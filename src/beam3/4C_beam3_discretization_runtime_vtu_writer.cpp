/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief Write visualization output for a beam discretization in vtk/vtu format at runtime

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

/* headers */
#include "4C_beam3_discretization_runtime_vtu_writer.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_periodic_boundingbox.hpp"
#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_lib_discret.hpp"
#include "4C_lib_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Comm.h>

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamDiscretizationRuntimeOutputWriter::BeamDiscretizationRuntimeOutputWriter(
    IO::VisualizationParameters parameters, const Epetra_Comm& comm)
    : visualization_manager_(Teuchos::rcp(
          new IO::VisualizationManager(std::move(parameters), comm, "structure-beams"))),
      use_absolute_positions_(true)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::Initialize(
    Teuchos::RCP<DRT::Discretization> discretization,
    bool use_absolute_positions_for_point_coordinates, const unsigned int n_subsegments,
    Teuchos::RCP<const CORE::GEO::MESHFREE::BoundingBox> const& periodic_boundingbox)
{
  discretization_ = discretization;
  use_absolute_positions_ = use_absolute_positions_for_point_coordinates;
  periodic_boundingbox_ = periodic_boundingbox;
  n_subsegments_ = n_subsegments;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::SetGeometryFromBeamDiscretization(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  /*  Note:
   *
   *  The centerline geometry of one element cannot be represented by one simple vtk cell type
   *  because we use cubic Hermite polynomials for the interpolation of the centerline.
   *
   *  Instead, we subdivide each beam element in several linear segments. This corresponds to a
   *  VTK_POLY_LINE (vtk cell type number 4). So one beam element will be visualized as one vtk
   * cell, but the number of points does not equal the number of FE nodes.
   *
   *  For a list of vtk cell types, see e.g.
   *  http://www.vtk.org/doc/nightly/html/vtkCellType_8h.html
   *
   *  Things get more complicated for 'cut' elements, i.e. when nodes have been 'shifted' due to
   *  periodic boundary conditions. Our approach here is to create two (or more) vtk cells from
   *  one beam element.
   *
   *
   *  Another approach would be to visualize the cubic Hermite polynomials as cubic Lagrange
   *  polynomials and specify four visualization points from e.g. the two FE nodes and two more
   * arbitrary points along the centerline. However, the representation of nonlinear geometry in
   * Paraview turned out to not work as expected (e.g. no visible refinement of subsegments if
   * corresponding parameter is changed).
   */

  // always use 3D for beams
  const unsigned int num_spatial_dimensions = 3;

  // determine and store local row indices of all beam elements in the given discretization
  // todo: maybe do this only if parallel distribution has changed, i.e not
  // ElementRowMapOld->SameAs(ElementRowMap)
  local_row_indices_beam_elements_.clear();
  local_row_indices_beam_elements_.reserve(discretization_->NumMyRowElements());
  for (unsigned int iele = 0; iele < static_cast<unsigned int>(discretization_->NumMyRowElements());
       ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    if (beamele != nullptr) local_row_indices_beam_elements_.push_back(iele);
  }

  num_cells_per_element_.clear();
  num_cells_per_element_.resize(local_row_indices_beam_elements_.size());

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);

  // do not need to store connectivity indices here because we create a
  // contiguous array by the order in which we fill the coordinates (otherwise
  // need to adjust order of filling in the coordinates).
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  std::vector<double>& point_coordinates = visualization_data.GetPointCoordinates();
  point_coordinates.clear();
  point_coordinates.reserve(num_spatial_dimensions * num_visualization_points);

  std::vector<uint8_t>& cell_types = visualization_data.GetCellTypes();
  cell_types.clear();
  cell_types.reserve(num_beam_row_elements);

  std::vector<int32_t>& cell_offsets = visualization_data.GetCellOffsets();
  cell_offsets.clear();
  cell_offsets.reserve(num_beam_row_elements);

  std::vector<bool> dummy_shift_in_dim(3, false);

  // loop over my elements and collect the geometry/grid data
  unsigned int pointcounter = 0;

  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    std::vector<double> beamelement_displacement_vector;

    if (use_absolute_positions_)
    {
      // this is needed in case your input file contains shifted/cut elements
      if (periodic_boundingbox_ != Teuchos::null)
      {
        BEAMINTERACTION::UTILS::GetCurrentUnshiftedElementDis(*discretization_, ele,
            displacement_state_vector, *periodic_boundingbox_, beamelement_displacement_vector);
      }
      // this is needed in case your input file does not contain shifted/cut elements
      else
      {
        BEAMINTERACTION::UTILS::GetCurrentElementDis(
            *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);
      }
    }

    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine their interpolated (initial) positions r */
    CORE::LINALG::Matrix<3, 1> interpolated_position(true);
    CORE::LINALG::Matrix<3, 1> interpolated_position_priorpoint(true);
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      interpolated_position.Clear();
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      if (use_absolute_positions_)
        beamele->GetPosAtXi(interpolated_position, xi, beamelement_displacement_vector);
      else
        beamele->GetRefPosAtXi(interpolated_position, xi);

      if (periodic_boundingbox_ != Teuchos::null)
        periodic_boundingbox_->Shift3D(interpolated_position);

      CORE::LINALG::Matrix<3, 1> unshift_interpolated_position = interpolated_position;

      // check if an element is cut by a periodic boundary
      bool shift = false;
      if (periodic_boundingbox_ != Teuchos::null)
        shift = periodic_boundingbox_->CheckIfShiftBetweenPoints(
            unshift_interpolated_position, interpolated_position_priorpoint, dummy_shift_in_dim);

      // if there is a shift between two consecutive points, double that point and create new cell
      // not for first and last point
      if (ipoint != 0 and ipoint != n_subsegments_ and shift)
      {
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
          point_coordinates.push_back(unshift_interpolated_position(idim));

        ++pointcounter;
        cell_offsets.push_back(pointcounter);
        cell_types.push_back(4);
        ++num_cells_per_element_[ibeamele];
      }

      // in case of last visualization point, we only add the unshifted (compared to former point)
      // configuration
      if (ipoint == n_subsegments_)
      {
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
          point_coordinates.push_back(unshift_interpolated_position(idim));
      }
      else
      {
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
          point_coordinates.push_back(interpolated_position(idim));
      }

      ++pointcounter;

      interpolated_position_priorpoint = interpolated_position;
    }
    // VTK_POLY_LINE (vtk cell type number 4)
    cell_types.push_back(4);
    ++num_cells_per_element_[ibeamele];
    cell_offsets.push_back(pointcounter);
  }

  // safety checks
  if (cell_types.size() != cell_offsets.size())
  {
    FOUR_C_THROW("RuntimeVtuWriter expected %d cell type values, but got %d", num_beam_row_elements,
        cell_types.size());
  }

  if (periodic_boundingbox_ != Teuchos::null and !periodic_boundingbox_->HavePBC() and
      (point_coordinates.size() != num_spatial_dimensions * num_visualization_points))
  {
    FOUR_C_THROW("RuntimeVtuWriter expected %d coordinate values, but got %d",
        num_spatial_dimensions * num_visualization_points, point_coordinates.size());
  }

  if (periodic_boundingbox_ != Teuchos::null and !periodic_boundingbox_->HavePBC() and
      (cell_offsets.size() != num_beam_row_elements))
  {
    FOUR_C_THROW("RuntimeVtuWriter expected %d cell offset values, but got %d",
        num_beam_row_elements, cell_offsets.size());
  }
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendDisplacementField(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  // triads only make sense in 3D
  const unsigned int num_spatial_dimensions = 3;

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);
  std::vector<int32_t> const& cell_offsets = visualization_data.GetCellOffsets();

  // disp vector
  std::vector<double> displacement_vector;
  displacement_vector.reserve(num_spatial_dimensions * num_visualization_points);

  // number of points so far
  int points_sofar = 0;

  // loop over myrank's beam elements and compute disp for each visualization point
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BEAMINTERACTION::UTILS::GetCurrentElementDis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);

    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine its disp state */
    CORE::LINALG::Matrix<3, 1> pos_visualization_point;
    CORE::LINALG::Matrix<3, 1> refpos_visualization_point;
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      pos_visualization_point.Clear();
      refpos_visualization_point.Clear();

      // interpolate
      beamele->GetRefPosAtXi(refpos_visualization_point, xi);
      beamele->GetPosAtXi(pos_visualization_point, xi, beamelement_displacement_vector);

      // in case of periodic boundary conditions, a point (except first and last point of an
      // element) can exists twice, we check this here by looking if current point is in cell offset
      // list and therefore starts of a new cell
      unsigned int num_point_exists = 1;
      if (ipoint != 0 and ipoint != n_subsegments_)
      {
        unsigned int curr_point_number = points_sofar + 1;
        if (std::find(cell_offsets.begin(), cell_offsets.end(), curr_point_number) !=
            cell_offsets.end())
          num_point_exists = 2;
      }

      // store the information in vectors that can be interpreted by vtu writer (disp = pos -
      // refpos) and update number of point data written
      for (unsigned int i = 0; i < num_point_exists; ++i)
      {
        ++points_sofar;
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
          displacement_vector.push_back(
              pos_visualization_point(idim, 0) - refpos_visualization_point(idim, 0));
      }
    }
  }

  // finally append the solution vectors to the visualization data of the vtu writer object
  visualization_data.SetPointDataVector(
      "displacement", displacement_vector, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendTriadField(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  // triads only make sense in 3D
  const unsigned int num_spatial_dimensions = 3;

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);
  std::vector<int32_t> const& cell_offsets = visualization_data.GetCellOffsets();

  // we write the triad field as three base vectors at every visualization point
  std::vector<double> base_vector_1;
  base_vector_1.reserve(num_spatial_dimensions * num_visualization_points);

  std::vector<double> base_vector_2;
  base_vector_2.reserve(num_spatial_dimensions * num_visualization_points);

  std::vector<double> base_vector_3;
  base_vector_3.reserve(num_spatial_dimensions * num_visualization_points);

  // number of points so far
  int points_sofar = 0;

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BEAMINTERACTION::UTILS::GetCurrentElementDis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);


    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine the triad */
    CORE::LINALG::Matrix<3, 3> triad_visualization_point;
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      triad_visualization_point.Clear();

      beamele->GetTriadAtXi(triad_visualization_point, xi, beamelement_displacement_vector);

      // in case of periodic boundary conditions, a point (except first and last point of an
      // element) can exists twice, we check this here by looking if current point is in cell offset
      // list and therefore starts of a new cell
      unsigned int num_point_exists = 1;
      if (ipoint != 0 and ipoint != n_subsegments_)
      {
        unsigned int curr_point_number = points_sofar + 1;
        if (std::find(cell_offsets.begin(), cell_offsets.end(), curr_point_number) !=
            cell_offsets.end())
          num_point_exists = 2;
      }

      // store the information in vectors that can be interpreted by vtu writer
      // and update number of point data written
      for (unsigned int i = 0; i < num_point_exists; ++i)
      {
        ++points_sofar;
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
        {
          // first column: first base vector
          base_vector_1.push_back(triad_visualization_point(idim, 0));

          // second column: second base vector
          base_vector_2.push_back(triad_visualization_point(idim, 1));

          // third column: third base vector
          base_vector_3.push_back(triad_visualization_point(idim, 2));
        }
      }
    }
  }

  // finally append the solution vectors to the visualization data of the vtu writer object
  visualization_data.SetPointDataVector("base_vector_1", base_vector_1, num_spatial_dimensions);
  visualization_data.SetPointDataVector("base_vector_2", base_vector_2, num_spatial_dimensions);
  visualization_data.SetPointDataVector("base_vector_3", base_vector_3, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementOwningProcessor()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // processor owning the element
  std::vector<double> owner;
  owner.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i) owner.push_back(ele->Owner());
  }

  // set the solution vector in the visualization data container
  visualization_manager_->GetVisualizationData().SetCellDataVector("element_owner", owner, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementGID()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // vector with the IDs of the beams on this processor.
  std::vector<double> gid;
  gid.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i) gid.push_back(ele->Id());
  }

  // append the solution vector to the visualization data
  visualization_manager_->GetVisualizationData().SetCellDataVector("element_gid", gid, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementGhostingInformation()
{
  const auto only_select_beam_elements = [](const DRT::Element* ele)
  { return dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele); };
  IO::AppendElementGhostingInformation(
      *discretization_, *visualization_manager_, only_select_beam_elements);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementInternalEnergy()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // processor owning the element
  std::vector<double> e_int;
  e_int.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);


    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      e_int.push_back(beamele->GetInternalEnergy());
  }

  // append the solution vector to the visualization data
  visualization_manager_->GetVisualizationData().SetCellDataVector(
      "element_internal_energy", e_int, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementKineticEnergy()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // processor owning the element
  std::vector<double> e_kin;
  e_kin.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);


    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      e_kin.push_back(beamele->GetKineticEnergy());
  }

  // append the solution vector to the visualization data
  visualization_manager_->GetVisualizationData().SetCellDataVector(
      "element_kinetic_energy", e_kin, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementFilamentIdAndType()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // processor owning the element
  std::vector<double> id, type;
  id.reserve(num_beam_row_elements);
  type.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    // get filament number (note so far only one filament for each element and node)
    DRT::Condition* cond = ele->Nodes()[0]->GetCondition("BeamLineFilamentCondition");
    if (cond == nullptr)
      FOUR_C_THROW(" No filament number assigned to element with gid %i .", ele->Id());

    double current_id = cond->parameters().Get<int>("FilamentId");
    double current_type =
        INPAR::BEAMINTERACTION::String2FilamentType((cond->parameters().Get<std::string>("Type")));

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
    {
      id.push_back(current_id);
      type.push_back(current_type);
    }
  }

  // append the solution vector to the visualization data
  auto& visualization_data = visualization_manager_->GetVisualizationData();
  visualization_data.SetCellDataVector("ele_filament_id", id, 1);
  visualization_data.SetCellDataVector("ele_filament_type", type, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementCircularCrossSectionRadius()
{
  // Todo we assume a circular cross-section shape here; generalize this to other shapes

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // we assume a constant cross-section radius over the element length
  std::vector<double> cross_section_radius;
  cross_section_radius.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // this needs to be done for all cells that make up a cut element
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      cross_section_radius.push_back(beamele->GetCircularCrossSectionRadiusForInteractions());
  }

  // append the solution vector to the visualization data
  visualization_manager_->GetVisualizationData().SetCellDataVector(
      "cross_section_radius", cross_section_radius, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendPointCircularCrossSectionInformationVector(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  // assume 3D here
  const unsigned int num_spatial_dimensions = 3;

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);


  // a beam with circular cross-section can be visualized as a 'chain' of straight cylinders
  // this is also supported as 'tube' in Paraview
  // to define one cylinder, we use the first base vector as its unit normal vector
  // and scale it with the cross-section radius of the beam
  // Edit: This approach seems not to work with Paraview because the functionality 'Vary Radius'
  //       of the Tube filter is different to what we expected.
  //       However, we keep this method as it could be useful for other visualization approaches
  //       in the future.
  std::vector<double> circular_cross_section_information_vector;
  circular_cross_section_information_vector.reserve(
      num_spatial_dimensions * num_visualization_points);
  std::vector<int32_t> const& cell_offsets = visualization_data.GetCellOffsets();

  // number of points so far
  int points_sofar = 0;

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    const double circular_cross_section_radius =
        beamele->GetCircularCrossSectionRadiusForInteractions();

    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BEAMINTERACTION::UTILS::GetCurrentElementDis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);


    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine the triad */
    CORE::LINALG::Matrix<3, 3> triad_visualization_point;
    double xi = 0.0;
    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      triad_visualization_point.Clear();

      beamele->GetTriadAtXi(triad_visualization_point, xi, beamelement_displacement_vector);

      // in case of periodic boundary conditions, a point (except first and last point of an
      // element) can exists twice, we check this here by looking if current point is in cell offset
      // list and therefore starts of a new cell
      unsigned int num_point_exists = 1;
      if (ipoint != 0 and ipoint != n_subsegments_)
      {
        unsigned int curr_point_number = points_sofar + 1;
        if (std::find(cell_offsets.begin(), cell_offsets.end(), curr_point_number) !=
            cell_offsets.end())
          num_point_exists = 2;
      }

      // store the information in vectors that can be interpreted by vtu writer (disp = pos -
      // refpos) and update number of point data written
      for (unsigned int i = 0; i < num_point_exists; ++i)
      {
        ++points_sofar;
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
        {
          // first column: first base vector
          circular_cross_section_information_vector.push_back(
              triad_visualization_point(idim, 0) * circular_cross_section_radius);
        }
      }
    }
  }

  // finally append the solution vectors to the visualization data of the vtu writer object
  visualization_data.SetPointDataVector("circular_cross_section_information_vector",
      circular_cross_section_information_vector, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendGaussPointMaterialCrossSectionStrainResultants()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();


  // storage for material strain measures at all GPs of all my row elements
  std::vector<double> axial_strain_GPs_all_row_elements;
  std::vector<double> shear_strain_2_GPs_all_row_elements;
  std::vector<double> shear_strain_3_GPs_all_row_elements;

  std::vector<double> twist_GPs_all_row_elements;
  std::vector<double> curvature_2_GPs_all_row_elements;
  std::vector<double> curvature_3_GPs_all_row_elements;


  // storage for material strain measures at all GPs of current element
  std::vector<double> axial_strain_GPs_current_element;
  std::vector<double> shear_strain_2_GPs_current_element;
  std::vector<double> shear_strain_3_GPs_current_element;

  std::vector<double> twist_GPs_current_element;
  std::vector<double> curvature_2_GPs_current_element;
  std::vector<double> curvature_3_GPs_current_element;


  // number of Gauss points must be the same for all elements in the grid
  unsigned int num_GPs_per_element_strains_translational = 0;
  unsigned int num_GPs_per_element_strains_rotational = 0;


  // loop over my elements and collect the data
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    axial_strain_GPs_current_element.clear();
    shear_strain_2_GPs_current_element.clear();
    shear_strain_3_GPs_current_element.clear();

    twist_GPs_current_element.clear();
    curvature_2_GPs_current_element.clear();
    curvature_3_GPs_current_element.clear();


    // get GP strain values from previous element evaluation call
    beamele->GetMaterialStrainResultantsAtAllGPs(axial_strain_GPs_current_element,
        shear_strain_2_GPs_current_element, shear_strain_3_GPs_current_element,
        twist_GPs_current_element, curvature_2_GPs_current_element,
        curvature_3_GPs_current_element);

    // special treatment for Kirchhoff beam elements where shear mode does not exist
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (shear_strain_2_GPs_current_element.size() == 0 and
        shear_strain_3_GPs_current_element.size() == 0)
    {
      shear_strain_2_GPs_current_element.resize(axial_strain_GPs_current_element.size());
      std::fill(shear_strain_2_GPs_current_element.begin(),
          shear_strain_2_GPs_current_element.end(), 0.0);

      shear_strain_3_GPs_current_element.resize(axial_strain_GPs_current_element.size());
      std::fill(shear_strain_3_GPs_current_element.begin(),
          shear_strain_3_GPs_current_element.end(), 0.0);
    }

    // special treatment for reduced Kirchhoff beam element where torsion mode does not exist
    // and due to isotropic formulation only one component of curvature and bending moment exists
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (twist_GPs_current_element.size() == 0 and curvature_3_GPs_current_element.size() == 0)
    {
      twist_GPs_current_element.resize(curvature_2_GPs_current_element.size());
      std::fill(twist_GPs_current_element.begin(), twist_GPs_current_element.end(), 0.0);

      curvature_3_GPs_current_element.resize(curvature_2_GPs_current_element.size());
      std::fill(
          curvature_3_GPs_current_element.begin(), curvature_3_GPs_current_element.end(), 0.0);
    }


    // safety check for number of Gauss points per element
    // initialize numbers from first element
    if (ibeamele == 0)
    {
      num_GPs_per_element_strains_translational = axial_strain_GPs_current_element.size();
      num_GPs_per_element_strains_rotational = curvature_2_GPs_current_element.size();
    }

    if (axial_strain_GPs_current_element.size() != num_GPs_per_element_strains_translational or
        shear_strain_2_GPs_current_element.size() != num_GPs_per_element_strains_translational or
        shear_strain_3_GPs_current_element.size() != num_GPs_per_element_strains_translational or
        twist_GPs_current_element.size() != num_GPs_per_element_strains_rotational or
        curvature_2_GPs_current_element.size() != num_GPs_per_element_strains_rotational or
        curvature_3_GPs_current_element.size() != num_GPs_per_element_strains_rotational)
    {
      FOUR_C_THROW("number of Gauss points must be the same for all elements in discretization!");
    }

    // store the values of current element in the large vectors collecting data from all elements
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
    {
      InsertVectorValuesAtBackOfOtherVector(
          axial_strain_GPs_current_element, axial_strain_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          shear_strain_2_GPs_current_element, shear_strain_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          shear_strain_3_GPs_current_element, shear_strain_3_GPs_all_row_elements);


      InsertVectorValuesAtBackOfOtherVector(twist_GPs_current_element, twist_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          curvature_2_GPs_current_element, curvature_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          curvature_3_GPs_current_element, curvature_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_strains_translational);
  int global_num_GPs_per_element_rotational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_strains_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  visualization_data.SetCellDataVector("axial_strain_GPs", axial_strain_GPs_all_row_elements,
      global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("shear_strain_2_GPs", shear_strain_2_GPs_all_row_elements,
      global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("shear_strain_3_GPs", shear_strain_3_GPs_all_row_elements,
      global_num_GPs_per_element_translational);


  visualization_data.SetCellDataVector(
      "twist_GPs", twist_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector(
      "curvature_2_GPs", curvature_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector(
      "curvature_3_GPs", curvature_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    AppendGaussPointMaterialCrossSectionStrainResultantsContinuous()
{
  AppendContinuousStressStrainResultants(StressStrainField::material_strain);
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendGaussPointMaterialCrossSectionStressResultants()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();


  // storage for material stress resultants at all GPs of all my row elements
  std::vector<double> material_axial_force_GPs_all_row_elements;
  std::vector<double> material_shear_force_2_GPs_all_row_elements;
  std::vector<double> material_shear_force_3_GPs_all_row_elements;

  std::vector<double> material_torque_GPs_all_row_elements;
  std::vector<double> material_bending_moment_2_GPs_all_row_elements;
  std::vector<double> material_bending_moment_3_GPs_all_row_elements;


  // storage for material stress resultants at all GPs of current element
  std::vector<double> material_axial_force_GPs_current_element;
  std::vector<double> material_shear_force_2_GPs_current_element;
  std::vector<double> material_shear_force_3_GPs_current_element;

  std::vector<double> material_torque_GPs_current_element;
  std::vector<double> material_bending_moment_2_GPs_current_element;
  std::vector<double> material_bending_moment_3_GPs_current_element;


  // number of Gauss points must be the same for all elements in the grid
  unsigned int num_GPs_per_element_stresses_translational = 0;
  unsigned int num_GPs_per_element_stresses_rotational = 0;


  // loop over my elements and collect the data
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    material_axial_force_GPs_current_element.clear();
    material_shear_force_2_GPs_current_element.clear();
    material_shear_force_3_GPs_current_element.clear();

    material_torque_GPs_current_element.clear();
    material_bending_moment_2_GPs_current_element.clear();
    material_bending_moment_3_GPs_current_element.clear();


    // get GP stress values from previous element evaluation call
    beamele->GetMaterialStressResultantsAtAllGPs(material_axial_force_GPs_current_element,
        material_shear_force_2_GPs_current_element, material_shear_force_3_GPs_current_element,
        material_torque_GPs_current_element, material_bending_moment_2_GPs_current_element,
        material_bending_moment_3_GPs_current_element);


    // special treatment for Kirchhoff beam elements where shear mode does not exist
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (material_shear_force_2_GPs_current_element.size() == 0 and
        material_shear_force_3_GPs_current_element.size() == 0)
    {
      material_shear_force_2_GPs_current_element.resize(
          material_axial_force_GPs_current_element.size());
      std::fill(material_shear_force_2_GPs_current_element.begin(),
          material_shear_force_2_GPs_current_element.end(), 0.0);

      material_shear_force_3_GPs_current_element.resize(
          material_axial_force_GPs_current_element.size());
      std::fill(material_shear_force_3_GPs_current_element.begin(),
          material_shear_force_3_GPs_current_element.end(), 0.0);
    }

    // special treatment for reduced Kirchhoff beam element where torsion mode does not exist
    // and due to isotropic formulation only one component of curvature and bending moment exists
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (material_torque_GPs_current_element.size() == 0 and
        material_bending_moment_3_GPs_current_element.size() == 0)
    {
      material_torque_GPs_current_element.resize(
          material_bending_moment_2_GPs_current_element.size());
      std::fill(material_torque_GPs_current_element.begin(),
          material_torque_GPs_current_element.end(), 0.0);

      material_bending_moment_3_GPs_current_element.resize(
          material_bending_moment_2_GPs_current_element.size());
      std::fill(material_bending_moment_3_GPs_current_element.begin(),
          material_bending_moment_3_GPs_current_element.end(), 0.0);
    }


    // safety check for number of Gauss points per element
    // initialize numbers from first element
    if (ibeamele == 0)
    {
      num_GPs_per_element_stresses_translational = material_axial_force_GPs_current_element.size();
      num_GPs_per_element_stresses_rotational =
          material_bending_moment_2_GPs_current_element.size();
    }

    if (material_axial_force_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        material_shear_force_2_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        material_shear_force_3_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        material_torque_GPs_current_element.size() != num_GPs_per_element_stresses_rotational or
        material_bending_moment_2_GPs_current_element.size() !=
            num_GPs_per_element_stresses_rotational or
        material_bending_moment_3_GPs_current_element.size() !=
            num_GPs_per_element_stresses_rotational)
    {
      FOUR_C_THROW("number of Gauss points must be the same for all elements in discretization!");
    }

    // store the values of current element in the large vectors collecting data from all elements
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
    {
      InsertVectorValuesAtBackOfOtherVector(
          material_axial_force_GPs_current_element, material_axial_force_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          material_shear_force_2_GPs_current_element, material_shear_force_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          material_shear_force_3_GPs_current_element, material_shear_force_3_GPs_all_row_elements);


      InsertVectorValuesAtBackOfOtherVector(
          material_torque_GPs_current_element, material_torque_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(material_bending_moment_2_GPs_current_element,
          material_bending_moment_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(material_bending_moment_3_GPs_current_element,
          material_bending_moment_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_stresses_translational);
  int global_num_GPs_per_element_rotational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_stresses_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  visualization_data.SetCellDataVector("material_axial_force_GPs",
      material_axial_force_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("material_shear_force_2_GPs",
      material_shear_force_2_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("material_shear_force_3_GPs",
      material_shear_force_3_GPs_all_row_elements, global_num_GPs_per_element_translational);


  visualization_data.SetCellDataVector("material_torque_GPs", material_torque_GPs_all_row_elements,
      global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector("material_bending_moment_2_GPs",
      material_bending_moment_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector("material_bending_moment_3_GPs",
      material_bending_moment_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    AppendGaussPointMaterialCrossSectionStressResultantsContinuous()
{
  AppendContinuousStressStrainResultants(StressStrainField::material_stress);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendGaussPointSpatialCrossSectionStressResultants()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();


  // storage for material stress resultants at all GPs of all my row elements
  std::vector<double> spatial_axial_force_GPs_all_row_elements;
  std::vector<double> spatial_shear_force_2_GPs_all_row_elements;
  std::vector<double> spatial_shear_force_3_GPs_all_row_elements;

  std::vector<double> spatial_torque_GPs_all_row_elements;
  std::vector<double> spatial_bending_moment_2_GPs_all_row_elements;
  std::vector<double> spatial_bending_moment_3_GPs_all_row_elements;


  // storage for material stress resultants at all GPs of current element
  std::vector<double> spatial_axial_force_GPs_current_element;
  std::vector<double> spatial_shear_force_2_GPs_current_element;
  std::vector<double> spatial_shear_force_3_GPs_current_element;

  std::vector<double> spatial_torque_GPs_current_element;
  std::vector<double> spatial_bending_moment_2_GPs_current_element;
  std::vector<double> spatial_bending_moment_3_GPs_current_element;


  // number of Gauss points must be the same for all elements in the grid
  unsigned int num_GPs_per_element_stresses_translational = 0;
  unsigned int num_GPs_per_element_stresses_rotational = 0;


  // loop over my elements and collect the data
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    spatial_axial_force_GPs_current_element.clear();
    spatial_shear_force_2_GPs_current_element.clear();
    spatial_shear_force_3_GPs_current_element.clear();

    spatial_torque_GPs_current_element.clear();
    spatial_bending_moment_2_GPs_current_element.clear();
    spatial_bending_moment_3_GPs_current_element.clear();


    // get GP stress values from previous element evaluation call
    beamele->GetSpatialStressResultantsAtAllGPs(spatial_axial_force_GPs_current_element,
        spatial_shear_force_2_GPs_current_element, spatial_shear_force_3_GPs_current_element,
        spatial_torque_GPs_current_element, spatial_bending_moment_2_GPs_current_element,
        spatial_bending_moment_3_GPs_current_element);


    // special treatment for Kirchhoff beam elements where shear mode does not exist
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (spatial_shear_force_2_GPs_current_element.size() == 0 and
        spatial_shear_force_3_GPs_current_element.size() == 0)
    {
      spatial_shear_force_2_GPs_current_element.resize(
          spatial_axial_force_GPs_current_element.size());
      std::fill(spatial_shear_force_2_GPs_current_element.begin(),
          spatial_shear_force_2_GPs_current_element.end(), 0.0);

      spatial_shear_force_3_GPs_current_element.resize(
          spatial_axial_force_GPs_current_element.size());
      std::fill(spatial_shear_force_3_GPs_current_element.begin(),
          spatial_shear_force_3_GPs_current_element.end(), 0.0);
    }

    // special treatment for reduced Kirchhoff beam element where torsion mode does not exist
    // and due to isotropic formulation only one component of curvature and bending moment exists
    // Todo add option where only the relevant modes are written to file and let the user decide
    //      whether to write zeros or nothing for non-applicable modes
    if (spatial_torque_GPs_current_element.size() == 0 and
        spatial_bending_moment_3_GPs_current_element.size() == 0)
    {
      spatial_torque_GPs_current_element.resize(
          spatial_bending_moment_2_GPs_current_element.size());
      std::fill(spatial_torque_GPs_current_element.begin(),
          spatial_torque_GPs_current_element.end(), 0.0);

      spatial_bending_moment_3_GPs_current_element.resize(
          spatial_bending_moment_2_GPs_current_element.size());
      std::fill(spatial_bending_moment_3_GPs_current_element.begin(),
          spatial_bending_moment_3_GPs_current_element.end(), 0.0);
    }


    // safety check for number of Gauss points per element
    // initialize numbers from first element
    if (ibeamele == 0)
    {
      num_GPs_per_element_stresses_translational = spatial_axial_force_GPs_current_element.size();
      num_GPs_per_element_stresses_rotational = spatial_bending_moment_2_GPs_current_element.size();
    }

    if (spatial_axial_force_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        spatial_shear_force_2_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        spatial_shear_force_3_GPs_current_element.size() !=
            num_GPs_per_element_stresses_translational or
        spatial_torque_GPs_current_element.size() != num_GPs_per_element_stresses_rotational or
        spatial_bending_moment_2_GPs_current_element.size() !=
            num_GPs_per_element_stresses_rotational or
        spatial_bending_moment_3_GPs_current_element.size() !=
            num_GPs_per_element_stresses_rotational)
    {
      FOUR_C_THROW("number of Gauss points must be the same for all elements in discretization!");
    }

    // store the values of current element in the large vectors collecting data from all elements
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
    {
      InsertVectorValuesAtBackOfOtherVector(
          spatial_axial_force_GPs_current_element, spatial_axial_force_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          spatial_shear_force_2_GPs_current_element, spatial_shear_force_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(
          spatial_shear_force_3_GPs_current_element, spatial_shear_force_3_GPs_all_row_elements);


      InsertVectorValuesAtBackOfOtherVector(
          spatial_torque_GPs_current_element, spatial_torque_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(spatial_bending_moment_2_GPs_current_element,
          spatial_bending_moment_2_GPs_all_row_elements);

      InsertVectorValuesAtBackOfOtherVector(spatial_bending_moment_3_GPs_current_element,
          spatial_bending_moment_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_stresses_translational);
  int global_num_GPs_per_element_rotational =
      GetGlobalNumberOfGaussPointsPerBeam(num_GPs_per_element_stresses_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->GetVisualizationData();

  visualization_data.SetCellDataVector("spatial_axial_force_GPs",
      spatial_axial_force_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("spatial_shear_force_2_GPs",
      spatial_shear_force_2_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.SetCellDataVector("spatial_shear_force_3_GPs",
      spatial_shear_force_3_GPs_all_row_elements, global_num_GPs_per_element_translational);


  visualization_data.SetCellDataVector("spatial_torque_GPs", spatial_torque_GPs_all_row_elements,
      global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector("spatial_bending_moment_2_GPs",
      spatial_bending_moment_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.SetCellDataVector("spatial_bending_moment_3_GPs",
      spatial_bending_moment_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementOrientationParamater(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  /*
   * see
   * [1] Chandran and Barocas, "Affine Versus Non_Affine Fibril Kinamtics in Collagen Networks:
   * Theoretical Studies of Network Behavior", 2006.
   * [2] D.L. Humphries et al., "Mechnanical Cell-Cell Communication in Fibrous Networks: The
   * Importance of Network Geometry", 2017.
   */

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // define variables
  std::vector<double> local_orientation_parameter(3, 0.0);

  std::vector<double> orientation_parameter_for_each_element;
  orientation_parameter_for_each_element.reserve(num_beam_row_elements * 3);
  std::vector<double> orientation_parameter_for_global_network;
  orientation_parameter_for_global_network.reserve(num_beam_row_elements * 3);

  double local_accumulated_ele_lengths = 0.0;

  // loop over my elements and collect data about orientation and length of elements/filaments
  //(assignment of elements to filaments not needed in case as parameter is calculated as sum over
  // all elements)
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // length of element is approximated linearly, as also the direction of a element is calculated
    // linearly independent of centerline interpolation
    CORE::LINALG::Matrix<3, 1> dirvec(true);

    std::vector<double> pos(2, 0.0);
    for (int dim = 0; dim < 3; ++dim)
    {
      pos[0] = ele->Nodes()[0]->X()[dim] +
               (*displacement_state_vector)[displacement_state_vector->Map().LID(
                   discretization_->Dof(ele->Nodes()[0])[dim])];
      pos[1] = ele->Nodes()[1]->X()[dim] +
               (*displacement_state_vector)[displacement_state_vector->Map().LID(
                   discretization_->Dof(ele->Nodes()[1])[dim])];
      dirvec(dim) = pos[1] - pos[0];
    }

    // current element length (linear)
    double curr_lin_ele_length = dirvec.Norm2();

    // loop over all base vectors for orientation index x,y and z
    CORE::LINALG::Matrix<3, 1> unit_base_vec(true);
    std::vector<double> curr_ele_orientation_parameter(3, 0.0);
    for (int unsigned ibase = 0; ibase < 3; ++ibase)
    {
      // init current base vector
      unit_base_vec.Clear();
      unit_base_vec(ibase) = 1.0;

      double cos_squared = dirvec.Dot(unit_base_vec) / curr_lin_ele_length;
      cos_squared *= cos_squared;

      curr_ele_orientation_parameter[ibase] = cos_squared;
      local_orientation_parameter[ibase] += curr_lin_ele_length * cos_squared;
    }

    local_accumulated_ele_lengths += curr_lin_ele_length;

    // in case of cut elements by a periodic boundary
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      InsertVectorValuesAtBackOfOtherVector(
          curr_ele_orientation_parameter, orientation_parameter_for_each_element);
  }

  // calculate length of all (linear) elements
  double global_linear_filament_length = 0;
  discretization_->Comm().SumAll(&local_accumulated_ele_lengths, &global_linear_filament_length, 1);

  //
  for (int unsigned i = 0; i < 3; ++i)
    local_orientation_parameter[i] /= global_linear_filament_length;

  // calculate global orientation parameter
  std::vector<double> global_orientation_parameter(3, 0.0);
  discretization_->Comm().SumAll(
      local_orientation_parameter.data(), global_orientation_parameter.data(), 3);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      InsertVectorValuesAtBackOfOtherVector(
          global_orientation_parameter, orientation_parameter_for_global_network);

  auto& visualization_data = visualization_manager_->GetVisualizationData();

  // append the solution vector to the visualization data
  visualization_data.SetCellDataVector(
      "orientation_parameter_element", orientation_parameter_for_each_element, 3);

  // append the solution vector to the visualization data
  visualization_data.SetCellDataVector(
      "orientation_parameter", orientation_parameter_for_global_network, 3);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendRVECrosssectionForces(
    Teuchos::RCP<const Epetra_Vector> const& displacement_state_vector)
{
  // NOTE: so far force in node 0 is written
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // storage for spatial stress resultants at all GPs of all my row elements
  std::vector<double> sum_spatial_force_rve_crosssection_xdir;
  std::vector<double> sum_spatial_force_rve_crosssection_ydir;
  std::vector<double> sum_spatial_force_rve_crosssection_zdir;
  sum_spatial_force_rve_crosssection_xdir.reserve(num_beam_row_elements);
  sum_spatial_force_rve_crosssection_ydir.reserve(num_beam_row_elements);
  sum_spatial_force_rve_crosssection_zdir.reserve(num_beam_row_elements);
  std::vector<double> spatial_x_force_GPs_current_element;
  std::vector<double> spatial_y_force_2_GPs_current_element;
  std::vector<double> spatial_z_force_3_GPs_current_element;

  std::vector<int> nodedofs;
  std::vector<std::vector<double>> fint_sum(3, std::vector<double>(3, 0.0));
  std::vector<double> beamelement_displacement_vector;
  std::vector<double> beamelement_shift_displacement_vector;
  CORE::LINALG::Matrix<3, 1> pos_node_1(true);
  CORE::LINALG::Matrix<3, 1> pos_node_2(true);

  // create pseudo planes through center of RVE (like this it also works if
  // your box is not periodic, i.e. you do not have cut element on the box edges)
  CORE::LINALG::Matrix<3, 2> box(true);
  if (periodic_boundingbox_ != Teuchos::null)
  {
    for (unsigned dim = 0; dim < 3; ++dim)
    {
      box(dim, 0) = periodic_boundingbox_->Box()(dim, 0);
      box(dim, 1) =
          periodic_boundingbox_->Box()(dim, 0) + 0.5 * periodic_boundingbox_->EdgeLength(dim);
    }
  }

  CORE::LINALG::Matrix<3, 1> xi_intersect(true);

  // loop over all my elements and build force sum of myrank's cut element
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    BEAMINTERACTION::UTILS::GetCurrentElementDis(
        *discretization_, ele, displacement_state_vector, beamelement_shift_displacement_vector);
    BEAMINTERACTION::UTILS::GetCurrentUnshiftedElementDis(*discretization_, ele,
        displacement_state_vector, *periodic_boundingbox_, beamelement_displacement_vector);

    beamele->GetPosAtXi(pos_node_1, -1.0, beamelement_displacement_vector);
    beamele->GetPosAtXi(pos_node_2, 1.0, beamelement_displacement_vector);
    periodic_boundingbox_->GetXiOfIntersection3D(pos_node_1, pos_node_2, xi_intersect, box);

    // todo: change from just using first gauss point to linear inter-/extrapolation
    // between two closest gauss points

    spatial_x_force_GPs_current_element.clear();
    spatial_y_force_2_GPs_current_element.clear();
    spatial_z_force_3_GPs_current_element.clear();

    for (int dir = 0; dir < 3; ++dir)
    {
      if (xi_intersect(dir) > 1.0) continue;

      beamele->GetSpatialForcesAtAllGPs(spatial_x_force_GPs_current_element,
          spatial_y_force_2_GPs_current_element, spatial_z_force_3_GPs_current_element);

      fint_sum[dir][0] += spatial_x_force_GPs_current_element[0];
      fint_sum[dir][1] += spatial_y_force_2_GPs_current_element[0];
      fint_sum[dir][2] += spatial_z_force_3_GPs_current_element[0];
    }
  }

  std::vector<std::vector<double>> global_sum(3, std::vector<double>(3, 0.0));
  for (int dir = 0; dir < 3; ++dir)
    discretization_->Comm().SumAll(fint_sum[dir].data(), global_sum[dir].data(), 3);

  // loop over all my elements and build force sum of myrank's cut element
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      for (int dim = 0; dim < 3; ++dim)
      {
        sum_spatial_force_rve_crosssection_xdir.push_back(global_sum[0][dim]);
        sum_spatial_force_rve_crosssection_ydir.push_back(global_sum[1][dim]);
        sum_spatial_force_rve_crosssection_zdir.push_back(global_sum[2][dim]);
      }

  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->GetVisualizationData();
  visualization_data.SetCellDataVector(
      "sum_spatial_force_rve_crosssection_xdir", sum_spatial_force_rve_crosssection_xdir, 3);
  visualization_data.SetCellDataVector(
      "sum_spatial_force_rve_crosssection_ydir", sum_spatial_force_rve_crosssection_ydir, 3);
  visualization_data.SetCellDataVector(
      "sum_spatial_force_rve_crosssection_zdir", sum_spatial_force_rve_crosssection_zdir, 3);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendElementElasticEnergy()
{
  FOUR_C_THROW("not implemented yet");

  //  // count number of nodes and number for each processor; output is completely independent of
  //  // the number of processors involved
  //  unsigned int num_row_elements = discretization_->NumMyRowElements();
  //
  //  // processor owning the element
  //  std::vector<double> energy_elastic;
  //  energy_elastic.reserve( num_row_elements );
  //
  //
  //  // loop over my elements and collect the data about triads/base vectors
  //  for (unsigned int iele=0; iele<num_row_elements; ++iele)
  //  {
  //    const DRT::Element* ele = discretization_->lRowElement(iele);
  //
  //    // check for beam element
  //    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const
  //    DRT::ELEMENTS::Beam3Base*>(ele);
  //
  //    // Todo for now, simply skip all other elements
  //    if ( beamele == nullptr )
  //      continue;
  //
  //
  //    // Todo get Eint_ from previous element evaluation call
  //  for( int i = 0; i < num_cells_per_element_[iele]; ++i )
  //    energy_elastic.push_back( beamele->GetElasticEnergy() );
  //  }
  //
  //  // append the solution vector to the visualization data of the vtu writer object
  //  runtime_vtuwriter_->AppendVisualizationCellDataVector(
  //      energy_elastic, 1, "element_elastic_energy" );
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendRefLength()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  std::vector<double> ref_lengths;
  ref_lengths.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    auto beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    // this needs to be done for all cells that make up a cut element
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      ref_lengths.push_back(beamele->RefLength());
  }

  // append the solution vector to the visualization data
  visualization_manager_->GetVisualizationData().SetCellDataVector("ref_length", ref_lengths, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::WriteToDisk(
    const double visualization_time, const int visualization_step)
{
  visualization_manager_->WriteToDisk(visualization_time, visualization_step);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::InsertVectorValuesAtBackOfOtherVector(
    const std::vector<double>& vector_input, std::vector<double>& vector_output)
{
  vector_output.reserve(vector_output.size() + vector_input.size());

  std::copy(vector_input.begin(), vector_input.end(), std::back_inserter(vector_output));
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
int BeamDiscretizationRuntimeOutputWriter::GetGlobalNumberOfGaussPointsPerBeam(
    unsigned int my_num_gp) const
{
  int my_num_gp_signed = (int)my_num_gp;
  int global_num_gp = 0;
  discretization_->Comm().MaxAll(&my_num_gp_signed, &global_num_gp, 1);

  // Safety checks.
  if (my_num_gp_signed > 0 and my_num_gp_signed != global_num_gp)
    FOUR_C_THROW("The number of Gauss points must be the same for all elements in discretization!");
  else if (global_num_gp < 0)
    FOUR_C_THROW("The number of Gauss points must be zero or a positve integer!");

  return global_num_gp;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::CalcInterpolationPolynomialCoefficients(
    const CORE::FE::GaussRule1D& gauss_rule, const std::vector<double>& gauss_point_values,
    std::vector<double>& polynomial_coefficients) const
{
  // Get the coefficients for the interpolation functions at the Gauss points.
  std::size_t n_gp = 3;
  std::array<std::array<double, 3>, 3> lagrange_coefficients;
  switch (gauss_rule)
  {
    case CORE::FE::GaussRule1D::line_3point:
    {
      lagrange_coefficients[0][0] = 0.0;
      lagrange_coefficients[0][1] = -0.645497224367889;
      lagrange_coefficients[0][2] = 0.8333333333333333;

      lagrange_coefficients[1][0] = 1.0;
      lagrange_coefficients[1][1] = 0.0;
      lagrange_coefficients[1][2] = -1.6666666666666667;

      lagrange_coefficients[2][0] = 0.0;
      lagrange_coefficients[2][1] = 0.645497224367889;
      lagrange_coefficients[2][2] = 0.8333333333333333;
    }
    break;
    case CORE::FE::GaussRule1D::line_lobatto3point:
    {
      lagrange_coefficients[0][0] = 0.0;
      lagrange_coefficients[0][1] = -0.5;
      lagrange_coefficients[0][2] = 0.5;

      lagrange_coefficients[1][0] = 1.0;
      lagrange_coefficients[1][1] = 0.0;
      lagrange_coefficients[1][2] = -1.0;

      lagrange_coefficients[2][0] = 0.0;
      lagrange_coefficients[2][1] = 0.5;
      lagrange_coefficients[2][2] = 0.5;
    }
    break;
    default:
      FOUR_C_THROW("Interpolation for Gauss rule not yet implemented.");
      break;
  }

  // Calculate the coefficients of the polynomial to interpolate the Gauss points.
  polynomial_coefficients.resize(n_gp);
  std::fill(polynomial_coefficients.begin(), polynomial_coefficients.end(), 0.0);
  for (std::size_t i_gp = 0; i_gp < n_gp; i_gp++)
    for (std::size_t p = 0; p < n_gp; p++)
      polynomial_coefficients[p] += gauss_point_values[i_gp] * lagrange_coefficients[i_gp][p];
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double BeamDiscretizationRuntimeOutputWriter::EvaluatePolynomialCoefficients(
    const std::vector<double>& polynomial_coefficients, const double& xi) const
{
  double interpolated_value = 0.0;
  for (std::size_t p = 0; p < polynomial_coefficients.size(); p++)
    interpolated_value += polynomial_coefficients[p] * pow(xi, p);
  return interpolated_value;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::AppendContinuousStressStrainResultants(
    const StressStrainField stress_strain_field)
{
  // storage for stress / strain measures at all GPs of current element
  std::vector<std::vector<double>> stress_strain_GPs_current_element(6);

  // storage for coefficient vectors for the stress / strain interpolation.
  std::vector<std::vector<double>> stress_strain_coefficients(6);

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  std::size_t num_beam_row_elements = local_row_indices_beam_elements_.size();
  std::size_t num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);

  // Set up global vectors
  std::vector<std::vector<double>> stress_strain_vector(6);
  for (std::size_t i = 0; i < 6; i++) stress_strain_vector[i].reserve(num_visualization_points);

  // loop over myrank's beam elements and compute strain resultants for each visualization point
  for (std::size_t ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const DRT::Element* ele =
        discretization_->lRowElement(local_row_indices_beam_elements_[ibeamele]);

    // cast to SR beam element
    const auto* sr_beam = dynamic_cast<const DRT::ELEMENTS::Beam3r*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (sr_beam == nullptr)
      FOUR_C_THROW("Continuous cross section output only implemented for SR beams.");

    // get GP stress / strain values from previous element evaluation call
    for (std::size_t i = 0; i < 6; i++) stress_strain_GPs_current_element[i].clear();
    {
      switch (stress_strain_field)
      {
        case StressStrainField::material_strain:
          sr_beam->GetMaterialStrainResultantsAtAllGPs(stress_strain_GPs_current_element[0],
              stress_strain_GPs_current_element[1], stress_strain_GPs_current_element[2],
              stress_strain_GPs_current_element[3], stress_strain_GPs_current_element[4],
              stress_strain_GPs_current_element[5]);
          break;
        case StressStrainField::material_stress:
          sr_beam->GetMaterialStressResultantsAtAllGPs(stress_strain_GPs_current_element[0],
              stress_strain_GPs_current_element[1], stress_strain_GPs_current_element[2],
              stress_strain_GPs_current_element[3], stress_strain_GPs_current_element[4],
              stress_strain_GPs_current_element[5]);
          break;
        default:
          FOUR_C_THROW("Type of stress strain field not yet implemented.");
      }
    }

    // Calculate the interpolated coefficients
    CORE::FE::GaussRule1D force_int_rule =
        sr_beam->MyGaussRule(DRT::ELEMENTS::Beam3r::res_elastic_force);
    for (std::size_t i = 0; i < 3; i++)
      CalcInterpolationPolynomialCoefficients(
          force_int_rule, stress_strain_GPs_current_element[i], stress_strain_coefficients[i]);
    CORE::FE::GaussRule1D moment_int_rule =
        sr_beam->MyGaussRule(DRT::ELEMENTS::Beam3r::res_elastic_moment);
    for (std::size_t i = 3; i < 6; i++)
      CalcInterpolationPolynomialCoefficients(
          moment_int_rule, stress_strain_GPs_current_element[i], stress_strain_coefficients[i]);

    // loop over the chosen visualization points (equidistant distribution in the element
    // parameter space xi \in [-1,1] ) and determine its disp state
    double xi = 0.0;

    for (std::size_t ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      // store the information in vectors that can be interpreted by vtu writer and update number
      // of point data written
      for (std::size_t i = 0; i < 6; i++)
        stress_strain_vector[i].push_back(
            EvaluatePolynomialCoefficients(stress_strain_coefficients[i], xi));
    }
  }

  std::vector<std::string> field_names;
  switch (stress_strain_field)
  {
    case StressStrainField::material_strain:
      field_names = {"axial_strain", "shear_strain_2", "shear_strain_3", "twist", "curvature_2",
          "curvature_3"};
      break;
    case StressStrainField::material_stress:
      field_names = {"material_axial_force", "material_shear_force_2", "material_shear_force_3",
          "material_torque", "material_bending_moment_2", "material_bending_moment_3"};
      break;
    default:
      FOUR_C_THROW("Type of stress strain field not yet implemented.");
  }

  // finally append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->GetVisualizationData();
  for (std::size_t i = 0; i < 6; i++)
    visualization_data.SetPointDataVector(field_names[i], stress_strain_vector[i], 1);
}

FOUR_C_NAMESPACE_CLOSE
