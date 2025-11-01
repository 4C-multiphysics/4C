// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_discretization_runtime_vtu_writer.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamDiscretizationRuntimeOutputWriter::BeamDiscretizationRuntimeOutputWriter(
    Core::IO::VisualizationParameters parameters, MPI_Comm comm)
    : visualization_manager_(std::make_shared<Core::IO::VisualizationManager>(
          std::move(parameters), comm, "structure-beams")),
      use_absolute_positions_(true)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::initialize(
    std::shared_ptr<Core::FE::Discretization> discretization,
    bool use_absolute_positions_for_point_coordinates, const unsigned int n_subsegments,
    std::shared_ptr<const Core::Geo::MeshFree::BoundingBox> const& periodic_boundingbox)
{
  discretization_ = discretization;
  use_absolute_positions_ = use_absolute_positions_for_point_coordinates;
  periodic_boundingbox_ = periodic_boundingbox;
  n_subsegments_ = n_subsegments;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::set_geometry_from_beam_discretization(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
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
  local_row_indices_beam_elements_.reserve(discretization_->num_my_row_elements());
  for (unsigned int iele = 0;
      iele < static_cast<unsigned int>(discretization_->num_my_row_elements()); ++iele)
  {
    const Core::Elements::Element* ele = discretization_->l_row_element(iele);

    // check for beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

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
  auto& visualization_data = visualization_manager_->get_visualization_data();

  // Clear the visualization data, especially the cell connectivity such that it is rebuilt later.
  visualization_data.clear_data();

  std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
  point_coordinates.reserve(num_spatial_dimensions * num_visualization_points);

  std::vector<uint8_t>& cell_types = visualization_data.get_cell_types();
  cell_types.reserve(num_beam_row_elements);

  std::vector<int32_t>& cell_offsets = visualization_data.get_cell_offsets();
  cell_offsets.reserve(num_beam_row_elements);

  std::vector<bool> dummy_shift_in_dim(3, false);

  // loop over my elements and collect the geometry/grid data
  unsigned int pointcounter = 0;

  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    std::vector<double> beamelement_displacement_vector;

    if (use_absolute_positions_)
    {
      // this is needed in case your input file contains shifted/cut elements
      if (periodic_boundingbox_ != nullptr)
      {
        BeamInteraction::Utils::get_current_unshifted_element_dis(*discretization_, ele,
            displacement_state_vector, *periodic_boundingbox_, beamelement_displacement_vector);
      }
      // this is needed in case your input file does not contain shifted/cut elements
      else
      {
        BeamInteraction::Utils::get_current_element_dis(
            *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);
      }
    }

    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine their interpolated (initial) positions r */
    Core::LinAlg::Matrix<3, 1> interpolated_position(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> interpolated_position_priorpoint(Core::LinAlg::Initialization::zero);
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      interpolated_position.clear();
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      if (use_absolute_positions_)
        beamele->get_pos_at_xi(interpolated_position, xi, beamelement_displacement_vector);
      else
        beamele->get_ref_pos_at_xi(interpolated_position, xi);

      if (periodic_boundingbox_ != nullptr) periodic_boundingbox_->shift_3d(interpolated_position);

      Core::LinAlg::Matrix<3, 1> unshift_interpolated_position = interpolated_position;

      // check if an element is cut by a periodic boundary
      bool shift = false;
      if (periodic_boundingbox_ != nullptr)
        shift = periodic_boundingbox_->check_if_shift_between_points(
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
    FOUR_C_THROW("RuntimeVtuWriter expected {} cell type values, but got {}", num_beam_row_elements,
        cell_types.size());
  }

  if (periodic_boundingbox_ != nullptr and !periodic_boundingbox_->have_pbc() and
      (point_coordinates.size() != num_spatial_dimensions * num_visualization_points))
  {
    FOUR_C_THROW("RuntimeVtuWriter expected {} coordinate values, but got {}",
        num_spatial_dimensions * num_visualization_points, point_coordinates.size());
  }

  if (periodic_boundingbox_ != nullptr and !periodic_boundingbox_->have_pbc() and
      (cell_offsets.size() != num_beam_row_elements))
  {
    FOUR_C_THROW("RuntimeVtuWriter expected {} cell offset values, but got {}",
        num_beam_row_elements, cell_offsets.size());
  }
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_displacement_field(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->get_visualization_data();

  // triads only make sense in 3D
  const unsigned int num_spatial_dimensions = 3;

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);
  std::vector<int32_t> const& cell_offsets = visualization_data.get_cell_offsets();

  // disp vector
  std::vector<double> displacement_vector;
  displacement_vector.reserve(num_spatial_dimensions * num_visualization_points);

  // number of points so far
  int points_sofar = 0;

  // loop over myrank's beam elements and compute disp for each visualization point
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BeamInteraction::Utils::get_current_element_dis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);

    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine its disp state */
    Core::LinAlg::Matrix<3, 1> pos_visualization_point;
    Core::LinAlg::Matrix<3, 1> refpos_visualization_point;
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      pos_visualization_point.clear();
      refpos_visualization_point.clear();

      // interpolate
      beamele->get_ref_pos_at_xi(refpos_visualization_point, xi);
      beamele->get_pos_at_xi(pos_visualization_point, xi, beamelement_displacement_vector);

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
  visualization_data.set_point_data_vector(
      "displacement", displacement_vector, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_triad_field(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->get_visualization_data();

  // triads only make sense in 3D
  const unsigned int num_spatial_dimensions = 3;

  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  unsigned int num_visualization_points = num_beam_row_elements * (n_subsegments_ + 1);
  std::vector<int32_t> const& cell_offsets = visualization_data.get_cell_offsets();

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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BeamInteraction::Utils::get_current_element_dis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);


    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine the triad */
    Core::LinAlg::Matrix<3, 3> triad_visualization_point;
    double xi = 0.0;

    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      triad_visualization_point.clear();

      beamele->get_triad_at_xi(triad_visualization_point, xi, beamelement_displacement_vector);

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
  visualization_data.set_point_data_vector("base_vector_1", base_vector_1, num_spatial_dimensions);
  visualization_data.set_point_data_vector("base_vector_2", base_vector_2, num_spatial_dimensions);
  visualization_data.set_point_data_vector("base_vector_3", base_vector_3, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_owning_processor()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i) owner.push_back(ele->owner());
  }

  // set the solution vector in the visualization data container
  visualization_manager_->get_visualization_data().set_cell_data_vector("element_owner", owner, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_gid()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // vector with the IDs of the beams on this processor.
  std::vector<int> gid;
  gid.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i) gid.push_back(ele->id());
  }

  // append the solution vector to the visualization data
  visualization_manager_->get_visualization_data().set_cell_data_vector("element_gid", gid, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_ghosting_information()
{
  const auto only_select_beam_elements = [](const Core::Elements::Element* ele)
  { return dynamic_cast<const Discret::Elements::Beam3Base*>(ele); };
  Core::IO::append_element_ghosting_information(
      *discretization_, *visualization_manager_, only_select_beam_elements);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_internal_energy()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);


    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      e_int.push_back(beamele->get_internal_energy());
  }

  // append the solution vector to the visualization data
  visualization_manager_->get_visualization_data().set_cell_data_vector(
      "element_internal_energy", e_int, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_kinetic_energy()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);


    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");
#endif

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      e_kin.push_back(beamele->get_kinetic_energy());
  }

  // append the solution vector to the visualization data
  visualization_manager_->get_visualization_data().set_cell_data_vector(
      "element_kinetic_energy", e_kin, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_filament_id_and_type()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();

  // processor owning the element
  std::vector<double> id, type;
  id.reserve(num_beam_row_elements);
  type.reserve(num_beam_row_elements);

  std::vector<const Core::Conditions::Condition*> conditions;
  discretization_->get_condition("BeamLineFilamentCondition", conditions);
  const auto beam_line_filament_conditioned_nodes =
      Core::Conditions::find_conditioned_node_ids_and_conditions(
          *discretization_, conditions, Core::Conditions::LookFor::locally_owned_and_ghosted);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    // get filament number (note so far only one filament for each element and node)
    FOUR_C_ASSERT_ALWAYS(beam_line_filament_conditioned_nodes.count(ele->nodes()[0]->id()) == 1,
        " No filament number assigned to element with gid {} .", ele->id());

    const Core::Conditions::Condition* cond =
        beam_line_filament_conditioned_nodes.find(ele->nodes()[0]->id())->second;
    double current_id = cond->parameters().get<int>("ID");
    double current_type = Inpar::BeamInteraction::string_to_filament_type(
        (cond->parameters().get<std::string>("TYPE")));

    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
    {
      id.push_back(current_id);
      type.push_back(current_type);
    }
  }

  // append the solution vector to the visualization data
  auto& visualization_data = visualization_manager_->get_visualization_data();
  visualization_data.set_cell_data_vector("ele_filament_id", id, 1);
  visualization_data.set_cell_data_vector("ele_filament_type", type, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_circular_cross_section_radius()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    // this needs to be done for all cells that make up a cut element
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      cross_section_radius.push_back(beamele->get_circular_cross_section_radius_for_interactions());
  }

  // append the solution vector to the visualization data
  visualization_manager_->get_visualization_data().set_cell_data_vector(
      "cross_section_radius", cross_section_radius, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_point_circular_cross_section_information_vector(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
{
  auto& visualization_data = visualization_manager_->get_visualization_data();

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
  std::vector<int32_t> const& cell_offsets = visualization_data.get_cell_offsets();

  // number of points so far
  int points_sofar = 0;

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");


    const double circular_cross_section_radius =
        beamele->get_circular_cross_section_radius_for_interactions();

    // get the displacement state vector for this element
    std::vector<double> beamelement_displacement_vector;

    BeamInteraction::Utils::get_current_element_dis(
        *discretization_, ele, displacement_state_vector, beamelement_displacement_vector);


    /* loop over the chosen visualization points (equidistant distribution in the element
     * parameter space xi \in [-1,1] ) and determine the triad */
    Core::LinAlg::Matrix<3, 3> triad_visualization_point;
    double xi = 0.0;
    for (unsigned int ipoint = 0; ipoint < n_subsegments_ + 1; ++ipoint)
    {
      xi = -1.0 + ipoint * 2.0 / n_subsegments_;

      triad_visualization_point.clear();

      beamele->get_triad_at_xi(triad_visualization_point, xi, beamelement_displacement_vector);

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
  visualization_data.set_point_data_vector("circular_cross_section_information_vector",
      circular_cross_section_information_vector, num_spatial_dimensions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    append_gauss_point_material_cross_section_strain_resultants()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

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
    beamele->get_material_strain_resultants_at_all_gps(axial_strain_GPs_current_element,
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
      insert_vector_values_at_back_of_other_vector(
          axial_strain_GPs_current_element, axial_strain_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          shear_strain_2_GPs_current_element, shear_strain_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          shear_strain_3_GPs_current_element, shear_strain_3_GPs_all_row_elements);


      insert_vector_values_at_back_of_other_vector(
          twist_GPs_current_element, twist_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          curvature_2_GPs_current_element, curvature_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          curvature_3_GPs_current_element, curvature_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_strains_translational);
  int global_num_GPs_per_element_rotational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_strains_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->get_visualization_data();

  visualization_data.set_cell_data_vector("axial_strain_GPs", axial_strain_GPs_all_row_elements,
      global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("shear_strain_2_GPs", shear_strain_2_GPs_all_row_elements,
      global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("shear_strain_3_GPs", shear_strain_3_GPs_all_row_elements,
      global_num_GPs_per_element_translational);


  visualization_data.set_cell_data_vector(
      "twist_GPs", twist_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector(
      "curvature_2_GPs", curvature_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector(
      "curvature_3_GPs", curvature_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    append_gauss_point_material_cross_section_strain_resultants_continuous()
{
  append_continuous_stress_strain_resultants(StressStrainField::material_strain);
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    append_gauss_point_material_cross_section_stress_resultants()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

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
    beamele->get_material_stress_resultants_at_all_gps(material_axial_force_GPs_current_element,
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
      insert_vector_values_at_back_of_other_vector(
          material_axial_force_GPs_current_element, material_axial_force_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          material_shear_force_2_GPs_current_element, material_shear_force_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          material_shear_force_3_GPs_current_element, material_shear_force_3_GPs_all_row_elements);


      insert_vector_values_at_back_of_other_vector(
          material_torque_GPs_current_element, material_torque_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(material_bending_moment_2_GPs_current_element,
          material_bending_moment_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(material_bending_moment_3_GPs_current_element,
          material_bending_moment_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_stresses_translational);
  int global_num_GPs_per_element_rotational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_stresses_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->get_visualization_data();

  visualization_data.set_cell_data_vector("material_axial_force_GPs",
      material_axial_force_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("material_shear_force_2_GPs",
      material_shear_force_2_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("material_shear_force_3_GPs",
      material_shear_force_3_GPs_all_row_elements, global_num_GPs_per_element_translational);


  visualization_data.set_cell_data_vector("material_torque_GPs",
      material_torque_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector("material_bending_moment_2_GPs",
      material_bending_moment_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector("material_bending_moment_3_GPs",
      material_bending_moment_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    append_gauss_point_material_cross_section_stress_resultants_continuous()
{
  append_continuous_stress_strain_resultants(StressStrainField::material_stress);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::
    append_gauss_point_spatial_cross_section_stress_resultants()
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

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
    beamele->get_spatial_stress_resultants_at_all_gps(spatial_axial_force_GPs_current_element,
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
      insert_vector_values_at_back_of_other_vector(
          spatial_axial_force_GPs_current_element, spatial_axial_force_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          spatial_shear_force_2_GPs_current_element, spatial_shear_force_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(
          spatial_shear_force_3_GPs_current_element, spatial_shear_force_3_GPs_all_row_elements);


      insert_vector_values_at_back_of_other_vector(
          spatial_torque_GPs_current_element, spatial_torque_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(spatial_bending_moment_2_GPs_current_element,
          spatial_bending_moment_2_GPs_all_row_elements);

      insert_vector_values_at_back_of_other_vector(spatial_bending_moment_3_GPs_current_element,
          spatial_bending_moment_3_GPs_all_row_elements);
    }
  }


  int global_num_GPs_per_element_translational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_stresses_translational);
  int global_num_GPs_per_element_rotational =
      get_global_number_of_gauss_points_per_beam(num_GPs_per_element_stresses_rotational);


  // append the solution vectors to the visualization data of the vtu writer object
  auto& visualization_data = visualization_manager_->get_visualization_data();

  visualization_data.set_cell_data_vector("spatial_axial_force_GPs",
      spatial_axial_force_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("spatial_shear_force_2_GPs",
      spatial_shear_force_2_GPs_all_row_elements, global_num_GPs_per_element_translational);

  visualization_data.set_cell_data_vector("spatial_shear_force_3_GPs",
      spatial_shear_force_3_GPs_all_row_elements, global_num_GPs_per_element_translational);


  visualization_data.set_cell_data_vector("spatial_torque_GPs", spatial_torque_GPs_all_row_elements,
      global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector("spatial_bending_moment_2_GPs",
      spatial_bending_moment_2_GPs_all_row_elements, global_num_GPs_per_element_rotational);

  visualization_data.set_cell_data_vector("spatial_bending_moment_3_GPs",
      spatial_bending_moment_3_GPs_all_row_elements, global_num_GPs_per_element_rotational);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_orientation_parameter(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // length of element is approximated linearly, as also the direction of a element is calculated
    // linearly independent of centerline interpolation
    Core::LinAlg::Matrix<3, 1> dirvec(Core::LinAlg::Initialization::zero);

    std::vector<double> pos(2, 0.0);
    for (int dim = 0; dim < 3; ++dim)
    {
      pos[0] =
          ele->nodes()[0]->x()[dim] +
          displacement_state_vector.local_values_as_span()[displacement_state_vector.get_map().lid(
              discretization_->dof(ele->nodes()[0])[dim])];
      pos[1] =
          ele->nodes()[1]->x()[dim] +
          displacement_state_vector.local_values_as_span()[displacement_state_vector.get_map().lid(
              discretization_->dof(ele->nodes()[1])[dim])];
      dirvec(dim) = pos[1] - pos[0];
    }

    // current element length (linear)
    double curr_lin_ele_length = dirvec.norm2();

    // loop over all base vectors for orientation index x,y and z
    Core::LinAlg::Matrix<3, 1> unit_base_vec(Core::LinAlg::Initialization::zero);
    std::vector<double> curr_ele_orientation_parameter(3, 0.0);
    for (int unsigned ibase = 0; ibase < 3; ++ibase)
    {
      // init current base vector
      unit_base_vec.clear();
      unit_base_vec(ibase) = 1.0;

      double cos_squared = dirvec.dot(unit_base_vec) / curr_lin_ele_length;
      cos_squared *= cos_squared;

      curr_ele_orientation_parameter[ibase] = cos_squared;
      local_orientation_parameter[ibase] += curr_lin_ele_length * cos_squared;
    }

    local_accumulated_ele_lengths += curr_lin_ele_length;

    // in case of cut elements by a periodic boundary
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      insert_vector_values_at_back_of_other_vector(
          curr_ele_orientation_parameter, orientation_parameter_for_each_element);
  }

  // calculate length of all (linear) elements
  double global_linear_filament_length = 0;
  global_linear_filament_length =
      Core::Communication::sum_all(local_accumulated_ele_lengths, discretization_->get_comm());

  //
  for (int unsigned i = 0; i < 3; ++i)
    local_orientation_parameter[i] /= global_linear_filament_length;

  // calculate global orientation parameter
  std::vector<double> global_orientation_parameter(3, 0.0);
  global_orientation_parameter =
      Core::Communication::sum_all(local_orientation_parameter, discretization_->get_comm());

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      insert_vector_values_at_back_of_other_vector(
          global_orientation_parameter, orientation_parameter_for_global_network);

  auto& visualization_data = visualization_manager_->get_visualization_data();

  // append the solution vector to the visualization data
  visualization_data.set_cell_data_vector(
      "orientation_parameter_element", orientation_parameter_for_each_element, 3);

  // append the solution vector to the visualization data
  visualization_data.set_cell_data_vector(
      "orientation_parameter", orientation_parameter_for_global_network, 3);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_rve_crosssection_forces(
    const Core::LinAlg::Vector<double>& displacement_state_vector)
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
  Core::LinAlg::Matrix<3, 1> pos_node_1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> pos_node_2(Core::LinAlg::Initialization::zero);

  // create pseudo planes through center of RVE (like this it also works if
  // your box is not periodic, i.e. you do not have cut element on the box edges)
  Core::LinAlg::Matrix<3, 2> box(Core::LinAlg::Initialization::zero);
  if (periodic_boundingbox_ != nullptr)
  {
    for (unsigned dim = 0; dim < 3; ++dim)
    {
      box(dim, 0) = periodic_boundingbox_->box()(dim, 0);
      box(dim, 1) =
          periodic_boundingbox_->box()(dim, 0) + 0.5 * periodic_boundingbox_->edge_length(dim);
    }
  }

  Core::LinAlg::Matrix<3, 1> xi_intersect(Core::LinAlg::Initialization::zero);

  // loop over all my elements and build force sum of myrank's cut element
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    BeamInteraction::Utils::get_current_element_dis(
        *discretization_, ele, displacement_state_vector, beamelement_shift_displacement_vector);
    BeamInteraction::Utils::get_current_unshifted_element_dis(*discretization_, ele,
        displacement_state_vector, *periodic_boundingbox_, beamelement_displacement_vector);

    beamele->get_pos_at_xi(pos_node_1, -1.0, beamelement_displacement_vector);
    beamele->get_pos_at_xi(pos_node_2, 1.0, beamelement_displacement_vector);
    periodic_boundingbox_->get_xi_of_intersection_3d(pos_node_1, pos_node_2, xi_intersect, box);

    // todo: change from just using first gauss point to linear inter-/extrapolation
    // between two closest gauss points

    spatial_x_force_GPs_current_element.clear();
    spatial_y_force_2_GPs_current_element.clear();
    spatial_z_force_3_GPs_current_element.clear();

    for (int dir = 0; dir < 3; ++dir)
    {
      if (xi_intersect(dir) > 1.0) continue;

      beamele->get_spatial_forces_at_all_gps(spatial_x_force_GPs_current_element,
          spatial_y_force_2_GPs_current_element, spatial_z_force_3_GPs_current_element);

      fint_sum[dir][0] += spatial_x_force_GPs_current_element[0];
      fint_sum[dir][1] += spatial_y_force_2_GPs_current_element[0];
      fint_sum[dir][2] += spatial_z_force_3_GPs_current_element[0];
    }
  }

  std::vector<std::vector<double>> global_sum(3, std::vector<double>(3, 0.0));
  for (int dir = 0; dir < 3; ++dir)
    global_sum[dir] = Core::Communication::sum_all(fint_sum[dir], discretization_->get_comm());

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
  auto& visualization_data = visualization_manager_->get_visualization_data();
  visualization_data.set_cell_data_vector(
      "sum_spatial_force_rve_crosssection_xdir", sum_spatial_force_rve_crosssection_xdir, 3);
  visualization_data.set_cell_data_vector(
      "sum_spatial_force_rve_crosssection_ydir", sum_spatial_force_rve_crosssection_ydir, 3);
  visualization_data.set_cell_data_vector(
      "sum_spatial_force_rve_crosssection_zdir", sum_spatial_force_rve_crosssection_zdir, 3);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_element_elastic_energy()
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
  //    const Core::Elements::Element* ele = discretization_->lRowElement(iele);
  //
  //    // check for beam element
  //    const Discret::Elements::Beam3Base* beamele = dynamic_cast<const
  //    Discret::Elements::Beam3Base*>(ele);
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
void BeamDiscretizationRuntimeOutputWriter::append_ref_length()
{
  // determine number of row BEAM elements for each processor
  // output is completely independent of the number of processors involved
  unsigned int num_beam_row_elements = local_row_indices_beam_elements_.size();
  std::vector<double> ref_lengths;
  ref_lengths.reserve(num_beam_row_elements);

  // loop over my elements and collect the data about triads/base vectors
  for (unsigned int ibeamele = 0; ibeamele < num_beam_row_elements; ++ibeamele)
  {
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to beam element
    auto beamele = dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    if (beamele == nullptr)
      FOUR_C_THROW("BeamDiscretizationRuntimeOutputWriter expects a beam element here!");

    // this needs to be done for all cells that make up a cut element
    for (int i = 0; i < num_cells_per_element_[ibeamele]; ++i)
      ref_lengths.push_back(beamele->ref_length());
  }

  // append the solution vector to the visualization data
  visualization_manager_->get_visualization_data().set_cell_data_vector(
      "ref_length", ref_lengths, 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::write_to_disk(
    const double visualization_time, const int visualization_step)
{
  visualization_manager_->write_to_disk(visualization_time, visualization_step);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::insert_vector_values_at_back_of_other_vector(
    const std::vector<double>& vector_input, std::vector<double>& vector_output)
{
  vector_output.reserve(vector_output.size() + vector_input.size());

  std::copy(vector_input.begin(), vector_input.end(), std::back_inserter(vector_output));
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
int BeamDiscretizationRuntimeOutputWriter::get_global_number_of_gauss_points_per_beam(
    unsigned int my_num_gp) const
{
  int my_num_gp_signed = (int)my_num_gp;
  int global_num_gp = 0;
  global_num_gp = Core::Communication::max_all(my_num_gp_signed, discretization_->get_comm());

  // Safety checks.
  if (my_num_gp_signed > 0 and my_num_gp_signed != global_num_gp)
    FOUR_C_THROW("The number of Gauss points must be the same for all elements in discretization!");
  else if (global_num_gp < 0)
    FOUR_C_THROW("The number of Gauss points must be zero or a positive integer!");

  return global_num_gp;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::calc_interpolation_polynomial_coefficients(
    const Core::FE::GaussRule1D& gauss_rule, const std::vector<double>& gauss_point_values,
    std::vector<double>& polynomial_coefficients) const
{
  // Get the coefficients for the interpolation functions at the Gauss points.
  std::size_t n_gp = 3;
  std::array<std::array<double, 3>, 3> lagrange_coefficients;
  switch (gauss_rule)
  {
    case Core::FE::GaussRule1D::line_3point:
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
    case Core::FE::GaussRule1D::line_lobatto3point:
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
double BeamDiscretizationRuntimeOutputWriter::evaluate_polynomial_coefficients(
    const std::vector<double>& polynomial_coefficients, const double& xi) const
{
  double interpolated_value = 0.0;
  for (std::size_t p = 0; p < polynomial_coefficients.size(); p++)
    interpolated_value += polynomial_coefficients[p] * pow(xi, p);
  return interpolated_value;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamDiscretizationRuntimeOutputWriter::append_continuous_stress_strain_resultants(
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
    const Core::Elements::Element* ele =
        discretization_->l_row_element(local_row_indices_beam_elements_[ibeamele]);

    // cast to SR beam element
    const auto* sr_beam = dynamic_cast<const Discret::Elements::Beam3r*>(ele);

    // Todo safety check for now, may be removed when better tested
    if (sr_beam == nullptr)
      FOUR_C_THROW("Continuous cross section output only implemented for SR beams.");

    // get GP stress / strain values from previous element evaluation call
    for (std::size_t i = 0; i < 6; i++) stress_strain_GPs_current_element[i].clear();
    {
      switch (stress_strain_field)
      {
        case StressStrainField::material_strain:
          sr_beam->get_material_strain_resultants_at_all_gps(stress_strain_GPs_current_element[0],
              stress_strain_GPs_current_element[1], stress_strain_GPs_current_element[2],
              stress_strain_GPs_current_element[3], stress_strain_GPs_current_element[4],
              stress_strain_GPs_current_element[5]);
          break;
        case StressStrainField::material_stress:
          sr_beam->get_material_stress_resultants_at_all_gps(stress_strain_GPs_current_element[0],
              stress_strain_GPs_current_element[1], stress_strain_GPs_current_element[2],
              stress_strain_GPs_current_element[3], stress_strain_GPs_current_element[4],
              stress_strain_GPs_current_element[5]);
          break;
        default:
          FOUR_C_THROW("Type of stress strain field not yet implemented.");
      }
    }

    // Calculate the interpolated coefficients
    Core::FE::GaussRule1D force_int_rule =
        sr_beam->my_gauss_rule(Discret::Elements::Beam3r::res_elastic_force);
    for (std::size_t i = 0; i < 3; i++)
      calc_interpolation_polynomial_coefficients(
          force_int_rule, stress_strain_GPs_current_element[i], stress_strain_coefficients[i]);
    Core::FE::GaussRule1D moment_int_rule =
        sr_beam->my_gauss_rule(Discret::Elements::Beam3r::res_elastic_moment);
    for (std::size_t i = 3; i < 6; i++)
      calc_interpolation_polynomial_coefficients(
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
            evaluate_polynomial_coefficients(stress_strain_coefficients[i], xi));
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
  auto& visualization_data = visualization_manager_->get_visualization_data();
  for (std::size_t i = 0; i < 6; i++)
    visualization_data.set_point_data_vector(field_names[i], stress_strain_vector[i], 1);
}

FOUR_C_NAMESPACE_CLOSE
