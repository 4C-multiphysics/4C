/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief Write visualization output for a discretization in vtk/vtu format at runtime

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

/* headers */
#include "discretization_runtime_vtu_writer.H"

#include "runtime_vtu_writer.H"

#include "io_control.H"

#include "../drt_lib/drt_element_vtk_cell_type_register.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_lib/drt_dserror.H"

#include "../drt_beam3/beam3_base.H"

#include "Epetra_FEVector.h"


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
DiscretizationRuntimeVtuWriter::DiscretizationRuntimeVtuWriter()
    : runtime_vtuwriter_(Teuchos::rcp(new RuntimeVtuWriter()))
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::Initialize(
    Teuchos::RCP<const DRT::Discretization> discretization,
    unsigned int max_number_timesteps_to_be_written, double time, bool write_binary_output)
{
  discretization_ = discretization;

  // determine path of output directory
  const std::string outputfilename(DRT::Problem::Instance()->OutputControlFile()->FileName());

  size_t pos = outputfilename.find_last_of("/");

  if (pos == outputfilename.npos)
    pos = 0ul;
  else
    pos++;

  const std::string output_directory_path(outputfilename.substr(0ul, pos));

  runtime_vtuwriter_->Initialize(discretization_->Comm().MyPID(), discretization_->Comm().NumProc(),
      max_number_timesteps_to_be_written, output_directory_path,
      DRT::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix(), discretization_->Name(),
      DRT::Problem::Instance()->OutputControlFile()->RestartName(), time, write_binary_output);

  // todo: if you want to use absolute positions, do this every time you write output
  // with current col displacements
  SetGeometryFromDiscretizationStandard();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::SetGeometryFromDiscretizationStandard()
{
  /*  Note:
   *
   *  this will only work for 'standard' elements, whose discretization can directly be
   *  represented by one of the VTK cell types
   *  (see e.g. http://www.vtk.org/doc/nightly/html/vtkCellType_8h.html)
   *
   *  examples where it does not work:
   *    - beam elements using cubic Hermite polynomials for centerline interpolation
   *    - elements using NURBS
   *
   *  see method SetGeometryFromDiscretizationNonStandard() for how to handle these cases
   */

  // Todo assume 3D for now
  const unsigned int num_spatial_dimensions = 3;

  // count number of nodes and number for each processor; output is completely independent of
  // the number of processors involved
  unsigned int num_row_elements = discretization_->NumMyRowElements();
  unsigned int num_nodes = 0;
  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  // do not need to store connectivity indices here because we create a
  // contiguous array by the order in which we fill the coordinates (otherwise
  // need to adjust order of filling in the coordinates).
  std::vector<double>& point_coordinates = runtime_vtuwriter_->GetMutablePointCoordinateVector();
  point_coordinates.clear();
  point_coordinates.reserve(num_spatial_dimensions * num_nodes);

  std::vector<uint8_t>& cell_types = runtime_vtuwriter_->GetMutableCellTypeVector();
  cell_types.clear();
  cell_types.reserve(num_row_elements);

  std::vector<int32_t>& cell_offsets = runtime_vtuwriter_->GetMutableCellOffsetVector();
  cell_offsets.clear();
  cell_offsets.reserve(num_row_elements);


  // loop over my elements and collect the geometry/grid data
  unsigned int pointcounter = 0;
  unsigned int num_skipped_eles = 0;

  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if (beamele != NULL)
    {
      ++num_skipped_eles;
      continue;
    }

    cell_types.push_back(DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType(ele->Shape()).first);

    const std::vector<int>& numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType(ele->Shape()).second;

    const DRT::Node* const* nodes = ele->Nodes();

    for (int inode = 0; inode < ele->NumNode(); ++inode)
      for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
        point_coordinates.push_back(nodes[numbering[inode]]->X()[idim]);

    pointcounter += ele->NumNode();

    cell_offsets.push_back(pointcounter);
  }


  // safety checks
  if (point_coordinates.size() != num_spatial_dimensions * pointcounter)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d coordinate values, but got %d",
        num_spatial_dimensions * pointcounter, point_coordinates.size());
  }

  if (cell_types.size() != num_row_elements - num_skipped_eles)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d cell type values, but got %d",
        num_row_elements, cell_types.size());
  }

  if (cell_offsets.size() != num_row_elements - num_skipped_eles)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d cell offset values, but got %d",
        num_row_elements, cell_offsets.size());
  }

  // store node row and col maps (needed to check for changed parallel distribution)
  noderowmap_last_geometry_set_ = Teuchos::rcp(new Epetra_Map(*discretization_->NodeRowMap()));
  nodecolmap_last_geometry_set_ = Teuchos::rcp(new Epetra_Map(*discretization_->NodeColMap()));
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::ResetTimeAndTimeStep(double time, unsigned int timestep)
{
  // check if parallel distribution of discretization changed
  int map_changed = ((not noderowmap_last_geometry_set_->SameAs(*discretization_->NodeRowMap())) or
                     (not nodecolmap_last_geometry_set_->SameAs(*discretization_->NodeColMap())));
  int map_changed_allproc(0);
  discretization_->Comm().MaxAll(&map_changed, &map_changed_allproc, 1);

  // reset geometry of runtime vtu writer
  if (map_changed_allproc) SetGeometryFromDiscretizationStandard();

  // Todo allow for independent setting of time/timestep and geometry name
  runtime_vtuwriter_->SetupForNewTimeStepAndGeometry(time, timestep, discretization_->Name());
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendDofBasedResultDataVector(
    const Teuchos::RCP<Epetra_Vector>& result_data_dofbased, unsigned int result_num_dofs_per_node,
    unsigned int read_result_data_from_dofindex, const std::string& resultname)
{
  /* the idea is to transform the given data to a 'point data vector' and append it to the
   * collected solution data vectors by calling AppendVisualizationPointDataVector() */

  // safety checks
  if (!discretization_->DofColMap()->SameAs(result_data_dofbased->Map()))
  {
    dserror(
        "DiscretizationRuntimeVtpWriter: Received DofBasedResult's map does not match the "
        "discretization's dof col map.");
  }

  // count number of nodes and number of elements for each processor
  unsigned int num_row_elements = (unsigned int)discretization_->NumMyRowElements();

  unsigned int num_nodes = 0;
  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  std::vector<double> vtu_point_result_data;
  vtu_point_result_data.reserve(result_num_dofs_per_node * num_nodes);

  std::vector<int> nodedofs;
  unsigned int pointcounter = 0;

  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if (beamele != NULL) continue;


    const std::vector<int>& numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType(ele->Shape()).second;

    for (unsigned int inode = 0; inode < (unsigned int)ele->NumNode(); ++inode)
    {
      nodedofs.clear();

      // local storage position of desired dof gid
      discretization_->Dof(ele->Nodes()[numbering[inode]], nodedofs);

      // adjust resultdofs according to elements dof
      if (nodedofs.size() < result_num_dofs_per_node)
      {
        result_num_dofs_per_node = nodedofs.size();
      }

      for (unsigned int idof = 0; idof < result_num_dofs_per_node; ++idof)
      {
        const int lid =
            result_data_dofbased->Map().LID(nodedofs[idof + read_result_data_from_dofindex]);

        if (lid > -1)
          vtu_point_result_data.push_back((*result_data_dofbased)[lid]);
        else
          dserror("received illegal dof local id: %d", lid);
      }
    }

    pointcounter += ele->NumNode();
  }

  // sanity check
  if (vtu_point_result_data.size() != result_num_dofs_per_node * pointcounter)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d result values, but got %d",
        result_num_dofs_per_node * pointcounter, vtu_point_result_data.size());
  }

  runtime_vtuwriter_->AppendVisualizationPointDataVector(
      vtu_point_result_data, result_num_dofs_per_node, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendNodeBasedResultDataVector(
    const Teuchos::RCP<Epetra_MultiVector>& result_data_nodebased,
    unsigned int result_num_components_per_node, const std::string& resultname)
{
  /* the idea is to transform the given data to a 'point data vector' and append it to the
   * collected solution data vectors by calling AppendVisualizationPointDataVector() */

  // safety checks
  if ((unsigned int)result_data_nodebased->NumVectors() != result_num_components_per_node)
    dserror(
        "DiscretizationRuntimeVtpWriter: expected Epetra_MultiVector with %d columns but got %d",
        result_num_components_per_node, result_data_nodebased->NumVectors());

  if (!discretization_->NodeColMap()->SameAs(result_data_nodebased->Map()))
  {
    dserror(
        "DiscretizationRuntimeVtpWriter: Received NodeBasedResult's map does not match the "
        "discretization's node col map.");
  }

  // count number of nodes and number of elements for each processor
  unsigned int num_row_elements = (unsigned int)discretization_->NumMyRowElements();

  unsigned int num_nodes = 0;
  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  std::vector<double> vtu_point_result_data;
  vtu_point_result_data.reserve(result_num_components_per_node * num_nodes);

  unsigned int pointcounter = 0;

  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if (beamele != NULL) continue;

    const std::vector<int>& numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType(ele->Shape()).second;

    for (unsigned int inode = 0; inode < (unsigned int)ele->NumNode(); ++inode)
    {
      const DRT::Node* node = ele->Nodes()[numbering[inode]];

      const int lid = node->LID();

      for (unsigned int icpn = 0; icpn < result_num_components_per_node; ++icpn)
      {
        Epetra_Vector* column = (*result_data_nodebased)(icpn);

        if (lid > -1)
          vtu_point_result_data.push_back((*column)[lid]);
        else
          dserror("received illegal node local id: %d", lid);
      }
    }

    pointcounter += ele->NumNode();
  }

  // sanity check
  if (vtu_point_result_data.size() != result_num_components_per_node * pointcounter)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d result values, but got %d",
        result_num_components_per_node * pointcounter, vtu_point_result_data.size());
  }

  runtime_vtuwriter_->AppendVisualizationPointDataVector(
      vtu_point_result_data, result_num_components_per_node, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendElementBasedResultDataVector(
    const Teuchos::RCP<Epetra_MultiVector>& result_data_elementbased,
    unsigned int result_num_components_per_element, const std::string& resultname)
{
  /* the idea is to transform the given data to a 'cell data vector' and append it to the
   *  collected solution data vectors by calling AppendVisualizationCellDataVector() */

  // safety check
  if ((unsigned int)result_data_elementbased->NumVectors() != result_num_components_per_element)
    dserror(
        "DiscretizationRuntimeVtpWriter: expected Epetra_MultiVector with %d columns but got %d",
        result_num_components_per_element, result_data_elementbased->NumVectors());

  if (!discretization_->ElementRowMap()->SameAs(result_data_elementbased->Map()))
  {
    dserror(
        "DiscretizationRuntimeVtpWriter: Received ElementBasedResult's map does not match the "
        "discretization's element row map.");
  }

  // count number of elements for each processor
  unsigned int num_row_elements = (unsigned int)discretization_->NumMyRowElements();

  std::vector<double> vtu_cell_result_data;
  vtu_cell_result_data.reserve(result_num_components_per_element * num_row_elements);

  unsigned int cellcounter = 0;

  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if (beamele != NULL) continue;

    for (unsigned int icpe = 0; icpe < result_num_components_per_element; ++icpe)
    {
      Epetra_Vector* column = (*result_data_elementbased)(icpe);

      vtu_cell_result_data.push_back((*column)[iele]);
    }

    ++cellcounter;
  }

  // sanity check
  if (vtu_cell_result_data.size() != result_num_components_per_element * cellcounter)
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d result values, but got %d",
        result_num_components_per_element * cellcounter, vtu_cell_result_data.size());
  }

  runtime_vtuwriter_->AppendVisualizationCellDataVector(
      vtu_cell_result_data, result_num_components_per_element, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendElementOwner(const std::string resultname)
{
  // Vector with element owner for elements in the row map.
  std::vector<double> owner_of_row_elements;
  owner_of_row_elements.reserve(discretization_->NumMyRowElements());

  const int my_pid = discretization_->Comm().MyPID();
  for (int iele = 0; iele < discretization_->NumMyRowElements(); ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // Since we do not output beam elements we filter them here.
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    if (beamele != nullptr) continue;

    owner_of_row_elements.push_back(my_pid);
  }

  // Pass data to the output writer.
  runtime_vtuwriter_->AppendVisualizationCellDataVector(owner_of_row_elements, 1, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendElementGID(const std::string& resultname)
{
  // Vector with element IDs for elements in the row map.
  std::vector<double> gid_of_row_elements;
  gid_of_row_elements.reserve(discretization_->NumMyRowElements());

  for (int iele = 0; iele < discretization_->NumMyRowElements(); ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // Since we do not output beam elements we filter them here.
    auto beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    if (beamele != nullptr) continue;

    gid_of_row_elements.push_back(ele->Id());
  }

  // Pass data to the output writer.
  runtime_vtuwriter_->AppendVisualizationCellDataVector(gid_of_row_elements, 1, resultname);
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendElementGhostingInformation()
{
  IO::AppendElementGhostingInformation(discretization_, runtime_vtuwriter_);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::AppendNodeGID(const std::string& resultname)
{
  // count number of nodes and number for each processor; output is completely independent of
  // the number of processors involved
  int num_row_elements = discretization_->NumMyRowElements();
  int num_nodes = 0;
  for (int iele = 0; iele < num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  // Setup the vector with the GIDs of the nodes.
  std::vector<double> gid_of_nodes;
  gid_of_nodes.reserve(num_nodes);

  // Loop over each element and add the node GIDs.
  for (int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    auto beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    if (beamele != nullptr) continue;

    // Add the node GIDs.
    const std::vector<int>& numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType(ele->Shape()).second;
    const DRT::Node* const* nodes = ele->Nodes();
    for (int inode = 0; inode < ele->NumNode(); ++inode)
      gid_of_nodes.push_back(nodes[numbering[inode]]->Id());
  }

  runtime_vtuwriter_->AppendVisualizationPointDataVector(gid_of_nodes, 1, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::WriteFiles() { runtime_vtuwriter_->WriteFiles(); }

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DiscretizationRuntimeVtuWriter::WriteCollectionFileOfAllWrittenFiles()
{
  runtime_vtuwriter_->WriteCollectionFileOfAllWrittenFiles(
      DRT::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix() + "-" +
      discretization_->Name());
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void IO::AppendElementGhostingInformation(
    const Teuchos::RCP<const DRT::Discretization>& discretization,
    const Teuchos::RCP<RuntimeVtuWriter>& runtime_vtuwriter, bool is_beam)
{
  // Set up a multivector which will be populated with all ghosting informations.
  const Epetra_Comm& comm = discretization->ElementColMap()->Comm();
  const int n_proc = comm.NumProc();
  const int my_proc = comm.MyPID();

  // Create Vectors to store the ghosting information.
  Epetra_FEVector ghosting_information(*discretization->ElementRowMap(), n_proc);

  // Get elements ghosted by this rank.
  std::vector<int> my_ghost_elements;
  my_ghost_elements.clear();
  int count = 0;
  for (int iele = 0; iele < discretization->NumMyColElements(); ++iele)
  {
    const DRT::Element* ele = discretization->lColElement(iele);
    const auto* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    const bool is_beam_element = beamele != nullptr;
    if ((is_beam_element and is_beam) or ((not is_beam_element) and (not is_beam)))
    {
      count++;
      if (ele->Owner() != my_proc) my_ghost_elements.push_back(ele->Id());
    }
  }

  // Add to the multi vector.
  std::vector<double> values(my_ghost_elements.size(), 1.0);
  ghosting_information.SumIntoGlobalValues(
      my_ghost_elements.size(), my_ghost_elements.data(), values.data(), my_proc);

  // Assemble over all processors.
  ghosting_information.GlobalAssemble();

  // Output the ghosting data of the elements owned by this proc.
  std::vector<double> ghosted_elements;
  ghosted_elements.reserve(count * n_proc);
  for (int iele = 0; iele < discretization->NumMyRowElements(); ++iele)
  {
    const DRT::Element* ele = discretization->lRowElement(iele);
    const auto* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);
    const bool is_beam_element = beamele != nullptr;
    if ((is_beam_element and is_beam) or ((not is_beam_element) and (not is_beam)))
    {
      const int local_row = ghosting_information.Map().LID(ele->Id());
      if (local_row == -1) dserror("The element has to exist in the row map.");
      for (int i_proc = 0; i_proc < n_proc; i_proc++)
        ghosted_elements.push_back(ghosting_information[i_proc][local_row]);
    }
  }

  runtime_vtuwriter->AppendVisualizationCellDataVector(
      ghosted_elements, n_proc, "element_ghosting");
}
