/*-----------------------------------------------------------------------------------------------*/
/*!
\file discretization_runtime_vtu_writer.cpp

\brief Write visualization output for a discretization in vtk/vtu format at runtime

\level 3

\maintainer Maximilian Grill
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


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
DiscretizationRuntimeVtuWriter::DiscretizationRuntimeVtuWriter() :
    runtime_vtuwriter_( Teuchos::rcp( new RuntimeVtuWriter() ) )
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::Initialize(
    Teuchos::RCP<const DRT::Discretization> discretization,
    unsigned int max_number_timesteps_to_be_written,
    double time,
    bool write_binary_output )
{
  discretization_ = discretization;

  // determine path of output directory
  const std::string outputfilename( DRT::Problem::Instance()->OutputControlFile()->FileName() );

  size_t pos = outputfilename.find_last_of("/");

  if (pos == outputfilename.npos)
    pos = 0ul;
  else
    pos++;

  const std::string output_directory_path( outputfilename.substr(0ul, pos) );

  runtime_vtuwriter_->Initialize(
      discretization_->Comm().MyPID(),
      discretization_->Comm().NumProc(),
      max_number_timesteps_to_be_written,
      output_directory_path,
      DRT::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix(),
      discretization_->Name(),
      DRT::Problem::Instance()->OutputControlFile()->RestartName(),
      time,
      write_binary_output
      );

  // todo: if you want to use absolute positions, do this every time you write output
  // with current col displacements
  // fixme: if parallel distribution changes during simulation, do this each time this is the case
  SetGeometryFromDiscretizationStandard();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::SetGeometryFromDiscretizationStandard()
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
  for (unsigned int iele=0; iele<num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  // do not need to store connectivity indices here because we create a
  // contiguous array by the order in which we fill the coordinates (otherwise
  // need to adjust order of filling in the coordinates).
  std::vector<double>& point_coordinates = runtime_vtuwriter_->GetMutablePointCoordinateVector();
  point_coordinates.clear();
  point_coordinates.reserve( num_spatial_dimensions * num_nodes );

  std::vector<uint8_t>& cell_types = runtime_vtuwriter_->GetMutableCellTypeVector();
  cell_types.clear();
  cell_types.reserve(num_row_elements);

  std::vector<int32_t>& cell_offsets = runtime_vtuwriter_->GetMutableCellOffsetVector();
  cell_offsets.clear();
  cell_offsets.reserve(num_row_elements);


  // loop over my elements and collect the geometry/grid data
  unsigned int pointcounter = 0;
  unsigned int num_skipped_eles = 0;

  for (unsigned int iele=0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if ( beamele != NULL )
    {
      ++num_skipped_eles;
      continue;
    }


    cell_types.push_back(
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType( ele->Shape() ).first );

    const std::vector<int> & numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType( ele->Shape() ).second;

    const DRT::Node* const* nodes = ele->Nodes();

    for (int inode=0; inode<ele->NumNode(); ++inode)
      for (unsigned int idim=0; idim<num_spatial_dimensions; ++idim)
        point_coordinates.push_back( nodes[ numbering[inode] ]->X()[idim] );

    pointcounter += ele->NumNode();

    cell_offsets.push_back(pointcounter);
  }


  // safety checks
  if ( point_coordinates.size() != num_spatial_dimensions * pointcounter )
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d coordinate values, but got %d",
        num_spatial_dimensions * pointcounter, point_coordinates.size() );
  }

  if ( cell_types.size() != num_row_elements - num_skipped_eles )
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d cell type values, but got %d",
        num_row_elements, cell_types.size() );
  }

  if ( cell_offsets.size() != num_row_elements - num_skipped_eles )
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d cell offset values, but got %d",
        num_row_elements, cell_offsets.size() );
  }

}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::ResetTimeAndTimeStep(
    double time,
    unsigned int timestep)
{
  // Todo allow for independent setting of time/timestep and geometry name
  runtime_vtuwriter_->SetupForNewTimeStepAndGeometry( time, timestep, discretization_->Name() );
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::AppendDofBasedResultDataVector(
    const Teuchos::RCP<Epetra_Vector> & result_data_dofbased,
    unsigned int result_num_dofs_per_node,
    unsigned int read_result_data_from_dofindex,
    const std::string & resultname
    )
{
  /* the idea is to transform the given data to a 'point data vector' and append it to the
   * collected solution data vectors by calling AppendVisualizationPointDataVector() */

  // count number of nodes and number of elements for each processor
  unsigned int num_row_elements = (unsigned int) discretization_->NumMyRowElements();

  unsigned int num_nodes = 0;
  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
    num_nodes += discretization_->lRowElement(iele)->NumNode();

  std::vector<double> vtu_point_result_data;
  vtu_point_result_data.reserve( result_num_dofs_per_node * num_nodes );

  std::vector<int> nodedofs;
  unsigned int pointcounter = 0;

  for (unsigned int iele = 0; iele < num_row_elements; ++iele)
  {
    const DRT::Element* ele = discretization_->lRowElement(iele);

    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const DRT::ELEMENTS::Beam3Base* beamele = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(ele);

    // simply skip beam elements here (handled by BeamDiscretizationRuntimeVtuWriter)
    if ( beamele != NULL )
      continue;


    const std::vector<int> & numbering =
        DRT::ELEMENTS::GetVtkCellTypeFromBaciElementShapeType( ele->Shape() ).second;

    for (unsigned int inode = 0; inode < (unsigned int) ele->NumNode(); ++inode)
    {
      nodedofs.clear();

      // local storage position of desired dof gid
      discretization_->Dof( ele->Nodes()[ numbering[inode] ], nodedofs );

      for (unsigned int idof = 0; idof < result_num_dofs_per_node; ++idof)
      {
        const int lid =
            result_data_dofbased->Map().LID( nodedofs[ idof + read_result_data_from_dofindex ] );

        if ( lid > -1 )
          vtu_point_result_data.push_back( (*result_data_dofbased)[lid] );
        else
        {
          // Fixme what about the 'fillzeros' flag? for now -> dserror
//          if( fillzeros )
//            vtu_point_result_data.push_back(0.);
//          else
            dserror("received illegal dof local id: %d", lid);
        }
      }

      // Fixme what about the following lines?
//      for (int d=numdf; d<ncomponents; ++d)
//        vtu_point_result_data.push_back(0.);
    }

    pointcounter += ele->NumNode();
  }

  // sanity check
  if ( vtu_point_result_data.size() != result_num_dofs_per_node * pointcounter )
  {
    dserror("DiscretizationRuntimeVtuWriter expected %d result values, but got %d",
        result_num_dofs_per_node * pointcounter, vtu_point_result_data.size() );
  }


  runtime_vtuwriter_->AppendVisualizationPointDataVector(
      vtu_point_result_data, result_num_dofs_per_node, resultname);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::AppendNodeBasedResultDataVector(
    const Teuchos::RCP<Epetra_MultiVector> & result_data_nodebased,
    unsigned int result_num_components_per_node,
    const std::string & resultname)
{
  // Todo
  /*  see PostVtuWriter::WriteDofResultStep() for implementation based on PostField
   *
   *  the idea is to transform the given data to a 'point data vector' and append it to the
   *  collected solution data vectors by calling AppendVisualizationPointDataVector()
   */

  dserror("not implemented yet");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::AppendElementBasedResultDataVector(
    const Teuchos::RCP<Epetra_MultiVector> & result_data_elementbased,
    unsigned int result_num_components_per_element,
    const std::string & resultname)
{
  // Todo
  /*  see PostVtuWriter::WriteDofResultStep() for implementation based on PostField
   *
   *  the idea is to transform the given data to a 'cell data vector' and append it to the
   *  collected solution data vectors by calling AppendVisualizationCellDataVector()
   */

  dserror("not implemented yet");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::WriteFiles()
{
  runtime_vtuwriter_->WriteFiles();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void
DiscretizationRuntimeVtuWriter::WriteCollectionFileOfAllWrittenFiles()
{
  runtime_vtuwriter_->WriteCollectionFileOfAllWrittenFiles(
      DRT::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix() +
      "-" + discretization_->Name() );
}