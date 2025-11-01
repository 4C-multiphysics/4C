// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_vtk_vtu_writer.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io_element_vtk_cell_type_register.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_post_common.hpp"
#include "4C_utils_exceptions.hpp"

#include <sstream>

FOUR_C_NAMESPACE_OPEN

//! Number of linear subsegments used for visualization of beam centerline.
#define BEAMSVTUVISUALSUBSEGMENTS 5


PostVtuWriter::PostVtuWriter(PostField* field, const std::string& filename)
    : PostVtkWriter(field, filename), proc_file_padding_(ceil(log10(numproc_)))
{
}

const std::string& PostVtuWriter::writer_string() const
{
  static std::string name("UnstructuredGrid");
  return name;
}

const std::string& PostVtuWriter::writer_opening_tag() const
{
  static std::string tag("<UnstructuredGrid>");
  return tag;
}

const std::string& PostVtuWriter::writer_p_opening_tag() const
{
  static std::string tag("<PUnstructuredGrid GhostLevel=\"0\">");
  return tag;
}

const std::vector<std::string>& PostVtuWriter::writer_p_piece_tags() const
{
  static std::vector<std::string> tags;
  tags.clear();
  for (size_t i = 0; i < numproc_; ++i)
  {
    std::stringstream stream;
    stream << "<Piece Source=\"" << filenamebase_ << "-" << std::setfill('0')
           << std::setw(proc_file_padding_) << i << ".vtu\"/>";
    tags.push_back(std::string(stream.str()));
  }
  return tags;
}

const std::string& PostVtuWriter::writer_suffix() const
{
  static std::string name(".vtu");
  return name;
}

const std::string& PostVtuWriter::writer_p_suffix() const
{
  static std::string name(".pvtu");
  return name;
}

void PostVtuWriter::write_geo()
{
  using namespace FourC;

  std::shared_ptr<Core::FE::Discretization> dis = this->get_field()->discretization();

  // count number of nodes and number for each processor; output is completely independent of
  // the number of processors involved
  int nelements = dis->num_my_row_elements();
  int nnodes = 0;
  for (int e = 0; e < nelements; ++e) nnodes += dis->l_row_element(e)->num_node();

  // do not need to store connectivity indices here because we create a
  // contiguous array by the order in which we fill the coordinates (otherwise
  // need to adjust filling in the coordinates).
  std::vector<double> coordinates;
  coordinates.reserve(3 * nnodes);
  std::vector<uint8_t> celltypes;
  celltypes.reserve(nelements);
  std::vector<int32_t> celloffset;
  celloffset.reserve(nelements);

  // loop over my elements and write the data
  int outNodeId = 0;
  for (int e = 0; e < nelements; ++e)
  {
    const Core::Elements::Element* ele = dis->l_row_element(e);
    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    if (Core::FE::is_nurbs_celltype(ele->shape()))
    {
      write_geo_nurbs_ele(ele, celltypes, outNodeId, celloffset, coordinates);
    }
    else if (beamele != nullptr)
    {
      write_geo_beam_ele(beamele, celltypes, outNodeId, celloffset, coordinates);
    }
    else
    {
      celltypes.push_back(Core::IO::get_vtk_cell_type_from_element_cell_type(ele->shape()).first);
      const std::vector<int>& numbering =
          Core::IO::get_vtk_cell_type_from_element_cell_type(ele->shape()).second;
      const Core::Nodes::Node* const* nodes = ele->nodes();
      for (int n = 0; n < ele->num_node(); ++n)
        for (int d = 0; d < 3; ++d) coordinates.push_back(nodes[numbering[n]]->x()[d]);
      outNodeId += ele->num_node();
      celloffset.push_back(outNodeId);
    }
  }
  FOUR_C_ASSERT((int)coordinates.size() == 3 * outNodeId, "internal error");

  // step 1: write node coordinates into file
  currentout_ << "<Piece NumberOfPoints=\"" << outNodeId << "\" NumberOfCells=\"" << nelements
              << "\" >\n"
              << "  <Points>\n"
              << "    <DataArray type=\"Float64\" NumberOfComponents=\"3\"";

  if (write_binary_output_)
  {
    currentout_ << " format=\"binary\">\n";
    LibB64::write_compressed_block(coordinates, currentout_);
  }
  else
  {
    currentout_ << " format=\"ascii\">\n";
    int counter = 1;
    for (std::vector<double>::const_iterator it = coordinates.begin(); it != coordinates.end();
        ++it)
    {
      currentout_ << std::setprecision(15) << std::scientific << *it << " ";
      // dimension is hard coded to three, thus
      if (counter % 3 == 0) currentout_ << '\n';
      counter++;
    }
    currentout_ << std::resetiosflags(std::ios::scientific);
  }


  currentout_ << "    </DataArray>\n"
              << "  </Points>\n\n";

  // avoid too much memory consumption -> clear coordinates vector now that we're done
  {
    std::vector<double> empty;
    empty.swap(coordinates);
  }

  // step 2: write mesh-node topology into file. we assumed contiguous order of coordinates
  // in this format, so fill the vector only here
  currentout_ << "  <Cells>\n"
              << "    <DataArray type=\"Int32\" Name=\"connectivity\"";
  if (write_binary_output_)
    currentout_ << " format=\"binary\">\n";
  else
    currentout_ << " format=\"ascii\">\n";

  {
    std::vector<int32_t> connectivity;
    connectivity.reserve(outNodeId);
    for (int i = 0; i < outNodeId; ++i) connectivity.push_back(i);
    if (write_binary_output_)
      LibB64::write_compressed_block(connectivity, currentout_);
    else
    {
      for (std::vector<int32_t>::const_iterator it = connectivity.begin(); it != connectivity.end();
          ++it)
        currentout_ << *it << " ";
    }
  }
  currentout_ << "    </DataArray>\n";

  // step 3: write start indices for individual cells
  currentout_ << "    <DataArray type=\"Int32\" Name=\"offsets\"";
  if (write_binary_output_)
  {
    currentout_ << " format=\"binary\">\n";
    LibB64::write_compressed_block(celloffset, currentout_);
  }
  else
  {
    currentout_ << " format=\"ascii\">\n";
    for (std::vector<int32_t>::const_iterator it = celloffset.begin(); it != celloffset.end(); ++it)
      currentout_ << *it << " ";
  }

  currentout_ << "    </DataArray>\n";

  // step 4: write cell types
  currentout_ << "    <DataArray type=\"UInt8\" Name=\"types\"";
  if (write_binary_output_)
  {
    currentout_ << " format=\"binary\">\n";
    LibB64::write_compressed_block(celltypes, currentout_);
  }
  else
  {
    currentout_ << " format=\"ascii\">\n";
    for (std::vector<uint8_t>::const_iterator it = celltypes.begin(); it != celltypes.end(); ++it)
      currentout_ << (unsigned int)*it << " ";
  }
  currentout_ << "    </DataArray>\n";

  currentout_ << "  </Cells>\n\n";

  if (myrank_ == 0)
  {
    currentmasterout_ << "    <PPoints>\n";
    currentmasterout_ << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n";
    currentmasterout_ << "    </PPoints>\n";
  }
}



void PostVtuWriter::write_dof_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from,
    const bool fillzeros)
{
  using namespace FourC;

  if (myrank_ == 0 && timestep_ == 0) std::cout << "writing dof-based field " << name << std::endl;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();

  // For parallel computations, we need to access all dofs on the elements, including the
  // nodes owned by other processors. Therefore, we need to import that data here.
  const Core::LinAlg::Map& vecmap = data->get_map();
  const Core::LinAlg::Map* colmap = dis->dof_col_map(0);

  int offset = vecmap.min_all_gid() - dis->dof_row_map()->min_all_gid();
  if (fillzeros) offset = 0;

  std::shared_ptr<Core::LinAlg::Vector<double>> ghostedData;
  if (colmap->same_as(vecmap))
    ghostedData = data;
  else
  {
    // There is one more complication: The map of the vector and the map governed by the
    // degrees of freedom in the discretization might be offset (e.g. pressure for fluid).
    // Therefore, we need to adjust the numbering in the vector to the numbering in the
    // discretization by the variable 'offset'.
    std::vector<int> gids(vecmap.num_my_elements());
    for (int i = 0; i < vecmap.num_my_elements(); ++i)
      gids[i] = vecmap.my_global_elements()[i] - offset;
    Core::LinAlg::Map rowmap(
        vecmap.num_global_elements(), vecmap.num_my_elements(), gids.data(), 0, vecmap.get_comm());
    std::shared_ptr<Core::LinAlg::Vector<double>> dofvec =
        Core::LinAlg::create_vector(rowmap, false);
    for (int i = 0; i < vecmap.num_my_elements(); ++i)
      (*dofvec).get_values()[i] = (*data).get_values()[i];

    ghostedData = Core::LinAlg::create_vector(*colmap, true);
    Core::LinAlg::export_to(*dofvec, *ghostedData);
  }

  int ncomponents = numdf;
  if (numdf > 1 && numdf == field_->problem()->num_dim()) ncomponents = 3;

  // count number of nodes and number of elements for each processor
  int nnodes = 0;
  for (int e = 0; e < dis->num_my_row_elements(); ++e) nnodes += dis->l_row_element(e)->num_node();

  std::vector<double> solution;
  solution.reserve(ncomponents * nnodes);

  std::vector<int> nodedofs;
  for (int e = 0; e < dis->num_my_row_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_row_element(e);
    // check for beam element that potentially needs special treatment due to Hermite interpolation
    const Discret::Elements::Beam3Base* beamele =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

    if (Core::FE::is_nurbs_celltype(ele->shape()))
    {
      write_dof_result_step_nurbs_ele(
          ele, ncomponents, numdf, solution, ghostedData, from, fillzeros);
    }
    else if (beamele != nullptr)
    {
      write_dof_result_step_beam_ele(
          beamele, ncomponents, numdf, solution, ghostedData, from, fillzeros);
    }
    else
    {
      const std::vector<int>& numbering =
          Core::IO::get_vtk_cell_type_from_element_cell_type(ele->shape()).second;

      for (int n = 0; n < ele->num_node(); ++n)
      {
        nodedofs.clear();

        // local storage position of desired dof gid
        dis->dof(ele->nodes()[numbering[n]], nodedofs);

        for (int d = 0; d < numdf; ++d)
        {
          const int lid = ghostedData->get_map().lid(nodedofs[d + from]);
          if (lid > -1)
            solution.push_back(ghostedData->local_values_as_span()[lid]);
          else
          {
            if (fillzeros)
              solution.push_back(0.);
            else
              FOUR_C_THROW("received illegal dof local id: {}", lid);
          }
        }
        for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.);
      }
    }

  }  // loop over all elements
     //  FOUR_C_ASSERT((int)solution.size() == ncomponents*nnodes, "internal error");

  // start the scalar fields that will later be written
  if (currentPhase_ == INIT)
  {
    currentout_ << "  <PointData>\n";  // Scalars=\"scalars\">\n";
    if (myrank_ == 0)
    {
      currentmasterout_ << "    <PPointData>\n";  // Scalars=\"scalars\">\n";
    }
    currentPhase_ = POINTS;
  }

  if (currentPhase_ != POINTS)
    FOUR_C_THROW(
        "Cannot write point data at this stage. Most likely cell and point data fields are mixed.");

  this->write_solution_vector(solution, ncomponents, name, file);
}



void PostVtuWriter::write_nodal_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf)
{
  using namespace FourC;

  if (myrank_ == 0 && timestep_ == 0) std::cout << "writing node-based field " << name << std::endl;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();

  // Here is the only thing we need to do for parallel computations: We need read access to all dofs
  // on the row elements, so need to get the NodeColMap to have this access
  const Core::LinAlg::Map* colmap = dis->node_col_map();
  const Core::LinAlg::Map& vecmap = data->get_map();

  FOUR_C_ASSERT(colmap->max_all_gid() == vecmap.max_all_gid() &&
                    colmap->min_all_gid() == vecmap.min_all_gid(),
      "Given data vector does not seem to match discretization node map");

  std::shared_ptr<Core::LinAlg::MultiVector<double>> ghostedData;
  if (colmap->same_as(vecmap))
    ghostedData = data;
  else
  {
    ghostedData =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*colmap, data->num_vectors(), false);
    Core::LinAlg::export_to(*data, *ghostedData);
  }

  int ncomponents = numdf;
  if (numdf > 1 && numdf == field_->problem()->num_dim()) ncomponents = 3;


  // count number of nodes and number of elements for each processor
  int nnodes = 0;
  for (int e = 0; e < dis->num_my_row_elements(); ++e) nnodes += dis->l_row_element(e)->num_node();

  std::vector<double> solution;
  solution.reserve(ncomponents * nnodes);

  for (int e = 0; e < dis->num_my_row_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_row_element(e);

    if (Core::FE::is_nurbs_celltype(ele->shape()))
    {
      write_nodal_result_step_nurbs_ele(ele, ncomponents, numdf, solution, ghostedData);
    }
    else
    {
      const std::vector<int>& numbering =
          Core::IO::get_vtk_cell_type_from_element_cell_type(ele->shape()).second;
      for (int n = 0; n < ele->num_node(); ++n)
      {
        for (int idf = 0; idf < numdf; ++idf)
        {
          // Core::LinAlg::Vector<double>* column = (*ghostedData)(idf);
          Core::LinAlg::Vector<double> column((*ghostedData)(idf));
          int lid = ghostedData->get_map().lid(ele->nodes()[numbering[n]]->id());

          if (lid > -1)
            solution.push_back(column.local_values_as_span()[lid]);
          else
          {
            FOUR_C_THROW("received illegal node local id: {}", lid);
          }
        }
        for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.);
      }
    }
  }  // loop over all elements
     //  FOUR_C_ASSERT((int)solution.size() == ncomponents*nnodes, "internal error");


  // start the scalar fields that will later be written
  if (currentPhase_ == INIT)
  {
    currentout_ << "  <PointData>\n";  // Scalars=\"scalars\">\n";
    if (myrank_ == 0)
    {
      currentmasterout_ << "    <PPointData>\n";  // Scalars=\"scalars\">\n";
    }
    currentPhase_ = POINTS;
  }

  if (currentPhase_ != POINTS)
    FOUR_C_THROW(
        "Cannot write point data at this stage. Most likely cell and point data fields are mixed.");

  this->write_solution_vector(solution, ncomponents, name, file);
}



void PostVtuWriter::write_element_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from)
{
  using namespace FourC;

  if (myrank_ == 0 && timestep_ == 0)
    std::cout << "writing element-based field " << name << std::endl;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();

  int ncomponents = numdf;
  if (numdf > 1 && numdf == field_->problem()->num_dim()) ncomponents = 3;
  int neles = dis->num_my_row_elements();

  std::vector<double> solution;
  solution.reserve(ncomponents * neles);

  const int numcol = data->num_vectors();
  if (numdf + from > numcol)
    FOUR_C_THROW("violated column range of Core::LinAlg::MultiVector<double>: {}", numcol);

  std::shared_ptr<Core::LinAlg::MultiVector<double>> importedData;
  if (dis->element_row_map()->same_as(data->get_map()))
    importedData = data;
  else
  {
    importedData = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *dis->element_row_map(), data->num_vectors(), false);
    Core::LinAlg::export_to(*data, *importedData);
  }

  for (int e = 0; e < dis->num_my_row_elements(); ++e)
  {
    for (int d = 0; d < numdf; ++d)
    {
      Core::LinAlg::Vector<double> column((*importedData)(d + from));
      solution.push_back(column.local_values_as_span()[e]);
    }
    for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.0);
  }
  FOUR_C_ASSERT((int)solution.size() == ncomponents * neles, "internal error");

  // start the scalar fields that will later be written
  if (currentPhase_ == POINTS)
  {
    // end the scalar fields
    currentout_ << "  </PointData>\n";
    if (myrank_ == 0)
    {
      currentmasterout_ << "    </PPointData>\n";
    }
  }

  if (currentPhase_ == INIT || currentPhase_ == POINTS)
  {
    currentout_ << "  <CellData>\n";  // Scalars=\"scalars\">\n";
    if (myrank_ == 0)
    {
      currentmasterout_ << "    <PCellData>\n";  // Scalars=\"scalars\">\n";
    }
    currentPhase_ = CELLS;
  }

  if (currentPhase_ != CELLS)
    FOUR_C_THROW(
        "Cannot write cell data at this stage. Most likely cell and point data fields are mixed.");

  this->write_solution_vector(solution, ncomponents, name, file);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void PostVtuWriter::write_geo_nurbs_ele(const Core::Elements::Element* ele,
    std::vector<uint8_t>& celltypes, int& outNodeId, std::vector<int32_t>& celloffset,
    std::vector<double>& coordinates) const
{
  using namespace FourC;

  switch (ele->shape())
  {
    case Core::FE::CellType::nurbs2:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs2>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs3>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    case Core::FE::CellType::nurbs4:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs4>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs9>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    case Core::FE::CellType::nurbs8:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs8>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    case Core::FE::CellType::nurbs27:
    {
      write_geo_nurbs_ele<Core::FE::CellType::nurbs27>(
          ele, celltypes, outNodeId, celloffset, coordinates);
      break;
    }
    default:
      FOUR_C_THROW("VTK output not yet implemented for given NURBS element");
  }

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType nurbs_type>
void PostVtuWriter::write_geo_nurbs_ele(const Core::Elements::Element* ele,
    std::vector<uint8_t>& celltypes, int& outNodeId, std::vector<int32_t>& celloffset,
    std::vector<double>& coordinates) const
{
  using namespace FourC;

  const unsigned NUMNODES = Core::FE::num_nodes(nurbs_type);
  const unsigned DIM = Core::FE::dim<nurbs_type>;

  const Core::FE::CellType mapped_dis_type = map_nurbs_dis_type_to_lagrange_dis_type(nurbs_type);

  const std::vector<int>& numbering =
      Core::IO::get_vtk_cell_type_from_element_cell_type(mapped_dis_type).second;

  std::shared_ptr<const Core::FE::Discretization> dis = this->get_field()->discretization();

  celltypes.push_back(Core::IO::get_vtk_cell_type_from_element_cell_type(mapped_dis_type).first);

  Core::LinAlg::Matrix<NUMNODES, 1> weights;
  const Core::Nodes::Node* const* nodes = ele->nodes();
  for (unsigned inode = 0; inode < NUMNODES; inode++)
  {
    const Core::FE::Nurbs::ControlPoint* cp =
        dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

    weights(inode) = cp->w();
  }
  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(dis.get());

  if (not nurbsdis) FOUR_C_THROW("Cast to NURBS discretization failed.\n");

  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);

  const bool zero_ele = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, ele->id());

  if (zero_ele) return;

  for (int n = 0; n < ele->num_node(); ++n)
  {
    Core::LinAlg::Matrix<NUMNODES, 1> funct;
    Core::LinAlg::Matrix<DIM, NUMNODES> deriv;  // dummy
    Core::LinAlg::Matrix<3, 1> gpa;

    gpa = Core::FE::get_node_coordinates(numbering[n], mapped_dis_type);

    Core::FE::Nurbs::nurbs_get_funct_deriv(funct, deriv, gpa, myknots, weights, nurbs_type);

    std::array<double, 3> X = {0., 0., 0.};
    for (unsigned i = 0; i < 3; ++i)
      for (unsigned m = 0; m < NUMNODES; ++m) X[i] += funct(m) * (nodes[m]->x()[i]);

    for (unsigned d = 0; d < 3; ++d) coordinates.push_back(X[d]);
  }
  outNodeId += ele->num_node();
  celloffset.push_back(outNodeId);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::CellType PostVtuWriter::map_nurbs_dis_type_to_lagrange_dis_type(
    const Core::FE::CellType nurbs_dis_type) const
{
  using namespace FourC;

  switch (nurbs_dis_type)
  {
    case Core::FE::CellType::nurbs2:
      return Core::FE::CellType::line2;
    case Core::FE::CellType::nurbs3:
      return Core::FE::CellType::line3;
    case Core::FE::CellType::nurbs4:
      return Core::FE::CellType::quad4;
    case Core::FE::CellType::nurbs9:
      return Core::FE::CellType::quad9;
    case Core::FE::CellType::nurbs8:
      return Core::FE::CellType::hex8;
    case Core::FE::CellType::nurbs27:
      return Core::FE::CellType::hex27;
    default:
      FOUR_C_THROW("No known mapping from NURBS to Lagrange.");
  }
}

void PostVtuWriter::write_geo_beam_ele(const Discret::Elements::Beam3Base* beamele,
    std::vector<uint8_t>& celltypes, int& outNodeId, std::vector<int32_t>& celloffset,
    std::vector<double>& coordinates)
{
  using namespace FourC;

  /* visualize the beam centerline as a sequence of straight line segments (POLY_LINE)
   * which is supported as vtkCellType number 4 (see also list in GetVtkElementType) */
  celltypes.push_back(4);

  /* loop over the chosen visualization points (equidistant distribution in the element
   * parameter space xi \in [-1,1] ) and determine their interpolated initial positions r */
  Core::LinAlg::Matrix<3, 1> r;
  double xi = 0.0;

  for (unsigned int i = 0; i < BEAMSVTUVISUALSUBSEGMENTS + 1; ++i)
  {
    r.clear();
    xi = -1.0 + i * 2.0 / BEAMSVTUVISUALSUBSEGMENTS;

    beamele->get_ref_pos_at_xi(r, xi);

    for (int d = 0; d < 3; ++d) coordinates.push_back(r(d));
  }

  outNodeId += BEAMSVTUVISUALSUBSEGMENTS + 1;
  celloffset.push_back(outNodeId);
}

void PostVtuWriter::write_dof_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::Vector<double>> ghostedData, const int from,
    const bool fillzeros) const
{
  using namespace FourC;

  switch (ele->shape())
  {
    case Core::FE::CellType::nurbs2:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs2>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs3>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    case Core::FE::CellType::nurbs4:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs4>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs9>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    case Core::FE::CellType::nurbs8:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs8>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    case Core::FE::CellType::nurbs27:
    {
      write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs27>(
          ele, ncomponents, numdf, solution, *ghostedData, from, fillzeros);
      break;
    }
    default:
      FOUR_C_THROW("VTK output not yet implemented for given NURBS element");
      break;
  }  // end switch shape

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType nurbs_type>
void PostVtuWriter::write_dof_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    Core::LinAlg::Vector<double>& ghostedData, const int from, const bool fillzeros) const
{
  using namespace FourC;

  const unsigned NUMNODES = Core::FE::num_nodes(nurbs_type);
  const unsigned DIM = Core::FE::dim<nurbs_type>;

  const std::shared_ptr<const Core::FE::Discretization> dis = field_->discretization();
  std::vector<int> nodedofs;

  const Core::FE::CellType mapped_dis_type = map_nurbs_dis_type_to_lagrange_dis_type(nurbs_type);

  const std::vector<int>& numbering =
      Core::IO::get_vtk_cell_type_from_element_cell_type(mapped_dis_type).second;

  Core::LinAlg::Matrix<NUMNODES, 1> weights;
  const Core::Nodes::Node* const* nodes = ele->nodes();
  for (unsigned inode = 0; inode < NUMNODES; inode++)
  {
    const Core::FE::Nurbs::ControlPoint* cp =
        dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

    weights(inode) = cp->w();
  }
  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(dis.get());

  if (not nurbsdis) FOUR_C_THROW("Cast to NURBS discretization failed.\n");

  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
  const bool zero_ele = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, ele->id());

  if (zero_ele) return;

  for (unsigned n = 0; n < NUMNODES; ++n)
  {
    Core::LinAlg::Matrix<NUMNODES, 1> funct;
    Core::LinAlg::Matrix<DIM, NUMNODES> deriv;
    Core::LinAlg::Matrix<3, 1> gpa;

    gpa = Core::FE::get_node_coordinates(numbering[n], mapped_dis_type);

    Core::FE::Nurbs::nurbs_get_funct_deriv(funct, deriv, gpa, myknots, weights, nurbs_type);

    std::vector<double> val(numdf, 0.0);

    for (unsigned m = 0; m < NUMNODES; ++m)
    {
      nodedofs.clear();
      dis->dof(ele->nodes()[m], nodedofs);
      for (int d = 0; d < numdf; ++d)
      {
        const int lid = ghostedData.get_map().lid(nodedofs[d + from]);
        if (lid > -1)
          val[d] += funct(m) * ghostedData.local_values_as_span()[lid];
        else
        {
          if (fillzeros)
            val[d] += 0.;
          else
            FOUR_C_THROW("received illegal dof local id: {}", lid);
        }
      }
    }

    for (int d = 0; d < numdf; ++d) solution.push_back(val[d]);

    for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.);
  }
}


void PostVtuWriter::write_dof_result_step_beam_ele(const Discret::Elements::Beam3Base* beamele,
    const int& ncomponents, const int& numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::Vector<double>>& ghostedData, const int& from,
    const bool fillzeros)
{
  using namespace FourC;

  if (numdf != ncomponents or numdf != 3)
  {
    FOUR_C_THROW(
        "writing of dof-based result for beams with Hermite centerline interpolation is "
        "restricted to centerline displacements where numdof = ncomponents = 3");
  }

  const std::shared_ptr<Core::FE::Discretization> dis = this->get_field()->discretization();
  std::vector<int> nodedofs;
  std::vector<double> elementdofvals;

  for (int n = 0; n < beamele->num_node(); ++n)
  {
    nodedofs.clear();

    // local storage position of desired dof gid
    dis->dof(beamele->nodes()[n], nodedofs);

    for (std::vector<int>::const_iterator it = nodedofs.begin(); it != nodedofs.end(); ++it)
    {
      const int lid = ghostedData->get_map().lid(*it);
      if (lid > -1)
        elementdofvals.push_back(ghostedData->local_values_as_span()[lid]);
      else
      {
        if (fillzeros)
          elementdofvals.push_back(0.);
        else
          FOUR_C_THROW("received illegal dof local id: {}", lid);
      }
    }
  }

  /* visualize the beam centerline as a sequence of straight line segments (POLY_LINE)
   * which is supported as vtkCellType number 4 (see also list in GetVtkElementType)
   * loop over the chosen visualization points (equidistant distribution in the element
   * parameter space xi \in [-1,1] ) and determine their interpolated initial positions r */
  Core::LinAlg::Matrix<3, 1> pos, refpos;
  double xi = 0.0;

  for (unsigned int i = 0; i < BEAMSVTUVISUALSUBSEGMENTS + 1; ++i)
  {
    pos.clear();
    refpos.clear();
    xi = -1.0 + i * 2.0 / BEAMSVTUVISUALSUBSEGMENTS;

    // let the beam element do the interpolation
    beamele->get_ref_pos_at_xi(refpos, xi);
    beamele->get_pos_at_xi(pos, xi, elementdofvals);

    for (int d = 0; d < 3; ++d) solution.push_back(pos(d) - refpos(d));
  }
}

void PostVtuWriter::write_nodal_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> ghostedData) const
{
  using namespace FourC;

  switch (ele->shape())
  {
    case Core::FE::CellType::nurbs2:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs2>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs3>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    case Core::FE::CellType::nurbs4:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs4>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs9>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    case Core::FE::CellType::nurbs8:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs8>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    case Core::FE::CellType::nurbs27:
    {
      write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs27>(
          ele, ncomponents, numdf, solution, *ghostedData);
      break;
    }
    default:
      FOUR_C_THROW("VTK output not yet implemented for given NURBS element");
      break;
  }  // end switch ele shape
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType nurbs_type>
void PostVtuWriter::write_nodal_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    Core::LinAlg::MultiVector<double>& ghostedData) const
{
  using namespace FourC;

  const unsigned NUMNODES = Core::FE::num_nodes(nurbs_type);
  const unsigned DIM = Core::FE::dim<nurbs_type>;

  const std::shared_ptr<const Core::FE::Discretization> dis = field_->discretization();
  std::vector<int> nodedofs;

  const Core::FE::CellType mapped_dis_type = map_nurbs_dis_type_to_lagrange_dis_type(nurbs_type);

  const std::vector<int>& numbering =
      Core::IO::get_vtk_cell_type_from_element_cell_type(mapped_dis_type).second;

  Core::LinAlg::Matrix<NUMNODES, 1> weights;
  const Core::Nodes::Node* const* nodes = ele->nodes();
  for (unsigned inode = 0; inode < NUMNODES; inode++)
  {
    const Core::FE::Nurbs::ControlPoint* cp =
        dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

    weights(inode) = cp->w();
  }

  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(dis.get());

  if (not nurbsdis) FOUR_C_THROW("Cast to NURBS discretization failed.\n");

  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
  bool zero_ele = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, ele->id());
  if (zero_ele) return;
  std::vector<double> val(numdf);
  for (unsigned n = 0; n < NUMNODES; ++n)
  {
    Core::LinAlg::Matrix<NUMNODES, 1> funct;
    Core::LinAlg::Matrix<DIM, NUMNODES> deriv;
    Core::LinAlg::Matrix<3, 1> gpa;

    gpa = Core::FE::get_node_coordinates(numbering[n], mapped_dis_type);

    Core::FE::Nurbs::nurbs_get_funct_deriv(funct, deriv, gpa, myknots, weights, nurbs_type);

    for (int i = 0; i < numdf; ++i) val[i] = 0.;

    for (int idf = 0; idf < numdf; ++idf)
    {
      // Core::LinAlg::Vector<double>* column = (*ghostedData)(idf);
      Core::LinAlg::Vector<double> column((ghostedData)(idf));
      for (unsigned m = 0; m < NUMNODES; ++m)
      {
        int lid = ghostedData.get_map().lid(ele->nodes()[m]->id());
        if (lid > -1)
          val[idf] += funct(m) * column.local_values_as_span()[lid];
        else
          FOUR_C_THROW("received illegal node local id: {}", lid);
      }
    }

    for (int d = 0; d < numdf; ++d) solution.push_back(val[d]);

    for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.);
  }
}

/*----------------------------------------------------------------------------*/
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs2>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs3>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs4>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs9>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs8>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;
template void PostVtuWriter::write_geo_nurbs_ele<Core::FE::CellType::nurbs27>(
    const Core::Elements::Element* ele, std::vector<uint8_t>& celltypes, int& outNodeId,
    std::vector<int32_t>& celloffset, std::vector<double>& coordinates) const;

/*----------------------------------------------------------------------------*/
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs2>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs3>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs4>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs9>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs8>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;
template void PostVtuWriter::write_dof_result_step_nurbs_ele<Core::FE::CellType::nurbs27>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::Vector<double>& ghostedData, const int from,
    const bool fillzeros) const;

/*----------------------------------------------------------------------------*/
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs2>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs3>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs4>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs9>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs8>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;
template void PostVtuWriter::write_nodal_result_step_nurbs_ele<Core::FE::CellType::nurbs27>(
    const Core::Elements::Element* ele, int ncomponents, const int numdf,
    std::vector<double>& solution, Core::LinAlg::MultiVector<double>& ghostedData) const;

FOUR_C_NAMESPACE_CLOSE
