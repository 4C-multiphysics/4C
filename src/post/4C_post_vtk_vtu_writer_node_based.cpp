// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_vtk_vtu_writer_node_based.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io_element_vtk_cell_type_register.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_post_common.hpp"
#include "4C_post_vtk_vtu_writer.hpp"
#include "4C_utils_exceptions.hpp"

#include <sstream>

FOUR_C_NAMESPACE_OPEN

PostVtuWriterNode::PostVtuWriterNode(PostField* field, const std::string& filename)
    : PostVtuWriter(field, filename)
{
  static_assert(29 == static_cast<int>(Core::FE::CellType::max_distype),
      "The number of element types defined by Core::FE::CellType does not match the "
      "number of element types supported by the post vtu filter.");
  if (myrank_ != 0) FOUR_C_THROW("Node based filtering only works in serial mode");
}


const std::string& PostVtuWriterNode::writer_string() const
{
  static std::string name("UnstructuredGrid");
  return name;
}

const std::string& PostVtuWriterNode::writer_opening_tag() const
{
  static std::string tag("<UnstructuredGrid>");
  return tag;
}

const std::string& PostVtuWriterNode::writer_p_opening_tag() const
{
  static std::string tag("<PUnstructuredGrid GhostLevel=\"0\">");
  return tag;
}

const std::vector<std::string>& PostVtuWriterNode::writer_p_piece_tags() const
{
  static std::vector<std::string> tags;
  tags.clear();
  for (size_t i = 0; i < numproc_; ++i)
  {
    std::stringstream stream;
    stream << "<Piece Source=\"" << filenamebase_ << "-" << i << ".vtu\"/>";
    tags.push_back(std::string(stream.str()));
  }
  return tags;
}

const std::string& PostVtuWriterNode::writer_suffix() const
{
  static std::string name(".vtu");
  return name;
}

const std::string& PostVtuWriterNode::writer_p_suffix() const
{
  static std::string name(".pvtu");
  return name;
}


void PostVtuWriterNode::write_geo()
{
  using namespace FourC;

  std::shared_ptr<Core::FE::Discretization> dis = this->get_field()->discretization();

  // count number of nodes and number for each processor; output is completely independent of
  // the number of processors involved
  int nelements = dis->num_my_row_elements();
  int nnodes = dis->num_my_row_nodes();
  std::vector<int32_t> connectivity;
  connectivity.reserve(nnodes);
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
      {
        connectivity.push_back(nodes[numbering[n]]->lid());
      }
      outNodeId += ele->num_node();
      celloffset.push_back(outNodeId);
    }
  }

  outNodeId = 0;
  for (int n = 0; n < nnodes; n++)
  {
    for (int d = 0; d < 3; ++d)
    {
      coordinates.push_back(dis->l_row_node(n)->x()[d]);
      outNodeId++;
    }
  }

  FOUR_C_ASSERT((int)coordinates.size() == 3 * outNodeId, "internal error");

  // step 1: write node coordinates into file
  currentout_ << "<Piece NumberOfPoints=\"" << nnodes << "\" NumberOfCells=\"" << nelements
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


  if (write_binary_output_)
    LibB64::write_compressed_block(connectivity, currentout_);
  else
  {
    for (std::vector<int32_t>::const_iterator it = connectivity.begin(); it != connectivity.end();
        ++it)
      currentout_ << *it << " ";
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



void PostVtuWriterNode::write_dof_result_step(std::ofstream& file,
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
      (*dofvec).get_values()[i] = data->local_values_as_span()[i];

    ghostedData = Core::LinAlg::create_vector(*colmap, true);
    Core::LinAlg::export_to(*dofvec, *ghostedData);
  }

  int ncomponents = numdf;
  if (numdf > 1 && numdf == field_->problem()->num_dim()) ncomponents = 3;

  // count number of nodes for each processor
  int nnodes = dis->num_my_row_nodes();
  std::vector<double> solution;

  solution.reserve(ncomponents * nnodes);

  std::vector<int> nodedofs;

  for (int i = 0; i < nnodes; i++)
  {
    nodedofs.clear();

    // local storage position of desired dof gid
    dis->dof(dis->l_row_node(i), nodedofs);
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

  }  // loop over all nodes

  FOUR_C_ASSERT((int)solution.size() == ncomponents * nnodes, "internal error");


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


void PostVtuWriterNode::write_nodal_result_step(std::ofstream& file,
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



  // count number of nodes for each processor
  int nnodes = dis->num_my_row_nodes();

  std::vector<double> solution;
  solution.reserve(ncomponents * nnodes);

  for (int i = 0; i < nnodes; i++)
  {
    for (int idf = 0; idf < numdf; ++idf)
    {
      Core::LinAlg::Vector<double> column((*ghostedData)(idf));
      solution.push_back(column.local_values_as_span()[i]);
    }
    for (int d = numdf; d < ncomponents; ++d) solution.push_back(0.);
  }  // loop over all nodes

  FOUR_C_ASSERT((int)solution.size() == ncomponents * nnodes, "internal error");


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



void PostVtuWriterNode::write_element_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from)
{
  if (myrank_ == 0 && timestep_ == 0)
  {
    std::cout << "WARNING: Cannot write element-based quantity in node based vtu filter "
              << std::endl;
    std::cout << "Skipping field " << name.c_str() << std::endl;
  }
}

void PostVtuWriterNode::write_geo_nurbs_ele(const Core::Elements::Element* ele,
    std::vector<uint8_t>& celltypes, int& outNodeId, std::vector<int32_t>& celloffset,
    std::vector<double>& coordinates)
{
  FOUR_C_THROW("VTU node based filter cannot handle NURBS elements");
}

void PostVtuWriterNode::write_geo_beam_ele(const Discret::Elements::Beam3Base* beamele,
    std::vector<uint8_t>& celltypes, int& outNodeId, std::vector<int32_t>& celloffset,
    std::vector<double>& coordinates)
{
  FOUR_C_THROW("VTU node based filter cannot handle beam elements");
}

void PostVtuWriterNode::write_dof_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::Vector<double>> ghostedData, const int from, const bool fillzeros)
{
  FOUR_C_THROW("VTU node based filter cannot handle NURBS elements");
}

void PostVtuWriterNode::write_dof_result_step_beam_ele(const Discret::Elements::Beam3Base* beamele,
    const int& ncomponents, const int& numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::Vector<double>>& ghostedData, const int& from,
    const bool fillzeros)
{
  FOUR_C_THROW("VTU node based filter cannot handle beam elements");
}

void PostVtuWriterNode::write_nodal_result_step_nurbs_ele(const Core::Elements::Element* ele,
    int ncomponents, const int numdf, std::vector<double>& solution,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> ghostedData)
{
  FOUR_C_THROW("VTU node based filter cannot handle NURBS elements");
}

FOUR_C_NAMESPACE_CLOSE
