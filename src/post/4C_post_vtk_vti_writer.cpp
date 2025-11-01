// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_vtk_vti_writer.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_post_common.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

#ifdef TOL_N
#undef TOL_N
#endif
#define TOL_N 1.0e-14

template <class T>
struct LessTol
{
  bool operator()(const T& x, const T& y) const { return x < y - TOL_N; }
};

PostVtiWriter::PostVtiWriter(PostField* field, const std::string& filename)
    : PostVtkWriter(field, filename)
{
  for (size_t i = 0; i < sizeof(origin_) / sizeof(origin_[0]); ++i) origin_[i] = 0.0;
  for (size_t i = 0; i < sizeof(spacing_) / sizeof(spacing_[0]); ++i) spacing_[i] = 0.0;
  for (size_t i = 0; i < sizeof(globalextent_) / sizeof(globalextent_[0]); ++i)
    globalextent_[i] = 0;
  for (size_t i = 0; i < sizeof(localextent_) / sizeof(localextent_[0]); ++i) localextent_[i] = 0;
}


const std::string& PostVtiWriter::writer_string() const
{
  static std::string name("ImageData");
  return name;
}

const std::string& PostVtiWriter::writer_opening_tag() const
{
  static std::string tag;
  std::stringstream stream;
  stream << "<ImageData WholeExtent=\"" << localextent_[0] << " " << localextent_[1] << " "
         << localextent_[2] << " " << localextent_[3] << " " << localextent_[4] << " "
         << localextent_[5] << "\" Origin=\"" << origin_[0] << " " << origin_[1] << " "
         << origin_[2] << "\" Spacing=\"" << spacing_[0] << " " << spacing_[1] << " " << spacing_[2]
         << "\">";
  tag.assign(stream.str());

  return tag;
}

const std::string& PostVtiWriter::writer_p_opening_tag() const
{
  static std::string tag;
  std::stringstream stream;
  stream << "<PImageData WholeExtent=\"" << globalextent_[0] << " " << globalextent_[1] << " "
         << globalextent_[2] << " " << globalextent_[3] << " " << globalextent_[4] << " "
         << globalextent_[5] << "\" GhostLevel=\"0\" Origin=\"" << origin_[0] << " " << origin_[1]
         << " " << origin_[2] << "\" Spacing=\"" << spacing_[0] << " " << spacing_[1] << " "
         << spacing_[2] << "\">";
  tag.assign(stream.str());

  return tag;
}

const std::vector<std::string>& PostVtiWriter::writer_p_piece_tags() const
{
  static std::vector<std::string> tags;
  tags.clear();

  std::vector<int> allextents(numproc_ * 6);
  Core::Communication::gather_all(
      (int*)localextent_, allextents.data(), 6, field_->problem()->get_comm());

  if (myrank_ == 0)
  {
    for (size_t i = 0; i < numproc_; ++i)
    {
      std::stringstream stream;
      stream << "<Piece Extent=\"" << allextents[6 * i] << " " << allextents[6 * i + 1] << " "
             << allextents[6 * i + 2] << " " << allextents[6 * i + 3] << " "
             << allextents[6 * i + 4] << " " << allextents[6 * i + 5] << "\" Source=\""
             << filenamebase_ << "-" << std::setfill('0') << std::setw(npdigits_) << i
             << ".vti\"/>";
      tags.push_back(stream.str());
    }
  }

  return tags;
}

const std::string& PostVtiWriter::writer_suffix() const
{
  static std::string name(".vti");
  return name;
}

const std::string& PostVtiWriter::writer_p_suffix() const
{
  static std::string name(".pvti");
  return name;
}



void PostVtiWriter::write_geo()
{
  // There is no such thing as geometry for ImageData, however we need to prepare some things

  // start the piece
  currentout_ << "<Piece Extent=\"" << localextent_[0] << " " << localextent_[1] << " "
              << localextent_[2] << " " << localextent_[3] << " " << localextent_[4] << " "
              << localextent_[5] << "\">\n";
  return;
}



void PostVtiWriter::write_dof_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from,
    const bool fillzeros)
{
  using namespace FourC;

  if (myrank_ == 0 && timestep_ == 0) std::cout << "writing dof-based field " << name << std::endl;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();

  // Here is the only thing we need to do for parallel computations: We need read access to all dofs
  // on the row elements, so need to get the DofColMap to have this access
  const Core::LinAlg::Map* colmap = dis->dof_col_map(0);
  const Core::LinAlg::Map& vecmap = data->get_map();

  // TODO: wic, once the vtu pressure writer is fixed, apply the same solution here.
  const int offset = (fillzeros) ? 0 : vecmap.min_all_gid() - dis->dof_row_map(0)->min_all_gid();

  std::shared_ptr<Core::LinAlg::Vector<double>> ghostedData;
  if (colmap->same_as(vecmap))
    ghostedData = data;
  else
  {
    ghostedData = Core::LinAlg::create_vector(*colmap, false);
    Core::LinAlg::export_to(*data, *ghostedData);
  }

  const int ncomponents = (numdf > 1 && numdf == field_->problem()->num_dim()) ? 3 : numdf;

  std::vector<double> solution;
  if (fillzeros)
    solution.resize(ncomponents * ((localextent_[1] - localextent_[0] + 1) *
                                      (localextent_[3] - localextent_[2] + 1) *
                                      (localextent_[5] - localextent_[4] + 1)),
        0.0);
  else
    solution.resize(ncomponents * ((localextent_[1] - localextent_[0] + 1) *
                                      (localextent_[3] - localextent_[2] + 1) *
                                      (localextent_[5] - localextent_[4] + 1)));

  std::vector<int> nodedofs;
  for (int e = 0; e < dis->num_my_col_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_col_element(e);

    for (int n = 0; n < ele->num_node(); ++n)
    {
      nodedofs.clear();

      // node gid for insertion in solution
      const int ngid = ele->nodes()[n]->id();
      const int inpos = ncomponents * idmapping_.find(ngid)->second;

      // local storage position of desired dof gid
      dis->dof(ele->nodes()[n], nodedofs);

      for (int d = 0; d < numdf; ++d)
      {
        const int lid = ghostedData->get_map().lid(nodedofs[d + from] + offset);
        if (lid > -1)
        {
          solution[inpos + d] = ghostedData->local_values_as_span()[lid];
        }
        else
        {
          if (!fillzeros) FOUR_C_THROW("received illegal dof local id: {}", lid);
        }
      }
      for (int d = numdf; d < ncomponents; ++d) solution[inpos + d] = 0.0;
    }
  }

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

  return;
}



void PostVtiWriter::write_nodal_result_step(std::ofstream& file,
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

  const int ncomponents = (numdf > 1 && numdf == field_->problem()->num_dim()) ? 3 : numdf;

  std::vector<double> solution(ncomponents * ((localextent_[1] - localextent_[0] + 1) *
                                                 (localextent_[3] - localextent_[2] + 1) *
                                                 (localextent_[5] - localextent_[4] + 1)));

  for (int e = 0; e < dis->num_my_col_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_col_element(e);

    for (int n = 0; n < ele->num_node(); ++n)
    {
      const int gid = ele->nodes()[n]->id();
      const int inpos = ncomponents * idmapping_.find(gid)->second;

      for (int idf = 0; idf < numdf; ++idf)
      {
        Core::LinAlg::Vector<double> column((*ghostedData)(idf));

        const int lid = ghostedData->get_map().lid(gid);

        if (lid > -1)
        {
          solution[inpos + idf] = column.local_values_as_span()[lid];
        }
        else
        {
          FOUR_C_THROW("received illegal node local id: {}", lid);
        }
      }
      for (int d = numdf; d < ncomponents; ++d) solution[inpos + d] = 0.;
    }
  }

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

  return;
}



void PostVtiWriter::write_element_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from)
{
  using namespace FourC;

  if (myrank_ == 0 && timestep_ == 0)
    std::cout << "writing element-based field " << name << std::endl;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();

  const int ncomponents = (numdf > 1 && numdf == field_->problem()->num_dim()) ? 3 : numdf;

  std::vector<double> solution(
      ncomponents * ((localextent_[1] - localextent_[0]) * (localextent_[3] - localextent_[2]) *
                        (localextent_[5] - localextent_[4])),
      0.0);

  const int numcol = data->num_vectors();
  if (numdf + from > numcol)
    FOUR_C_THROW("violated column range of Core::LinAlg::MultiVector<double>: {}", numcol);

  std::shared_ptr<Core::LinAlg::MultiVector<double>> importedData;
  if (dis->element_col_map()->same_as(data->get_map()))
    importedData = data;
  else
  {
    importedData = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *dis->element_col_map(), data->num_vectors(), false);
    Core::LinAlg::export_to(*data, *importedData);
  }

  for (int e = 0; e < dis->num_my_col_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_col_element(e);
    const int egid = ele->id();
    const int inpos = ncomponents * (eidmapping_.find(egid)->second);
    for (int d = 0; d < numdf; ++d)
    {
      Core::LinAlg::Vector<double> column((*importedData)(d + from));
      solution[inpos + d] = column.local_values_as_span()[e];
    }
  }

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

  return;
}



void PostVtiWriter::writer_prep_timestep()
{
  using namespace FourC;

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();
  // collect all possible values of the x-, y- and z-coordinate
  using set_tol = std::set<double, LessTol<double>>;
  set_tol collected_coords[3];
  for (int n = 0; n < dis->num_my_col_nodes(); ++n)
  {
    const auto& coord = dis->l_col_node(n)->x();
    for (int i = 0; i < 3; ++i) collected_coords[i].insert(coord[i]);
  }

  // determine local and global domain size
  std::array<double, 3> lorigin{}, gorigin{};
  std::array<double, 3> lextent{}, gextent{};
  for (int i = 0; i < 3; ++i)
  {
    lorigin[i] = *collected_coords[i].begin();
    lextent[i] = *collected_coords[i].rbegin();
  }
  gorigin = Core::Communication::min_all(lorigin, field_->discretization()->get_comm());
  gextent = Core::Communication::max_all(lextent, field_->discretization()->get_comm());

  // determine spacing and check whether it is consistent for ImageData
  for (int i = 0; i < 3; ++i)
    spacing_[i] = (*(collected_coords[i].rbegin()) - *(collected_coords[i].begin())) /
                  (collected_coords[i].size() - 1);
  for (int i = 0; i < 3; ++i)
  {
    int k = 0;
    for (set_tol::const_iterator it = collected_coords[i].begin(); it != collected_coords[i].end();
        ++it, ++k)
      if (*it - lorigin[i] - k * spacing_[i] > TOL_N || *it - lorigin[i] - k * spacing_[i] < -TOL_N)
        FOUR_C_THROW(
            "The mesh is not a uniform rectangular grid: grid coordinate[{}]: {}, "
            "node coordinate[{}]: {}, difference: {}",
            i, lorigin[i] + k * spacing_[i], i, *it, *it - lorigin[i] - k * spacing_[i]);
  }

  // determine extents
  for (int i = 0; i < 3; ++i)
  {
    localextent_[0 + 2 * i] = round((lorigin[i] - gorigin[i]) / spacing_[i]);
    localextent_[1 + 2 * i] = round((lextent[i] - gorigin[i]) / spacing_[i]);
    globalextent_[0 + 2 * i] = 0;
    globalextent_[1 + 2 * i] = round((gextent[i] - gorigin[i]) / spacing_[i]);
  }

  // finally create the mapping
  int nx = localextent_[1] - localextent_[0] + 1;
  int ny = localextent_[3] - localextent_[2] + 1;
  idmapping_.clear();
  for (int n = 0; n < dis->num_my_col_nodes(); ++n)
  {
    const auto& coord = dis->l_col_node(n)->x();
    int i = round((coord[0] - lorigin[0]) / spacing_[0]);
    int j = round((coord[1] - lorigin[1]) / spacing_[1]);
    int k = round((coord[2] - lorigin[2]) / spacing_[2]);
    idmapping_[dis->node_col_map()->gid(n)] = (k * ny + j) * nx + i;
  }

  // create element id mapping by the coordinates of the lowest node coordinate
  nx = localextent_[1] - localextent_[0];
  ny = localextent_[3] - localextent_[2];
  eidmapping_.clear();
  for (int e = 0; e < dis->num_my_col_elements(); ++e)
  {
    const Core::Elements::Element* ele = dis->l_col_element(e);
    double mincoord[] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()};
    for (int n = 0; n < ele->num_node(); ++n)
    {
      const auto& coord = ele->nodes()[n]->x();
      mincoord[0] = std::min(mincoord[0], coord[0]);
      mincoord[1] = std::min(mincoord[1], coord[1]);
      mincoord[2] = std::min(mincoord[2], coord[2]);
    }
    int i = round((mincoord[0] - lorigin[0]) / spacing_[0]);
    int j = round((mincoord[1] - lorigin[1]) / spacing_[1]);
    int k = round((mincoord[2] - lorigin[2]) / spacing_[2]);
    eidmapping_[dis->element_col_map()->gid(e)] = (k * ny + j) * nx + i;
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE
