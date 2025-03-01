// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POST_VTK_VTI_WRITER_HPP
#define FOUR_C_POST_VTK_VTI_WRITER_HPP


#include "4C_config.hpp"

#include "4C_post_vtk_writer.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

// forward declarations

FOUR_C_NAMESPACE_OPEN
class PostField;
class PostResult;

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

/*
 \brief Base class for VTU output generation

*/
class PostVtiWriter : public PostVtkWriter
{
 public:
  //! constructor. Initializes the writer to a certain field.
  PostVtiWriter(PostField* field, const std::string& name);

 protected:
  //! Return the opening xml tag for this writer type
  const std::string& writer_opening_tag() const override;

  //! Return the parallel opening xml tag for this writer type
  const std::string& writer_p_opening_tag() const override;

  //! Return a vector of parallel piece tags for each file
  const std::vector<std::string>& writer_p_piece_tags() const override;

  //! Give every writer a chance to do preparations before writing
  void writer_prep_timestep() override;

  //! Return the parallel file suffix including the dot for this file type
  const std::string& writer_p_suffix() const override;

  //! Return the string of this writer type
  const std::string& writer_string() const override;

  //! Return the file suffix including the dot for this file type
  const std::string& writer_suffix() const override;

  //! Write a single result step
  void write_dof_result_step(std::ofstream& file,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& data,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf, const int from,
      const bool fillzeros) override;

  //! Write a single result step
  void write_nodal_result_step(std::ofstream& file,
      const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf) override;

  //! Write a single result step
  void write_element_result_step(std::ofstream& file,
      const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf,
      const int from) override;

  //! write the geometry of one time step
  void write_geo() override;

  //! origin of the ImageData-grid
  double origin_[3];

  //! spacing of the ImageData-grid
  double spacing_[3];

  //! global extent of the ImageData-grid (x_min x_max y_min y_max z_min z_max)
  int globalextent_[6];

  //! local extent of the ImageData-grid (x_min x_max y_min y_max z_min z_max)
  int localextent_[6];

  //! Mapping between nodeids and their position on an ImageData-grid in a (z*Ny+y)*Nx+x form
  std::map<int, int> idmapping_;

  //! Mapping between elementids and their position on an ImageData-grid in a (z*Ny+y)*Nx+x form
  std::map<int, int> eidmapping_;
};

FOUR_C_NAMESPACE_CLOSE

#endif
