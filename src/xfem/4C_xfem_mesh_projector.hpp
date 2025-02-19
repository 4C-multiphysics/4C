// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_MESH_PROJECTOR_HPP
#define FOUR_C_XFEM_MESH_PROJECTOR_HPP

#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_MpiComm.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Geo
{
  class SearchTree;
}

namespace XFEM
{
  class MeshProjector
  {
   public:
    //! ctor
    MeshProjector(std::shared_ptr<const Core::FE::Discretization> sourcedis,
        std::shared_ptr<const Core::FE::Discretization> targetdis,
        const Teuchos::ParameterList& params,
        std::shared_ptr<const Core::LinAlg::Vector<double>> sourcedisp = nullptr);

    //! set current displacements of source discretization
    void set_source_position_vector(
        std::shared_ptr<const Core::LinAlg::Vector<double>> sourcedisp = nullptr);

    //! set state vectors - mandatory for interpolation
    void set_source_state_vectors(
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> source_statevecs)
    {
      source_statevecs_ = source_statevecs;
    }

    //! main projection routine (pass a map of the target node ids)
    void project(std::map<int, std::set<int>>&
                     projection_nodeToDof,  //< node-to-dof map of target nodes demanding projection
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>
            target_statevecs,  //< state vectors of target discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>> targetdisp = nullptr);

    //! projection routine for projection for all nodes of the target discretization
    void project_in_full_target_discretization(
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> target_statevecs,
        std::shared_ptr<const Core::LinAlg::Vector<double>> targetdisp = nullptr);

   private:
    /// determine the search radius for the search tree
    template <Core::FE::CellType distype>
    void find_search_radius();

    //! build a search tree for elements of source discretization
    void setup_search_tree();

    //! for every node search for a covering element from the source discretization
    void find_covering_elements_and_interpolate_values(
        std::vector<Core::LinAlg::Matrix<3, 1>>& tar_nodepositions,
        std::vector<Core::LinAlg::Matrix<8, 1>>& interpolated_vecs,
        std::vector<int>& projection_targetnodes, std::vector<int>& have_values);

    //! compute position of target node w.r.t. source element and interpolate when covered by it
    template <Core::FE::CellType distype>
    bool check_position_and_project(const Core::Elements::Element* src_ele,
        const Core::LinAlg::Matrix<3, 1>& node_xyz, Core::LinAlg::Matrix<8, 1>& interpolatedvec);

    //! communicate nodes demanding reconstruction in a Round-Robin pattern
    void communicate_nodes(std::vector<Core::LinAlg::Matrix<3, 1>>& tar_nodepositions,
        std::vector<Core::LinAlg::Matrix<8, 1>>& interpolated_vecs,
        std::vector<int>& projection_targetnodes, std::vector<int>& have_values);

    /// receive a block in the round robin communication pattern
    void receive_block(
        std::vector<char>& rblock, Core::Communication::Exporter& exporter, MPI_Request& request);

    /// send a block in the round robin communication pattern
    void send_block(
        std::vector<char>& sblock, Core::Communication::Exporter& exporter, MPI_Request& request);

    /// pack values in the round robin communication pattern
    void pack_values(std::vector<Core::LinAlg::Matrix<3, 1>>& tar_nodepositions,
        std::vector<Core::LinAlg::Matrix<8, 1>>& interpolated_vecs,
        std::vector<int>& projection_targetnodes, std::vector<int>& have_values,
        std::vector<char>& sblock);

    std::shared_ptr<const Core::FE::Discretization> sourcedis_;
    std::shared_ptr<const Core::FE::Discretization> targetdis_;

    //! search radius factor
    double searchradius_fac_;

    //! 3D search tree for embedded discretization
    std::shared_ptr<Core::Geo::SearchTree> search_tree_;

    //! min. radius needed for the search tree
    double searchradius_;

    //! map of source node to coordinates (including possible displacements)
    std::map<int, Core::LinAlg::Matrix<3, 1>> src_nodepositions_n_;

    //! state vectors from projection source
    std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> source_statevecs_;

    //! map between target node id and parent element id
    std::map<int, int> targetnode_to_parent_;
  };
}  // namespace XFEM


FOUR_C_NAMESPACE_CLOSE

#endif
