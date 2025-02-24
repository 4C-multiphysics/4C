// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_COMBINTERSECTION_HPP
#define FOUR_C_CUT_COMBINTERSECTION_HPP

#include "4C_config.hpp"

#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_meshintersection.hpp"

#include <mpi.h>

FOUR_C_NAMESPACE_OPEN


namespace Core::LinAlg
{
  class SerialDenseMatrix;
}


namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Element;

  /*!
  \brief Interface class for the level set cut.
  */
  class CombIntersection : public LevelSetIntersection, public MeshIntersection
  {
   public:
    /// constructor for LevelSetIntersection class
    CombIntersection(MPI_Comm comm);

    void cut(bool screenoutput);

    void find_node_positions();

    void add_element(int eid, const std::vector<int>& nids,
        const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, const double* lsv,
        const bool lsv_only_plus_domain);

    void add_level_set_side(int levelset_sid);

    void add_mesh_cutting_side(int sid, const std::vector<int>& nids,
        const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, int mi);
  };

}  // namespace Cut

FOUR_C_NAMESPACE_CLOSE

#endif
