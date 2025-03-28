// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_INTERSECTION_SERVICE_TEMPLATES_HPP
#define FOUR_C_FEM_GEOMETRY_INTERSECTION_SERVICE_TEMPLATES_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_geometry_geo_utils.hpp"
#include "4C_fem_geometry_intersection_math.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo
{
  //! template of extended aligned bounding boxes
  template <int ndim, Core::FE::CellType distype, class M>
  static inline Core::LinAlg::Matrix<3, 2> compute_fast_xaabbt(
      const M& xyze, const Core::Geo::EleGeoType eleGeoType)
  {
    Core::LinAlg::Matrix<3, 2> XAABB;

    // first node
    for (int dim = 0; dim < 3; ++dim)
    {
      XAABB(dim, 0) = xyze(dim, 0) - TOL7;
      XAABB(dim, 1) = xyze(dim, 0) + TOL7;
    }
    // remaining nodes
    const int numNodes = Core::FE::num_nodes<distype>;
    for (int i = 1; i < numNodes; ++i)
      for (int dim = 0; dim < ndim; dim++)
      {
        XAABB(dim, 0) = std::min(XAABB(dim, 0), xyze(dim, i) - TOL7);
        XAABB(dim, 1) = std::max(XAABB(dim, 1), xyze(dim, i) + TOL7);
      }

    return XAABB;
  }

  /*!
  \brief Computes a rough overestimating extended
         axis-aligned bounding box for an element (XAABB)
  \param distype        (in)  distype of element
  \param xyze           (in)  nodal position array (3,numnode)
  \param eleGeoType     (in)  element geometric type CARTESIAN LINEAR or HIGHERORDER
  \return extended axis-aligned bounding box  (XAABB) for an element
   */
  template <class M>
  Core::LinAlg::Matrix<3, 2> compute_fast_xaabb(
      Core::FE::CellType distype, const M& xyze, const Core::Geo::EleGeoType eleGeoType)
  {
    switch (distype)
    {
      case Core::FE::CellType::hex8:
        return compute_fast_xaabbt<3, Core::FE::CellType::hex8>(xyze, eleGeoType);
      case Core::FE::CellType::quad4:
        return compute_fast_xaabbt<3, Core::FE::CellType::quad4>(xyze, eleGeoType);
      case Core::FE::CellType::hex20:
        return compute_fast_xaabbt<3, Core::FE::CellType::hex20>(xyze, eleGeoType);
      case Core::FE::CellType::hex27:
        return compute_fast_xaabbt<3, Core::FE::CellType::hex27>(xyze, eleGeoType);
      case Core::FE::CellType::tet4:
        return compute_fast_xaabbt<3, Core::FE::CellType::tet4>(xyze, eleGeoType);
      case Core::FE::CellType::tet10:
        return compute_fast_xaabbt<3, Core::FE::CellType::tet10>(xyze, eleGeoType);
      case Core::FE::CellType::line2:
        return compute_fast_xaabbt<3, Core::FE::CellType::line2>(xyze, eleGeoType);
      case Core::FE::CellType::line3:
        return compute_fast_xaabbt<3, Core::FE::CellType::line3>(xyze, eleGeoType);
      case Core::FE::CellType::tri3:
        return compute_fast_xaabbt<3, Core::FE::CellType::tri3>(xyze, eleGeoType);
      case Core::FE::CellType::tri6:
        return compute_fast_xaabbt<3, Core::FE::CellType::tri6>(xyze, eleGeoType);
      case Core::FE::CellType::quad8:
        return compute_fast_xaabbt<3, Core::FE::CellType::quad8>(xyze, eleGeoType);
      case Core::FE::CellType::quad9:
        return compute_fast_xaabbt<3, Core::FE::CellType::quad9>(xyze, eleGeoType);
      case Core::FE::CellType::pyramid5:
        return compute_fast_xaabbt<3, Core::FE::CellType::pyramid5>(xyze, eleGeoType);
      default:
        std::cout << Core::FE::cell_type_to_string(distype) << std::endl;
        FOUR_C_THROW("add your distype to this switch!");
    }
    return Core::LinAlg::Matrix<3, 2>(Core::LinAlg::Initialization::zero);
  }


}  // namespace Core::Geo


FOUR_C_NAMESPACE_CLOSE

#endif
