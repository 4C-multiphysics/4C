// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_gausspoints.hpp"

#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  namespace
  {
    void fill_pyramid5(Kokkos::DynRankView<double, Kokkos::HostSpace>& cub_points,
        Kokkos::DynRankView<double, Kokkos::HostSpace>& cub_weights)
    {
      Kokkos::resize(cub_points, 8, 3);
      Kokkos::resize(cub_weights, 8);
      cub_points(0, 0) = -0.26318405556971;
      cub_points(1, 0) = -0.50661630334979;
      cub_points(2, 0) = -0.26318405556971;
      cub_points(3, 0) = -0.50661630334979;
      cub_points(4, 0) = 0.26318405556971;
      cub_points(5, 0) = 0.50661630334979;
      cub_points(6, 0) = 0.26318405556971;
      cub_points(7, 0) = 0.50661630334979;
      cub_points(0, 1) = -0.26318405556971;
      cub_points(1, 1) = -0.50661630334979;
      cub_points(2, 1) = 0.26318405556971;
      cub_points(3, 1) = 0.50661630334979;
      cub_points(4, 1) = -0.26318405556971;
      cub_points(5, 1) = -0.50661630334979;
      cub_points(6, 1) = 0.26318405556971;
      cub_points(7, 1) = 0.50661630334979;
      cub_points(0, 2) = 0.54415184401122;
      cub_points(1, 2) = 0.12251482265544;
      cub_points(2, 2) = 0.54415184401122;
      cub_points(3, 2) = 0.12251482265544;
      cub_points(4, 2) = 0.54415184401122;
      cub_points(5, 2) = 0.12251482265544;
      cub_points(6, 2) = 0.54415184401122;
      cub_points(7, 2) = 0.12251482265544;

      cub_weights(0) = 0.10078588207983;
      cub_weights(1) = 0.23254745125351;
      cub_weights(2) = 0.10078588207983;
      cub_weights(3) = 0.23254745125351;
      cub_weights(4) = 0.10078588207983;
      cub_weights(5) = 0.23254745125351;
      cub_weights(6) = 0.10078588207983;
      cub_weights(7) = 0.23254745125351;
    }

    /// wrapper to intrepid gauss point implementation
    class IntrepidGaussPoints : public GaussPoints
    {
     public:
      explicit IntrepidGaussPoints(const shards::CellTopology& cell_topology, int cubDegree)
          : cub_points_("cubature_points", 1, 1),
            cub_weights_("cubature_weights", 1),
            cell_topology_(cell_topology)
      {
        // Special case that Intrepid does not support
        if (cell_topology ==
            shards::CellTopology(shards::getCellTopologyData<shards::Pyramid<5>>()))
        {
          fill_pyramid5(cub_points_, cub_weights_);
          return;
        }

        // retrieve spatial dimension
        const int spaceDim = cell_topology_.getDimension();

        Teuchos::RCP<Intrepid2::Cubature<Kokkos::HostSpace, double, double>> myCub =
            Intrepid2::DefaultCubatureFactory::create<Kokkos::HostSpace>(cell_topology_, cubDegree);

        // retrieve number of cubature points
        int numCubPoints = myCub->getNumPoints();

        Kokkos::resize(cub_points_, numCubPoints, spaceDim);
        Kokkos::resize(cub_weights_, numCubPoints);

        // retrieve cubature points and weights
        myCub->getCubature(cub_points_, cub_weights_);
      }

      int num_points() const override { return cub_points_.extent(0); }

      int num_dimension() const override { return cub_points_.extent(1); }

      const double* point(int point) const override { return &cub_points_(point, 0); }

      double weight(int point) const override { return cub_weights_(point); }

      void print() const override
      {
        std::cout << cell_topology_.getName() << " gauss points:\n";
        for (int i = 0; i < num_points(); ++i)
        {
          std::cout << "coordinate: ";
          for (int j = 0; j < num_dimension(); ++j) std::cout << cub_points_(i, j) << " ";
          std::cout << "weight: " << cub_weights_(i) << "\n";
        }
      }

     private:
      Kokkos::DynRankView<double, Kokkos::HostSpace> cub_points_;
      Kokkos::DynRankView<double, Kokkos::HostSpace> cub_weights_;

      shards::CellTopology cell_topology_;
    };

    template <typename Topology>
    std::shared_ptr<IntrepidGaussPoints> make_intrepid_gauss_points(int cub_degree)
    {
      const shards::CellTopology cell_topology = shards::getCellTopologyData<Topology>();
      return std::make_shared<IntrepidGaussPoints>(cell_topology, cub_degree);
    }

  }  // namespace
}  // namespace Core::FE

Core::FE::GaussIntegration::GaussIntegration(Core::FE::CellType distype)
    : gp_(create_gauss_points_default(distype))
{
}

Core::FE::GaussIntegration::GaussIntegration(Core::FE::CellType distype, int degree)
    : gp_(create_gauss_points(distype, degree))
{
}

std::shared_ptr<Core::FE::GaussPoints> Core::FE::create_gauss_points(
    Core::FE::CellType distype, int degree)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
      return make_intrepid_gauss_points<shards::Quadrilateral<4>>(degree);
    case Core::FE::CellType::quad8:
      return make_intrepid_gauss_points<shards::Quadrilateral<8>>(degree);
    case Core::FE::CellType::quad9:
      return make_intrepid_gauss_points<shards::Quadrilateral<9>>(degree);
    case Core::FE::CellType::tri3:
      return make_intrepid_gauss_points<shards::Triangle<3>>(degree);
    case Core::FE::CellType::tri6:
      return make_intrepid_gauss_points<shards::Triangle<6>>(degree);
    case Core::FE::CellType::hex8:
      return make_intrepid_gauss_points<shards::Hexahedron<8>>(degree);
    case Core::FE::CellType::hex20:
      return make_intrepid_gauss_points<shards::Hexahedron<20>>(degree);
    case Core::FE::CellType::hex27:
      return make_intrepid_gauss_points<shards::Hexahedron<27>>(degree);
    case Core::FE::CellType::tet4:
      return make_intrepid_gauss_points<shards::Tetrahedron<4>>(degree);
    case Core::FE::CellType::tet10:
      return make_intrepid_gauss_points<shards::Tetrahedron<10>>(degree);
    case Core::FE::CellType::wedge6:
      return make_intrepid_gauss_points<shards::Wedge<6>>(degree);
    case Core::FE::CellType::wedge15:
      return make_intrepid_gauss_points<shards::Wedge<15>>(degree);
    case Core::FE::CellType::pyramid5:
      return make_intrepid_gauss_points<shards::Pyramid<5>>(degree);
    case Core::FE::CellType::line2:
      return make_intrepid_gauss_points<shards::Line<2>>(degree);
    case Core::FE::CellType::line3:
      return make_intrepid_gauss_points<shards::Line<3>>(degree);
    default:
      FOUR_C_THROW("unsupported element shape");
  }
}


std::shared_ptr<Core::FE::GaussPoints> Core::FE::create_gauss_points_default(Core::FE::CellType distype)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
      return create_gauss_points(Core::FE::CellType::quad4, 3);
    case Core::FE::CellType::quad8:
      return create_gauss_points(Core::FE::CellType::quad8, 4);
    case Core::FE::CellType::quad9:
      return create_gauss_points(Core::FE::CellType::quad9, 4);
    case Core::FE::CellType::tri3:
      return create_gauss_points(Core::FE::CellType::tri3, 3);
    case Core::FE::CellType::tri6:
      return create_gauss_points(Core::FE::CellType::tri6, 4);
    case Core::FE::CellType::hex8:
      return create_gauss_points(Core::FE::CellType::hex8, 3);
    case Core::FE::CellType::hex20:
      return create_gauss_points(Core::FE::CellType::hex20, 4);
    case Core::FE::CellType::hex27:
      return create_gauss_points(Core::FE::CellType::hex27, 4);
    case Core::FE::CellType::tet4:
      return create_gauss_points(Core::FE::CellType::tet4, 3);
    case Core::FE::CellType::tet10:
      return create_gauss_points(Core::FE::CellType::tet10, 4);
    case Core::FE::CellType::wedge6:
      return create_gauss_points(Core::FE::CellType::wedge6, 3);
    case Core::FE::CellType::wedge15:
      return create_gauss_points(Core::FE::CellType::wedge15, 4);
    case Core::FE::CellType::pyramid5:
      return create_gauss_points(Core::FE::CellType::pyramid5, 3);
    case Core::FE::CellType::line2:
      return create_gauss_points(Core::FE::CellType::line2, 3);
    case Core::FE::CellType::line3:
      return create_gauss_points(Core::FE::CellType::line3, 4);
    case Core::FE::CellType::nurbs2:
      return create_gauss_points(Core::FE::CellType::line2, 3);
    case Core::FE::CellType::nurbs3:
      return create_gauss_points(Core::FE::CellType::line3, 4);
    case Core::FE::CellType::nurbs4:
      return create_gauss_points(Core::FE::CellType::quad4, 3);
    case Core::FE::CellType::nurbs8:
      return create_gauss_points(Core::FE::CellType::hex8, 3);
    case Core::FE::CellType::nurbs9:
      return create_gauss_points(Core::FE::CellType::quad9, 4);
    case Core::FE::CellType::nurbs27:
      return create_gauss_points(Core::FE::CellType::hex27, 4);
    default:
      FOUR_C_THROW("unsupported element shape");
  }
}

FOUR_C_NAMESPACE_CLOSE
