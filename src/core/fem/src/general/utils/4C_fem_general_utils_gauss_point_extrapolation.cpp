// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_gauss_point_extrapolation.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <Core::FE::CellType distype>
    requires(Core::FE::is_tet<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 4) return Core::FE::CellType::point1;
    if (numgp < 10) return Core::FE::CellType::tet4;
    return Core::FE::CellType::tet10;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_hex<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 8) return Core::FE::CellType::point1;
    if (numgp < 20) return Core::FE::CellType::hex8;
    if (numgp < 27) return Core::FE::CellType::hex20;
    return Core::FE::CellType::hex27;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_nurbs<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 8) return Core::FE::CellType::point1;
    return Core::FE::CellType::nurbs27;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_quad<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 4) return Core::FE::CellType::point1;
    if (numgp < 8) return Core::FE::CellType::quad4;
    if (numgp < 9) return Core::FE::CellType::quad8;
    return Core::FE::CellType::quad9;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_tri<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 3) return Core::FE::CellType::point1;
    if (numgp < 6) return Core::FE::CellType::tri3;
    return Core::FE::CellType::tri6;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_wedge<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 6) return Core::FE::CellType::point1;
    if (numgp < 15) return Core::FE::CellType::wedge6;
    return Core::FE::CellType::wedge15;
  }

  template <Core::FE::CellType distype>
    requires(Core::FE::is_pyramid<distype>)
  inline Core::FE::CellType get_gauss_point_extrapolation_base_distype(unsigned numgp)
  {
    if (numgp < 5) return Core::FE::CellType::point1;
    return Core::FE::CellType::pyramid5;
  }

  template <Core::FE::CellType distype, class GaussIntegration>
  Core::LinAlg::SerialDenseMatrix evaluate_base_shape_functions_at_gauss_points(
      const Core::FE::CellType base_distype, const GaussIntegration& intpoints)
  {
    constexpr int nsd = Core::FE::dim<distype>;
    int base_numnod = Core::FE::get_number_of_element_nodes(base_distype);
    Core::LinAlg::SerialDenseMatrix mat(intpoints.num_points(), base_numnod);


    for (int gp = 0; gp < intpoints.num_points(); ++gp)
    {
      Core::LinAlg::Matrix<nsd, 1> xi(intpoints.point(gp), true);

      Core::LinAlg::SerialDenseVector shape_functions(base_numnod);
      Core::FE::shape_function_dim<Core::LinAlg::Matrix<nsd, 1>, Core::LinAlg::SerialDenseVector,
          nsd>(xi, shape_functions, base_distype);

      for (int inode = 0; inode < base_numnod; ++inode)
      {
        mat(gp, inode) = shape_functions(inode);
      }
    }
    return mat;
  }

  template <Core::FE::CellType distype, class GaussIntegration>
  Core::LinAlg::SerialDenseMatrix evaluate_nurbs_base_shape_functions_at_gauss_points(
      const Core::FE::Discretization& dis, const Core::Elements::Element& ele,
      const Core::FE::CellType base_distype, const GaussIntegration& intpoints)
  {
    constexpr int nsd = Core::FE::dim<distype>;
    int base_numnod = Core::FE::get_number_of_element_nodes(base_distype);
    Core::LinAlg::SerialDenseMatrix mat(intpoints.num_points(), base_numnod);

    // Obtain weights and knot vector of element
    Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, 1> weights(
        Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::SerialDenseVector> myknots(true);

    const bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(dis, &ele, myknots, weights);
    if (zero_size) FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");


    for (int gp = 0; gp < intpoints.num_points(); ++gp)
    {
      Core::LinAlg::Matrix<nsd, 1> xi(intpoints.point(gp), true);

      Core::LinAlg::SerialDenseVector shape_functions(base_numnod);
      Core::FE::Nurbs::nurbs_shape_function_dim(
          shape_functions, xi, myknots, weights, base_distype);

      for (int inode = 0; inode < base_numnod; ++inode)
      {
        mat(gp, inode) = shape_functions(inode);
      }
    }
    return mat;
  }

  Core::LinAlg::SerialDenseMatrix evaluate_projection_gauss_points_to_base_distype(
      const Core::LinAlg::SerialDenseMatrix& shapefcns_at_gps)
  {
    Core::LinAlg::SerialDenseMatrix shapefunctions_at_gps_copy(shapefcns_at_gps);
    if (shapefcns_at_gps.numRows() == shapefcns_at_gps.numCols())
    {
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> matrixInverter;
      matrixInverter.setMatrix(Teuchos::rcpFromRef(shapefunctions_at_gps_copy));
      int error_code = matrixInverter.invert();

      if (error_code != 0)
      {
        FOUR_C_THROW(
            "Failed to invert the matrix of the shapefunctions evaluated at the Gauss points. It "
            "looks like this element does not support the default way to extrapolate quantities "
            "from Gauss points to nodes. Error code: {}",
            error_code);
      }

      FOUR_C_ASSERT(
          shapefunctions_at_gps_copy.values() == matrixInverter.getFactoredMatrix()->values(),
          "Inverse of the matrix was not computed in place, but we expect that. Unfortunately, the "
          "Trilinos documentation is ambiguous here.");

      return shapefunctions_at_gps_copy;
    }

    // solve least square algorithm
    Core::LinAlg::SerialDenseMatrix matTmat(shapefcns_at_gps.numCols(), shapefcns_at_gps.numCols());
    Core::LinAlg::multiply_tn(matTmat, shapefcns_at_gps, shapefcns_at_gps);

    {
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> matrixInverter;
      matrixInverter.setMatrix(Teuchos::rcpFromRef(matTmat));
      int error_code = matrixInverter.invert();

      if (error_code != 0)
      {
        FOUR_C_THROW(
            "Failed to invert the matrix of the shapefunctions evaluated at the Gauss points. It "
            "looks like this element does not support the default way to extrapolate quantities "
            "from Gauss points to nodes. Error code {}",
            error_code);
      }

      FOUR_C_ASSERT(matTmat.values() == matrixInverter.getFactoredMatrix()->values(),
          "Inverse of the matrix was not computed in place, but we expect that. Unfortunately, the "
          "Trilinos documentation is ambiguous here.");
    }

    Core::LinAlg::SerialDenseMatrix matrix_gp_to_base(
        shapefcns_at_gps.numCols(), shapefcns_at_gps.numRows());
    Core::LinAlg::multiply_nt(matrix_gp_to_base, matTmat, shapefunctions_at_gps_copy);

    return matrix_gp_to_base;
  }

  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix evaluate_projection_gauss_points_to_distype(
      const Core::LinAlg::SerialDenseMatrix& matrix_gp_to_base, Core::FE::CellType base_distype)
  {
    if (base_distype == distype)
    {
      return matrix_gp_to_base;
    }
    constexpr int nsd = Core::FE::dim<distype>;
    int base_numnod = Core::FE::get_number_of_element_nodes(base_distype);

    Core::LinAlg::SerialDenseMatrix matrix_base_to_dis(Core::FE::num_nodes<distype>, base_numnod);

    for (int dis_inode = 0; dis_inode < Core::FE::num_nodes<distype>; ++dis_inode)
    {
      Core::LinAlg::SerialDenseMatrix reference_nodes =
          Core::FE::get_ele_node_numbering_nodes_paramspace(distype);

      Core::LinAlg::SerialDenseVector shape_functions(base_numnod);
      switch (nsd)
      {
        case 3:
        {
          Core::FE::shape_function_3d(shape_functions, reference_nodes(0, dis_inode),
              reference_nodes(1, dis_inode), reference_nodes(2, dis_inode), base_distype);
        }
        break;
        case 2:
        {
          Core::FE::shape_function_2d(shape_functions, reference_nodes(0, dis_inode),
              reference_nodes(1, dis_inode), base_distype);
        }
        break;
        case 1:
        {
          Core::FE::shape_function_1d(shape_functions, reference_nodes(0, dis_inode), base_distype);
        }
        break;
        default:
          FOUR_C_THROW("This function is not implemented for space dimension {}.", nsd);
      }
      for (int basedis_inode = 0; basedis_inode < base_numnod; ++basedis_inode)
      {
        matrix_base_to_dis(dis_inode, basedis_inode) = shape_functions(basedis_inode);
      }
    }

    Core::LinAlg::SerialDenseMatrix matrix_gp_to_nodes(
        Core::FE::num_nodes<distype>, matrix_gp_to_base.numCols());

    // extend matrix from base_distype to distype
    Core::LinAlg::multiply(matrix_gp_to_nodes, matrix_base_to_dis, matrix_gp_to_base);

    return matrix_gp_to_nodes;
  }

  template <class T>
  void assemble_extrapolated_nodal_values(Core::LinAlg::MultiVector<double>& global_data,
      const T& nodal_data, const Core::Elements::Element& ele, bool nodal_average)
  {
    for (decltype(nodal_data.numRows()) i = 0; i < nodal_data.numRows(); ++i)
    {
      const int lid = global_data.Map().LID(ele.node_ids()[i]);
      if (lid >= 0)  // rownode
      {
        const double invmyadjele = (nodal_average) ? 1.0 / ele.nodes()[i]->num_element() : 1.0;
        for (decltype(nodal_data.numCols()) j = 0; j < nodal_data.numCols(); ++j)
        {
          (global_data(j))[lid] += nodal_data(i, j) * invmyadjele;
        }
      }
    }
  }
}  // namespace


template <Core::FE::CellType distype, class GaussIntegration>
Core::LinAlg::SerialDenseMatrix Core::FE::evaluate_gauss_points_to_nodes_extrapolation_matrix(
    const GaussIntegration& intpoints)
{
  static std::unordered_map<unsigned, Core::LinAlg::SerialDenseMatrix> extrapolation_matrix_cache{};

  if (extrapolation_matrix_cache.find(intpoints.num_points()) == extrapolation_matrix_cache.end())
  {
    Core::FE::CellType base_distype =
        get_gauss_point_extrapolation_base_distype<distype>(intpoints.num_points());

    FOUR_C_ASSERT(Core::FE::get_number_of_element_nodes(base_distype) <= intpoints.num_points(),
        "The base discretization has more nodes than Gauss points. The extrapolation is not "
        "unique! "
        "This should not happen. The evaluation of the base extrapolation type for the number of "
        "gauss points is not correct.");

    Core::LinAlg::SerialDenseMatrix shapefcns_at_gps =
        evaluate_base_shape_functions_at_gauss_points<distype>(base_distype, intpoints);

    Core::LinAlg::SerialDenseMatrix matrix_gp_to_base =
        evaluate_projection_gauss_points_to_base_distype(shapefcns_at_gps);

    extrapolation_matrix_cache[intpoints.num_points()] =
        evaluate_projection_gauss_points_to_distype<distype>(matrix_gp_to_base, base_distype);
  }

  return extrapolation_matrix_cache[intpoints.num_points()];
}

// template specialization for pyramid 5 elements
// The default procedure of extrapolation by using the shape functions results in different results
// than with our previous method. The 8 Gauss points create a HEX-element inside the pyramid. The
// extrapolation matrix holds the shapefunction-values of the HEX-element, evaluated at the
// pyramid-nodes.
template <>
Core::LinAlg::SerialDenseMatrix
Core::FE::evaluate_gauss_points_to_nodes_extrapolation_matrix<Core::FE::CellType::pyramid5>(
    const Core::FE::IntegrationPoints3D& intpoints)
{
  if (intpoints.num_points() != 8)
  {
    FOUR_C_THROW(
        "Gauss point extrapolation is not yet implemented for Pyramid5 elements with {} Gauss "
        "points. Currently, only 8 are supported",
        intpoints.num_points());
  }

  static Core::LinAlg::SerialDenseMatrix extrapolation_matrix = std::invoke(
      []()
      {
        Core::LinAlg::SerialDenseMatrix expol(5, 8);
        expol(0, 0) = 2.408235313815748;
        expol(0, 1) = -0.6452847075210328;
        expol(0, 2) = 0.1729035162684118;
        expol(0, 3) = -0.6452847075210328;
        expol(0, 4) = -0.542209910031327;
        expol(0, 5) = 0.1452847075210439;
        expol(0, 6) = -0.03892892005285509;
        expol(0, 7) = 0.1452847075210439;
        expol(1, 0) = -0.6452847075210328;
        expol(1, 1) = 2.408235313815748;
        expol(1, 2) = -0.6452847075210328;
        expol(1, 3) = 0.1729035162684118;
        expol(1, 4) = 0.1452847075210439;
        expol(1, 5) = -0.542209910031327;
        expol(1, 6) = 0.1452847075210439;
        expol(1, 7) = -0.03892892005285509;
        expol(2, 0) = 0.1729035162684118;
        expol(2, 1) = -0.6452847075210328;
        expol(2, 2) = 2.408235313815748;
        expol(2, 3) = -0.6452847075210328;
        expol(2, 4) = -0.03892892005285509;
        expol(2, 5) = 0.1452847075210439;
        expol(2, 6) = -0.542209910031327;
        expol(2, 7) = 0.1452847075210439;
        expol(3, 0) = -0.6452847075210328;
        expol(3, 1) = 0.1729035162684118;
        expol(3, 2) = -0.6452847075210328;
        expol(3, 3) = 2.408235313815748;
        expol(3, 4) = 0.1452847075210439;
        expol(3, 5) = -0.03892892005285509;
        expol(3, 6) = 0.1452847075210439;
        expol(3, 7) = -0.542209910031327;
        expol(4, 0) = -0.2702847075210531;
        expol(4, 1) = -0.2702847075210531;
        expol(4, 2) = -0.2702847075210531;
        expol(4, 3) = -0.2702847075210531;
        expol(4, 4) = 0.520284707521053;
        expol(4, 5) = 0.520284707521053;
        expol(4, 6) = 0.520284707521053;
        expol(4, 7) = 0.520284707521053;
        return expol;
      });

  return extrapolation_matrix;
}

template <Core::FE::CellType distype, class GaussIntegration>
Core::LinAlg::SerialDenseMatrix Core::FE::evaluate_gauss_points_to_nurbs_knots_extrapolation_matrix(
    const Core::FE::Discretization& dis, const Core::Elements::Element& ele,
    const GaussIntegration& intpoints)
{
  static std::unordered_map<unsigned, Core::LinAlg::SerialDenseMatrix> extrapolation_matrix_cache{};

  if (extrapolation_matrix_cache.find(intpoints.num_points()) == extrapolation_matrix_cache.end())
  {
    Core::FE::CellType base_distype =
        get_gauss_point_extrapolation_base_distype<distype>(intpoints.num_points());

    FOUR_C_ASSERT(Core::FE::get_number_of_element_nodes(base_distype) <= intpoints.num_points(),
        "The base discretization has more nodes than Gauss points. The extrapolation is not "
        "unique! "
        "This should not happen. The evaluation of the base extrapolation type for the number of "
        "gauss points is not correct.");

    LinAlg::SerialDenseMatrix shapefcns_at_gps =
        evaluate_nurbs_base_shape_functions_at_gauss_points<distype>(
            dis, ele, base_distype, intpoints);

    LinAlg::SerialDenseMatrix matrix_gp_to_base =
        evaluate_projection_gauss_points_to_base_distype(shapefcns_at_gps);

    extrapolation_matrix_cache[intpoints.num_points()] =
        evaluate_projection_gauss_points_to_distype<distype>(matrix_gp_to_base, base_distype);
  }

  return extrapolation_matrix_cache[intpoints.num_points()];
}

template <Core::FE::CellType distype, class GaussIntegration>
void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const GaussIntegration& integration)
{
  Core::LinAlg::SerialDenseMatrix nodal_quantity(Core::FE::num_nodes<distype>, gp_data.numCols());
  Core::LinAlg::multiply(nodal_quantity,
      evaluate_gauss_points_to_nodes_extrapolation_matrix<distype>(integration), gp_data);

  assemble_extrapolated_nodal_values(global_data, nodal_quantity, ele, nodal_average);
}

template <Core::FE::CellType distype, class GaussIntegration>
void Core::FE::extrapolate_gp_quantity_to_nurbs_knots_and_assemble(
    const Core::FE::Discretization& dis, const Core::Elements::Element& ele,
    const LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const GaussIntegration& integration)
{
  Core::LinAlg::SerialDenseMatrix nodal_quantity(Core::FE::num_nodes<distype>, gp_data.numCols());
  Core::LinAlg::multiply(nodal_quantity,
      evaluate_gauss_points_to_nurbs_knots_extrapolation_matrix<distype>(dis, ele, integration),
      gp_data);

  assemble_extrapolated_nodal_values(global_data, nodal_quantity, ele, nodal_average);
}

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex8,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex8,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex18,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex18,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex20,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex20,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex27,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex27,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::nurbs27,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::nurbs27,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet4,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet4,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet10,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tet10,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge6,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge6,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge15,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::wedge15,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::pyramid5,
    Core::FE::GaussIntegration>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::pyramid5,
    Core::FE::IntegrationPoints3D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints3D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad4,
    Core::FE::IntegrationPoints2D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints2D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad8,
    Core::FE::IntegrationPoints2D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints2D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::quad9,
    Core::FE::IntegrationPoints2D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints2D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tri3,
    Core::FE::IntegrationPoints2D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints2D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::tri6,
    Core::FE::IntegrationPoints2D>(const Core::Elements::Element& ele,
    const Core::LinAlg::SerialDenseMatrix& gp_data, Core::LinAlg::MultiVector<double>& global_data,
    bool nodal_average, const Core::FE::IntegrationPoints2D& integration);

template void Core::FE::extrapolate_gp_quantity_to_nurbs_knots_and_assemble<
    Core::FE::CellType::nurbs27, Core::FE::GaussIntegration>(const Core::FE::Discretization& dis,
    const Core::Elements::Element& ele, const LinAlg::SerialDenseMatrix& gp_data,
    Core::LinAlg::MultiVector<double>& global_data, bool nodal_average,
    const GaussIntegration& integration);
template void Core::FE::extrapolate_gp_quantity_to_nurbs_knots_and_assemble<
    Core::FE::CellType::nurbs27, Core::FE::IntegrationPoints3D>(const Core::FE::Discretization& dis,
    const Core::Elements::Element& ele, const LinAlg::SerialDenseMatrix& gp_data,
    Core::LinAlg::MultiVector<double>& global_data, bool nodal_average,
    const Core::FE::IntegrationPoints3D& integration);

FOUR_C_NAMESPACE_CLOSE
