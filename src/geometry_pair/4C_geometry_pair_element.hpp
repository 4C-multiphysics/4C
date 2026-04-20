// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_ELEMENT_HPP
#define FOUR_C_GEOMETRY_PAIR_ELEMENT_HPP


#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"

#include <type_traits>

FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  /**
   * \brief Geometry discretization type of element.
   */
  enum class DiscretizationTypeGeometry
  {
    //! none
    none,
    //! 1D curve
    line,
    //! triangle
    triangle,
    //! quadrilateral
    quad,
    //! hexahedron
    hexahedron,
    //! tetraeder
    tetraeder,
    //! wedge
    wedge
  };

  /**
   * \brief type of basis used for the shape functions.
   */
  enum class BasisType
  {
    //! standard
    standard,
    //! dual
    dual,
  };

  /**
   * \brief This structure "converts" the discretization type to a geometry type.
   *
   * For some geometry pairs we need to know if a geometry is a triangle / a quad / tetrahedra or
   * hexahedron (linear, quadratic, ...) this structure "returns" the correct type depending on the
   * discretization type of the element.
   */
  template <Core::FE::CellType discretization>
  struct ElementDiscretizationToGeometryType
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::none;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::line2>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::line;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::tri3>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::triangle;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::tri6>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::triangle;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::quad4>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::quad;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::quad8>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::quad;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::quad9>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::quad;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::nurbs9>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::quad;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::hex8>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::hexahedron;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::hex20>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::hexahedron;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::hex27>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::hexahedron;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::tet4>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::tetraeder;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::tet10>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::tetraeder;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::nurbs27>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::hexahedron;
  };
  template <>
  struct ElementDiscretizationToGeometryType<Core::FE::CellType::wedge6>
  {
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        GeometryPair::DiscretizationTypeGeometry::wedge;
  };

  /**
   * \brief Base class for the geometry pair element type.
   *
   * @tparam discretization Type of shape function.
   * @tparam values_per_node Number of nodal values per node (standard elements have 1, Hermitian
   * shape functions have 2)
   * @tparam spatial_dim Number of spatial dimensions. This affects the number of degrees of freedom
   * of the element
   */
  template <Core::FE::CellType discretization, unsigned int values_per_node,
      unsigned int spatial_dim = 3, BasisType type = BasisType::standard>
  class ElementDiscretizationBase
  {
   public:
    //! Type of shape function that will be used when evaluating the shape functions.
    static constexpr Core::FE::CellType discretization_ = discretization;

    //! Dimension of element (curve=1, surface=2, volume=3).
    static constexpr unsigned int element_dim_ = Core::FE::dim<discretization_>;

    //! Number of values per node.
    static constexpr unsigned int n_val_ = values_per_node;

    //! Number of nodes for this element.
    static constexpr unsigned int n_nodes_ = Core::FE::num_nodes(discretization_);

    //! Number of spatial dimensions.
    static const unsigned int spatial_dim_ = spatial_dim;

    //! Number of unknowns for this element.
    static constexpr unsigned int n_dof_ = spatial_dim_ * n_val_ * n_nodes_;

    //! Basis type used for shape function.
    static constexpr BasisType type_ = type;

    //! Geometry type of the element.
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        ElementDiscretizationToGeometryType<discretization_>::geometry_type_;
  };

  /**
   * Shortcuts to element types are created here, so the explicit template initializations are
   * better readable.
   */

  //! 1D elements
  using t_hermite = ElementDiscretizationBase<Core::FE::CellType::line2, 2>;
  using t_hermite_dual =
      ElementDiscretizationBase<Core::FE::CellType::line2, 2, 3, BasisType::dual>;
  using t_line2 = ElementDiscretizationBase<Core::FE::CellType::line2, 1>;
  using t_line2_scalar = ElementDiscretizationBase<Core::FE::CellType::line2, 1, 1>;
  using t_line3 = ElementDiscretizationBase<Core::FE::CellType::line3, 1>;
  using t_line4 = ElementDiscretizationBase<Core::FE::CellType::line4, 1>;
  using t_line5 = ElementDiscretizationBase<Core::FE::CellType::line5, 1>;

  //! 2D elements
  using t_tri3 = ElementDiscretizationBase<Core::FE::CellType::tri3, 1>;
  using t_tri6 = ElementDiscretizationBase<Core::FE::CellType::tri6, 1>;
  using t_quad4 = ElementDiscretizationBase<Core::FE::CellType::quad4, 1>;
  using t_quad8 = ElementDiscretizationBase<Core::FE::CellType::quad8, 1>;
  using t_quad9 = ElementDiscretizationBase<Core::FE::CellType::quad9, 1>;
  using t_nurbs9 = ElementDiscretizationBase<Core::FE::CellType::nurbs9, 1>;

  //! 3D elements
  using t_hex8 = ElementDiscretizationBase<Core::FE::CellType::hex8, 1>;
  using t_hex20 = ElementDiscretizationBase<Core::FE::CellType::hex20, 1>;
  using t_hex27 = ElementDiscretizationBase<Core::FE::CellType::hex27, 1>;
  using t_tet4 = ElementDiscretizationBase<Core::FE::CellType::tet4, 1>;
  using t_tet10 = ElementDiscretizationBase<Core::FE::CellType::tet10, 1>;
  using t_nurbs27 = ElementDiscretizationBase<Core::FE::CellType::nurbs27, 1>;
  using t_wedge6 = ElementDiscretizationBase<Core::FE::CellType::wedge6, 1>;


  /**
   * \brief Compile time struct to check if an element type is based on Lagrange shape functions
   */
  template <typename ElementType>
  struct IsLagrangeElement
  {
    static const bool value_ = false;
  };

  template <>
  struct IsLagrangeElement<t_line2>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_line2_scalar>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_line3>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_line4>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_line5>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_hex8>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_hex20>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_hex27>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_tet4>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_tet10>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsLagrangeElement<t_wedge6>
  {
    static const bool value_ = true;
  };


  /**
   * \brief Compile time struct to check if an element type is based on NURBS shape functions
   */
  template <typename ElementType>
  struct IsNurbsElement
  {
    static const bool value_ = false;
  };

  template <>
  struct IsNurbsElement<t_nurbs9>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsNurbsElement<t_nurbs27>
  {
    static const bool value_ = true;
  };

  /**
   * \brief Compile time struct to check if an element type is a surface element with averaged nodal
   * normals
   */
  template <typename ElementType>
  struct IsSurfaceAveragedNormalsElement
  {
    static const bool value_ = false;
  };

  template <>
  struct IsSurfaceAveragedNormalsElement<t_quad4>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsSurfaceAveragedNormalsElement<t_quad8>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsSurfaceAveragedNormalsElement<t_quad9>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsSurfaceAveragedNormalsElement<t_tri3>
  {
    static const bool value_ = true;
  };

  template <>
  struct IsSurfaceAveragedNormalsElement<t_tri6>
  {
    static const bool value_ = true;
  };

  /**
   * \brief Data container with additional data needed to evaluate the shape functions of an element
   *
   * Per default this is empty as no additional data (besides the parameter coordinates) are needed
   * to evaluate the shape functions
   */
  template <typename ElementType, typename Enable = void>
  struct ShapeFunctionData
  {
  };

  /**
   * \brief Specialization for hermite elements which need a reference length
   */
  template <>
  struct ShapeFunctionData<t_hermite>
  {
    double ref_length_;
  };

  /**
   * \brief Specialization for hermite elements which need a reference length
   */
  template <>
  struct ShapeFunctionData<t_hermite_dual>
  {
    double ref_length_;
  };

  /**
   * \brief Specialization for nurbs9 surface elements which require an additional factor to specify
   * outward pointing normals
   */
  template <>
  struct ShapeFunctionData<t_nurbs9>
  {
    Core::LinAlg::Matrix<t_nurbs9::n_nodes_, 1, double> weights_;
    std::vector<Core::LinAlg::SerialDenseVector> myknots_;
    double surface_normal_factor_;
  };

  /**
   * \brief Specialization for nurbs27 which require knot vectors and weights
   */
  template <>
  struct ShapeFunctionData<t_nurbs27>
  {
    Core::LinAlg::Matrix<t_nurbs27::n_nodes_, 1, double> weights_;
    std::vector<Core::LinAlg::SerialDenseVector> myknots_;
  };

  /**
   * \brief Structure to set the shape function data container
   */
  template <typename ElementType, typename Enable = void>
  struct SetShapeFunctionData
  {
    static void set(
        ShapeFunctionData<ElementType>& shape_function_data, const Core::Elements::Element* element)
    {
      // Per default this is empty, for all shape functions which don't need additional data
    }
  };


  /**
   * \brief Specialization for nurbs9 elements
   *
   * Here we have to store the knot vector, weights and the normal factor for the nurbs surface
   * element
   */
  template <>
  struct SetShapeFunctionData<t_nurbs9>
  {
    static void set(
        ShapeFunctionData<t_nurbs9>& shape_function_data, const Core::Elements::Element* element)
    {
      const auto* discretization = Global::Problem::instance()->get_dis("structure").get();
      if (dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(discretization) == nullptr)
        FOUR_C_THROW(
            "Evaluation of the shape function data for nurbs requires a valid nurbs discretization "
            "pointer");

      auto face_element = dynamic_cast<const Core::Elements::FaceElement*>(element);
      if (face_element == nullptr)
      {
        // The element is a direct surface element, e.g., shell
        const bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
            *(discretization), element, shape_function_data.myknots_, shape_function_data.weights_);
        if (zero_size)
          FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");
        shape_function_data.surface_normal_factor_ = 1.0;
      }
      else
      {
        // The element is the face of a nurbs27 element
        std::vector<Core::LinAlg::SerialDenseVector> my_parent_knots(3);
        shape_function_data.myknots_.resize(2);
        const bool zero_size = Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(
            face_element, face_element->face_master_number(), face_element->parent_element_id(),
            *(discretization), my_parent_knots, shape_function_data.myknots_,
            shape_function_data.weights_, shape_function_data.surface_normal_factor_);
        if (zero_size)
          FOUR_C_THROW(
              "get_knot_vector_and_weights_for_nurbs_boundary has to return a non zero size.");
      }
    }
  };

  /**
   * \brief Specialization for nurbs27 elements
   */
  template <>
  struct SetShapeFunctionData<t_nurbs27>
  {
    static void set(
        ShapeFunctionData<t_nurbs27>& shape_function_data, const Core::Elements::Element* element)
    {
      const auto* discretization = Global::Problem::instance()->get_dis("structure").get();
      if (dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(discretization) == nullptr)
        FOUR_C_THROW(
            "Evaluation of the shape function data for nurbs requires a valid nurbs discretization "
            "pointer");

      const bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
          *discretization, element, shape_function_data.myknots_, shape_function_data.weights_);
      if (zero_size) FOUR_C_THROW("get_my_nurbs_knots_and_weights has to return a non zero size.");
    }
  };

  /**
   * \brief Data container warping everything required to evaluate field functions on the elements
   */
  template <typename ElementType, typename ScalarType, typename Enable = void>
  struct ElementData
  {
    Core::LinAlg::Matrix<ElementType::n_dof_, 1, ScalarType> element_position_;
    ShapeFunctionData<ElementType> shape_function_data_;
  };

  /**
   * \brief Specialization for elements with averaged nodal normals, here we have to also store the
   * normals at the nodes to ensure we can fully evaluate the element
   */
  template <typename ElementType, typename ScalarType>
  struct ElementData<ElementType, ScalarType,
      typename std::enable_if<IsSurfaceAveragedNormalsElement<ElementType>::value_>::type>
  {
    Core::LinAlg::Matrix<ElementType::n_dof_, 1, ScalarType> element_position_;
    Core::LinAlg::Matrix<ElementType::n_dof_, 1, ScalarType> nodal_normals_;
    ShapeFunctionData<ElementType> shape_function_data_;
  };

  /**
   * \brief Struct to initialize element data containers with the correct shape function data
   */
  template <typename ElementType, typename ScalarType>
  struct InitializeElementData
  {
    static GeometryPair::ElementData<ElementType, ScalarType> initialize(
        const Core::Elements::Element* element)
    {
      GeometryPair::ElementData<ElementType, ScalarType> element_data;
      SetShapeFunctionData<ElementType>::set(element_data.shape_function_data_, element);
      return element_data;
    }
  };

  /**
   * \brief Struct to convert a FAD element data container to an element data container of
   * type double
   */
  template <typename ElementType, typename Enable = void>
  struct ElementDataToDouble
  {
    template <typename ScalarType>
    static GeometryPair::ElementData<ElementType, double> to_double(
        const GeometryPair::ElementData<ElementType, ScalarType>& element_data)
    {
      auto element_data_double = ElementData<ElementType, double>();
      element_data_double.shape_function_data_ = element_data.shape_function_data_;
      element_data_double.element_position_ =
          Core::FADUtils::cast_to_double(element_data.element_position_);
      return element_data_double;
    }
  };

  /**
   * \brief Specialization for Elements with averaged nodal normals
   */
  template <typename ElementType>
  struct ElementDataToDouble<ElementType,
      typename std::enable_if<IsSurfaceAveragedNormalsElement<ElementType>::value_>::type>
  {
    template <typename ScalarType>
    static GeometryPair::ElementData<ElementType, double> to_double(
        const GeometryPair::ElementData<ElementType, ScalarType>& element_data)
    {
      auto element_data_double = ElementData<ElementType, double>();
      element_data_double.shape_function_data_ = element_data.shape_function_data_;
      element_data_double.element_position_ =
          Core::FADUtils::cast_to_double(element_data.element_position_);
      element_data_double.nodal_normals_ =
          Core::FADUtils::cast_to_double(element_data.nodal_normals_);
      return element_data_double;
    }
  };

  /**
   * \brief Struct to print the element data container to screen
   */
  template <typename ElementType, typename Enable = void>
  struct PrintElementData
  {
    template <typename ScalarType>
    static void print(const ElementData<ElementType, ScalarType>& element_data, std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
    }
  };

  /**
   * \brief Specialization for NURBS elements
   */
  template <typename ElementType>
  struct PrintElementData<ElementType,
      typename std::enable_if<IsNurbsElement<ElementType>::value_>::type>
  {
    template <typename ScalarType>
    static void print(const ElementData<ElementType, ScalarType>& element_data, std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
      out << "\nElement knot vectors: ";
      for (const auto& knot : element_data.shape_function_data_.myknots_) knot.print(out);
      out << "\nElement weight vector: ";
      element_data.shape_function_data_.weights_.print(out);
    }
  };

  /**
   * \brief Specialization for Hermite elements
   */
  template <>
  struct PrintElementData<t_hermite>
  {
    template <typename ScalarType>
    static void print(const ElementData<t_hermite, ScalarType>& element_data, std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement reference length: " << element_data.shape_function_data_.ref_length_;
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
    }
  };

  /**
   * \brief Specialization for dual Hermite elements
   */
  template <>
  struct PrintElementData<t_hermite_dual>
  {
    template <typename ScalarType>
    static void print(
        const ElementData<t_hermite_dual, ScalarType>& element_data, std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement reference length: " << element_data.shape_function_data_.ref_length_;
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
    }
  };

  /**
   * \brief Specialization for elements with averaged nodal normals
   */
  template <typename ElementType>
  struct PrintElementData<ElementType,
      typename std::enable_if<IsSurfaceAveragedNormalsElement<ElementType>::value_>::type>
  {
    template <typename ScalarType>
    static void print(const ElementData<ElementType, ScalarType>& element_data, std::ostream& out)
    {
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      out << "\nElement state vector: ";
      element_data.element_position_.print(out);
      out << "\nElement nodal normals: ";
      element_data.nodal_normals_.print(out);
    }
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
