// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL_KL_NURBS_HPP
#define FOUR_C_SHELL_KL_NURBS_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <memory>


FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class KirchhoffLoveShellNurbsType : public Core::Elements::ElementType
    {
     public:
      [[nodiscard]] std::string name() const override { return "KirchhoffLoveShellNurbsType"; }

      static KirchhoffLoveShellNurbsType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, int const numdof, int const dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static KirchhoffLoveShellNurbsType instance_;
    };

    /**
     * @brief A Kirchhoff-Love shell element based on Kiendl, Josef & Bletzinger, Kai-Uwe & Linhard,
     * J. & Wuechner, Roland. (2009). Isogeometric shell analysis with Kirchhoff-Love elements.
     */
    class KirchhoffLoveShellNurbs : public Core::Elements::Element
    {
     public:
      /**
       * @brief Standard constructor
       */
      KirchhoffLoveShellNurbs(int id, int owner);

      /**
       * @brief Copy constructor
       */
      KirchhoffLoveShellNurbs(const KirchhoffLoveShellNurbs& old);

      /**
       * @brief Deep copy this instance of KirchhoffLoveShellNurbs and return pointer to the copy
       */
      [[nodiscard]] Core::Elements::Element* clone() const override;

      /**
       * @brief Get shape type of element
       */
      [[nodiscard]] Core::FE::CellType shape() const override { return Core::FE::CellType::nurbs9; }

      /**
       * @brief Set discretization type of element
       */
      virtual void set_dis_type(Core::FE::CellType shape)
      {
        if (shape != Core::FE::CellType::nurbs9)
          FOUR_C_THROW("The KirchhoffLoveShellNurbs element is only implemented for NURBS9");
      };

      /**
       * @brief Return number of lines of this element
       */
      [[nodiscard]] int num_line() const override { return 4; }

      /**
       * @brief Return number of surfaces of this element
       */
      [[nodiscard]] int num_surface() const override { return 1; }

      /**
       * @brief Get vector of std::shared_ptrs to the surfaces of this element
       */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      /**
       * @brief Return unique ParObject id
       *
       * Every class implementing ParObject needs a unique id defined at the top of this file.
       */
      int unique_par_object_id() const override
      {
        return KirchhoffLoveShellNurbsType::instance().unique_par_object_id();
      }

      /**
       * @brief Pack this class so it can be communicated
       */
      void pack(Core::Communication::PackBuffer& data) const override;

      /**
       * @brief Unpack data from a char vector into this class
       */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      /**
       * @brief Get number of degrees of freedom for a certain node
       */
      [[nodiscard]] int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

      /**
       * @brief Get number of degrees of freedom per element, this element does not carry individual
       * DOFs
       */
      [[nodiscard]] int num_dof_per_element() const override { return 0; }

      /**
       * @brief Return the element type instance
       */
      [[nodiscard]] Core::Elements::ElementType& element_type() const override
      {
        return KirchhoffLoveShellNurbsType::instance();
      }

      /**
       * @brief Read input for this element
       */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      /**
       * @brief Set the parameter interface ptr for the solid elements
       */
      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      /**
       * @brief Returns true if the parameter interface is defined and initialized, otherwise false
       */
      [[nodiscard]] bool is_params_interface() const override
      {
        return (interface_ptr_ != nullptr);
      }

      /**
       * @brief Evaluate the element
       */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      /**
       * @brief This method evaluates a surfaces Neumann condition on the shell element
       */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      /**
       * @brief Calculate the residuum of this shell element
       *
       * The code is automatically generated with AceGen. The corresponding AceGen script is located
       * in the `script` subfolder. Functional changes should only be made there.
       */
      static void evaluate_residuum_auto_generated(const double young, const double poisson,
          const double thickness, const Core::FE::IntegrationPoints1D& intpointsXi,
          const Core::FE::IntegrationPoints1D& intpointsEta,
          const std::vector<Core::LinAlg::SerialDenseVector>& knots,
          const Core::LinAlg::Matrix<9, 1>& weights, const Core::LinAlg::Matrix<9, 3>& X,
          const Core::LinAlg::Matrix<27, 1>& q, Core::LinAlg::SerialDenseVector& res);

      /**
       * @brief Calculate the residuum and Jacobian of this shell element
       *
       * The code is automatically generated with AceGen. The corresponding AceGen script is located
       * in the `script` subfolder. Functional changes should only be made there.
       */
      static void evaluate_residuum_and_jacobian_auto_generated(const double young,
          const double poisson, const double thickness,
          const Core::FE::IntegrationPoints1D& intpointsXi,
          const Core::FE::IntegrationPoints1D& intpointsEta,
          const std::vector<Core::LinAlg::SerialDenseVector>& knots,
          const Core::LinAlg::Matrix<9, 1>& weights, const Core::LinAlg::Matrix<9, 3>& X,
          const Core::LinAlg::Matrix<27, 1>& q, Core::LinAlg::SerialDenseVector& res,
          Core::LinAlg::SerialDenseMatrix& stiff);

      /**
       * @brief Calculate a body load on this shell element
       *
       * The code is automatically generated with AceGen. The corresponding AceGen script is located
       * in the `script` subfolder. Functional changes should only be made there.
       */
      static void evaluate_body_load_auto_generated(const Core::FE::IntegrationPoints1D& intpoints,
          const std::vector<Core::LinAlg::SerialDenseVector>& knots,
          const Core::LinAlg::Matrix<9, 1>& weights, const Core::LinAlg::Matrix<9, 3>& X,
          const std::function<Core::LinAlg::Matrix<3, 1>(const double*)>& bodyload,
          Core::LinAlg::SerialDenseVector& elementload);

     private:
      //! We don't want assignment operator
      KirchhoffLoveShellNurbs& operator=(const KirchhoffLoveShellNurbs& old);

     private:
      //! Number of the material law
      int material_;

      //! Gaussian points in each parameter direction
      std::array<Core::FE::GaussRule1D, 2> gaussrule_;

      //! Data exchange between the element and the time integrator
      std::shared_ptr<Solid::Elements::ParamsInterface> interface_ptr_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
