// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_RED_AIRWAYS_INTERACINARDEP_IMPL_HPP
#define FOUR_C_RED_AIRWAYS_INTERACINARDEP_IMPL_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_red_airways_elementbase.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Discret
{
  namespace Elements
  {
    /// Interface base class for inter_acinar_dep_impl
    /*!
      This class exists to provide a common interface for all template
      versions of inter_acinar_dep_impl. The only function
      this class actually defines is Impl, which returns a pointer to
      the appropriate version of inter_acinar_dep_impl.
     */
    class RedInterAcinarDepImplInterface
    {
     public:
      /// Empty constructor
      RedInterAcinarDepImplInterface() {}
      /// Empty destructor
      virtual ~RedInterAcinarDepImplInterface() = default;  /// Evaluate the element
      virtual int evaluate(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra,
          std::shared_ptr<Core::Mat::Material> mat) = 0;

      virtual void initial(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec3_epetra,
          std::shared_ptr<const Core::Mat::Material> material) = 0;

      virtual void evaluate_terminal_bc(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          std::shared_ptr<Core::Mat::Material> mat) = 0;

      virtual void calc_flow_rates(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::LinAlg::SerialDenseVector& a_volumen,
          Core::LinAlg::SerialDenseVector& a_volumenp, std::vector<int>& lm,
          std::shared_ptr<Core::Mat::Material> mat) = 0;

      virtual void get_coupled_values(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          std::shared_ptr<Core::Mat::Material> material) = 0;

      /// Internal implementation class for inter-acinar linker element
      static RedInterAcinarDepImplInterface* impl(Discret::Elements::RedInterAcinarDep* acinus);
    };


    /// Internal inter-acinar linker implementation
    /*!
      This internal class keeps all the working arrays needed to
      calculate the inter-acinar linker element. Additionally the
      method Sysmat() provides a clean and fast element implementation.

      <h3>Purpose</h3>

      \author ismail
      \date 01/09
    */

    template <Core::FE::CellType distype>
    class InterAcinarDepImpl : public RedInterAcinarDepImplInterface
    {
     public:
      /// Constructor
      explicit InterAcinarDepImpl();

      //! number of nodes
      static constexpr int iel = Core::FE::num_nodes<distype>;


      /// Evaluate
      /*!
        The evaluate function for the general inter-acinar linker case.
       */
      int evaluate(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra,
          std::shared_ptr<Core::Mat::Material> mat) override;

      /*!
        \brief calculate element matrix and rhs

        \param ele              (i) the element those matrix is calculated
        \param eqnp             (i) nodal volumetric flow rate at n+1
        \param evelnp           (i) nodal velocity at n+1
        \param eareanp          (i) nodal cross-sectional area at n+1
        \param eprenp           (i) nodal pressure at n+1
        \param estif            (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) acinus material/dimesion
        \param time             (i) current simulation time
        \param dt               (i) timestep
        */
      void sysmat(std::vector<double>& ial, Core::LinAlg::SerialDenseMatrix& sysmat,
          Core::LinAlg::SerialDenseVector& rhs);


      void evaluate_terminal_bc(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& rhs,
          std::shared_ptr<Core::Mat::Material> material) override;

      /*!
        \brief get the initial values of the degrees of freedom at the node

        \param ele              (i) the element those matrix is calculated
        \param eqnp             (i) nodal volumetric flow rate at n+1
        \param evelnp           (i) nodal velocity at n+1
        \param eareanp          (i) nodal cross-sectional area at n+1
        \param eprenp           (i) nodal pressure at n+1
        \param estif            (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) acinus material/dimesion
        \param time             (i) current simulation time
        \param dt               (i) timestep
        */
      void initial(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& n_intr_acn_l,
          std::shared_ptr<const Core::Mat::Material> material) override;

      /*!
       \Essential functions to compute the results of essential matrices
      */
      void calc_flow_rates(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization,
          Core::LinAlg::SerialDenseVector& a_volumen_strain_np,
          Core::LinAlg::SerialDenseVector& a_volumenp, std::vector<int>& lm,
          std::shared_ptr<Core::Mat::Material> mat) override {};

      /*!
       \Essential functions to evaluate the coupled results
      */
      void get_coupled_values(RedInterAcinarDep* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          std::shared_ptr<Core::Mat::Material> material) override {};

     private:
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
