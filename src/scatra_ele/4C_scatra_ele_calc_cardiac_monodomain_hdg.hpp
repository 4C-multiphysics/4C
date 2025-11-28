// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_CARDIAC_MONODOMAIN_HDG_HPP
#define FOUR_C_SCATRA_ELE_CALC_CARDIAC_MONODOMAIN_HDG_HPP


#include "4C_config.hpp"

#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_scatra_ele_calc_hdg.hpp"
#include "4C_scatra_ele_hdg.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    /// Scatra HDG element implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcHDGCardiacMonodomain : public ScaTraEleCalcHDG<distype, probdim>
    {
     protected:
     private:
      /// (private) protected constructor, since we are a Singleton.
      /// this constructor is called from a derived class
      /// -> therefore, it has to be protected instead of private
      ScaTraEleCalcHDGCardiacMonodomain(
          const int numdofpernode, const int numscal, const std::string& disname);

      std::vector<Core::LinAlg::SerialDenseVector> values_mat_gp_all_;
      std::vector<double> gp_mat_alpha_;

     public:
      /// Singleton access method
      static ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>* instance(const int numdofpernode,
          const int numscal, const std::string& disname, bool create = true);

      //! evaluate the element
      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
      {
        return 0;
      };

     protected:
      /*========================================================================*/
      //! @name material and related and related functions
      /*========================================================================*/

      //! evaluate material
      void prepare_materials(Core::Elements::Element* ele,  //!< the element we are dealing with
          const std::shared_ptr<const Core::Mat::Material>
              material,  //!< pointer to current material
          const int k,   //!< id of current scalar
          std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor) override;

      //! evaluate material
      virtual void prepare_materials_all(
          Core::Elements::Element* ele,  //!< the element we are dealing with
          const std::shared_ptr<const Core::Mat::Material>
              material,  //!< pointer to current material
          const int k,   //!< id of current scalar
          std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor);

      //! evaluate material
      virtual void prepare_materials_tet(
          Core::Elements::Element* ele,  //!< the element we are dealing with
          const std::shared_ptr<const Core::Mat::Material>
              material,  //!< pointer to current material
          const int k,   //!< id of current scalar
          std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor);


      //! evaluate material
      void materials(const std::shared_ptr<const Core::Mat::Material>
                         material,  //!< pointer to current material
          const int k,              //!< id of current scalar
          Core::LinAlg::SerialDenseMatrix& difftensor, Core::LinAlg::SerialDenseVector& ivecn,
          Core::LinAlg::SerialDenseVector& ivecnp,
          Core::LinAlg::SerialDenseMatrix& ivecnpderiv) override;

      //! material ScaTra
      void mat_myocard(const std::shared_ptr<const Core::Mat::Material>
                           material,  //!< pointer to current material
          const int k,                //!< id of current scalar
          Core::LinAlg::SerialDenseMatrix& difftensor, Core::LinAlg::SerialDenseVector& ivecn,
          Core::LinAlg::SerialDenseVector& ivecnp, Core::LinAlg::SerialDenseMatrix& ivecnpderiv);

      //! update time dependent material
      void time_update_material(
          const Core::Elements::Element* ele  //!< the element we are dealing with
          ) override;

      //! get material internal state for output
      void get_material_internal_state(const Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization) override;

      //! set material internal state after restart
      void set_material_internal_state(const Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization) override;

      //! project material field
      int project_material_field(const Core::Elements::Element* ele) override;

      //! project material field
      int project_material_field_all(const Core::Elements::Element* ele);

      //! project material field for Tet elements, because quadrature not working for higher order
      //! polynomials
      int project_material_field_tet(const Core::Elements::Element* ele);

      /*!
       * @brief Setup cardiac fibers. If the fiber direction fiber1 is directly given, it is
       * returned. Otherwise, the fiber will be calculated with the RAD-AXI-CIR coordinate system
       * and the helix and transversal angles
       *
       * @tparam dim space dimension of the problem
       * @param fibers Fiber data projected
       * @param f Cardiac fiber direction setup from the fiber or coordinate system data
       */
      template <std::size_t dim>
      void setup_cardiac_fibers(const Core::Nodes::NodalFiberHolder& fibers,
          std::vector<Core::LinAlg::Tensor<double, dim>>& f);

      /// polynomial space for element interior for various Gauss Points for the evaluation of the
      /// material
      std::shared_ptr<Core::FE::PolynomialSpace<probdim>> polySpace_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
