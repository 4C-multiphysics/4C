// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_BOUNDARY_CALC_LOMA_HPP
#define FOUR_C_SCATRA_ELE_BOUNDARY_CALC_LOMA_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_boundary_calc.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // class implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype> + 1>
    class ScaTraEleBoundaryCalcLoma : public ScaTraEleBoundaryCalc<distype, probdim>
    {
      using my = Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

     public:
      //! singleton access method
      static ScaTraEleBoundaryCalcLoma<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);


      //! evaluate action
      int evaluate_action(Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, ScaTra::BoundaryAction action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

     private:
      //! private constructor for singletons
      ScaTraEleBoundaryCalcLoma(
          const int numdofpernode, const int numscal, const std::string& disname);

      //! evaluate loma thermal press
      void calc_loma_therm_press(Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la);

      //! calculate Neumann inflow boundary conditions
      void neumann_inflow(const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs) override;

      //! integral of normal diffusive flux and velocity over boundary surface
      void norm_diff_flux_and_vel_integral(const Core::Elements::Element* ele,
          Teuchos::ParameterList& params, const std::vector<double>& enormdiffflux,
          const std::vector<double>& enormvel);

      //! thermodynamic pressure
      double thermpress_;
    };
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
