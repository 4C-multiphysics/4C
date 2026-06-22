// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_NONLOCAL_STIMULUS_HPP
#define FOUR_C_SCATRA_ELE_CALC_NONLOCAL_STIMULUS_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class MixtureConstituentRemodelFiberSsi;
}

namespace Mat
{
  class Mixture;
}

namespace Discret::Elements
{
  /*!
   * @brief Scatra element calculator for the Helmholtz non-local G&R stimulus equation:
   *
   *   \f$\psi - \ell_c^2 \Delta\psi = (\sigma - \sigma_h)\f$
   *
   * The solution \psi is then used by the structural mixture constituent
   * MixtureConstituentRemodelFiberSsi (non-local mode) to drive growth.
   */
  template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
  class ScaTraEleCalcNonlocalStimulus : public ScaTraEleCalc<distype, probdim>
  {
    using my = ScaTraEleCalc<distype, probdim>;

   public:
    /// singleton access method
    static ScaTraEleCalcNonlocalStimulus<distype, probdim>* instance(
        const int numdofpernode, const int numscal, const std::string& disname);

    int setup_calc(Core::Elements::Element* ele, Core::FE::Discretization& discretization) override;

   protected:
    ScaTraEleCalcNonlocalStimulus(
        const int numdofpernode, const int numscal, const std::string& disname);

    void get_material_params(const Core::Elements::Element* ele, std::vector<double>& densn,
        std::vector<double>& densnp, std::vector<double>& densam, double& visc,
        const int iquad) override;

    void materials(const std::shared_ptr<const Core::Mat::Material> material, const int k,
        double& densn, double& densnp, double& densam, double& visc, const int iquad) override;

    /// evaluates the material, sets up the coefficients of the Helmholtz equation
    /// and caches the local stimulus, therefore source term at the gauss point
    void mat_scatra_nls(const std::shared_ptr<const Core::Mat::Material> material, const int k,
        double& densn, double& densnp, double& densam, double& visc, const int iquad);

    void get_rhs_int(double& rhsint, const double densnp, const int k) override;

    int evaluate_action_od(Core::Elements::Element* ele, Teuchos::ParameterList& params,
        Core::FE::Discretization& discretization, const ScaTra::Action& action,
        Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

   private:
    void sysmat_od_mesh_nls(Core::LinAlg::SerialDenseMatrix& emat, int ndofpernodemesh);
    static std::shared_ptr<Mat::Mixture> get_struct_material(const Core::Elements::Element* ele);

    /// SSI RemodelFiber constituents per scalar (set in setup_calc)
    std::vector<const Mixture::MixtureConstituentRemodelFiberSsi*> constituent_;

    /// Local stimulus \f$(\sigma - \sigma_h)\f$ at the current Gauss point, per scalar
    std::vector<double> source_at_gp_;
  };

}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_SCATRA_ELE_CALC_NONLOCAL_STIMULUS_HPP
