// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_poro_p1.hpp"

#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_parameter_poro.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"

FOUR_C_NAMESPACE_OPEN


template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcPoroP1<distype>*
Discret::Elements::FluidEleCalcPoroP1<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidEleCalcPoroP1<distype>>(
            new Discret::Elements::FluidEleCalcPoroP1<distype>());
      });

  return singleton_owner.instance(action);
}

template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcPoroP1<distype>::FluidEleCalcPoroP1()
    : Discret::Elements::FluidEleCalcPoro<distype>::FluidEleCalcPoro()
{
}

template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcPoroP1<distype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, const Core::FE::GaussIntegration& intpoints)
{
  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (Base::isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
        discretization, ele, Base::myknots_, Base::weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  // set element id
  Base::eid_ = ele->id();
  // get structure material
  Base::get_struct_material(ele);

  // rotationally symmetric periodic bc's: do setup for current element
  // (only required to be set up for routines "extract_values_from_global_vector")
  Base::rotsymmpbc_->setup(ele);

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes,
  // with pressure gradient prescribed as body force included for turbulent
  // channel flow and with scatra body force included for variable-density flow
  // (evaluation at time n+alpha_F for generalized-alpha scheme,
  //  and at time n+1 otherwise)
  // ---------------------------------------------------------------------
  static Core::LinAlg::Matrix<nsd_, nen_> ebofoaf(Core::LinAlg::Initialization::zero);
  ebofoaf.clear();
  static Core::LinAlg::Matrix<nsd_, nen_> eprescpgaf(Core::LinAlg::Initialization::zero);
  eprescpgaf.clear();
  static Core::LinAlg::Matrix<nen_, 1> escabofoaf(Core::LinAlg::Initialization::zero);
  escabofoaf.clear();
  Base::body_force(ele, ebofoaf, eprescpgaf, escabofoaf);

  // ---------------------------------------------------------------------
  // get all general state vectors: velocity/pressure, acceleration
  // and history
  // velocity/pressure values are at time n+alpha_F/n+alpha_M for
  // generalized-alpha scheme and at time n+1/n for all other schemes
  // acceleration values are at time n+alpha_M for
  // generalized-alpha scheme and at time n+1 for all other schemes
  // ---------------------------------------------------------------------
  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1
  // ost:         velocity/pressure at time n+1
  static Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  evelaf.clear();
  static Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  epreaf.clear();
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &evelaf, &epreaf, "velaf");

  // np_genalpha: additional vector for velocity at time n+1
  static Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  evelnp.clear();
  static Core::LinAlg::Matrix<nen_, 1> eprenp(Core::LinAlg::Initialization::zero);
  eprenp.clear();
  if (FluidEleCalc<distype>::fldparatimint_->is_genalpha_np())
    Base::extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, &evelnp, &eprenp, "velnp");

  static Core::LinAlg::Matrix<nsd_, nen_> emhist(Core::LinAlg::Initialization::zero);
  emhist.clear();
  static Core::LinAlg::Matrix<nen_, 1> echist(Core::LinAlg::Initialization::zero);
  echist.clear();
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &emhist, &echist, "hist");

  static Core::LinAlg::Matrix<nsd_, nen_> eaccam(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nen_, 1> epressam_timederiv(Core::LinAlg::Initialization::zero);
  eaccam.clear();
  epressam_timederiv.clear();

  if (Base::fldparatimint_->is_genalpha())
    Base::extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, &eaccam, &epressam_timederiv, "accam");

  static Core::LinAlg::Matrix<nen_, 1> epressn_timederiv(Core::LinAlg::Initialization::zero);
  epressn_timederiv.clear();
  if (Base::fldparatimint_->is_genalpha())
    Base::extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, nullptr, &epressn_timederiv, "accn");

  static Core::LinAlg::Matrix<nen_, 1> epren(Core::LinAlg::Initialization::zero);
  epren.clear();
  static Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
  eveln.clear();
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &eveln, &epren, "veln");

  static Core::LinAlg::Matrix<nen_, 1> epressnp_timederiv(Core::LinAlg::Initialization::zero);
  epressnp_timederiv.clear();
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, nullptr, &epressnp_timederiv, "accnp");

  static Core::LinAlg::Matrix<nen_, 1> escaaf(Core::LinAlg::Initialization::zero);
  escaaf.clear();
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, nullptr, &escaaf, "scaaf");

  // ---------------------------------------------------------------------
  // get additional state vectors for ALE case: grid displacement and vel.
  // ---------------------------------------------------------------------
  static Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  edispnp.clear();
  static Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);
  egridv.clear();
  static Core::LinAlg::Matrix<nsd_, nen_> egridvn(Core::LinAlg::Initialization::zero);
  egridvn.clear();
  static Core::LinAlg::Matrix<nsd_, nen_> edispn(Core::LinAlg::Initialization::zero);
  edispn.clear();

  static Core::LinAlg::Matrix<nen_, 1> eporositynp(Core::LinAlg::Initialization::zero);
  eporositynp.clear();
  static Core::LinAlg::Matrix<nen_, 1> eporositydot(Core::LinAlg::Initialization::zero);
  eporositydot.clear();
  static Core::LinAlg::Matrix<nen_, 1> eporositydotn(Core::LinAlg::Initialization::zero);
  eporositydotn.clear();

  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &edispnp, &eporositynp, "dispnp");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &egridv, &eporositydot, "gridv");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &egridvn, &eporositydotn, "gridvn");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &edispn, nullptr, "dispn");

  // get node coordinates and number of elements per node
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, Base::xyze_);

  // construct views
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> elemat1(elemat1_epetra, true);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1> elevec1(elevec1_epetra, true);
  // elemat2 and elevec2+3 are currently not in use

  // call inner evaluate (does not know about element or discretization object)
  int result = Base::evaluate(params, ebofoaf, elemat1, elevec1, evelaf, epreaf, evelnp, eveln,
      eprenp, epren, emhist, echist, epressnp_timederiv, epressam_timederiv, epressn_timederiv,
      eaccam, edispnp, edispn, egridv, egridvn, escaaf, &eporositynp, &eporositydot, &eporositydotn,
      mat, ele->is_ale(), intpoints);

  return result;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::compute_porosity(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const Core::LinAlg::Matrix<nen_, 1>& shapfct, const Core::LinAlg::Matrix<nen_, 1>* myporosity,
    double& porosity, double* dphi_dp, double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ,
    double* dphi_dpp, bool save)
{
  if (myporosity == nullptr)
    FOUR_C_THROW("no porosity values given!!");
  else
    porosity = shapfct.dot(*myporosity);
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::compute_porosity_gradient(const double& dphidp,
    const double& dphidJ, const Core::LinAlg::Matrix<nsd_, 1>& gradJ,
    const Core::LinAlg::Matrix<nsd_, 1>& gradp, const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
    Core::LinAlg::Matrix<nsd_, 1>& grad_porosity, Core::LinAlg::Matrix<nsd_, 1>& refgrad_porosity)
{
  if (eporositynp == nullptr)
    FOUR_C_THROW("no porosity values given for calculation of porosity gradient!!");

  //--------------------- current porosity gradient
  grad_porosity.multiply(Base::derxy_, *eporositynp);

  refgrad_porosity.multiply(Base::xjm_, grad_porosity);
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::evaluate_pressure_equation(
    Teuchos::ParameterList& params, const double& timefacfacpre, const double& rhsfac,
    const double& dphi_dp, const double& dphi_dJ, const double& dphi_dJdp, const double& dphi_dpp,
    const Core::LinAlg::Matrix<nen_, 1>* eporositydot,
    const Core::LinAlg::Matrix<nen_, 1>* eporositydotn, const Core::LinAlg::Matrix<nen_, 1>& echist,
    const Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp,
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, nen_>& ppmat,
    Core::LinAlg::Matrix<nen_, 1>& preforce)
{
  // first evaluate terms without porosity time derivative
  Base::evaluate_pressure_equation_non_transient(params, timefacfacpre, rhsfac, dphi_dp, dphi_dJ,
      dphi_dJdp, dphi_dpp, dgradphi_dp, estif_q_u, ppmat, preforce);

  // now the porosity time derivative (different for standard poro and poro_p1 elements)
  if (!Base::porofldpara_->is_stationary_conti())
  {
    // inertia terms on the right hand side for instationary fluids

    if (eporositydot)
    {
      double porositydot = Base::funct_.dot(*eporositydot);
      // double porositydot =  Base::funct_.Dot(*eporositydotn);

      for (int vi = 0; vi < nen_; ++vi)
      {
        // check genalpha case
        preforce(vi) -= rhsfac * porositydot * Base::funct_(vi);
      }

      // just update internal variables, no contribution to rhs
      const double porositydotn = Base::funct_.dot(*eporositydotn);

      Base::hist_con_ =
          Base::fldparatimint_->om_theta() * Base::fldparatimint_->dt() * porositydotn;

      // rhs from last time step
      Base::rhscon_ =
          1.0 / Base::fldparatimint_->dt() / Base::fldparatimint_->theta() * Base::hist_con_;

      // transient part of continuity equation residual
      Base::conres_old_ += porositydot - Base::rhscon_;
    }
    else
      FOUR_C_THROW("no porosity time derivative given for poro_p1 element!");
  }
}

template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcPoroP1<distype>::evaluate_od(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, const Core::FE::GaussIntegration& intpoints)
{
  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (Base::isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
        discretization, ele, Base::myknots_, Base::weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  // set element id
  Base::eid_ = ele->id();

  // get structure material
  Base::get_struct_material(ele);

  // rotationally symmetric periodic bc's: do setup for current element
  // (only required to be set up for routines "extract_values_from_global_vector")
  Base::rotsymmpbc_->setup(ele);

  // construct views
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> elemat1(elemat1_epetra, true);
  //  Core::LinAlg::Matrix<(nsd_+1)*nen_,(nsd_+1)*nen_>
  //  elemat2(elemat2_epetra,true);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1> elevec1(elevec1_epetra, true);
  // elevec2 and elevec3 are currently not in use

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes,
  // with pressure gradient prescribed as body force included for turbulent
  // channel flow and with scatra body force included for variable-density flow
  // (evaluation at time n+alpha_F for generalized-alpha scheme,
  //  and at time n+1 otherwise)
  // ---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd_, nen_> ebofoaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> eprescpgaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escabofoaf(Core::LinAlg::Initialization::zero);
  this->body_force(ele, ebofoaf, eprescpgaf, escabofoaf);

  // ---------------------------------------------------------------------
  // get all general state vectors: velocity/pressure, acceleration
  // and history
  // velocity/pressure values are at time n+alpha_F/n+alpha_M for
  // generalized-alpha scheme and at time n+1/n for all other schemes
  // acceleration values are at time n+alpha_M for
  // generalized-alpha scheme and at time n+1 for all other schemes
  // ---------------------------------------------------------------------
  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1
  // ost:         velocity/pressure at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &evelaf, &epreaf, "velaf");

  // np_genalpha: additional vector for velocity at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> eprenp(Core::LinAlg::Initialization::zero);
  if (Base::fldparatimint_->is_genalpha_np())
    this->extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, &evelnp, &eprenp, "velnp");

  // np_genalpha: additional vector for velocity at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epren(Core::LinAlg::Initialization::zero);
  if (Base::fldparatimint_->is_genalpha_np())
    this->extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, &eveln, &epren, "veln");

  static Core::LinAlg::Matrix<nsd_, nen_> eaccam(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nen_, 1> epressam_timederiv(Core::LinAlg::Initialization::zero);
  eaccam.clear();
  epressam_timederiv.clear();

  if (Base::fldparatimint_->is_genalpha())
    Base::extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, &eaccam, &epressam_timederiv, "accam");

  static Core::LinAlg::Matrix<nen_, 1> epressn_timederiv(Core::LinAlg::Initialization::zero);
  epressn_timederiv.clear();
  if (Base::fldparatimint_->is_genalpha())
    Base::extract_values_from_global_vector(
        discretization, lm, *Base::rotsymmpbc_, nullptr, &epressn_timederiv, "accn");

  Core::LinAlg::Matrix<nen_, 1> epressnp_timederiv(Core::LinAlg::Initialization::zero);
  this->extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, nullptr, &epressnp_timederiv, "accnp");

  Core::LinAlg::Matrix<nen_, 1> escaaf(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, nullptr, &escaaf, "scaaf");

  Core::LinAlg::Matrix<nsd_, nen_> emhist(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> echist(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &emhist, &echist, "hist");

  // ---------------------------------------------------------------------
  // get additional state vectors for ALE case: grid displacement and vel.
  // ---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> edispn(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> egridvn(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<nen_, 1> eporositynp(Core::LinAlg::Initialization::zero);

  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &edispnp, &eporositynp, "dispnp");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &egridv, nullptr, "gridv");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &edispn, nullptr, "dispn");
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &egridvn, nullptr, "gridvn");

  // get node coordinates and number of elements per node
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, Base::xyze_);

  // call inner evaluate (does not know about element or discretization object)
  int result = evaluate_od(params, ebofoaf, elemat1, elevec1, evelaf, epreaf, evelnp, eveln, eprenp,
      epren, emhist, echist, epressnp_timederiv, epressam_timederiv, epressn_timederiv, eaccam,
      edispnp, edispn, egridv, egridvn, escaaf, &eporositynp, mat, ele->is_ale(), intpoints);

  return result;
}

template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcPoroP1<distype>::evaluate_od(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& elemat1,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& elevec1,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelnp, const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
    const Core::LinAlg::Matrix<nen_, 1>& eprenp, const Core::LinAlg::Matrix<nen_, 1>& epren,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nen_, 1>& echist,
    const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& edispn, const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridvn, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>* eporositynp, std::shared_ptr<Core::Mat::Material> mat,
    bool isale, const Core::FE::GaussIntegration& intpoints)
{
  // flag for higher order elements
  Base::is_higher_order_ele_ = IsHigherOrder<distype>::ishigherorder;
  // overrule higher_order_ele if input-parameter is set
  // this might be interesting for fast (but slightly
  // less accurate) computations
  if (Base::fldpara_->is_inconsistent()) Base::is_higher_order_ele_ = false;

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  sysmat_od(params, ebofoaf, evelaf, evelnp, eveln, epreaf, eprenp, epren, emhist, echist,
      epressnp_timederiv, epressam_timederiv, epressn_timederiv, eaccam, edispnp, edispn, egridv,
      egridvn, escaaf, eporositynp, elemat1, elevec1, mat, isale, intpoints);

  return 0;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::sysmat_od(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf, const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelnp, const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
    const Core::LinAlg::Matrix<nen_, 1>& epren, const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
    const Core::LinAlg::Matrix<nen_, 1>& echist,
    const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& edispn, const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridvn, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& ecoupl,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
    std::shared_ptr<const Core::Mat::Material> material, bool isale,
    const Core::FE::GaussIntegration& intpoints)
{
  //------------------------------------------------------------------------
  //  preliminary definitions and evaluations
  //------------------------------------------------------------------------
  // definition of matrices
  static Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_> ecoupl_u(
      Core::LinAlg::Initialization::zero);  // coupling matrix for momentum equation
  static Core::LinAlg::Matrix<nen_, nen_ * nsd_> ecoupl_p(
      Core::LinAlg::Initialization::zero);  // coupling matrix for continuity equation

  static Core::LinAlg::Matrix<nen_ * nsd_, nen_> ecouplp1_u(
      Core::LinAlg::Initialization::zero);  // coupling matrix for momentum equation
  static Core::LinAlg::Matrix<nen_, nen_> ecouplp1_p(
      Core::LinAlg::Initialization::zero);  // coupling matrix for continuity equation

  ecoupl_u.clear();
  ecoupl_p.clear();
  ecouplp1_u.clear();
  ecouplp1_p.clear();

  // material coordinates xyze0
  Base::xyze0_ = Base::xyze_;

  // add displacement when fluid nodes move in the ALE case (in poroelasticity this is always the
  // case)
  Base::xyze_ += edispnp;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // evaluate shape functions and derivatives at element center
  Base::eval_shape_func_and_derivs_at_ele_center();

  //------------------------------------------------------------------------
  //  start loop over integration points
  //------------------------------------------------------------------------
  gauss_point_loop_p1_od(params, ebofoaf, evelaf, evelnp, eveln, epreaf, eprenp, epren, emhist,
      echist, epressnp_timederiv, epressam_timederiv, epressn_timederiv, eaccam, edispnp, edispn,
      egridv, egridvn, escaaf, eporositynp, eforce, ecoupl_u, ecoupl_p, ecouplp1_u, ecouplp1_p,
      material, intpoints);
  //------------------------------------------------------------------------
  //  end loop over integration points
  //------------------------------------------------------------------------

  //------------------------------------------------------------------------
  //  add contributions to element matrix
  //------------------------------------------------------------------------

  // add fluid velocity-structure displacement part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int nsd_ui = nsd_ * ui;
    const int nsdp1_ui = (nsd_ + 1) * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int nsd_ui_jdim = nsd_ui + jdim;
      const int nsdp1_ui_jdim = nsdp1_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = Base::numdofpernode_ * vi;
        const int nsd_vi = nsd_ * vi;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          ecoupl(numdof_vi + idim, nsdp1_ui_jdim) += ecoupl_u(nsd_vi + idim, nsd_ui_jdim);
        }
      }
    }
  }

  // add fluid pressure-structure displacement part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int nsd_ui = nsd_ * ui;
    const int nsdp1_ui = (nsd_ + 1) * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int nsd_ui_jdim = nsd_ui + jdim;
      const int nsdp1_ui_jdim = nsdp1_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
      {
        ecoupl(Base::numdofpernode_ * vi + nsd_, nsdp1_ui_jdim) += ecoupl_p(vi, nsd_ui_jdim);
      }
    }
  }

  // add fluid velocity-structure porosity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int nsdp1_ui = (nsd_ + 1) * ui;

    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int nsdp1_ui_nsd = nsdp1_ui + nsd_;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = Base::numdofpernode_ * vi;
        const int nsd_vi = nsd_ * vi;

        ecoupl(numdof_vi + idim, nsdp1_ui_nsd) += ecouplp1_u(nsd_vi + idim, ui);
      }
    }
  }

  // add fluid pressure-structure porosity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int nsdp1_ui_nsd = (nsd_ + 1) * ui + nsd_;

    for (int vi = 0; vi < nen_; ++vi)
      ecoupl(Base::numdofpernode_ * vi + nsd_, nsdp1_ui_nsd) += ecouplp1_p(vi, ui);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::gauss_point_loop_p1_od(
    Teuchos::ParameterList& params, const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nen_, 1>& eprenp, const Core::LinAlg::Matrix<nen_, 1>& epren,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nen_, 1>& echist,
    const Core::LinAlg::Matrix<nen_, 1>& epressnp_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressam_timederiv,
    const Core::LinAlg::Matrix<nen_, 1>& epressn_timederiv,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& edispn, const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridvn, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>* eporositynp,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& ecoupl_u,
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& ecoupl_p,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& ecouplp1_u,
    Core::LinAlg::Matrix<nen_, nen_>& ecouplp1_p,
    std::shared_ptr<const Core::Mat::Material> material,
    const Core::FE::GaussIntegration& intpoints)
{
  // definition of velocity-based momentum residual vectors
  static Core::LinAlg::Matrix<nsd_, nen_ * nsd_> lin_resM_Dus(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nsd_, nen_ * nsd_> lin_resM_Dus_gridvel(
      Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nsd_, nen_> lin_resM_Dphi(Core::LinAlg::Initialization::zero);

  // set element area or volume
  const double vol = Base::fac_;

  for (Core::FE::GaussIntegration::const_iterator iquad = intpoints.begin();
      iquad != intpoints.end(); ++iquad)
  {
    lin_resM_Dus.clear();
    lin_resM_Dus_gridvel.clear();
    lin_resM_Dphi.clear();

    // evaluate shape functions and derivatives at integration point
    Base::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    // evaluate shape function derivatives w.r.t. to material coordinates at integration point
    Base::setup_material_derivatives();

    // -------------------------(material) deformation gradient F = d xyze_ / d XYZE = xyze_ *
    // N_XYZ_^T
    static Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
    Base::compute_def_gradient(defgrd, Base::N_XYZ_, Base::xyze_);

    // inverse deformation gradient F^-1
    static Core::LinAlg::Matrix<nsd_, nsd_> defgrd_inv(Core::LinAlg::Initialization::uninitialized);
    defgrd_inv.invert(defgrd);

    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J and the volume change
    Base::compute_jacobian_determinant_volume_change(
        Base::J_, volchange, defgrd, Base::N_XYZ_, edispnp);

    Base::evaluate_variables_at_gauss_point_od(params, ebofoaf, evelaf, evelnp, eveln, epreaf,
        eprenp, epren, epressnp_timederiv, epressam_timederiv, epressn_timederiv, eaccam, edispnp,
        edispn, egridv, egridvn, escaaf, emhist, echist, eporositynp);

    //************************************************auxiliary variables for computing the porosity

    double dphi_dp = 0.0;
    double dphi_dJ = 0.0;
    double dphi_dJdp = 0.0;
    double dphi_dJJ = 0.0;
    Base::porosity_ = 0.0;

    // compute scalar at n+alpha_F or n+1
    const double scalaraf = Base::funct_.dot(escaaf);
    params.set<double>("scalar", scalaraf);
    compute_porosity(params, Base::press_, volchange, *(iquad), Base::funct_, eporositynp,
        Base::porosity_, &dphi_dp, &dphi_dJ, &dphi_dJdp, &dphi_dJJ,
        nullptr,  // dphi_dpp not needed
        false);

    double refporositydot = Base::struct_mat_->ref_porosity_time_deriv();

    //---------------------------  dJ/dx = dJ/dF : dF/dx = JF^-T : dF/dx at gausspoint
    static Core::LinAlg::Matrix<nsd_, 1> gradJ(Core::LinAlg::Initialization::uninitialized);
    // spatial porosity gradient
    static Core::LinAlg::Matrix<nsd_, 1> grad_porosity(Core::LinAlg::Initialization::uninitialized);
    //--------------------- linearization of porosity w.r.t. structure displacements
    static Core::LinAlg::Matrix<1, nsd_ * nen_> dphi_dus(
        Core::LinAlg::Initialization::uninitialized);

    //------------------------------------------------dJ/dus = dJ/dF : dF/dus = J * F^-T . N_X = J *
    // N_x
    static Core::LinAlg::Matrix<1, nsd_ * nen_> dJ_dus(Core::LinAlg::Initialization::uninitialized);
    //------------------ d( grad(\phi) ) / du_s = d\phi/(dJ du_s) * dJ/dx+ d\phi/dJ * dJ/(dx*du_s) +
    // d\phi/(dp*du_s) * dp/dx
    static Core::LinAlg::Matrix<nsd_, nen_ * nsd_> dgradphi_dus(
        Core::LinAlg::Initialization::uninitialized);

    //------------------------------------ build F^-T as vector 9x1
    static Core::LinAlg::Matrix<nsd_ * nsd_, 1> defgrd_IT_vec(
        Core::LinAlg::Initialization::uninitialized);
    for (int i = 0; i < nsd_; i++)
      for (int j = 0; j < nsd_; j++) defgrd_IT_vec(i * nsd_ + j) = defgrd_inv(j, i);

    // dF/dx
    static Core::LinAlg::Matrix<nsd_ * nsd_, nsd_> F_x(Core::LinAlg::Initialization::uninitialized);

    // dF/dX
    static Core::LinAlg::Matrix<nsd_ * nsd_, nsd_> F_X(Core::LinAlg::Initialization::uninitialized);

    Base::compute_f_derivative(edispnp, defgrd_inv, F_x, F_X);

    // compute gradients if needed
    Base::compute_gradients(Base::J_, dphi_dp, dphi_dJ, defgrd_IT_vec, F_x, Base::gradp_,
        eporositynp, gradJ, Base::grad_porosity_, Base::refgrad_porosity_);

    compute_linearization_od(dphi_dJ, dphi_dJJ, dphi_dJdp, defgrd_inv, defgrd_IT_vec, F_x, F_X,
        gradJ, dJ_dus, dphi_dus, dgradphi_dus);

    //----------------------------------------------------------------------
    // potential evaluation of material parameters and/or stabilization
    // parameters at integration point
    //----------------------------------------------------------------------
    // get material parameters at integration point
    Base::get_material_parameters(material);

    // set viscous term from previous iteration to zero (required for
    // using routine for evaluation of momentum rhs/residual as given)
    Base::visc_old_.clear();
    Base::viscs2_.clear();
    // compute viscous term from previous iteration and viscous operator
    if (Base::is_higher_order_ele_) Base::calc_div_eps(evelaf);

    Base::compute_spatial_reaction_terms(material, defgrd_inv);

    // compute linearization of spatial reaction tensor w.r.t. structural displacements
    Base::compute_lin_spatial_reaction_terms(material, defgrd_inv, &dJ_dus, nullptr);

    // get stabilization parameters at integration point
    Base::compute_stabilization_parameters(vol);

    // compute old RHS of momentum equation and subgrid scale velocity
    Base::compute_old_rhs_and_subgrid_scale_velocity();

    // compute old RHS of continuity equation
    Base::compute_old_rhs_conti(dphi_dp);

    // compute strong residual of mixture (structural) equation
    if (Base::porofldpara_->stab_biot() and (not Base::porofldpara_->is_stationary_conti()) and
        Base::struct_mat_->poro_law_type() != Core::Materials::m_poro_law_constant)
      Base::compute_mixture_strong_residual(params, defgrd, edispnp, edispn, F_X, *iquad, true);

    //----------------------------------------------------------------------
    // set time-integration factors for left- and right-hand side
    //----------------------------------------------------------------------
    const double timefacfac = Base::fldparatimint_->time_fac() * Base::fac_;
    const double timefacfacpre = Base::fldparatimint_->time_fac_pre() * Base::fac_;

    //***********************************************************************************************
    // 1) coupling terms in momentum balance

    Base::fill_matrix_momentum_od(timefacfac, evelaf, egridv, epreaf, dgradphi_dus, dphi_dp,
        dphi_dJ, dphi_dus, refporositydot, lin_resM_Dus, lin_resM_Dus_gridvel, ecoupl_u);

    //*************************************************************************************************************
    // 2) coupling terms in continuity equation

    Base::fill_matrix_conti_od(timefacfacpre, dphi_dp, dphi_dJ, dphi_dJJ, dphi_dJdp, refporositydot,
        dgradphi_dus, dphi_dus, dJ_dus, egridv, lin_resM_Dus, lin_resM_Dus_gridvel, ecoupl_p);

    //*************************************************************************************************************
    // 3) additionale terms due to p1 approach (derivatives w.r.t. porosity)
    // 3.1) Momentum equation

    /*  reaction */
    /*
      /                           \
     |                             |
  -  |    sigma * v_f D(phi), v    |
     |                             |
      \                           /
     */
    {
      const double porosity_inv = 1.0 / Base::porosity_;
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          lin_resM_Dphi(idim, ui) +=
              timefacfac * porosity_inv * Base::reac_tensor_vel_(idim) * Base::funct_(ui);
        }
      }
    }


    if (not Base::porofldpara_->is_stationary_momentum())
    // transient terms
    /*  reaction  */
    /*
      /                           \
     |                             |
  -  |    sigma * v_s D(phi), v    |
     |                             |
      \                           /
     */
    {
      const double porosity_inv = 1.0 / Base::porosity_;
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          lin_resM_Dphi(idim, ui) +=
              timefacfac * porosity_inv * (-Base::reac_tensor_gridvel_(idim)) * Base::funct_(ui);
        }
      }
    }

    // viscous terms (brinkman terms)
    if (Base::visceff_)
    {
      static Core::LinAlg::Matrix<nsd_, nsd_> viscstress(
          Core::LinAlg::Initialization::uninitialized);
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          viscstress(idim, jdim) =
              Base::visceff_ * (Base::vderxy_(jdim, idim) + Base::vderxy_(idim, jdim));
        }
      }

      static Core::LinAlg::Matrix<nsd_, 1> viscstress_gradphi(
          Core::LinAlg::Initialization::uninitialized);
      viscstress_gradphi.multiply(viscstress, Base::grad_porosity_);

      static Core::LinAlg::Matrix<nsd_, nen_> viscstress_derxy(
          Core::LinAlg::Initialization::uninitialized);
      viscstress_derxy.multiply(viscstress, Base::derxy_);

      const double porosity_inv = 1.0 / Base::porosity_;

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          lin_resM_Dphi(idim, ui) += timefacfac * porosity_inv *
                                     (porosity_inv * viscstress_gradphi(idim) * Base::funct_(ui) -
                                         viscstress_derxy(idim, ui));
        }
      }
    }

    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = nsd_ * vi;
        for (int idim = 0; idim < nsd_; ++idim)
        {
          ecouplp1_u(fvi + idim, ui) += Base::funct_(vi) * lin_resM_Dphi(idim, ui);
        }
      }
    }

    //*************************************************************************************************************
    // 3.2) Continuity equation

    // transient terms
    /* time derivative*/
    /*
      /                           \
     |                             |
  -  |    D(phi), v                |
     |                             |
      \                           /
     */
    {
      for (int ui = 0; ui < nen_; ++ui)
        for (int vi = 0; vi < nen_; ++vi)
          ecouplp1_p(vi, ui) += Base::fac_ * Base::funct_(vi) * Base::funct_(ui);
    }

    static Core::LinAlg::Matrix<nen_, 1> derxy_convel(Core::LinAlg::Initialization::uninitialized);
    derxy_convel.clear();

    for (int i = 0; i < nen_; i++)
      for (int j = 0; j < nsd_; j++) derxy_convel(i) += Base::derxy_(j, i) * Base::velint_(j);

    if (not Base::porofldpara_->is_stationary_conti())
    {
      for (int i = 0; i < nen_; i++)
        for (int j = 0; j < nsd_; j++)
          derxy_convel(i) += Base::derxy_(j, i) * (-Base::gridvel_int_(j));
    }

    if (!static_cast<Discret::Elements::FluidEleParameterPoro*>(Base::fldpara_)
            ->poro_conti_part_int())
    {
      /*
        /                           \     /                             \
       |                             |    |                              |
       |    \nabla v_f D(phi), v     | +  |  (v_f-v_s) \nabla  D(phi), v |
       |                             |    |                              |
        \                           /     \                             /
       */
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          ecouplp1_p(vi, ui) += +timefacfacpre * Base::vdiv_ * Base::funct_(vi) * Base::funct_(ui) +
                                timefacfacpre * Base::funct_(vi) * derxy_convel(ui);
        }
      }
    }
    else
    {
      /*
          /                             \
          |                              |
       -  |  (v_f-v_s) \nabla  D(phi), v |
          |                              |
          \                             /
       */
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          ecouplp1_p(vi, ui) += -1.0 * timefacfacpre * derxy_convel(vi) * Base::funct_(ui);
        }
      }
      /*
          /                             \
          |                              |
          |  \nabla v_s D(phi), v        |
          |                              |
          \                             /
       */
      if (not Base::porofldpara_->is_stationary_conti())
      {
        for (int ui = 0; ui < nen_; ++ui)
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            ecouplp1_p(vi, ui) +=
                timefacfacpre * Base::funct_(vi) * Base::gridvel_div_ * Base::funct_(ui);
          }
        }
      }
    }

    //*************************************************************************************************************
    // PSPG
    if (Base::fldpara_->pspg())
    {
      double scal_grad_q = 0.0;

      if (Base::fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
      {
        scal_grad_q = Base::tau_(1);
      }
      else
      {
        scal_grad_q = 0.0;  // Base::fldpara_->AlphaF()*fac3;
      }

      {
        const double v1 = -timefacfacpre * Base::dtau_dphi_(1) / scal_grad_q;
        for (int ui = 0; ui < nen_; ++ui)
        {
          for (int idim = 0; idim < nsd_; ++idim)
          {
            const double v = v1 * Base::sgvelint_(idim) * Base::funct_(ui);

            for (int vi = 0; vi < nen_; ++vi)
            {
              ecouplp1_p(vi, ui) += v * Base::derxy_(idim, vi);
            }
          }
        }
      }

      // linearization of residual in stabilization term w.r.t. porosity
      if (Base::is_higher_order_ele_ || Base::fldpara_->is_newton())
      {
        static Core::LinAlg::Matrix<nen_, nen_> temp(Core::LinAlg::Initialization::uninitialized);
        temp.clear();

        for (int vi = 0; vi < nen_; ++vi)
        {
          for (int ui = 0; ui < nen_; ++ui)
            for (int idim = 0; idim < nsd_; ++idim)
              temp(vi, ui) += Base::derxy_(idim, vi) * lin_resM_Dphi(idim, ui);
        }

        for (int ui = 0; ui < nen_; ++ui)
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            ecouplp1_p(vi, ui) += scal_grad_q * temp(vi, ui);
          }
        }
      }
    }

    //*************************************************************************************************************
    if (Base::fldpara_->r_stab() != Inpar::FLUID::reactive_stab_none)
    {
      double reac_tau;
      if (Base::fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
        reac_tau = Base::fldpara_->visc_rea_stab_fac() * Base::reacoeff_ * Base::tau_(1);
      else
      {
        FOUR_C_THROW("Is this factor correct? Check for bugs!");
        reac_tau = 0.0;
      }

      if (Base::is_higher_order_ele_ or Base::fldpara_->is_newton())
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double v = reac_tau * Base::funct_(vi);

          for (int idim = 0; idim < nsd_; ++idim)
          {
            const int fvi_p_idim = nsd_ * vi + idim;

            for (int ui = 0; ui < nen_; ++ui)
            {
              ecouplp1_u(fvi_p_idim, ui) += v * lin_resM_Dphi(idim, ui);
            }
          }
        }
      }

      {  // linearization of stabilization parameter w.r.t. porosity
        const double v = timefacfac * Base::fldpara_->visc_rea_stab_fac() *
                         (Base::reacoeff_ * Base::dtau_dphi_(1) / Base::tau_(1) +
                             Base::reacoeff_ / Base::porosity_);
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double w = -1.0 * v * Base::funct_(vi);

          for (int idim = 0; idim < nsd_; ++idim)
          {
            const double w_sgvelint = w * Base::sgvelint_(idim);
            const int fvi = nsd_ * vi + idim;

            for (int ui = 0; ui < nen_; ++ui)
            {
              ecouplp1_u(fvi, ui) += w_sgvelint * Base::funct_(ui);
            }
          }
        }
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::compute_linearization(const double& dphi_dp,
    const double& dphi_dpp, const double& dphi_dJp, const Core::LinAlg::Matrix<nsd_, 1>& gradJ,
    Core::LinAlg::Matrix<nsd_, nen_>& dgradphi_dp)
{
  // porosity is a primary variable -> d(grad(phi)/d(pressure)) is zero!
  dgradphi_dp.clear();
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::compute_linearization_od(const double& dphi_dJ,
    const double& dphi_dJJ, const double& dphi_dJp,
    const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd_inv,
    const Core::LinAlg::Matrix<nsd_ * nsd_, 1>& defgrd_IT_vec,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_x,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nsd_>& F_X, const Core::LinAlg::Matrix<nsd_, 1>& gradJ,
    Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dus, Core::LinAlg::Matrix<1, nsd_ * nen_>& dphi_dus,
    Core::LinAlg::Matrix<nsd_, nen_ * nsd_>& dgradphi_dus)
{
  //------------------------------------------------dJ/dus = dJ/dF : dF/dus = J * F^-T . N_X = J *
  // N_x
  for (int i = 0; i < nen_; i++)
    for (int j = 0; j < nsd_; j++) dJ_dus(j + i * nsd_) = Base::J_ * Base::derxy_(j, i);

  //--------------------- linearization of porosity w.r.t. structure displacements
  // porosity is a primary variable -> d(grad(phi)/d(displacement)) is zero!
  dphi_dus.clear();

  //------------------ d( grad(\phi) ) / du_s is also zero
  dgradphi_dus.clear();

  if (!static_cast<Discret::Elements::FluidEleParameterPoro*>(Base::fldpara_)
          ->poro_conti_part_int() or
      Base::visceff_)
  {
    //---------------------d(gradJ)/dus =  dJ/dus * F^-T . : dF/dx + J * dF^-T/dus : dF/dx + J *
    // F^-T : N_X_x

    // dF^-T/dus : dF/dx = - (F^-1. dN/dx . u_s)^T  : dF/dx
    static Core::LinAlg::Matrix<nsd_, nsd_ * nen_> dFinvdus_dFdx(
        Core::LinAlg::Initialization::uninitialized);
    dFinvdus_dFdx.clear();
    for (int i = 0; i < nsd_; i++)
    {
      for (int n = 0; n < nen_; n++)
      {
        for (int j = 0; j < nsd_; j++)
        {
          const int gid = nsd_ * n + j;
          const double defgrd_inv_ij = defgrd_inv(i, j);
          for (int k = 0; k < nsd_; k++)
          {
            const double derxy_kn = Base::derxy_(k, n);
            for (int p = 0; p < nsd_; p++)
              dFinvdus_dFdx(p, gid) += -defgrd_inv_ij * derxy_kn * F_x(k * nsd_ + i, p);
          }
        }
      }
    }

    // F^-T : d(dF/dx)/dus =  F^-T : (N,XX * F^ -1 + dF/dX * F^-1 * N,x)
    static Core::LinAlg::Matrix<nsd_, nsd_ * nen_> FinvT_dFx_dus(
        Core::LinAlg::Initialization::uninitialized);
    FinvT_dFx_dus.clear();

    for (int n = 0; n < nen_; n++)
    {
      for (int j = 0; j < nsd_; j++)
      {
        const int gid = nsd_ * n + j;
        for (int p = 0; p < nsd_; p++)
        {
          double val = 0.0;
          const double derxy_p_n = Base::derxy_(p, n);
          for (int k = 0; k < nsd_; k++)
          {
            const double defgrd_inv_kj = defgrd_inv(k, j);
            const double defgrd_inv_kp = defgrd_inv(k, p);
            for (int i = 0; i < nsd_; i++)
            {
              val += defgrd_inv(i, j) * Base::N_XYZ2full_(i * nsd_ + k, n) * defgrd_inv_kp;
              for (int l = 0; l < nsd_; l++)
                val += -defgrd_inv(i, l) * F_X(i * nsd_ + l, k) * defgrd_inv_kj * derxy_p_n;
            }
          }
          FinvT_dFx_dus(p, gid) += val;
        }
      }
    }

    //----d(gradJ)/dus =  dJ/dus * F^-T . : dF/dx + J * dF^-T/dus : dF/dx + J * F^-T : N_X_x
    static Core::LinAlg::Matrix<1, nsd_> temp;
    temp.multiply_tn(defgrd_IT_vec, F_x);

    //----d(gradJ)/dus =  dJ/dus * F^-T . : dF/dx + J * dF^-T/dus : dF/dx + J * F^-T : N_X_x
    static Core::LinAlg::Matrix<nsd_, nen_ * nsd_> dgradJ_dus;

    dgradJ_dus.multiply_tn(temp, dJ_dus);

    dgradJ_dus.update(Base::J_, dFinvdus_dFdx, 1.0);

    dgradJ_dus.update(Base::J_, FinvT_dFx_dus, 1.0);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::pspg(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, nen_>& ppmat,
    Core::LinAlg::Matrix<nen_, 1>& preforce,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resMRea_Du,
    const Core::LinAlg::Matrix<nsd_, nen_>& lin_resM_Dp, const double& dphi_dp, const double& fac3,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac)
{
  Base::pspg(estif_q_u, ppmat, preforce, lin_resM_Du, lin_resMRea_Du, lin_resM_Dp, dphi_dp, fac3,
      timefacfac, timefacfacpre, rhsfac);
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcPoroP1<distype>::reac_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const Core::LinAlg::Matrix<nsd_, nen_>& lin_resM_Dp, const double& dphi_dp,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac, const double& fac3)
{
  Base::reac_stab(estif_u, estif_p_v, velforce, lin_resM_Du, lin_resM_Dp, dphi_dp, timefacfac,
      timefacfacpre, rhsfac, fac3);
}

template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcPoroP1<distype>::compute_volume(Teuchos::ParameterList& params,
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, Base::xyze_);
  // set element id
  Base::eid_ = ele->id();

  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> eporositynp(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &edispnp, &eporositynp, "dispnp");

  Core::LinAlg::Matrix<nsd_, nen_> egridvnp(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, lm, *Base::rotsymmpbc_, &egridvnp, nullptr, "gridv");

  // get new node positions of ALE mesh
  Base::xyze_ += edispnp;

  // integration loop
  for (Core::FE::GaussIntegration::iterator iquad = Base::intpoints_.begin();
      iquad != Base::intpoints_.end(); ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    Base::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    //-----------------------------------computing the porosity
    Base::porosity_ = Base::funct_.dot(eporositynp);

    // get structure velocity derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    Core::LinAlg::Matrix<nsd_, nsd_> gridvelderxy;
    gridvelderxy.multiply_nt(egridvnp, Base::derxy_);

    Base::gridvel_div_ = 0.0;
    for (int idim = 0; idim < nsd_; ++idim) Base::gridvel_div_ += gridvelderxy(idim, idim);

    elevec1(0) += Base::porosity_ * Base::fac_;
  }

  return 0;
}

template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::hex8>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::hex20>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::hex27>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::tet4>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::tet10>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::wedge6>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::wedge15>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::pyramid5>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::quad4>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::quad8>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::quad9>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::tri3>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::tri6>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::nurbs9>;
template class Discret::Elements::FluidEleCalcPoroP1<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
