// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_loma.hpp"

#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcLoma<distype>*
Discret::Elements::FluidEleCalcLoma<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidEleCalcLoma<distype>>(
            new Discret::Elements::FluidEleCalcLoma<distype>());
      });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcLoma<distype>::FluidEleCalcLoma()
    : Discret::Elements::FluidEleCalc<distype>::FluidEleCalc()
{
  // we use the standard parameter list here, since there are not any additional
  // loma-specific parameters required in this derived class
  my::fldpara_ = Discret::Elements::FluidEleParameterStd::instance();
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcLoma<distype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, bool offdiag)
{
  if (not offdiag)
    return my::evaluate(ele, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2,
        elevec3, my::intpoints_);
  else
    return evaluate_od(ele, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2,
        elevec3, my::intpoints_);
}


/*----------------------------------------------------------------------*
 * evaluation of off-diagonal matrix block for monolithic loma solver (2)
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcLoma<distype>::evaluate_od(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, const Core::FE::GaussIntegration& intpoints)
{
  // rotationally symmetric periodic bc's: do setup for current element
  my::rotsymmpbc_->setup(discretization, ele);

  // construct view
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, nen_> elemat_1(elemat1, true);

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
  my::body_force(ele, ebofoaf, eprescpgaf, escabofoaf);

  // ---------------------------------------------------------------------
  // get all general state vectors: velocity/pressure, scalar,
  // acceleration/scalar time derivative and history
  // velocity/pressure and scalar values are at time n+alpha_F/n+alpha_M
  // for generalized-alpha scheme and at time n+1/n for all other schemes
  // acceleration/scalar time derivative values are at time n+alpha_M for
  // generalized-alpha scheme and at time n+1 for all other schemes
  // ---------------------------------------------------------------------
  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F and n+alpha_M
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1
  // ost:         velocity/pressure at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &evelaf, &epreaf, "velaf");

  Core::LinAlg::Matrix<nsd_, nen_> evelam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epream(Core::LinAlg::Initialization::zero);
  if (my::fldpara_->physical_type() == Inpar::FLUID::weakly_compressible &&
      my::fldparatimint_->is_genalpha())
  {
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &evelam, &epream, "velam");
  }
  if (my::fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes &&
      my::fldparatimint_->is_genalpha())
  {
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &evelam, &epream, "velam");
  }

  Core::LinAlg::Matrix<nen_, 1> escaaf(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, nullptr, &escaaf, "scaaf");

  Core::LinAlg::Matrix<nsd_, nen_> emhist(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &emhist, nullptr, "hist");

  Core::LinAlg::Matrix<nsd_, nen_> eaccam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escadtam(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &eaccam, &escadtam, "accam");

  Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escaam(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &eveln, &escaam, "scaam");

  if (not my::fldparatimint_->is_genalpha()) eaccam.clear();

  // ---------------------------------------------------------------------
  // get additional state vectors for ALE case: grid displacement and vel.
  // ---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);

  if (ele->is_ale())
  {
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &edispnp, nullptr, "dispnp");
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &egridv, nullptr, "gridv");
  }

  // get node coordinates and number of elements per node
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, my::xyze_);

  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (my::isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
        discretization, ele, my::myknots_, my::weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  //----------------------------------------------------------------
  // prepare some dynamic Smagorinsky related stuff
  //----------------------------------------------------------------
  double CsDeltaSq = 0.0;
  double CiDeltaSq = 0.0;
  if (my::fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_CsDeltaSq =
        params.sublist("TURBULENCE MODEL")
            .get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_Cs_delta_sq");
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_CiDeltaSq =
        params.sublist("TURBULENCE MODEL")
            .get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_Ci_delta_sq");
    const int id = ele->lid();
    CsDeltaSq = ele_CsDeltaSq->local_values_as_span()[id];
    CiDeltaSq = ele_CiDeltaSq->local_values_as_span()[id];
  }

  // set element id
  my::eid_ = ele->id();
  // call inner evaluate (does not know about element or discretization object)
  int result = evaluate_od(params, ebofoaf, eprescpgaf, elemat_1, evelaf, epreaf, epream, escaaf,
      emhist, eaccam, escadtam, escabofoaf, eveln, escaam, edispnp, egridv, mat, ele->is_ale(),
      CsDeltaSq, CiDeltaSq, intpoints);

  return result;
}


/*----------------------------------------------------------------------*
 * evaluation of off-diagonal matrix block for monolithic loma solver (3)
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcLoma<distype>::evaluate_od(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, nen_>& elemat1,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nen_, 1>& epream, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nsd_, nen_>& eaccam,
    const Core::LinAlg::Matrix<nen_, 1>& escadtam, const Core::LinAlg::Matrix<nen_, 1>& escabofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nen_, 1>& escaam,
    const Core::LinAlg::Matrix<nsd_, nen_>& edispnp, const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    std::shared_ptr<Core::Mat::Material> mat, bool isale, double CsDeltaSq, double CiDeltaSq,
    const Core::FE::GaussIntegration& intpoints)
{
  // flag for higher order elements
  my::is_higher_order_ele_ = IsHigherOrder<distype>::ishigherorder;
  // overrule higher_order_ele if input-parameter is set
  // this might be interesting for fast (but slightly
  // less accurate) computations
  if (my::fldpara_->is_inconsistent() == true) my::is_higher_order_ele_ = false;

  // stationary formulation does not support ALE formulation
  if (isale and my::fldparatimint_->is_stationary())
    FOUR_C_THROW("No ALE support within stationary fluid solver.");

  // set thermodynamic pressure at n+1/n+alpha_F and n+alpha_M/n and
  // its time derivative at n+alpha_M/n+1
  const double thermpressaf = params.get<double>("thermpress at n+alpha_F/n+1");
  const double thermpressam = params.get<double>("thermpress at n+alpha_M/n");
  const double thermpressdtaf = params.get<double>("thermpressderiv at n+alpha_F/n+1");
  const double thermpressdtam = params.get<double>("thermpressderiv at n+alpha_M/n+1");

  // ---------------------------------------------------------------------
  // set parameters for classical turbulence models
  // ---------------------------------------------------------------------
  Teuchos::ParameterList& turbmodelparams = params.sublist("TURBULENCE MODEL");

  double Cs_delta_sq = 0.0;
  double Ci_delta_sq = 0.0;
  my::visceff_ = 0.0;

  // remember the layer of averaging for the dynamic Smagorinsky model
  int nlayer = 0;

  my::get_turbulence_params(
      turbmodelparams, Cs_delta_sq, Ci_delta_sq, nlayer, CsDeltaSq, CiDeltaSq);

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix
  // ---------------------------------------------------------------------
  sysmat_od(ebofoaf, eprescpgaf, evelaf, eveln, epreaf, epream, eaccam, escaaf, escaam, escadtam,
      escabofoaf, emhist, edispnp, egridv, elemat1, thermpressaf, thermpressam, thermpressdtaf,
      thermpressdtam, mat, Cs_delta_sq, Ci_delta_sq, isale, intpoints);

  return 0;
}


/*----------------------------------------------------------------------*
 |  calculate element matrix for off-diagonal matrix block              |
 |  for monolithic low-Mach-number solver                      vg 10/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcLoma<distype>::sysmat_od(
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epream,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nen_, 1>& escabofoaf, const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
    const Core::LinAlg::Matrix<nsd_, nen_>& edispnp, const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, nen_>& estif, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    std::shared_ptr<const Core::Mat::Material> material, double& Cs_delta_sq, double& Ci_delta_sq,
    bool isale, const Core::FE::GaussIntegration& intpoints)
{
  // definition of temperature-based residual vector for continuity
  // and energy-conservation equation
  Core::LinAlg::Matrix<nen_, 1> lin_resC_DT(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> lin_resE_DT(Core::LinAlg::Initialization::zero);

  // add displacement when fluid nodes move in the ALE case
  if (isale) my::xyze_ += edispnp;

  // evaluate shape functions and derivatives at element center
  my::eval_shape_func_and_derivs_at_ele_center();

  // set element area or volume
  const double vol = my::fac_;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // get material parameters at element center
  if (not my::fldpara_->mat_gp() or not my::fldpara_->tau_gp())
  {
    my::get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
        thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);

    // calculate all-scale subgrid viscosity at element center
    my::visceff_ = my::visc_;
    if (my::fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        my::fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        my::fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
    {
      my::calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
      // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
      my::visceff_ += my::sgvisc_;
    }
  }

  // calculate stabilization parameter at element center
  if (not my::fldpara_->tau_gp())
  {
    // get convective velocity at element center for evaluation of
    // stabilization parameter
    my::velint_.multiply(evelaf, my::funct_);
    my::convvelint_.update(my::velint_);
    if (isale) my::convvelint_.multiply(-1.0, egridv, my::funct_, 1.0);

    // calculate stabilization parameters at element center
    my::calc_stab_parameter(vol);
  }

  //------------------------------------------------------------------------
  //  start loop over integration points
  //------------------------------------------------------------------------
  for (Core::FE::GaussIntegration::const_iterator iquad = intpoints.begin();
      iquad != intpoints.end(); ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    my::eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    // get convective velocity at integration point
    // (including grid velocity in ALE case,
    // values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    my::convvelint_.multiply(evelaf, my::funct_);
    if (isale)
    {
      my::gridvelint_.multiply(egridv, my::funct_);
      my::convvelint_.update(-1.0, my::gridvelint_, 1.0);
    }

    //----------------------------------------------------------------------
    // potential evaluation of material parameters, subgrid viscosity
    // and/or stabilization parameters at integration point
    //----------------------------------------------------------------------
    // get material parameters at integration point
    if (my::fldpara_->mat_gp())
    {
      my::get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
          thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);
      // calculate all-scale or fine-scale subgrid viscosity at integration point
      my::visceff_ = my::visc_;
      if (my::fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      {
        my::calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
        // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
        my::visceff_ += my::sgvisc_;
      }
    }

    // calculate stabilization parameter at integration point
    if (my::fldpara_->tau_gp()) my::calc_stab_parameter(vol);

    // evaluation of convective operator
    my::conv_c_.multiply_tn(my::derxy_, my::convvelint_);

    // compute additional Galerkin terms on right-hand side of continuity equation
    // -> different for generalized-alpha and other time-integration schemes
    my::compute_gal_rhs_cont_eq(eveln, escaaf, escaam, escadtam, isale);

    // compute subgrid-scale part of scalar
    // -> different for generalized-alpha and other time-integration schemes
    my::compute_subgrid_scale_scalar(escaaf, escaam);

    // update material parameters including subgrid-scale part of scalar
    if (my::fldpara_->update_mat())
    {
      if (my::fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
        FOUR_C_THROW("No material update in combination with smagorinsky model!");
      my::update_material_params(material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
          thermpressam, my::sgscaint_);
      my::visceff_ = my::visc_;
      if (my::fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          my::fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
        my::visceff_ += my::sgvisc_;
    }
    //----------------------------------------------------------------------
    //  evaluate temperature-based residual vector for continuity equation
    //----------------------------------------------------------------------
    // set residual vector for continuity equation to zero
    lin_resC_DT.clear();

    // transient term
    if (not my::fldparatimint_->is_stationary())
    {
      const double scadtfacfac = my::scadtfac_ * my::fac_;
      for (int ui = 0; ui < nen_; ++ui)
      {
        lin_resC_DT(ui) += scadtfacfac * my::funct_(ui);
      }
    }

    // convective term
    const double timefac_scaconvfacaf =
        my::fldparatimint_->time_fac() * my::fac_ * my::scaconvfacaf_;
    for (int ui = 0; ui < nen_; ++ui)
    {
      lin_resC_DT(ui) += timefac_scaconvfacaf * my::conv_c_(ui);
    }

    //----------------------------------------------------------------------
    // subgrid-scale-velocity term (governed by cross-stress flag here)
    //----------------------------------------------------------------------
    if (my::fldpara_->conti_cross() == Inpar::FLUID::cross_stress_stab or
        my::fldpara_->conti_reynolds() == Inpar::FLUID::reynolds_stress_stab)
    {
      //----------------------------------------------------------------------
      //  evaluation of various values at integration point:
      //  1) velocity derivatives
      //  2) pressure (including derivatives)
      //  3) body-force vector
      //  4) "history" vector for momentum equation
      //----------------------------------------------------------------------
      // get velocity derivatives at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      my::vderxy_.multiply_nt(evelaf, my::derxy_);

      // get pressure gradient at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      my::gradp_.multiply(my::derxy_, epreaf);

      // get bodyforce at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      my::bodyforce_.multiply(ebofoaf, my::funct_);
      // get prescribed pressure gradient acting as body force
      // (required for turbulent channel flow)
      my::generalbodyforce_.multiply(eprescpgaf, my::funct_);

      // get momentum history data at integration point
      // (only required for one-step-theta and BDF2 time-integration schemes)
      my::histmom_.multiply(emhist, my::funct_);

      // convective term from previous iteration
      my::conv_old_.multiply(my::vderxy_, my::convvelint_);

      // compute viscous term from previous iteration
      if (my::is_higher_order_ele_)
        my::calc_div_eps(evelaf, eveln);
      else
        my::visc_old_.clear();

      // compute residual of momentum equation and subgrid-scale velocity
      // -> residual of momentum equation different for generalized-alpha
      //    and other time-integration schemes
      // -> no time-dependent subgrid scales considered here
      double fac1 = 0.0;
      double fac2 = 0.0;
      double fac3 = 0.0;
      double facMtau = 0.0;
      double* saccn = nullptr;
      double* sveln = nullptr;
      double* svelnp = nullptr;
      my::compute_subgrid_scale_velocity(
          eaccam, fac1, fac2, fac3, facMtau, *iquad, saccn, sveln, svelnp);

      if (my::fldpara_->conti_cross() == Inpar::FLUID::cross_stress_stab)
      {
        // evaluate subgrid-scale-velocity term
        for (int ui = 0; ui < nen_; ++ui)
        {
          lin_resC_DT(ui) += timefac_scaconvfacaf * my::sgconv_c_(ui);
        }
      }
    }

    //----------------------------------------------------------------------
    // computation of standard Galerkin contributions to element matrix:
    // transient and convective term (potentially incl. cross-stress term)
    //----------------------------------------------------------------------
    /*
            /                                        \
           |         1     / dT     /         \   \   |
       -   |    q , --- * | ---- + | u o nabla | T |  |
           |         T     \ dt     \         /   /   |
            \                                        /
    */
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int numdof_vi_p_nsd = my::numdofpernode_ * vi + nsd_;
      for (int ui = 0; ui < nen_; ++ui)
      {
        estif(numdof_vi_p_nsd, ui) -= my::funct_(vi) * lin_resC_DT(ui);
      }
    }

    //----------------------------------------------------------------------
    // computation of SUPG and contributions to element matrix
    // (potentially including Reynolds-stress term)
    //----------------------------------------------------------------------
    if (my::fldpara_->supg())
    {
      // weighting functions for SUPG term
      Core::LinAlg::Matrix<nen_, 1> supg_rey_weight;
      const double prefac = my::scaconvfacaf_ * my::tau_(0);
      for (int vi = 0; vi < nen_; ++vi)
      {
        supg_rey_weight(vi) = prefac * my::conv_c_(vi);
      }

      // weighting functions for Reynolds-stress term
      if (my::fldpara_->reynolds() == Inpar::FLUID::reynolds_stress_stab)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          supg_rey_weight(vi) += prefac * my::sgconv_c_(vi);
        }
      }

      //----------------------------------------------------------------------
      //  evaluate residual vector for energy-conservation equation
      //----------------------------------------------------------------------
      // set residual vector for energy-conservation equation to zero
      lin_resE_DT.clear();

      // transient term
      if (not my::fldparatimint_->is_stationary())
      {
        const double densamfac = my::fac_ * my::densam_;
        for (int ui = 0; ui < nen_; ++ui)
        {
          lin_resE_DT(ui) += densamfac * my::funct_(ui);
        }
      }

      // convective term
      const double denstimefac = my::fldparatimint_->time_fac() * my::fac_ * my::densaf_;
      for (int ui = 0; ui < nen_; ++ui)
      {
        lin_resE_DT(ui) += denstimefac * my::conv_c_(ui);
      }

      // diffusive term
      if (my::is_higher_order_ele_)
      {
        // compute second derivatives of shape functions
        Core::LinAlg::Matrix<nen_, 1> diff;
        diff.clear();
        // compute N,xx + N,yy + N,zz for each shape function
        for (int i = 0; i < nen_; ++i)
        {
          for (int j = 0; j < nsd_; ++j)
          {
            diff(i) += my::derxy2_(j, i);
          }
        }

        const double difftimefac = my::fldparatimint_->time_fac() * my::fac_ * my::diffus_;
        for (int ui = 0; ui < nen_; ++ui)
        {
          lin_resE_DT(ui) -= difftimefac * diff(ui);
        }
      }

      /*    SUPG/Reynolds-stress term
          /                                                                      \
         |   /         \            dDT          /          \                     |
     -   |  | u o nabla | q , rho * ---- + rho * | u o nabla | DT - diff * lap DT |
         |   \         /             dt          \           /                    |
          \                                                                      /
      */
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi_p_nsd = my::numdofpernode_ * vi + nsd_;
        for (int ui = 0; ui < nen_; ++ui)
        {
          estif(numdof_vi_p_nsd, ui) -= supg_rey_weight(vi) * lin_resE_DT(ui);
        }
      }
    }
  }
  //------------------------------------------------------------------------
  //  end loop over integration points
  //------------------------------------------------------------------------

  return;
}


// Ursula is responsible for this comment!
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::hex8>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::hex20>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::hex27>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::tet4>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::tet10>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::wedge6>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::wedge15>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::pyramid5>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::quad4>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::quad8>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::quad9>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::tri3>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::tri6>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::nurbs9>;
template class Discret::Elements::FluidEleCalcLoma<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
