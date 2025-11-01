// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_gder2.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_ele_tds.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_mat_carreauyasuda.hpp"
#include "4C_mat_fluid_linear_density_viscosity.hpp"
#include "4C_mat_fluid_murnaghantait.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_herschelbulkley.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_modpowerlaw.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * Constructor
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
Discret::Elements::FluidEleCalc<distype, enrtype>::FluidEleCalc()
    : rotsymmpbc_(nullptr),
      eid_(-1.0),
      is_higher_order_ele_(false),
      weights_(Core::LinAlg::Initialization::zero),
      myknots_(nsd_),
      intpoints_(distype),
      is_inflow_ele_(false),
      estif_u_(Core::LinAlg::Initialization::zero),
      estif_p_v_(Core::LinAlg::Initialization::zero),
      estif_q_u_(Core::LinAlg::Initialization::zero),
      ppmat_(Core::LinAlg::Initialization::zero),
      preforce_(Core::LinAlg::Initialization::zero),
      velforce_(Core::LinAlg::Initialization::zero),
      lin_resM_Du_(Core::LinAlg::Initialization::zero),
      resM_Du_(Core::LinAlg::Initialization::zero),
      ebofoaf_(Core::LinAlg::Initialization::zero),
      eprescpgaf_(Core::LinAlg::Initialization::zero),
      escabofoaf_(Core::LinAlg::Initialization::zero),
      ebofon_(Core::LinAlg::Initialization::zero),
      eprescpgn_(Core::LinAlg::Initialization::zero),
      escabofon_(Core::LinAlg::Initialization::zero),
      evelaf_(Core::LinAlg::Initialization::zero),
      epreaf_(Core::LinAlg::Initialization::zero),
      evelam_(Core::LinAlg::Initialization::zero),
      epream_(Core::LinAlg::Initialization::zero),
      evelnp_(Core::LinAlg::Initialization::zero),
      eprenp_(Core::LinAlg::Initialization::zero),
      eveln_(Core::LinAlg::Initialization::zero),
      epren_(Core::LinAlg::Initialization::zero),
      eaccam_(Core::LinAlg::Initialization::zero),
      escadtam_(Core::LinAlg::Initialization::zero),
      eveldtam_(Core::LinAlg::Initialization::zero),
      epredtam_(Core::LinAlg::Initialization::zero),
      escaaf_(Core::LinAlg::Initialization::zero),
      escaam_(Core::LinAlg::Initialization::zero),
      emhist_(Core::LinAlg::Initialization::zero),
      eporo_(Core::LinAlg::Initialization::zero),
      gradphiele_(Core::LinAlg::Initialization::zero),
      curvatureele_(Core::LinAlg::Initialization::zero),
      gradphielen_(Core::LinAlg::Initialization::zero),
      curvatureelen_(Core::LinAlg::Initialization::zero),
      gradphieletot_(Core::LinAlg::Initialization::zero),
      curvatureeletot_(Core::LinAlg::Initialization::zero),
      edispnp_(Core::LinAlg::Initialization::zero),
      egridv_(Core::LinAlg::Initialization::zero),
      fsevelaf_(Core::LinAlg::Initialization::zero),
      fsescaaf_(Core::LinAlg::Initialization::zero),
      evel_hat_(Core::LinAlg::Initialization::zero),
      ereynoldsstress_hat_(Core::LinAlg::Initialization::zero),
      xyze_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      deriv2_(Core::LinAlg::Initialization::zero),
      xjm_(Core::LinAlg::Initialization::zero),
      xji_(Core::LinAlg::Initialization::zero),
      vderxy_(Core::LinAlg::Initialization::zero),
      derxy_(Core::LinAlg::Initialization::zero),
      derxy2_(Core::LinAlg::Initialization::zero),
      bodyforce_(Core::LinAlg::Initialization::zero),
      dens_theta_(0.0),
      bodyforcen_(Core::LinAlg::Initialization::zero),
      conv_oldn_(Core::LinAlg::Initialization::zero),
      visc_oldn_(Core::LinAlg::Initialization::zero),
      gradpn_(Core::LinAlg::Initialization::zero),
      velintn_(Core::LinAlg::Initialization::zero),
      viscn_(0.0),
      conres_oldn_(0.0),
      generalbodyforcen_(Core::LinAlg::Initialization::zero),
      generalbodyforce_(Core::LinAlg::Initialization::zero),
      histmom_(Core::LinAlg::Initialization::zero),
      velint_(Core::LinAlg::Initialization::zero),
      sgvelint_(Core::LinAlg::Initialization::zero),
      gridvelint_(Core::LinAlg::Initialization::zero),
      gridvelintn_(Core::LinAlg::Initialization::zero),
      convvelint_(Core::LinAlg::Initialization::zero),
      eadvvel_(Core::LinAlg::Initialization::zero),
      accint_(Core::LinAlg::Initialization::zero),
      gradp_(Core::LinAlg::Initialization::zero),
      tau_(Core::LinAlg::Initialization::zero),
      viscs2_(Core::LinAlg::Initialization::zero),
      conv_c_(Core::LinAlg::Initialization::zero),
      sgconv_c_(Core::LinAlg::Initialization::zero),
      vdiv_(0.0),
      rhsmom_(Core::LinAlg::Initialization::zero),
      conv_old_(Core::LinAlg::Initialization::zero),
      visc_old_(Core::LinAlg::Initialization::zero),
      momres_old_(Core::LinAlg::Initialization::zero),
      conres_old_(true),
      xder2_(Core::LinAlg::Initialization::zero),
      vderiv_(Core::LinAlg::Initialization::zero),
      xsi_(Core::LinAlg::Initialization::zero),
      det_(0.0),
      fac_(0.0),
      visc_(0.0),
      visceff_(0.0),
      reacoeff_(0.0),
      gamma_(0.0),
      // LOMA-specific variables
      diffus_(0.0),
      rhscon_(0.0),
      densaf_(1.0),         // initialized to 1.0 (filled in Fluid::get_material_params)
      densam_(1.0),         // initialized to 1.0 (filled in Fluid::get_material_params)
      densn_(1.0),          // initialized to 1.0 (filled in Fluid::get_material_params)
      scadtfac_(0.0),       // initialized to 0.0 (filled in Fluid::get_material_params)
      scaconvfacaf_(0.0),   // initialized to 0.0 (filled in Fluid::get_material_params)
      scaconvfacn_(0.0),    // initialized to 0.0 (filled in Fluid::get_material_params)
      thermpressadd_(0.0),  // initialized to 0.0 (filled in Fluid::get_material_params)
      convvelintn_(Core::LinAlg::Initialization::zero),
      vderxyn_(Core::LinAlg::Initialization::zero),
      vdivn_(0.0),
      grad_scaaf_(Core::LinAlg::Initialization::zero),
      grad_scan_(Core::LinAlg::Initialization::zero),
      scaaf_(0.0),
      scan_(0.0),
      tder_sca_(0.0),
      conv_scaaf_(0.0),
      conv_scan_(0.0),
      scarhs_(0.0),
      sgscaint_(0.0),
      // weakly_compressible-specific variables
      preaf_(0.0),
      pream_(0.0),
      preconvfacaf_(0.0),  // initialized to 0.0 (filled in Fluid::get_material_params)
      tder_pre_(0.0),
      predtfac_(0.0),  // initialized to 0.0 (filled in Fluid::get_material_params)
      grad_preaf_(Core::LinAlg::Initialization::zero),
      conv_preaf_(0.0),
      // turbulence-specific variables
      fsvelint_(Core::LinAlg::Initialization::zero),
      mffsvelint_(Core::LinAlg::Initialization::zero),
      fsvderxy_(Core::LinAlg::Initialization::zero),
      mffsvderxy_(Core::LinAlg::Initialization::zero),
      mffsvdiv_(0.0),
      sgvisc_(0.0),
      fssgvisc_(0.0),
      q_sq_(0.0),
      mfssgscaint_(0.0),
      grad_fsscaaf_(Core::LinAlg::Initialization::zero),
      tds_(nullptr),
      evelafgrad_(Core::LinAlg::Initialization::zero),
      evelngrad_(Core::LinAlg::Initialization::zero)
{
  rotsymmpbc_ =
      std::make_shared<FLD::RotationallySymmetricPeriodicBC<distype, nsd_ + 1, enrtype>>();

  // pointer to class FluidEleParameter (access to the general parameter)
  fldparatimint_ = Discret::Elements::FluidEleParameterTimInt::instance();
  // initialize also general parameter list, also it will be overwritten in derived subclasses
  fldpara_ = Discret::Elements::FluidEleParameterStd::instance();

  // Nurbs
  isNurbs_ = Core::FE::is_nurbs<distype>;
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, bool offdiag)
{
  return evaluate(ele, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3,
      intpoints_, offdiag);
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, const Core::FE::GaussIntegration& intpoints,
    bool offdiag)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "FLD::FluidEleCalc::Evaluate" );

  // rotationally symmetric periodic bc's: do setup for current element
  rotsymmpbc_->setup(discretization, ele);

  // construct views
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> elemat_1(elemat1, true);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> elemat_2(elemat2, true);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1> elevec_1(elevec1, true);
  // elevec2 and elevec3 are currently not in use

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes,
  // with pressure gradient prescribed as body force included for turbulent
  // channel flow and with scatra body force included for variable-density flow
  // (evaluation at time n+alpha_F for generalized-alpha scheme,
  //  and at time n+1 otherwise)
  // ---------------------------------------------------------------------
  ebofoaf_.clear();
  eprescpgaf_.clear();
  escabofoaf_.clear();
  body_force(ele, ebofoaf_, eprescpgaf_, escabofoaf_);
  if (params.get("forcing", false))
  {
    static Core::LinAlg::Matrix<nsd_, nen_> interiorebofoaf;
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &interiorebofoaf, nullptr, "forcing");
    ebofoaf_.update(1.0, interiorebofoaf, 1.0);
  }

  ebofon_.clear();
  eprescpgn_.clear();
  escabofon_.clear();  // TODO: Used for #LOMA#, scatrabodyforce. Implement later
  // TODO: Provide "forcing" for #Turbulence#!
  if (fldparatimint_->is_new_ost_implementation())
  {
    body_force(ele, (fldparatimint_->time() - fldparatimint_->dt()), fldpara_->physical_type(),
        ebofon_, eprescpgn_, escabofon_);
  }  // end is_new_ost_implementation

  // if not available, the arrays for the subscale quantities have to be
  // resized and initialised to zero
  double* saccn = nullptr;
  double* sveln = nullptr;
  double* svelnp = nullptr;
  if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)
  {
    ele->activate_tds(intpoints.num_points(), nsd_, &saccn, &sveln, &svelnp);
    tds_ = ele->tds();  // store reference to the required tds element data
  }

  // add correction term to RHS of continuity equation in order to match
  // the approximated analytical solution of the channal_weakly_compressible problem
  // given in "New analytical solutions for weakly compressible Newtonian
  // Poiseuille flows with pressure-dependent viscosity"
  // Kostas D. Housiadas, Georgios C. Georgiou
  const Teuchos::ParameterList& fluidparams = Global::Problem::instance()->fluid_dynamic_params();
  int corrtermfuncnum = (fluidparams.get<int>("CORRTERMFUNCNO"));
  ecorrectionterm_.clear();
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes && corrtermfuncnum > 0)
  {
    correction_term(ele, ecorrectionterm_);
  }

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
  evelaf_.clear();
  epreaf_.clear();
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelaf_, &epreaf_, "velaf");

  evelam_.clear();
  epream_.clear();
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible &&
      fldparatimint_->is_genalpha())
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &evelam_, &epream_, "velam");
  }
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes &&
      fldparatimint_->is_genalpha())
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &evelam_, &epream_, "velam");
  }

  // np_genalpha: additional vector for velocity at time n+1
  evelnp_.clear();
  eprenp_.clear();
  if (fldparatimint_->is_genalpha_np())
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &evelnp_, &eprenp_, "velnp");

  eveln_.clear();
  epren_.clear();
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &eveln_, &epren_, "veln");

  eaccam_.clear();
  escadtam_.clear();
  extract_values_from_global_vector(
      discretization, lm, *rotsymmpbc_, &eaccam_, &escadtam_, "accam");

  // changing names for consistency
  eveldtam_.clear();
  epredtam_.clear();
  eveldtam_ = eaccam_;
  epredtam_ = escadtam_;

  escaaf_.clear();
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, nullptr, &escaaf_, "scaaf");

  escaam_.clear();
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, nullptr, &escaam_, "scaam");

  emhist_.clear();
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &emhist_, nullptr, "hist");

  if (fldpara_->is_reconstruct_der())
  {
    // extract gradient projection for consistent residual
    const std::shared_ptr<Core::LinAlg::MultiVector<double>> velafgrad =
        params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("velafgrad");
    Core::FE::extract_my_node_based_values(ele, evelafgrad_, *velafgrad, nsd_ * nsd_);
    if (fldparatimint_->is_new_ost_implementation())
    {
      const std::shared_ptr<Core::LinAlg::MultiVector<double>> velngrad =
          params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("velngrad");
      Core::FE::extract_my_node_based_values(ele, evelngrad_, *velngrad, nsd_ * nsd_);
    }
  }

  // clear vectors not required for respective time-integration scheme
  if (fldparatimint_->is_genalpha())
  {
    eveln_.clear();
    epren_.clear();
  }
  else
    eaccam_.clear();


  // set element advective field for Oseen problems
  if (fldpara_->physical_type() == Inpar::FLUID::oseen) set_advective_vel_oseen(ele);


  gradphiele_.clear();
  curvatureele_.clear();
  gradphielen_.clear();
  curvatureelen_.clear();

  gradphieletot_.clear();
  curvatureeletot_.clear();

  if (fldpara_->get_include_surface_tension())
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &gradphiele_, &curvatureele_, "tpf_gradphi_curvaf");

    if (fldparatimint_->is_new_ost_implementation())
    {
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, &gradphielen_, &curvatureelen_, "tpf_gradphi_curvn");
    }

    for (int i = 0; i < nen_; i++)
    {
      for (int idim = 0; idim < nsd_; idim++)
      {
        gradphieletot_(idim, i) = gradphiele_(idim, i);
        gradphieletot_(idim, i + nen_) = gradphielen_(idim, i);
      }
      curvatureeletot_(i, 0) = curvatureele_(i, 0);
      curvatureeletot_(i, 1) = curvatureelen_(i, 0);
    }

    double epsilon = fldpara_->get_interface_thickness();

    if (fldpara_->get_enhanced_gauss_rule_in_interface())
    {
      for (int i = 0; i < nen_; i++)
      {
        //      if(escaaf(i,1) < epsilon && escaaf(i,1) > -epsilon)
        if (abs(escaaf_(i, 0)) <= epsilon)
        {
          // Intpoints are changed. For sine and cosine, this new rule is utilized for error
          // computation compared to analytical solution.
          Core::FE::GaussIntegration intpoints_tmp(distype, ele->degree() * 2 + 3);
          intpoints_ = intpoints_tmp;
          break;
        }
        if (fldparatimint_->is_new_ost_implementation())  // To add enhanced Gaussrule to elements
                                                          // at time n as well.
        {
          for (int i = 0; i < nen_; i++)
          {
            if (abs(escaam_(i, 0)) <= epsilon)
            {
              // Intpoints are changed. For sine and cosine, this new rule is utilized for error
              // computation compared to analytical solution.
              Core::FE::GaussIntegration intpoints_tmp(distype, ele->degree() * 2 + 3);
              intpoints_ = intpoints_tmp;
              break;
            }
          }
        }  // fldparatimint_->is_new_ost_implementation()
      }
    }  // fldpara_->get_enhanced_gauss_rule_in_interface()
  }  // fldpara_->get_include_surface_tension()

  eporo_.clear();

  // ---------------------------------------------------------------------
  // get initial node coordinates for element
  // ---------------------------------------------------------------------
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // ---------------------------------------------------------------------
  // get additional state vectors for ALE case: grid displacement and vel.
  // ---------------------------------------------------------------------
  edispnp_.clear();
  egridv_.clear();

  if (ele->is_ale())
  {
    if (not fldparatimint_->is_new_ost_implementation())
    {
      get_grid_disp_vel_ale(discretization, lm, edispnp_, egridv_);
      egridvn_.clear();
    }
    else
      get_grid_disp_vel_aleost_new(discretization, lm, edispnp_, egridv_, egridvn_);
  }


  // ---------------------------------------------------------------------
  // get additional state vector for AVM3 case: fine-scale velocity
  // values are at time n+alpha_F for generalized-alpha scheme and at
  // time n+1 for all other schemes
  // ---------------------------------------------------------------------
  fsevelaf_.clear();
  fsescaaf_.clear();
  if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv or
      fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &fsevelaf_, nullptr, "fsvelaf");
    if (fldpara_->physical_type() == Inpar::FLUID::loma and
        fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, nullptr, &fsescaaf_, "fsscaaf");
  }


  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, ele, myknots_, weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  //----------------------------------------------------------------
  // prepare dynamic Smagorinsky model, if included
  //----------------------------------------------------------------
  double CsDeltaSq = 0.0;
  double CiDeltaSq = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
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
  // identify elements of inflow section
  inflow_element(ele);

  // set element id
  eid_ = ele->id();

  // call inner evaluate (does not know about element or discretization object)
  int result = evaluate(params, ebofoaf_, eprescpgaf_, ebofon_, eprescpgn_, elemat_1, elemat_2,
      elevec_1, evelaf_, epreaf_, evelam_, epream_, eprenp_, evelnp_, escaaf_, emhist_, eaccam_,
      escadtam_, eveldtam_, epredtam_, escabofoaf_, escabofon_, eveln_, epren_, escaam_, edispnp_,
      egridv_, egridvn_, fsevelaf_, fsescaaf_, evel_hat_, ereynoldsstress_hat_, eporo_,
      gradphieletot_,    // gradphiele,
      curvatureeletot_,  // curvatureele,
      mat, ele->is_ale(),
      ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()), CsDeltaSq,
      CiDeltaSq, saccn, sveln, svelnp, intpoints, offdiag);

  // rotate matrices and vectors if we have a rotationally symmetric problem
  rotsymmpbc_->rotate_matand_vec_if_necessary(elemat_1, elemat_2, elevec_1);

  return result;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::evaluate(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgn,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& elemat1,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& elemat2,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& elevec1,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelam, const Core::LinAlg::Matrix<nen_, 1>& epream,
    const Core::LinAlg::Matrix<nen_, 1>& eprenp, const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
    const Core::LinAlg::Matrix<nen_, 1>& escaaf, const Core::LinAlg::Matrix<nsd_, nen_>& emhist,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveldtam, const Core::LinAlg::Matrix<nen_, 1>& epredtam,
    const Core::LinAlg::Matrix<nen_, 1>& escabofoaf, const Core::LinAlg::Matrix<nen_, 1>& escabofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nen_, 1>& epren,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridv, const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const Core::LinAlg::Matrix<nen_, 1>& fsescaaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evel_hat,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& ereynoldsstress_hat,
    const Core::LinAlg::Matrix<nen_, 1>& eporo,
    const Core::LinAlg::Matrix<nsd_, 2 * nen_>& egradphi,
    const Core::LinAlg::Matrix<nen_, 2 * 1>& ecurvature, std::shared_ptr<Core::Mat::Material> mat,
    bool isale, bool isowned, double CsDeltaSq, double CiDeltaSq, double* saccn, double* sveln,
    double* svelnp, const Core::FE::GaussIntegration& intpoints, bool offdiag)
{
  if (offdiag) FOUR_C_THROW("No-off-diagonal matrix evaluation in standard fluid implementation!!");

  // flag for higher order elements
  is_higher_order_ele_ = IsHigherOrder<distype>::ishigherorder;
  // overrule higher_order_ele if input-parameter is set
  // this might be interesting for fast (but slightly
  // less accurate) computations
  if (fldpara_->is_inconsistent() == true) is_higher_order_ele_ = false;

  // stationary formulation does not support ALE formulation
  if (isale and fldparatimint_->is_stationary())
    FOUR_C_THROW("No ALE support within stationary fluid solver.");

  // set thermodynamic pressure at n+1/n+alpha_F and n+alpha_M/n and
  // its time derivative at n+alpha_M/n+1
  const double thermpressaf = params.get<double>("thermpress at n+alpha_F/n+1", 1.0);
  const double thermpressam = params.get<double>("thermpress at n+alpha_M/n", 1.0);
  const double thermpressdtaf = params.get<double>("thermpressderiv at n+alpha_F/n+1", 0.0);
  const double thermpressdtam = params.get<double>("thermpressderiv at n+alpha_M/n+1", 0.0);


  // ---------------------------------------------------------------------
  // set parameters for classical turbulence models
  // ---------------------------------------------------------------------
  Teuchos::ParameterList& turbmodelparams = params.sublist("TURBULENCE MODEL");

  double Cs_delta_sq = 0.0;
  double Ci_delta_sq = 0.0;
  double Cv = 0.0;
  visceff_ = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
    Cv = params.get<double>("C_vreman");


  // remember the layer of averaging for the dynamic Smagorinsky model
  int nlayer = 0;

  get_turbulence_params(turbmodelparams, Cs_delta_sq, Ci_delta_sq, nlayer, CsDeltaSq, CiDeltaSq);

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  if (not fldparatimint_->is_new_ost_implementation())
  {
    sysmat(ebofoaf, eprescpgaf, ebofon, eprescpgn, evelaf, evelam, eveln, evelnp, fsevelaf,
        fsescaaf, evel_hat, ereynoldsstress_hat, epreaf, epream, epren, eprenp, eaccam, escaaf,
        escaam, escadtam, eveldtam, epredtam, escabofoaf, escabofon, emhist, edispnp, egridv,
        elemat1,
        elemat2,  // -> emesh
        elevec1, eporo, egradphi, ecurvature, thermpressaf, thermpressam, thermpressdtaf,
        thermpressdtam, mat, Cs_delta_sq, Ci_delta_sq, Cv, isale, saccn, sveln, svelnp, intpoints);
  }
  else
  {
    sysmat_ost_new(ebofoaf, eprescpgaf, ebofon, eprescpgn, evelaf, evelam, eveln, evelnp, fsevelaf,
        fsescaaf, evel_hat, ereynoldsstress_hat, epreaf, epream, epren, eprenp, eaccam, escaaf,
        escaam, escadtam, escabofoaf, escabofon, emhist, edispnp, egridv, egridvn, elemat1,
        elemat2,  // -> emesh
        elevec1, eporo, egradphi, ecurvature, thermpressaf, thermpressam, thermpressdtaf,
        thermpressdtam, mat, Cs_delta_sq, Ci_delta_sq, Cv, isale, saccn, sveln, svelnp, intpoints);
  }

  // Fix if here. Sysmat with and without escabofon, ebofon, eprescpgn,

  // ---------------------------------------------------------------------
  // output values of Cs, visceff and Cs_delta_sq
  // ---------------------------------------------------------------------
  store_model_parameters_for_output(Cs_delta_sq, Ci_delta_sq, nlayer, isowned, turbmodelparams);

  return 0;
}

/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)   g.bau 03/07|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::sysmat(
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgn,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& evelam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const Core::LinAlg::Matrix<nen_, 1>& fsescaaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evel_hat,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& ereynoldsstress_hat,
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epream,
    const Core::LinAlg::Matrix<nen_, 1>& epren, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveldtam, const Core::LinAlg::Matrix<nen_, 1>& epredtam,
    const Core::LinAlg::Matrix<nen_, 1>& escabofoaf, const Core::LinAlg::Matrix<nen_, 1>& escabofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& estif,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce, const Core::LinAlg::Matrix<nen_, 1>& eporo,
    const Core::LinAlg::Matrix<nsd_, 2 * nen_>& egradphi,
    const Core::LinAlg::Matrix<nen_, 2 * 1>& ecurvature, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    std::shared_ptr<const Core::Mat::Material> material, double& Cs_delta_sq, double& Ci_delta_sq,
    double& Cv, bool isale, double* saccn, double* sveln, double* svelnp,
    const Core::FE::GaussIntegration& intpoints)
{
  //------------------------------------------------------------------------
  //  preliminary definitions and evaluations
  //------------------------------------------------------------------------
  // definition of matrices
  estif_u_.clear();
  estif_p_v_.clear();
  estif_q_u_.clear();
  ppmat_.clear();

  // definition of vectors
  preforce_.clear();
  velforce_.clear();

  // definition of velocity-based momentum residual vectors
  lin_resM_Du_.clear();
  resM_Du_.clear();

  // if polynomial pressure projection: reset variables
  if (fldpara_->ppp())
  {
    D_ = 0;
    E_.clear();
  }

  // evaluate shape functions and derivatives at element center
  eval_shape_func_and_derivs_at_ele_center();

  // set element area or volume
  const double vol = fac_;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // get material parameters at element center
  if (not fldpara_->mat_gp() or not fldpara_->tau_gp())
  {
    get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
        thermpressam, thermpressdtaf, thermpressdtam, vol);

    // calculate all-scale or fine-scale subgrid viscosity at element center
    visceff_ = visc_;

    if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::vreman or
        fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
    {
      if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
        Cs_delta_sq = Cv;  // use the declaration of Cs_delta_sq for the dynamic Vreman constant
      calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
      // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
      visceff_ += sgvisc_;
    }
    else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
      calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
  }

  // potential evaluation of multifractal subgrid-scales at element center
  // coefficient B of fine-scale velocity
  static Core::LinAlg::Matrix<nsd_, 1> B_mfs(Core::LinAlg::Initialization::zero);
  B_mfs.clear();

  // coefficient D of fine-scale scalar (loma only)
  double D_mfs = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    if (not fldpara_->b_gp())
    {
      // make sure to get material parameters at element center
      if (fldpara_->mat_gp())
        // get_material_params(material,evelaf,epreaf,epream,escaaf,escaam,thermpressaf,thermpressam,thermpressdtam,vol);
        get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
            thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);

      // provide necessary velocities and gradients at element center
      velint_.multiply(evelaf, funct_);
      fsvelint_.multiply(fsevelaf, funct_);
      vderxy_.multiply_nt(evelaf, derxy_);
      // calculate parameters of multifractal subgrid-scales and, finally,
      // calculate coefficient for multifractal modeling of subgrid velocity
      // if loma, calculate coefficient for multifractal modeling of subgrid scalar
      prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      // clear all velocities and gradients
      velint_.clear();
      fsvelint_.clear();
      vderxy_.clear();
    }
  }


  // calculate stabilization parameter at element center
  if (not fldpara_->tau_gp() and fldpara_->stab_type() == Inpar::FLUID::stabtype_residualbased)
  {
    // get convective velocity at element center
    // for evaluation of stabilization parameter
    velint_.multiply(evelaf, funct_);

    // get the grid velocity in case of ALE
    if (isale) gridvelint_.multiply(egridv, funct_);

    // get convective velocity at integration point
    set_convective_velint(isale);


    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)
    {
      // get velocity derivatives at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      vderxy_.multiply_nt(evelaf, derxy_);  // required for time-dependent subscales

      // compute velnp at integration point (required for time-dependent subscales)
      static Core::LinAlg::Matrix<nsd_, 1> velintnp(Core::LinAlg::Initialization::zero);
      velintnp.multiply(evelnp, funct_);
      vel_normnp_ = velintnp.norm2();
    }

    // calculate stabilization parameters at element center
    calc_stab_parameter(vol);
  }

  // get Gaussian integration points
  // const Core::FE::IntegrationPoints3D intpoints(ele->gaussrule_);
  // const Core::FE::IntPointsAndWeights<nsd_>
  // intpoints(Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  //------------------------------------------------------------------------
  //  start loop over integration points
  //------------------------------------------------------------------------
  // for (int iquad=0; iquad<intpoints.ip().nquad; ++iquad)

  for (Core::FE::GaussIntegration::const_iterator iquad = intpoints.begin();
      iquad != intpoints.end(); ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    //----------------------------------------------------------------------
    //  evaluation of various values at integration point:
    //  1) velocity (including derivatives, fine-scale and grid velocity)
    //  2) pressure (including derivatives)
    //  3) body-force vector
    //  4) "history" vector for momentum equation
    //----------------------------------------------------------------------
    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    velint_.multiply(evelaf, funct_);

    // get velocity derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    vderxy_.multiply_nt(evelaf, derxy_);

    // get fine-scale velocity and its derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvderxy_.clear();
    }
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      fsvelint_.multiply(fsevelaf, funct_);
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvelint_.clear();
    }

    // get the grid velocity in case of ALE
    if (isale) gridvelint_.multiply(egridv, funct_);

    // get convective velocity at integration point
    set_convective_velint(isale);


    // get pressure at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP scheme, n+1 otherwise)
    double press = 0.0;
    if (fldparatimint_->is_genalpha_np())
      press = funct_.dot(eprenp);
    else
      press = funct_.dot(epreaf);

    // get pressure gradient at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP scheme, n+1 otherwise)
    if (fldparatimint_->is_genalpha_np())
      gradp_.multiply(derxy_, eprenp);
    else
      gradp_.multiply(derxy_, epreaf);

    // get bodyforce at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    bodyforce_.multiply(ebofoaf, funct_);
    // get prescribed pressure gradient acting as body force
    // (required for turbulent channel flow)
    // If one wants to have SURF-tension only at ele-center. Then this might need to be revised.
    generalbodyforce_.multiply(eprescpgaf, funct_);

    // get momentum history data at integration point
    // (only required for one-step-theta and BDF2 time-integration schemes)
    histmom_.multiply(emhist, funct_);

    //----------------------------------------------------------------------
    // potential evaluation of material parameters, subgrid viscosity
    // and/or stabilization parameters at integration point
    //----------------------------------------------------------------------
    // get material parameters at integration point

    if (fldpara_->mat_gp())
    {
      get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
          thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);

      // calculate all-scale or fine-scale subgrid viscosity at integration point
      visceff_ = visc_;

      if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      {
        calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
        // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
        visceff_ += sgvisc_;
      }
      else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
        calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
    }

    // calculate stabilization parameter at integration point
    if (fldpara_->tau_gp() and fldpara_->stab_type() == Inpar::FLUID::stabtype_residualbased)
      calc_stab_parameter(vol);

    // potential evaluation of coefficient of multifractal subgrid-scales at integration point
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      if (fldpara_->b_gp())
      {
        // make sure to get material parameters at gauss point
        if (not fldpara_->mat_gp())
        {
          // get_material_params(material,evelaf,epreaf,epream,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam);
          // would overwrite materials at the element center, hence BGp() should always be combined
          // with MatGp()
          FOUR_C_THROW(
              "evaluation of B and D at gauss-point should always be combined with evaluation "
              "material at gauss-point!");
        }

        // calculate parameters of multifractal subgrid-scales
        prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      }

      // calculate fine-scale velocity, its derivative and divergence for multifractal subgrid-scale
      // modeling
      for (int idim = 0; idim < nsd_; idim++)
        mffsvelint_(idim, 0) = fsvelint_(idim, 0) * B_mfs(idim, 0);

      for (int idim = 0; idim < nsd_; idim++)
      {
        for (int jdim = 0; jdim < nsd_; jdim++)
          mffsvderxy_(idim, jdim) = fsvderxy_(idim, jdim) * B_mfs(idim, 0);
      }

      mffsvdiv_ = mffsvderxy_(0, 0) + mffsvderxy_(1, 1) + mffsvderxy_(2, 2);

      // only required for variable-density flow at low Mach number
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
      {
        if (isale) FOUR_C_THROW("Multifractal subgrid-scales with ale and loma not supported");
        mfssgscaint_ = D_mfs * funct_.dot(fsescaaf);
        grad_fsscaaf_.multiply(derxy_, fsescaaf);
        for (int dim = 0; dim < nsd_; dim++) grad_fsscaaf_(dim, 0) *= D_mfs;
      }
      else
      {
        mfssgscaint_ = 0.0;
        grad_fsscaaf_.clear();
      }
    }
    else
    {
      mffsvelint_.clear();
      mffsvderxy_.clear();
      mffsvdiv_ = 0.0;
    }

    // Adds surface tension force to the Gausspoint.
    // Note: has to be called after get_material_params(), otherwise gamma_ is uninitialized!!
    if (fldpara_->get_include_surface_tension())
      add_surface_tension_force(escaaf, escaam, egradphi, ecurvature);

    //----------------------------------------------------------------------
    //  evaluation of various partial operators at integration point
    //  1) convective term from previous iteration and convective operator
    //  2) viscous term from previous iteration and viscous operator
    //  3) divergence of velocity from previous iteration
    //----------------------------------------------------------------------

    // compute convective term from previous iteration and convective operator
    conv_old_.multiply(vderxy_, convvelint_);
    conv_c_.multiply_tn(derxy_, convvelint_);

    // compute viscous term from previous iteration and viscous operator
    if (is_higher_order_ele_)
      calc_div_eps(evelaf);
    else
    {
      visc_old_.clear();
      viscs2_.clear();
    }

    // compute divergence of velocity from previous iteration
    vdiv_ = 0.0;
    if (not fldparatimint_->is_genalpha_np())
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        vdiv_ += vderxy_(idim, idim);
      }
    }
    else
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        // get vdiv at time n+1 for np_genalpha,
        static Core::LinAlg::Matrix<nsd_, nsd_> vderxy(Core::LinAlg::Initialization::zero);
        vderxy.multiply_nt(evelnp, derxy_);
        vdiv_ += vderxy(idim, idim);
      }
    }

    //----------------------------------------------------------------------
    // set time-integration factors for left- and right-hand side
    //----------------------------------------------------------------------
    const double timefacfac = fldparatimint_->time_fac() * fac_;
    const double timefacfacpre = fldparatimint_->time_fac_pre() * fac_;
    const double rhsfac = fldparatimint_->time_fac_rhs() * fac_;

    //----------------------------------------------------------------------
    // computation of various subgrid-scale values and residuals
    //----------------------------------------------------------------------
    // compute residual of momentum equation and subgrid-scale velocity
    // -> residual of momentum equation different for generalized-alpha
    //    and other time-integration schemes
    double fac1 = 0.0;
    double fac2 = 0.0;
    double fac3 = 0.0;
    double facMtau = 0.0;
    compute_subgrid_scale_velocity(eaccam, fac1, fac2, fac3, facMtau, *iquad, saccn, sveln, svelnp);

    // compute residual of continuity equation
    // residual contains velocity divergence only for incompressible flow
    conres_old_ = vdiv_;

    // following computations only required for variable-density flow at low Mach number
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      compute_gal_rhs_cont_eq(eveln, escaaf, escaam, escadtam, isale);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;

      if (fldpara_->update_mat() or fldpara_->conti_supg() or
          fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none or
          fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none or
          fldpara_->multi_frac_loma_conti())
      {
        // compute subgrid-scale part of scalar
        // -> different for generalized-alpha and other time-integration schemes
        compute_subgrid_scale_scalar(escaaf, escaam);

        // update material parameters including subgrid-scale part of scalar
        if (fldpara_->update_mat())
        {
          // since we update the viscosity in the next step, a potential subgrid-scale velocity
          // would be overwritten
          if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
            FOUR_C_THROW("No material update in combination with smagorinsky model!");

          if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
            update_material_params(material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
                thermpressam, mfssgscaint_);
          else
            update_material_params(material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
                thermpressam, sgscaint_);
          visceff_ = visc_;
          if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
            visceff_ += sgvisc_;
        }

        // right-hand side of continuity equation based on updated material parameters
        // and including all stabilization terms
        // -> different for generalized-alpha and other time-integration schemes
        recompute_gal_and_compute_cross_rhs_cont_eq();
      }
    }
    else if (fldpara_->physical_type() == Inpar::FLUID::artcomp)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      compute_gal_rhs_cont_eq_art_comp(epreaf, epren, escadtam);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;
    }
    else if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
             fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
    {
      // update material parameters
      update_material_params(
          material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf, thermpressam, sgscaint_);

      // compute additional Galerkin terms on right-hand side of continuity equation
      compute_gal_rhs_cont_eq_weak_comp(epreaf, epredtam, isale);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;
    }


    // set velocity-based momentum residual vectors to zero
    lin_resM_Du_.clear();
    resM_Du_.clear();

    // compute first version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part),
    // reaction term and cross-stress term
    lin_gal_mom_res_u(lin_resM_Du_, timefacfac);

    // potentially rescale first version of velocity-based momentum residual
    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
        fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
    {
      lin_gal_mom_res_u_subscales(estif_p_v_, lin_resM_Du_, resM_Du_, timefacfac, facMtau);
    }

    //----------------------------------------------------------------------
    // computation of standard Galerkin and stabilization contributions to
    // element matrix and right-hand-side vector
    //----------------------------------------------------------------------
    // 1) standard Galerkin inertia, convection and reaction terms
    //    (convective and reactive part for convection term)
    //    as well as first part of cross-stress term on left-hand side
    inertia_convection_reaction_gal_part(estif_u_, velforce_, lin_resM_Du_, resM_Du_, rhsfac);

    // 2) standard Galerkin viscous term
    //    (including viscous stress computation,
    //     excluding viscous part for low-Mach-number flow)
    static Core::LinAlg::Matrix<nsd_, nsd_> viscstress(Core::LinAlg::Initialization::zero);
    viscstress.clear();
    viscous_gal_part(estif_u_, velforce_, viscstress, timefacfac, rhsfac);

    // 3) stabilization of continuity equation,
    //    standard Galerkin viscous part for low-Mach-number flow and
    //    right-hand-side part of standard Galerkin viscous term
    if (fldpara_->c_stab() or fldpara_->physical_type() == Inpar::FLUID::loma or
        fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
        fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
      cont_stab(estif_u_, velforce_, fldparatimint_->time_fac(), timefacfac, timefacfacpre, rhsfac);

    // 4) standard Galerkin pressure term
    pressure_gal_part(estif_p_v_, velforce_, timefacfac, timefacfacpre, rhsfac, press);

    // 5) standard Galerkin continuity term
    continuity_gal_part(estif_q_u_, preforce_, timefacfac, timefacfacpre, rhsfac);

    // 6) standard Galerkin bodyforce term on right-hand side
    body_force_rhs_term(velforce_, rhsfac);

    // 7) additional standard Galerkin terms due to conservative formulation
    if (fldpara_->is_conservative())
    {
      conservative_formulation(estif_u_, velforce_, timefacfac, rhsfac);
    }

    // 8) additional standard Galerkin terms for low-Mach-number flow and
    //    artificial compressibility (only right-hand side in latter case)
    if (fldpara_->physical_type() == Inpar::FLUID::loma or
        fldpara_->physical_type() == Inpar::FLUID::artcomp or
        fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
        fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
    {
      loma_gal_part(estif_q_u_, preforce_, timefacfac, rhsfac);
    }

    // 9) additional standard Galerkin term for temporal derivative of pressure
    //    in case of artificial compressibility (only left-hand side)
    if (fldpara_->physical_type() == Inpar::FLUID::artcomp and not fldparatimint_->is_stationary())
      art_comp_pressure_inertia_gal_partand_cont_stab(estif_p_v_, ppmat_);

    // 10) additional standard Galerkin term for temporal derivative of pressure
    //     in case of weakly_compressible flow
    if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
        fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
    {
      weak_comp_pressure_inertia_gal_part(estif_p_v_, ppmat_);
    }

    //----------------------------------------------------------------------
    // compute second version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part) and
    // viscous term
    //----------------------------------------------------------------------
    stab_lin_gal_mom_res_u(lin_resM_Du_, timefacfac);

    // 10) PSPG term
    if (fldpara_->pspg())
    {
      pspg(estif_q_u_, ppmat_, preforce_, lin_resM_Du_, fac3, timefacfac, timefacfacpre, rhsfac,
          *iquad);
    }

    // 11) SUPG term as well as first part of Reynolds-stress term on
    //     left-hand side and Reynolds-stress term on right-hand side
    if (fldpara_->supg())
    {
      supg(estif_u_, estif_p_v_, velforce_, preforce_, lin_resM_Du_, fac3, timefacfac,
          timefacfacpre, rhsfac);
    }

    // 12) reactive stabilization term
    if (fldpara_->r_stab() != Inpar::FLUID::reactive_stab_none)
    {
      reac_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // 13) viscous stabilization term
    if (is_higher_order_ele_ and (fldpara_->v_stab() != Inpar::FLUID::viscous_stab_none))
    {
      visc_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // if ConvDivStab for XFEM
    //    {
    //      ConvDivStab(estif_u,
    //           velforce,
    //           timefacfac,
    //           rhsfac);
    //    }


    // 14) cross-stress term: second part on left-hand side (only for Newton
    //     iteration) as well as cross-stress term on right-hand side
    if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none)
    {
      cross_stress_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // 15) Reynolds-stress term: second part on left-hand side
    //     (only for Newton iteration)
    if (fldpara_->reynolds() == Inpar::FLUID::reynolds_stress_stab and fldpara_->is_newton())
    {
      reynolds_stress_stab(estif_u_, estif_p_v_, lin_resM_Du_, timefacfac, timefacfacpre, fac3);
    }

    // 16) fine-scale subgrid-viscosity term
    //     (contribution only to right-hand-side vector)
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      const double fssgviscfac = fssgvisc_ * rhsfac;

      fine_scale_sub_grid_viscosity_term(velforce_, fssgviscfac);
    }

    // 17) subgrid-stress term (multifractal subgrid scales)
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      multfrac_sub_grid_scales_cross(estif_u_, velforce_, timefacfac, rhsfac);

      multfrac_sub_grid_scales_reynolds(estif_u_, velforce_, timefacfac, rhsfac);
    }

    // 18) polynomial pressure projection term (Dohrmann, Bochev IJNME 2004)
    //     (parameter-free inf-sub-stabilization, e.g. used instead of PSPG)
    if (fldpara_->ppp())
    {
      pressure_projection(ppmat_);
    }

    // linearization wrt mesh motion
    if (emesh.is_initialized())
    {
      if (nsd_ == 3)
        lin_mesh_motion_3d(emesh, evelaf, press, fldparatimint_->time_fac(), timefacfac);
      else if (nsd_ == 2)
        lin_mesh_motion_2d(emesh, evelaf, press, fldparatimint_->time_fac(), timefacfac);
      else
        FOUR_C_THROW("Linearization of the mesh motion is not available in 1D");
    }
  }
  //------------------------------------------------------------------------
  //  end loop over integration points
  //------------------------------------------------------------------------

  // if polynomial pressure projection: finalize matrices and rhs
  if (fldpara_->ppp())
  {
    if (fldparatimint_->is_genalpha_np())
      pressure_projection_finalize(ppmat_, preforce_, eprenp);
    else
      pressure_projection_finalize(ppmat_, preforce_, epreaf);
  }

  //------------------------------------------------------------------------
  //  add contributions to element matrix and right-hand-side vector
  //------------------------------------------------------------------------
  // add pressure part to right-hand-side vector
  for (int vi = 0; vi < nen_; ++vi)
  {
    eforce(numdofpernode_ * vi + nsd_) += preforce_(vi);
  }

  // add velocity part to right-hand-side vector
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      eforce(numdofpernode_ * vi + idim) += velforce_(idim, vi);
    }
  }

  // add pressure-pressure part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int fuippp = numdofpernode_ * ui + nsd_;

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int numdof_vi_p_nsd = numdofpernode_ * vi + nsd_;

      estif(numdof_vi_p_nsd, fuippp) += ppmat_(vi, ui);
    }
  }

  // add velocity-velocity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_ * ui;
    const int nsd_ui = nsd_ * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int numdof_ui_jdim = numdof_ui + jdim;
      const int nsd_ui_jdim = nsd_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = numdofpernode_ * vi;
        const int nsd_vi = nsd_ * vi;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif(numdof_vi + idim, numdof_ui_jdim) += estif_u_(nsd_vi + idim, nsd_ui_jdim);
        }
      }
    }
  }

  // add velocity-pressure part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui_nsd = numdofpernode_ * ui + nsd_;

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int nsd_vi = nsd_ * vi;
      const int numdof_vi = numdofpernode_ * vi;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif(numdof_vi + idim, numdof_ui_nsd) += estif_p_v_(nsd_vi + idim, ui);
      }
    }
  }

  // add pressure-velocity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_ * ui;
    const int nsd_ui = nsd_ * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int numdof_ui_jdim = numdof_ui + jdim;
      const int nsd_ui_jdim = nsd_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
        estif(numdofpernode_ * vi + nsd_, numdof_ui_jdim) += estif_q_u_(vi, nsd_ui_jdim);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  compute body force at element nodes (protected)            vg 10/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::body_force(Discret::Elements::Fluid* ele,
    Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf, Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    Core::LinAlg::Matrix<nen_, 1>& escabofoaf)
{
  body_force(
      ele, fldparatimint_->time(), fldpara_->physical_type(), ebofoaf, eprescpgaf, escabofoaf);
}



/*----------------------------------------------------------------------*
 |  compute body force at element nodes (public)               vg 10/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::body_force(Discret::Elements::Fluid* ele,
    const double time, const Inpar::FLUID::PhysicalType physicaltype,
    Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf, Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    Core::LinAlg::Matrix<nen_, 1>& escabofoaf)
{
  std::vector<const Core::Conditions::Condition*> myneumcond;

  // check whether all nodes have a unique Neumann condition
  if (nsd_ == 3)
    Core::Conditions::find_element_conditions(ele, "VolumeNeumann", myneumcond);
  else if (nsd_ == 2)
    Core::Conditions::find_element_conditions(ele, "SurfaceNeumann", myneumcond);
  else
    FOUR_C_THROW("Body force for 1D problem not yet implemented!");

  if (myneumcond.size() > 1) FOUR_C_THROW("More than one Neumann condition on one node!");

  if (myneumcond.size() == 1)
  {
    const auto condtype = myneumcond[0]->parameters().get<std::string>("TYPE");

    // get values and switches from the condition
    const auto onoff = myneumcond[0]->parameters().get<std::vector<int>>("ONOFF");
    const auto val = myneumcond[0]->parameters().get<std::vector<double>>("VAL");
    const auto functions =
        myneumcond[0]->parameters().get<std::vector<std::optional<int>>>("FUNCT");

    // factor given by spatial function
    double functionfac = 1.0;

    // set this condition to the ebofoaf array
    for (int isd = 0; isd < nsd_; isd++)
    {
      double num = onoff[isd] * val[isd];

      if (enrtype == Discret::Elements::Fluid::xwall)
      {
        // for xwall, only the linear shape functions contribute to the bodyforce,
        // since the sum of all shape functions sum(N_j) != 1.0 if the enriched shape functions are
        // included
        for (int jnode = 0; jnode < nen_; jnode += 2)
        {
          if (functions[isd].has_value() && functions[isd].value() > 0)
          {
            // evaluate function at the position of the current node if it is not none
            // ------------------------------------------------------
            // comment: this introduces an additional error compared to an
            // evaluation at the integration point. However, we need a node
            // based element bodyforce vector for prescribed pressure gradients
            // in some fancy turbulance stuff.
            functionfac =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[isd].value())
                    .evaluate((ele->nodes()[jnode])->x(), time, isd);
          }
          else
            functionfac = 1.0;

          // get usual body force
          if (condtype == "Dead" or condtype == "Live") ebofoaf(isd, jnode) = num * functionfac;
          // get prescribed pressure gradient
          else if (condtype == "PressureGrad")
            eprescpgaf(isd, jnode) = num * functionfac;
          else
            FOUR_C_THROW("Unknown Neumann condition");
        }
      }
      else
        for (int jnode = 0; jnode < nen_; ++jnode)
        {
          if (functions[isd].has_value() && functions[isd].value() > 0)
          {
            // evaluate function at the position of the current node
            // ------------------------------------------------------
            // comment: this introduces an additional error compared to an
            // evaluation at the integration point. However, we need a node
            // based element bodyforce vector for prescribed pressure gradients
            // in some fancy turbulance stuff.
            functionfac =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[isd].value())
                    .evaluate((ele->nodes()[jnode])->x(), time, isd);
          }
          else
            functionfac = 1.0;

          // get usual body force
          if (condtype == "Dead" or condtype == "Live") ebofoaf(isd, jnode) = num * functionfac;
          // get prescribed pressure gradient
          else if (condtype == "PressureGrad")
            eprescpgaf(isd, jnode) = num * functionfac;
          else
            FOUR_C_THROW("Unknown Neumann condition");
        }
    }
  }

  // get nodal values of scatra bodyforce for variable-density flow
  // at low Mach number
  if (physicaltype == Inpar::FLUID::loma)
  {
    std::vector<const Core::Conditions::Condition*> myscatraneumcond;

    // check whether all nodes have a unique Neumann condition
    if (nsd_ == 3)
      Core::Conditions::find_element_conditions(ele, "TransportVolumeNeumann", myscatraneumcond);
    else if (nsd_ == 2)
      Core::Conditions::find_element_conditions(ele, "TransportSurfaceNeumann", myscatraneumcond);
    else
      FOUR_C_THROW("Body force for 1D problem not yet implemented!");

    if (myscatraneumcond.size() > 1) FOUR_C_THROW("More than one Neumann condition on one node!");

    if (myscatraneumcond.size() == 1)
    {
      // check for potential time curve
      const auto funct =
          myscatraneumcond[0]->parameters().get<std::vector<std::optional<int>>>("FUNCT");

      // initialization of time-curve factor
      double functfac = 0.0;

      // compute potential time curve or set time-curve factor to one
      if (funct[0].has_value() && funct[0] > 0)
      {
        // time factor (negative time indicating error)
        if (time >= 0.0)
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfTime>(funct[0].value())
                         .evaluate(time);
        else
          FOUR_C_THROW("Negative time in bodyforce calculation: time = {}", time);
      }
      else
        functfac = 1.0;

      // get values and switches from the condition
      const auto onoff = myscatraneumcond[0]->parameters().get<std::vector<int>>("ONOFF");
      const auto val = myscatraneumcond[0]->parameters().get<std::vector<double>>("VAL");

      // set this condition to the bodyforce array
      for (int jnode = 0; jnode < nen_; jnode++)
      {
        escabofoaf(jnode) = onoff[0] * val[0] * functfac;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  compute correction term at element nodes                            |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::correction_term(
    Discret::Elements::Fluid* ele, Core::LinAlg::Matrix<1, nen_>& ecorrectionterm)
{
  // fill the element correction term
  const Teuchos::ParameterList& fluidparams = Global::Problem::instance()->fluid_dynamic_params();
  int functnum = (fluidparams.get<int>("CORRTERMFUNCNO"));
  if (functnum < 0) FOUR_C_THROW("Please provide a correct function number");
  for (int i = 0; i < nen_; ++i)
  {
    ecorrectionterm(i) = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum)
                             .evaluate((ele->nodes()[i])->x(), 0.0, 0);
  }
}

/*----------------------------------------------------------------------*
 |  compute surface tension force                              mw 05/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::add_surface_tension_force(
    const Core::LinAlg::Matrix<nen_, 1>& escaaf, const Core::LinAlg::Matrix<nen_, 1>& escaam,
    const Core::LinAlg::Matrix<nsd_, 2 * nen_>& egradphitot,
    const Core::LinAlg::Matrix<nen_, 2 * 1>& ecurvaturetot)
{
  double gaussescaaf;
  gaussescaaf = funct_.dot(escaaf);

  double epsilon = fldpara_->get_interface_thickness();  // Thickness in one direction.

  // Add surface force if inside interface thickness, otherwise do not.
  if (abs(gaussescaaf) <= epsilon)
  {
    static Core::LinAlg::Matrix<nsd_, nen_> egradphi(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<nsd_, nen_> egradphin(Core::LinAlg::Initialization::zero);

    static Core::LinAlg::Matrix<nen_, 1> ecurvature(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<nen_, 1> ecurvaturen(Core::LinAlg::Initialization::zero);

    // Extract values from gradient and curvature vectors (who been compressed to not use too many
    // unnecessary variables)
    //==================================================
    for (int i = 0; i < nen_; i++)
    {
      for (int idim = 0; idim < nsd_; idim++)
      {
        egradphi(idim, i) = egradphitot(idim, i);
        egradphin(idim, i) = egradphitot(idim, i + nen_);
      }
      ecurvature(i, 0) = ecurvaturetot(i, 0);
      ecurvaturen(i, 0) = ecurvaturetot(i, 1);
    }
    //==================================================

    static Core::LinAlg::Matrix<nsd_, 1> gradphi;

    // NON-smoothed gradient!!! Should be correct
    gradphi.multiply(derxy_, escaaf);

    if (fldparatimint_->is_new_ost_implementation())
    {
      static Core::LinAlg::Matrix<nsd_, 1> gradphin;
      gradphin.multiply(derxy_, escaam);
    }
  }


  return;
}

/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives at element center  vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::eval_shape_func_and_derivs_at_ele_center()
{
  // use one-point Gauss rule
  Core::FE::IntPointsAndWeights<nsd_> intpoints_stab(
      Discret::Elements::DisTypeToStabGaussRule<distype>::rule);

  eval_shape_func_and_derivs_at_int_point(
      (intpoints_stab.ip().qxg)[0], intpoints_stab.ip().qwgt[0]);

  return;
}


/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives at integr. point   vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::eval_shape_func_and_derivs_at_int_point(
    const double* gpcoord,  // actual integration point (coords)
    double gpweight         // actual integration point (weight)
)
{
  for (int idim = 0; idim < nsd_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  if (not isNurbs_)
  {
    // shape functions and their first derivatives
    Core::FE::shape_function<distype>(xsi_, funct_);
    Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);
    derxy2_.clear();
    if (is_higher_order_ele_)
    {
      // get the second derivatives of standard element at current GP
      Core::FE::shape_function_deriv2<distype>(xsi_, deriv2_);
    }
  }
  else
  {
    if (is_higher_order_ele_)
      Core::FE::Nurbs::nurbs_get_funct_deriv_deriv2(
          funct_, deriv_, deriv2_, xsi_, myknots_, weights_, distype);
    else
      Core::FE::Nurbs::nurbs_get_funct_deriv(funct_, deriv_, xsi_, myknots_, weights_, distype);
  }

  //

  // get Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
  */
  xjm_.multiply_nt(deriv_, xyze_);
  det_ = xji_.invert(xjm_);

  if (det_ < 1E-16)
    FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", eid_, det_);

  // compute integration factor
  fac_ = gpweight * det_;

  // compute global first derivatives
  derxy_.multiply(xji_, deriv_);

  //--------------------------------------------------------------
  //             compute global second derivatives
  //--------------------------------------------------------------
  if (is_higher_order_ele_)
  {
    Core::FE::gder2<distype, nen_>(xjm_, derxy_, deriv2_, xyze_, derxy2_);
  }
  else
    derxy2_.clear();

  return;
}

/*---------------------------------------------------------------------------*
 | get ALE grid displacements and grid velocity for element     schott 11/14 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_grid_disp_vel_ale(
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::Matrix<nsd_, nen_>& edispnp, Core::LinAlg::Matrix<nsd_, nen_>& egridv)
{
  switch (fldpara_->physical_type())
  {
    case Inpar::FLUID::oseen:
    case Inpar::FLUID::stokes:
    {
      FOUR_C_THROW(
          "ALE with Oseen or Stokes seems to be a tricky combination. Think deep before removing "
          "FOUR_C_THROW!");
      break;
    }
    default:
    {
      get_grid_disp_ale(discretization, lm, edispnp);
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, &egridv, nullptr, "gridv");
      break;
    }
  }
}

/*---------------------------------------------------------------------------*
 | get ALE grid displacements only for element                      bk 02/15 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_grid_disp_ale(
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::Matrix<nsd_, nen_>& edispnp)
{
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

  // add displacement when fluid nodes move in the ALE case
  xyze_ += edispnp;
}

/*---------------------------------------------------------------------------*
 |  set the (relative) convective velocity at integration point schott 11/14 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::set_convective_velint(const bool isale)
{
  // get convective velocity at integration point
  switch (fldpara_->physical_type())
  {
    case Inpar::FLUID::incompressible:
    case Inpar::FLUID::weakly_compressible:
    case Inpar::FLUID::artcomp:
    case Inpar::FLUID::varying_density:
    case Inpar::FLUID::loma:
    case Inpar::FLUID::tempdepwater:
    case Inpar::FLUID::boussinesq:
    {
      convvelint_.update(velint_);
      break;
    }
    case Inpar::FLUID::oseen:
    {
      convvelint_.multiply(eadvvel_, funct_);
      break;
    }
    case Inpar::FLUID::stokes:
    case Inpar::FLUID::weakly_compressible_stokes:
    {
      convvelint_.clear();
      break;
    }
    default:
      FOUR_C_THROW(
          "Physical type not implemented here. For Poro-problems see derived class "
          "FluidEleCalcPoro.");
      break;
  }

  // (ALE case handled implicitly here using the (potential
  //  mesh-movement-dependent) convective velocity, avoiding
  //  various ALE terms used to be calculated before)
  if (isale)
  {
    convvelint_.update(-1.0, gridvelint_, 1.0);
  }
}


/*---------------------------------------------------------------------------*
 |  set element advective field for Oseen problems              schott 11/14 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::set_advective_vel_oseen(
    Discret::Elements::Fluid* ele)
{
  // set element advective field for Oseen problems
  if (fldpara_->physical_type() == Inpar::FLUID::oseen)
  {
    const int funcnum = fldpara_->oseen_field_func_no();
    const double time = fldparatimint_->time();
    for (int jnode = 0; jnode < nen_; ++jnode)
    {
      const auto jx = ele->nodes()[jnode]->x();
      for (int idim = 0; idim < nsd_; ++idim)
        eadvvel_(idim, jnode) = Global::Problem::instance()
                                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
                                    .evaluate(jx, time, idim);
    }
  }
}


///*----------------------------------------------------------------------*
// |  compute material parameters                                vg 09/09 |
// *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_material_params(
    std::shared_ptr<const Core::Mat::Material> material,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nen_, 1>& epream, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escabofoaf,
    const double thermpressaf, const double thermpressam, const double thermpressdtaf,
    const double thermpressdtam, const double vol)
{
  get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
      thermpressam, thermpressdtaf, thermpressdtam, vol, densam_, densaf_, densn_, visc_, viscn_,
      gamma_);
}


/*----------------------------------------------------------------------*
 |  compute material parameters                                vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_material_params(
    std::shared_ptr<const Core::Mat::Material> material,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nen_, 1>& epream, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escabofoaf,
    const double thermpressaf, const double thermpressam, const double thermpressdtaf,
    const double thermpressdtam, const double vol, double& densam, double& densaf, double& densn,
    double& visc, double& viscn, double& gamma)
{
  // initially set density values and values with respect to continuity rhs
  densam = 1.0;
  densaf = 1.0;
  densn = 1.0;
  scadtfac_ = 0.0;
  scaconvfacaf_ = 0.0;
  scaconvfacn_ = 0.0;
  thermpressadd_ = 0.0;
  preconvfacaf_ = 0.0;
  predtfac_ = 0.0;

  if (material->material_type() == Core::Materials::m_fluid)
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(material.get());

    // get constant dynamic viscosity
    visc = actmat->viscosity();
    viscn = visc;

    // artificial compressibility
    if (fldpara_->physical_type() == Inpar::FLUID::artcomp)
    {
      // get norm of convective velocity
      const double vel_norm = convvelint_.norm2();

      // calculate characteristic element length
      double h_u = 0.0;
      double h_p = 0.0;
      calc_char_ele_length(vol, vel_norm, h_u, h_p);

      // get constant density
      densaf = actmat->density();
      densam = densaf;
      densn = densaf;

      // compute compressibility parameter c as squared value of
      // maximum of convective velocity, viscous velocity and constant
      // value 1/2, as proposed in Nithiarasu (2003)
      double maxvel = std::max(vel_norm, (visc / (h_p * densaf)));
      maxvel = std::max(maxvel, 0.5);
      scadtfac_ = 1.0 / std::pow(maxvel, 2);
    }
    // varying Density
    else if (fldpara_->physical_type() == Inpar::FLUID::varying_density)
    {
      const double density_0 = actmat->density();

      densaf = funct_.dot(escaaf) * density_0;
      densam = densaf;
      densn = funct_.dot(escaam) * density_0;
    }
    // Boussinesq approximation: Calculation of delta rho
    else if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
    {
      const double density_0 = actmat->density();

      if (escaaf(0) < 1e-12) FOUR_C_THROW("Boussinesq approximation: density in escaaf is zero");
      densaf = density_0;
      densam = densaf;
      densn = densaf;

      deltadens_ = (funct_.dot(escaaf) - 1.0) * density_0;
      // division by density_0 was removed here since we keep the density in all
      // terms of the momentum equation (no division by rho -> using dynamic viscosity)
    }
    // incompressible flow (standard case)
    else
    {
      densaf = actmat->density();
      densam = densaf;
      densn = densaf;
    }

    gamma = actmat->gamma();
  }
  else if (material->material_type() == Core::Materials::m_carreauyasuda)
  {
    const Mat::CarreauYasuda* actmat = static_cast<const Mat::CarreauYasuda*>(material.get());

    densaf = actmat->density();
    densam = densaf;
    densn = densaf;

    double nu_0 = actmat->nu0();       // parameter for zero-shear viscosity
    double nu_inf = actmat->nu_inf();  // parameter for infinite-shear viscosity
    double lambda = actmat->lambda();  // parameter for characteristic time
    double a = actmat->a_param();      // constant parameter
    double b = actmat->b_param();      // constant parameter

    // compute rate of strain at n+alpha_F or n+1
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(evelaf);

    // compute viscosity according to the Carreau-Yasuda model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    const double tmp = std::pow(lambda * rateofstrain, b);
    // kinematic viscosity
    visc = nu_inf + ((nu_0 - nu_inf) / pow((1 + tmp), a));
    // dynamic viscosity
    visc *= densaf;
  }
  else if (material->material_type() == Core::Materials::m_modpowerlaw)
  {
    const Mat::ModPowerLaw* actmat = static_cast<const Mat::ModPowerLaw*>(material.get());

    densaf = actmat->density();
    densam = densaf;
    densn = densaf;

    // get material parameters
    double m = actmat->m_cons();     // consistency constant
    double delta = actmat->delta();  // safety factor
    double a = actmat->a_exp();      // exponent

    // compute rate of strain at n+alpha_F or n+1
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(evelaf);

    // compute viscosity according to a modified power law model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    // kinematic viscosity
    visc = m * pow((delta + rateofstrain), (-1) * a);
    // dynamic viscosity
    visc *= densaf;
  }
  else if (material->material_type() == Core::Materials::m_herschelbulkley)
  {
    const Mat::HerschelBulkley* actmat = static_cast<const Mat::HerschelBulkley*>(material.get());

    densaf = actmat->density();
    densam = densaf;
    densn = densaf;

    double tau0 = actmat->tau0();                         // yield stress
    double kfac = actmat->k_fac();                        // constant factor
    double nexp = actmat->n_exp();                        // exponent
    double mexp = actmat->m_exp();                        // exponent
    double uplimshearrate = actmat->up_lim_shear_rate();  // upper limit of shear rate
    double lolimshearrate = actmat->lo_lim_shear_rate();  // lower limit of shear rate

    // compute rate of strain at n+alpha_F or n+1
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(evelaf);

    // calculate dynamic viscosity according to Herschel-Bulkley model
    // (within lower and upper limit of shear rate)
    if (rateofstrain < lolimshearrate)
      visc = tau0 * ((1.0 - exp(-mexp * lolimshearrate)) / lolimshearrate) +
             kfac * pow(lolimshearrate, (nexp - 1.0));
    else if (rateofstrain > uplimshearrate)
      visc = tau0 * ((1.0 - exp(-mexp * uplimshearrate)) / uplimshearrate) +
             kfac * pow(uplimshearrate, (nexp - 1.0));
    else
      visc = tau0 * ((1.0 - exp(-mexp * rateofstrain)) / rateofstrain) +
             kfac * pow(rateofstrain, (nexp - 1.0));
  }
  else if (material->material_type() == Core::Materials::m_sutherland)
  {
    const Mat::Sutherland* actmat = static_cast<const Mat::Sutherland*>(material.get());

    // compute temperature at n+alpha_F or n+1 and check whether it is positive
    const double tempaf = funct_.dot(escaaf);
    if (tempaf < 0.0) FOUR_C_THROW("Negative temperature in Fluid Sutherland material evaluation!");


    // compute viscosity according to Sutherland law
    visc = actmat->compute_viscosity(tempaf);

    // compute diffusivity according to Sutherland law
    diffus_ = actmat->compute_diffusivity(tempaf);

    // compute density at n+alpha_F or n+1 based on temperature
    // and thermodynamic pressure
    densaf = actmat->compute_density(tempaf, thermpressaf);

    // factor for convective scalar term at n+alpha_F or n+1
    scaconvfacaf_ = 1.0 / tempaf;

    if (fldparatimint_->is_genalpha())
    {
      // compute temperature at n+alpha_M
      const double tempam = funct_.dot(escaam);

      // factor for scalar time derivative at n+alpha_M
      scadtfac_ = 1.0 / tempam;

      // compute density at n+alpha_M based on temperature
      densam = actmat->compute_density(tempam, thermpressam);

      // addition due to thermodynamic pressure at n+alpha_M
      thermpressadd_ = -thermpressdtam / thermpressam;

      // first part of right-hand side for scalar equation:
      // time derivative of thermodynamic pressure at n+alpha_F
      scarhs_ = thermpressdtaf / actmat->shc();
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam = densaf;

      if (not fldparatimint_->is_stationary())
      {
        // compute temperature at n
        const double tempn = funct_.dot(escaam);

        // compute density at n based on temperature at n and
        // (approximately) thermodynamic pressure at n+1
        densn = actmat->compute_density(tempn, thermpressaf);

        // factor for convective scalar term at n
        scaconvfacn_ = 1.0 / tempn;

        // factor for scalar time derivative
        scadtfac_ = scaconvfacaf_;

        // addition due to thermodynamic pressure
        thermpressadd_ = -(thermpressaf - thermpressam) / (fldparatimint_->dt() * thermpressaf);

        // first part of right-hand side for scalar equation:
        // time derivative of thermodynamic pressure
        scarhs_ = (thermpressaf - thermpressam) / fldparatimint_->dt() / actmat->shc();
      }
    }

    // second part of right-hand side for scalar equation: body force
    // (value at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    const double scatrabodyforce = funct_.dot(escabofoaf);
    scarhs_ += scatrabodyforce / actmat->shc();
  }
  else if (material->material_type() == Core::Materials::m_fluid_linear_density_viscosity)
  {
    const Mat::LinearDensityViscosity* actmat =
        static_cast<const Mat::LinearDensityViscosity*>(material.get());

    double RefPressure = actmat->ref_pressure();    // reference pressure
    double CoeffDensity = actmat->coeff_density();  // density-pressure coefficient

    // compute pressure at n+alpha_F or n+1
    preaf_ = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf_);

    // compute viscosity based on pressure
    visc_ = actmat->compute_viscosity(preaf_);

    // factor for convective pressure term at n+alpha_F or n+1
    preconvfacaf_ = -1.0 / ((preaf_ - RefPressure) + 1.0 / CoeffDensity);

    if (fldparatimint_->is_genalpha())
    {
      // compute pressure at n+alpha_M
      pream_ = funct_.dot(epream);

      // compute density at n+alpha_M based on pressure
      densam_ = actmat->compute_density(pream_);

      // factor for pressure time derivative at n+alpha_M
      predtfac_ = -1.0 / ((pream_ - RefPressure) + 1.0 / CoeffDensity);
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam_ = densaf_;

      if (not fldparatimint_->is_stationary())
      {
        FOUR_C_THROW("Genalpha is the only scheme implemented for weakly compressibility");
      }
    }
  }
  else if (material->material_type() == Core::Materials::m_fluid_murnaghantait)
  {
    const Mat::MurnaghanTaitFluid* actmat =
        static_cast<const Mat::MurnaghanTaitFluid*>(material.get());

    double RefPressure = actmat->ref_pressure();         // reference pressure
    double RefBulkModulus = actmat->ref_bulk_modulus();  // reference bulk modulus
    double MatParameter =
        actmat->mat_parameter();  // material parameter according to Murnaghan-Tait

    // dynamic viscosity
    visc = actmat->viscosity();

    // compute pressure at n+alpha_F or n+1
    preaf_ = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf_);

    // factor for convective pressure term at n+alpha_F or n+1
    preconvfacaf_ = -1.0 / (MatParameter * (preaf_ - RefPressure) + RefBulkModulus);

    if (fldparatimint_->is_genalpha())
    {
      // compute pressure at n+alpha_M
      pream_ = funct_.dot(epream);

      // compute density at n+alpha_M based on pressure
      densam_ = actmat->compute_density(pream_);

      // factor for pressure time derivative at n+alpha_M
      predtfac_ = -1.0 / (MatParameter * (pream_ - RefPressure) + RefBulkModulus);
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam_ = densaf_;

      if (not fldparatimint_->is_stationary())
      {
        FOUR_C_THROW("Genalpha is the only scheme implemented for weakly compressibility");
      }
    }
  }
  else if (material->material_type() == Core::Materials::m_matlist)
  {
    // get material list for this element
    const Mat::MatList* matlist = static_cast<const Mat::MatList*>(material.get());

    int numofmaterials = matlist->num_mat();

    // Error messages
    if (numofmaterials > 2)
    {
      FOUR_C_THROW("More than two materials is currently not supported.");
    }
    if (not fldpara_->mat_gp())
    {
      FOUR_C_THROW(
          "For Two Phase Flow, the material should be evaluated at the Gauss-points. Switch "
          "EVALUATION_MAT to integration_point.");
    }

    std::vector<double> density(numofmaterials);  // Assume density[0] is on positive side, and
                                                  // density[1] is on negative side.
    std::vector<double> viscosity(numofmaterials);
    std::vector<double> gamma_vector(numofmaterials);

    for (int nmaterial = 0; nmaterial < numofmaterials; nmaterial++)
    {
      // set default id in list of materials
      int matid = -1;
      matid = matlist->mat_id(nmaterial);

      std::shared_ptr<const Core::Mat::Material> matptr = matlist->material_by_id(matid);
      Core::Materials::MaterialType mattype = matptr->material_type();

      // choose from different materials
      switch (mattype)
      {
        //--------------------------------------------------------
        // Newtonian fluid for incompressible flow (standard case)
        //--------------------------------------------------------
        case Core::Materials::m_fluid:
        {
          const Mat::NewtonianFluid* mat = static_cast<const Mat::NewtonianFluid*>(matptr.get());
          density[nmaterial] = mat->density();
          viscosity[nmaterial] = mat->viscosity();
          gamma_vector[nmaterial] = mat->gamma();
          break;
        }
        //------------------------------------------------
        // different types of materials (to be added here)
        //------------------------------------------------
        default:
          FOUR_C_THROW("Only Newtonian fluids supported as input.");
          break;
      }
    }

    double epsilon = fldpara_->get_interface_thickness();

    const double gpscaaf = funct_.dot(escaaf);  // Scalar function at gausspoint evaluated
    const double gpscaam = funct_.dot(escaam);  // Scalar function at gausspoint evaluated

    // Assign material parameter values to positive side by default.
    double heavyside_epsilon = 1.0;
    densaf = density[0];
    visc = viscosity[0];
    viscn = visc;
    densam = densaf;
    densn = densam;


    // Calculate material parameters with phiaf
    if (abs(gpscaaf) <= epsilon)
    {
      heavyside_epsilon =
          0.5 * (1.0 + gpscaaf / epsilon +
                    1.0 / std::numbers::pi * sin(std::numbers::pi * gpscaaf / epsilon));

      densaf = heavyside_epsilon * density[0] + (1.0 - heavyside_epsilon) * density[1];
      visc = heavyside_epsilon * viscosity[0] + (1.0 - heavyside_epsilon) * viscosity[1];
    }
    else if (gpscaaf < epsilon)
    {
      heavyside_epsilon = 0.0;

      densaf = density[1];
      visc = viscosity[1];
    }

    //  //Calculate material parameters with phiam
    if (abs(gpscaam) <= epsilon)
    {
      heavyside_epsilon =
          0.5 * (1.0 + gpscaam / epsilon +
                    1.0 / std::numbers::pi * sin(std::numbers::pi * gpscaam / epsilon));

      densam = heavyside_epsilon * density[0] + (1.0 - heavyside_epsilon) * density[1];
      densn = densam;

      viscn = heavyside_epsilon * viscosity[0] + (1.0 - heavyside_epsilon) * viscosity[1];
    }
    else if (gpscaam < epsilon)
    {
      heavyside_epsilon = 0.0;

      densam = density[1];
      densn = densam;

      viscn = viscosity[1];
    }


    if (gamma_vector[0] != gamma_vector[1]) FOUR_C_THROW("Surface tensions have to be equal");

    // Surface tension coefficient assigned.
    gamma = gamma_vector[0];

  }  // end else if m_matlist
  else
    FOUR_C_THROW("Material type is not supported");

  // check whether there is zero or negative (physical) viscosity
  if (visc < 1e-15) FOUR_C_THROW("zero or negative (physical) diffusivity");

  return;
}  // FluidEleCalc::get_material_params

/*----------------------------------------------------------------------*
 |  Get mk                                                     bk 09/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalc<distype, enrtype>::get_mk()
{
  return Discret::Elements::mk<distype>();
}

/*----------------------------------------------------------------------*
 |  calculation of stabilization parameter                     vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_stab_parameter(const double vol)
{
  //---------------------------------------------------------------------
  // preliminary definition of values which will already be computed for
  // tau_M and later be used for tau_C again by some of the subsequent
  // stabilization parameter definitions
  //---------------------------------------------------------------------
  double traceG = 0.0;
  double Gnormu = 0.0;
  double Gvisc = 0.0;

  double h_u = 0.0;
  double h_p = 0.0;
  double vel_norm = 0.0;
  double re12 = 0.0;
  double c3 = 0.0;

  //---------------------------------------------------------------------
  // first step: computation of tau_M with the following options
  // (both with or without inclusion of dt-part):
  // A) definition according to Taylor et al. (1998)
  //    -> see also Gravemeier and Wall (2010) for version for
  //       variable-density flow at low Mach number
  // B) combined definition according to Franca and Valentin (2000) as
  //    well as Barrenechea and Valentin (2002)
  //    -> differentiating tau_Mu and tau_Mp for this definition
  // C) definition according to Shakib (1989) / Shakib and Hughes (1991)
  //    -> differentiating tau_Mu and tau_Mp for this definition
  // D) definition according to Codina (1998)
  //    -> differentiating tau_Mu and tau_Mp for this definition
  // E) definition according to Franca et al. (2005) as well as Badia
  //    and Codina (2010)
  //    -> only for Darcy or Darcy-Stokes/Brinkman flow, hence only
  //       tau_Mp for this definition
  //---------------------------------------------------------------------
  // get element-type constant for tau
  const double mk = get_mk();


  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)  // quasistatic case
  {
    // computation depending on which parameter definition is used
    switch (fldpara_->which_tau())
    {
      case Inpar::FLUID::tau_taylor_hughes_zarins:
      case Inpar::FLUID::tau_taylor_hughes_zarins_wo_dt:
      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
      case Inpar::FLUID::tau_taylor_hughes_zarins_scaled:
      case Inpar::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt:
      {
        /*

        literature:
        1) C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
           of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
           (1998) 155-196.
        2) V. Gravemeier, W.A. Wall, An algebraic variational multiscale-
           multigrid method for large-eddy simulation of turbulent variable-
           density flow at low Mach number, J. Comput. Phys. 229 (2010)
           6047-6070.
           -> version for variable-density low-Mach-number flow as implemented
              here, which corresponds to version for incompressible flow as
              given in the previous publications when density is constant

                                                                               1
                         +-                                               -+ - -
                         |        2                                        |   2
                         | c_1*rho                                  2      |
              tau  = C * | -------   +  c_2*rho*u*G*rho*u  +  c_3*mu *G:G  |
                 M       |     2                                           |
                         |   dt                                            |
                         +-                                               -+

              with the constants and covariant metric tensor defined as follows:

              C   = 1.0 (not explicitly defined here),
              c_1 = 4.0 (for version with dt), 0.0 (for version without dt),
              c_2 = 1.0 (not explicitly defined here),
              c_3 = 12.0/m_k (36.0 for linear and 144.0 for quadratic elements)

                      +-           -+   +-           -+   +-           -+
                      |             |   |             |   |             |
                      |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
                G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
                 ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                      |    i     j  |   |    i     j  |   |    i     j  |
                      +-           -+   +-           -+   +-           -+

                      +----
                       \
              G : G =   +   G   * G
                       /     ij    ij
                      +----
                       i,j
                                 +----
                                 \
              rho*u*G*rho*u  =   +   rho*u * G  *rho*u
                                 /        i   ij      j
                                +----
                                  i,j
        */

        // total reaction coefficient sigma_tot: sum of "artificial" reaction
        // due to time factor and reaction coefficient (reaction coefficient
        // ensured to remain zero in get_material_params for non-reactive material)
        double sigma_tot = reacoeff_;
        if (fldpara_->which_tau() == Inpar::FLUID::tau_taylor_hughes_zarins or
            fldpara_->which_tau() == Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen or
            fldpara_->which_tau() == Inpar::FLUID::tau_taylor_hughes_zarins_scaled)
          sigma_tot += 1.0 / fldparatimint_->dt();

        // definition of constants as described above
        const double c1 = 4.0;
        c3 = 12.0 / mk;

        // computation of various values derived from covariant metric tensor
        // (trace of covariant metric tensor required for computation of tau_C below)
        double G;
        double normG = 0.0;
        const double dens_sqr = densaf_ * densaf_;
        for (int nn = 0; nn < nsd_; ++nn)
        {
          const double dens_sqr_velint_nn = dens_sqr * convvelint_(nn);
          for (int mm = 0; mm < nsd_; ++mm)
          {
            traceG += xji_(nn, mm) * xji_(nn, mm);
          }
          for (int rr = 0; rr < nsd_; ++rr)
          {
            G = xji_(nn, 0) * xji_(rr, 0);
            for (int mm = 1; mm < nsd_; ++mm)
            {
              G += xji_(nn, mm) * xji_(rr, mm);
            }
            normG += G * G;
            Gnormu += dens_sqr_velint_nn * G * convvelint_(rr);
          }
        }

        // compute viscous part
        Gvisc = c3 * visceff_ * visceff_ * normG;

        // compute stabilization parameter tau_Mu
        tau_(0) = 1.0 / (sqrt(c1 * dens_sqr * ((sigma_tot) * (sigma_tot)) + Gnormu + Gvisc));

        // compute stabilization parameter tau_Mp
        // ensure that tau_Mp does not become too small for viscosity-dominated flow
        // lower limit proportional to squared (Braack et al. (2007)) or cubic
        // Barth et al. (2004) characteristic length
        // here: lower-limit constant chosen to be 1.0 and cubic char. length
        const double llc = 1.0;
        const double powerfac = 3.0;
        if ((Gnormu < Gvisc) and (std::pow(traceG, (powerfac / 2.0)) < llc * sqrt(Gvisc)))
          tau_(1) = 1.0 / (sqrt(c1 * dens_sqr * ((sigma_tot) * (sigma_tot)) + Gnormu +
                                (std::pow(traceG, powerfac) / ((llc) * (llc)))));
        else
          tau_(1) = tau_(0);
      }
      break;

      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall:
      {
        /*

        literature:
        1) L.P. Franca, F. Valentin, On an improved unusual stabilized
           finite element method for the advective-reactive-diffusive
           equation, Comput. Methods Appl. Mech. Engrg. 190 (2000) 1785-1800.
        2) G.R. Barrenechea, F. Valentin, An unusual stabilized finite
           element method for a generalized Stokes problem, Number. Math.
           92 (2002) 652-677.


                      xi1,xi2 ^
                              |      /
                              |     /
                              |    /
                            1 +---+
                              |
                              |
                              |
                              +--------------> re1,re2
                                  1

        */
        // get norm of convective velocity
        vel_norm = convvelint_.norm2();

        // total reaction coefficient sigma_tot: sum of "artificial" reaction
        // due to time factor and reaction coefficient (reaction coefficient
        // ensured to remain zero in get_material_params for non-reactive material)
        const double sigma_tot = 1.0 / fldparatimint_->time_fac() + reacoeff_;
        // NOTE: Gen_Alpha (implementation by Peter Gamnitzer) used a different time factor!

        // calculate characteristic element length
        calc_char_ele_length(vol, vel_norm, h_u, h_p);

        // various parameter computations for case with dt:
        // relating viscous to reactive part (re01: tau_Mu, re11: tau_Mp)
        const double re01 = 4.0 * visceff_ / (mk * densaf_ * sigma_tot * ((h_u) * (h_u)));
        const double re11 = 4.0 * visceff_ / (mk * densaf_ * sigma_tot * ((h_p) * (h_p)));

        // relating convective to viscous part (re02: tau_Mu, re12: tau_Mp)
        const double re02 = mk * densaf_ * vel_norm * h_u / (2.0 * visceff_);
        re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);

        // respective "switching" parameters
        const double xi01 = std::max(re01, 1.0);
        const double xi11 = std::max(re11, 1.0);
        const double xi02 = std::max(re02, 1.0);
        const double xi12 = std::max(re12, 1.0);

        // compute stabilization parameter tau_Mu
        tau_(0) = ((h_u) * (h_u)) /
                  (((h_u) * (h_u)) * densaf_ * sigma_tot * xi01 + (4.0 * visceff_ / mk) * xi02);

        // compute stabilization parameter tau_Mp
        // ensure that tau_Mp does not become too small for viscosity-dominated flow
        // lower limit proportional to squared (Braack et al. (2007)) or cubic
        // Barth et al. (2004) characteristic length
        // here: lower-limit constant chosen to be 1.0 and cubic char. length
        const double llc = 1.0;
        const double powerfac = 3.0;
        if ((re12 < 1.0) and
            (llc * std::pow(h_p, powerfac) > ((h_p) * (h_p)) / (4.0 * visceff_ / mk)))
        {
          if (re11 < 1.0)
            tau_(1) = 1.0 / (densaf_ * sigma_tot + (1.0 / (llc * std::pow(h_p, powerfac))));
          else
            tau_(1) = llc * std::pow(h_p, powerfac);
        }
        else
          tau_(1) = ((h_p) * (h_p)) /
                    (((h_p) * (h_p)) * densaf_ * sigma_tot * xi11 + (4.0 * visceff_ / mk) * xi12);
      }
      break;

      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
      {
        /*

         stabilization parameter as above without inclusion of dt-part

        */
        // get norm of convective velocity
        vel_norm = convvelint_.norm2();

        // calculate characteristic element length
        calc_char_ele_length(vol, vel_norm, h_u, h_p);

        // various parameter computations for case without dt:
        // relating viscous to reactive part (re01: tau_Mu, re11: tau_Mp)
        double re01 = 0.0;
        double re11 = 0.0;
        if (fldpara_->reaction())  // TODO Martin: check influence of reaction to stabilization
        {
          re01 = 4.0 * visceff_ / (mk * densaf_ * reacoeff_ * ((h_u) * (h_u)));
          re11 = 4.0 * visceff_ / (mk * densaf_ * reacoeff_ * ((h_p) * (h_p)));
        }
        // relating convective to viscous part (re02: tau_Mu, re12: tau_Mp)
        const double re02 = mk * densaf_ * vel_norm * h_u / (2.0 * visceff_);
        re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);

        // respective "switching" parameters
        const double xi01 = std::max(re01, 1.0);
        const double xi11 = std::max(re11, 1.0);
        const double xi02 = std::max(re02, 1.0);
        const double xi12 = std::max(re12, 1.0);

        // compute stabilization parameter tau_Mu
        tau_(0) = ((h_u) * (h_u)) /
                  (((h_u) * (h_u)) * densaf_ * reacoeff_ * xi01 + (4.0 * visceff_ / mk) * xi02);

        // compute stabilization parameter tau_Mp
        // ensure that tau_Mp does not become too small for viscosity-dominated flow
        // lower limit proportional to squared (Braack et al. (2007)) or cubic
        // Barth et al. (2004) characteristic length
        // here: lower-limit constant chosen to be 1.0 and cubic char. length
        const double llc = 1.0;
        const double powerfac = 3.0;
        if ((re12 < 1.0) and
            (llc * std::pow(h_p, powerfac) > ((h_p) * (h_p)) / (4.0 * visceff_ / mk)))
        {
          if (re11 < 1.0)
            tau_(1) = 1.0 / (densaf_ * reacoeff_ + (1.0 / (llc * std::pow(h_p, powerfac))));
          else
            tau_(1) = llc * std::pow(h_p, powerfac);
        }
        else
          tau_(1) = ((h_p) * (h_p)) /
                    (((h_p) * (h_p)) * densaf_ * reacoeff_ * xi11 + (4.0 * visceff_ / mk) * xi12);
      }
      break;

      case Inpar::FLUID::tau_shakib_hughes_codina:
      case Inpar::FLUID::tau_shakib_hughes_codina_wo_dt:
      {
        /*

        literature:
        1) F. Shakib, Finite element analysis of the compressible Euler and
           Navier-Stokes equations, PhD thesis, Division of Applied Mechanics,
           Stanford University, Stanford, CA, USA, 1989.
        2) F. Shakib, T.J.R. Hughes, A new finite element formulation for
           computational fluid dynamics: IX. Fourier analysis of space-time
           Galerkin/least-squares algorithms, Comput. Methods Appl. Mech.
           Engrg. 87 (1991) 35-58.
        3) R. Codina, Stabilized finite element approximation of transient
           incompressible flows using orthogonal subscales, Comput. Methods
           Appl. Mech. Engrg. 191 (2002) 4295-4321.

           constants defined as in Shakib (1989) / Shakib and Hughes (1991),
           merely slightly different with respect to c_3:

           c_1 = 4.0 (for version with dt), 0.0 (for version without dt),
           c_2 = 4.0,
           c_3 = 4.0/(m_k*m_k) (36.0 for linear, 576.0 for quadratic ele.)

           Codina (2002) proposed present version without dt and explicit
           definition of constants
           (condition for constants as defined here: c_2 <= sqrt(c_3)).

        */
        // get norm of convective velocity
        vel_norm = convvelint_.norm2();

        // calculate characteristic element length
        calc_char_ele_length(vol, vel_norm, h_u, h_p);

        // total reaction coefficient sigma_tot: sum of "artificial" reaction
        // due to time factor and reaction coefficient (reaction coefficient
        // ensured to remain zero in get_material_params for non-reactive material)
        double sigma_tot = reacoeff_;
        if (fldpara_->which_tau() == Inpar::FLUID::tau_shakib_hughes_codina)
          sigma_tot += 1.0 / fldparatimint_->dt();

        // definition of constants as described above
        const double c1 = 4.0;
        const double c2 = 4.0;
        c3 = 4.0 / (mk * mk);
        // alternative value as proposed in Shakib (1989): c3 = 16.0/(mk*mk);

        // compute stabilization parameter tau_Mu
        tau_(0) =
            1.0 / (sqrt(c1 * ((densaf_) * (densaf_)) * ((sigma_tot) * (sigma_tot)) +
                        c2 * ((densaf_) * (densaf_)) * ((vel_norm) * (vel_norm)) / ((h_u) * (h_u)) +
                        c3 * ((visceff_) * (visceff_)) / (((h_u) * (h_u)) * ((h_u) * (h_u)))));

        // compute stabilization parameter tau_Mp
        // ensure that tau_Mp does not become too small for viscosity-dominated flow
        // lower limit proportional to squared (Braack et al. (2007)) or cubic
        // Barth et al. (2004) characteristic length
        // here: lower-limit constant chosen to be 1.0 and cubic char. length
        const double llc = 1.0;
        const double powerfac = 3.0;
        const double re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
        if ((re12 < 1.0) and
            (llc * std::pow(h_p, powerfac) > ((h_p) * (h_p)) / (sqrt(c3) * visceff_)))
          tau_(1) =
              1.0 /
              (sqrt(c1 * ((densaf_) * (densaf_)) * ((sigma_tot) * (sigma_tot)) +
                    c2 * ((densaf_) * (densaf_)) * ((vel_norm) * (vel_norm)) / ((h_p) * (h_p)) +
                    1.0 / ((llc * std::pow(h_p, powerfac)) * (llc * std::pow(h_p, powerfac)))));
        else
          tau_(1) =
              1.0 /
              (sqrt(c1 * ((densaf_) * (densaf_)) * ((sigma_tot) * (sigma_tot)) +
                    c2 * ((densaf_) * (densaf_)) * ((vel_norm) * (vel_norm)) / ((h_p) * (h_p)) +
                    c3 * ((visceff_) * (visceff_)) / (((h_p) * (h_p)) * ((h_p) * (h_p)))));
      }
      break;

      case Inpar::FLUID::tau_codina:
      case Inpar::FLUID::tau_codina_wo_dt:
      case Inpar::FLUID::tau_codina_convscaled:
      {
        /*

          literature:
             R. Codina, Comparison of some finite element methods for solving
             the diffusion-convection-reaction equation, Comput. Methods
             Appl. Mech. Engrg. 156 (1998) 185-210.

             constants:
             c_1 = 1.0 (for version with dt), 0.0 (for version without dt),
             c_2 = 2.0,
             c_3 = 4.0/m_k (12.0 for linear, 48.0 for quadratic elements)

             Codina (1998) proposed present version without dt.

        */
        // get norm of convective velocity
        vel_norm = convvelint_.norm2();

        // calculate characteristic element length
        calc_char_ele_length(vol, vel_norm, h_u, h_p);

        // total reaction coefficient sigma_tot: sum of "artificial" reaction
        // due to time factor and reaction coefficient (reaction coefficient
        // ensured to remain zero in get_material_params for non-reactive material)
        double sigma_tot = reacoeff_;
        if (fldpara_->which_tau() == Inpar::FLUID::tau_codina ||
            fldpara_->which_tau() == Inpar::FLUID::tau_codina_convscaled)
          sigma_tot += 1.0 / fldparatimint_->dt();

        // definition of constants as described above
        const double c1 = 1.0;
        double c2 = 2.0;
        c3 = 4.0 / mk;

        // for high-order elements or non-polynomial enrichments,
        // a scaling of the convective term like this gives good results
        if (fldpara_->which_tau() == Inpar::FLUID::tau_codina_convscaled) c2 = sqrt(c3 / 3.0);

        // compute stabilization parameter tau_Mu
        tau_(0) = 1.0 / (c1 * densaf_ * sigma_tot + c2 * densaf_ * vel_norm / h_u +
                            c3 * visceff_ / ((h_u) * (h_u)));

        // compute stabilization parameter tau_Mp
        // ensure that tau_Mp does not become too small for viscosity-dominated flow
        // lower limit proportional to squared (Braack et al. (2007)) or cubic
        // Barth et al. (2004) characteristic length
        // here: lower-limit constant chosen to be 1.0 and cubic char. length
        const double llc = 1.0;
        const double powerfac = 3.0;
        const double re12 = mk * densaf_ * vel_norm * h_p / (2.0 * visceff_);
        if ((re12 < 1.0) and (llc * std::pow(h_p, powerfac) > ((h_p) * (h_p)) / (c3 * visceff_)))
          tau_(1) = 1.0 / (c1 * densaf_ * sigma_tot + c2 * densaf_ * vel_norm / h_p +
                              1.0 / (llc * std::pow(h_p, powerfac)));
        else
          tau_(1) = 1.0 / (c1 * densaf_ * sigma_tot + c2 * densaf_ * vel_norm / h_p +
                              c3 * visceff_ / ((h_p) * (h_p)));
      }
      break;

      case Inpar::FLUID::tau_franca_madureira_valentin_badia_codina:
      case Inpar::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt:
      {
        /*

        This stabilization parameter is only intended to be used for
        (viscous-)reactive problems such as Darcy(-Stokes/Brinkman) problems.

        literature:
        1) L.P. Franca, A.L. Madureira, F. Valentin, Towards multiscale
           functions: enriching finite element spaces with local but not
           bubble-like functions, Comput. Methods Appl. Mech. Engrg. 194
           (2005) 3006-3021.
        2) S. Badia, R. Codina, Stabilized continuous and discontinuous
           Galerkin techniques for Darcy flow, Comput. Methods Appl.
           Mech. Engrg. 199 (2010) 1654-1667.

        */
        // total reaction coefficient sigma_tot: sum of "artificial" reaction
        // due to time factor and reaction coefficient
        double sigma_tot = reacoeff_;
        if (fldpara_->which_tau() == Inpar::FLUID::tau_franca_madureira_valentin_badia_codina)
          sigma_tot += 1.0 / fldparatimint_->time_fac();

        // calculate characteristic element length
        calc_char_ele_length(vol, 0.0, h_u, h_p);

        // various parameter computations for case with dt:
        // relating viscous to reactive part
        const double re11 = 2.0 * visceff_ / (mk * densaf_ * sigma_tot * ((h_p) * (h_p)));

        // respective "switching" parameter
        const double xi11 = std::max(re11, 1.0);

        // constant c_u as suggested in Badia and Codina (2010), method A
        // (set to be 4.0 in Badia and Codina (2010), 1.0 in Franca et al. (2005))
        const double c_u = 4.0;

        // compute stabilization parameter tau_Mp (tau_Mu not required)
        tau_(0) = 0.0;
        tau_(1) = ((h_p) * (h_p)) /
                  (c_u * ((h_p) * (h_p)) * densaf_ * sigma_tot * xi11 + (2.0 * visceff_ / mk));
      }
      break;

      case Inpar::FLUID::tau_hughes_franca_balestra_wo_dt:
      {
        /*----------------------------------------------------------------------*/
        /*
         *  This stabilization parameter is only intended to be used for
         *  stationary Stokes problems.
         *
         *  literature:
         *  1) T.J.R. Hughes, L.P. Franca, and M. Balestra, A new finite element
         *     formulation for computational fluid dynamics: V. circumventing the
         *     Babuska-Brezzi condition: a stable Petrov-Galerkin formulation of
         *     the Stokes problem accommodating equal-order interpolations,
         *     Comput. Methods Appl. Mech. Engrg. 59 (1986) 85-99.
         *
         *  2) J. Donea and A. Huerta, Finite element methods for flow problems.
         *     (for alpha_0 = 1/3)
         */
        /*----------------------------------------------------------------------*/

        // calculate characteristic element length
        calc_char_ele_length(vol, 0.0, h_u, h_p);

        // compute stabilization parameter tau_Mp (tau_Mu not required)
        tau_(0) = 0.0;
        tau_(1) = (1.0 / 3.0) * std::pow(h_p, 2.0) / (4.0 * visceff_);
      }
      break;

      default:
      {
        if (not(fldpara_->stab_type() == Inpar::FLUID::stabtype_edgebased and
                fldpara_->which_tau() == Inpar::FLUID::tau_not_defined))
          FOUR_C_THROW("unknown definition for tau_M\n {}  ", fldpara_->which_tau());

        break;
      }
    }  // end switch (fldpara_->WhichTau())


    //---------------------------------------------------------------------
    // second step: computation of tau_C with the following options:
    // A) definition according to Taylor et al. (1998)
    // B) definition according to Whiting (1999)/Whiting and Jansen (2001)
    // C) scaled version of definition according to Taylor et al. (1998)
    // D) definition according to Wall (1999)
    // E) definition according to Codina (2002)
    // F) definition according to Badia and Codina (2010)
    //    (only for Darcy or Darcy-Stokes/Brinkman flow)
    //---------------------------------------------------------------------
    // computation depending on which parameter definition is used
    switch (fldpara_->which_tau())
    {
      case Inpar::FLUID::tau_taylor_hughes_zarins:
      case Inpar::FLUID::tau_taylor_hughes_zarins_wo_dt:
      {
        /*

        literature:
           C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
           of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
           (1998) 155-196.

                                                  1/2
                               (c_2*rho*u*G*rho*u)
                        tau  = -------------------
                           C       trace (G)


           -> see respective definitions for computation of tau_M above

        */

        tau_(2) = sqrt(Gnormu) / traceG;
      }
      break;

      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
      {
        /*

        literature:
        1) C.H. Whiting, Stabilized finite element methods for fluid dynamics
           using a hierarchical basis, PhD thesis, Rensselaer Polytechnic
           Institute, Troy, NY, USA, 1999.
        2) C.H. Whiting, K.E. Jansen, A stabilized finite element method for
           the incompressible Navier-Stokes equations using a hierarchical
           basis, Int. J. Number. Meth. Fluids 35 (2001) 93-116.

                                      1.0
                        tau  = ------------------
                           C    tau  * trace (G)
                                   M

           -> see respective definitions for computation of tau_M above

        */

        tau_(2) = 1.0 / (tau_(0) * traceG);
      }
      break;

      case Inpar::FLUID::tau_taylor_hughes_zarins_scaled:
      case Inpar::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt:
      {
        /*

          Caution: This is an experimental version of a stabilization
                   parameter definition which scales the definition
                   for tau_C by Taylor et al. (1998) in a similar
                   way as proposed below by Franca and Frey (1992)
                   and Wall (1999) by appropriately defining an
                   element Reynolds number based on the covariant
                   metric tensor.

                      /                        1/2    \
                      |  /                    \       |                       1/2
                      | |  c_2*rho*u*G*rho*u  |       |    (c_2*rho*u*G*rho*u)
          tau  =  MIN | | ------------------- | | 1.0 | *  -------------------
             C        | |          2          |       |         trace (G)
                      | \    c_3*mu *G:G      /       |
                      \                               /
                        |                     |
                        -----------------------
                        element Reynolds number
                          based on covariant
                            metric tensor

           -> see respective definitions for computation of tau_M above

        */

        // element Reynolds number based on covariant metric tensor
        const double reG = sqrt(Gnormu / Gvisc);

        // "switching" parameter
        const double xi_tau_c = std::min(reG, 1.0);

        tau_(2) = xi_tau_c * sqrt(Gnormu) / traceG;
      }
      break;

      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall:
      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
      {
        /*

        literature:
        1) L.P. Franca, S.L. Frey, Stabilized finite element methods:
           II. The incompressible Navier-Stokes equations, Comput. Methods
           Appl. Mech. Engrg. 99 (1992) 209-293.
        2) W.A. Wall, Fluid-Struktur-Interaction mit stabilisierten Finiten
           Elementen, Dissertation, Universitaet Stuttgart, 1999.

                     xi_tau_c ^
                              |
                            1 |   +-----------
                              |  /
                              | /
                              |/
                              +--------------> re12
                                  1

           -> see respective definitions for computation of tau_M above

        */

        // "switching" parameter
        const double xi_tau_c = std::min(re12, 1.0);

        tau_(2) = 0.5 * densaf_ * vel_norm * h_p * xi_tau_c;
      }
      break;

      case Inpar::FLUID::tau_shakib_hughes_codina:
      case Inpar::FLUID::tau_shakib_hughes_codina_wo_dt:
      {
        /*

        literature:
           R. Codina, Stabilized finite element approximations of transient
           incompressible flows using orthogonal subscales, Comput. Methods
           Appl. Mech. Engrg. 191 (2002) 4295-4321.

           -> see respective definitions for computation of tau_M above

        */

        tau_(2) = ((h_p) * (h_p)) / (sqrt(c3) * tau_(1));
      }
      break;

      case Inpar::FLUID::tau_codina:
      case Inpar::FLUID::tau_codina_wo_dt:
      case Inpar::FLUID::tau_codina_convscaled:
      {
        /*

        literature:
           R. Codina, Stabilized finite element approximations of transient
           incompressible flows using orthogonal subscales, Comput. Methods
           Appl. Mech. Engrg. 191 (2002) 4295-4321.

           -> see respective definitions for computation of tau_M above

           fixed bug: before we used sqrt(c3) instead of c3 here   bk 2014
           see for example Diss Peter Gamnitzer for correct definition
        */

        tau_(2) = ((h_p) * (h_p)) / (c3 * tau_(1));
      }
      break;

      case Inpar::FLUID::tau_franca_madureira_valentin_badia_codina:
      case Inpar::FLUID::tau_franca_madureira_valentin_badia_codina_wo_dt:
      {
        /*

        This stabilization parameter is only intended to be used for
        (viscous-)reactive problems such as Darcy(-Stokes/Brinkman) problems.

        literature:
           S. Badia, R. Codina, Stabilized continuous and discontinuous
           Galerkin techniques for Darcy flow, Comput. Methods Appl.
           Mech. Engrg. 199 (2010) 1654-1667.

        */

        // constant c_p as suggested in Badia and Codina (2010), method A
        // (set to be 4.0 in Badia and Codina (2010))
        const double c_p = 4.0;

        tau_(2) = c_p * ((h_p) * (h_p)) * reacoeff_;
      }
      break;

      case Inpar::FLUID::tau_hughes_franca_balestra_wo_dt:
      {
        /*----------------------------------------------------------------------*/
        /*
         *  This stabilization parameter is only intended to be used for
         *  stationary Stokes problems.
         *
         *  literature:
         *  1) T.J.R. Hughes, L.P. Franca, and M. Balestra, A new finite element
         *     formulation for computational fluid dynamics: V. circumventing the
         *     Babuska-Brezzi condition: a stable Petrov-Galerkin formulation of
         *     the Stokes problem accommodating equal-order interpolations,
         *     Comput. Methods Appl. Mech. Engrg. 59 (1986) 85-99.
         *
         *  2) J. Donea and A. Huerta, Finite element methods for flow problems.
         *     (for alpha_0 = 1/3)
         */
        /*----------------------------------------------------------------------*/

        // tau_C not required)
        tau_(2) = 0.0;
      }
      break;

      default:
      {
        if (not(fldpara_->stab_type() == Inpar::FLUID::stabtype_edgebased and
                fldpara_->which_tau() == Inpar::FLUID::tau_not_defined))
          FOUR_C_THROW("unknown definition for tau_C\n {}  ", fldpara_->which_tau());

        break;
      }
    }  // end switch (fldpara_->WhichTau())

  }  // end of quasistatic case

  //-----------------------------------------------------------
  //-----------------------------------------------------------

  else  // Inpar::FLUID::subscales_time_dependent
  {
    // norms of velocity in gausspoint, time n+af and time n+1
    const double vel_normaf = convvelint_.norm2();
    const double vel_normnp = vel_normnp_;

    // get velocity (n+alpha_F,i) derivatives at integration point
    //
    //       n+af      +-----  dN (x)
    //   dvel    (x)    \        k         n+af
    //   ----------- =   +     ------ * vel
    //       dx         /        dx        k
    //         j       +-----      j
    //                 node k
    //
    // j : direction of derivative x/y/z
    //
    static Core::LinAlg::Matrix<nsd_, nsd_> vderxyaf_(Core::LinAlg::Initialization::zero);
    vderxyaf_.update(1.0, vderxy_, 0.0);

    // Now we are ready. Let's go on!


    //-------------------------------------------------------
    //          TAUS FOR TIME DEPENDENT SUBSCALES
    //-------------------------------------------------------

    switch (fldpara_->which_tau())
    {
      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen:
      case Inpar::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
      {
        /* INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA

           tau_M: Bazilevs et al. + ideas from Codina
                                                           1.0
                   +-                                 -+ - ---
                   |                                   |   2.0
               td  |  n+af      n+af         2         |
            tau  = | u     * G u     + C * nu  * G : G |
               M   |         -          I        -   - |
                   |         -                   -   - |
                   +-                                 -+

           tau_C: Bazilevs et al., derived from the fine scale complement Shur
                                   operator of the pressure equation


                         td         1.0
                      tau  = -----------------
                         C       td   /     \
                              tau  * | g * g |
                                 M    \-   -/
        */

        /*          +-           -+   +-           -+   +-           -+
                    |             |   |             |   |             |
                    |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
              G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
               ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                    |    i     j  |   |    i     j  |   |    i     j  |
                    +-           -+   +-           -+   +-           -+
        */
        static Core::LinAlg::Matrix<nsd_, nsd_> G;

        for (int nn = 0; nn < nsd_; ++nn)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            G(nn, rr) = xji_(nn, 0) * xji_(rr, 0);
            for (int mm = 1; mm < nsd_; ++mm)
            {
              G(nn, rr) += xji_(nn, mm) * xji_(rr, mm);
            }
          }
        }

        /*          +----
                     \
            G : G =   +   G   * G
            -   -    /     ij    ij
            -   -   +----
                     i,j
        */
        double normG = 0;
        for (int nn = 0; nn < nsd_; ++nn)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            normG += G(nn, rr) * G(nn, rr);
          }
        }

        /*                    +----
             n+af      n+af    \     n+af         n+af
            u     * G u     =   +   u    * G   * u
                    -          /     i     -ij    j
                    -         +----        -
                               i,j
        */
        double Gnormu = 0;
        for (int nn = 0; nn < nsd_; ++nn)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            Gnormu += convvelint_(nn) * G(nn, rr) * convvelint_(rr);
          }
        }

        // definition of constant
        // (Akkerman et al. (2008) used 36.0 for quadratics, but Stefan
        //  brought 144.0 from Austin...)
        const double CI = 12.0 / mk;

        /*                                                 1.0
                   +-                                 -+ - ---
                   |                                   |   2.0
                   |  n+af      n+af         2         |
            tau  = | u     * G u     + C * nu  * G : G |
               M   |         -          I        -   - |
                   |         -                   -   - |
                   +-                                 -+
        */
        tau_(0) = 1.0 / sqrt(Gnormu + CI * visceff_ * visceff_ * normG);
        tau_(1) = tau_(0);

        /*         +-     -+   +-     -+   +-     -+
                   |       |   |       |   |       |
                   |  dr   |   |  ds   |   |  dt   |
              g  = |  ---  | + |  ---  | + |  ---  |
               i   |  dx   |   |  dx   |   |  dx   |
                   |    i  |   |    i  |   |    i  |
                   +-     -+   +-     -+   +-     -+
        */
        static Core::LinAlg::Matrix<nsd_, 1> g;

        for (int rr = 0; rr < nsd_; ++rr)
        {
          g(rr) = xji_(rr, 0);
          for (int mm = 1; mm < nsd_; ++mm)
          {
            g(rr) += xji_(rr, mm);
          }
        }

        /*         +----
                    \
           g * g =   +   g * g
           -   -    /     i   i
                   +----
                     i
        */
        double normgsq = 0.0;

        for (int rr = 0; rr < nsd_; ++rr)
        {
          normgsq += g(rr) * g(rr);
        }

        /*
                                  1.0
                    tau  = -----------------
                       C            /      \
                            tau  * | g * g |
                               M    \-   -/
        */
        tau_(2) = 1. / (tau_(0) * normgsq);
      }
      break;
      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall:
      case Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt:
      {
        // INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA
        //
        // tau_M: modification of
        //
        //    Franca, L.P. and Valentin, F.: On an Improved Unusual Stabilized
        //    Finite Element Method for the Advective-Reactive-Diffusive
        //    Equation. Computer Methods in Applied Mechanics and Engineering,
        //    Vol. 190, pp. 1785-1800, 2000.
        //    http://www.lncc.br/~valentin/publication.htm                   */
        //
        // tau_Mp: modification of Barrenechea, G.R. and Valentin, F.
        //
        //    Barrenechea, G.R. and Valentin, F.: An unusual stabilized finite
        //    element method for a generalized Stokes problem. Numerische
        //    Mathematik, Vol. 92, pp. 652-677, 2002.
        //    http://www.lncc.br/~valentin/publication.htm
        //
        //
        // tau_C: kept Wall definition
        //
        // for the modifications see Codina, Principe, Guasch, Badia
        //    "Time dependent subscales in the stabilized finite  element
        //     approximation of incompressible flow problems"
        //
        //
        // see also: Codina, R. and Soto, O.: Approximation of the incompressible
        //    Navier-Stokes equations using orthogonal subscale stabilisation
        //    and pressure segregation on anisotropic finite element meshes.
        //    Computer methods in Applied Mechanics and Engineering,
        //    Vol 193, pp. 1403-1419, 2004.

        // calculate characteristic element length
        calc_char_ele_length(vol, vel_norm, h_u, h_p);

        //---------------------------------------------- compute tau_Mu = tau_Mp
        /* convective : viscous forces (element reynolds number)*/
        const double re_convectaf = (vel_normaf * h_p / visceff_) * (mk / 2.0);
        const double xi_convectaf = std::max(re_convectaf, 1.0);

        /*
                 xi_convect ^
                            |      /
                            |     /
                            |    /
                          1 +---+
                            |
                            |
                            |
                            +--------------> re_convect
                                1
        */

        /* the 4.0 instead of the Franca's definition 2.0 results from the viscous
         * term in the Navier-Stokes-equations, which is scaled by 2.0*nu         */

        tau_(0) = ((h_p) * (h_p)) / (4.0 * visceff_ / mk + (4.0 * visceff_ / mk) * xi_convectaf);

        tau_(1) = tau_(0);

        /*------------------------------------------------------ compute tau_C ---*/

        //-- stability parameter definition according to Wall Diss. 99
        /*
                 xi_convect ^
                            |
                          1 |   +-----------
                            |  /
                            | /
                            |/
                            +--------------> Re_convect
                                1
        */
        const double re_convectnp = (vel_normnp * h_p / visceff_) * (mk / 2.0);

        const double xi_tau_c = std::min(re_convectnp, 1.0);

        tau_(2) = vel_normnp * h_p * 0.5 * xi_tau_c;
      }
      break;
      case Inpar::FLUID::tau_codina:
      {
        // Parameter from Codina, Badia (Constants are chosen according to
        // the values in the standard definition above)

        const double CI = 4.0 / mk;
        const double CII = 2.0 / mk;

        // in contrast to the original definition, we neglect the influence of
        // the subscale velocity on velnormaf
        tau_(0) = 1.0 / (CI * visceff_ / (h_p * h_p) + CII * vel_normaf / h_p);

        tau_(1) = tau_(0);

        tau_(2) = (h_p * h_p) / (CI * tau_(0));
      }
      break;
      default:
      {
        FOUR_C_THROW(
            "Unknown definition of stabilization parameter for time-dependent formulation\n");
      }
      break;
    }  // Switch TauType

  }  // end Fluid::subscales_time_dependent

  return;
}


/*----------------------------------------------------------------------*
 |  calculation of characteristic element length               vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_char_ele_length(
    const double vol, const double vel_norm, double& h_u, double& h_p)
{
  // check for potential 1-D computations
  if (nsd_ == 1) FOUR_C_THROW("Element length not implemented for 1-D computation!");

  //---------------------------------------------------------------------
  // select from various definitions for characteristic element length
  // for tau_Mu
  //---------------------------------------------------------------------
  switch (fldpara_->char_ele_length_u())
  {
    // a) streamlength due to Tezduyar et al. (1992) -> default
    // normed velocity vector
    case Inpar::FLUID::streamlength_u:
    {
      static Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
      if (vel_norm >= 1e-6)
        velino.update(1.0 / vel_norm, convvelint_);
      else
      {
        velino.clear();
        velino(0, 0) = 1;
      }

      // get streamlength using the normed velocity at element centre
      static Core::LinAlg::Matrix<nen_, 1> tmp;

      // enriched dofs are not interpolatory with respect to geometry
      if (enrtype == Discret::Elements::Fluid::xwall)
      {
        static Core::LinAlg::Matrix<nsd_, nen_> derxy_copy(derxy_);
        for (int inode = 1; inode < nen_; inode += 2)
        {
          for (int idim = 0; idim < nsd_; idim++) derxy_copy(idim, inode) = 0.0;
        }
        tmp.multiply_tn(derxy_copy, velino);
      }
      else
        tmp.multiply_tn(derxy_, velino);

      const double val = tmp.norm1();
      h_u = 2.0 / val;  // h=streamlength
    }
    break;

    // b) volume-equivalent diameter (warning: 3-D formula!)
    case Inpar::FLUID::volume_equivalent_diameter_u:
    {
      h_u = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);
    }
    break;

    // c) cubic/square root of element volume/area or element length (3- and 2-D)
    case Inpar::FLUID::root_of_volume_u:
    {
      // cast dimension to a double variable -> pow()
      const double dim = double(nsd_);
      h_u = std::pow(vol, 1.0 / dim);
    }
    break;

    default:
      FOUR_C_THROW("unknown characteristic element length for tau_Mu\n");
      break;
  }  // switch (charelelengthu_)

  //---------------------------------------------------------------------
  // select from various definitions for characteristic element length
  // for tau_Mp and tau_C
  //---------------------------------------------------------------------
  switch (fldpara_->char_ele_length_pc())
  {
    // a) streamlength due to Tezduyar et al. (1992) -> default
    // normed velocity vector
    case Inpar::FLUID::streamlength_pc:
    {
      static Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
      if (vel_norm >= 1e-6)
        velino.update(1.0 / vel_norm, convvelint_);
      else
      {
        velino.clear();
        velino(0, 0) = 1;
      }

      // get streamlength using the normed velocity at element centre
      static Core::LinAlg::Matrix<nen_, 1> tmp;
      // enriched dofs are not interpolatory with respect to geometry
      if (enrtype == Discret::Elements::Fluid::xwall)
      {
        static Core::LinAlg::Matrix<nsd_, nen_> derxy_copy(derxy_);
        for (int inode = 1; inode < nen_; inode += 2)
        {
          for (int idim = 0; idim < nsd_; idim++) derxy_copy(idim, inode) = 0.0;
        }
        tmp.multiply_tn(derxy_copy, velino);
      }
      else
        tmp.multiply_tn(derxy_, velino);
      const double val = tmp.norm1();
      h_p = 2.0 / val;  // h=streamlength
    }
    break;

    // b) volume-equivalent diameter (warning: 3-D formula!)
    case Inpar::FLUID::volume_equivalent_diameter_pc:
    {
      h_p = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);
    }
    break;

    // c) cubic/square root of element volume/area or element length (3- and 2-D)
    case Inpar::FLUID::root_of_volume_pc:
    {
      // cast dimension to a double variable -> pow()
      const double dim = double(nsd_);
      h_p = std::pow(vol, 1 / dim);
    }
    break;

    default:
      FOUR_C_THROW("unknown characteristic element length for tau_Mu and tau_C\n");
      break;
  }  // switch (charelelengthpc_)

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_div_eps(
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf)
{
  /*--- viscous term: div(epsilon(u)) --------------------------------*/
  /*   /                                                \
       |  2 N_x,xx + N_x,yy + N_y,xy + N_x,zz + N_z,xz  |
     1 |                                                |
     - |  N_y,xx + N_x,yx + 2 N_y,yy + N_z,yz + N_y,zz  |
     2 |                                                |
       |  N_z,xx + N_x,zx + N_y,zy + N_z,yy + 2 N_z,zz  |
       \                                                /

       with N_x .. x-line of N
       N_y .. y-line of N                                             */

  /*--- subtraction for low-Mach-number flow: div((1/3)*(div u)*I) */
  /*   /                            \
       |  N_x,xx + N_y,yx + N_z,zx  |
     1 |                            |
  -  - |  N_x,xy + N_y,yy + N_z,zy  |
     3 |                            |
       |  N_x,xz + N_y,yz + N_z,zz  |
       \                            /

         with N_x .. x-line of N
         N_y .. y-line of N                                             */

  // set visc_old to zero
  visc_old_.clear();

  double prefac;
  if (fldpara_->physical_type() == Inpar::FLUID::loma or
      fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
      fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
  // if(loma_)
  {
    prefac = 1.0 / 3.0;
    derxy2_.scale(prefac);
  }
  else
    prefac = 1.0;

  // reconstruction of second derivative via projection or superconvergent patch recovery
  if (fldpara_->is_reconstruct_der())
  {
    if (is_higher_order_ele_ == false) FOUR_C_THROW("this doesn't make sense");

    // global second derivatives of evalaf (projected velgrad)
    // not symmetric!
    Core::LinAlg::Matrix<nsd_ * nsd_, nsd_> evelgradderxy;
    evelgradderxy.multiply_nt(evelafgrad_, derxy_);
    evelgradderxy.scale(prefac);
    if (nsd_ == 3)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy + u,zz)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1) +
                          evelgradderxy(nsd_idim + 2, 2)) /
                      prefac;
        // interpolate mixed terms
        double sum2 = 0.0;
        switch (idim)
        {
          case 0:
            // uy,xy + uz,xz
            sum2 = 0.5 * (evelgradderxy(3, 1) + evelgradderxy(4, 0) + evelgradderxy(6, 2) +
                             evelgradderxy(8, 0));
            break;
          case 1:
            // ux,xy + uz,yz
            sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1) + evelgradderxy(7, 2) +
                             evelgradderxy(8, 1));
            break;
          case 2:
            // ux,xz + uy,yz
            sum2 = 0.5 * (evelgradderxy(2, 0) + evelgradderxy(0, 2) + evelgradderxy(5, 1) +
                             evelgradderxy(4, 2));
            break;
          default:
            FOUR_C_THROW("only 3d");
            break;
        }
        // assemble each row of div epsilon(evelaf)
        visc_old_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else if (nsd_ == 2)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1)) / prefac;
        // interpolate mixed terms
        double sum2 = 0.0;
        switch (idim)
        {
          case 0:
            // uy,xy
            sum2 = 0.5 * (evelgradderxy(2, 1) + evelgradderxy(3, 0));
            break;
          case 1:
            // ux,xy
            sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1));
            break;
          default:
            FOUR_C_THROW("only 2d");
            break;
        }

        // assemble each row of div epsilon(evelaf)
        visc_old_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else
      FOUR_C_THROW("Epsilon(N) is not implemented for the 1D case");
  }
  else  // get second derivatives via shape functions
  {
    if (nsd_ == 3)
    {
      for (int inode = 0; inode < nen_; ++inode)
      {
        double sum = (derxy2_(0, inode) + derxy2_(1, inode) + derxy2_(2, inode)) / prefac;
        viscs2_(0, inode) = 0.5 * (sum + derxy2_(0, inode));
        viscs2_(1, inode) = 0.5 * derxy2_(3, inode);
        viscs2_(2, inode) = 0.5 * derxy2_(4, inode);
        viscs2_(3, inode) = 0.5 * derxy2_(3, inode);
        viscs2_(4, inode) = 0.5 * (sum + derxy2_(1, inode));
        viscs2_(5, inode) = 0.5 * derxy2_(5, inode);
        viscs2_(6, inode) = 0.5 * derxy2_(4, inode);
        viscs2_(7, inode) = 0.5 * derxy2_(5, inode);
        viscs2_(8, inode) = 0.5 * (sum + derxy2_(2, inode));

        for (int idim = 0; idim < nsd_; ++idim)
        {
          const int nsd_idim = idim * nsd_;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            visc_old_(idim) += viscs2_(nsd_idim + jdim, inode) * evelaf(jdim, inode);
          }
        }
      }
    }
    else if (nsd_ == 2)
    {
      for (int inode = 0; inode < nen_; ++inode)
      {
        double sum = (derxy2_(0, inode) + derxy2_(1, inode)) / prefac;
        viscs2_(0, inode) = 0.5 * (sum + derxy2_(0, inode));
        viscs2_(1, inode) = 0.5 * derxy2_(2, inode);
        viscs2_(2, inode) = 0.5 * derxy2_(2, inode);
        viscs2_(3, inode) = 0.5 * (sum + derxy2_(1, inode));

        for (int idim = 0; idim < nsd_; ++idim)
        {
          const int nsd_idim = idim * nsd_;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            visc_old_(idim) += viscs2_(nsd_idim + jdim, inode) * evelaf(jdim, inode);
          }
        }
      }
    }
    else
      FOUR_C_THROW("Epsilon(N) is not implemented for the 1D case");
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_subgrid_scale_velocity(
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, double& fac1, double& fac2, double& fac3,
    double& facMtau, int iquad, double* saccn, double* sveln, double* svelnp)
{
  //----------------------------------------------------------------------
  // compute residual of momentum equation
  // -> different for generalized-alpha and other time-integration schemes
  //----------------------------------------------------------------------
  if (fldparatimint_->is_genalpha())
  {
    // rhs of momentum equation: density*bodyforce at n+alpha_F
    if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
    {
      // safety check
      if (fldparatimint_->alpha_f() != 1.0 or fldparatimint_->gamma() != 1.0)
        FOUR_C_THROW(
            "Boussinesq approximation in combination with generalized-alpha time integration "
            "has only been tested for BDF2-equivalent time integration parameters! "
            "Feel free to remove this error at your own risk!");

      rhsmom_.update(deltadens_, bodyforce_, 0.0);
    }
    else
      rhsmom_.update(densaf_, bodyforce_, 0.0);

    // add pressure gradient prescribed as body force (caution: not density weighted)
    rhsmom_.update(1.0, generalbodyforce_, 1.0);

    // get acceleration at time n+alpha_M at integration point
    accint_.multiply(eaccam, funct_);

    // evaluate momentum residual once for all stabilization right hand sides
    for (int rr = 0; rr < nsd_; ++rr)
      momres_old_(rr) = densam_ * accint_(rr) + densaf_ * conv_old_(rr) + gradp_(rr) -
                        2 * visceff_ * visc_old_(rr) + reacoeff_ * velint_(rr) - rhsmom_(rr);

    // add consistency terms for MFS if applicable
    multfrac_sub_grid_scales_consistent_residual();
  }
  else
  {
    if (not fldparatimint_->is_stationary())
    {
      // rhs of instationary momentum equation:
      // density*theta*bodyforce at n+1 + density*(histmom/dt)
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g Changed density from densn_
      // to densaf_. Makes the OST consistent with the gen-alpha.
      if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
        rhsmom_.update((densaf_ / fldparatimint_->dt() / fldparatimint_->theta()), histmom_,
            deltadens_, bodyforce_);
      else
        rhsmom_.update((densaf_ / fldparatimint_->dt() / fldparatimint_->theta()), histmom_,
            densaf_, bodyforce_);

      // add pressure gradient prescribed as body force (caution: not density weighted)
      rhsmom_.update(1.0, generalbodyforce_, 1.0);

      // compute instationary momentum residual:
      // momres_old = u_(n+1)/dt + theta ( ... ) - histmom_/dt - theta*bodyforce_
      for (int rr = 0; rr < nsd_; ++rr)
        momres_old_(rr) = ((densaf_ * velint_(rr) / fldparatimint_->dt() +
                               fldparatimint_->theta() *
                                   (densaf_ * conv_old_(rr) + gradp_(rr) -
                                       2 * visceff_ * visc_old_(rr) + reacoeff_ * velint_(rr))) /
                              fldparatimint_->theta()) -
                          rhsmom_(rr);
    }
    else
    {
      // rhs of stationary momentum equation: density*bodyforce
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g and pressure gradient
      // prescribed as body force (not density weighted)
      if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
        rhsmom_.update(deltadens_, bodyforce_, 1.0, generalbodyforce_);
      else
        rhsmom_.update(densaf_, bodyforce_, 1.0, generalbodyforce_);

      // compute stationary momentum residual:
      for (int rr = 0; rr < nsd_; ++rr)
      {
        momres_old_(rr) = densaf_ * conv_old_(rr) + gradp_(rr) - 2 * visceff_ * visc_old_(rr) +
                          reacoeff_ * velint_(rr) - rhsmom_(rr);
      }

      // add consistency terms for MFS if applicable
      multfrac_sub_grid_scales_consistent_residual();
    }
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale velocity
  //----------------------------------------------------------------------
  // 1) quasi-static subgrid scales
  // Definition of subgrid-scale velocity is not consistent for the SUPG term and Franca, Valentin,
  // ... Definition of subgrid velocity used by Hughes
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    sgvelint_.update(-tau_(1), momres_old_, 0.0);
  }
  // 2) time-dependent subgrid scales
  else
  {
    // some checking
    if (fldparatimint_->is_stationary())
      FOUR_C_THROW("there is no time dependent subgrid scale closure for stationary problems\n");
    if (saccn == nullptr or sveln == nullptr or svelnp == nullptr)
      FOUR_C_THROW("no subscale array provided");

    // parameter definitions
    double alphaF = fldparatimint_->alpha_f();
    double alphaM = fldparatimint_->alpha_m();
    double gamma = fldparatimint_->gamma();
    double dt = fldparatimint_->dt();

    /*
                                            1.0
       facMtau =  -------------------------------------------------------
                     n+aM                      n+aF
                  rho     * alphaM * tauM + rho     * alphaF * gamma * dt
    */
    facMtau = 1.0 / (densam_ * alphaM * tau_(1) + densaf_ * fldparatimint_->afgdt());

    /*
       factor for old subgrid velocities:

                 n+aM                      n+aF
       fac1 = rho     * alphaM * tauM + rho     * gamma * dt * (alphaF-1)
    */
    fac1 = (densam_ * alphaM * tau_(1) + densaf_ * gamma * dt * (alphaF - 1.0)) * facMtau;
    /*
      factor for old subgrid accelerations

                 n+aM
       fac2 = rho     * tauM * dt * (alphaM-gamma)
    */
    fac2 = (densam_ * dt * tau_(1) * (alphaM - gamma)) * facMtau;
    /*
      factor for residual in current subgrid velocities:

       fac3 = gamma * dt * tauM
    */
    fac3 = (gamma * dt * tau_(1)) * facMtau;

    // warning: time-dependent subgrid closure requires generalized-alpha time
    // integration
    if (!fldparatimint_->is_genalpha())
    {
      FOUR_C_THROW("the time-dependent subgrid closure requires a genalpha time integration\n");
    }

    /*         +-                                       -+
        ~n+1   |        ~n           ~ n            n+1  |
        u    = | fac1 * u  + fac2 * acc  -fac3 * res     |
         (i)   |                                    (i)  |
               +-                                       -+
    */

    /* compute the intermediate value of subscale velocity

            ~n+af            ~n+1                   ~n
            u     = alphaF * u     + (1.0-alphaF) * u
             (i)              (i)

    */

    static Core::LinAlg::Matrix<1, nsd_> sgvelintaf(Core::LinAlg::Initialization::zero);
    sgvelintaf.clear();
    for (int rr = 0; rr < nsd_; ++rr)
    {
      tds_->update_svelnp_in_one_direction(fac1, fac2, fac3, momres_old_(rr),
          fldparatimint_->alpha_f(), rr, iquad,
          sgvelint_(rr),  // sgvelint_ is set to sgvelintnp, but is then overwritten below anyway!
          sgvelintaf(rr));

      int pos = rr + nsd_ * iquad;

      /*
       *  ~n+1           ~n           ~ n            n+1
       *  u    =  fac1 * u  + fac2 * acc  -fac3 * res
       *   (i)
       *
       */

      svelnp[pos] = fac1 * sveln[pos] + fac2 * saccn[pos] - fac3 * momres_old_(rr);

      /* compute the intermediate value of subscale velocity
       *
       *          ~n+af            ~n+1                   ~n
       *          u     = alphaF * u     + (1.0-alphaF) * u
       *           (i)              (i)
       *
       */
      sgvelint_(rr) = alphaF * svelnp[pos] + (1.0 - alphaF) * sveln[pos];
    }
  }  // end time dependent subgrid scale closure

  //----------------------------------------------------------------------
  // include computed subgrid-scale velocity in convective term
  // -> only required for cross- and Reynolds-stress terms
  //----------------------------------------------------------------------
  if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none or
      fldpara_->reynolds() != Inpar::FLUID::reynolds_stress_stab_none or
      fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none or
      fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    sgconv_c_.multiply_tn(derxy_, sgvelint_);
  else
    sgconv_c_.clear();
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::lin_gal_mom_res_u(
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac)
{
  /*
      instationary                          cross-stress, part 1
       +-----+                             +-------------------+
       |     |                             |                   |

                 /       n+1       \        /      ~n+1       \
       rho*Du + |   rho*u   o nabla | Du + |   rho*u   o nabla | Du +
                 \      (i)        /        \      (i)        /

                 /                \  n+1
              + |   rho*Du o nabla | u      +  sigma*Du
                 \                /   (i)
                |                        |     |       |
                +------------------------+     +-------+
                        Newton                  reaction
  */

  int idim_nsd_p_idim[nsd_];

  for (int idim = 0; idim < nsd_; ++idim)
  {
    idim_nsd_p_idim[idim] = idim * nsd_ + idim;
  }

  if (fldparatimint_->is_stationary() == false)
  {
    const double fac_densam = fac_ * densam_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v = fac_densam * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  const double timefacfac_densaf = timefacfac * densaf_;

  // convection, reactive
  for (int ui = 0; ui < nen_; ++ui)
  {
    const double v = timefacfac_densaf * conv_c_(ui);

    for (int idim = 0; idim < nsd_; ++idim)
    {
      lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
    }
  }

  // convection, convective (only for Newton)
  if (fldpara_->is_newton())
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      const double temp = timefacfac_densaf * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int idim_nsd = idim * nsd_;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          lin_resM_Du(idim_nsd + jdim, ui) += temp * vderxy_(idim, jdim);
        }
      }
    }
  }

  if (fldpara_->reaction())
  {
    const double fac_reac = timefacfac * reacoeff_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v = fac_reac * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  if (fldpara_->cross() == Inpar::FLUID::cross_stress_stab)
  {
    // const double rhsresfac_densaf=rhsresfac*densaf_;
    for (int ui = 0; ui < nen_; ++ui)
    {
      // const double v=rhsresfac_densaf*sgconv_c_(ui);
      const double v = timefacfac_densaf * sgconv_c_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  return;
}



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::lin_gal_mom_res_u_subscales(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, Core::LinAlg::Matrix<nsd_, 1>& resM_Du,
    const double& timefacfac, const double& facMtau)
{
  // rescale Galerkin residual of all terms which have not been
  // integrated by parts

  const double C_saccGAL = densaf_ * fldparatimint_->afgdt() * facMtau;

  for (int ui = 0; ui < nen_; ++ui)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int idim_nsd = idim * nsd_;

      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        lin_resM_Du(idim_nsd + jdim, ui) *= C_saccGAL;
      }
    }
  }

  // include all contributions which have been integrated by parts
  // and thus can not be rescaled

  /* viscous term (intermediate) */
  /*  factor:
                                rhoaM*alphaM*tauM                 gamma*dt
          2*nu*alphaF*---------------------------------------,  * --------
                      rhoaM*alphaM*tauM+rhoaf*alphaF*gamma*dt      alphaM


             /                         \
            |               /    \      |
            |  nabla o eps | Dacc | , v |
            |               \    /      |
             \                         /

  */

  if (is_higher_order_ele_)
  {
    const double v = 2.0 * visceff_ * timefacfac * (1.0 - C_saccGAL);
    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int nsd_idim = nsd_ * idim;

      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        const int nsd_idim_p_jdim = nsd_idim + jdim;

        for (int ui = 0; ui < nen_; ++ui)
        {
          lin_resM_Du(nsd_idim_p_jdim, ui) += v * viscs2_(nsd_idim_p_jdim, ui);
        }
      }
    }
  }

  /*  factor:
                              rhoaM*alphaM*tauM                gamma*dt
          alphaF * ---------------------------------------,  * --------
                   rhoaM*alphaM*tauM+rhoaF*alphaF*gamma*dt      alphaM

                       /               \
                      |                 |
                      |  nabla Dp ,  v  |
                      |                 |
                       \               /
  */
  for (int ui = 0; ui < nen_; ++ui)
  {
    const double v = (1.0 - C_saccGAL) * fac_ * fldparatimint_->time_fac_pre();
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int fvi = nsd_ * vi;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_p_v(fvi + idim, ui) -= v * derxy_(idim, ui) * funct_(vi);
      }
    }
  }

  /*  factor: +1

           /                       \
          |     n+am    ~ n+am      |
          |  rho     * acc     , v  |
          |               (i)       |
           \                       /


         using
                                  n+af             /
           n+am    ~ n+am      rho        ~n+af   |    n+am      n+am
        rho     * acc     = - --------- * u     - | rho     * acc     +
                     (i)           n+af    (i)    |              (i)
                               tau_M               \

                                  n+af    / n+af        \   n+af            n+1
                             + rho     * | c     o nabla | u     + nabla o p    -
                                          \ (i)         /   (i)             (i)

                                                        / n+af \
                             - 2 * mu * grad o epsilon | u      | -
                                                        \ (i)  /
                                               \
                                  n+af    n+af  |
                             - rho     * f      |
                                                |
                                               /
  */

  /*
  For time-dependent subgrid scales closure, we take the implementation
  by Peter Gamnitzer, reading as documented below. Note that certain terms cancel out,
  such as those containing acc_(i)^n+am and the convective term!
  */

  //---------------------------------------------------------------
  //
  //      GALERKIN PART AND SUBSCALE ACCELERATION STABILISATION
  //
  //---------------------------------------------------------------
  /*  factor: +1

         /             \     /                     \
        |   ~ n+am      |   |     n+am    n+af      |
        |  acc     , v  | + |  acc     - f     , v  |
        |     (i)       |   |     (i)               |
         \             /     \                   /


       using
                                                  /
                  ~ n+am        1.0      ~n+af   |    n+am
                 acc     = - --------- * u     - | acc     +
                    (i)           n+af    (i)    |    (i)
                             tau_M                \

                              / n+af        \   n+af            n+1
                           + | c     o nabla | u     + nabla o p    -
                              \ (i)         /   (i)             (i)

                                                      / n+af \
                           - 2 * nu * grad o epsilon | u      | -
                                                      \ (i)  /
                                   \
                              n+af  |
                           - f      |
                                    |
                                   /

  */


  for (int idim = 0; idim < nsd_; ++idim)
  {
    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)
      // see implementation by Peter Gamnitzer
      resM_Du(idim) += (timefacfac / fldparatimint_->alpha_f()) *
                       (-densaf_ * sgvelint_(idim) / tau_(0) - gradp_(idim) +
                           2 * visceff_ * visc_old_(idim));  //-momres_old_(idim));
    else
      resM_Du(idim) += fac_ * (-densaf_ * sgvelint_(idim) / tau_(1) - momres_old_(idim));
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::stab_lin_gal_mom_res_u(
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac)
{
  /*
                 /       n+1       \        /                \  n+1
       rho*Du + |   rho*u   o nabla | Du + |   rho*Du o nabla | u   +
                 \      (i)        /        \                /   (i)

                               /  \
     + sigma*Du + nabla o eps | Du |
                               \  /
  */
  if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent ||
      fldpara_->cross() == Inpar::FLUID::cross_stress_stab)
  {
    //----------------------------------------------------------------------
    /* GALERKIN residual was rescaled and cannot be reused; so rebuild it */

    lin_resM_Du.clear();

    int idim_nsd_p_idim[nsd_];

    for (int idim = 0; idim < nsd_; ++idim)
    {
      idim_nsd_p_idim[idim] = idim * nsd_ + idim;
    }

    if (fldparatimint_->is_stationary() == false)
    {
      const double fac_densam = fac_ * densam_;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const double v = fac_densam * funct_(ui);

        for (int idim = 0; idim < nsd_; ++idim)
        {
          lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
        }
      }
    }

    const double timefacfac_densaf = timefacfac * densaf_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      // deleted +sgconv_c_(ui)
      const double v = timefacfac_densaf * conv_c_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }

    if (fldpara_->is_newton())
    {
      //
      //
      // dr_j   d    /    du_j \          du_j         dN_B
      // ----= ---- | u_i*----  | = N_B * ---- + u_i * ---- * d_jk
      // du_k  du_k  \    dx_i /          dx_k         dx_i

      for (int ui = 0; ui < nen_; ++ui)
      {
        const double temp = timefacfac_densaf * funct_(ui);

        for (int idim = 0; idim < nsd_; ++idim)
        {
          const int idim_nsd = idim * nsd_;

          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            lin_resM_Du(idim_nsd + jdim, ui) += temp * vderxy_(idim, jdim);
          }
        }
      }
    }

    if (fldpara_->reaction())
    {
      const double fac_reac = timefacfac * reacoeff_;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const double v = fac_reac * funct_(ui);

        for (int idim = 0; idim < nsd_; ++idim)
        {
          lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
        }
      }
    }
  }

  if (is_higher_order_ele_)
  {
    const double v = -2.0 * visceff_ * timefacfac;
    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int nsd_idim = nsd_ * idim;

      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        const int nsd_idim_p_jdim = nsd_idim + jdim;

        for (int ui = 0; ui < nen_; ++ui)
        {
          lin_resM_Du(nsd_idim_p_jdim, ui) += v * viscs2_(nsd_idim_p_jdim, ui);
        }
      }
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::inertia_convection_reaction_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, Core::LinAlg::Matrix<nsd_, 1>& resM_Du,
    const double& rhsfac)
{
  /* inertia (contribution to mass matrix) if not is_stationary */
  /*
            /              \
           |                |
           |    rho*Du , v  |
           |                |
            \              /
  */
  /* convection, convective part (convective form) */
  /*
            /                             \
           |  /       n+1       \          |
           | |   rho*u   o nabla | Du , v  |
           |  \      (i)        /          |
            \                             /
  */
  /*  convection, reactive part (convective form)
            /                               \
           |  /                \   n+1       |
           | |  rho*Du o nabla  | u     , v  |
           |  \                /   (i)       |
            \                               /
  */
  /*  reaction */
  /*
            /                \
           |                  |
           |    sigma*Du , v  |
           |                  |
            \                /
  */
  if ((fldpara_->is_newton() or
          (is_higher_order_ele_ and fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)))
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          for (int idim = 0; idim < nsd_; ++idim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                funct_(vi) * lin_resM_Du(idim * nsd_ + jdim, ui);
          }  // end for (idim)
        }  // end for (jdim)
      }  // end for (vi)
    }  // end for (ui)
  }
  else
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_ * vi + idim, nsd_ * ui + idim) +=
              funct_(vi) * lin_resM_Du(idim * nsd_ + idim, ui);
        }  // end for (idim)
      }  // vi
    }  // ui
  }

  // inertia terms on the right hand side for instationary fluids
  if (not fldparatimint_->is_stationary())
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      if (fldparatimint_->is_genalpha())
      {
        if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
            fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
        {
          ;  // do nothing here! Whole term already set in lin_gal_mom_res_u_subscales()
        }
        else
          resM_Du(idim) += rhsfac * densam_ * accint_(idim);
      }
      else
        resM_Du(idim) += fac_ * densaf_ * velint_(idim);
    }
  }  // end if (not stationary)

  // convective terms of rhs
  for (int idim = 0; idim < nsd_; ++idim)
  {
    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
        fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
    {
      ;  // do nothing here! Whole term already set in lin_gal_mom_res_u_subscales()
    }
    else
      resM_Du(idim) += rhsfac * densaf_ * conv_old_(idim);
  }  // end for(idim)


  if (fldpara_->reaction())
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      resM_Du(idim) += rhsfac * reacoeff_ * velint_(idim);
    }
  }  // end if (reaction_)

  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      velforce(idim, vi) -= resM_Du(idim) * funct_(vi);
    }
  }
  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::viscous_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, Core::LinAlg::Matrix<nsd_, nsd_>& viscstress,
    const double& timefacfac, const double& rhsfac)
{
  const double visceff_timefacfac = visceff_ * timefacfac;

  /* viscosity term */
  /*
                   /                        \
                  |       /  \         / \   |
            2 mu  |  eps | Du | , eps | v |  |
                  |       \  /         \ /   |
                   \                        /
  */

  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const double temp = visceff_timefacfac * derxy_(jdim, vi);

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) += temp * derxy_(idim, ui);
        }
      }
    }
  }

  static Core::LinAlg::Matrix<nen_, nen_> tmp_dyad;
  tmp_dyad.multiply_tn(derxy_, derxy_);
  tmp_dyad.scale(visceff_timefacfac);

  for (int ui = 0; ui < nen_; ++ui)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const double tmp_val = tmp_dyad(vi, ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_u(nsd_ * vi + idim, nsd_ * ui + idim) += tmp_val;
      }  // end for (idim)
    }  // ui
  }  // vi


  const double v = visceff_ * rhsfac;

  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      viscstress(idim, jdim) = v * (vderxy_(jdim, idim) + vderxy_(idim, jdim));
    }
  }

  static Core::LinAlg::Matrix<nsd_, nen_> tmp;
  tmp.multiply(viscstress, derxy_);
  velforce.update(-1.0, tmp, 1.0);


  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::cont_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefac, const double& timefacfac,
    const double& timefacfacpre, const double& rhsfac)
{
  // In the case no continuity stabilization and no LOMA:
  // the factors 'conti_stab_and_vol_visc_fac' and 'conti_stab_and_vol_visc_rhs' are zero
  // therefore there is no contribution to the element stiffness matrix and
  // the viscous stress tensor is NOT altered!!
  //
  // ONLY
  // the rhs contribution of the viscous term is added!!

  double conti_stab_and_vol_visc_fac = 0.0;
  double conti_stab_and_vol_visc_rhs = 0.0;

  if (fldpara_->c_stab())
  {
    conti_stab_and_vol_visc_fac += timefacfacpre * tau_(2);
    conti_stab_and_vol_visc_rhs -= rhsfac * tau_(2) * conres_old_;
  }
  if (fldpara_->physical_type() == Inpar::FLUID::loma or
      fldpara_->physical_type() == Inpar::FLUID::weakly_compressible or
      fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
  {
    conti_stab_and_vol_visc_fac -= (2.0 / 3.0) * visceff_ * timefacfac;
    conti_stab_and_vol_visc_rhs += (2.0 / 3.0) * rhsfac * visceff_ * vdiv_;
    // additional term q_sq_ for dynamics Smagorisnky only
    if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      conti_stab_and_vol_visc_rhs += (2.0 / 3.0) * rhsfac * q_sq_;
  }

  /* continuity stabilisation on left-hand side */
  /*
              /                        \
             |                          |
        tauC | nabla o Du  , nabla o v  |
             |                          |
              \                        /
  */
  /* viscosity term - subtraction for low-Mach-number flow */
  /*
             /                             \             /                        \
            |  1                      / \   |     2 mu  |                          |
     - 2 mu |  - (nabla o u) I , eps | v |  | = - ----- | nabla o Du  , nabla o v  |
            |  3                      \ /   |       3   |                          |
             \                             /             \                        /
  */
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int fui = nsd_ * ui;

    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int fui_p_idim = fui + idim;
      const double v0 = conti_stab_and_vol_visc_fac * derxy_(idim, ui);
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = nsd_ * vi;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          estif_u(fvi + jdim, fui_p_idim) += v0 * derxy_(jdim, vi);
        }
      }
    }  // end for(idim)
  }

  // computation of right-hand-side viscosity term
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      /* viscosity term on right-hand side */
      velforce(idim, vi) += conti_stab_and_vol_visc_rhs * derxy_(idim, vi);
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pressure_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac,
    const double& press)
{
  for (int ui = 0; ui < nen_; ++ui)
  {
    const double v = -timefacfacpre * funct_(ui);
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int fvi = nsd_ * vi;
      /* pressure term */
      /*
           /                \
          |                  |
          |  Dp , nabla o v  |
          |                  |
           \                /
      */
      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_p_v(fvi + idim, ui) += v * derxy_(idim, vi);
      }
    }
  }

  // pressure term on right-hand side
  velforce.update(press * rhsfac, derxy_, 1.0);

  return;
}



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::continuity_gal_part(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, 1>& preforce,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac)
{
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = timefacfacpre * funct_(vi);
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int fui = nsd_ * ui;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        /* continuity term */
        /*
             /                \
            |                  |
            | nabla o Du  , q  |
            |                  |
             \                /
        */
        estif_q_u(vi, fui + idim) += v * derxy_(idim, ui);
      }
    }
  }  // end for(idim)

  // continuity term on right-hand side
  preforce.update(-rhsfac * vdiv_, funct_, 1.0);

  return;
}


/*----------------------------------------------------------------------------*
 |  add integration point contribution to pressure matrices         nis Jan13 |
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pressure_projection(
    Core::LinAlg::Matrix<nen_, nen_>& ppmat)
{
  // mass matrix of pressure basis functions - used as temp
  ppmat.multiply_nt(fac_, funct_, funct_, 1.0);

  // "mass matrix" of projection modes - here element volume_equivalent_diameter_pc
  D_ += fac_;

  // prolongator(?) of projection
  E_.update(fac_, funct_, 1.0);
}

/*----------------------------------------------------------------------------*
 |  finalize pressure projection and compute rhs-contribution       nis Jan13 |
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pressure_projection_finalize(
    Core::LinAlg::Matrix<nen_, nen_>& ppmat, Core::LinAlg::Matrix<nen_, 1>& preforce,
    const Core::LinAlg::Matrix<nen_, 1>& epre)
{
  // check whether we deal with linear elements
  if (not(distype == Core::FE::CellType::hex8 or distype == Core::FE::CellType::tet4 or
          distype == Core::FE::CellType::quad4 or distype == Core::FE::CellType::tri3))
  {
    FOUR_C_THROW("Polynomial pressure projection only implemented for linear elements so far.");
  }

  // compute difference of consistent and projection pressure mass matrices
  ppmat.multiply_nt(-1.0 / D_, E_, E_, 1.0);
  ppmat.scale(1.0 / visc_);

  // compute rhs-contribution
  static Core::LinAlg::Matrix<nen_, 1> temp(Core::LinAlg::Initialization::uninitialized);
  temp.multiply(ppmat, epre);
  preforce.update(-fldparatimint_->time_fac_rhs(), temp, 1.0);

  // scale pressure-pressure matrix with pressure time factor
  ppmat.scale(fldparatimint_->time_fac_pre());
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::body_force_rhs_term(
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& rhsfac)
{
  for (int idim = 0; idim < nsd_; ++idim)
  {
    const double scaled_rhsmom = rhsfac * rhsmom_(idim);

    for (int vi = 0; vi < nen_; ++vi)
    {
      velforce(idim, vi) += scaled_rhsmom * funct_(vi);
    }
  }  // end for(idim)

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::conservative_formulation(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefacfac, const double& rhsfac)
{
  //----------------------------------------------------------------------
  // computation of additions to convection term (convective and
  // reactive part) for conservative form of convection term including
  // right-hand-side contribution
  //----------------------------------------------------------------------

  /* convection, convective part (conservative addition) */
  /*
    /                                                \
    |      /              n+1    n+1           \      |
    |  Du | rho*nabla o u    +  u   *nabla rho | , v  |
    |      \             (i)     (i)          /       |
    \                                                 /
  */

  for (int idim = 0; idim < nsd_; ++idim)
  {
    // left hand side
    {
      // compute prefactor
      double v = timefacfac * densaf_ * vdiv_;
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
        v -= timefacfac * densaf_ * scaconvfacaf_ * conv_scaaf_;
      else if (fldpara_->physical_type() == Inpar::FLUID::varying_density)
      {
        v += timefacfac * conv_scaaf_;
        //         o
        // (v, Du rho)
        /*{
          // interpolation to GP
          double densdtngp = densdtn.Dot(funct_);
          v += timefacfac*densdtngp;
        }*/
      }

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int fui = nsd_ * ui + idim;
        const double v1 = v * funct_(ui);

        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = nsd_ * vi + idim;
          estif_u(fvi, fui) += funct_(vi) * v1;
        }
      }

      /*  convection, reactive part (conservative addition) */
      /*
        /                              \
        |  n+1  /               \      |
        | u    | rho*nabla o Du | , v  |
        |  (i)  \              /       |
        \                             /
      */

      if (fldpara_->is_newton())
      {
        const double v_idim = timefacfac * densaf_ * velint_(idim);
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = nsd_ * vi + idim;
          const double v1_idim = v_idim * funct_(vi);

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = nsd_ * ui;

            for (int jdim = 0; jdim < nsd_; ++jdim)
              estif_u(fvi, fui + jdim) += v1_idim * derxy_(jdim, ui);
          }
        }

        /*  convection, reactive part (conservative addition) */
        /*
         /                           \
         |  n+1  /             \      |
         | u    | Du*nabla rho | , v  |
         |  (i)  \            /       |
         \                           /
        */
        if (fldpara_->physical_type() == Inpar::FLUID::loma or
            fldpara_->physical_type() == Inpar::FLUID::varying_density)
        {
          double v_idim = 0.0;
          if (fldpara_->physical_type() == Inpar::FLUID::loma)
            v_idim = -timefacfac * densaf_ * scaconvfacaf_ * grad_scaaf_(idim) * velint_(idim);
          else if (fldpara_->physical_type() == Inpar::FLUID::varying_density)
            v_idim = +timefacfac * grad_scaaf_(idim) * velint_(idim);

          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = nsd_ * vi + idim;
            const double v1_idim = v_idim * funct_(vi);

            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = nsd_ * ui;

              for (int jdim = 0; jdim < nsd_; ++jdim)
                estif_u(fvi, fui + jdim) += v1_idim * funct_(ui);
            }
          }
        }
      }
    }

    // right hand side
    {
      /* convection (conservative addition) on right-hand side */
      double v = -rhsfac * densaf_ * velint_(idim) * vdiv_;

      if (fldpara_->physical_type() == Inpar::FLUID::loma)
        v += rhsfac * velint_(idim) * densaf_ * scaconvfacaf_ * conv_scaaf_;
      else if (fldpara_->physical_type() == Inpar::FLUID::varying_density)
        v -= rhsfac * velint_(idim) * conv_scaaf_;

      for (int vi = 0; vi < nen_; ++vi) velforce(idim, vi) += v * funct_(vi);
    }
  }  // end for(idim)

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pspg(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, nen_>& ppmat,
    Core::LinAlg::Matrix<nen_, 1>& preforce, Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const double& fac3, const double& timefacfac, const double& timefacfacpre, const double& rhsfac,
    const int iquad)
{
  // conservative, stabilization terms are neglected (Hughes)

  /* pressure stabilisation:                                            */
  /*
              /                 \
             |  ~n+af            |
           - |  u     , nabla q  |
             |                   |
              \                 /
  */

  double scal_grad_q = 0.0;

  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    scal_grad_q = tau_(1);
  }
  else  // time-dependent subgrid-scales
  {
    scal_grad_q = fac3;
  }

  /* pressure stabilisation: inertia if not stationary*/
  /*
            /                  \
           |                    |
           |  rho*Du , nabla q  |
           |                    |
            \                  /
  */
  /* pressure stabilisation: convection, convective part */
  /*
            /                                   \
           |  /       n+1       \                |
           | |   rho*u   o nabla | Du , nabla q  |
           |  \      (i)        /                |
            \                                   /
  */
  /* pressure stabilisation: convection, reactive part if Newton */
  /*
            /                                   \
           |  /                \   n+1           |
           | |   rho*Du o nabla | u     , grad q |
           |  \                /   (i)           |
            \                                   /
  */
  /* pressure stabilisation: reaction if included */
  /*
            /                     \
           |                      |
           |  sigma*Du , nabla q  |
           |                      |
            \                    /
  */
  /* pressure stabilisation: viscosity (-L_visc_u) */
  /*
            /                              \
           |               /  \             |
       mu  |  nabla o eps | Du | , nabla q  |
           |               \  /             |
            \                              /
  */

  if (is_higher_order_ele_ || fldpara_->is_newton())
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const double temp_vi_idim = derxy_(idim, vi) * scal_grad_q;
        for (int ui = 0; ui < nen_; ++ui)
        {
          const int nsd_ui = nsd_ * ui;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_q_u(vi, nsd_ui + jdim) += lin_resM_Du(nsd_ * idim + jdim, ui) * temp_vi_idim;
          }  // jdim
        }  // ui
      }  // idim
    }  // vi
  }  // end if (is_higher_order_ele_) or (newton_)
  else
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_q_u(vi, nsd_ * ui + idim) +=
              lin_resM_Du(nsd_ * idim + idim, ui) * derxy_(idim, vi) * scal_grad_q;
        }  // vi
      }  // ui
    }  // idim
  }  // end if not (is_higher_order_ele_) nor (newton_)


  for (int ui = 0; ui < nen_; ++ui)
  {
    /* pressure stabilisation: pressure( L_pres_p) */
    /*
         /                    \
        |                      |
        |  nabla Dp , nabla q  |
        |                      |
         \                    /
    */
    for (int vi = 0; vi < nen_; ++vi)
    {
      double sum = 0.;
      for (int idim = 0; idim < nsd_; ++idim) sum += derxy_(idim, ui) * derxy_(idim, vi);

      ppmat(vi, ui) += timefacfacpre * scal_grad_q * sum;
    }  // vi
  }  // ui

  for (int idim = 0; idim < nsd_; ++idim)
  {
    double sgvel = 0.0;
    if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    {
      sgvel = sgvelint_(idim);
    }
    else  // time-dependent subgrid-scales, Np_Genal_Alpha!
    {
      sgvel = (tds_->svelnp())(idim, iquad);
    }
    const double temp = rhsfac * sgvel;

    for (int vi = 0; vi < nen_; ++vi)
    {
      // pressure stabilisation
      preforce(vi) += temp * derxy_(idim, vi);
    }
  }  // end for(idim)

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::supg(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nen_, 1>& preforce, Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const double& fac3, const double& timefacfac, const double& timefacfacpre, const double& rhsfac)
{
  /*
                    /                                \
                   |  ~n+af    /     n+af       \     |
                 - |  u     , | rho*u    o nabla | v  |
                   |           \     (i)        /     |
                    \                                /
   */

  static Core::LinAlg::Matrix<nsd_, 1> temp;

  double supgfac;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    supgfac = densaf_ * tau_(0);
  else
    supgfac = densaf_ * fldparatimint_->alpha_f() * fac3;

  static Core::LinAlg::Matrix<nen_, 1> supg_test;
  for (int vi = 0; vi < nen_; ++vi)
  {
    supg_test(vi) = supgfac * conv_c_(vi);
  }

  if (fldpara_->reynolds() == Inpar::FLUID::reynolds_stress_stab)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      supg_test(vi) += supgfac * sgconv_c_(vi);
    }
  }

  /* supg stabilisation: inertia if not stationary */
  /*
         /                                \
        |            /     n+1       \     |
        |  rho*Du , | rho*u   o nabla | v  |
        |            \     (i)       /     |
         \                                /
  */
  /* supg stabilisation: convective part ( L_conv_u) , convective term */
  /*
         /                                                     \
        |    /       n+1        \        /      n+1       \     |
        |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
        |    \       (i)        /        \      (i)       /     |
         \                                                     /
  */
  /* supg stabilisation: convective part ( L_conv_u) , reactive term if Newton */
  /*
         /                                                     \
        |    /       n+1        \        /     n+1        \     |
        |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
        |    \       (i)        /        \     (i)        /     |
         \                                                     /
  */
  /* supg stabilisation: reaction if included */
  /*
         /                                  \
        |              /     n+1       \     |
        |  sigma*Du , | rho*u   o nabla | v  |
        |              \     (i)       /     |
         \                                  /
  */
  /* supg stabilisation: viscous part  (-L_visc_u) if is_higher_order_ele_ */
  /*
         /                                              \
        |               /  \    /       n+1        \     |
        |  nabla o eps | Du |, |   rho*u    o nabla | v  |
        |               \  /    \       (i)        /     |
         \                                              /
  */

  /* supg stabilisation: inertia, linearisation of testfunction if Newton */
  /*
              /                                       \
             |         n+1       /              \      |
             |    rho*u      ,  | rho*Du o nabla | v   |
             |         (i)       \              /      |
              \                                       /
  */
  /* supg stabilisation: reactive part of convection and linearisation of testfunction ( L_conv_u)
   * if Newton */
  /*
              /                                                       \
             |    /       n+1        \   n+1     /              \      |
             |   |   rho*u    o nabla | u    ,  | rho*Du o nabla | v   |
             |    \       (i)        /   (i)     \              /      |
              \                                                       /
  */
  /* supg stabilisation: reaction, linearisation of testfunction if Newton */
  /*
              /                                         \
             |           n+1       /              \      |
             |    sigma*u      ,  | rho*Du o nabla | v   |
             |           (i)       \              /      |
              \                                         /
  */
  /* supg stabilisation: pressure part, linearisation of test function if Newton ( L_pres_p) */
  /*
             /                                     \
            |         n+1    /                \     |
            |  nabla p    , |   rho*Du o nabla | v  |
            |         (i)    \                /     |
             \                                     /
  */
  /* supg stabilisation: viscous part, linearisation of test function if Newton (-L_visc_u) */
  /*
             /                                               \
            |               / n+1 \    /               \      |
            |  nabla o eps | u     |, |  rho*Du o nabla | v   |
            |               \ (i) /    \               /      |
             \                                               /
  */
  /* supg stabilisation: bodyforce part, linearisation of test function if Newton */
  /*
             /                                      \
            |                  /               \     |
            |  rho*rhsint   , |  rho*Du o nabla | v  |
            |                  \               /     |
             \                                      /
  */
  if (fldpara_->is_newton())
  {
    if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    {
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        temp(jdim) = timefacfac * supgfac * momres_old_(jdim);
      }
    }
    else
    {
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        temp(jdim) = -timefacfac * densaf_ * sgvelint_(jdim);
      }
    }
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          const double w = temp(idim) * funct_(ui);
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                lin_resM_Du(nsd_ * idim + jdim, ui) * supg_test(vi) + derxy_(jdim, vi) * w;
          }  // jdim
        }  // vi
      }  // ui
    }  // idim
  }  // end if (fldpara_->IsNewton())
  else if (is_higher_order_ele_)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                lin_resM_Du(nsd_ * idim + jdim, ui) * supg_test(vi);
          }
        }
      }
    }
  }  // end if (is_higher_order_ele_)
  else
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const int nsd_vi = nsd_ * vi;
      for (int ui = 0; ui < nen_; ++ui)
      {
        const int nsd_ui = nsd_ * ui;
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_vi + idim, nsd_ui + idim) +=
              lin_resM_Du(nsd_ * idim + idim, ui) * supg_test(vi);
        }  // ui
      }  // idim
    }  // vi
  }  // end if not (is_higher_order_ele_) nor (newton_)

  /* supg stabilisation: pressure part  ( L_pres_p) */
  /*
           /                                    \
          |              /       n+1       \     |
          |  nabla Dp , |   rho*u   o nabla | v  |
          |              \       (i)       /     |
           \                                    /
  */
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = timefacfacpre * supg_test(vi);
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_p_v(nsd_ * vi + idim, ui) += v * derxy_(idim, ui);
      }
    }
  }  // end for(idim)

  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      temp(jdim) = rhsfac * momres_old_(jdim);
    }
  }
  else
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      temp(jdim) = -rhsfac * densaf_ * sgvelint_(jdim) / (fac3 * fldparatimint_->alpha_f());
    }
  }

  for (int idim = 0; idim < nsd_; ++idim)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      // supg stabilisation
      velforce(idim, vi) -= temp(idim) * supg_test(vi);
    }
  }  // end for(idim)

  // SUPG and Reynolds-stress term on right-hand side of
  // continuity equation for low-Mach-number flow
  if (fldpara_->physical_type() == Inpar::FLUID::loma and
      (fldpara_->conti_supg() or
          fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none))
  {
    const double temp_supg = rhsfac * scaconvfacaf_ * sgscaint_;

    if (fldpara_->conti_supg())
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        preforce(vi) -= temp_supg * conv_c_(vi);
      }
    }

    if (fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        preforce(vi) -= temp_supg * sgconv_c_(vi);
      }
    }
  }  // loma

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::reac_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac,
    const double& timefacfacpre, const double& rhsfac, const double& fac3)
{
  double reac_tau;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    reac_tau = fldpara_->visc_rea_stab_fac() * reacoeff_ * tau_(1);
  else
  {
    FOUR_C_THROW("Is this factor correct? Check for bugs!");
    reac_tau = fldpara_->visc_rea_stab_fac() * reacoeff_ * fldparatimint_->alpha_f() * fac3;
  }


  /* reactive stabilisation, inertia part if not stationary */
  /*
               /                    \
              |                      |
          -/+ |    rho*Du , sigma*v  |
              |                      |
               \                    /
  */
  /* reactive stabilisation, convective part, convective type */
  /*
             /                                  \
            |  /       n+1       \               |
        -/+ | |   rho*u   o nabla | Du , sigma*v |
            |  \       (i)       /               |
             \                                  /
  */
  /* reactive stabilisation, reactive part of convection */
  /*
             /                                   \
            |  /                \   n+1           |
        -/+ | |   rho*Du o nabla | u    , sigma*v |
            |  \                /   (i)           |
             \                                   /
  */
  /* reactive stabilisation, reaction part if included */
  /*
               /                      \
              |                        |
          -/+ |    sigma*Du , sigma*v  |
              |                        |
               \                      /
  */
  /* reactive stabilisation, viscous part (-L_visc_u) */
  /*
             /                             \
            |               /  \            |
       +/-  |  nabla o eps | Du | , sigma*v |
            |               \  /            |
             \                             /
  */
  if (is_higher_order_ele_ or fldpara_->is_newton())
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const double v = reac_tau * funct_(vi);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = nsd_ * idim;

        const int fvi_p_idim = nsd_ * vi + idim;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          const int nsd_idim_p_jdim = nsd_idim + jdim;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui_p_jdim = nsd_ * ui + jdim;

            estif_u(fvi_p_idim, fui_p_jdim) += v * lin_resM_Du(nsd_idim_p_jdim, ui);
          }  // jdim
        }  // vi
      }  // ui
    }  // idim
  }  // end if (is_higher_order_ele_) or (newton_)
  else
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const double v = reac_tau * funct_(vi);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int fvi_p_idim = nsd_ * vi + idim;

        const int nsd_idim = nsd_ * idim;

        for (int ui = 0; ui < nen_; ++ui)
        {
          const int fui_p_idim = nsd_ * ui + idim;

          estif_u(fvi_p_idim, fui_p_idim) += v * lin_resM_Du(nsd_idim + idim, ui);
        }  // ui
      }  // idim
    }  // vi
  }  // end if not (is_higher_order_ele_) nor (newton_)


  /* reactive stabilisation, pressure part ( L_pres_p) */
  /*
             /                    \
            |                      |
       -/+  |  nabla Dp , sigma*v  |
            |                      |
             \                    /
  */
  const double reac_tau_timefacfacpre = reac_tau * timefacfacpre;
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = reac_tau_timefacfacpre * funct_(vi);

    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int fvi = nsd_ * vi + idim;

      for (int ui = 0; ui < nen_; ++ui)
      {
        estif_p_v(fvi, ui) += v * derxy_(idim, ui);
      }
    }
  }  // end for(idim)

  const double reac_fac = fldpara_->visc_rea_stab_fac() * rhsfac * reacoeff_;
  for (int idim = 0; idim < nsd_; ++idim)
  {
    const double v = reac_fac * sgvelint_(idim);

    for (int vi = 0; vi < nen_; ++vi)
    {
      velforce(idim, vi) += v * funct_(vi);
    }
  }  // end for(idim)

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::visc_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac,
    const double& timefacfacpre, const double& rhsfac, const double& fac3)
{
  // preliminary parameter computation
  double two_visc_tau;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    two_visc_tau = -fldpara_->visc_rea_stab_fac() * 2.0 * visc_ * tau_(1);
  else
    two_visc_tau = -fldpara_->visc_rea_stab_fac() * 2.0 * visc_ * fldparatimint_->alpha_f() * fac3;

  /* viscous stabilisation, inertia part if not stationary */
  /*
                /                        \
               |                          |
           +/- |    rho*Du , div eps (v)  |
               |                          |
                \                        /
  */
  /* viscous stabilisation, convective part, convective type */
  /*
              /                                      \
             |  /       n+1       \                   |
         +/- | |   rho*u   o nabla | Du , div eps (v) |
             |  \       (i)       /                   |
              \                                      /
  */
  /* viscous stabilisation, reactive part of convection */
  /*
              /                                       \
             |  /                \   n+1               |
         +/- | |   rho*Du o nabla | u    , div eps (v) |
             |  \                /   (i)               |
              \                                       /
  */
  /* viscous stabilisation, reaction part if included */
  /*
                /                          \
               |                            |
           +/- |    sigma*Du , div eps (v)  |
               |                            |
                \                          /
  */
  /* viscous stabilisation, viscous part (-L_visc_u) */
  /*
              /                                 \
             |               /  \                |
        -/+  |  nabla o eps | Du | , div eps (v) |
             |               \  /                |
              \                                 /
  */
  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int fui_p_jdim = nsd_ * ui + jdim;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        for (int kdim = 0; kdim < nsd_; ++kdim)
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi_p_idim = nsd_ * vi + idim;

            estif_u(fvi_p_idim, fui_p_jdim) += two_visc_tau * lin_resM_Du(nsd_ * kdim + jdim, ui) *
                                               viscs2_(nsd_ * idim + kdim, vi);
          }  // vi
        }  // kdim
      }  // idim
    }  // ui
  }  // jdim


  /* viscous stabilisation, pressure part ( L_pres_p) */
  /*
              /                        \
             |                          |
        +/-  |  nabla Dp , div eps (v)  |
             |                          |
              \                        /
  */
  const double two_visc_tau_timefacfacpre = two_visc_tau * timefacfacpre;
  for (int idim = 0; idim < nsd_; ++idim)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          estif_p_v(vi * nsd_ + idim, ui) +=
              two_visc_tau_timefacfacpre * derxy_(jdim, ui) * viscs2_(jdim + (idim * nsd_), vi);
        }
      }
    }
  }  // end for(idim)

  // viscous stabilization term on right-hand side
  const double two_visc_fac = -fldpara_->visc_rea_stab_fac() * rhsfac * 2.0 * visc_;
  for (int idim = 0; idim < nsd_; ++idim)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* viscous stabilisation */
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        velforce(idim, vi) += two_visc_fac * sgvelint_(jdim) * viscs2_(jdim + (idim * nsd_), vi);
      }
    }
  }  // end for(idim)

  return;
}



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::conv_div_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefacfac, const double& rhsfac)
{
  /* additional convective stabilization, when continuity is not satisfied*/
  /*
              /                           \
          1  |                             |
      +  --- |  (nabla o u)  Du  , v       |
          2  |                             |
              \                           /
  */


  // compute divergence of u
  double divergence_timefacfac = 0.5 * (vderxy_(0, 0) + vderxy_(1, 1) + vderxy_(2, 2)) * timefacfac;
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int ijdim = 0; ijdim < nsd_; ijdim++)
      {
        const int fui_ijdim = nsd_ * ui + ijdim;
        const int fvi_ijdim = nsd_ * vi + ijdim;

        estif_u(fvi_ijdim, fui_ijdim) += divergence_timefacfac * funct_(vi) * funct_(ui);
      }
    }
  }


  for (int idim = 0; idim < nsd_; ++idim)
  {
    const double rhs_divergencefac = divergence_timefacfac * velint_(idim);

    for (int vi = 0; vi < nen_; ++vi)
    {
      velforce(idim, vi) -= rhs_divergencefac * funct_(vi);
    }
  }  // end for(idim)


  return;
}



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::cross_stress_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac,
    const double& timefacfacpre, const double& rhsfac, const double& fac3)
{
  /*
                               this part is linearised in
                              combination with the standard
                                  Galerkin term above
                                          +----+
                                          |    |
                    /                                \
                   |   /    ~n+af       \   n+af      |
                 + |  | rho*u    o nabla | u     , v  |
                   |   \     (i)        /   (i)       |
                    \                                /
                        |       |
                        +-------+
                     linearisation of
                  this part is performed
                     in the following

   */

  double crossfac;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    crossfac = densaf_ * tau_(1);
  else
    crossfac = densaf_ * fldparatimint_->alpha_f() * fac3;

  // Stabilization of lhs and the rhs
  if (fldpara_->cross() == Inpar::FLUID::cross_stress_stab and fldpara_->is_newton())
  {
    /*
           /                         \
          |  /          \   n+af      |
          | | Du o nabla | u     , v  |
          |  \          /             |
           \                         /
    */
    /*
           /                                              \
          |  / / /          \   n+af \         \   n+af    |
          | | | | Du o nabla | u      | o nabla | u   , v  |
          |  \ \ \          /        /         /           |
           \                                              /
    */
    /*
           /                                               \
          |  / / / n+af        \     \         \   n+af     |
          | | | | u     o nabla | Du  | o nabla | u    , v  |
          |  \ \ \             /     /         /            |
           \                                               /
    */
    /*
           /                               \
          |  /                \   n+af      |
          | | sigma*Du o nabla | u     , v  |
          |  \                /             |
           \                               /
    */
    /*
           /                                             \
          |  / /             /  \ \         \   n+af      |
          | | | nabla o eps | Du | | o nabla | u     , v  |
          |  \ \             \  / /         /             |
           \                                             /
    */
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        const int fui_p_jdim = nsd_ * ui + jdim;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi_p_idim = nsd_ * vi + idim;

            for (int kdim = 0; kdim < nsd_; ++kdim)
            {
              estif_u(fvi_p_idim, fui_p_jdim) -=
                  crossfac * lin_resM_Du(nsd_ * kdim + jdim, ui) * vderxy_(idim, kdim) * funct_(vi);
            }
          }  // jdim
        }  // vi
      }  // ui
    }  // idim

    /*
                    /                               \
                   |  /                \   n+af      |
                   | | nabla Dp o nabla | u     , v  |
                   |  \                /             |
                    \                               /
    */
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int fvi = nsd_ * vi + idim;

        for (int ui = 0; ui < nen_; ++ui)
        {
          for (int kdim = 0; kdim < nsd_; ++kdim)
          {
            estif_p_v(fvi, ui) -=
                crossfac * timefacfacpre * vderxy_(idim, kdim) * derxy_(kdim, ui) * funct_(vi);
          }
        }
      }  // end for(idim)
    }  // vi
  }  // end if (cross_ == Inpar::FLUID::cross_stress_stab) and (is_newton)

  // Stabilization only of the rhs
  static Core::LinAlg::Matrix<nsd_, 1> temp;
  temp.clear();

  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int kdim = 0; kdim < nsd_; ++kdim)
    {
      temp(jdim) += rhsfac * densaf_ * sgvelint_(kdim) * vderxy_(jdim, kdim);
    }
  }

  for (int idim = 0; idim < nsd_; ++idim)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      velforce(idim, vi) -= temp(idim) * funct_(vi);
    }
  }  // end for(idim)


  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::reynolds_stress_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac,
    const double& timefacfacpre, const double& fac3)
{
  /*
                            linearisation of
                         this part is performed
                            in the following
                                +--------+
                                |        |
                   /                                 \
                  |  ~n+af     /    ~n+af       \     |
                - |  u     ,  | rho*u    o nabla | v  |
                  |   (i)      \     (i)        /     |
                   \                                 /
                     |   |
                     +---+
            this part is linearised
          in combination with the SUPG
                  term above

  */

  double reyfac;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    // if(fldpara_->IsGenalpha())
    reyfac = densaf_ * tau_(1);
    // else
    // reyfac=densaf_*tau_(1)/fldpara_->Theta();
  }
  else
    reyfac = densaf_ * fldparatimint_->alpha_f() * fac3;

  /*
          /                          \
         |  ~n+af                     |
         |  u     , ( Du o nabla ) v  |
         |                            |
          \                          /
  */
  /*
          /                                                 \
         |  ~n+af    / / / n+af        \     \         \     |
         |  u     , | | | u     o nabla | Du  | o nabla | v  |
         |           \ \ \             /     /         /     |
          \                                                 /
  */
  /*
          /                                                 \
         |  ~n+af    / / /          \   n+af \         \     |
         |  u     , | | | Du o nabla | u      | o nabla | v  |
         |           \ \ \          /        /         /     |
          \                                                 /
  */
  /*
          /                                \
         |  ~n+af                           |
         |  u     , ( sigma*Du o nabla ) v  |
         |                                  |
          \                                /
  */
  /*
          /                                               \
         |  ~n+af    / /             /  \  \         \     |
         |  u     , | | nabla o eps | Du |  | o nabla | v  |
         |           \ \             \  /  /         /     |
          \                                               /
  */
  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int fui_p_jdim = nsd_ * ui + jdim;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi_p_idim = nsd_ * vi + idim;

          for (int kdim = 0; kdim < nsd_; ++kdim)
          {
            estif_u(fvi_p_idim, fui_p_jdim) +=
                reyfac * lin_resM_Du(nsd_ * kdim + jdim, ui) * sgvelint_(idim) * derxy_(kdim, vi);
          }
        }  // jdim
      }  // vi
    }  // ui
  }  // idim

  /*
          /                                \
         |  ~n+af    /                \     |
         |  u     , | nabla Dp o nabla | v  |
         |           \                /     |
          \                                /
  */
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int fvi_p_idim = nsd_ * vi + idim;

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int kdim = 0; kdim < nsd_; ++kdim)
        {
          estif_p_v(fvi_p_idim, ui) +=
              reyfac * timefacfacpre * sgvelint_(idim) * derxy_(kdim, ui) * derxy_(kdim, vi);
        }
      }
    }  // end for(idim)
  }  // vi

  return;
}


/*----------------------------------------------------------------------*
 * Evaluate supporting methods of the element
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::evaluate_service(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  switch (act)
  {
    case FLD::calc_div_u:
    {
      // compute divergence of velocity field at the element
      return compute_div_u(ele, discretization, lm, elevec1);
    }
    break;
    case FLD::calc_fluid_error:
    {
      // compute error for a known analytical solution
      return compute_error(ele, params, mat, discretization, lm, elevec1);
    }
    break;
    case FLD::calc_dissipation:
    {
      if (nsd_ == 3)
      {
        if (ele->owner() ==
            Core::Communication::my_mpi_rank(
                discretization.get_comm()))  // don't store values of ghosted elements
        {
          return calc_dissipation(ele, params, discretization, lm, mat);
        }
      }
      else
        FOUR_C_THROW("{} D elements does not support calculation of dissipation", nsd_);
    }
    break;
    case FLD::integrate_shape:
    {
      // integrate shape function for this element
      // (results assembled into element vector)
      return integrate_shape_function(ele, discretization, lm, elevec1);
    }
    break;
    case FLD::calc_divop:
    {
      // calculate the integrated divergence operator
      return calc_div_op(ele, discretization, lm, elevec1);
    }
    break;
    case FLD::calc_mass_matrix:
    {
      // compute element mass matrix
      return calc_mass_matrix(ele, discretization, lm, mat, elemat1);
    }
    break;
      break;
    case FLD::calc_turbulence_statistics:
    {
      if (nsd_ == 3)
      {
        if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        {
          // it is quite expensive to calculate second order derivatives, since a matrix has to be
          // inverted... so let's save time here
          is_higher_order_ele_ = false;
          return calc_channel_statistics(ele, params, discretization, lm, mat);
        }
      }  // end if (nsd == 3)
      else
        FOUR_C_THROW("action 'calc_turbulence_statistics' is a 3D specific action");
      return 0;
    }
    break;
    case FLD::velgradient_projection:
    {
      // project velocity gradient to nodal level
      return vel_gradient_projection(ele, params, discretization, lm, elemat1, elemat2);
    }
    break;
    case FLD::presgradient_projection:
    {
      // project velocity gradient to nodal level
      return pres_gradient_projection(ele, params, discretization, lm, elemat1, elemat2);
    }
    break;
    case FLD::calc_dt_via_cfl:
    {
      return calc_time_step(ele, discretization, lm, elevec1);
    }
    break;
    case FLD::calc_velgrad_ele_center:
    {
      return calc_vel_gradient_ele_center(ele, discretization, lm, elevec1, elevec2);
    }
    break;
    case FLD::calc_mass_flow_periodic_hill:
    {
      // compute element mass matrix
      return calc_mass_flow_periodic_hill(ele, params, discretization, lm, elevec1, mat);
    }
    break;
    default:
      FOUR_C_THROW("Unknown type of action '{}' for Fluid evaluate_service()", act);
      break;
  }  // end of switch(act)

  return 0;
}


/*----------------------------------------------------------------------*
 * Action type: Integrate shape function
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::integrate_shape_function(
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1)
{
  // integrations points and weights
  return integrate_shape_function(ele, discretization, lm, elevec1, intpoints_);
}


/*----------------------------------------------------------------------*
 * Action type: Integrate shape function
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::integrate_shape_function(
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    const Core::FE::GaussIntegration& intpoints)
{
  // --------------------------------------------------
  // construct views
  Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> vector(elevec1.values(), true);

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // set element id
  eid_ = ele->id();

  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, ele, myknots_, weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

    // get new node positions for isale
    xyze_ += edispnp;
  }

  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------

  for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    for (int ui = 0; ui < nen_; ++ui)  // loop rows  (test functions)
    {
      // integrated shape function is written into the pressure dof
      int fuippp = numdofpernode_ * ui + nsd_;
      vector(fuippp) += fac_ * funct_(ui);
    }
  }

  return 0;
}


/*---------------------------------------------------------------------*
 | Action type: calc_divop                                             |
 | calculate integrated divergence operator              mayr.mt 04/12 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_div_op(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // set element id
  eid_ = ele->id();

  if (ele->is_ale())  // Do ALE specific updates if necessary
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

    // get new node positions of ALE mesh
    xyze_ += edispnp;
  }

  // integration loop
  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    for (int nodes = 0; nodes < nen_; nodes++)  // loop over nodes
    {
      for (int dim = 0; dim < nsd_; dim++)  // loop over spatial dimensions
      {
        elevec1((nsd_ + 1) * nodes + dim) += derxy_(dim, nodes) * fac_;
      }
    }
  }  // end of integration loop

  return 0;
}


/*---------------------------------------------------------------------*
 | Action type: velgradient_projection                                 |
 | project velocity gradient to nodal level                ghamm 06/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::vel_gradient_projection(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2)
{
  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  Core::LinAlg::Matrix<nsd_, nen_> evel(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evel, nullptr, "vel");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // set element id
  eid_ = ele->id();

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "disp");

    // get new node positions for isale
    xyze_ += edispnp;
  }

  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------

  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    vderxy_.multiply_nt(evel, derxy_);

    // fill element matrix (mass matrix) and elevec in each node
    for (int vi = 0; vi < nen_; ++vi)  // loop rows
    {
      for (int ui = 0; ui < nen_; ++ui)  // loop columns
      {
        elemat1(vi, ui) += fac_ * funct_(ui) * funct_(vi);
      }
    }

    // fill elemat which is of size node x (nsd_*nsd_) with rhs
    for (int i = 0; i < nsd_; ++i)  // loop rows of vderxy
    {
      for (int j = 0; j < nsd_; ++j)  // loop columns of vderxy
      {
        for (int vi = 0; vi < nen_; ++vi)  // loop nodes
        {
          elemat2(vi, i * nsd_ + j) += fac_ * funct_(vi) * vderxy_(i, j);
        }
      }
    }

  }  // end of integration loop

  return 0;
}

/*---------------------------------------------------------------------*
 | Action type: presgradient_projection                                |
 | project pressure gradient to nodal level               winter 09/15 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::pres_gradient_projection(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2)
{
  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  Core::LinAlg::Matrix<nen_, 1> epres(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, nullptr, &epres, "pres");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // set element id
  eid_ = ele->id();

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "disp");

    // get new node positions for isale
    xyze_ += edispnp;
  }

  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------

  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    gradp_.multiply(derxy_, epres);

    // fill element matrix (mass matrix) and elevec in each node
    for (int vi = 0; vi < nen_; ++vi)  // loop rows
    {
      for (int ui = 0; ui < nen_; ++ui)  // loop columns
      {
        elemat1(vi, ui) += fac_ * funct_(ui) * funct_(vi);
      }
    }

    // fill elemat which is of size node x (1*nsd_) with rhs
    for (int i = 0; i < nsd_; ++i)  // loop rows of vderxy
    {
      //      for (int j=0; j<nsd_; ++j) // loop columns of vderxy
      //      {
      for (int vi = 0; vi < nen_; ++vi)  // loop nodes
      {
        elemat2(vi, i) += fac_ * funct_(vi) * gradp_(i, 0);
      }
      //      }
    }

  }  // end of integration loop

  return 0;
}


/*----------------------------------------------------------------------*
 * Action type: Compute Div u                                 ehrl 12/12|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::compute_div_u(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  double area = 0.0;
  double divu = 0.0;

  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1, velocity for continuity equ. at
  // time n+1 ost:         velocity/pressure at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelaf, nullptr, "velaf");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // set element id
  eid_ = ele->id();

  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, ele, myknots_, weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

    // get new node positions for isale
    xyze_ += edispnp;
  }

  {
    // evaluate shape functions and derivatives at element center
    eval_shape_func_and_derivs_at_ele_center();

    //------------------------------------------------------------------
    //                       INTEGRATION LOOP
    //------------------------------------------------------------------

    // option to evaluate div u at Gauss point

    // loop over Gauss points if div u needs to be evaluated at the Gauss points
    /*
  for ( Core::FE::GaussIntegration::iterator iquad=intpoints_.begin();
  iquad!=intpoints_.end();
  ++iquad )
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.Point(),iquad.Weight());
  */

    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    vderxy_.multiply_nt(evelaf, derxy_);

    vdiv_ = 0.0;
    for (int idim = 0; idim < nsd_; ++idim)
    {
      vdiv_ += vderxy_(idim, idim);
    }

    divu += vdiv_ * fac_;
    area += fac_;
  }
  elevec1[0] = divu / area;
  return 0;
}


/*----------------------------------------------------------------------*
 * Action type: Compute Error                              shahmiri 01/12
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::compute_error(Discret::Elements::Fluid* ele,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // integrations points and weights
  // more GP than usual due to (possible) cos/exp fcts in analytical solutions
  // degree 5
  const Core::FE::GaussIntegration intpoints(distype, ele->degree() * 2 + 3);
  return compute_error(ele, params, mat, discretization, lm, elevec1, intpoints);
}


/*----------------------------------------------------------------------*
 * Action type: Compute Error                              shahmiri 01/12
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::compute_error(Discret::Elements::Fluid* ele,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, const Core::FE::GaussIntegration& intpoints)
{
  // analytical solution
  Core::LinAlg::Matrix<nsd_, 1> u(Core::LinAlg::Initialization::zero);
  double p = 0.0;
  Core::LinAlg::Matrix<nsd_, nsd_> dervel(Core::LinAlg::Initialization::zero);

  // error
  Core::LinAlg::Matrix<nsd_, 1> deltavel(Core::LinAlg::Initialization::zero);
  double deltap = 0.0;
  Core::LinAlg::Matrix<nsd_, nsd_> deltadervel(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nsd_> dervelint(Core::LinAlg::Initialization::zero);

  const auto calcerr =
      Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(params, "calculate error");
  const int calcerrfunctno = params.get<int>("error function number");

  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1
  // ost:         velocity/pressure at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelaf, &epreaf, "velaf");

  // np_genalpha: additional vector for velocity at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> eprenp(Core::LinAlg::Initialization::zero);
  if (fldparatimint_->is_genalpha_np())
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelnp, &eprenp, "velnp");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // set element id
  eid_ = ele->id();

  //----------------------------------------------------------------
  // Now do the nurbs specific stuff (for isogeometric elements)
  //----------------------------------------------------------------
  if (isNurbs_)
  {
    // access knots and weights for this element
    bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, ele, myknots_, weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return (0);
  }  // Nurbs specific stuff

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

    // get new node positions for isale
    xyze_ += edispnp;
  }

  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------

  for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    velint_.multiply(evelaf, funct_);

    // get pressure at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    double preint(true);
    if (fldparatimint_->is_genalpha_np())
      preint = funct_.dot(eprenp);
    else
      preint = funct_.dot(epreaf);

    // H1 -error norm
    // compute first derivative of the velocity
    dervelint.multiply_nt(evelaf, derxy_);

    // get coordinates at integration point
    Core::LinAlg::Matrix<nsd_, 1> xyzint(Core::LinAlg::Initialization::zero);
    xyzint.multiply(xyze_, funct_);

    //  the error is evaluated at the specific time of the used time integration scheme
    //  n+alpha_F for generalized-alpha scheme
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)

    evaluate_analytic_solution_point(xyzint, fldparatimint_->time(), calcerr, calcerrfunctno, mat,
        u, p, dervel, fldparatimint_->is_full_impl_pressure_and_cont(), fldparatimint_->dt());

    // compute difference between analytical solution and numerical solution
    deltap = preint - p;
    deltavel.update(1.0, velint_, -1.0, u);

    // H1 -error norm
    // compute error for first velocity derivative
    for (int i = 0; i < nsd_; ++i)
      for (int j = 0; j < nsd_; ++j) deltadervel(i, j) = dervelint(i, j) - dervel(i, j);

    // 0: delta velocity L2-error norm
    // 1: delta p L2-error norm
    // 2: delta velocity H1-error norm
    // 3: analytical velocity L2 norm
    // 4: analytical p L2 norm
    // 5: analytical velocity H1 norm

    // the error for the L2 and H1 norms are evaluated at the Gauss point
    for (int isd = 0; isd < nsd_; isd++)
    {
      // integrate delta velocity for L2-error norm
      elevec1[0] += deltavel(isd) * deltavel(isd) * fac_;
      // integrate delta velocity for H1-error norm
      elevec1[2] += deltavel(isd) * deltavel(isd) * fac_;
      // integrate analytical velocity for L2 norm
      elevec1[3] += u(isd) * u(isd) * fac_;
      // integrate analytical velocity for H1 norm
      elevec1[5] += u(isd) * u(isd) * fac_;
    }
    // integrate delta p for L2-error norm
    elevec1[1] += deltap * deltap * fac_;
    // integrate analytical p for L2 norm
    elevec1[4] += p * p * fac_;

    // integrate delta velocity derivative for H1-error norm
    elevec1[2] += deltadervel.dot(deltadervel) * fac_;
    // integrate analytical velocity for H1 norm
    elevec1[5] += dervel.dot(dervel) * fac_;
  }

  return 0;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::evaluate_analytic_solution_point(
    const Core::LinAlg::Matrix<nsd_, 1>& xyzint, const double t,
    const Inpar::FLUID::CalcError calcerr, const int calcerrfunctno,
    const std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::Matrix<nsd_, 1>& u, double& p,
    Core::LinAlg::Matrix<nsd_, nsd_>& dervel, bool isFullImplPressure, double deltat)
{
  // Compute analytical solution
  switch (calcerr)
  {
    case Inpar::FLUID::beltrami_flow:
    {
      if (nsd_ == 3)
      {
        double visc = 1.;
        // get viscosity
        if (mat->material_type() == Core::Materials::m_fluid)
        {
          const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());

          // get constant kinematic viscosity
          visc = actmat->viscosity() / actmat->density();
        }
        else
          FOUR_C_THROW("Material is not Newtonian Fluid");

        const double a = std::numbers::pi / 4.0;
        const double d = std::numbers::pi / 2.0;

        // compute analytical pressure
        if (not isFullImplPressure)
        {
          p = -a * a / 2.0 *
              (std::exp(2.0 * a * xyzint(0)) + std::exp(2.0 * a * xyzint(1)) +
                  std::exp(2.0 * a * xyzint(2)) +
                  2.0 * std::sin(a * xyzint(0) + d * xyzint(1)) *
                      std::cos(a * xyzint(2) + d * xyzint(0)) *
                      std::exp(a * (xyzint(1) + xyzint(2))) +
                  2.0 * std::sin(a * xyzint(1) + d * xyzint(2)) *
                      std::cos(a * xyzint(0) + d * xyzint(1)) *
                      std::exp(a * (xyzint(2) + xyzint(0))) +
                  2.0 * std::sin(a * xyzint(2) + d * xyzint(0)) *
                      std::cos(a * xyzint(1) + d * xyzint(2)) *
                      std::exp(a * (xyzint(0) + xyzint(1)))) *
              std::exp(-2.0 * visc * d * d * t);
        }
        else  // pressure for full implicit OST scheme:
        {
          p = -a * a / 2.0 *
              (std::exp(2.0 * a * xyzint(0)) + std::exp(2.0 * a * xyzint(1)) +
                  std::exp(2.0 * a * xyzint(2)) +
                  2.0 * std::sin(a * xyzint(0) + d * xyzint(1)) *
                      std::cos(a * xyzint(2) + d * xyzint(0)) *
                      std::exp(a * (xyzint(1) + xyzint(2))) +
                  2.0 * std::sin(a * xyzint(1) + d * xyzint(2)) *
                      std::cos(a * xyzint(0) + d * xyzint(1)) *
                      std::exp(a * (xyzint(2) + xyzint(0))) +
                  2.0 * std::sin(a * xyzint(2) + d * xyzint(0)) *
                      std::cos(a * xyzint(1) + d * xyzint(2)) *
                      std::exp(a * (xyzint(0) + xyzint(1)))) *
              std::exp(-2.0 * visc * d * d * (t - 0.5 * deltat));
        }

        // H1 -error norm
        // sacado data type replaces "double"
        using FAD = Sacado::Fad::DFad<double>;  // for first derivs

        FAD x = xyzint(0);
        x.diff(0, 3);  // independent variable 0 out of a total of 3

        FAD y = xyzint(1);
        y.diff(1, 3);  // independent variable 1 out of a total of 3

        FAD z = xyzint(2);
        z.diff(2, 3);  // independent variable 2 out of a total of 3

        // compute the function itself AND its derivatives w.r.t. ALL indep. variables
        FAD uu = -a *
                 (std::exp(a * x) * std::sin(a * y + d * z) +
                     std::exp(a * z) * std::cos(a * x + d * y)) *
                 std::exp(-visc * d * d * t);
        FAD vv = -a *
                 (std::exp(a * y) * std::sin(a * z + d * x) +
                     std::exp(a * x) * std::cos(a * y + d * z)) *
                 std::exp(-visc * d * d * t);
        FAD ww = -a *
                 (std::exp(a * z) * std::sin(a * x + d * y) +
                     std::exp(a * y) * std::cos(a * z + d * x)) *
                 std::exp(-visc * d * d * t);

        u(0) = uu.val();
        u(1) = vv.val();
        u(2) = ww.val();

        dervel(0, 0) = uu.dx(0);
        dervel(0, 1) = uu.dx(1);
        dervel(0, 2) = uu.dx(2);
        dervel(1, 0) = vv.dx(0);
        dervel(1, 1) = vv.dx(1);
        dervel(1, 2) = vv.dx(2);
        dervel(2, 0) = ww.dx(0);
        dervel(2, 1) = ww.dx(1);
        dervel(2, 2) = ww.dx(2);
      }
      else
        FOUR_C_THROW("action 'calc_fluid_beltrami_error' is a 3D specific action");
    }
    break;
    case Inpar::FLUID::shear_flow:
    {
      const double maxvel = 1.0;
      const double height = 1.0;

      // y=0 is located in the middle of the domain
      if (nsd_ == 2)
      {
        p = 1.0;
        u(0) = xyzint(1) * maxvel + height / 2 * maxvel;
        u(1) = 0.0;
      }
      if (nsd_ == 3)
      {
        p = 0.0;
        u(0) = xyzint(1) * maxvel + height / 2 * maxvel;
        u(1) = 0.0;
        u(2) = 0.0;
      }
    }
    break;
    case Inpar::FLUID::gravitation:
    {
      const double gravity = 10.0;
      const double height = 1.0;

      // 2D: rectangle 1.0x1.0
      // 3D: cube 1.0x1.0x1.0
      // y=0 is located in the middle of the domain
      if (nsd_ == 2)
      {
        p = -xyzint(1) * gravity + height / 2 * gravity;
        u(0) = 0.0;
        u(1) = 0.0;
      }
      if (nsd_ == 3)
      {
        p = -xyzint(1) * gravity + height / 2 * gravity;
        u(0) = 0.0;
        u(1) = 0.0;
        u(2) = 0.0;
      }
    }
    break;
    case Inpar::FLUID::channel2D:
    {
      const double maxvel = 1.0;
      const double height = 1.0;
      const double visc = 1.0;
      const double pressure_gradient = 10.0;

      // u_max = 1.25
      // y=0 is located in the middle of the channel
      if (nsd_ == 2)
      {
        p = 1.0;
        // p = -10*xyzint(0)+20;
        u(0) = maxvel - ((height * height) / (2.0 * visc) * pressure_gradient *
                            (xyzint(1) / height) * (xyzint(1) / height));
        u(1) = 0.0;
      }
      else
        FOUR_C_THROW("3D analytical solution is not implemented yet");
    }
    break;
    case Inpar::FLUID::byfunct:
    {
      // function evaluation requires a 3D position vector!!
      double position[3];

      if (nsd_ == 2)
      {
        position[0] = xyzint(0);
        position[1] = xyzint(1);
        position[2] = 0.0;
      }
      else if (nsd_ == 3)
      {
        position[0] = xyzint(0);
        position[1] = xyzint(1);
        position[2] = xyzint(2);
      }
      else
        FOUR_C_THROW("invalid nsd {}", nsd_);

      if (nsd_ == 2)
      {
        const double u_exact_x =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 0);
        const double u_exact_y =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 1);
        const double p_exact =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 2);

        u(0) = u_exact_x;
        u(1) = u_exact_y;
        p = p_exact;

        std::vector<double> uder_exact_x =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate_spatial_derivative(position, t, 0);
        std::vector<double> uder_exact_y =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate_spatial_derivative(position, t, 1);

        if (uder_exact_x.size())
        {
          dervel(0, 0) = uder_exact_x[0];
          dervel(0, 1) = uder_exact_x[1];
        }

        if (uder_exact_y.size())
        {
          dervel(1, 0) = uder_exact_y[0];
          dervel(1, 1) = uder_exact_y[1];
        }
      }
      else if (nsd_ == 3)
      {
        const double u_exact_x =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 0);
        const double u_exact_y =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 1);
        const double u_exact_z =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 2);
        const double p_exact =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate(position, t, 3);

        u(0) = u_exact_x;
        u(1) = u_exact_y;
        u(2) = u_exact_z;
        p = p_exact;

        std::vector<double> uder_exact_x =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate_spatial_derivative(position, t, 0);
        std::vector<double> uder_exact_y =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate_spatial_derivative(position, t, 1);
        std::vector<double> uder_exact_z =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(calcerrfunctno)
                .evaluate_spatial_derivative(position, t, 2);

        if (uder_exact_x.size())
        {
          dervel(0, 0) = uder_exact_x[0];
          dervel(0, 1) = uder_exact_x[1];
          dervel(0, 2) = uder_exact_x[2];
        }

        if (uder_exact_y.size())
        {
          dervel(1, 0) = uder_exact_y[0];
          dervel(1, 1) = uder_exact_y[1];
          dervel(1, 2) = uder_exact_y[2];
        }

        if (uder_exact_z.size())
        {
          dervel(2, 0) = uder_exact_z[0];
          dervel(2, 1) = uder_exact_z[1];
          dervel(2, 2) = uder_exact_z[2];
        }
      }
      else
        FOUR_C_THROW("invalid dimension");
    }
    break;
    case Inpar::FLUID::fsi_fluid_pusher:
    {
      /* Since the fluid pusher solution depends only on time, but not on spatial
       * coordinates x,y,z, we only compute the L2-error and no H1-error.
       */

      // get pointer to material in order to access density
      if (mat->material_type() == Core::Materials::m_fluid)
      {
        const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());

        // available solution types
        enum SOLUTIONTYPE
        {
          quadratic,  // d(t) = -t^2
          cubic,      // d(t) = -t^3
          quartic,    // d(t) = -t^4
          quintic     // d(t) = -t^5
        };

        // choose solution type
        SOLUTIONTYPE solutiontype = quintic;

        // compute analytical velocities and pressure for different prescribed time curves
        // Note: consider offset of coordinate system
        switch (solutiontype)
        {
          case quadratic:
          {
            u(0) = -2.0 * t;
            u(1) = 0.0;
            u(2) = 0.0;
            p = actmat->density() * 2.0 * (xyzint(0) + 1.5);

            break;
          }
          case cubic:
          {
            u(0) = -3.0 * t * t;
            u(1) = 0.0;
            u(2) = 0.0;
            p = actmat->density() * 6.0 * t * (xyzint(0) + 1.5);

            break;
          }
          case quartic:
          {
            u(0) = -4.0 * t * t * t;
            u(1) = 0.0;
            u(2) = 0.0;
            p = actmat->density() * 12.0 * t * t * (xyzint(0) + 1.5);

            break;
          }
          case quintic:
          {
            u(0) = -5.0 * t * t * t * t;
            u(1) = 0.0;
            u(2) = 0.0;
            p = actmat->density() * 20.0 * t * t * t * (xyzint(0) + 1.5);

            break;
          }
          default:
          {
            FOUR_C_THROW("Unknown solution type.");
            break;
          }
        }
      }
      else
        FOUR_C_THROW("Material is not a Newtonian Fluid");
    }
    break;
    case Inpar::FLUID::channel_weakly_compressible:
    {
      // Steady, weakly compressible isothermal flow of a Newtonian fluid
      // with pressure-dependent density according to Murnaghan-Tait law
      // Comparison to analytical solution obtained in:
      // "New analytical solutions for weakly compressible Newtonian
      // Poiseuille flows with pressure-dependent viscosity"
      // Kostas D. Housiadas, Georgios C. Georgiou

      if (mat->material_type() == Core::Materials::m_fluid_murnaghantait)
      {
        const Mat::MurnaghanTaitFluid* actmat =
            static_cast<const Mat::MurnaghanTaitFluid*>(mat.get());

        if (actmat->mat_parameter() != 1.0)
        {
          FOUR_C_THROW("The analytical solution is only valid for material parameter = 1");
        }

        double x = xyzint(0);
        double y = xyzint(1);

        double length = 10.0;
        double radius = 1.0;
        double aspect_ratio = radius / length;
        double mean_velocity_channel_exit = 1.0;
        double viscosity = actmat->viscosity();
        double reference_pressure = actmat->ref_pressure();
        double reference_bulk_modulus = actmat->ref_bulk_modulus();
        double linear_coefficient_density = (3.0 * (1.0 / reference_bulk_modulus) * viscosity *
                                                length * mean_velocity_channel_exit) /
                                            std::pow(radius, 2.0);

        if (nsd_ == 2)
        {
          u(0) = 3.0 / 2.0 * (1 - std::pow(y / radius, 2.0)) *
                 (1.0 + linear_coefficient_density * (x / length - 1.0));
          u(1) = 0.0;
          p = 1.0 - x / length -
              linear_coefficient_density *
                  (1.0 / 6.0 * std::pow(aspect_ratio, 2.0) * (std::pow(y / radius, 2.0) - 1.0) +
                      1.0 / 2.0 * std::pow(1.0 - x / length, 2.0));
          dervel(0, 0) =
              3.0 / 2.0 * (1.0 - std::pow(y / radius, 2.0)) * linear_coefficient_density / length;
          dervel(0, 1) = -3.0 / std::pow(radius, 2.0) * y *
                         (1.0 + linear_coefficient_density * (x / length - 1.0));
          dervel(1, 0) = 0.0;
          dervel(1, 1) = 0.0;

          // scaling correctly the variables
          u(0) = u(0) * mean_velocity_channel_exit;
          u(1) = u(1) * mean_velocity_channel_exit * radius / length;
          p = p * (3.0 * viscosity * length * mean_velocity_channel_exit / std::pow(radius, 2.0)) +
              reference_pressure;
          dervel(0, 0) = dervel(0, 0) * mean_velocity_channel_exit;
          dervel(0, 1) = dervel(0, 1) * mean_velocity_channel_exit;
          dervel(1, 0) = dervel(1, 0) * mean_velocity_channel_exit * radius / length;
          dervel(1, 1) = dervel(1, 1) * mean_velocity_channel_exit * radius / length;
        }
        else
          FOUR_C_THROW("3D analytical solution is not implemented");
      }
      else if (mat->material_type() == Core::Materials::m_fluid_linear_density_viscosity)
      {
        const Mat::LinearDensityViscosity* actmat =
            static_cast<const Mat::LinearDensityViscosity*>(mat.get());

        double x = xyzint(0);
        double y = xyzint(1);

        double length = 10.0;
        double radius = 1.0;
        double aspect_ratio = radius / length;
        double mean_velocity_channel_exit = 1.0;
        double reference_viscosity = actmat->ref_viscosity();
        double reference_pressure = actmat->ref_pressure();
        double coefficient_density = actmat->coeff_density();
        double coefficient_viscosity = actmat->coeff_viscosity();
        double coefficient_density_adim = (3.0 * coefficient_density * reference_viscosity *
                                              length * mean_velocity_channel_exit) /
                                          std::pow(radius, 2.0);
        double coefficient_viscosity_adim = (3.0 * coefficient_viscosity * reference_viscosity *
                                                length * mean_velocity_channel_exit) /
                                            std::pow(radius, 2.0);

        // parameters according with the paper
        double z = x / length;
        double r = y / radius;
        double alfa = aspect_ratio;
        double beta = coefficient_viscosity_adim;
        double epsilon = coefficient_density_adim;
        double a = aspect_ratio;
        double B = alfa * beta;
        double lambda = 1.0 + (1.0 / 5.0) * std::pow(B, 2.0) + (11.0 / 175.0) * std::pow(B, 4.0) +
                        (533.0 / 23625.0) * std::pow(B, 6.0) +
                        (5231.0 / 606375.0) * std::pow(B, 8.0);
        double p_0_hat = std::cosh(alfa * beta * lambda * r) / std::cosh(alfa * beta * lambda);
        double u_r1_hat =
            -(11.0 * r * std::pow(1.0 - std::pow(r, 2.0), 2.0)) / 40.0 * std::pow(B, 2.0) *
            (1.0 + ((173.0 - 85.0 * std::pow(r, 2.0)) / (770.0)) * std::pow(B, 2.0) +
                ((5793.0 - 7190.0 * std::pow(r, 2.0) + 3965.0 * std::pow(r, 4.0)) / (83160.0)) *
                    std::pow(B, 4.0) +
                ((7435723.0 - 16839665.0 * std::pow(r, 2.0) + 16836225.0 * std::pow(r, 4.0) -
                     5021275.0 * std::pow(r, 6.0)) /
                    (320166000.0)) *
                    std::pow(B, 6.0));
        double u_r1_hat_first =
            (11.0 * std::pow(B, 2.0) * std::pow(std::pow(r, 2.0) - 1.0, 2.0) *
                (((4099.0 * std::pow(r, 6.0)) / 261360.0 - (32069.0 * std::pow(r, 4.0)) / 609840.0 +
                     (3367933.0 * std::pow(r, 2.0)) / 64033200.0 - 7435723.0 / 320166000.0) *
                        std::pow(B, 6.0) +
                    (-(793.0 * std::pow(r, 4.0)) / 16632.0 + (719.0 * std::pow(r, 2.0)) / 8316.0 -
                        1931.0 / 27720.0) *
                        std::pow(B, 4.0) +
                    ((17.0 * std::pow(r, 2.0)) / 154.0 - 173.0 / 770.0) * std::pow(B, 2.0) - 1.0)) /
                40.0 +
            (11.0 * std::pow(B, 2.0) * std::pow(r, 2.0) * (std::pow(r, 2.0) - 1.0) *
                (((4099.0 * std::pow(r, 6.0)) / 261360.0 - (32069.0 * std::pow(r, 4.0)) / 609840.0 +
                     (3367933.0 * std::pow(r, 2.0)) / 64033200.0 - 7435723.0 / 320166000.0) *
                        std::pow(B, 6.0) +
                    (-(793.0 * std::pow(r, 4.0)) / 16632.0 + (719.0 * std::pow(r, 2.0)) / 8316.0 -
                        1931.0 / 27720.0) *
                        std::pow(B, 4.0) +
                    ((17.0 * std::pow(r, 2.0)) / 154.0 - 173.0 / 770.0) * std::pow(B, 2.0) - 1.0)) /
                10.0 +
            (11.0 * std::pow(B, 2.0) * r * std::pow(std::pow(r, 2.0) - 1.0, 2.0) *
                (((4099.0 * std::pow(r, 5.0)) / 43560.0 - (32069.0 * std::pow(r, 3.0)) / 152460.0 +
                     (3367933.0 * r) / 32016600.0) *
                        std::pow(B, 6.0) +
                    ((719.0 * r) / 4158.0 - (793.0 * std::pow(r, 3.0)) / 4158.0) *
                        std::pow(B, 4.0) +
                    (17.0 * r * std::pow(B, 2.0)) / 77.0)) /
                40.0;
        double h = 1.0 / std::pow(beta, 2.0) *
                   (-1.0 + ((11.0 - 10.0 * std::pow(r, 2.0)) / (15.0)) * std::pow(B, 2.0) +
                       ((359.0 - 126.0 * std::pow(r, 2.0) + 35.0 * std::pow(r, 4.0)) / (1260.0)) *
                           std::pow(B, 4.0) +
                       ((13761.0 - 17790.0 * std::pow(r, 2.0) + 34125.0 * std::pow(r, 4.0) -
                            17500.0 * std::pow(r, 6.0)) /
                           (94500.0)) *
                           std::pow(B, 6.0) +
                       ((225311.0 - 614515.0 * std::pow(r, 2.0) + 1492755.0 * std::pow(r, 4.0) -
                            1324785.0 * std::pow(r, 6.0) + 394350.0 * std::pow(r, 8.0)) /
                           (3118500.0)) *
                           std::pow(B, 8.0));
        double h_1 = 1.0 / std::pow(beta, 2.0) *
                     (-1.0 + ((11.0 - 10.0) / (15.0)) * std::pow(B, 2.0) +
                         ((359.0 - 126.0 + 35.0) / (1260.0)) * std::pow(B, 4.0) +
                         ((13761.0 - 17790.0 + 34125.0 - 17500.0) / (94500.0)) * std::pow(B, 6.0) +
                         ((225311.0 - 614515.0 + 1492755.0 - 1324785.0 + 394350.0) / (3118500.0)) *
                             std::pow(B, 8.0));

        if (nsd_ == 2)
        {
          u(0) =
              -(3.0 * std::log(p_0_hat)) / (std::pow(B, 2.0) * lambda) +
              epsilon * ((3 * (std::tanh(B * lambda) - r * std::tanh(B * lambda * r)) +
                             std::log(std::pow(p_0_hat, 3.0)) / (B * lambda)) /
                                (beta * (3 * std::tanh(B * lambda) - 2 * B)) +
                            (std::exp(lambda * beta * (1.0 - z))) / (lambda * beta) *
                                ((p_0_hat * std::log(std::pow(p_0_hat, 3.0))) / (std::pow(B, 2.0)) +
                                    u_r1_hat_first));
          u(1) = epsilon * u_r1_hat * std::exp(lambda * beta * (1.0 - z));
          p = (p_0_hat * std::exp(lambda * beta * (1.0 - z)) - 1.0) / beta +
              epsilon * p_0_hat * std::exp(lambda * beta * (1.0 - z)) *
                  ((lambda * a *
                       (1.0 - z + a * (r * std::tanh(B * lambda * r) - std::tanh(B * lambda)))) /
                          (3.0 * std::tanh(B * lambda) - 2.0 * B) +
                      p_0_hat * h * std::exp(lambda * beta * (1.0 - z)) - h_1);

          // scaling correctly the variables
          u(0) = u(0) * mean_velocity_channel_exit;
          u(1) = u(1) * mean_velocity_channel_exit * radius / length;
          p = p * (3.0 * reference_viscosity * length * mean_velocity_channel_exit /
                      std::pow(radius, 2.0)) +
              reference_pressure;
        }
        else
          FOUR_C_THROW("3D analytical solution is not implemented");
      }
      else
      {
        FOUR_C_THROW("The analytical solution is not implemented for this material");
      }
    }
    break;
    default:
      FOUR_C_THROW("analytical solution is not defined");
      break;
  }
}


/*!
 * \brief fill element matrix and vectors with the global values
 */
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::extract_values_from_global_vector(
    const Core::FE::Discretization& discretization,  ///< discretization
    const std::vector<int>& lm,                      ///<
    FLD::RotationallySymmetricPeriodicBC<distype, nsd_ + 1, enrtype>& rotsymmpbc,  ///<
    Core::LinAlg::Matrix<nsd_, nen_>* matrixtofill,                                ///< vector field
    Core::LinAlg::Matrix<nen_, 1>* vectortofill,                                   ///< scalar field
    const std::string state)  ///< state of the global vector
{
  // get state of the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(state);

  if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state);

  // extract local values of the global vectors
  std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  if (matrixtofill != nullptr) rotsymmpbc.rotate_my_values_if_necessary(mymatrix);

  for (int inode = 0; inode < nen_; ++inode)  // number of nodes
  {
    // fill a vector field via a pointer
    if (matrixtofill != nullptr)
    {
      for (int idim = 0; idim < nsd_; ++idim)  // number of dimensions
      {
        (*matrixtofill)(idim, inode) = mymatrix[idim + (inode * numdofpernode_)];
      }  // end for(idim)
    }
    // fill a scalar field via a pointer
    if (vectortofill != nullptr)
      (*vectortofill)(inode, 0) = mymatrix[nsd_ + (inode * numdofpernode_)];
  }
}


/*--------------------------------------------------------------------------------
 * additional output for turbulent channel flow                    rasthofer 12/10
 * -> dissipation
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_dissipation(Fluid* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> mat)
{
  //----------------------------------------------------------------------
  // get all nodal values
  // ---------------------------------------------------------------------
  if (not fldparatimint_->is_genalpha())
    FOUR_C_THROW("this routine supports only GenAlpha currently");
  // call routine for calculation of body force in element nodes,
  // with pressure gradient prescribed as body force included for turbulent
  // channel flow and with scatra body force included for variable-density flow
  // (evaluation at time n+alpha_F for generalized-alpha scheme,
  //  and at time n+1 otherwise)
  Core::LinAlg::Matrix<nsd_, nen_> ebofoaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> eprescpgaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escabofoaf(Core::LinAlg::Initialization::zero);
  body_force(ele, ebofoaf, eprescpgaf, escabofoaf);

  // if not available, the arrays for the subscale quantities have to be
  // resized and initialised to zero
  if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)
    FOUR_C_THROW("Time-dependent subgrid scales not supported");

  // get all general state vectors: velocity/pressure, scalar,
  // acceleration/scalar time derivative and history
  // velocity/pressure and scalar values are at time n+alpha_F/n+alpha_M
  // for generalized-alpha scheme and at time n+1/n for all other schemes
  // acceleration/scalar time derivative values are at time n+alpha_M for
  // generalized-alpha scheme and at time n+1 for all other schemes
  // fill the local element vector/matrix with the global values
  // af_genalpha: velocity/pressure at time n+alpha_F and n+alpha_M
  // np_genalpha: velocity at time n+alpha_F, pressure at time n+1
  // ost:         velocity/pressure at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelaf, &epreaf, "velaf");

  Core::LinAlg::Matrix<nsd_, nen_> evelam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> epream(Core::LinAlg::Initialization::zero);
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible &&
      fldparatimint_->is_genalpha())
  {
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelam, &epream, "velam");
  }
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes &&
      fldparatimint_->is_genalpha())
  {
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelam, &epream, "velam");
  }

  // np_genalpha: additional vector for velocity at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> eprenp(Core::LinAlg::Initialization::zero);
  if (fldparatimint_->is_genalpha_np())
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelnp, &eprenp, "velnp");

  Core::LinAlg::Matrix<nen_, 1> escaaf(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, nullptr, &escaaf, "scaaf");

  Core::LinAlg::Matrix<nsd_, nen_> emhist(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &emhist, nullptr, "hist");

  Core::LinAlg::Matrix<nsd_, nen_> eaccam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escadtam(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &eaccam, &escadtam, "accam");

  Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> escaam(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &eveln, &escaam, "scaam");

  if (fldparatimint_->is_genalpha())
    eveln.clear();
  else
    eaccam.clear();

  if (fldpara_->is_reconstruct_der())
  {
    const std::shared_ptr<Core::LinAlg::MultiVector<double>> velafgrad =
        params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("velafgrad");
    Core::FE::extract_my_node_based_values(ele, evelafgrad_, *velafgrad, nsd_ * nsd_);
  }

  // get additional state vectors for ALE case: grid displacement and vel.
  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);

  if (ele->is_ale())
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &egridv, nullptr, "gridv");
  }

  // get additional state vector for AVM3 case and multifractal subgrid scales:
  // fine-scale velocity values are at time n+alpha_F for generalized-alpha
  // scheme and at time n+1 for all other schemes
  Core::LinAlg::Matrix<nsd_, nen_> fsevelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> fsescaaf(Core::LinAlg::Initialization::zero);
  if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv or
      fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &fsevelaf, nullptr, "fsvelaf");
    if (fldpara_->physical_type() == Inpar::FLUID::loma and
        fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, nullptr, &fsescaaf, "fsscaaf");
  }

  // get node coordinates and number of elements per node
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // set element id
  eid_ = ele->id();

  // flag for higher order elements
  is_higher_order_ele_ = IsHigherOrder<distype>::ishigherorder;
  // overrule higher_order_ele if input-parameter is set
  // this might be interesting for fast (but slightly
  // less accurate) computations
  if (fldpara_->is_inconsistent() == true) is_higher_order_ele_ = false;

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

  double Ci_delta_sq = 0.0;
  double Cs_delta_sq = 0.0;
  visceff_ = 0.0;

  // remember the layer of averaging for the dynamic Smagorinsky model
  int smaglayer = 0;

  double CsDeltaSq = 0.0;
  double CiDeltaSq = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
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
  get_turbulence_params(turbmodelparams, Cs_delta_sq, Ci_delta_sq, smaglayer, CsDeltaSq, CiDeltaSq);


  //----------------------------------------------------------------------
  // prepare mean values
  // ---------------------------------------------------------------------

  // the coordinates of the element layers in the channel
  // planecoords are named nodeplanes in turbulence_statistics_channel!
  std::shared_ptr<std::vector<double>> planecoords =
      params.get<std::shared_ptr<std::vector<double>>>("planecoords_", nullptr);
  if (planecoords == nullptr)
    FOUR_C_THROW("planecoords is null, but need channel_flow_of_height_2\n");

  // this will be the y-coordinate of a point in the element interior
  double center = 0.0;
  // get node coordinates of element
  for (int inode = 0; inode < ele->num_node(); inode++) center += xyze_(1, inode);

  center /= (double)ele->num_node();

  // working arrays for the quantities we want to compute
  Core::LinAlg::Matrix<nsd_, 1> mean_res;
  Core::LinAlg::Matrix<nsd_, 1> mean_sacc;
  Core::LinAlg::Matrix<nsd_, 1> mean_svelaf;
  Core::LinAlg::Matrix<nsd_, 1> mean_res_sq;
  Core::LinAlg::Matrix<nsd_, 1> mean_sacc_sq;
  Core::LinAlg::Matrix<nsd_, 1> mean_svelaf_sq;
  Core::LinAlg::Matrix<nsd_, 1> mean_tauinvsvel;

  Core::LinAlg::Matrix<2 * nsd_, 1> mean_crossstress;
  Core::LinAlg::Matrix<2 * nsd_, 1> mean_reystress;

  double vol = 0.0;

  double h = 0.0;
  double h_bazilevs = 0.0;
  double strle = 0.0;
  double gradle = 0.0;
  double averaged_tauC = 0.0;
  double averaged_tauM = 0.0;

  double abs_res = 0.0;
  double abs_svel = 0.0;

  double mean_resC = 0.0;
  double mean_resC_sq = 0.0;
  double mean_sprenp = 0.0;
  double mean_sprenp_sq = 0.0;

  double eps_visc = 0.0;
  double eps_conv = 0.0;
  double eps_smag = 0.0;
  double eps_avm3 = 0.0;
  double eps_mfs = 0.0;
  double eps_mfscross = 0.0;
  double eps_mfsrey = 0.0;
  double eps_supg = 0.0;
  double eps_cross = 0.0;
  double eps_rey = 0.0;
  double eps_graddiv = 0.0;
  double eps_pspg = 0.0;

  mean_res.clear();
  mean_sacc.clear();
  mean_svelaf.clear();
  mean_res_sq.clear();
  mean_sacc_sq.clear();
  mean_svelaf_sq.clear();
  mean_tauinvsvel.clear();
  mean_crossstress.clear();
  mean_reystress.clear();


  // ---------------------------------------------------------------------
  // calculate volume and evaluate material, tau ... at element center
  // ---------------------------------------------------------------------

  // evaluate shape functions and derivatives at element center
  eval_shape_func_and_derivs_at_ele_center();

  // set element area or volume
  vol = fac_;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // get material parameters at element center
  if (not fldpara_->mat_gp() or not fldpara_->tau_gp())
  {
    get_material_params(mat, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
        thermpressam, thermpressdtaf, thermpressdtam, vol);

    // calculate all-scale or fine-scale subgrid viscosity at element center
    visceff_ = visc_;
    if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
    {
      calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
      // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
      visceff_ += sgvisc_;
    }
    else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
      calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
  }

  // potential evaluation of multifractal subgrid-scales at element center
  // coefficient B of fine-scale velocity
  Core::LinAlg::Matrix<nsd_, 1> B_mfs(Core::LinAlg::Initialization::zero);
  // coefficient D of fine-scale scalar (loma only)
  double D_mfs = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    if (not fldpara_->b_gp())
    {
      // make sure to get material parameters at element center
      if (fldpara_->mat_gp())
        // get_material_params(material,evelaf,epreaf,epream,escaaf,escaam,thermpressaf,thermpressam,thermpressdtam,vol);
        get_material_params(mat, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
            thermpressam, thermpressdtaf, thermpressdtam, vol);

      // provide necessary velocities and gradients at element center
      velint_.multiply(evelaf, funct_);
      fsvelint_.multiply(fsevelaf, funct_);
      vderxy_.multiply_nt(evelaf, derxy_);
      // calculate parameters of multifractal subgrid-scales and, finally,
      // calculate coefficient for multifractal modeling of subgrid velocity
      // if loma, calculate coefficient for multifractal modeling of subgrid scalar
      prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      // clear all velocities and gradients
      velint_.clear();
      fsvelint_.clear();
      vderxy_.clear();
    }
  }


  // calculate stabilization parameter at element center
  if (not fldpara_->tau_gp())
  {
    // get convective velocity at element center for evaluation of
    // stabilization parameter
    velint_.multiply(evelaf, funct_);
    convvelint_.update(velint_);
    if (ele->is_ale()) convvelint_.multiply(-1.0, egridv, funct_, 1.0);

    // calculate stabilization parameters at element center
    calc_stab_parameter(vol);
  }


  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------
  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    //---------------------------------------------------------------
    // evaluate shape functions and derivatives at integration point
    //---------------------------------------------------------------
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    velint_.multiply(evelaf, funct_);

    // get velocity derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    vderxy_.multiply_nt(evelaf, derxy_);

    // get fine-scale velocity and its derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvderxy_.clear();
    }
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      fsvelint_.multiply(fsevelaf, funct_);
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvelint_.clear();
    }

    // get convective velocity at integration point
    // (ALE case handled implicitly here using the (potential
    //  mesh-movement-dependent) convective velocity, avoiding
    //  various ALE terms used to be calculated before)
    convvelint_.update(velint_);
    if (ele->is_ale())
    {
      gridvelint_.multiply(egridv, funct_);
      convvelint_.update(-1.0, gridvelint_, 1.0);
    }

    // get pressure gradient at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    if (fldparatimint_->is_genalpha_np())
      gradp_.multiply(derxy_, eprenp);
    else
      gradp_.multiply(derxy_, epreaf);

    // get bodyforce at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    bodyforce_.multiply(ebofoaf, funct_);
    // get prescribed pressure gradient acting as body force
    // (required for turbulent channel flow)
    generalbodyforce_.multiply(eprescpgaf, funct_);

    // get momentum history data at integration point
    // (only required for one-step-theta and BDF2 time-integration schemes)
    histmom_.multiply(emhist, funct_);

    // evaluation of various partial operators at integration point
    // compute convective term from previous iteration and convective operator
    conv_old_.multiply(vderxy_, convvelint_);

    // compute viscous term from previous iteration and viscous operator
    if (is_higher_order_ele_)
      calc_div_eps(evelaf);
    else
      visc_old_.clear();

    // compute divergence of velocity from previous iteration
    vdiv_ = 0.0;
    if (not fldparatimint_->is_genalpha_np())
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        vdiv_ += vderxy_(idim, idim);
      }
    }
    else
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        // get vdiv at time n+1 for np_genalpha,
        Core::LinAlg::Matrix<nsd_, nsd_> vderxy(Core::LinAlg::Initialization::zero);
        vderxy.multiply_nt(evelnp, derxy_);
        vdiv_ += vderxy(idim, idim);
      }
    }

    // get material parameters at integration point
    if (fldpara_->mat_gp())
    {
      get_material_params(mat, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
          thermpressam, thermpressdtaf, thermpressdtam, vol);

      // calculate all-scale or fine-scale subgrid viscosity at integration point
      visceff_ = visc_;
      if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      {
        calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
        // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
        visceff_ += sgvisc_;
      }
      else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
        calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
    }

    // potential evaluation of coefficient of multifractal subgrid-scales at integration point
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      if (fldpara_->b_gp())
      {
        // make sure to get material parameters at gauss point
        if (not fldpara_->mat_gp())
          get_material_params(mat, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
              thermpressam, thermpressdtaf, thermpressdtam, vol);

        // calculate parameters of multifractal subgrid-scales
        prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      }

      // calculate fine-scale velocity, its derivative and divergence for multifractal subgrid-scale
      // modeling
      for (int idim = 0; idim < nsd_; idim++)
        mffsvelint_(idim, 0) = fsvelint_(idim, 0) * B_mfs(idim, 0);
    }
    else
    {
      mffsvelint_.clear();
      mffsvderxy_.clear();
      mffsvdiv_ = 0.0;
    }

    // calculate stabilization parameter at integration point
    if (fldpara_->tau_gp()) calc_stab_parameter(vol);


    // compute residual of continuity equation
    // residual contains velocity divergence only for incompressible flow
    conres_old_ = vdiv_;

    // following computations only required for variable-density flow at low Mach number
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      compute_gal_rhs_cont_eq(eveln, escaaf, escaam, escadtam, ele->is_ale());
      if (not fldparatimint_->is_genalpha())
        FOUR_C_THROW("Does compute_gal_rhs_cont_eq() for ost really the right thing?");
      // remark: I think the term theta*u^n+1*nabla T^n+1 is missing.
      //         Moreover, the resulting conres_old_ should be multiplied by theta (see
      //         monres_old_).

      // add to residual of continuity equation
      conres_old_ -= rhscon_;

      // compute subgrid-scale part of scalar
      // -> different for generalized-alpha and other time-integration schemes
      compute_subgrid_scale_scalar(escaaf, escaam);

      // update material parameters including subgrid-scale part of scalar
      if (fldpara_->update_mat())
      {
        if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
          update_material_params(mat, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
              thermpressam, mfssgscaint_);
        else
          update_material_params(
              mat, evelaf, epreaf, epream, escaaf, escaam, thermpressaf, thermpressam, sgscaint_);
        visceff_ = visc_;
        if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
            fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
            fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
          visceff_ += sgvisc_;
      }
    }

    // evaluate momentum residual once for all stabilization right hand sides
    if (fldparatimint_->is_genalpha())
    {
      // get acceleration at time n+alpha_M at integration point
      accint_.multiply(eaccam, funct_);

      for (int rr = 0; rr < nsd_; ++rr)
      {
        momres_old_(rr) = densam_ * accint_(rr) + densaf_ * conv_old_(rr) + gradp_(rr) -
                          2 * visceff_ * visc_old_(rr) - densaf_ * bodyforce_(rr) -
                          generalbodyforce_(rr);
      }
      // add consistency terms for MFS if applicable
      multfrac_sub_grid_scales_consistent_residual();
    }
    else
    {
      rhsmom_.update(
          (densn_ / fldparatimint_->dt()), histmom_, densaf_ * fldparatimint_->theta(), bodyforce_);
      // and pressure gradient prescribed as body force
      // caution: not density weighted
      rhsmom_.update(fldparatimint_->theta(), generalbodyforce_, 1.0);
      // compute instationary momentum residual:
      // momres_old = u_(n+1)/dt + theta ( ... ) - histmom_/dt - theta*bodyforce_
      for (int rr = 0; rr < nsd_; ++rr)
      {
        momres_old_(rr) = (densaf_ * velint_(rr) / fldparatimint_->dt() +
                              fldparatimint_->theta() * (densaf_ * conv_old_(rr) + gradp_(rr) -
                                                            2 * visceff_ * visc_old_(rr))) -
                          rhsmom_(rr);
      }
      // add consistency terms for MFS if applicable
      multfrac_sub_grid_scales_consistent_residual();
    }


    //---------------------------------------------------------------
    // element average dissipation and production rates
    //---------------------------------------------------------------

    //---------------------------------------------------------------
    // residual-based subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation by supg-stabilization
    if (fldpara_->supg())
    {
      for (int rr = 0; rr < nsd_; rr++)
      {
        eps_supg += densaf_ * fac_ * tau_(0) * momres_old_(rr, 0) * conv_old_(rr, 0);
      }
    }

    // dissipation by cross-stress-stabilization
    if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none)
    {
      for (int rr = 0; rr < nsd_; rr++)
      {
        eps_cross += densaf_ * fac_ * tau_(0) * velint_(rr, 0) *
                     (momres_old_(0, 0) * vderxy_(rr, 0) + momres_old_(1, 0) * vderxy_(rr, 1) +
                         momres_old_(2, 0) * vderxy_(rr, 2));
      }
    }

    // dissipation by reynolds-stress-stabilization
    if (fldpara_->reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    {
      for (int rr = 0; rr < nsd_; rr++)
      {
        eps_rey -= densaf_ * fac_ * tau_(0) * tau_(0) * momres_old_(rr, 0) *
                   (momres_old_(0, 0) * vderxy_(rr, 0) + momres_old_(1, 0) * vderxy_(rr, 1) +
                       momres_old_(2, 0) * vderxy_(rr, 2));
      }
    }

    // dissipation by pspg-stabilization
    if (fldpara_->pspg())
    {
      for (int rr = 0; rr < nsd_; rr++)
      {
        eps_pspg += fac_ * gradp_(rr, 0) * tau_(1) * momres_old_(rr, 0);
      }
    }

    // dissipation by continuity-stabilization
    if (fldpara_->c_stab())
    {
      eps_graddiv += fac_ * vdiv_ * tau_(2) * conres_old_;
    }

    //---------------------------------------------------------------
    // multifractal subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation multifractal subgrid-scales
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      for (int rr = 0; rr < nsd_; rr++)
      {
        eps_mfs -= densaf_ * fac_ *
                   (mffsvelint_(rr, 0) * conv_old_(rr, 0) +
                       velint_(rr, 0) * (mffsvelint_(0, 0) * vderxy_(rr, 0) +
                                            mffsvelint_(1, 0) * vderxy_(rr, 1) +
                                            mffsvelint_(2, 0) * vderxy_(rr, 2)) +
                       mffsvelint_(rr, 0) * (mffsvelint_(0, 0) * vderxy_(rr, 0) +
                                                mffsvelint_(1, 0) * vderxy_(rr, 1) +
                                                mffsvelint_(2, 0) * vderxy_(rr, 2)));

        eps_mfscross -= densaf_ * fac_ *
                        (mffsvelint_(rr, 0) * conv_old_(rr, 0) +
                            velint_(rr, 0) * (mffsvelint_(0, 0) * vderxy_(rr, 0) +
                                                 mffsvelint_(1, 0) * vderxy_(rr, 1) +
                                                 mffsvelint_(2, 0) * vderxy_(rr, 2)));

        eps_mfsrey -= densaf_ * fac_ * mffsvelint_(rr, 0) *
                      (mffsvelint_(0, 0) * vderxy_(rr, 0) + mffsvelint_(1, 0) * vderxy_(rr, 1) +
                          mffsvelint_(2, 0) * vderxy_(rr, 2));
      }
    }

    //---------------------------------------------------------------
    // small-scale subgrid-viscosity subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation AVM3
    /*
                       /                                \
                      |       /  n+1 \         / n+1 \   |
        2* visc    *  |  eps | du     | , eps | u     |  |
               turb   |       \      /         \     /   |
                       \                                /
  */
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      Core::LinAlg::Matrix<nsd_, nsd_> fstwo_epsilon;
      for (int rr = 0; rr < nsd_; ++rr)
      {
        for (int mm = 0; mm < nsd_; ++mm)
        {
          fstwo_epsilon(rr, mm) = fsvderxy_(rr, mm) + fsvderxy_(mm, rr);
        }
      }
      for (int rr = 0; rr < nsd_; ++rr)
      {
        for (int mm = 0; mm < nsd_; ++mm)
        {
          //          eps_avm3 += 0.5*fssgvisc_*fac_*fstwo_epsilon(rr,mm)*two_epsilon(rr,mm);
          eps_avm3 += 0.5 * fssgvisc_ * fac_ * fstwo_epsilon(rr, mm) * fstwo_epsilon(rr, mm);
        }
      }
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
      {
        FOUR_C_THROW("Read warning before usage!");
        // Warning: Here, we should use the deviatoric part of the strain-rate tensor.
        //          However, I think this is not done in the element Sysmat-routine.
        //          Hence, I skipped it here.
      }
    }

    //---------------------------------------------------------------
    // Smagorinsky model
    //---------------------------------------------------------------

    // dissipation (Smagorinsky)
    /*
                       /                                \
                      |       / n+1 \         / n+1 \   |
        2* visc    *  |  eps | u     | , eps | u     |  |
               turb   |       \     /         \     /   |
                       \                                /
  */
    Core::LinAlg::Matrix<nsd_, nsd_> two_epsilon;
    for (int rr = 0; rr < nsd_; ++rr)
    {
      for (int mm = 0; mm < nsd_; ++mm)
      {
        two_epsilon(rr, mm) = vderxy_(rr, mm) + vderxy_(mm, rr);
      }
    }
    if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky)
    {
      for (int rr = 0; rr < nsd_; ++rr)
      {
        for (int mm = 0; mm < nsd_; ++mm)
        {
          eps_smag += 0.5 * sgvisc_ * fac_ * two_epsilon(rr, mm) * two_epsilon(rr, mm);
        }
      }
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
        eps_smag -= (2.0 / 3.0) * fac_ * (sgvisc_ * vdiv_ + q_sq_) * vdiv_;
    }


    //---------------------------------------------------------------
    // standard Galerkin terms
    //---------------------------------------------------------------

    // convective (Galerkin)
    /*
               /                          \
              |   n+1   / n+1 \   /  n+1\  |
              |  u    , | u   | o | u   |  |
              |         \     /   \     /  |
               \                          /
  */
    for (int rr = 0; rr < nsd_; rr++)
    {
      eps_conv -= densaf_ * fac_ * velint_(rr, 0) *
                  (velint_(0, 0) * vderxy_(rr, 0) + velint_(1, 0) * vderxy_(rr, 1) +
                      velint_(2, 0) * vderxy_(rr, 2));
    }

    // dissipation (Galerkin)
    /*
                   /                                \
                  |       / n+1 \         / n+1 \   |
        2* visc * |  eps | u     | , eps | u     |  |
                  |       \     /         \     /   |
                   \                                /
  */
    for (int rr = 0; rr < nsd_; ++rr)
    {
      for (int mm = 0; mm < nsd_; ++mm)
      {
        eps_visc += 0.5 * visc_ * fac_ * two_epsilon(rr, mm) * two_epsilon(rr, mm);
      }
    }
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
      eps_visc -= (2.0 / 3.0) * visc_ * fac_ * vdiv_ * vdiv_;


    //---------------------------------------------------------------
    // reference length for stabilization parameters
    //---------------------------------------------------------------
    // volume based element size
    double hk = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);
    h += fac_ * hk;

    // streamlength based element size
    // The stream length is not correct for xwall as calculated here, because the virtual (enriched)
    // dofs don't have any geometric meaning
    if (enrtype == Discret::Elements::Fluid::xwall)
      strle = 10000000;
    else
    {
      const double vel_norm = velint_.norm2();

      // this copy of velintaf_ will be used to store the normed velocity
      Core::LinAlg::Matrix<3, 1> normed_velint;

      // normed velocity at element center (we use the copy for safety reasons!)
      if (vel_norm >= 1e-6)
      {
        for (int rr = 0; rr < 3; ++rr) /* loop element nodes */
        {
          normed_velint(rr) = velint_(rr) / vel_norm;
        }
      }
      else
      {
        normed_velint(0) = 1.;
        for (int rr = 1; rr < 3; ++rr) /* loop element nodes */
        {
          normed_velint(rr) = 0.0;
        }
      }

      // get streamlength
      double val = 0.0;
      for (int rr = 0; rr < nen_; ++rr) /* loop element nodes */
      {
        val += fabs(normed_velint(0) * derxy_(0, rr) + normed_velint(1) * derxy_(1, rr) +
                    normed_velint(2) * derxy_(2, rr));
      } /* end of loop over element nodes */
      strle += 2.0 / val * fac_;
    }

    // element size in main gradient direction
    {
      // this copy of velintaf_ will be used to store the normed velocity
      Core::LinAlg::Matrix<3, 1> normed_velgrad;

      for (int rr = 0; rr < 3; ++rr)
      {
        normed_velgrad(rr) =
            sqrt(vderxy_(0, rr) * vderxy_(0, rr) + vderxy_(1, rr) * vderxy_(1, rr) +
                 vderxy_(2, rr) * vderxy_(2, rr));
      }
      double norm = normed_velgrad.norm2();

      // normed gradient
      if (norm > 1e-6)
      {
        for (int rr = 0; rr < 3; ++rr)
        {
          normed_velgrad(rr) /= norm;
        }
      }
      else
      {
        normed_velgrad(0) = 1.;
        for (int rr = 1; rr < 3; ++rr)
        {
          normed_velgrad(rr) = 0.0;
        }
      }

      // get length in this direction
      double val = 0.0;
      for (int rr = 0; rr < nen_; ++rr) /* loop element nodes */
      {
        val += fabs(normed_velgrad(0) * derxy_(0, rr) + normed_velgrad(1) * derxy_(1, rr) +
                    normed_velgrad(2) * derxy_(2, rr));
      } /* end of loop over element nodes */
      gradle += 2.0 / val * fac_;
    }

    {
      /*          +-           -+   +-           -+   +-           -+
                |             |   |             |   |             |
                |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
          G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
           ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                |    i     j  |   |    i     j  |   |    i     j  |
                +-           -+   +-           -+   +-           -+
    */
      Core::LinAlg::Matrix<3, 3> G;

      for (int nn = 0; nn < 3; ++nn)
      {
        for (int rr = 0; rr < 3; ++rr)
        {
          G(nn, rr) = xji_(nn, 0) * xji_(rr, 0);
          for (int mm = 1; mm < 3; ++mm)
          {
            G(nn, rr) += xji_(nn, mm) * xji_(rr, mm);
          }
        }
      }

      /*          +----
                 \
        G : G =   +   G   * G
        -   -    /     ij    ij
        -   -   +----
                 i,j
    */
      double normG = 0;
      for (int nn = 0; nn < 3; ++nn)
      {
        for (int rr = 0; rr < 3; ++rr)
        {
          normG += G(nn, rr) * G(nn, rr);
        }
      }

      h_bazilevs += 1. / sqrt(sqrt(normG)) * fac_;
    }


    //---------------------------------------------------------------
    // element averages of residual and subgrid scales
    //---------------------------------------------------------------
    for (int rr = 0; rr < 3; ++rr)
    {
      mean_res(rr) += momres_old_(rr) * fac_;
      mean_res_sq(rr) += momres_old_(rr) * momres_old_(rr) * fac_;
    }
    abs_res += sqrt(momres_old_(0) * momres_old_(0) + momres_old_(1) * momres_old_(1) +
                    momres_old_(2) * momres_old_(2)) *
               fac_;

    for (int rr = 0; rr < 3; ++rr)
    {
      const double aux = tau_(0) * momres_old_(rr);

      mean_svelaf(rr) -= aux * fac_;
      mean_svelaf_sq(rr) += aux * aux * fac_;
    }

    abs_svel += sqrt(momres_old_(0) * momres_old_(0) + momres_old_(1) * momres_old_(1) +
                     momres_old_(2) * momres_old_(2)) *
                tau_(0) * fac_;

    for (int rr = 0; rr < 3; ++rr)
    {
      mean_tauinvsvel(rr) += mean_svelaf(rr) / tau_(0);
    }


    {
      const double aux = tau_(2) * conres_old_;

      mean_sprenp -= aux * fac_;
      mean_sprenp_sq += aux * aux * fac_;
    }


    //---------------------------------------------------------------
    // element averages of cross stresses and cross stresses
    //---------------------------------------------------------------
    if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none)
    {
      mean_crossstress(0) +=
          fac_ * tau_(0) * (momres_old_(0) * velint_(0) + velint_(0) * momres_old_(0));
      mean_crossstress(1) +=
          fac_ * tau_(0) * (momres_old_(1) * velint_(1) + velint_(1) * momres_old_(1));
      mean_crossstress(2) +=
          fac_ * tau_(0) * (momres_old_(2) * velint_(2) + velint_(2) * momres_old_(2));
      mean_crossstress(3) +=
          fac_ * tau_(0) * (momres_old_(0) * velint_(1) + velint_(0) * momres_old_(1));
      mean_crossstress(4) +=
          fac_ * tau_(0) * (momres_old_(1) * velint_(2) + velint_(1) * momres_old_(2));
      mean_crossstress(5) +=
          fac_ * tau_(0) * (momres_old_(2) * velint_(0) + velint_(2) * momres_old_(0));
    }

    if (fldpara_->reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    {
      mean_reystress(0) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(0) * momres_old_(0) + momres_old_(0) * momres_old_(0));
      mean_reystress(1) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(1) * momres_old_(1) + momres_old_(1) * momres_old_(1));
      mean_reystress(2) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(2) * momres_old_(2) + momres_old_(2) * momres_old_(2));
      mean_reystress(3) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(0) * momres_old_(1) + momres_old_(1) * momres_old_(0));
      mean_reystress(4) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(1) * momres_old_(2) + momres_old_(2) * momres_old_(1));
      mean_reystress(5) -= fac_ * tau_(0) * tau_(0) *
                           (momres_old_(2) * momres_old_(0) + momres_old_(0) * momres_old_(2));
    }


    //---------------------------------------------------------------
    // element averages of tau_Mu and tau_C
    //---------------------------------------------------------------
    averaged_tauM += tau_(0) * fac_;
    averaged_tauC += tau_(2) * fac_;

    mean_resC += conres_old_ * fac_;
    mean_resC_sq += conres_old_ * conres_old_ * fac_;
  }  // end integration loop


  for (int rr = 0; rr < 3; ++rr)
  {
    mean_res(rr) /= vol;
    mean_res_sq(rr) /= vol;
    mean_sacc(rr) /= vol;
    mean_sacc_sq(rr) /= vol;
    mean_svelaf(rr) /= vol;
    mean_svelaf_sq(rr) /= vol;
    mean_tauinvsvel(rr) /= vol;
  }


  for (int rr = 0; rr < 6; ++rr)
  {
    mean_crossstress(rr) /= vol;
    mean_reystress(rr) /= vol;
  }

  abs_res /= vol;
  abs_svel /= vol;

  mean_resC /= vol;
  mean_resC_sq /= vol;
  mean_sprenp /= vol;
  mean_sprenp_sq /= vol;

  h /= vol;
  h_bazilevs /= vol;
  strle /= vol;
  gradle /= vol;

  averaged_tauC /= vol;
  averaged_tauM /= vol;

  eps_visc /= vol;
  eps_conv /= vol;
  eps_smag /= vol;
  eps_avm3 /= vol;
  eps_mfs /= vol;
  eps_mfscross /= vol;
  eps_mfsrey /= vol;
  eps_supg /= vol;
  eps_cross /= vol;
  eps_rey /= vol;
  eps_graddiv /= vol;
  eps_pspg /= vol;

  std::shared_ptr<std::vector<double>> incrvol =
      params.get<std::shared_ptr<std::vector<double>>>("incrvol");

  std::shared_ptr<std::vector<double>> incr_eps_visc =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_visc");
  std::shared_ptr<std::vector<double>> incr_eps_conv =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_conv");
  std::shared_ptr<std::vector<double>> incr_eps_smag =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_eddyvisc");
  std::shared_ptr<std::vector<double>> incr_eps_avm3 =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_avm3");
  std::shared_ptr<std::vector<double>> incr_eps_mfs =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_mfs");
  std::shared_ptr<std::vector<double>> incr_eps_mfscross =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_mfscross");
  std::shared_ptr<std::vector<double>> incr_eps_mfsrey =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_mfsrey");
  std::shared_ptr<std::vector<double>> incr_eps_supg =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_supg");
  std::shared_ptr<std::vector<double>> incr_eps_cross =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_cross");
  std::shared_ptr<std::vector<double>> incr_eps_rey =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_rey");
  std::shared_ptr<std::vector<double>> incr_eps_graddiv =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_graddiv");
  std::shared_ptr<std::vector<double>> incr_eps_pspg =
      params.get<std::shared_ptr<std::vector<double>>>("incr_eps_pspg");

  std::shared_ptr<std::vector<double>> incrhk =
      params.get<std::shared_ptr<std::vector<double>>>("incrhk");
  std::shared_ptr<std::vector<double>> incrhbazilevs =
      params.get<std::shared_ptr<std::vector<double>>>("incrhbazilevs");
  std::shared_ptr<std::vector<double>> incrstrle =
      params.get<std::shared_ptr<std::vector<double>>>("incrstrle");
  std::shared_ptr<std::vector<double>> incrgradle =
      params.get<std::shared_ptr<std::vector<double>>>("incrgradle");

  std::shared_ptr<std::vector<double>> incrmk =
      params.get<std::shared_ptr<std::vector<double>>>("incrmk");

  std::shared_ptr<std::vector<double>> incrres =
      params.get<std::shared_ptr<std::vector<double>>>("incrres");
  std::shared_ptr<std::vector<double>> incrres_sq =
      params.get<std::shared_ptr<std::vector<double>>>("incrres_sq");
  std::shared_ptr<std::vector<double>> incrabsres =
      params.get<std::shared_ptr<std::vector<double>>>("incrabsres");
  std::shared_ptr<std::vector<double>> incrtauinvsvel =
      params.get<std::shared_ptr<std::vector<double>>>("incrtauinvsvel");

  std::shared_ptr<std::vector<double>> incrsvelaf =
      params.get<std::shared_ptr<std::vector<double>>>("incrsvelaf");
  std::shared_ptr<std::vector<double>> incrsvelaf_sq =
      params.get<std::shared_ptr<std::vector<double>>>("incrsvelaf_sq");
  std::shared_ptr<std::vector<double>> incrabssvelaf =
      params.get<std::shared_ptr<std::vector<double>>>("incrabssvelaf");

  std::shared_ptr<std::vector<double>> incrresC =
      params.get<std::shared_ptr<std::vector<double>>>("incrresC");
  std::shared_ptr<std::vector<double>> incrresC_sq =
      params.get<std::shared_ptr<std::vector<double>>>("incrresC_sq");
  std::shared_ptr<std::vector<double>> spressnp =
      params.get<std::shared_ptr<std::vector<double>>>("incrspressnp");
  std::shared_ptr<std::vector<double>> spressnp_sq =
      params.get<std::shared_ptr<std::vector<double>>>("incrspressnp_sq");

  std::shared_ptr<std::vector<double>> incrtauC =
      params.get<std::shared_ptr<std::vector<double>>>("incrtauC");
  std::shared_ptr<std::vector<double>> incrtauM =
      params.get<std::shared_ptr<std::vector<double>>>("incrtauM");

  std::shared_ptr<std::vector<double>> incrcrossstress =
      params.get<std::shared_ptr<std::vector<double>>>("incrcrossstress");
  std::shared_ptr<std::vector<double>> incrreystress =
      params.get<std::shared_ptr<std::vector<double>>>("incrreystress");

  bool found = false;

  int nlayer = 0;
  for (nlayer = 0; nlayer < (int)(*planecoords).size() - 1;)
  {
    if (center < (*planecoords)[nlayer + 1])
    {
      found = true;
      break;
    }
    nlayer++;
  }
  if (found == false)
  {
    FOUR_C_THROW("could not determine element layer");
  }

  // collect layer volume
  (*incrvol)[nlayer] += vol;

  // element length in stabilisation parameter
  (*incrhk)[nlayer] += h;

  // element length in viscous regime defined by the Bazilevs parameter
  (*incrhbazilevs)[nlayer] += h_bazilevs;

  // stream length
  (*incrstrle)[nlayer] += strle;

  // gradient based element length
  (*incrgradle)[nlayer] += gradle;

  // averages of stabilisation parameters
  (*incrtauC)[nlayer] += averaged_tauC;
  (*incrtauM)[nlayer] += averaged_tauM;

  // element mk in stabilisation parameter
  (*incrmk)[nlayer] += get_mk();

  // averages of momentum residuals, subscale velocity and accelerations
  for (int mm = 0; mm < 3; ++mm)
  {
    (*incrres)[3 * nlayer + mm] += mean_res(mm);
    (*incrres_sq)[3 * nlayer + mm] += mean_res_sq(mm);

    (*incrsvelaf)[3 * nlayer + mm] += mean_svelaf(mm);
    (*incrsvelaf_sq)[3 * nlayer + mm] += mean_svelaf_sq(mm);

    (*incrtauinvsvel)[3 * nlayer + mm] += mean_tauinvsvel(mm);
  }

  (*incrabsres)[nlayer] += abs_res;
  (*incrabssvelaf)[nlayer] += abs_svel;

  // averages of subscale pressure and continuity residuals
  (*incrresC)[nlayer] += mean_resC;
  (*incrresC_sq)[nlayer] += mean_resC_sq;

  (*spressnp)[nlayer] += mean_sprenp;
  (*spressnp_sq)[nlayer] += mean_sprenp_sq;


  (*incr_eps_visc)[nlayer] += eps_visc;
  (*incr_eps_conv)[nlayer] += eps_conv;
  (*incr_eps_smag)[nlayer] += eps_smag;
  (*incr_eps_avm3)[nlayer] += eps_avm3;
  (*incr_eps_mfs)[nlayer] += eps_mfs;
  (*incr_eps_mfscross)[nlayer] += eps_mfscross;
  (*incr_eps_mfsrey)[nlayer] += eps_mfsrey;
  (*incr_eps_supg)[nlayer] += eps_supg;
  (*incr_eps_cross)[nlayer] += eps_cross;
  (*incr_eps_rey)[nlayer] += eps_rey;
  (*incr_eps_graddiv)[nlayer] += eps_graddiv;
  (*incr_eps_pspg)[nlayer] += eps_pspg;

  // averages of subgrid stress tensors
  for (int mm = 0; mm < 6; ++mm)
  {
    (*incrcrossstress)[6 * nlayer + mm] += mean_crossstress(mm);
    (*incrreystress)[6 * nlayer + mm] += mean_reystress(mm);
  }

  return 0;
}


/*!
      \brief do finite difference check for given element ID
             --> for debugging purposes only

      \param ele              (i) the element those matrix is calculated
                                  (pass-through)
      \param evelaf           (i) nodal velocities at n+alpha_F/n+1 (pass-through)
      \param eveln            (i) nodal velocities at n (pass-through)
      \param fsevelaf         (i) fine-scale nodal velocities at n+alpha_F/n+1
                                  (pass-through)
      \param epreaf           (i) nodal pressure at n+alpha_F/n+1 (pass-through)
      \param eaccam           (i) nodal accelerations at n+alpha_M (pass-through)
      \param escaaf           (i) nodal scalar at n+alpha_F/n+1 (pass-through)
      \param escaam           (i) nodal scalar at n+alpha_M/n (pass-through)
      \param escadtam         (i) nodal scalar derivatives at n+alpha_M/n+1
                                  (pass-through)
      \param emhist           (i) time rhs for momentum equation (pass-through)
      \param edispnp          (i) nodal displacements (on moving mesh)
                                  (pass-through)
      \param egridv           (i) grid velocity (on moving mesh) (pass-through)
      \param estif            (i) element matrix to calculate (pass-through)
      \param emesh            (i) linearization wrt mesh motion (pass-through)
      \param eforce           (i) element rhs to calculate (pass-through)
      \param material         (i) fluid material (pass-through)
      \param time             (i) current simulation time (pass-through)
      \param timefac          (i) time discretization factor (pass-through)
      \param newton           (i) boolean flag for linearisation (pass-through)
      \param loma             (i) boolean flag for potential low-Mach-number solver
                                  (pass-through)
      \param conservative     (i) boolean flag for conservative form (pass-through)
      \param is_genalpha      (i) boolean flag for generalized-alpha time
                                  integration (pass-through)
      \param higher_order_ele (i) keep or drop second derivatives (pass-through)
      \param fssgv            (i) flag for type of fine-scale subgrid viscosity
                                  (pass-through)
      \param pspg             (i) boolean flag for stabilisation (pass-through)
      \param supg             (i) boolean flag for stabilisation (pass-through)
      \param vstab            (i) boolean flag for stabilisation (pass-through)
      \param graddiv            (i) boolean flag for stabilisation (pass-through)
      \param cross            (i) boolean flag for stabilisation (pass-through)
      \param reynolds         (i) boolean flag for stabilisation (pass-through)
      \param turb_mod_action  (i) selecting turbulence model (none, Smagorisky,
                                  dynamic Smagorinsky, Smagorinsky with van Driest
                                  damping for channel flows) (pass-through)
      \param Cs               (i) Smagorinsky model parameter (pass-through)
      \param Cs_delta_sq      (i) Model parameter computed by dynamic Smagorinsky
                                  approach (Cs*h*h) (pass-through)
      \param l_tau            (i) viscous length scale, required for van driest
                                  damping function and defined on input (pass-through)
*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::f_dcheck(
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& eveln,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    const Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& estif,
    const Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    const Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    const std::shared_ptr<const Core::Mat::Material> material, const double timefac,
    const double& Cs, const double& Cs_delta_sq, const double& l_tau)
{
  // magnitude of dof perturbation
  const double epsilon = 1e-14;

  if (fldparatimint_->is_genalpha_np()) FOUR_C_THROW("FD check not available for NP genalpha!!");

  // make a copy of all input parameters potentially modified by Sysmat
  // call --- they are not intended to be modified
  //  double copy_Cs         =Cs;
  //  double copy_Cs_delta_sq=Cs_delta_sq;
  //  double copy_l_tau      =l_tau;

  std::shared_ptr<const Core::Mat::Material> copy_material = material;

  // allocate arrays to compute element matrices and vectors at perturbed
  // positions
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> checkmat1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_> checkmat2(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1> checkvec1(Core::LinAlg::Initialization::zero);

  // alloc the vectors that will contain the perturbed velocities or
  // pressures
  Core::LinAlg::Matrix<nsd_, nen_> checkevelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> checkeaccam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> checkepreaf(Core::LinAlg::Initialization::zero);

  // echo to screen
  printf("+-------------------------------------------+\n");
  printf("| FINITE DIFFERENCE CHECK FOR ELEMENT %5d |\n", eid_);
  printf("+-------------------------------------------+\n");
  printf("\n");
  // loop columns of matrix by looping nodes and then dof per nodes

  // loop nodes
  for (int nn = 0; nn < nen_; ++nn)
  {
    printf("-------------------------------------\n");
    printf("-------------------------------------\n");
    printf("NODE of element local id %d\n", nn);
    // loop dofs
    for (int rr = 0; rr < (nsd_ + 1); ++rr)
    {
      // number of the matrix column to check
      int dof = nn * (nsd_ + 1) + rr;

      // clear element matrices and vectors to assemble
      checkmat1.clear();
      checkmat2.clear();
      checkvec1.clear();

      // copy velocities and pressures to perturbed arrays
      for (int mm = 0; mm < nen_; ++mm)
      {
        for (int dim = 0; dim < nsd_; ++dim)
        {
          checkevelaf(dim, mm) = evelaf(dim, mm);

          checkeaccam(dim, mm) = eaccam(dim, mm);
        }

        checkepreaf(mm) = epreaf(mm);
      }

      // perturb the respective elemental quantities
      if (rr == nsd_)
      {
        printf("pressure dof (%d) %f\n", nn, epsilon);

        if (fldparatimint_->is_genalpha())
        {
          checkepreaf(nn) += fldparatimint_->alpha_f() * epsilon;
        }
        else
        {
          checkepreaf(nn) += epsilon;
        }
      }
      else
      {
        printf("velocity dof %d (%d)\n", rr, nn);

        if (fldparatimint_->is_genalpha())
        {
          checkevelaf(rr, nn) += fldparatimint_->alpha_f() * epsilon;
          checkeaccam(rr, nn) += fldparatimint_->alpha_m() /
                                 (fldparatimint_->gamma() * fldparatimint_->dt()) * epsilon;
        }
        else
        {
          checkevelaf(rr, nn) += epsilon;
        }
      }

      // TODO: Andi
      // calculate the right hand side for the perturbed vector
      //      Sysmat2D3D(checkevelaf,
      //                 eveln,
      //                 fsevelaf,
      //                 checkepreaf,
      //                 checkeaccam,
      //                 escaaf,
      //                 escaam,
      //                 escadtam,
      //                 emhist,
      //                 edispnp,
      //                 egridv,
      //                 checkmat1,
      //                 checkmat2,
      //                 checkvec1,
      //                 thermpressaf,
      //                 thermpressam,
      //                 thermpressdtaf,
      //                 thermpressdtam,
      //                 copy_material,
      //                 timefac,
      //                 copy_Cs,
      //                 copy_Cs_delta_sq,
      //                 copy_l_tau);

      // compare the difference between linear approximation and
      // (nonlinear) right hand side evaluation

      // note that it makes more sense to compare these quantities
      // than to compare the matrix entry to the difference of the
      // the right hand sides --- the latter causes numerical problems
      // do to deletion

      for (int mm = 0; mm < (nsd_ + 1) * nen_; ++mm)
      {
        double val;
        double lin;
        double nonlin;

        // For af-generalized-alpha scheme, the residual vector for the
        // solution rhs is scaled on the time-integration level...
        if (fldparatimint_->is_genalpha())
        {
          val = -(eforce(mm) / (epsilon)) * (fldparatimint_->gamma() * fldparatimint_->dt()) /
                (fldparatimint_->alpha_m());
          lin = -(eforce(mm) / (epsilon)) * (fldparatimint_->gamma() * fldparatimint_->dt()) /
                    (fldparatimint_->alpha_m()) +
                estif(mm, dof);
          nonlin = -(checkvec1(mm) / (epsilon)) * (fldparatimint_->gamma() * fldparatimint_->dt()) /
                   (fldparatimint_->alpha_m());
        }
        else
        {
          val = -eforce(mm) / epsilon;
          lin = -eforce(mm) / epsilon + estif(mm, dof);
          nonlin = -checkvec1(mm) / epsilon;
        }

        double norm = abs(lin);
        if (norm < 1e-12)
        {
          norm = 1e-12;
        }

        // output to screen
        printf("relerr         %+12.5e ", (lin - nonlin) / norm);
        printf("abserr         %+12.5e ", lin - nonlin);
        printf("orig. value    %+12.5e ", val);
        printf("lin. approx.   %+12.5e ", lin);
        printf("nonlin. funct. %+12.5e ", nonlin);
        printf("matrix entry   %+12.5e ", estif(mm, dof));
        printf("\n");
      }
    }
  }

  return;
}


/*-------------------------------------------------------------------------------*
 |find elements of inflow section                                rasthofer 10/12 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::inflow_element(Core::Elements::Element* ele)
{
  is_inflow_ele_ = false;

  std::vector<const Core::Conditions::Condition*> myinflowcond;

  // check whether all nodes have a unique inflow condition
  Core::Conditions::find_element_conditions(ele, "TurbulentInflowSection", myinflowcond);
  if (myinflowcond.size() > 1) FOUR_C_THROW("More than one inflow condition on one node!");

  if (myinflowcond.size() == 1) is_inflow_ele_ = true;

  return;
}

/*-----------------------------------------------------------------------------*
 | Calculate element mass matrix                              la spina 06/2017 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_mass_matrix(
    Discret::Elements::Fluid* ele,
    //    Teuchos::ParameterList&              params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1)
{
  // set element id
  eid_ = ele->id();

  // ---------------------------------------------------------------------------
  // Prepare material parameters
  // ---------------------------------------------------------------------------
  // Since we need only the density, we use a lot of dummy values.

  // create dummy matrices
  Core::LinAlg::Matrix<nsd_, nen_> mat1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> mat2(Core::LinAlg::Initialization::zero);

  get_material_params(mat, mat1, mat2, mat2, mat2, mat2, mat2, 0.0, 0.0, 0.0, 0.0, 0.0);

  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // Do ALE specific updates if necessary
  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(
        discretization, lm, *rotsymmpbc_, &edispnp, nullptr, "dispnp");

    // get new node positions of ALE mesh
    xyze_ += edispnp;
  }

  // definition of matrices
  Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_> estif_u(Core::LinAlg::Initialization::zero);

  // ---------------------------------------------------------------------------
  // Integration loop
  // ---------------------------------------------------------------------------
  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          for (int idim = 0; idim < nsd_; ++idim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) += funct_(vi) * funct_(ui) * fac_ * densaf_;
          }  // end for (idim)
        }  // end for (jdim)
      }  // end for (vi)
    }  // end for (ui)
  }  // end of integration loop

  // ---------------------------------------------------------------------------
  // Add velocity-velocity part to matrix
  // ---------------------------------------------------------------------------
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_ * ui;
    const int nsd_ui = nsd_ * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int numdof_ui_jdim = numdof_ui + jdim;
      const int nsd_ui_jdim = nsd_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = numdofpernode_ * vi;
        const int nsd_vi = nsd_ * vi;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          elemat1(numdof_vi + idim, numdof_ui_jdim) += estif_u(nsd_vi + idim, nsd_ui_jdim);
        }  // end for (idim)
      }  // end for (vi)
    }  // end for (jdim)
  }  // end for (ui)

  // add terms associated to pressure dofs for weakly_compressible flows
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible)
  {
    FOUR_C_THROW("Evaluation of the mass matrix for pressure dofs");
    // check fluid material
    if (mat->material_type() != Core::Materials::m_fluid_murnaghantait)
    {
      FOUR_C_THROW(
          "The evaluation of the mass matrix for pressure dofs is implemented only for "
          "Murnaghan-Tait equation of state");
    }

    // extract fluid material parameters
    const Mat::MurnaghanTaitFluid* actmat = static_cast<const Mat::MurnaghanTaitFluid*>(mat.get());
    double RefPressure = actmat->ref_pressure();         // reference pressure
    double RefBulkModulus = actmat->ref_bulk_modulus();  // reference bulk modulus
    double MatParameter =
        actmat->mat_parameter();  // material parameter according to Murnaghan-Tait

    // evaluation of the "compressibility factor"
    double compr_fac = 1.0 / (RefBulkModulus + MatParameter * (preaf_ - RefPressure));

    // definition of matrices
    Core::LinAlg::Matrix<nen_, nen_> ppmat(Core::LinAlg::Initialization::zero);

    // ---------------------------------------------------------------------------
    // Integration loop
    // ---------------------------------------------------------------------------
    for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
        ++iquad)
    {
      // evaluate shape functions and derivatives at integration point
      eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          ppmat(vi, ui) += funct_(vi) * funct_(ui) * fac_ * compr_fac;
        }  // end for (vi)
      }  // end for (ui)
    }  // end of integration loop

    // ---------------------------------------------------------------------------
    // Add pressure-pressure part to matrix
    // ---------------------------------------------------------------------------
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int numdof_ui = numdofpernode_ * ui;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = numdofpernode_ * vi;

        elemat1(numdof_vi + nsd_, numdof_ui + nsd_) += ppmat(vi, ui);
      }  // end for (vi)
    }  // end for (ui)
  }

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate channel statistics                                     bk 05/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_channel_statistics(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material>& mat)
{
  // moved here from fluid_ele_evaluate_utils,  f3_calc_means()

  /*!
  \brief calculate spatial mean values for channel flow
  (requires wall parallel layers of elements)

                                                           gammi 07/07


  this method assumes that each 2 dimensional integration element
  in the homogeneous plane is parallel to the wall!!!

  The necessary element integration is done in here. The element
  is cut into two (HEX8) or three (quadratic elements) planes (plus
  additional planes for visualisation purposes, defined by planes
  vector), the spatial functions (velocity, pressure etc.) are
  integrated over this plane and this element contribution is added
  to a processor local vector (see formulas below for a exact
  description of the output).
  It is assumed that the sampling planes are distributed equidistant
  in the element. The result is normalized by the area afterwards


                     ^ normdirect       integration plane
                     |                /
                     |               /
                     |
               +-----|-------------+
              /|     |            /|
             / |     |           / |
            /  |     |          /  |
           /   |     |         /   |
          /    +-----|--------/----+ ---- additional integration
         /    /|     |       /    /|      plane (for quadratic elements)
        /    / |     |      /    / |
       +-------------------+    /  |
       |   /   |     *-----|---+------------>
       |  /    +----/------|--/----+         inplanedirect[1]
       | /    /    /       | /    /
       |/    /    /        |/    /   \
       +---------+---------+    /     \
       |   /    /          |   /       integration plane
       |  /    /           |  /
       | /    /            | /
       |/    /             |/
       +----/--------------+
           /
          /   inplanedirect[0]


  Example for a mean value evaluation:

         1.0       /                     1.0      /
  _               |                              |            detJ
  u = -------- *  | u(x,y,z) dx dz =  -------- * | u(r,s,t) * ---- dr ds
      +---        |                   +---       |             h
       \         / A                   \        /  [-1:1]^2     y
       / area                          / area
      +---                            +---                    +--+
                                                             Jacobi-
                                                           determinant
                                                            computed
                                                         from 3d mapping
                                                       h  is the (constant)
                                                        y
                                                      height of the element

  The method computes:
                      _             _             _             _
             numele * u  , numele * v  , numele * w  , numele * p
                      ___           ___           ___           ___
                       ^2            ^2            ^2            ^2
  and        numele * u  , numele * v  , numele * w  , numele * p

                      _ _           _ _           _ _
  as well as  numele * u*v, numele * u*w, numele * v*w


  as well as numele and the element area.
  All results are communicated via the parameter list!


 */

  // --------------------------------------------------
  // extract velocity and pressure from global
  // distributed vectors
  // --------------------------------------------------
  // velocity and pressure values (n+1)
  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
      discretization.get_state("u and p (n+1,converged)");
  if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velnp'");

  // extract local values from the global vectors
  std::vector<double> mysol = Core::FE::extract_values(*velnp, lm);
  // get view of solution and subgrid-viscosity vector
  Core::LinAlg::Matrix<4 * nen_, 1> sol(mysol.data(), true);

  // the plane normal tells you in which plane the integration takes place
  const int normdirect = params.get<int>("normal direction to homogeneous plane");

  // the vector planes contains the coordinates of the homogeneous planes (in
  // wall normal direction)
  std::shared_ptr<std::vector<double>> planes =
      params.get<std::shared_ptr<std::vector<double>>>("coordinate vector for home. planes");

  // get the pointers to the solution vectors
  std::shared_ptr<std::vector<double>> sumarea =
      params.get<std::shared_ptr<std::vector<double>>>("element layer area");

  std::shared_ptr<std::vector<double>> sumu =
      params.get<std::shared_ptr<std::vector<double>>>("mean velocity u");
  std::shared_ptr<std::vector<double>> sumv =
      params.get<std::shared_ptr<std::vector<double>>>("mean velocity v");
  std::shared_ptr<std::vector<double>> sumw =
      params.get<std::shared_ptr<std::vector<double>>>("mean velocity w");
  std::shared_ptr<std::vector<double>> sump =
      params.get<std::shared_ptr<std::vector<double>>>("mean pressure p");

  std::shared_ptr<std::vector<double>> sumsqu =
      params.get<std::shared_ptr<std::vector<double>>>("mean value u^2");
  std::shared_ptr<std::vector<double>> sumsqv =
      params.get<std::shared_ptr<std::vector<double>>>("mean value v^2");
  std::shared_ptr<std::vector<double>> sumsqw =
      params.get<std::shared_ptr<std::vector<double>>>("mean value w^2");
  std::shared_ptr<std::vector<double>> sumuv =
      params.get<std::shared_ptr<std::vector<double>>>("mean value uv");
  std::shared_ptr<std::vector<double>> sumuw =
      params.get<std::shared_ptr<std::vector<double>>>("mean value uw");
  std::shared_ptr<std::vector<double>> sumvw =
      params.get<std::shared_ptr<std::vector<double>>>("mean value vw");
  std::shared_ptr<std::vector<double>> sumsqp =
      params.get<std::shared_ptr<std::vector<double>>>("mean value p^2");

  // get node coordinates of element
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // Do ALE specific updates if necessary
  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    // get new node positions of ALE mesh
    get_grid_disp_ale(discretization, lm, edispnp);

    for (int inode = 0; inode < nen_; inode++)
    {
      if (abs(edispnp(normdirect, inode)) > 1e-6)
      {
        FOUR_C_THROW("no sampling possible if homogeneous planes are not conserved\n");
      }
    }
  }

  // we have to fill the virtual nodes of the xyze_ matrix, since we only get
  // coordinates of the real nodes
  if (enrtype == Discret::Elements::Fluid::xwall)
  {
    for (int inode = 0; inode < nen_ / 2; inode++)
    {
      for (int sdm = 0; sdm < nsd_; ++sdm)
      {
        xyze_(sdm, inode + nen_ / 2) =
            xyze_(sdm, inode);  // watch out, this is not in the correct order
        // but otherwise we would have to introduce a new copy which is costly
      }
    }
  }

  if (distype == Core::FE::CellType::hex8 || distype == Core::FE::CellType::hex27 ||
      distype == Core::FE::CellType::hex20)
  {
    // decide first, if this element is taken into account!
    double inflowmax = params.get<double>("INFLOW_CHA_SIDE");
    // get the minimum x coordinate of this element and compare with the maximum of the inflow
    // channel
    double minx = 9998.0;
    for (int inode = 0; inode < nen_; inode++)
    {
      if (minx > xyze_(0, inode))
      {
        minx = xyze_(0, inode);
      }
    }
    if (inflowmax < minx) return 0;


    double min = xyze_(normdirect, 0);
    double max = xyze_(normdirect, 0);

    // set maximum and minimum value in wall normal direction
    for (int inode = 0; inode < nen_; inode++)
    {
      if (min > xyze_(normdirect, inode))
      {
        min = xyze_(normdirect, inode);
      }
      if (max < xyze_(normdirect, inode))
      {
        max = xyze_(normdirect, inode);
      }
    }

    // determine the ids of the homogeneous planes intersecting this element
    std::set<int> planesinele;
    for (unsigned nplane = 0; nplane < planes->size(); ++nplane)
    {
      // get all available wall normal coordinates
      for (int nn = 0; nn < nsd_; ++nn)
      {
        if (min - 2e-9 < (*planes)[nplane] && max + 2e-9 > (*planes)[nplane])
        {
          planesinele.insert(nplane);
        }
      }
    }

    // remove lowest layer from planesinele to avoid double calculations. This is not done
    // for the first level (index 0) --- if deleted, shift the first integration point in
    // wall normal direction
    // the shift depends on the number of sampling planes in the element
    double shift = 0;

    // set the number of planes which cut the element
    const int numplanesinele = planesinele.size();

    if (*planesinele.begin() != 0)
    {
      // this is not an element of the lowest element layer
      planesinele.erase(planesinele.begin());

      shift = 2.0 / (static_cast<double>(numplanesinele - 1));
    }
    else
    {
      // this is an element of the lowest element layer. Increase the counter
      // in order to compute the total number of elements in one layer
      int* count = params.get<int*>("count processed elements");

      (*count)++;
    }

    // determine the orientation of the rst system compared to the xyz system
    int elenormdirect = -1;
    bool upsidedown = false;
    // the only thing of interest is how normdirect is oriented in the
    // element coordinate system
    if (xyze_(normdirect, 4) - xyze_(normdirect, 0) > 2e-9)
    {
      // t aligned
      elenormdirect = 2;
    }
    else if (xyze_(normdirect, 3) - xyze_(normdirect, 0) > 2e-9)
    {
      // s aligned
      elenormdirect = 1;
    }
    else if (xyze_(normdirect, 1) - xyze_(normdirect, 0) > 2e-9)
    {
      // r aligned
      elenormdirect = 0;
    }
    else if (xyze_(normdirect, 4) - xyze_(normdirect, 0) < -2e-9)
    {
      // -t aligned
      elenormdirect = 2;
      upsidedown = true;
    }
    else if (xyze_(normdirect, 3) - xyze_(normdirect, 0) < -2e-9)
    {
      // -s aligned
      elenormdirect = 1;
      upsidedown = true;
    }
    else if (xyze_(normdirect, 1) - xyze_(normdirect, 0) < -2e-9)
    {
      // -r aligned
      elenormdirect = 0;
      upsidedown = true;
    }
    else
    {
      FOUR_C_THROW(
          "cannot determine orientation of plane normal in local coordinate system of element");
    }
    std::vector<int> inplanedirect;
    {
      std::set<int> inplanedirectset;
      for (int i = 0; i < 3; ++i)
      {
        inplanedirectset.insert(i);
      }
      inplanedirectset.erase(elenormdirect);

      for (std::set<int>::iterator id = inplanedirectset.begin(); id != inplanedirectset.end();
          ++id)
      {
        inplanedirect.push_back(*id);
      }
    }

    // get the quad9 gaussrule for the in plane integration
    Core::FE::GaussIntegration intpoints(Core::FE::CellType::quad9);

    // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
    // (now we allow even more)
    double layershift = 0;
    //    if(elenormdirect==0)
    //    {
    //      std::cout << "numplanes in ele:   " << numplanesinele<< std::endl;
    //      for(std::set<int>::const_iterator id = planesinele.begin();id!=planesinele.end() ;++id)
    //        std::cout <<"planesinele id:  "<< *id << std::endl;
    //
    //    }
    // loop all levels in element
    for (std::set<int>::const_iterator id = planesinele.begin(); id != planesinele.end(); ++id)
    {
      // reset temporary values
      double area = 0;

      double ubar = 0;
      double vbar = 0;
      double wbar = 0;
      double pbar = 0;

      double usqbar = 0;
      double vsqbar = 0;
      double wsqbar = 0;
      double uvbar = 0;
      double uwbar = 0;
      double vwbar = 0;
      double psqbar = 0;

      // get the integration point in wall normal direction
      double e[3];

      e[elenormdirect] = -1.0 + shift + layershift;
      if (upsidedown)
      {
        e[elenormdirect] *= -1;
      }

      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
          ++iquad)
      {
        // get the other gauss point coordinates
        for (int i = 0; i < 2; ++i)
        {
          e[inplanedirect[i]] = iquad.point()[i];
        }
        {
          const double* econst = e;

          // evaluate shape functions and derivatives at integration point
          eval_shape_func_and_derivs_at_int_point(econst, iquad.weight());
        }

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)
        if (xjm_(elenormdirect, normdirect) < 0)
        {
          xjm_(elenormdirect, normdirect) = -1.0;
        }
        else
        {
          xjm_(elenormdirect, normdirect) = 1.0;
        }

        // get local copy of determinant/ adjusted to plane integration
        const double det =
            xjm_(0, 0) * xjm_(1, 1) * xjm_(2, 2) + xjm_(0, 1) * xjm_(1, 2) * xjm_(2, 0) +
            xjm_(0, 2) * xjm_(1, 0) * xjm_(2, 1) - xjm_(0, 2) * xjm_(1, 1) * xjm_(2, 0) -
            xjm_(0, 0) * xjm_(1, 2) * xjm_(2, 1) - xjm_(0, 1) * xjm_(1, 0) * xjm_(2, 2);

        // check for degenerated elements
        if (det <= 0.0)
        {
          FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nNEGATIVE JACOBIAN DETERMINANT: {}", ele->id(), det);
        }

#ifdef FOUR_C_ENABLE_ASSERTIONS
        // check whether this gausspoint is really inside the desired plane
        {
          double x[3];
          x[0] = 0;
          x[1] = 0;
          x[2] = 0;
          for (int inode = 0; inode < nen_; inode++)
          {
            if (enrtype == Discret::Elements::Fluid::xwall)
            {
              if (inode < nen_ / 2)
              {  // funct and xyze not in the same order
                x[0] += funct_(inode * 2) * xyze_(0, inode);
                x[1] += funct_(inode * 2) * xyze_(1, inode);
                x[2] += funct_(inode * 2) * xyze_(2, inode);
              }
            }
            else
            {
              x[0] += funct_(inode) * xyze_(0, inode);
              x[1] += funct_(inode) * xyze_(1, inode);
              x[2] += funct_(inode) * xyze_(2, inode);
            }
          }

          if (abs(x[normdirect] - (*planes)[*id]) > 2e-9)
          {
            FOUR_C_THROW("Mixing up element cut planes during integration");
          }
        }
#endif

        // interpolated values at gausspoints
        double ugp = 0;
        double vgp = 0;
        double wgp = 0;
        double pgp = 0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac = det * iquad.weight();

        // increase area of cutting plane in element
        area += fac;

        for (int inode = 0; inode < nen_; inode++)
        {
          int finode = inode * 4;

          ugp += funct_(inode) * sol(finode++);
          vgp += funct_(inode) * sol(finode++);
          wgp += funct_(inode) * sol(finode++);
          pgp += funct_(inode) * sol(finode);
        }

        // add contribution to integral

        double dubar = ugp * fac;
        double dvbar = vgp * fac;
        double dwbar = wgp * fac;
        double dpbar = pgp * fac;

        ubar += dubar;
        vbar += dvbar;
        wbar += dwbar;
        pbar += dpbar;

        usqbar += ugp * dubar;
        vsqbar += vgp * dvbar;
        wsqbar += wgp * dwbar;
        uvbar += ugp * dvbar;
        uwbar += ugp * dwbar;
        vwbar += vgp * dwbar;
        psqbar += pgp * dpbar;
      }  // end loop integration points

      // add increments from this layer to processor local vectors
      (*sumarea)[*id] += area;

      (*sumu)[*id] += ubar;
      (*sumv)[*id] += vbar;
      (*sumw)[*id] += wbar;
      (*sump)[*id] += pbar;

      (*sumsqu)[*id] += usqbar;
      (*sumsqv)[*id] += vsqbar;
      (*sumsqw)[*id] += wsqbar;
      (*sumuv)[*id] += uvbar;
      (*sumuw)[*id] += uwbar;
      (*sumvw)[*id] += vwbar;
      (*sumsqp)[*id] += psqbar;

      // jump to the next layer in the element.
      // in case of an hex8 element, the two coordinates are -1 and 1(+2)
      // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

      layershift += 2.0 / (static_cast<double>(numplanesinele - 1));
    }
  }
  else if (distype == Core::FE::CellType::nurbs8 || distype == Core::FE::CellType::nurbs27)
  {
    // get size of planecoords
    int size = planes->size();

    Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

    if (nurbsdis == nullptr)
    {
      FOUR_C_THROW("we need a nurbs discretisation for nurbs elements\n");
    }

    // get nurbs dis' element numbers
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(0));

    // use size of planes and mele to determine number of layers
    int numsublayers = (size - 1) / nele_x_mele_x_lele[1];

    // get the knotvector itself
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

    Core::Nodes::Node** nodes = ele->nodes();

    // get gid, location in the patch
    int gid = ele->id();

    std::vector<int> ele_cart_id(3);

    int npatch = -1;

    knots->convert_ele_gid_to_knot_ids(gid, npatch, ele_cart_id);
    if (npatch != 0)
    {
      FOUR_C_THROW("expected single patch nurbs problem for calculating means");
    }

    bool zero_size = false;
    zero_size = knots->get_ele_knots(myknots_, gid);

    // if we have a zero sized element due to a interpolated
    // point --- exit here
    if (zero_size)
    {
      return 0;
    }

    for (int inode = 0; inode < nen_; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

      weights_(inode) = cp->w();
    }

    // there's one additional plane for the last element layer
    int endlayer = 0;
    if (ele_cart_id[1] != nele_x_mele_x_lele[1] - 1)
    {
      endlayer = numsublayers;
    }
    else
    {
      endlayer = numsublayers + 1;
    }



    //!!see below for more information in green!!
    // this is dangerous because we don't check anywhere, if the wall normal points in y direction.
    // please make this more general!
    // we don't have a test case for this routine either
    FOUR_C_THROW(
        "Warning: Nurbs channel statistics work only if the element wall normal points in y "
        "direction.");


    // loop layers in element
    for (int rr = 0; rr < endlayer; ++rr)
    {
      // set gauss point coordinates
      double gp[3];
      gp[1] = -1.0 + rr * 2.0 / ((double)numsublayers);

      // get the quad9 gaussrule for the in plane integration
      Core::FE::GaussIntegration intpoints(Core::FE::CellType::quad9);

      // reset temporary values
      double area = 0;

      double ubar = 0;
      double vbar = 0;
      double wbar = 0;
      double pbar = 0;

      double usqbar = 0;
      double vsqbar = 0;
      double wsqbar = 0;
      double uvbar = 0;
      double uwbar = 0;
      double vwbar = 0;
      double psqbar = 0;


      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
          ++iquad)
      {
        // get the other gauss point coordinates
        // here we assume that the element wall normal points in y direction
        gp[0] = iquad.point()[0];
        gp[2] = iquad.point()[1];

        const double* gpconst = gp;
        eval_shape_func_and_derivs_at_int_point(gpconst, iquad.weight());

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)

        // here we still have the bug with the element normal directions
        // but we have to find out the element wall normal first to correct it
        // this part of the code works only if all the normals point in y direction!
        // please change the following lines to xjm_(elenormdirect,normdirect)
        // but you have to get elenormdirect first...
        if (xjm_(normdirect, normdirect) < 0)
        {
          xjm_(normdirect, normdirect) = -1.0;
        }
        else
        {
          xjm_(normdirect, normdirect) = 1.0;
        }

        const double det =
            xjm_(0, 0) * xjm_(1, 1) * xjm_(2, 2) + xjm_(0, 1) * xjm_(1, 2) * xjm_(2, 0) +
            xjm_(0, 2) * xjm_(1, 0) * xjm_(2, 1) - xjm_(0, 2) * xjm_(1, 1) * xjm_(2, 0) -
            xjm_(0, 0) * xjm_(1, 2) * xjm_(2, 1) - xjm_(0, 1) * xjm_(1, 0) * xjm_(2, 2);

        // check for degenerated elements
        if (det <= 0.0)
        {
          FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nNEGATIVE JACOBIAN DETERMINANT: {}", ele->id(), det);
        }

        // interpolated values at gausspoints
        double ugp = 0;
        double vgp = 0;
        double wgp = 0;
        double pgp = 0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac = det * iquad.weight();

        // increase area of cutting plane in element
        area += fac;

        for (int inode = 0; inode < nen_; inode++)
        {
          ugp += funct_(inode) * sol(inode * 4);
          vgp += funct_(inode) * sol(inode * 4 + 1);
          wgp += funct_(inode) * sol(inode * 4 + 2);
          pgp += funct_(inode) * sol(inode * 4 + 3);
        }

        // add contribution to integral
        ubar += ugp * fac;
        vbar += vgp * fac;
        wbar += wgp * fac;
        pbar += pgp * fac;

        usqbar += ugp * ugp * fac;
        vsqbar += vgp * vgp * fac;
        wsqbar += wgp * wgp * fac;
        uvbar += ugp * vgp * fac;
        uwbar += ugp * wgp * fac;
        vwbar += vgp * wgp * fac;
        psqbar += pgp * pgp * fac;
      }  // end loop integration points


      // add increments from this layer to processor local vectors
      (*sumarea)[ele_cart_id[1] * numsublayers + rr] += area;

      (*sumu)[ele_cart_id[1] * numsublayers + rr] += ubar;
      (*sumv)[ele_cart_id[1] * numsublayers + rr] += vbar;
      (*sumw)[ele_cart_id[1] * numsublayers + rr] += wbar;
      (*sump)[ele_cart_id[1] * numsublayers + rr] += pbar;

      (*sumsqu)[ele_cart_id[1] * numsublayers + rr] += usqbar;
      (*sumsqv)[ele_cart_id[1] * numsublayers + rr] += vsqbar;
      (*sumsqw)[ele_cart_id[1] * numsublayers + rr] += wsqbar;
      (*sumuv)[ele_cart_id[1] * numsublayers + rr] += uvbar;
      (*sumuw)[ele_cart_id[1] * numsublayers + rr] += uwbar;
      (*sumvw)[ele_cart_id[1] * numsublayers + rr] += vwbar;
      (*sumsqp)[ele_cart_id[1] * numsublayers + rr] += psqbar;
    }
  }
  else
  {
    FOUR_C_THROW("Unknown element type for mean value evaluation\n");
  }

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate properties for adaptive time step based on CFL number  bk 08/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_time_step(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // Do ALE specific updates if necessary
  // ---------------------------------------------------------------------
  // get additional state vectors for ALE case: grid displacement and vel.
  // ---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);
  if (ele->is_ale()) get_grid_disp_vel_ale(discretization, lm, edispnp, egridv);


  // evaluate shape functions and derivatives element center
  eval_shape_func_and_derivs_at_ele_center();

  // np_genalpha: additional vector for velocity at time n+1
  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);

  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelnp, nullptr, "velnp");

  convvelint_.multiply(evelnp, funct_);

  // calculate element length via the stream length definition, see corresponding implementation
  // in fluid_ele_calc for calculation of the stabilization parameter
  double h = 0.0;
  double vel_norm = 0.0;

  if (ele->is_ale())
  {
    gridvelint_.multiply(egridv, funct_);
    convvelint_.update(-1.0, gridvelint_, 1.0);
  }

  vel_norm = convvelint_.norm2();

  if (vel_norm > 1.0e-6)
  {
    Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
    velino.update(1.0 / vel_norm, convvelint_);

    // get streamlength using the normed velocity at element centre
    Core::LinAlg::Matrix<nen_, 1> tmp;
    // enriched dofs are not interpolatory with respect to geometry
    if (enrtype == Discret::Elements::Fluid::xwall)
    {
      Core::LinAlg::Matrix<nsd_, nen_> derxy_copy(derxy_);
      for (int inode = 1; inode < nen_; inode += 2)
      {
        for (int idim = 0; idim < nsd_; idim++) derxy_copy(idim, inode) = 0.0;
      }
      tmp.multiply_tn(derxy_copy, velino);
    }
    else
      tmp.multiply_tn(derxy_, velino);

    const double val = tmp.norm1();
    h = 2.0 / val;  // h=streamlength

    elevec1[0] = h / vel_norm;
  }
  else
    elevec1[0] = 1.0e12;

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate properties for adaptive forcing of periodic hill       bk 12/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_mass_flow_periodic_hill(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, std::shared_ptr<Core::Mat::Material>& mat)
{
  // set element id
  eid_ = ele->id();

  // ---------------------------------------------------------------------------
  // Prepare material parameters
  // ---------------------------------------------------------------------------
  // Since we need only the density, we use a lot of dummy values.
  // create dummy matrices
  Core::LinAlg::Matrix<nsd_, nen_> mat1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nen_, 1> mat2(Core::LinAlg::Initialization::zero);

  get_material_params(mat, mat1, mat2, mat2, mat2, mat2, mat2, 0.0, 0.0, 0.0, 0.0, 0.0);

  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // Do ALE specific updates if necessary
  if (ele->is_ale()) FOUR_C_THROW("no ale for periodic hill");

  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evelnp, nullptr, "velnp");

  // definition of matrices
  Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_> estif_u(Core::LinAlg::Initialization::zero);

  // length of whole domain
  double length = params.get<double>("length");

  // ---------------------------------------------------------------------------
  // Integration loop
  // ---------------------------------------------------------------------------
  double massf = 0.0;
  for (Core::FE::GaussIntegration::iterator iquad = intpoints_.begin(); iquad != intpoints_.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    // create dummy matrices
    Core::LinAlg::Matrix<nsd_, nen_> mat1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nen_, 1> mat2(Core::LinAlg::Initialization::zero);

    get_material_params(mat, mat1, mat2, mat2, mat2, mat2, mat2, 0.0, 0.0, 0.0, 0.0, 0.0);

    velint_.multiply(evelnp, funct_);

    massf += velint_(0) * densaf_ * fac_;
  }
  massf /= length;
  elevec1[0] = massf;

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate coordinates and velocities and element center          bk 01/2015 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalc<distype, enrtype>::calc_vel_gradient_ele_center(
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2)
{
  if (distype != Core::FE::CellType::hex8 && distype != Core::FE::CellType::tet4 &&
      distype != Core::FE::CellType::quad4 && distype != Core::FE::CellType::tri3)
    FOUR_C_THROW("this is currently only implemented for linear elements");
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  // Do ALE specific updates if necessary
  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edisp(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &edisp, nullptr, "disp");

    // get new node positions of ALE mesh
    xyze_ += edisp;
  }

  // evaluate shape functions and derivatives element center
  eval_shape_func_and_derivs_at_ele_center();

  if (discretization.has_state("vel"))
  {
    // extract element velocities
    Core::LinAlg::Matrix<nsd_, nen_> evel(Core::LinAlg::Initialization::zero);
    extract_values_from_global_vector(discretization, lm, *rotsymmpbc_, &evel, nullptr, "vel");

    // get gradient of velocity at element center
    vderxy_.multiply_nt(evel, derxy_);

    // write values into element vector (same order as in vel_gradient_projection())
    for (int i = 0; i < nsd_; ++i)  // loop rows of vderxy
    {
      for (int j = 0; j < nsd_; ++j)  // loop columns of vderxy
      {
        elevec1[i * nsd_ + j] = vderxy_(i, j);
      }
    }
  }

  // get position of element centroid
  Core::LinAlg::Matrix<nsd_, 1> x_centroid(Core::LinAlg::Initialization::zero);
  x_centroid.multiply(xyze_, funct_);
  for (int i = 0; i < nsd_; ++i)
  {
    elevec2[i] = x_centroid(i);
  }

  return 0;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::lin_mesh_motion_2d(
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const double& press, const double& timefac,
    const double& timefacfac)
{
  // xGderiv_ = sum(gridx(k,i) * deriv_(j,k), k);
  // xGderiv_ == xjm_

  const double fac0 =
      densam_ * velint_(0) - rhsmom_(0) * fldparatimint_->dt() * fldparatimint_->theta();
  const double fac1 =
      densam_ * velint_(1) - rhsmom_(1) * fldparatimint_->dt() * fldparatimint_->theta();

  // mass + rhs
  for (int vi = 0; vi < nen_; ++vi)
  {
    const int tvi = 3 * vi;
    const int tvip = tvi + 1;

    const double v = fac_ * funct_(vi);
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int tui = 3 * ui;
      const int tuip = tui + 1;

      emesh(tvi, tui) += v * fac0 * derxy_(0, ui);
      emesh(tvi, tuip) += v * fac0 * derxy_(1, ui);

      emesh(tvip, tui) += v * fac1 * derxy_(0, ui);
      emesh(tvip, tuip) += v * fac1 * derxy_(1, ui);
    }
  }

  vderiv_.multiply_nt(evelaf, deriv_);

  const double vderiv_0_0 = vderiv_(0, 0);
  const double vderiv_0_1 = vderiv_(0, 1);
  const double vderiv_1_0 = vderiv_(1, 0);
  const double vderiv_1_1 = vderiv_(1, 1);

  {
    const double convvelint_0 = convvelint_(0);
    const double convvelint_1 = convvelint_(1);
    const double densaftimefacfac_det = densaf_ * timefacfac / det_;

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int tvi = 3 * vi;
      const int tvip = tvi + 1;
      const double v = densaftimefacfac_det * funct_(vi);
      for (int ui = 0; ui < nen_; ++ui)
      {
        const int tui = 3 * ui;
        const int tuip = tui + 1;

        emesh(tvi, tui) +=
            v * (+convvelint_1 * (-vderiv_0_0 * deriv_(1, ui) + vderiv_0_1 * deriv_(0, ui)));

        emesh(tvi, tuip) -=
            v * (+convvelint_0 * (-vderiv_0_0 * deriv_(1, ui) + vderiv_0_1 * deriv_(0, ui)));

        emesh(tvip, tui) +=
            v * (+convvelint_1 * (-vderiv_1_0 * deriv_(1, ui) + vderiv_1_1 * deriv_(0, ui)));

        emesh(tvip, tuip) -=
            v * (+convvelint_0 * (-vderiv_1_0 * deriv_(1, ui) + vderiv_1_1 * deriv_(0, ui)));
      }
    }
  }

  // pressure
  const double v = press * timefacfac / det_;
  for (int vi = 0; vi < nen_; ++vi)
  {
    const int tvi = 3 * vi;
    const int tvip = tvi + 1;
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int tui = 3 * ui;
      emesh(tvi, tui + 1) -= v * (deriv_(0, vi) * deriv_(1, ui) - deriv_(0, ui) * deriv_(1, vi));
      emesh(tvip, tui) += v * (deriv_(0, vi) * deriv_(1, ui) - deriv_(0, ui) * deriv_(1, vi));
    }
  }

  // div u
  const double timefacfac_det = timefacfac / det_;
  for (int vi = 0; vi < nen_; ++vi)
  {
    const int tvipp = 3 * vi + 2;
    const double v = timefacfac_det * funct_(vi);
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int tui = 3 * ui;
      emesh(tvipp, tui) += v * (deriv_(0, ui) * vderiv_1_1 - deriv_(1, ui) * vderiv_1_0);

      emesh(tvipp, tui + 1) -= v * (deriv_(0, ui) * vderiv_0_1 - deriv_(1, ui) * vderiv_0_0);
    }
  }


  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::lin_mesh_motion_3d(
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const double& press, const double& timefac,
    const double& timefacfac)
{
  // xGderiv_ = sum(gridx(k,i) * deriv_(j,k), k);
  // xGderiv_ == xjm_

  // mass + rhs
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = fac_ * funct_(vi, 0);
    const double fac0 =
        v * (densam_ * velint_(0) - rhsmom_(0) * fldparatimint_->dt() * fldparatimint_->theta());
    for (int ui = 0; ui < nen_; ++ui)
    {
      emesh(vi * 4, ui * 4) += fac0 * derxy_(0, ui);
      emesh(vi * 4, ui * 4 + 1) += fac0 * derxy_(1, ui);
      emesh(vi * 4, ui * 4 + 2) += fac0 * derxy_(2, ui);
    }

    const double fac1 =
        v * (densam_ * velint_(1) - rhsmom_(1) * fldparatimint_->dt() * fldparatimint_->theta());
    for (int ui = 0; ui < nen_; ++ui)
    {
      emesh(vi * 4 + 1, ui * 4) += fac1 * derxy_(0, ui);
      emesh(vi * 4 + 1, ui * 4 + 1) += fac1 * derxy_(1, ui);
      emesh(vi * 4 + 1, ui * 4 + 2) += fac1 * derxy_(2, ui);
    }

    const double fac2 =
        v * (densam_ * velint_(2) - rhsmom_(2) * fldparatimint_->dt() * fldparatimint_->theta());
    for (int ui = 0; ui < nen_; ++ui)
    {
      emesh(vi * 4 + 2, ui * 4) += fac2 * derxy_(0, ui);
      emesh(vi * 4 + 2, ui * 4 + 1) += fac2 * derxy_(1, ui);
      emesh(vi * 4 + 2, ui * 4 + 2) += fac2 * derxy_(2, ui);
    }
  }

  // vderiv_  = sum(evelaf(i,k) * deriv_(j,k), k);
  vderiv_.multiply_nt(evelaf, deriv_);

#define derxjm_(r, c, d, i) derxjm_##r##c##d(i)

#define derxjm_001(ui) (deriv_(2, ui) * xjm_1_2 - deriv_(1, ui) * xjm_2_2)
#define derxjm_002(ui) (deriv_(1, ui) * xjm_2_1 - deriv_(2, ui) * xjm_1_1)

#define derxjm_100(ui) (deriv_(1, ui) * xjm_2_2 - deriv_(2, ui) * xjm_1_2)
#define derxjm_102(ui) (deriv_(2, ui) * xjm_1_0 - deriv_(1, ui) * xjm_2_0)

#define derxjm_200(ui) (deriv_(2, ui) * xjm_1_1 - deriv_(1, ui) * xjm_2_1)
#define derxjm_201(ui) (deriv_(1, ui) * xjm_2_0 - deriv_(2, ui) * xjm_1_0)

#define derxjm_011(ui) (deriv_(0, ui) * xjm_2_2 - deriv_(2, ui) * xjm_0_2)
#define derxjm_012(ui) (deriv_(2, ui) * xjm_0_1 - deriv_(0, ui) * xjm_2_1)

#define derxjm_110(ui) (deriv_(2, ui) * xjm_0_2 - deriv_(0, ui) * xjm_2_2)
#define derxjm_112(ui) (deriv_(0, ui) * xjm_2_0 - deriv_(2, ui) * xjm_0_0)

#define derxjm_210(ui) (deriv_(0, ui) * xjm_2_1 - deriv_(2, ui) * xjm_0_1)
#define derxjm_211(ui) (deriv_(2, ui) * xjm_0_0 - deriv_(0, ui) * xjm_2_0)

#define derxjm_021(ui) (deriv_(1, ui) * xjm_0_2 - deriv_(0, ui) * xjm_1_2)
#define derxjm_022(ui) (deriv_(0, ui) * xjm_1_1 - deriv_(1, ui) * xjm_0_1)

#define derxjm_120(ui) (deriv_(0, ui) * xjm_1_2 - deriv_(1, ui) * xjm_0_2)
#define derxjm_122(ui) (deriv_(1, ui) * xjm_0_0 - deriv_(0, ui) * xjm_1_0)

#define derxjm_220(ui) (deriv_(1, ui) * xjm_0_1 - deriv_(0, ui) * xjm_1_1)
#define derxjm_221(ui) (deriv_(0, ui) * xjm_1_0 - deriv_(1, ui) * xjm_0_0)

  const double vderiv_0_0 = vderiv_(0, 0);
  const double vderiv_0_1 = vderiv_(0, 1);
  const double vderiv_0_2 = vderiv_(0, 2);
  const double vderiv_1_0 = vderiv_(1, 0);
  const double vderiv_1_1 = vderiv_(1, 1);
  const double vderiv_1_2 = vderiv_(1, 2);
  const double vderiv_2_0 = vderiv_(2, 0);
  const double vderiv_2_1 = vderiv_(2, 1);
  const double vderiv_2_2 = vderiv_(2, 2);

  const double xjm_0_0 = xjm_(0, 0);
  const double xjm_0_1 = xjm_(0, 1);
  const double xjm_0_2 = xjm_(0, 2);
  const double xjm_1_0 = xjm_(1, 0);
  const double xjm_1_1 = xjm_(1, 1);
  const double xjm_1_2 = xjm_(1, 2);
  const double xjm_2_0 = xjm_(2, 0);
  const double xjm_2_1 = xjm_(2, 1);
  const double xjm_2_2 = xjm_(2, 2);

  {
    const double convvelint_0 = convvelint_(0);
    const double convvelint_1 = convvelint_(1);
    const double convvelint_2 = convvelint_(2);
    const double denstimefacfac_det = densaf_ * timefacfac / det_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v00 =
          +convvelint_1 * (vderiv_0_0 * derxjm_(0, 0, 1, ui) + vderiv_0_1 * derxjm_(0, 1, 1, ui) +
                              vderiv_0_2 * derxjm_(0, 2, 1, ui)) +
          convvelint_2 * (vderiv_0_0 * derxjm_(0, 0, 2, ui) + vderiv_0_1 * derxjm_(0, 1, 2, ui) +
                             vderiv_0_2 * derxjm_(0, 2, 2, ui));
      const double v01 =
          +convvelint_0 * (vderiv_0_0 * derxjm_(1, 0, 0, ui) + vderiv_0_1 * derxjm_(1, 1, 0, ui) +
                              vderiv_0_2 * derxjm_(1, 2, 0, ui)) +
          convvelint_2 * (vderiv_0_0 * derxjm_(1, 0, 2, ui) + vderiv_0_1 * derxjm_(1, 1, 2, ui) +
                             vderiv_0_2 * derxjm_(1, 2, 2, ui));
      const double v02 =
          +convvelint_0 * (vderiv_0_0 * derxjm_(2, 0, 0, ui) + vderiv_0_1 * derxjm_(2, 1, 0, ui) +
                              vderiv_0_2 * derxjm_(2, 2, 0, ui)) +
          convvelint_1 * (vderiv_0_0 * derxjm_(2, 0, 1, ui) + vderiv_0_1 * derxjm_(2, 1, 1, ui) +
                             vderiv_0_2 * derxjm_(2, 2, 1, ui));
      const double v10 =
          +convvelint_1 * (vderiv_1_0 * derxjm_(0, 0, 1, ui) + vderiv_1_1 * derxjm_(0, 1, 1, ui) +
                              vderiv_1_2 * derxjm_(0, 2, 1, ui)) +
          convvelint_2 * (vderiv_1_0 * derxjm_(0, 0, 2, ui) + vderiv_1_1 * derxjm_(0, 1, 2, ui) +
                             vderiv_1_2 * derxjm_(0, 2, 2, ui));
      const double v11 =
          +convvelint_0 * (vderiv_1_0 * derxjm_(1, 0, 0, ui) + vderiv_1_1 * derxjm_(1, 1, 0, ui) +
                              vderiv_1_2 * derxjm_(1, 2, 0, ui)) +
          convvelint_2 * (vderiv_1_0 * derxjm_(1, 0, 2, ui) + vderiv_1_1 * derxjm_(1, 1, 2, ui) +
                             vderiv_1_2 * derxjm_(1, 2, 2, ui));
      const double v12 =
          +convvelint_0 * (vderiv_1_0 * derxjm_(2, 0, 0, ui) + vderiv_1_1 * derxjm_(2, 1, 0, ui) +
                              vderiv_1_2 * derxjm_(2, 2, 0, ui)) +
          convvelint_1 * (vderiv_1_0 * derxjm_(2, 0, 1, ui) + vderiv_1_1 * derxjm_(2, 1, 1, ui) +
                             vderiv_1_2 * derxjm_(2, 2, 1, ui));
      const double v20 =
          +convvelint_1 * (vderiv_2_0 * derxjm_(0, 0, 1, ui) + vderiv_2_1 * derxjm_(0, 1, 1, ui) +
                              vderiv_2_2 * derxjm_(0, 2, 1, ui)) +
          convvelint_2 * (vderiv_2_0 * derxjm_(0, 0, 2, ui) + vderiv_2_1 * derxjm_(0, 1, 2, ui) +
                             vderiv_2_2 * derxjm_(0, 2, 2, ui));
      const double v21 =
          +convvelint_0 * (vderiv_2_0 * derxjm_(1, 0, 0, ui) + vderiv_2_1 * derxjm_(1, 1, 0, ui) +
                              vderiv_2_2 * derxjm_(1, 2, 0, ui)) +
          convvelint_2 * (vderiv_2_0 * derxjm_(1, 0, 2, ui) + vderiv_2_1 * derxjm_(1, 1, 2, ui) +
                             vderiv_2_2 * derxjm_(1, 2, 2, ui));
      const double v22 =
          +convvelint_0 * (vderiv_2_0 * derxjm_(2, 0, 0, ui) + vderiv_2_1 * derxjm_(2, 1, 0, ui) +
                              vderiv_2_2 * derxjm_(2, 2, 0, ui)) +
          convvelint_1 * (vderiv_2_0 * derxjm_(2, 0, 1, ui) + vderiv_2_1 * derxjm_(2, 1, 1, ui) +
                             vderiv_2_2 * derxjm_(2, 2, 1, ui));

      for (int vi = 0; vi < nen_; ++vi)
      {
        const double v = denstimefacfac_det * funct_(vi);

        emesh(vi * 4 + 0, ui * 4 + 0) += v * v00;
        emesh(vi * 4 + 0, ui * 4 + 1) += v * v01;
        emesh(vi * 4 + 0, ui * 4 + 2) += v * v02;

        emesh(vi * 4 + 1, ui * 4 + 0) += v * v10;
        emesh(vi * 4 + 1, ui * 4 + 1) += v * v11;
        emesh(vi * 4 + 1, ui * 4 + 2) += v * v12;

        emesh(vi * 4 + 2, ui * 4 + 0) += v * v20;
        emesh(vi * 4 + 2, ui * 4 + 1) += v * v21;
        emesh(vi * 4 + 2, ui * 4 + 2) += v * v22;
      }
    }
  }

  // viscosity

  const double xji_00 = xji_(0, 0);
  const double xji_01 = xji_(0, 1);
  const double xji_02 = xji_(0, 2);
  const double xji_10 = xji_(1, 0);
  const double xji_11 = xji_(1, 1);
  const double xji_12 = xji_(1, 2);
  const double xji_20 = xji_(2, 0);
  const double xji_21 = xji_(2, 1);
  const double xji_22 = xji_(2, 2);

  // part 1: derivative of 1/det
  {
    const double vderxy_0_0 = 2.0 * vderxy_(0, 0);
    const double vderxy_1_1 = 2.0 * vderxy_(1, 1);
    const double vderxy_2_2 = 2.0 * vderxy_(2, 2);
    const double vderxy_0_1 = vderxy_(0, 1) + vderxy_(1, 0);
    const double vderxy_0_2 = vderxy_(0, 2) + vderxy_(2, 0);
    const double vderxy_1_2 = vderxy_(1, 2) + vderxy_(2, 1);

    const double v = visceff_ * timefac * fac_;
    for (int ui = 0; ui < nen_; ++ui)
    {
      const double derinvJ0 =
          -v * (deriv_(0, ui) * xji_00 + deriv_(1, ui) * xji_01 + deriv_(2, ui) * xji_02);
      const double derinvJ1 =
          -v * (deriv_(0, ui) * xji_10 + deriv_(1, ui) * xji_11 + deriv_(2, ui) * xji_12);
      const double derinvJ2 =
          -v * (deriv_(0, ui) * xji_20 + deriv_(1, ui) * xji_21 + deriv_(2, ui) * xji_22);
      for (int vi = 0; vi < nen_; ++vi)
      {
        const double visres0 =
            derxy_(0, vi) * vderxy_0_0 + derxy_(1, vi) * vderxy_0_1 + derxy_(2, vi) * vderxy_0_2;
        const double visres1 =
            derxy_(0, vi) * vderxy_0_1 + derxy_(1, vi) * vderxy_1_1 + derxy_(2, vi) * vderxy_1_2;
        const double visres2 =
            derxy_(0, vi) * vderxy_0_2 + derxy_(1, vi) * vderxy_1_2 + derxy_(2, vi) * vderxy_2_2;
        emesh(vi * 4 + 0, ui * 4 + 0) += derinvJ0 * visres0;
        emesh(vi * 4 + 1, ui * 4 + 0) += derinvJ0 * visres1;
        emesh(vi * 4 + 2, ui * 4 + 0) += derinvJ0 * visres2;

        emesh(vi * 4 + 0, ui * 4 + 1) += derinvJ1 * visres0;
        emesh(vi * 4 + 1, ui * 4 + 1) += derinvJ1 * visres1;
        emesh(vi * 4 + 2, ui * 4 + 1) += derinvJ1 * visres2;

        emesh(vi * 4 + 0, ui * 4 + 2) += derinvJ2 * visres0;
        emesh(vi * 4 + 1, ui * 4 + 2) += derinvJ2 * visres1;
        emesh(vi * 4 + 2, ui * 4 + 2) += derinvJ2 * visres2;
      }
    }
  }

  // part 2: derivative of viscosity residual

  {
    const double v = timefacfac * visceff_ / det_;
    for (int ui = 0; ui < nen_; ++ui)
    {
      const double ui_derxjm_001 = (deriv_(2, ui) * xjm_1_2 - deriv_(1, ui) * xjm_2_2);
      const double ui_derxjm_002 = (deriv_(1, ui) * xjm_2_1 - deriv_(2, ui) * xjm_1_1);
      const double ui_derxjm_100 = (deriv_(1, ui) * xjm_2_2 - deriv_(2, ui) * xjm_1_2);
      const double ui_derxjm_102 = (deriv_(2, ui) * xjm_1_0 - deriv_(1, ui) * xjm_2_0);
      const double ui_derxjm_200 = (deriv_(2, ui) * xjm_1_1 - deriv_(1, ui) * xjm_2_1);
      const double ui_derxjm_201 = (deriv_(1, ui) * xjm_2_0 - deriv_(2, ui) * xjm_1_0);
      const double ui_derxjm_011 = (deriv_(0, ui) * xjm_2_2 - deriv_(2, ui) * xjm_0_2);
      const double ui_derxjm_012 = (deriv_(2, ui) * xjm_0_1 - deriv_(0, ui) * xjm_2_1);
      const double ui_derxjm_110 = (deriv_(2, ui) * xjm_0_2 - deriv_(0, ui) * xjm_2_2);
      const double ui_derxjm_112 = (deriv_(0, ui) * xjm_2_0 - deriv_(2, ui) * xjm_0_0);
      const double ui_derxjm_210 = (deriv_(0, ui) * xjm_2_1 - deriv_(2, ui) * xjm_0_1);
      const double ui_derxjm_211 = (deriv_(2, ui) * xjm_0_0 - deriv_(0, ui) * xjm_2_0);
      const double ui_derxjm_021 = (deriv_(1, ui) * xjm_0_2 - deriv_(0, ui) * xjm_1_2);
      const double ui_derxjm_022 = (deriv_(0, ui) * xjm_1_1 - deriv_(1, ui) * xjm_0_1);
      const double ui_derxjm_120 = (deriv_(0, ui) * xjm_1_2 - deriv_(1, ui) * xjm_0_2);
      const double ui_derxjm_122 = (deriv_(1, ui) * xjm_0_0 - deriv_(0, ui) * xjm_1_0);
      const double ui_derxjm_220 = (deriv_(1, ui) * xjm_0_1 - deriv_(0, ui) * xjm_1_1);
      const double ui_derxjm_221 = (deriv_(0, ui) * xjm_1_0 - deriv_(1, ui) * xjm_0_0);

      {
        const double v0 =
            -vderiv_0_0 * (xji_10 * ui_derxjm_100 + xji_10 * ui_derxjm_100 +
                              xji_20 * ui_derxjm_200 + xji_20 * ui_derxjm_200) -
            vderiv_0_1 * (xji_11 * ui_derxjm_100 + xji_10 * ui_derxjm_110 + xji_21 * ui_derxjm_200 +
                             xji_20 * ui_derxjm_210) -
            vderiv_0_2 * (xji_12 * ui_derxjm_100 + xji_10 * ui_derxjm_120 + xji_22 * ui_derxjm_200 +
                             xji_20 * ui_derxjm_220) -
            vderiv_1_0 * (ui_derxjm_100 * xji_00) - vderiv_1_1 * (ui_derxjm_100 * xji_01) -
            vderiv_1_2 * (ui_derxjm_100 * xji_02) - vderiv_2_0 * (ui_derxjm_200 * xji_00) -
            vderiv_2_1 * (ui_derxjm_200 * xji_01) - vderiv_2_2 * (ui_derxjm_200 * xji_02);
        const double v1 =
            -vderiv_0_0 * (xji_10 * ui_derxjm_110 + xji_11 * ui_derxjm_100 +
                              xji_20 * ui_derxjm_210 + xji_21 * ui_derxjm_200) -
            vderiv_0_1 * (xji_11 * ui_derxjm_110 + xji_11 * ui_derxjm_110 + xji_21 * ui_derxjm_210 +
                             xji_21 * ui_derxjm_210) -
            vderiv_0_2 * (xji_12 * ui_derxjm_110 + xji_11 * ui_derxjm_120 + xji_22 * ui_derxjm_210 +
                             xji_21 * ui_derxjm_220) -
            vderiv_1_0 * (ui_derxjm_110 * xji_00) - vderiv_1_1 * (ui_derxjm_110 * xji_01) -
            vderiv_1_2 * (ui_derxjm_110 * xji_02) - vderiv_2_0 * (ui_derxjm_210 * xji_00) -
            vderiv_2_1 * (ui_derxjm_210 * xji_01) - vderiv_2_2 * (ui_derxjm_210 * xji_02);
        const double v2 =
            -vderiv_0_0 * (xji_10 * ui_derxjm_120 + xji_12 * ui_derxjm_100 +
                              xji_20 * ui_derxjm_220 + xji_22 * ui_derxjm_200) -
            vderiv_0_1 * (xji_11 * ui_derxjm_120 + xji_12 * ui_derxjm_110 + xji_21 * ui_derxjm_220 +
                             xji_22 * ui_derxjm_210) -
            vderiv_0_2 * (xji_12 * ui_derxjm_120 + xji_12 * ui_derxjm_120 + xji_22 * ui_derxjm_220 +
                             xji_22 * ui_derxjm_220) -
            vderiv_1_0 * (ui_derxjm_120 * xji_00) - vderiv_1_1 * (ui_derxjm_120 * xji_01) -
            vderiv_1_2 * (ui_derxjm_120 * xji_02) - vderiv_2_0 * (ui_derxjm_220 * xji_00) -
            vderiv_2_1 * (ui_derxjm_220 * xji_01) - vderiv_2_2 * (ui_derxjm_220 * xji_02);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 0, ui * 4 + 0) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 = -vderiv_0_0 * (2 * ui_derxjm_001 * xji_00 + 2 * ui_derxjm_001 * xji_00 +
                                            xji_20 * ui_derxjm_201 + xji_20 * ui_derxjm_201) -
                          vderiv_0_1 * (2 * ui_derxjm_011 * xji_00 + 2 * ui_derxjm_001 * xji_01 +
                                           xji_21 * ui_derxjm_201 + xji_20 * ui_derxjm_211) -
                          vderiv_0_2 * (2 * ui_derxjm_021 * xji_00 + 2 * ui_derxjm_001 * xji_02 +
                                           xji_22 * ui_derxjm_201 + xji_20 * ui_derxjm_221) -
                          vderiv_1_0 * (ui_derxjm_001 * xji_10) -
                          vderiv_1_1 * (ui_derxjm_011 * xji_10) -
                          vderiv_1_2 * (ui_derxjm_021 * xji_10) -
                          vderiv_2_0 * (ui_derxjm_201 * xji_00 + ui_derxjm_001 * xji_20) -
                          vderiv_2_1 * (ui_derxjm_201 * xji_01 + ui_derxjm_011 * xji_20) -
                          vderiv_2_2 * (ui_derxjm_201 * xji_02 + ui_derxjm_021 * xji_20);
        const double v1 = -vderiv_0_0 * (2 * ui_derxjm_011 * xji_00 + 2 * ui_derxjm_001 * xji_01 +
                                            xji_21 * ui_derxjm_201 + xji_20 * ui_derxjm_211) -
                          vderiv_0_1 * (2 * ui_derxjm_011 * xji_01 + 2 * ui_derxjm_011 * xji_01 +
                                           xji_21 * ui_derxjm_211 + xji_21 * ui_derxjm_211) -
                          vderiv_0_2 * (2 * ui_derxjm_011 * xji_02 + 2 * ui_derxjm_021 * xji_01 +
                                           xji_21 * ui_derxjm_221 + xji_22 * ui_derxjm_211) -
                          vderiv_1_0 * (ui_derxjm_001 * xji_11) -
                          vderiv_1_1 * (ui_derxjm_011 * xji_11) -
                          vderiv_1_2 * (ui_derxjm_021 * xji_11) -
                          vderiv_2_0 * (ui_derxjm_211 * xji_00 + ui_derxjm_001 * xji_21) -
                          vderiv_2_1 * (ui_derxjm_211 * xji_01 + ui_derxjm_011 * xji_21) -
                          vderiv_2_2 * (ui_derxjm_211 * xji_02 + ui_derxjm_021 * xji_21);
        const double v2 = -vderiv_0_0 * (2 * ui_derxjm_021 * xji_00 + 2 * ui_derxjm_001 * xji_02 +
                                            xji_22 * ui_derxjm_201 + xji_20 * ui_derxjm_221) -
                          vderiv_0_1 * (2 * ui_derxjm_011 * xji_02 + 2 * ui_derxjm_021 * xji_01 +
                                           xji_21 * ui_derxjm_221 + xji_22 * ui_derxjm_211) -
                          vderiv_0_2 * (2 * ui_derxjm_021 * xji_02 + 2 * ui_derxjm_021 * xji_02 +
                                           xji_22 * ui_derxjm_221 + xji_22 * ui_derxjm_221) -
                          vderiv_1_0 * (ui_derxjm_001 * xji_12) -
                          vderiv_1_1 * (ui_derxjm_011 * xji_12) -
                          vderiv_1_2 * (ui_derxjm_021 * xji_12) -
                          vderiv_2_0 * (ui_derxjm_221 * xji_00 + ui_derxjm_001 * xji_22) -
                          vderiv_2_1 * (ui_derxjm_221 * xji_01 + ui_derxjm_011 * xji_22) -
                          vderiv_2_2 * (ui_derxjm_221 * xji_02 + ui_derxjm_021 * xji_22);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 0, ui * 4 + 1) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 = -vderiv_0_0 * (2 * ui_derxjm_002 * xji_00 + 2 * ui_derxjm_002 * xji_00 +
                                            xji_10 * ui_derxjm_102 + xji_10 * ui_derxjm_102) -
                          vderiv_0_1 * (2 * ui_derxjm_012 * xji_00 + 2 * ui_derxjm_002 * xji_01 +
                                           xji_11 * ui_derxjm_102 + xji_10 * ui_derxjm_112) -
                          vderiv_0_2 * (2 * ui_derxjm_022 * xji_00 + 2 * ui_derxjm_002 * xji_02 +
                                           xji_12 * ui_derxjm_102 + xji_10 * ui_derxjm_122) -
                          vderiv_1_0 * (ui_derxjm_002 * xji_10 + ui_derxjm_102 * xji_00) -
                          vderiv_1_1 * (ui_derxjm_012 * xji_10 + ui_derxjm_102 * xji_01) -
                          vderiv_1_2 * (ui_derxjm_022 * xji_10 + ui_derxjm_102 * xji_02) -
                          vderiv_2_0 * (ui_derxjm_002 * xji_20) -
                          vderiv_2_1 * (ui_derxjm_012 * xji_20) -
                          vderiv_2_2 * (ui_derxjm_022 * xji_20);
        const double v1 = -vderiv_0_0 * (2 * ui_derxjm_012 * xji_00 + 2 * ui_derxjm_002 * xji_01 +
                                            xji_11 * ui_derxjm_102 + xji_10 * ui_derxjm_112) -
                          vderiv_0_1 * (2 * ui_derxjm_012 * xji_01 + 2 * ui_derxjm_012 * xji_01 +
                                           xji_11 * ui_derxjm_112 + xji_11 * ui_derxjm_112) -
                          vderiv_0_2 * (2 * ui_derxjm_012 * xji_02 + 2 * ui_derxjm_022 * xji_01 +
                                           xji_11 * ui_derxjm_122 + xji_12 * ui_derxjm_112) -
                          vderiv_1_0 * (ui_derxjm_002 * xji_11 + ui_derxjm_112 * xji_00) -
                          vderiv_1_1 * (ui_derxjm_012 * xji_11 + ui_derxjm_112 * xji_01) -
                          vderiv_1_2 * (ui_derxjm_022 * xji_11 + ui_derxjm_112 * xji_02) -
                          vderiv_2_0 * (ui_derxjm_002 * xji_21) -
                          vderiv_2_1 * (ui_derxjm_012 * xji_21) -
                          vderiv_2_2 * (ui_derxjm_022 * xji_21);
        const double v2 = -vderiv_0_0 * (2 * ui_derxjm_022 * xji_00 + 2 * ui_derxjm_002 * xji_02 +
                                            xji_12 * ui_derxjm_102 + xji_10 * ui_derxjm_122) -
                          vderiv_0_1 * (2 * ui_derxjm_012 * xji_02 + 2 * ui_derxjm_022 * xji_01 +
                                           xji_11 * ui_derxjm_122 + xji_12 * ui_derxjm_112) -
                          vderiv_0_2 * (2 * ui_derxjm_022 * xji_02 + 2 * ui_derxjm_022 * xji_02 +
                                           xji_12 * ui_derxjm_122 + xji_12 * ui_derxjm_122) -
                          vderiv_1_0 * (ui_derxjm_002 * xji_12 + ui_derxjm_122 * xji_00) -
                          vderiv_1_1 * (ui_derxjm_012 * xji_12 + ui_derxjm_122 * xji_01) -
                          vderiv_1_2 * (ui_derxjm_022 * xji_12 + ui_derxjm_122 * xji_02) -
                          vderiv_2_0 * (ui_derxjm_002 * xji_22) -
                          vderiv_2_1 * (ui_derxjm_012 * xji_22) -
                          vderiv_2_2 * (ui_derxjm_022 * xji_22);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 0, ui * 4 + 2) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 = -vderiv_0_0 * (ui_derxjm_100 * xji_00) -
                          vderiv_0_1 * (ui_derxjm_110 * xji_00) -
                          vderiv_0_2 * (ui_derxjm_120 * xji_00) -
                          vderiv_1_0 * (2 * xji_10 * ui_derxjm_100 + 2 * xji_10 * ui_derxjm_100 +
                                           xji_20 * ui_derxjm_200 + xji_20 * ui_derxjm_200) -
                          vderiv_1_1 * (2 * xji_11 * ui_derxjm_100 + 2 * xji_10 * ui_derxjm_110 +
                                           xji_21 * ui_derxjm_200 + xji_20 * ui_derxjm_210) -
                          vderiv_1_2 * (2 * xji_12 * ui_derxjm_100 + 2 * xji_10 * ui_derxjm_120 +
                                           xji_22 * ui_derxjm_200 + xji_20 * ui_derxjm_220) -
                          vderiv_2_0 * (ui_derxjm_200 * xji_10 + ui_derxjm_100 * xji_20) -
                          vderiv_2_1 * (ui_derxjm_200 * xji_11 + ui_derxjm_110 * xji_20) -
                          vderiv_2_2 * (ui_derxjm_200 * xji_12 + ui_derxjm_120 * xji_20);
        const double v1 = -vderiv_0_0 * (ui_derxjm_100 * xji_01) -
                          vderiv_0_1 * (ui_derxjm_110 * xji_01) -
                          vderiv_0_2 * (ui_derxjm_120 * xji_01) -
                          vderiv_1_0 * (2 * xji_10 * ui_derxjm_110 + 2 * xji_11 * ui_derxjm_100 +
                                           xji_20 * ui_derxjm_210 + xji_21 * ui_derxjm_200) -
                          vderiv_1_1 * (2 * xji_11 * ui_derxjm_110 + 2 * xji_11 * ui_derxjm_110 +
                                           xji_21 * ui_derxjm_210 + xji_21 * ui_derxjm_210) -
                          vderiv_1_2 * (2 * xji_12 * ui_derxjm_110 + 2 * xji_11 * ui_derxjm_120 +
                                           xji_22 * ui_derxjm_210 + xji_21 * ui_derxjm_220) -
                          vderiv_2_0 * (ui_derxjm_210 * xji_10 + ui_derxjm_100 * xji_21) -
                          vderiv_2_1 * (ui_derxjm_210 * xji_11 + ui_derxjm_110 * xji_21) -
                          vderiv_2_2 * (ui_derxjm_210 * xji_12 + ui_derxjm_120 * xji_21);
        const double v2 = -vderiv_0_0 * (ui_derxjm_100 * xji_02) -
                          vderiv_0_1 * (ui_derxjm_110 * xji_02) -
                          vderiv_0_2 * (ui_derxjm_120 * xji_02) -
                          vderiv_1_0 * (2 * xji_10 * ui_derxjm_120 + 2 * xji_12 * ui_derxjm_100 +
                                           xji_20 * ui_derxjm_220 + xji_22 * ui_derxjm_200) -
                          vderiv_1_1 * (2 * xji_11 * ui_derxjm_120 + 2 * xji_12 * ui_derxjm_110 +
                                           xji_21 * ui_derxjm_220 + xji_22 * ui_derxjm_210) -
                          vderiv_1_2 * (2 * xji_12 * ui_derxjm_120 + 2 * xji_12 * ui_derxjm_120 +
                                           xji_22 * ui_derxjm_220 + xji_22 * ui_derxjm_220) -
                          vderiv_2_0 * (ui_derxjm_220 * xji_10 + ui_derxjm_100 * xji_22) -
                          vderiv_2_1 * (ui_derxjm_220 * xji_11 + ui_derxjm_110 * xji_22) -
                          vderiv_2_2 * (ui_derxjm_220 * xji_12 + ui_derxjm_120 * xji_22);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 1, ui * 4 + 0) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 =
            -vderiv_0_0 * (ui_derxjm_001 * xji_10) - vderiv_0_1 * (ui_derxjm_001 * xji_11) -
            vderiv_0_2 * (ui_derxjm_001 * xji_12) -
            vderiv_1_0 * (xji_00 * ui_derxjm_001 + xji_00 * ui_derxjm_001 + xji_20 * ui_derxjm_201 +
                             xji_20 * ui_derxjm_201) -
            vderiv_1_1 * (xji_01 * ui_derxjm_001 + xji_00 * ui_derxjm_011 + xji_21 * ui_derxjm_201 +
                             xji_20 * ui_derxjm_211) -
            vderiv_1_2 * (xji_02 * ui_derxjm_001 + xji_00 * ui_derxjm_021 + xji_22 * ui_derxjm_201 +
                             xji_20 * ui_derxjm_221) -
            vderiv_2_0 * (ui_derxjm_201 * xji_10) - vderiv_2_1 * (ui_derxjm_201 * xji_11) -
            vderiv_2_2 * (ui_derxjm_201 * xji_12);
        const double v1 =
            -vderiv_0_0 * (ui_derxjm_011 * xji_10) - vderiv_0_1 * (ui_derxjm_011 * xji_11) -
            vderiv_0_2 * (ui_derxjm_011 * xji_12) -
            vderiv_1_0 * (xji_00 * ui_derxjm_011 + xji_01 * ui_derxjm_001 + xji_20 * ui_derxjm_211 +
                             xji_21 * ui_derxjm_201) -
            vderiv_1_1 * (xji_01 * ui_derxjm_011 + xji_01 * ui_derxjm_011 + xji_21 * ui_derxjm_211 +
                             xji_21 * ui_derxjm_211) -
            vderiv_1_2 * (xji_02 * ui_derxjm_011 + xji_01 * ui_derxjm_021 + xji_22 * ui_derxjm_211 +
                             xji_21 * ui_derxjm_221) -
            vderiv_2_0 * (ui_derxjm_211 * xji_10) - vderiv_2_1 * (ui_derxjm_211 * xji_11) -
            vderiv_2_2 * (ui_derxjm_211 * xji_12);
        const double v2 =
            -vderiv_0_0 * (ui_derxjm_021 * xji_10) - vderiv_0_1 * (ui_derxjm_021 * xji_11) -
            vderiv_0_2 * (ui_derxjm_021 * xji_12) -
            vderiv_1_0 * (xji_00 * ui_derxjm_021 + xji_02 * ui_derxjm_001 + xji_20 * ui_derxjm_221 +
                             xji_22 * ui_derxjm_201) -
            vderiv_1_1 * (xji_01 * ui_derxjm_021 + xji_02 * ui_derxjm_011 + xji_21 * ui_derxjm_221 +
                             xji_22 * ui_derxjm_211) -
            vderiv_1_2 * (xji_02 * ui_derxjm_021 + xji_02 * ui_derxjm_021 + xji_22 * ui_derxjm_221 +
                             xji_22 * ui_derxjm_221) -
            vderiv_2_0 * (ui_derxjm_221 * xji_10) - vderiv_2_1 * (ui_derxjm_221 * xji_11) -
            vderiv_2_2 * (ui_derxjm_221 * xji_12);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 1, ui * 4 + 1) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 =
            -vderiv_0_0 * (ui_derxjm_002 * xji_10 + ui_derxjm_102 * xji_00) -
            vderiv_0_1 * (ui_derxjm_002 * xji_11 + ui_derxjm_112 * xji_00) -
            vderiv_0_2 * (ui_derxjm_002 * xji_12 + ui_derxjm_122 * xji_00) -
            vderiv_1_0 * (xji_00 * ui_derxjm_002 + xji_00 * ui_derxjm_002 +
                             2 * xji_10 * ui_derxjm_102 + 2 * xji_10 * ui_derxjm_102) -
            vderiv_1_1 * (xji_01 * ui_derxjm_002 + xji_00 * ui_derxjm_012 +
                             2 * xji_11 * ui_derxjm_102 + 2 * xji_10 * ui_derxjm_112) -
            vderiv_1_2 * (xji_02 * ui_derxjm_002 + xji_00 * ui_derxjm_022 +
                             2 * xji_12 * ui_derxjm_102 + 2 * xji_10 * ui_derxjm_122) -
            vderiv_2_0 * (ui_derxjm_102 * xji_20) - vderiv_2_1 * (ui_derxjm_112 * xji_20) -
            vderiv_2_2 * (ui_derxjm_122 * xji_20);
        const double v1 =
            -vderiv_0_0 * (ui_derxjm_012 * xji_10 + ui_derxjm_102 * xji_01) -
            vderiv_0_1 * (ui_derxjm_012 * xji_11 + ui_derxjm_112 * xji_01) -
            vderiv_0_2 * (ui_derxjm_012 * xji_12 + ui_derxjm_122 * xji_01) -
            vderiv_1_0 * (xji_00 * ui_derxjm_012 + xji_01 * ui_derxjm_002 +
                             2 * xji_10 * ui_derxjm_112 + 2 * xji_11 * ui_derxjm_102) -
            vderiv_1_1 * (xji_01 * ui_derxjm_012 + xji_01 * ui_derxjm_012 +
                             2 * xji_11 * ui_derxjm_112 + 2 * xji_11 * ui_derxjm_112) -
            vderiv_1_2 * (xji_02 * ui_derxjm_012 + xji_01 * ui_derxjm_022 +
                             2 * xji_12 * ui_derxjm_112 + 2 * xji_11 * ui_derxjm_122) -
            vderiv_2_0 * (ui_derxjm_102 * xji_21) - vderiv_2_1 * (ui_derxjm_112 * xji_21) -
            vderiv_2_2 * (ui_derxjm_122 * xji_21);
        const double v2 =
            -vderiv_0_0 * (ui_derxjm_022 * xji_10 + ui_derxjm_102 * xji_02) -
            vderiv_0_1 * (ui_derxjm_022 * xji_11 + ui_derxjm_112 * xji_02) -
            vderiv_0_2 * (ui_derxjm_022 * xji_12 + ui_derxjm_122 * xji_02) -
            vderiv_1_0 * (xji_00 * ui_derxjm_022 + xji_02 * ui_derxjm_002 +
                             2 * xji_10 * ui_derxjm_122 + 2 * xji_12 * ui_derxjm_102) -
            vderiv_1_1 * (xji_01 * ui_derxjm_022 + xji_02 * ui_derxjm_012 +
                             2 * xji_11 * ui_derxjm_122 + 2 * xji_12 * ui_derxjm_112) -
            vderiv_1_2 * (xji_02 * ui_derxjm_022 + xji_02 * ui_derxjm_022 +
                             2 * xji_12 * ui_derxjm_122 + 2 * xji_12 * ui_derxjm_122) -
            vderiv_2_0 * (ui_derxjm_102 * xji_22) - vderiv_2_1 * (ui_derxjm_112 * xji_22) -
            vderiv_2_2 * (ui_derxjm_122 * xji_22);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 1, ui * 4 + 2) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 =
            -vderiv_0_0 * (ui_derxjm_200 * xji_00) - vderiv_0_1 * (ui_derxjm_210 * xji_00) -
            vderiv_0_2 * (ui_derxjm_220 * xji_00) -
            vderiv_1_0 * (ui_derxjm_200 * xji_10 + ui_derxjm_100 * xji_20) -
            vderiv_1_1 * (ui_derxjm_210 * xji_10 + ui_derxjm_100 * xji_21) -
            vderiv_1_2 * (ui_derxjm_220 * xji_10 + ui_derxjm_100 * xji_22) -
            vderiv_2_0 * (xji_10 * ui_derxjm_100 + xji_10 * ui_derxjm_100 +
                             2 * xji_20 * ui_derxjm_200 + 2 * xji_20 * ui_derxjm_200) -
            vderiv_2_1 * (xji_11 * ui_derxjm_100 + xji_10 * ui_derxjm_110 +
                             2 * xji_21 * ui_derxjm_200 + 2 * xji_20 * ui_derxjm_210) -
            vderiv_2_2 * (xji_12 * ui_derxjm_100 + xji_10 * ui_derxjm_120 +
                             2 * xji_22 * ui_derxjm_200 + 2 * xji_20 * ui_derxjm_220);
        const double v1 =
            -vderiv_0_0 * (ui_derxjm_200 * xji_01) - vderiv_0_1 * (ui_derxjm_210 * xji_01) -
            vderiv_0_2 * (ui_derxjm_220 * xji_01) -
            vderiv_1_0 * (ui_derxjm_200 * xji_11 + ui_derxjm_110 * xji_20) -
            vderiv_1_1 * (ui_derxjm_210 * xji_11 + ui_derxjm_110 * xji_21) -
            vderiv_1_2 * (ui_derxjm_220 * xji_11 + ui_derxjm_110 * xji_22) -
            vderiv_2_0 * (xji_10 * ui_derxjm_110 + xji_11 * ui_derxjm_100 +
                             2 * xji_20 * ui_derxjm_210 + 2 * xji_21 * ui_derxjm_200) -
            vderiv_2_1 * (xji_11 * ui_derxjm_110 + xji_11 * ui_derxjm_110 +
                             2 * xji_21 * ui_derxjm_210 + 2 * xji_21 * ui_derxjm_210) -
            vderiv_2_2 * (xji_12 * ui_derxjm_110 + xji_11 * ui_derxjm_120 +
                             2 * xji_22 * ui_derxjm_210 + 2 * xji_21 * ui_derxjm_220);
        const double v2 =
            -vderiv_0_0 * (ui_derxjm_200 * xji_02) - vderiv_0_1 * (ui_derxjm_210 * xji_02) -
            vderiv_0_2 * (ui_derxjm_220 * xji_02) -
            vderiv_1_0 * (ui_derxjm_200 * xji_12 + ui_derxjm_120 * xji_20) -
            vderiv_1_1 * (ui_derxjm_210 * xji_12 + ui_derxjm_120 * xji_21) -
            vderiv_1_2 * (ui_derxjm_220 * xji_12 + ui_derxjm_120 * xji_22) -
            vderiv_2_0 * (xji_10 * ui_derxjm_120 + xji_12 * ui_derxjm_100 +
                             2 * xji_20 * ui_derxjm_220 + 2 * xji_22 * ui_derxjm_200) -
            vderiv_2_1 * (xji_11 * ui_derxjm_120 + xji_12 * ui_derxjm_110 +
                             2 * xji_21 * ui_derxjm_220 + 2 * xji_22 * ui_derxjm_210) -
            vderiv_2_2 * (xji_12 * ui_derxjm_120 + xji_12 * ui_derxjm_120 +
                             2 * xji_22 * ui_derxjm_220 + 2 * xji_22 * ui_derxjm_220);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 2, ui * 4 + 0) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 =
            -vderiv_0_0 * (ui_derxjm_201 * xji_00 + ui_derxjm_001 * xji_20) -
            vderiv_0_1 * (ui_derxjm_211 * xji_00 + ui_derxjm_001 * xji_21) -
            vderiv_0_2 * (ui_derxjm_221 * xji_00 + ui_derxjm_001 * xji_22) -
            vderiv_1_0 * (ui_derxjm_201 * xji_10) - vderiv_1_1 * (ui_derxjm_211 * xji_10) -
            vderiv_1_2 * (ui_derxjm_221 * xji_10) -
            vderiv_2_0 * (xji_00 * ui_derxjm_001 + xji_00 * ui_derxjm_001 +
                             2 * xji_20 * ui_derxjm_201 + 2 * xji_20 * ui_derxjm_201) -
            vderiv_2_1 * (xji_01 * ui_derxjm_001 + xji_00 * ui_derxjm_011 +
                             2 * xji_21 * ui_derxjm_201 + 2 * xji_20 * ui_derxjm_211) -
            vderiv_2_2 * (xji_02 * ui_derxjm_001 + xji_00 * ui_derxjm_021 +
                             2 * xji_22 * ui_derxjm_201 + 2 * xji_20 * ui_derxjm_221);
        const double v1 =
            -vderiv_0_0 * (ui_derxjm_201 * xji_01 + ui_derxjm_011 * xji_20) -
            vderiv_0_1 * (ui_derxjm_211 * xji_01 + ui_derxjm_011 * xji_21) -
            vderiv_0_2 * (ui_derxjm_221 * xji_01 + ui_derxjm_011 * xji_22) -
            vderiv_1_0 * (ui_derxjm_201 * xji_11) - vderiv_1_1 * (ui_derxjm_211 * xji_11) -
            vderiv_1_2 * (ui_derxjm_221 * xji_11) -
            vderiv_2_0 * (xji_00 * ui_derxjm_011 + xji_01 * ui_derxjm_001 +
                             2 * xji_20 * ui_derxjm_211 + 2 * xji_21 * ui_derxjm_201) -
            vderiv_2_1 * (xji_01 * ui_derxjm_011 + xji_01 * ui_derxjm_011 +
                             2 * xji_21 * ui_derxjm_211 + 2 * xji_21 * ui_derxjm_211) -
            vderiv_2_2 * (xji_02 * ui_derxjm_011 + xji_01 * ui_derxjm_021 +
                             2 * xji_22 * ui_derxjm_211 + 2 * xji_21 * ui_derxjm_221);
        const double v2 =
            -vderiv_0_0 * (ui_derxjm_201 * xji_02 + ui_derxjm_021 * xji_20) -
            vderiv_0_1 * (ui_derxjm_211 * xji_02 + ui_derxjm_021 * xji_21) -
            vderiv_0_2 * (ui_derxjm_221 * xji_02 + ui_derxjm_021 * xji_22) -
            vderiv_1_0 * (ui_derxjm_201 * xji_12) - vderiv_1_1 * (ui_derxjm_211 * xji_12) -
            vderiv_1_2 * (ui_derxjm_221 * xji_12) -
            vderiv_2_0 * (xji_00 * ui_derxjm_021 + xji_02 * ui_derxjm_001 +
                             2 * xji_20 * ui_derxjm_221 + 2 * xji_22 * ui_derxjm_201) -
            vderiv_2_1 * (xji_01 * ui_derxjm_021 + xji_02 * ui_derxjm_011 +
                             2 * xji_21 * ui_derxjm_221 + 2 * xji_22 * ui_derxjm_211) -
            vderiv_2_2 * (xji_02 * ui_derxjm_021 + xji_02 * ui_derxjm_021 +
                             2 * xji_22 * ui_derxjm_221 + 2 * xji_22 * ui_derxjm_221);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 2, ui * 4 + 1) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }

      ////////////////////////////////////////////////////////////////

      {
        const double v0 =
            -vderiv_0_0 * (ui_derxjm_002 * xji_20) - vderiv_0_1 * (ui_derxjm_002 * xji_21) -
            vderiv_0_2 * (ui_derxjm_002 * xji_22) - vderiv_1_0 * (ui_derxjm_102 * xji_20) -
            vderiv_1_1 * (ui_derxjm_102 * xji_21) - vderiv_1_2 * (ui_derxjm_102 * xji_22) -
            vderiv_2_0 * (xji_00 * ui_derxjm_002 + xji_00 * ui_derxjm_002 + xji_10 * ui_derxjm_102 +
                             xji_10 * ui_derxjm_102) -
            vderiv_2_1 * (xji_01 * ui_derxjm_002 + xji_00 * ui_derxjm_012 + xji_11 * ui_derxjm_102 +
                             xji_10 * ui_derxjm_112) -
            vderiv_2_2 * (xji_02 * ui_derxjm_002 + xji_00 * ui_derxjm_022 + xji_12 * ui_derxjm_102 +
                             xji_10 * ui_derxjm_122);
        const double v1 =
            -vderiv_0_0 * (ui_derxjm_012 * xji_20) - vderiv_0_1 * (ui_derxjm_012 * xji_21) -
            vderiv_0_2 * (ui_derxjm_012 * xji_22) - vderiv_1_0 * (ui_derxjm_112 * xji_20) -
            vderiv_1_1 * (ui_derxjm_112 * xji_21) - vderiv_1_2 * (ui_derxjm_112 * xji_22) -
            vderiv_2_0 * (xji_00 * ui_derxjm_012 + xji_01 * ui_derxjm_002 + xji_10 * ui_derxjm_112 +
                             xji_11 * ui_derxjm_102) -
            vderiv_2_1 * (xji_01 * ui_derxjm_012 + xji_01 * ui_derxjm_012 + xji_11 * ui_derxjm_112 +
                             xji_11 * ui_derxjm_112) -
            vderiv_2_2 * (xji_02 * ui_derxjm_012 + xji_01 * ui_derxjm_022 + xji_12 * ui_derxjm_112 +
                             xji_11 * ui_derxjm_122);
        const double v2 =
            -vderiv_0_0 * (ui_derxjm_022 * xji_20) - vderiv_0_1 * (ui_derxjm_022 * xji_21) -
            vderiv_0_2 * (ui_derxjm_022 * xji_22) - vderiv_1_0 * (ui_derxjm_122 * xji_20) -
            vderiv_1_1 * (ui_derxjm_122 * xji_21) - vderiv_1_2 * (ui_derxjm_122 * xji_22) -
            vderiv_2_0 * (xji_00 * ui_derxjm_022 + xji_02 * ui_derxjm_002 + xji_10 * ui_derxjm_122 +
                             xji_12 * ui_derxjm_102) -
            vderiv_2_1 * (xji_01 * ui_derxjm_022 + xji_02 * ui_derxjm_012 + xji_11 * ui_derxjm_122 +
                             xji_12 * ui_derxjm_112) -
            vderiv_2_2 * (xji_02 * ui_derxjm_022 + xji_02 * ui_derxjm_022 + xji_12 * ui_derxjm_122 +
                             xji_12 * ui_derxjm_122);

        for (int vi = 0; vi < nen_; ++vi)
        {
          emesh(vi * 4 + 2, ui * 4 + 2) +=
              v * (deriv_(0, vi) * v0 + deriv_(1, vi) * v1 + deriv_(2, vi) * v2);
        }
      }
    }
  }


  // pressure
  const double v = press * timefacfac / det_;
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      emesh(vi * 4, ui * 4 + 1) +=
          v * (deriv_(0, vi) * derxjm_(0, 0, 1, ui) + deriv_(1, vi) * derxjm_(0, 1, 1, ui) +
                  deriv_(2, vi) * derxjm_(0, 2, 1, ui));
      emesh(vi * 4, ui * 4 + 2) +=
          v * (deriv_(0, vi) * derxjm_(0, 0, 2, ui) + deriv_(1, vi) * derxjm_(0, 1, 2, ui) +
                  deriv_(2, vi) * derxjm_(0, 2, 2, ui));

      emesh(vi * 4 + 1, ui * 4 + 0) +=
          v * (deriv_(0, vi) * derxjm_(1, 0, 0, ui) + deriv_(1, vi) * derxjm_(1, 1, 0, ui) +
                  deriv_(2, vi) * derxjm_(1, 2, 0, ui));
      emesh(vi * 4 + 1, ui * 4 + 2) +=
          v * (deriv_(0, vi) * derxjm_(1, 0, 2, ui) + deriv_(1, vi) * derxjm_(1, 1, 2, ui) +
                  deriv_(2, vi) * derxjm_(1, 2, 2, ui));

      emesh(vi * 4 + 2, ui * 4 + 0) +=
          v * (deriv_(0, vi) * derxjm_(2, 0, 0, ui) + deriv_(1, vi) * derxjm_(2, 1, 0, ui) +
                  deriv_(2, vi) * derxjm_(2, 2, 0, ui));
      emesh(vi * 4 + 2, ui * 4 + 1) +=
          v * (deriv_(0, vi) * derxjm_(2, 0, 1, ui) + deriv_(1, vi) * derxjm_(2, 1, 1, ui) +
                  deriv_(2, vi) * derxjm_(2, 2, 1, ui));
    }
  }

  // div u
  const double timefacfac_det = timefacfac / det_;
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = timefacfac_det * funct_(vi, 0);
    for (int ui = 0; ui < nen_; ++ui)
    {
      emesh(vi * 4 + 3, ui * 4 + 0) +=
          v * (+vderiv_1_0 * derxjm_(0, 0, 1, ui) + vderiv_1_1 * derxjm_(0, 1, 1, ui) +
                  vderiv_1_2 * derxjm_(0, 2, 1, ui) + vderiv_2_0 * derxjm_(0, 0, 2, ui) +
                  vderiv_2_1 * derxjm_(0, 1, 2, ui) + vderiv_2_2 * derxjm_(0, 2, 2, ui));

      emesh(vi * 4 + 3, ui * 4 + 1) +=
          v * (+vderiv_0_0 * derxjm_(1, 0, 0, ui) + vderiv_0_1 * derxjm_(1, 1, 0, ui) +
                  vderiv_0_2 * derxjm_(1, 2, 0, ui) + vderiv_2_0 * derxjm_(1, 0, 2, ui) +
                  vderiv_2_1 * derxjm_(1, 1, 2, ui) + vderiv_2_2 * derxjm_(1, 2, 2, ui));

      emesh(vi * 4 + 3, ui * 4 + 2) +=
          v * (+vderiv_0_0 * derxjm_(2, 0, 0, ui) + vderiv_0_1 * derxjm_(2, 1, 0, ui) +
                  vderiv_0_2 * derxjm_(2, 2, 0, ui) + vderiv_1_0 * derxjm_(2, 0, 1, ui) +
                  vderiv_1_1 * derxjm_(2, 1, 1, ui) + vderiv_1_2 * derxjm_(2, 2, 1, ui));
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_gal_rhs_cont_eq(
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    bool isale)
{
  //----------------------------------------------------------------------
  // compute additional Galerkin terms on right-hand side of continuity
  // equation (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  /*

           /                                                dp   \
          |         1     / dT     /         \   \     1      th  |
          |    q , --- * | ---- + | u o nabla | T | - --- * ----  |
          |         T     \ dt     \         /   /    p      dt   |
           \                                           th        /
           +-----------------------------------------------------+
                           Galerkin part of rhscon_
  */

  // convective term (identical for all time-integration schemes,
  // while being the only component for stationary scheme)
  // gradient of scalar value at n+alpha_F/n+1
  grad_scaaf_.multiply(derxy_, escaaf);

  // convective scalar term at n+alpha_F/n+1
  conv_scaaf_ = convvelint_.dot(grad_scaaf_);

  // add to rhs of continuity equation
  rhscon_ = scaconvfacaf_ * conv_scaaf_;

  // further terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->is_genalpha())
  {
    // time derivative of scalar at n+alpha_M
    tder_sca_ = funct_.dot(escadtam);

    // add to rhs of continuity equation
    rhscon_ += scadtfac_ * tder_sca_ + thermpressadd_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->is_stationary())
    {
      // get velocity at n (including grid velocity in ALE case)
      convvelintn_.multiply(eveln, funct_);
      if (isale) convvelintn_.update(-1.0, gridvelint_, 1.0);

      // get velocity derivatives at n
      vderxyn_.multiply_nt(eveln, derxy_);

      // velocity divergence at n
      vdivn_ = 0.0;
      for (int idim = 0; idim < nsd_; ++idim)
      {
        vdivn_ += vderxyn_(idim, idim);
      }

      // scalar value at n+1
      scaaf_ = funct_.dot(escaaf);

      // scalar value at n
      scan_ = funct_.dot(escaam);

      // gradient of scalar value at n
      grad_scan_.multiply(derxy_, escaam);

      // convective scalar term at n
      conv_scan_ = convvelintn_.dot(grad_scan_);

      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ +=
          (scadtfac_ * (scaaf_ - scan_) / fldparatimint_->dt() +
              fldparatimint_->om_theta() * (scaconvfacn_ * conv_scan_ - vdivn_) + thermpressadd_) /
          fldparatimint_->theta();
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_gal_rhs_cont_eq_weak_comp(
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epredtam,
    bool isale)
{
  //----------------------------------------------------------------------
  // compute additional Galerkin terms on right-hand side of continuity
  // equation (only required for weakly compressibility)
  //----------------------------------------------------------------------

  /*

             /                                 \     /                                         \
            |         1     /               \   |   |               1          /             \  |
            |  q , - --- * |  grad(rho) o u  |  | = | q , - --------------- * |  grad(p) o u  | |
            |        rho    \               /   |   |        K_0+n*(p-p_0)     \             /  |
             \                                 /     \                                         /
             +---------------------------------+
                  Galerkin part of rhscon_
  */
  // recover the convective velocity for the evaluation of RHS of continuity equation
  // in case of a weakly_compressible_stokes problem
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
  {
    convvelint_.update(velint_);
    if (isale)
    {
      convvelint_.update(-1.0, gridvelint_, 1.0);
    }
  }

  // convective term (identical for all time-integration schemes,
  // while being the only component for stationary scheme)
  // gradient of pressure at n+alpha_F/n+1
  grad_preaf_.multiply(derxy_, epreaf);

  // convective pressure term at n+alpha_F/n+1
  conv_preaf_ = convvelint_.dot(grad_preaf_);

  // add to rhs of continuity equation
  rhscon_ = preconvfacaf_ * conv_preaf_;

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // add correction term to RHS of continuity equation in order to match
  // the approximated analytical solution of the channale_weakly_compressible problem
  // given in "New analytical solutions for weakly compressible Newtonian
  // Poiseuille flows with pressure-dependent viscosity"
  // Kostas D. Housiadas, Georgios C. Georgiou

  // correction term in gausspoint
  Core::LinAlg::Matrix<1, 1> correctionterm;

  // get the correction term at integration point
  correctionterm.multiply(ecorrectionterm_, funct_);

  // add correction term to rhs of continuity equation
  rhscon_ += correctionterm(0, 0);
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  // pressure inertia terms on the right hand side for instationary fluids
  if (not fldparatimint_->is_stationary())
  {
    // get derivative of pressure at time n+alpha_M at integration point
    // prederint_.multiply(epredtam,funct_);
    double prederint = 0.0;
    for (int ui = 0; ui < nen_; ++ui)
    {
      prederint += epredtam(ui) * funct_(ui);
    }

    // add to rhs of continuity equation
    rhscon_ += predtfac_ * prederint;
  }

  // delete again the convective velocity
  // in case of a weakly_compressible_stokes problem
  if (fldpara_->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
  {
    convvelint_.clear();
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_gal_rhs_cont_eq_art_comp(
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epren,
    const Core::LinAlg::Matrix<nen_, 1>& escadtam)
{
  //----------------------------------------------------------------------
  // compute additional Galerkin terms on right-hand side of continuity
  // equation for artificial compressibility
  //----------------------------------------------------------------------
  /*

            /                      \
           |           1      dp   |
       -   |    q ,   --- *  ----  |
           |           c^2    dt   |
            \                     /
            +----------------------+
            Galerkin part of rhscon_
  */

  // terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->is_genalpha())
  {
    // time derivative of scalar (i.e., pressure in this case) at n+alpha_M
    tder_sca_ = funct_.dot(escadtam);

    // add to rhs of continuity equation
    rhscon_ = -scadtfac_ * tder_sca_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->is_stationary())
    {
      // scalar value (i.e., pressure in this case) at n+1
      scaaf_ = funct_.dot(epreaf);

      // scalar value (i.e., pressure in this case) at n
      scan_ = funct_.dot(epren);

      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ = -scadtfac_ * (scaaf_ - scan_) / (fldparatimint_->dt() * fldparatimint_->theta());
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_subgrid_scale_scalar(
    const Core::LinAlg::Matrix<nen_, 1>& escaaf, const Core::LinAlg::Matrix<nen_, 1>& escaam)
{
  //----------------------------------------------------------------------
  // compute residual of scalar equation
  // -> different for generalized-alpha and other time-integration schemes
  // (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  // define residual
  double scares_old = 0.0;

  // compute diffusive term at n+alpha_F/n+1 for higher-order elements
  Core::LinAlg::Matrix<nen_, 1> diff;
  double diff_scaaf = 0.0;
  if (is_higher_order_ele_)
  {
    diff.clear();
    // compute N,xx + N,yy + N,zz for each shape function
    for (int i = 0; i < nen_; ++i)
    {
      for (int j = 0; j < nsd_; ++j)
      {
        diff(i) += derxy2_(j, i);
      }
    }
    diff.scale(diffus_);
    diff_scaaf = diff.dot(escaaf);
  }

  if (fldparatimint_->is_genalpha())
    scares_old = densam_ * tder_sca_ + densaf_ * conv_scaaf_ - diff_scaaf - scarhs_;
  else
  {
    if (not fldparatimint_->is_stationary())
    {
      // compute diffusive term at n for higher-order elements
      double diff_scan = 0.0;
      if (is_higher_order_ele_) diff_scan = diff.dot(escaam);

      scares_old = densaf_ * (scaaf_ - scan_) / fldparatimint_->dt() +
                   fldparatimint_->theta() * (densaf_ * conv_scaaf_ - diff_scaaf) +
                   fldparatimint_->om_theta() * (densn_ * conv_scan_ - diff_scan) - scarhs_;
    }
    else
      scares_old = densaf_ * conv_scaaf_ - diff_scaaf - scarhs_;
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale part of scalar
  // (For simplicity, stabilization parameter tau_Mu is used here instead
  //  of exactly calculating the stabilization parameter tau for the scalar
  //  equation; differences should be minor for Prandtl numbers or ratios
  //  of viscosity and diffusivity (for mixture-fraction equation),
  //  respectively, close to one.)
  //----------------------------------------------------------------------
  sgscaint_ = -tau_(0) * scares_old;

  return;
}


/*----------------------------------------------------------------------*
 |  update material parameters including s.-s. part of scalar  vg 10/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::update_material_params(
    std::shared_ptr<const Core::Mat::Material> material,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nen_, 1>& epreaf,
    const Core::LinAlg::Matrix<nen_, 1>& epream, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const double thermpressaf,
    const double thermpressam, const double sgsca)
{
  if (material->material_type() == Core::Materials::m_sutherland)
  {
    const Mat::Sutherland* actmat = static_cast<const Mat::Sutherland*>(material.get());

    // compute temperature at n+alpha_F or n+1
    double tempaf = funct_.dot(escaaf);

    // add subgrid-scale part to obtain complete temperature
    // and check whether it is positive
    tempaf += sgsca;
    if (tempaf < 0.0)
      FOUR_C_THROW("Negative temperature in Fluid Sutherland material-update evaluation!");

    // compute viscosity according to Sutherland law
    visc_ = actmat->compute_viscosity(tempaf);

    // compute density at n+alpha_F or n+1 based on temperature
    // and thermodynamic pressure
    densaf_ = actmat->compute_density(tempaf, thermpressaf);

    // factor for convective scalar term at n+alpha_F or n+1
    scaconvfacaf_ = 1.0 / tempaf;

    if (fldparatimint_->is_genalpha())
    {
      // compute temperature at n+alpha_M
      double tempam = funct_.dot(escaam);

      // add subgrid-scale part to obtain complete temperature
      tempam += sgsca;

      // factor for scalar time derivative at n+alpha_M
      scadtfac_ = 1.0 / tempam;

      // compute density at n+alpha_M based on temperature
      densam_ = actmat->compute_density(tempam, thermpressam);
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam_ = densaf_;

      if (not fldparatimint_->is_stationary())
      {
        // compute temperature at n
        double tempn = funct_.dot(escaam);

        // add subgrid-scale part to obtain complete temperature
        tempn += sgsca;

        // compute density at n based on temperature at n and
        // (approximately) thermodynamic pressure at n+1
        densn_ = actmat->compute_density(tempn, thermpressaf);

        // factor for convective scalar term at n
        scaconvfacn_ = 1.0 / tempn;

        // factor for scalar time derivative
        scadtfac_ = scaconvfacaf_;
      }
    }
  }
  else if (material->material_type() == Core::Materials::m_fluid_linear_density_viscosity)
  {
    const Mat::LinearDensityViscosity* actmat =
        static_cast<const Mat::LinearDensityViscosity*>(material.get());

    double RefPressure = actmat->ref_pressure();    // reference pressure
    double CoeffDensity = actmat->coeff_density();  // density-pressure coefficient

    // compute pressure at n+alpha_F or n+1
    preaf_ = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf_);

    // compute viscosity based on pressure
    visc_ = actmat->compute_viscosity(preaf_);

    // factor for convective pressure term at n+alpha_F or n+1
    preconvfacaf_ = -1.0 / ((preaf_ - RefPressure) + 1.0 / CoeffDensity);

    if (fldparatimint_->is_genalpha())
    {
      // compute pressure at n+alpha_M
      pream_ = funct_.dot(epream);

      // compute density at n+alpha_M based on pressure
      densam_ = actmat->compute_density(pream_);

      // factor for pressure time derivative at n+alpha_M
      predtfac_ = -1.0 / ((pream_ - RefPressure) + 1.0 / CoeffDensity);
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam_ = densaf_;

      if (not fldparatimint_->is_stationary())
      {
        FOUR_C_THROW("Genalpha is the only scheme implemented for weakly compressibility");
      }
    }
  }
  else if (material->material_type() == Core::Materials::m_fluid_murnaghantait)
  {
    const Mat::MurnaghanTaitFluid* actmat =
        static_cast<const Mat::MurnaghanTaitFluid*>(material.get());

    double RefPressure = actmat->ref_pressure();         // reference pressure
    double RefBulkModulus = actmat->ref_bulk_modulus();  // reference bulk modulus
    double MatParameter =
        actmat->mat_parameter();  // material parameter according to Murnaghan-Tait

    // compute pressure at n+alpha_F or n+1
    preaf_ = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf_);

    // factor for convective pressure term at n+alpha_F or n+1
    preconvfacaf_ = -1.0 / (MatParameter * (preaf_ - RefPressure) + RefBulkModulus);

    if (fldparatimint_->is_genalpha())
    {
      // compute pressure at n+alpha_M
      pream_ = funct_.dot(epream);

      // compute density at n+alpha_M based on pressure
      densam_ = actmat->compute_density(pream_);

      // factor for pressure time derivative at n+alpha_M
      predtfac_ = -1.0 / (MatParameter * (pream_ - RefPressure) + RefBulkModulus);
    }
    else
    {
      // set density at n+1 at location n+alpha_M as well
      densam_ = densaf_;

      if (not fldparatimint_->is_stationary())
      {
        FOUR_C_THROW("Genalpha is the only scheme implemented for weakly compressibility");
      }
    }
  }
  else
    FOUR_C_THROW("Update of material parameters not required for this material type!");

  return;
}  // FluidEleCalc::update_material_params



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype,
    enrtype>::recompute_gal_and_compute_cross_rhs_cont_eq()
{
  //----------------------------------------------------------------------
  // recompute Galerkin terms based on updated material parameters
  // including s.-s. part of scalar and compute cross-stress term on
  // right-hand side of continuity equation
  // (only required for variable-density flow at low Mach number)
  //----------------------------------------------------------------------
  /*

           /                                                       dp   \
          |         1     / dT     /     ^         \   \     1      th  |
          |    q , --- * | ---- + | (u + u) o nabla | T | - --- * ----  |
          |         T     \ dt     \               /   /    p      dt   |
           \                                                 th        /
           +-----------------------------------------------------+
            Galerkin part of rhscon_ including cross-stress term
  */

  // add convective term to rhs of continuity equation
  // (identical for all time-integration schemes)
  rhscon_ = scaconvfacaf_ * conv_scaaf_;

  // add (first) subgrid-scale-velocity part to rhs of continuity equation
  // (identical for all time-integration schemes)
  if (fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none)
  {
    rhscon_ += scaconvfacaf_ * sgvelint_.dot(grad_scaaf_);
  }

  if (fldpara_->multi_frac_loma_conti())
  {
    rhscon_ += scaconvfacaf_ * mffsvelint_.dot(grad_scaaf_);    // first cross-stress term
    rhscon_ += scaconvfacaf_ * velint_.dot(grad_fsscaaf_);      // second cross-stress term
    rhscon_ += scaconvfacaf_ * mffsvelint_.dot(grad_fsscaaf_);  // Reynolds-stress term
    //    rhscon_ -= mffsvdiv_; // multifractal divergence
  }

  // further terms different for general.-alpha and other time-int. schemes
  if (fldparatimint_->is_genalpha())
  {
    // add to rhs of continuity equation
    rhscon_ += scadtfac_ * tder_sca_ + thermpressadd_;
  }
  else
  {
    // instationary case
    if (not fldparatimint_->is_stationary())
    {
      // add to rhs of continuity equation
      // (prepared for later multiplication by theta*dt in
      //  evaluation of element matrix and vector contributions)
      rhscon_ +=
          (scadtfac_ * (scaaf_ - scan_) / fldparatimint_->dt() +
              fldparatimint_->om_theta() * (scaconvfacn_ * conv_scan_ - vdivn_) + thermpressadd_) /
          fldparatimint_->theta();

      // add second subgrid-scale-velocity part to rhs of continuity equation
      // (subgrid-scale velocity at n+1 also approximately used at n)
      if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none)
        rhscon_ += (fldparatimint_->om_theta() / fldparatimint_->theta()) * scaconvfacn_ *
                   sgvelint_.dot(grad_scan_);
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::loma_gal_part(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, 1>& preforce,
    const double& timefacfac, const double& rhsfac)
{
  //----------------------------------------------------------------------
  // computation of additional terms for low-Mach-number flow:
  // 2) additional rhs term of continuity equation
  //----------------------------------------------------------------------

  if (fldpara_->physical_type() != Inpar::FLUID::weakly_compressible &&
      fldpara_->physical_type() != Inpar::FLUID::weakly_compressible_stokes)
  {
    if (fldpara_->is_newton())
    {
      const double timefacfac_scaconvfacaf = timefacfac * scaconvfacaf_;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int fui = nsd_ * ui;

        const double timefacfac_scaconvfacaf_funct_ui = timefacfac_scaconvfacaf * funct_(ui);

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          const double temp = timefacfac_scaconvfacaf_funct_ui * grad_scaaf_(jdim);

          for (int vi = 0; vi < nen_; ++vi)
          {
            // const int fvippp= numdofpernode_*vi+nsd_;


            /*
                  factor afgtd/am

                          /                    \
                    1    |       /         \    |
                   --- * |  q , | Du o grad | T |
                    T    |       \         /    |
                          \                    /
             */
            estif_q_u(vi, fui + jdim) -= temp * funct_(vi);
          }
        }
      }
    }  // end if (is_newton_)

    const double rhsfac_rhscon = rhsfac * rhscon_;
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* additional rhs term of continuity equation */
      preforce(vi) += rhsfac_rhscon * funct_(vi);
    }
  }
  else
  {
    if (fldpara_->is_newton())
    {
      const double timefacfac_preconvfacaf = timefacfac * preconvfacaf_;

      for (int ui = 0; ui < nen_; ++ui)
      {
        const int fui = nsd_ * ui;

        const double timefacfac_preconvfacaf_funct_ui = timefacfac_preconvfacaf * funct_(ui);

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          const double pre = timefacfac_preconvfacaf_funct_ui * grad_preaf_(jdim);

          for (int vi = 0; vi < nen_; ++vi)
          {
            estif_q_u(vi, fui + jdim) -= pre * funct_(vi);
          }
        }
      }
    }  // end if (is_newton_)

    const double rhsfac_rhscon = rhsfac * rhscon_;
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* additional rhs term of continuity equation */
      preforce(vi) += rhsfac_rhscon * funct_(vi);
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::
    art_comp_pressure_inertia_gal_partand_cont_stab(
        Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nen_, nen_>& ppmat)
{
  /* pressure inertia term if not is_stationary */
  /*
            /             \
           |   1           |
           |  ---  Dp , q  |
           | beta^2         |
            \             /
  */
  double prefac = scadtfac_ * fac_;
  for (int ui = 0; ui < nen_; ++ui)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      ppmat(vi, ui) += prefac * funct_(ui) * funct_(vi);
    }  // vi
  }  // ui

  if (fldpara_->c_stab())
  {
    /* continuity stabilisation on left-hand side for artificial compressibility */
    /*
                /                      \
               |   1                   |
          tauC |  ---  Dp , nabla o v  |
               |   c^2                 |
                \                     /
    */

    prefac *= tau_(2);

    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = nsd_ * vi;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          estif_p_v(fvi + jdim, ui) += prefac * funct_(ui) * derxy_(jdim, vi);
        }
      }
    }
  }

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::weak_comp_pressure_inertia_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nen_, nen_>& ppmat)
{
  /* pressure inertia term for instationary fluids */
  /*

             /                           \
            |              1              |
            |  q ,  --------------- *  Dp |
            |        K_0+n*(p-p_0)        |
             \                           /

  */
  // instationary case
  if (not fldparatimint_->is_stationary())
  {
    // evaluation of the "compressibility factor"
    double compr_fac = -predtfac_ * fac_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        ppmat(vi, ui) += compr_fac * funct_(ui) * funct_(vi);
      }  // vi
    }  // ui

    if (fldpara_->c_stab())
    {
      /* continuity stabilisation on left-hand side for weakly_compressible flow */
      /*
                        /                                   \
                       |          1                          |
                  tauC |  --------------- *  Dp , nabla o v  |
                       |   K_0+n*(p-p_0)                     |
                        \                                   /
                         */

      compr_fac *= tau_(2);

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = nsd_ * vi;

          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_p_v(fvi + jdim, ui) += compr_fac * funct_(ui) * derxy_(jdim, vi);
          }
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)   g.bau 03/07|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::sysmat_ost_new(
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgn,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& evelam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const Core::LinAlg::Matrix<nen_, 1>& fsescaaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evel_hat,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& ereynoldsstress_hat,
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epream,
    const Core::LinAlg::Matrix<nen_, 1>& epren, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nen_, 1>& escabofoaf, const Core::LinAlg::Matrix<nen_, 1>& escabofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridv, const Core::LinAlg::Matrix<nsd_, nen_>& egridvn,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& estif,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce, const Core::LinAlg::Matrix<nen_, 1>& eporo,
    const Core::LinAlg::Matrix<nsd_, 2 * nen_>& egradphi,
    const Core::LinAlg::Matrix<nen_, 2 * 1>& ecurvature, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    std::shared_ptr<const Core::Mat::Material> material, double& Cs_delta_sq, double& Ci_delta_sq,
    double& Cv, bool isale, double* saccn, double* sveln, double* svelnp,
    const Core::FE::GaussIntegration& intpoints)
{
  //------------------------------------------------------------------------
  //  preliminary definitions and evaluations
  //------------------------------------------------------------------------
  // definition of matrices
  estif_u_.clear();
  estif_p_v_.clear();
  estif_q_u_.clear();
  ppmat_.clear();

  // definition of vectors
  preforce_.clear();
  velforce_.clear();

  // definition of velocity-based momentum residual vectors
  lin_resM_Du_.clear();
  resM_Du_.clear();

  // if polynomial pressure projection: reset variables
  if (fldpara_->ppp())
  {
    D_ = 0;
    E_.clear();
  }

  // evaluate shape functions and derivatives at element center
  eval_shape_func_and_derivs_at_ele_center();

  // set element area or volume
  const double vol = fac_;

  //------------------------------------------------------------------------
  // potential evaluation of material parameters, subgrid viscosity
  // and/or stabilization parameters at element center
  //------------------------------------------------------------------------
  // get material parameters at element center
  if (not fldpara_->mat_gp() or not fldpara_->tau_gp())
  {
    get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf, thermpressaf,
        thermpressam, thermpressdtaf, thermpressdtam, vol);

    // calculate all-scale or fine-scale subgrid viscosity at element center
    visceff_ = visc_;

    if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::vreman or
        fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
    {
      if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
        Cs_delta_sq = Cv;  // use the declaration of Cs_delta_sq for the dynamic Vreman constant
      calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
      // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
      visceff_ += sgvisc_;
    }
    else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
      calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
  }

  // potential evaluation of multifractal subgrid-scales at element center
  // coefficient B of fine-scale velocity
  static Core::LinAlg::Matrix<nsd_, 1> B_mfs(Core::LinAlg::Initialization::zero);
  B_mfs.clear();

  // coefficient D of fine-scale scalar (loma only)
  double D_mfs = 0.0;
  if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    if (not fldpara_->b_gp())
    {
      // make sure to get material parameters at element center
      if (fldpara_->mat_gp())
        // get_material_params(material,evelaf,epreaf,epream,escaaf,escaam,thermpressaf,thermpressam,thermpressdtam,vol);
        get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
            thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);

      // provide necessary velocities and gradients at element center
      velint_.multiply(evelaf, funct_);
      fsvelint_.multiply(fsevelaf, funct_);
      vderxy_.multiply_nt(evelaf, derxy_);
      // calculate parameters of multifractal subgrid-scales and, finally,
      // calculate coefficient for multifractal modeling of subgrid velocity
      // if loma, calculate coefficient for multifractal modeling of subgrid scalar
      prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      // clear all velocities and gradients
      velint_.clear();
      fsvelint_.clear();
      vderxy_.clear();
    }
  }


  // calculate stabilization parameter at element center
  if (not fldpara_->tau_gp() and fldpara_->stab_type() == Inpar::FLUID::stabtype_residualbased)
  {
    // get convective velocity at element center
    // for evaluation of stabilization parameter
    velint_.multiply(evelaf, funct_);

    // get the grid velocity in case of ALE
    if (isale)
    {
      gridvelint_.multiply(egridv, funct_);
      gridvelintn_.multiply(egridvn, funct_);
    }

    // get convective velocity at integration point
    set_convective_velint(isale);


    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)
    {
      // get velocity derivatives at integration point
      // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
      vderxy_.multiply_nt(evelaf, derxy_);  // required for time-dependent subscales

      // compute velnp at integration point (required for time-dependent subscales)
      static Core::LinAlg::Matrix<nsd_, 1> velintnp(Core::LinAlg::Initialization::zero);
      velintnp.multiply(evelnp, funct_);
      vel_normnp_ = velintnp.norm2();
    }

    // calculate stabilization parameters at element center
    calc_stab_parameter(vol);
  }

  // get Gaussian integration points
  // const Core::FE::IntegrationPoints3D intpoints(ele->gaussrule_);
  // const Core::FE::IntPointsAndWeights<nsd_>
  // intpoints(Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  //------------------------------------------------------------------------
  //  start loop over integration points
  //------------------------------------------------------------------------
  // for (int iquad=0; iquad<intpoints.ip().nquad; ++iquad)

  for (Core::FE::GaussIntegration::const_iterator iquad = intpoints.begin();
      iquad != intpoints.end(); ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    //----------------------------------------------------------------------
    //  evaluation of various values at integration point:
    //  1) velocity (including derivatives, fine-scale and grid velocity)
    //  2) pressure (including derivatives)
    //  3) body-force vector
    //  4) "history" vector for momentum equation
    //----------------------------------------------------------------------
    // get velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    velint_.multiply(evelaf, funct_);

    // get velocity derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    vderxy_.multiply_nt(evelaf, derxy_);

    // get fine-scale velocity and its derivatives at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvderxy_.clear();
    }
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      fsvelint_.multiply(fsevelaf, funct_);
      fsvderxy_.multiply_nt(fsevelaf, derxy_);
    }
    else
    {
      fsvelint_.clear();
    }

    // get the grid velocity in case of ALE
    if (isale)
    {
      gridvelint_.multiply(egridv, funct_);
      gridvelintn_.multiply(egridvn, funct_);
    }

    // get convective velocity at integration point
    set_convective_velint(isale);


    // get pressure at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    double press = 0.0;
    if (fldparatimint_->is_genalpha_np())
      press = funct_.dot(eprenp);
    else
      press = funct_.dot(epreaf);

    // get pressure gradient at integration point
    // (value at n+alpha_F for generalized-alpha scheme,
    //  value at n+alpha_F for generalized-alpha-NP schemen, n+1 otherwise)
    if (fldparatimint_->is_genalpha_np())
      gradp_.multiply(derxy_, eprenp);
    else
      gradp_.multiply(derxy_, epreaf);

    // get bodyforce at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    bodyforce_.multiply(ebofoaf, funct_);
    bodyforcen_.multiply(ebofon, funct_);

    // get prescribed pressure gradient acting as body force
    // (required for turbulent channel flow)
    generalbodyforce_.multiply(eprescpgaf, funct_);
    generalbodyforcen_.multiply(eprescpgn, funct_);

    // get momentum history data at integration point
    // (only required for one-step-theta and BDF2 time-integration schemes)
    histmom_.multiply(emhist, funct_);


    //----------------------------------------------------------------------
    // potential evaluation of material parameters, subgrid viscosity
    // and/or stabilization parameters at integration point
    //----------------------------------------------------------------------
    // get material parameters at integration point

    if (fldpara_->mat_gp())
    {
      get_material_params(material, evelaf, epreaf, epream, escaaf, escaam, escabofoaf,
          thermpressaf, thermpressam, thermpressdtaf, thermpressdtam, vol);

      // calculate all-scale or fine-scale subgrid viscosity at integration point
      visceff_ = visc_;

      if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
          fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      {
        calc_subgr_visc(evelaf, vol, Cs_delta_sq, Ci_delta_sq);
        // effective viscosity = physical viscosity + (all-scale) subgrid viscosity
        visceff_ += sgvisc_;
      }
      else if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
        calc_fine_scale_subgr_visc(evelaf, fsevelaf, vol);
    }

    // calculate stabilization parameter at integration point
    if (fldpara_->tau_gp() and fldpara_->stab_type() == Inpar::FLUID::stabtype_residualbased)
      calc_stab_parameter(vol);

    // potential evaluation of coefficient of multifractal subgrid-scales at integration point
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      if (fldpara_->b_gp())
      {
        // make sure to get material parameters at gauss point
        if (not fldpara_->mat_gp())
        {
          // get_material_params(material,evelaf,epreaf,epream,escaaf,escaam,escabofoaf,thermpressaf,thermpressam,thermpressdtaf,thermpressdtam);
          // would overwrite materials at the element center, hence BGp() should always be combined
          // with MatGp()
          FOUR_C_THROW(
              "evaluation of B and D at gauss-point should always be combined with evaluation "
              "material at gauss-point!");
        }

        // calculate parameters of multifractal subgrid-scales
        prepare_multifractal_subgr_scales(B_mfs, D_mfs, evelaf, fsevelaf, vol);
      }

      // calculate fine-scale velocity, its derivative and divergence for multifractal subgrid-scale
      // modeling
      for (int idim = 0; idim < nsd_; idim++)
        mffsvelint_(idim, 0) = fsvelint_(idim, 0) * B_mfs(idim, 0);

      for (int idim = 0; idim < nsd_; idim++)
      {
        for (int jdim = 0; jdim < nsd_; jdim++)
          mffsvderxy_(idim, jdim) = fsvderxy_(idim, jdim) * B_mfs(idim, 0);
      }

      mffsvdiv_ = mffsvderxy_(0, 0) + mffsvderxy_(1, 1) + mffsvderxy_(2, 2);

      // only required for variable-density flow at low Mach number
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
      {
        mfssgscaint_ = D_mfs * funct_.dot(fsescaaf);
        grad_fsscaaf_.multiply(derxy_, fsescaaf);
        for (int dim = 0; dim < nsd_; dim++) grad_fsscaaf_(dim, 0) *= D_mfs;
      }
      else
      {
        mfssgscaint_ = 0.0;
        grad_fsscaaf_.clear();
      }

      if (isale) FOUR_C_THROW("Multifractal subgrid-scales with ale not supported");
    }
    else
    {
      mffsvelint_.clear();
      mffsvderxy_.clear();
      mffsvdiv_ = 0.0;
    }

    // TODO: Need gradphi and curvature at time n. (Can be implemented for interface values at
    // timestep t^(n+1)) Adds surface tension force to the Gausspoint.
    // Note: has to be called after get_material_params(), otherwise gamma_ is uninitialized!!
    if (fldpara_->get_include_surface_tension())
      add_surface_tension_force(escaaf, escaam, egradphi, ecurvature);

    //----------------------------------------------------------------------
    //  evaluation of various partial operators at integration point
    //  1) convective term from previous iteration and convective operator
    //  2) viscous term from previous iteration and viscous operator
    //  3) divergence of velocity from previous iteration
    //----------------------------------------------------------------------

    // compute convective term from previous iteration and convective operator
    conv_old_.multiply(vderxy_, convvelint_);
    conv_c_.multiply_tn(derxy_, convvelint_);

    // compute viscous term from previous iteration and viscous operator
    if (is_higher_order_ele_)
      calc_div_eps(evelaf, eveln);
    else
    {
      visc_old_.clear();
      visc_oldn_.clear();
      viscs2_.clear();
    }

    // compute divergence of velocity from previous iteration
    vdiv_ = 0.0;
    vdivn_ = 0.0;
    vderxyn_.multiply_nt(eveln, derxy_);
    if (not fldparatimint_->is_genalpha_np())
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        vdiv_ += vderxy_(idim, idim);
        vdivn_ += vderxyn_(idim, idim);
      }
    }
    else
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        // get vdiv at time n+1 for np_genalpha,
        static Core::LinAlg::Matrix<nsd_, nsd_> vderxy(Core::LinAlg::Initialization::zero);
        vderxy.multiply_nt(evelnp, derxy_);
        vdiv_ += vderxy(idim, idim);
      }
    }

    // New One Step Theta variables (for old time step):
    velintn_.multiply(eveln, funct_);
    // get convective velocity at integration point
    set_convective_velint_n(isale);
    conv_oldn_.multiply(vderxyn_, convvelintn_);
    const double pressn = funct_.dot(epren);
    gradpn_.multiply(derxy_, epren);


    //-----------------------------------------------------------------------
    //       |          timefac         |  timefacpre     |    timefacrhs   |
    // ----------------------------------------------------------------------
    // OST   |                        dt*theta                              |
    //-----------------------------------------------------------------------
    // BDF2  |                        2/3 * dt                              |
    //-----------------------------------------------------------------------
    // Af GA |          alphaF*gamma*dt/alphaM            | gamma*dt/alphaM |
    //----------------------------------------------------------------------
    // NP GA | alphaF*gamma*dt/alphaM   | gamma*dt/alphaM | gamma*dt/alphaM |
    //-----------------------------------------------------------------------
    //----------------------------------------------------------------------
    // set time-integration factors for left- and right-hand side
    //----------------------------------------------------------------------
    const double timefacfac = fldparatimint_->time_fac() * fac_;
    const double timefacfacpre = fldparatimint_->time_fac_pre() * fac_;
    const double rhsfac = fldparatimint_->time_fac_rhs() * fac_;

    // For One Step Theta rhsfac is: \Delta t (1 - \theta)
    const double rhsfacn = (1 - fldparatimint_->theta()) * fldparatimint_->dt() * fac_;
    // For One Step Theta,
    // the density multiplied with the instationary term has to be evaluated at time = ( n + \theta
    // ).
    dens_theta_ = fldparatimint_->theta() * densaf_ + (1 - fldparatimint_->theta()) * densn_;

    //----------------------------------------------------------------------
    // computation of various subgrid-scale values and residuals
    //----------------------------------------------------------------------
    // compute residual of momentum equation and subgrid-scale velocity
    // -> residual of momentum equation different for generalized-alpha
    //    and other time-integration schemes
    double fac1 = 0.0;
    double fac2 = 0.0;
    double fac3 = 0.0;
    double facMtau = 0.0;
    compute_subgrid_scale_velocity_ost_new(
        eaccam, fac1, fac2, fac3, facMtau, *iquad, saccn, sveln, svelnp);

    // compute residual of continuity equation
    // residual contains velocity divergence only for incompressible flow
    conres_old_ = vdiv_;
    conres_oldn_ = vdivn_;

    // following computations only required for variable-density flow at low Mach number
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      compute_gal_rhs_cont_eq(eveln, escaaf, escaam, escadtam, isale);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;

      if (fldpara_->update_mat() or fldpara_->conti_supg() or
          fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none or
          fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none or
          fldpara_->multi_frac_loma_conti())
      {
        // compute subgrid-scale part of scalar
        // -> different for generalized-alpha and other time-integration schemes
        compute_subgrid_scale_scalar(escaaf, escaam);

        // update material parameters including subgrid-scale part of scalar
        if (fldpara_->update_mat())
        {
          // since we update the viscosity in the next step, a potential subgrid-scale velocity
          // would be overwritten
          if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
            FOUR_C_THROW("No material update in combination with smagorinsky model!");

          if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
            update_material_params(material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
                thermpressam, mfssgscaint_);
          else
            update_material_params(material, evelaf, epreaf, epream, escaaf, escaam, thermpressaf,
                thermpressam, sgscaint_);
          visceff_ = visc_;
          if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
              fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
            visceff_ += sgvisc_;
        }

        // right-hand side of continuity equation based on updated material parameters
        // and including all stabilization terms
        // -> different for generalized-alpha and other time-integration schemes
        recompute_gal_and_compute_cross_rhs_cont_eq();
      }
    }
    else if (fldpara_->physical_type() == Inpar::FLUID::artcomp)
    {
      // compute additional Galerkin terms on right-hand side of continuity equation
      // -> different for generalized-alpha and other time-integration schemes
      compute_gal_rhs_cont_eq_art_comp(epreaf, epren, escadtam);

      // add to residual of continuity equation
      conres_old_ -= rhscon_;
    }

    // set velocity-based momentum residual vectors to zero
    lin_resM_Du_.clear();
    resM_Du_.clear();

    // compute first version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part),
    // reaction term and cross-stress term
    lin_gal_mom_res_uost_new(lin_resM_Du_, timefacfac);

    // potentially rescale first version of velocity-based momentum residual
    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
        fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
    {
      lin_gal_mom_res_u_subscales(estif_p_v_, lin_resM_Du_, resM_Du_, timefacfac, facMtau);
    }

    //----------------------------------------------------------------------
    // computation of standard Galerkin and stabilization contributions to
    // element matrix and right-hand-side vector
    //----------------------------------------------------------------------
    // 1) standard Galerkin inertia, convection and reaction terms
    //    (convective and reactive part for convection term)
    //    as well as first part of cross-stress term on left-hand side
    inertia_convection_reaction_gal_part(
        estif_u_, velforce_, lin_resM_Du_, resM_Du_, rhsfac, rhsfacn);

    // 2) standard Galerkin viscous term
    //    (including viscous stress computation,
    //     excluding viscous part for low-Mach-number flow)
    static Core::LinAlg::Matrix<nsd_, nsd_> viscstress(Core::LinAlg::Initialization::zero);
    viscstress.clear();

    viscous_gal_part(estif_u_, velforce_, viscstress, timefacfac, rhsfac, rhsfacn);

    // 3) stabilization of continuity equation,
    //    standard Galerkin viscous part for low-Mach-number flow and
    //    right-hand-side part of standard Galerkin viscous term
    if (fldpara_->c_stab() or fldpara_->physical_type() == Inpar::FLUID::loma)
      cont_stab(estif_u_, velforce_, fldparatimint_->time_fac(), timefacfac, timefacfacpre, rhsfac,
          rhsfacn);

    // 4) standard Galerkin pressure term
    pressure_gal_part(
        estif_p_v_, velforce_, timefacfac, timefacfacpre, rhsfac, rhsfacn, press, pressn);

    // 5) standard Galerkin continuity term
    continuity_gal_part(estif_q_u_, preforce_, timefacfac, timefacfacpre, rhsfac, rhsfacn);

    // 6) standard Galerkin bodyforce term on right-hand side
    body_force_rhs_term(velforce_, rhsfac, rhsfacn);

    // 7) additional standard Galerkin terms due to conservative formulation
    //    New One Step Theta not implemented for this as of yet!
    if (fldpara_->is_conservative())
    {
      conservative_formulation(estif_u_, velforce_, timefacfac, rhsfac);
    }

    // 8) additional standard Galerkin terms for low-Mach-number flow and
    //    artificial compressibility (only right-hand side in latter case)
    //    New One Step Theta not implemented for LOMA as of yet.
    if (fldpara_->physical_type() == Inpar::FLUID::loma or
        fldpara_->physical_type() == Inpar::FLUID::artcomp)
    {
      loma_gal_part(estif_q_u_, preforce_, timefacfac, rhsfac);
    }

    // 9) additional standard Galerkin term for temporal derivative of pressure
    //    in case of artificial compressibility (only left-hand side)
    if (fldpara_->physical_type() == Inpar::FLUID::artcomp and not fldparatimint_->is_stationary())
      art_comp_pressure_inertia_gal_partand_cont_stab(estif_p_v_, ppmat_);

    //----------------------------------------------------------------------
    // compute second version of velocity-based momentum residual containing
    // inertia term, convection term (convective and reactive part) and
    // viscous term
    //----------------------------------------------------------------------
    stab_lin_gal_mom_res_u(lin_resM_Du_, timefacfac);

    // 10) PSPG term
    if (fldpara_->pspg())
    {
      pspgost_new(estif_q_u_, ppmat_, preforce_, lin_resM_Du_, fac3, timefacfac, timefacfacpre,
          rhsfac, *iquad);
    }

    // 11) SUPG term as well as first part of Reynolds-stress term on
    //     left-hand side and Reynolds-stress term on right-hand side
    if (fldpara_->supg())
    {
      supgost_new(estif_u_, estif_p_v_, velforce_, preforce_, lin_resM_Du_, fac3, timefacfac,
          timefacfacpre, rhsfac);
    }

    // 12) reactive stabilization term
    if (fldpara_->r_stab() != Inpar::FLUID::reactive_stab_none)
    {
      reac_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // 13) viscous stabilization term
    if (is_higher_order_ele_ and (fldpara_->v_stab() != Inpar::FLUID::viscous_stab_none))
    {
      visc_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // if ConvDivStab for XFEM
    //    {
    //      ConvDivStab(estif_u,
    //           velforce,
    //           timefacfac,
    //           rhsfac);
    //    }


    // 14) cross-stress term: second part on left-hand side (only for Newton
    //     iteration) as well as cross-stress term on right-hand side
    if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none)
    {
      cross_stress_stab(
          estif_u_, estif_p_v_, velforce_, lin_resM_Du_, timefacfac, timefacfacpre, rhsfac, fac3);
    }

    // 15) Reynolds-stress term: second part on left-hand side
    //     (only for Newton iteration)
    if (fldpara_->reynolds() == Inpar::FLUID::reynolds_stress_stab and fldpara_->is_newton())
    {
      reynolds_stress_stab(estif_u_, estif_p_v_, lin_resM_Du_, timefacfac, timefacfacpre, fac3);
    }

    // 16) fine-scale subgrid-viscosity term
    //     (contribution only to right-hand-side vector)
    if (fldpara_->fssgv() != Inpar::FLUID::no_fssgv)
    {
      const double fssgviscfac = fssgvisc_ * rhsfac;

      fine_scale_sub_grid_viscosity_term(velforce_, fssgviscfac);
    }

    // 17) subgrid-stress term (multifractal subgrid scales)
    if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      multfrac_sub_grid_scales_cross(estif_u_, velforce_, timefacfac, rhsfac);

      multfrac_sub_grid_scales_reynolds(estif_u_, velforce_, timefacfac, rhsfac);
    }

    // 18) polynomial pressure projection term (Dohrmann, Bochev IJNME 2004)
    //     (parameter-free inf-sub-stabilization, e.g. used instead of PSPG)
    if (fldpara_->ppp())
    {
      pressure_projection(ppmat_);
    }

    // linearization wrt mesh motion
    if (emesh.is_initialized())
    {
      if (nsd_ == 3)
        lin_mesh_motion_3d(emesh, evelaf, press, fldparatimint_->time_fac(), timefacfac);
      else if (nsd_ == 2)
        lin_mesh_motion_2d(emesh, evelaf, press, fldparatimint_->time_fac(), timefacfac);
      else
        FOUR_C_THROW("Linearization of the mesh motion is not available in 1D");
    }
  }
  //------------------------------------------------------------------------
  //  end loop over integration points
  //------------------------------------------------------------------------

  // if polynomial pressure projection: finalize matrices and rhs
  if (fldpara_->ppp())
  {
    if (fldparatimint_->is_genalpha_np())
      pressure_projection_finalize(ppmat_, preforce_, eprenp);
    else
      pressure_projection_finalize(ppmat_, preforce_, epreaf);
  }

  //------------------------------------------------------------------------
  //  add contributions to element matrix and right-hand-side vector
  //------------------------------------------------------------------------
  // add pressure part to right-hand-side vector
  for (int vi = 0; vi < nen_; ++vi)
  {
    eforce(numdofpernode_ * vi + nsd_) += preforce_(vi);
  }

  // add velocity part to right-hand-side vector
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      eforce(numdofpernode_ * vi + idim) += velforce_(idim, vi);
    }
  }

  // add pressure-pressure part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int fuippp = numdofpernode_ * ui + nsd_;

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int numdof_vi_p_nsd = numdofpernode_ * vi + nsd_;

      estif(numdof_vi_p_nsd, fuippp) += ppmat_(vi, ui);
    }
  }

  // add velocity-velocity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_ * ui;
    const int nsd_ui = nsd_ * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int numdof_ui_jdim = numdof_ui + jdim;
      const int nsd_ui_jdim = nsd_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
      {
        const int numdof_vi = numdofpernode_ * vi;
        const int nsd_vi = nsd_ * vi;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif(numdof_vi + idim, numdof_ui_jdim) += estif_u_(nsd_vi + idim, nsd_ui_jdim);
        }
      }
    }
  }

  // add velocity-pressure part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui_nsd = numdofpernode_ * ui + nsd_;

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int nsd_vi = nsd_ * vi;
      const int numdof_vi = numdofpernode_ * vi;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif(numdof_vi + idim, numdof_ui_nsd) += estif_p_v_(nsd_vi + idim, ui);
      }
    }
  }

  // add pressure-velocity part to matrix
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int numdof_ui = numdofpernode_ * ui;
    const int nsd_ui = nsd_ * ui;

    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const int numdof_ui_jdim = numdof_ui + jdim;
      const int nsd_ui_jdim = nsd_ui + jdim;

      for (int vi = 0; vi < nen_; ++vi)
        estif(numdofpernode_ * vi + nsd_, numdof_ui_jdim) += estif_q_u_(vi, nsd_ui_jdim);
    }
  }

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_div_eps(
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& eveln)
{
  /*--- viscous term: div(epsilon(u)) --------------------------------*/
  /*   /                                                \
       |  2 N_x,xx + N_x,yy + N_y,xy + N_x,zz + N_z,xz  |
     1 |                                                |
     - |  N_y,xx + N_x,yx + 2 N_y,yy + N_z,yz + N_y,zz  |
     2 |                                                |
       |  N_z,xx + N_x,zx + N_y,zy + N_z,yy + 2 N_z,zz  |
       \                                                /

       with N_x .. x-line of N
       N_y .. y-line of N                                             */

  /*--- subtraction for low-Mach-number flow: div((1/3)*(div u)*I) */
  /*   /                            \
       |  N_x,xx + N_y,yx + N_z,zx  |
     1 |                            |
  -  - |  N_x,xy + N_y,yy + N_z,zy  |
     3 |                            |
       |  N_x,xz + N_y,yz + N_z,zz  |
       \                            /

         with N_x .. x-line of N
         N_y .. y-line of N                                             */

  // set visc_old to zero
  visc_old_.clear();
  visc_oldn_.clear();

  double prefac;
  if (fldpara_->physical_type() == Inpar::FLUID::loma)
  // if(loma_)
  {
    prefac = 1.0 / 3.0;
    derxy2_.scale(prefac);
  }
  else
    prefac = 1.0;
  // reconstruction of second derivative via projection or superconvergent patch recovery
  if (fldpara_->is_reconstruct_der())
  {
    if (is_higher_order_ele_ == false) FOUR_C_THROW("this doesn't make sense");

    // global second derivatives of evalaf (projected velgrad)
    // not symmetric!
    Core::LinAlg::Matrix<nsd_ * nsd_, nsd_> evelgradderxy;
    evelgradderxy.multiply_nt(evelafgrad_, derxy_);
    //*VELNP*
    if (nsd_ == 3)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy + u,zz)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1) +
                          evelgradderxy(nsd_idim + 2, 2)) /
                      prefac;
        // interpolate mixed terms
        double sum2;
        if (idim == 0)
        {
          // uy,xy + uz,xz
          sum2 = 0.5 * (evelgradderxy(3, 1) + evelgradderxy(4, 0) + evelgradderxy(6, 2) +
                           evelgradderxy(8, 0));
        }
        else if (idim == 1)
        {
          // ux,xy + uz,yz
          sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1) + evelgradderxy(7, 2) +
                           evelgradderxy(8, 1));
        }
        else
        {
          // ux,xz + uy,yz
          sum2 = 0.5 * (evelgradderxy(2, 0) + evelgradderxy(0, 2) + evelgradderxy(5, 1) +
                           evelgradderxy(4, 2));
        }
        // assemble each row of div epsilon(evelaf)
        visc_old_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else if (nsd_ == 2)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1)) / prefac;
        // interpolate mixed terms
        double sum2;
        if (idim == 0)
        {
          // uy,xy
          sum2 = 0.5 * (evelgradderxy(2, 1) + evelgradderxy(3, 0));
        }
        else
        {
          // ux,xy
          sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1));
        }
        // assemble each row of div epsilon(evelaf)
        visc_old_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else
      FOUR_C_THROW("Epsilon(N) is not implemented for the 1D case");

    //*VELN*
    // Core::LinAlg::Matrix<nsd_*nsd_,nsd_> evelngradderxy;
    evelgradderxy.clear();
    evelgradderxy.multiply_nt(evelngrad_, derxy_);
    if (nsd_ == 3)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy + u,zz)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1) +
                          evelgradderxy(nsd_idim + 2, 2)) /
                      prefac;
        // interpolate mixed terms
        double sum2;
        if (idim == 0)
        {
          // uy,xy + uz,xz
          sum2 = 0.5 * (evelgradderxy(3, 1) + evelgradderxy(4, 0) + evelgradderxy(6, 2) +
                           evelgradderxy(8, 0));
        }
        else if (idim == 1)
        {
          // ux,xy + uz,yz
          sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1) + evelgradderxy(7, 2) +
                           evelgradderxy(8, 1));
        }
        else
        {
          // ux,xz + uy,yz
          sum2 = 0.5 * (evelgradderxy(2, 0) + evelgradderxy(0, 2) + evelgradderxy(5, 1) +
                           evelgradderxy(4, 2));
        }
        // assemble each row of div epsilon(evelaf)
        visc_oldn_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else if (nsd_ == 2)
    {
      // assemble div epsilon(evelaf)
      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int nsd_idim = idim * nsd_;
        // select diagonal entries (u,xx + u,yy)
        double sum1 = (evelgradderxy(nsd_idim, 0) + evelgradderxy(nsd_idim + 1, 1)) / prefac;
        // interpolate mixed terms
        double sum2;
        if (idim == 0)
        {
          // uy,xy
          sum2 = 0.5 * (evelgradderxy(2, 1) + evelgradderxy(3, 0));
        }
        else
        {
          // ux,xy
          sum2 = 0.5 * (evelgradderxy(1, 0) + evelgradderxy(0, 1));
        }
        // assemble each row of div epsilon(evelaf)
        visc_oldn_(idim) = 0.5 * (sum1 + evelgradderxy(nsd_idim + idim, idim) + sum2);
      }
    }
    else
      FOUR_C_THROW("Epsilon(N) is not implemented for the 1D case");
  }
  else
  {
    if (nsd_ == 3)
    {
      for (int inode = 0; inode < nen_; ++inode)
      {
        double sum = (derxy2_(0, inode) + derxy2_(1, inode) + derxy2_(2, inode)) / prefac;
        viscs2_(0, inode) = 0.5 * (sum + derxy2_(0, inode));
        viscs2_(1, inode) = 0.5 * derxy2_(3, inode);
        viscs2_(2, inode) = 0.5 * derxy2_(4, inode);
        viscs2_(3, inode) = 0.5 * derxy2_(3, inode);
        viscs2_(4, inode) = 0.5 * (sum + derxy2_(1, inode));
        viscs2_(5, inode) = 0.5 * derxy2_(5, inode);
        viscs2_(6, inode) = 0.5 * derxy2_(4, inode);
        viscs2_(7, inode) = 0.5 * derxy2_(5, inode);
        viscs2_(8, inode) = 0.5 * (sum + derxy2_(2, inode));

        for (int idim = 0; idim < nsd_; ++idim)
        {
          const int nsd_idim = idim * nsd_;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            visc_old_(idim) += viscs2_(nsd_idim + jdim, inode) * evelaf(jdim, inode);
            visc_oldn_(idim) += viscs2_(nsd_idim + jdim, inode) * eveln(jdim, inode);
          }
        }
      }
    }
    else if (nsd_ == 2)
    {
      for (int inode = 0; inode < nen_; ++inode)
      {
        double sum = (derxy2_(0, inode) + derxy2_(1, inode)) / prefac;
        viscs2_(0, inode) = 0.5 * (sum + derxy2_(0, inode));
        viscs2_(1, inode) = 0.5 * derxy2_(2, inode);
        viscs2_(2, inode) = 0.5 * derxy2_(2, inode);
        viscs2_(3, inode) = 0.5 * (sum + derxy2_(1, inode));

        for (int idim = 0; idim < nsd_; ++idim)
        {
          const int nsd_idim = idim * nsd_;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            visc_old_(idim) += viscs2_(nsd_idim + jdim, inode) * evelaf(jdim, inode);
            visc_oldn_(idim) += viscs2_(nsd_idim + jdim, inode) * eveln(jdim, inode);
          }
        }
      }
    }
    else
      FOUR_C_THROW("Epsilon(N) is not implemented for the 1D case");
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::compute_subgrid_scale_velocity_ost_new(
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, double& fac1, double& fac2, double& fac3,
    double& facMtau, int iquad, double* saccn, double* sveln, double* svelnp)
{
  //----------------------------------------------------------------------
  // compute residual of momentum equation
  // -> different for generalized-alpha and other time-integration schemes
  //----------------------------------------------------------------------
  if (fldparatimint_->is_genalpha())
  {
    if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
      FOUR_C_THROW(
          "The combination of generalized-alpha time integration and a Boussinesq approximation "
          "has not been implemented yet!");

    // rhs of momentum equation: density*bodyforce at n+alpha_F
    rhsmom_.update(densaf_, bodyforce_, 0.0);
    // and pressure gradient prescribed as body force
    // caution: not density weighted
    rhsmom_.update(1.0, generalbodyforce_, 1.0);

    // get acceleration at time n+alpha_M at integration point
    accint_.multiply(eaccam, funct_);

    // evaluate momentum residual once for all stabilization right hand sides
    for (int rr = 0; rr < nsd_; ++rr)
    {
      momres_old_(rr) = densam_ * accint_(rr) + densaf_ * conv_old_(rr) + gradp_(rr) -
                        2 * visceff_ * visc_old_(rr) + reacoeff_ * velint_(rr) -
                        densaf_ * bodyforce_(rr) - generalbodyforce_(rr);
    }

    // add consistency terms for MFS if applicable
    multfrac_sub_grid_scales_consistent_residual();
  }
  else
  {
    if (not fldparatimint_->is_stationary())
    {
      // rhs of instationary momentum equation:
      // density*theta*bodyforce at n+1 + density*(histmom/dt)
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g
      if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
      {
        //      Made old OST impl equivalent to gen-alpha (alpha_f=alpha_m=1) (multiplied with
        //      \rho_(n+1))
        rhsmom_.update((densaf_ / fldparatimint_->dt() / fldparatimint_->theta()), histmom_,
            deltadens_, bodyforce_);
        // and pressure gradient prescribed as body force
        // caution: not density weighted
        rhsmom_.update(1.0, generalbodyforce_, 1.0);
      }
      else
      {
        //      Made old OST impl equivalent to gen-alpha (alpha_f=alpha_m=1) (multiplied with
        //      \rho_(n+1))
        rhsmom_.update((densaf_ / fldparatimint_->dt() / fldparatimint_->theta()), histmom_,
            densaf_, bodyforce_);

        // and pressure gradient prescribed as body force
        // caution: not density weighted
        rhsmom_.update(1.0, generalbodyforce_, 1.0);
      }

      // compute instationary momentum residual:
      // momres_old = u_(n+1)/dt + theta ( ... ) +(1-theta) ( ... ) - theta*bodyforce_
      if (fldparatimint_->is_new_ost_implementation())
      {
        const double quotfac =
            (1.0 - fldparatimint_->theta()) * fldparatimint_->dt() / fldparatimint_->time_fac_rhs();
        for (int rr = 0; rr < nsd_; ++rr)
        {
          momres_old_(rr) = dens_theta_ * (velint_(rr) - velintn_(rr)) /
                            (fldparatimint_->dt() * fldparatimint_->theta());
          momres_old_(rr) -= (densaf_ * bodyforce_(rr) + densn_ * quotfac * bodyforcen_(rr));
          momres_old_(rr) +=
              reacoeff_ * (velint_(rr) + quotfac * velintn_(rr));  // TODO: Time dependant reacoef.
          momres_old_(rr) += (densaf_ * conv_old_(rr) + densn_ * quotfac * conv_oldn_(rr));
          momres_old_(rr) -= 2.0 * (visceff_ * visc_old_(rr) + viscn_ * quotfac * visc_oldn_(rr));
          momres_old_(rr) -= generalbodyforce_(rr) + quotfac * generalbodyforcen_(rr);
          if (not fldparatimint_->is_full_impl_pressure_and_cont())
          {
            momres_old_(rr) += gradp_(rr) + quotfac * gradpn_(rr);
          }
          else
          {
            momres_old_(rr) +=
                gradp_(rr) / fldparatimint_->theta();  // Gradient of p with no pre-factor.
          }

          // Left for implementation in ALE, possibly....
          //        //Furthermore, should rhsmom_ be calculated like this? Not with galerkin terms
          //        and integration???! rhsmom_(rr)=- fldparatimint_->Dt() *fldparatimint_->Theta()*
          //        (- densaf_*velintn_(rr)/(fldparatimint_->Dt()*fldparatimint_->Theta())
          //                                          - densn_*quotfac*bodyforcen_(rr)
          //                                          + reacoeff_*quotfac*velintn_(rr) +
          //                                          densaf_*quotfac*conv_oldn_(rr)
          //                                          - 2*viscn_*quotfac*visc_oldn_(rr)  +
          //                                          quotfac*generalbodyforcen_(rr)
          //                                          + quotfac*gradpn_(rr)); //Check what gradpn_
          //                                          to use! (This is needed for ALE
          //                                          implementations?)
        }
      }
      else
      {
        // momres_old = u_(n+1)/dt + theta ( ... ) - histmom_/dt - theta*bodyforce_
        for (int rr = 0; rr < nsd_; ++rr)
        {
          momres_old_(rr) = ((densaf_ * velint_(rr) / fldparatimint_->dt() +
                                 fldparatimint_->theta() *
                                     (densaf_ * conv_old_(rr) + gradp_(rr) -
                                         2 * visceff_ * visc_old_(rr) + reacoeff_ * velint_(rr))) /
                                fldparatimint_->theta()) -
                            rhsmom_(rr);
        }
      }  // end is_new_ost_implementation
    }
    else
    {
      // rhs of stationary momentum equation: density*bodyforce
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g and pressure gradient
      // prescribed as body force (not density weighted)
      if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
        rhsmom_.update(deltadens_, bodyforce_, 1.0, generalbodyforce_);
      else
        rhsmom_.update(densaf_, bodyforce_, 1.0, generalbodyforce_);

      // compute stationary momentum residual:
      for (int rr = 0; rr < nsd_; ++rr)
      {
        momres_old_(rr) = -rhsmom_(rr);
        momres_old_(rr) += gradp_(rr);
        momres_old_(rr) += reacoeff_ * velint_(rr);
        momres_old_(rr) += densaf_ * conv_old_(rr);
        momres_old_(rr) += -2 * visceff_ * visc_old_(rr);
      }

      // add consistency terms for MFS if applicable
      multfrac_sub_grid_scales_consistent_residual();
    }
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale velocity
  //----------------------------------------------------------------------
  // 1) quasi-static subgrid scales
  // Definition of subgrid-scale velocity is not consistent for the SUPG term and Franca, Valentin,
  // ... Definition of subgrid velocity used by Hughes
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    sgvelint_.update(-tau_(1), momres_old_, 0.0);
  }
  // 2) time-dependent subgrid scales
  else
  {
    // some checking
    if (fldparatimint_->is_stationary())
      FOUR_C_THROW("there is no time dependent subgrid scale closure for stationary problems\n");
    if (saccn == nullptr or sveln == nullptr or svelnp == nullptr)
      FOUR_C_THROW("no subscale array provided");

    // parameter definitions
    double alphaF = fldparatimint_->alpha_f();
    double alphaM = fldparatimint_->alpha_m();
    double gamma = fldparatimint_->gamma();
    double dt = fldparatimint_->dt();

    /*
                                            1.0
       facMtau =  -------------------------------------------------------
                     n+aM                      n+aF
                  rho     * alphaM * tauM + rho     * alphaF * gamma * dt
    */
    facMtau = 1.0 / (densam_ * alphaM * tau_(1) + densaf_ * fldparatimint_->afgdt());

    /*
       factor for old subgrid velocities:

                 n+aM                      n+aF
       fac1 = rho     * alphaM * tauM + rho     * gamma * dt * (alphaF-1)
    */
    fac1 = (densam_ * alphaM * tau_(1) + densaf_ * gamma * dt * (alphaF - 1.0)) * facMtau;
    /*
      factor for old subgrid accelerations

                 n+aM
       fac2 = rho     * tauM * dt * (alphaM-gamma)
    */
    fac2 = (densam_ * dt * tau_(1) * (alphaM - gamma)) * facMtau;
    /*
      factor for residual in current subgrid velocities:

       fac3 = gamma * dt * tauM
    */
    fac3 = (gamma * dt * tau_(1)) * facMtau;

    // warning: time-dependent subgrid closure requires generalized-alpha time
    // integration
    if (!fldparatimint_->is_genalpha())
    {
      FOUR_C_THROW("the time-dependent subgrid closure requires a genalpha time integration\n");
    }

    /*         +-                                       -+
        ~n+1   |        ~n           ~ n            n+1  |
        u    = | fac1 * u  + fac2 * acc  -fac3 * res     |
         (i)   |                                    (i)  |
               +-                                       -+
    */

    /* compute the intermediate value of subscale velocity

            ~n+af            ~n+1                   ~n
            u     = alphaF * u     + (1.0-alphaF) * u
             (i)              (i)

    */

    static Core::LinAlg::Matrix<1, nsd_> sgvelintaf(Core::LinAlg::Initialization::zero);
    sgvelintaf.clear();
    for (int rr = 0; rr < nsd_; ++rr)
    {
      tds_->update_svelnp_in_one_direction(fac1, fac2, fac3, momres_old_(rr),
          fldparatimint_->alpha_f(), rr, iquad,
          sgvelint_(rr),  // sgvelint_ is set to sgvelintnp, but is then overwritten below anyway!
          sgvelintaf(rr));

      int pos = rr + nsd_ * iquad;

      /*
       *  ~n+1           ~n           ~ n            n+1
       *  u    =  fac1 * u  + fac2 * acc  -fac3 * res
       *   (i)
       *
       */

      svelnp[pos] = fac1 * sveln[pos] + fac2 * saccn[pos] - fac3 * momres_old_(rr);

      /* compute the intermediate value of subscale velocity
       *
       *          ~n+af            ~n+1                   ~n
       *          u     = alphaF * u     + (1.0-alphaF) * u
       *           (i)              (i)
       *
       */
      sgvelint_(rr) = alphaF * svelnp[pos] + (1.0 - alphaF) * sveln[pos];
    }
  }  // end time dependent subgrid scale closure

  //----------------------------------------------------------------------
  // include computed subgrid-scale velocity in convective term
  // -> only required for cross- and Reynolds-stress terms
  //----------------------------------------------------------------------
  if (fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none or
      fldpara_->reynolds() != Inpar::FLUID::reynolds_stress_stab_none or
      fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none or
      fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    sgconv_c_.multiply_tn(derxy_, sgvelint_);
  else
    sgconv_c_.clear();
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::lin_gal_mom_res_uost_new(
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac)
{
  /*
      instationary                          cross-stress, part 1
       +-----+                             +-------------------+
       |     |                             |                   |

                 /       n+1       \        /      ~n+1       \
       rho*Du + |   rho*u   o nabla | Du + |   rho*u   o nabla | Du +
                 \      (i)        /        \      (i)        /

                 /                \  n+1
              + |   rho*Du o nabla | u      +  sigma*Du
                 \                /   (i)
                |                        |     |       |
                +------------------------+     +-------+
                        Newton                  reaction
  */

  int idim_nsd_p_idim[nsd_];

  for (int idim = 0; idim < nsd_; ++idim)
  {
    idim_nsd_p_idim[idim] = idim * nsd_ + idim;
  }

  if (fldparatimint_->is_stationary() == false)
  {
    //    double fac_densam= (fldparatimint_->IsOneStepTheta()) ? fac_*dens_theta_ : fac_*densam_;
    double fac_densam;
    if (fldparatimint_->is_new_ost_implementation())
      fac_densam = fac_ * dens_theta_;
    else
      fac_densam = fac_ * densam_;
    // End of is_new_ost_implementation()

    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v = fac_densam * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  const double timefacfac_densaf = timefacfac * densaf_;

  // convection, reactive
  for (int ui = 0; ui < nen_; ++ui)
  {
    const double v = timefacfac_densaf * conv_c_(ui);

    for (int idim = 0; idim < nsd_; ++idim)
    {
      lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
    }
  }

  // convection, convective (only for Newton)
  if (fldpara_->is_newton())
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      const double temp = timefacfac_densaf * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        const int idim_nsd = idim * nsd_;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          lin_resM_Du(idim_nsd + jdim, ui) += temp * vderxy_(idim, jdim);
        }
      }
    }
  }

  if (fldpara_->reaction())
  {
    const double fac_reac = timefacfac * reacoeff_;

    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v = fac_reac * funct_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  if (fldpara_->cross() == Inpar::FLUID::cross_stress_stab)
  {
    // const double rhsresfac_densaf=rhsresfac*densaf_;
    for (int ui = 0; ui < nen_; ++ui)
    {
      // const double v=rhsresfac_densaf*sgconv_c_(ui);
      const double v = timefacfac_densaf * sgconv_c_(ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += v;
      }
    }
  }

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::inertia_convection_reaction_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, Core::LinAlg::Matrix<nsd_, 1>& resM_Du,
    const double& rhsfac, const double& rhsfacn)
{
  /* inertia (contribution to mass matrix) if not is_stationary */
  /*
            /              \
           |                |
           |    rho*Du , v  |
           |                |
            \              /
  */
  /* convection, convective part (convective form) */
  /*
            /                             \
           |  /       n+1       \          |
           | |   rho*u   o nabla | Du , v  |
           |  \      (i)        /          |
            \                             /
  */
  /*  convection, reactive part (convective form)
            /                               \
           |  /                \   n+1       |
           | |  rho*Du o nabla  | u     , v  |
           |  \                /   (i)       |
            \                               /
  */
  /*  reaction */
  /*
            /                \
           |                  |
           |    sigma*Du , v  |
           |                  |
            \                /
  */
  if ((fldpara_->is_newton() or
          (is_higher_order_ele_ and fldpara_->tds() == Inpar::FLUID::subscales_time_dependent)))
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          for (int idim = 0; idim < nsd_; ++idim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                funct_(vi) * lin_resM_Du(idim * nsd_ + jdim, ui);
          }  // end for (idim)
        }  // end for (jdim)
      }  // end for (vi)
    }  // end for (ui)
  }
  else
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_ * vi + idim, nsd_ * ui + idim) +=
              funct_(vi) * lin_resM_Du(idim * nsd_ + idim, ui);
        }  // end for (idim)
      }  // vi
    }  // ui
  }

  // inertia terms on the right hand side for instationary fluids
  if (not fldparatimint_->is_stationary())
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      if (fldparatimint_->is_genalpha())
      {
        if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
            fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
        {
          ;  // do nothing here! Whole term already set in lin_gal_mom_res_u_subscales()
        }
        else
          resM_Du(idim) += rhsfac * densam_ * accint_(idim);
      }
      else
      {
        if (fldparatimint_->is_new_ost_implementation())
        {
          // this approximates \int_{t_n}^{t_{n+1}} \rho du/dt
          // It could be implemented differently like (through integration by parts):
          // \rho_{n+1}u_{n+1}-\rho_{n}u_{n} -  \int_{t_n}^{t_{n+1}} d \rho/dt u
          // But leaves the second integral to be approximated.
          resM_Du(idim) += fac_ * dens_theta_ * (velint_(idim) - velintn_(idim));
        }
        else
        {
          resM_Du(idim) += fac_ * densaf_ * velint_(idim);
        }  // end is_new_ost_implementation
      }
    }
  }  // end if (not stationary)

  // convective terms of rhs
  for (int idim = 0; idim < nsd_; ++idim)
  {
    if (fldpara_->tds() == Inpar::FLUID::subscales_time_dependent &&
        fldpara_->transient() == Inpar::FLUID::inertia_stab_keep)
    {
      ;  // do nothing here! Whole term already set in lin_gal_mom_res_u_subscales()
    }
    else
    {
      resM_Du(idim) += rhsfac * densaf_ * conv_old_(idim);
      if (fldparatimint_->is_new_ost_implementation())
      {
        if (fldparatimint_->is_one_step_theta())
        {
          resM_Du(idim) += rhsfacn * densn_ * conv_oldn_(idim);
        }
      }  // end is_new_ost_implementation
    }
  }  // end for(idim)

  if (fldpara_->reaction())
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      resM_Du(idim) += rhsfac * reacoeff_ * velint_(idim);
      if (fldparatimint_->is_new_ost_implementation())
      {
        if (fldparatimint_->is_one_step_theta())
        {
          resM_Du(idim) += rhsfacn * reacoeff_ * velintn_(idim);
        }
      }  // end is_new_ost_implementation
    }
  }  // end if (reaction_)

  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      velforce(idim, vi) -= resM_Du(idim) * funct_(vi);
    }
  }
  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::viscous_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, Core::LinAlg::Matrix<nsd_, nsd_>& viscstress,
    const double& timefacfac, const double& rhsfac, const double& rhsfacn)
{
  const double visceff_timefacfac = visceff_ * timefacfac;

  /* viscosity term */
  /*
                   /                        \
                  |       /  \         / \   |
            2 mu  |  eps | Du | , eps | v |  |
                  |       \  /         \ /   |
                   \                        /
  */

  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      const double temp = visceff_timefacfac * derxy_(jdim, vi);

      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) += temp * derxy_(idim, ui);
        }
      }
    }
  }

  static Core::LinAlg::Matrix<nen_, nen_> tmp_dyad;
  tmp_dyad.multiply_tn(derxy_, derxy_);
  tmp_dyad.scale(visceff_timefacfac);

  for (int ui = 0; ui < nen_; ++ui)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      const double tmp_val = tmp_dyad(vi, ui);

      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_u(nsd_ * vi + idim, nsd_ * ui + idim) += tmp_val;
      }  // end for (idim)
    }  // ui
  }  // vi

  static Core::LinAlg::Matrix<nsd_, nsd_> viscstressn;

  const double v = visceff_ * rhsfac;
  const double vn = viscn_ * rhsfacn;

  for (int jdim = 0; jdim < nsd_; ++jdim)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      viscstress(idim, jdim) = v * (vderxy_(jdim, idim) + vderxy_(idim, jdim));
      viscstressn(idim, jdim) = vn * (vderxyn_(jdim, idim) + vderxyn_(idim, jdim));
    }
  }


  static Core::LinAlg::Matrix<nsd_, nen_> tmp;


  if (fldparatimint_->is_new_ost_implementation())
  {
    static Core::LinAlg::Matrix<nsd_, nsd_> viscstress_added;

    viscstress_added.update(1.0, viscstress, 1.0, viscstressn, 0.0);
    tmp.multiply(viscstress_added, derxy_);
  }
  else
    tmp.multiply(viscstress, derxy_);

  velforce.update(-1.0, tmp, 1.0);

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::cont_stab(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefac, const double& timefacfac,
    const double& timefacfacpre, const double& rhsfac, const double& rhsfacn)
{
  // In the case no continuity stabilization and no LOMA:
  // the factors 'conti_stab_and_vol_visc_fac' and 'conti_stab_and_vol_visc_rhs' are zero
  // therefore there is no contribution to the element stiffness matrix and
  // the viscous stress tensor is NOT altered!!
  //
  // ONLY
  // the rhs contribution of the viscous term is added!!

  double conti_stab_and_vol_visc_fac = 0.0;
  double conti_stab_and_vol_visc_rhs = 0.0;

  if (fldpara_->c_stab())
  {
    if (not fldparatimint_->is_full_impl_pressure_and_cont())
    {
      conti_stab_and_vol_visc_fac += timefacfacpre * tau_(2);
      conti_stab_and_vol_visc_rhs -= rhsfac * tau_(2) * conres_old_;
      if (fldparatimint_->is_new_ost_implementation())
      {
        if (not fldparatimint_->is_impl_pressure())
        {
          conti_stab_and_vol_visc_rhs -= rhsfacn * tau_(2) * conres_oldn_;
        }
      }  // end is_new_ost_implementation
    }
    else
    {
      // Full impl pressure weighted with \Delta t only.
      conti_stab_and_vol_visc_fac += fac_ * fldparatimint_->dt() * tau_(2);
      conti_stab_and_vol_visc_rhs -= fac_ * fldparatimint_->dt() * tau_(2) * conres_old_;
    }
  }
  if (fldpara_->physical_type() == Inpar::FLUID::loma)
  {
    conti_stab_and_vol_visc_fac -= (2.0 / 3.0) * visceff_ * timefacfac;
    conti_stab_and_vol_visc_rhs += (2.0 / 3.0) * rhsfac * visceff_ * vdiv_;
    // additional term q_sq_ for dynamics Smagorisnky only
    if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
      conti_stab_and_vol_visc_rhs += (2.0 / 3.0) * rhsfac * q_sq_;
  }

  /* continuity stabilisation on left-hand side */
  /*
              /                        \
             |                          |
        tauC | nabla o Du  , nabla o v  |
             |                          |
              \                        /
  */
  /* viscosity term - subtraction for low-Mach-number flow */
  /*
             /                             \             /                        \
            |  1                      / \   |     2 mu  |                          |
     - 2 mu |  - (nabla o u) I , eps | v |  | = - ----- | nabla o Du  , nabla o v  |
            |  3                      \ /   |       3   |                          |
             \                             /             \                        /
  */
  for (int ui = 0; ui < nen_; ++ui)
  {
    const int fui = nsd_ * ui;

    for (int idim = 0; idim < nsd_; ++idim)
    {
      const int fui_p_idim = fui + idim;
      const double v0 = conti_stab_and_vol_visc_fac * derxy_(idim, ui);
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = nsd_ * vi;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          estif_u(fvi + jdim, fui_p_idim) += v0 * derxy_(jdim, vi);
        }
      }
    }  // end for(idim)
  }

  // computation of right-hand-side viscosity term
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      /* viscosity term on right-hand side */
      velforce(idim, vi) += conti_stab_and_vol_visc_rhs * derxy_(idim, vi);
    }
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pressure_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac,
    const double& rhsfacn, const double& press, const double& pressn)
{
  for (int ui = 0; ui < nen_; ++ui)
  {
    double v;
    if (not fldparatimint_->is_full_impl_pressure_and_cont())
    {
      v = -timefacfacpre * funct_(ui);
    }
    else
    {
      v = -fldparatimint_->dt() * fac_ * funct_(ui);
    }

    for (int vi = 0; vi < nen_; ++vi)
    {
      const int fvi = nsd_ * vi;
      /* pressure term */
      /*
           /                \
          |                  |
          |  Dp , nabla o v  |
          |                  |
           \                /
      */
      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_p_v(fvi + idim, ui) += v * derxy_(idim, vi);
      }
    }
  }

  // pressure term on right-hand side
  if (not fldparatimint_->is_full_impl_pressure_and_cont())
  {
    velforce.update(press * rhsfac, derxy_, 1.0);
    if (fldparatimint_->is_new_ost_implementation())
    {
      velforce.update(pressn * rhsfacn, derxy_, 1.0);
    }  // end is_new_ost_implementation
  }
  else
  {
    //     Full impl pressure weighted with \Delta t.
    velforce.update(fac_ * fldparatimint_->dt() * press, derxy_, 1.0);
  }
  return;
}



template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::continuity_gal_part(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, 1>& preforce,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac,
    const double& rhsfacn)
{
  for (int vi = 0; vi < nen_; ++vi)
  {
    double v;
    if (not fldparatimint_->is_full_impl_pressure_and_cont())
    {
      v = timefacfacpre * funct_(vi);
    }
    else
    {
      v = fac_ * fldparatimint_->dt() * funct_(vi);
    }

    for (int ui = 0; ui < nen_; ++ui)
    {
      const int fui = nsd_ * ui;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        /* continuity term */
        /*
             /                \
            |                  |
            | nabla o Du  , q  |
            |                  |
             \                /
        */
        estif_q_u(vi, fui + idim) += v * derxy_(idim, ui);
      }
    }
  }  // end for(idim)
  if (not fldparatimint_->is_full_impl_pressure_and_cont())
  {
    preforce.update(-rhsfac * vdiv_, funct_, 1.0);
    if (fldparatimint_->is_new_ost_implementation())
    {
      if (not fldparatimint_->is_impl_pressure())
      {
        preforce.update(-rhsfacn * vdivn_, funct_, 1.0);
      }
    }  // end is_new_ost_implementation
  }
  else
  {
    //     Full impl pressure weighted with \Delta t.
    preforce.update(-fac_ * fldparatimint_->dt() * vdiv_, funct_, 1.0);
  }
  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::body_force_rhs_term(
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& rhsfac, const double rhsfacn)
{
  for (int idim = 0; idim < nsd_; ++idim)
  {
    double scaled_rhsmom;
    if (fldparatimint_->is_genalpha())
      scaled_rhsmom = rhsfac * rhsmom_(idim);
    else
    {
      if (fldparatimint_->is_new_ost_implementation())
      {
        scaled_rhsmom = rhsfac * (densaf_ * bodyforce_(idim) + generalbodyforce_(idim));
        if (fldparatimint_->is_one_step_theta())
        {
          scaled_rhsmom += rhsfacn * (densn_ * bodyforcen_(idim) + generalbodyforcen_(idim));
        }
      }
      else
      {
        scaled_rhsmom = rhsfac * rhsmom_(idim);
      }  // end is_new_ost_implementation
    }

    for (int vi = 0; vi < nen_; ++vi)
    {
      velforce(idim, vi) += scaled_rhsmom * funct_(vi);
    }
  }  // end for(idim)

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::pspgost_new(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, nen_>& ppmat,
    Core::LinAlg::Matrix<nen_, 1>& preforce, Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const double& fac3, const double& timefacfac, const double& timefacfacpre, const double& rhsfac,
    const int iquad)
{
  // conservative, stabilization terms are neglected (Hughes)

  /* pressure stabilisation:                                            */
  /*
              /                 \
             |  ~n+af            |
           - |  u     , nabla q  |
             |                   |
              \                 /
  */

  double scal_grad_q = 0.0;

  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    scal_grad_q = tau_(1);
  }
  else  // time-dependent subgrid-scales
  {
    scal_grad_q = fac3;
  }

  /* pressure stabilisation: inertia if not stationary*/
  /*
            /                  \
           |                    |
           |  rho*Du , nabla q  |
           |                    |
            \                  /
  */
  /* pressure stabilisation: convection, convective part */
  /*
            /                                   \
           |  /       n+1       \                |
           | |   rho*u   o nabla | Du , nabla q  |
           |  \      (i)        /                |
            \                                   /
  */
  /* pressure stabilisation: convection, reactive part if Newton */
  /*
            /                                   \
           |  /                \   n+1           |
           | |   rho*Du o nabla | u     , grad q |
           |  \                /   (i)           |
            \                                   /
  */
  /* pressure stabilisation: reaction if included */
  /*
            /                     \
           |                      |
           |  sigma*Du , nabla q  |
           |                      |
            \                    /
  */
  /* pressure stabilisation: viscosity (-L_visc_u) */
  /*
            /                              \
           |               /  \             |
       mu  |  nabla o eps | Du | , nabla q  |
           |               \  /             |
            \                              /
  */

  if (is_higher_order_ele_ || fldpara_->is_newton())
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          const double temp_vi_idim = derxy_(idim, vi) * scal_grad_q;
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_q_u(vi, nsd_ * ui + jdim) += lin_resM_Du(nsd_ * idim + jdim, ui) * temp_vi_idim;
          }  // jdim
        }  // idim
      }  // vi
    }  // ui
  }  // end if (is_higher_order_ele_) or (newton_)
  else
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_q_u(vi, nsd_ * ui + idim) +=
              lin_resM_Du(nsd_ * idim + idim, ui) * derxy_(idim, vi) * scal_grad_q;
        }  // vi
      }  // ui
    }  // idim
  }  // end if not (is_higher_order_ele_) nor (newton_)


  for (int ui = 0; ui < nen_; ++ui)
  {
    /* pressure stabilisation: pressure( L_pres_p) */
    /*
         /                    \
        |                      |
        |  nabla Dp , nabla q  |
        |                      |
         \                    /
    */
    for (int vi = 0; vi < nen_; ++vi)
    {
      double sum = 0.;
      for (int idim = 0; idim < nsd_; ++idim) sum += derxy_(idim, ui) * derxy_(idim, vi);

      if (not fldparatimint_->is_full_impl_pressure_and_cont())
        ppmat(vi, ui) += timefacfacpre * scal_grad_q * sum;
      else
      {
        // Weighted with \Delta t
        ppmat(vi, ui) += fac_ * fldparatimint_->dt() * scal_grad_q * sum;
      }
    }  // vi
  }  // ui

  for (int idim = 0; idim < nsd_; ++idim)
  {
    double sgvel = 0.0;
    if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    {
      sgvel = sgvelint_(idim);
    }
    else  // time-dependent subgrid-scales, Np_Genal_Alpha!
    {
      sgvel = (tds_->svelnp())(idim, iquad);
    }
    const double temp = rhsfac * sgvel;

    for (int vi = 0; vi < nen_; ++vi)
    {
      // pressure stabilisation
      preforce(vi) += temp * derxy_(idim, vi);
    }
  }  // end for(idim)

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::supgost_new(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nen_ * nsd_, nen_>& estif_p_v, Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nen_, 1>& preforce, Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du,
    const double& fac3, const double& timefacfac, const double& timefacfacpre, const double& rhsfac)
{
  /*
                    /                                \
                   |  ~n+af    /     n+af       \     |
                 - |  u     , | rho*u    o nabla | v  |
                   |           \     (i)        /     |
                    \                                /
   */

  static Core::LinAlg::Matrix<nsd_, 1> temp;

  double supgfac;
  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    supgfac = densaf_ * tau_(0);
  else
    supgfac = densaf_ * fldparatimint_->alpha_f() * fac3;

  static Core::LinAlg::Matrix<nen_, 1> supg_test;
  for (int vi = 0; vi < nen_; ++vi)
  {
    supg_test(vi) = supgfac * conv_c_(vi);
  }

  if (fldpara_->reynolds() == Inpar::FLUID::reynolds_stress_stab)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      supg_test(vi) += supgfac * sgconv_c_(vi);
    }
  }

  /* supg stabilisation: inertia if not stationary */
  /*
         /                                \
        |            /     n+1       \     |
        |  rho*Du , | rho*u   o nabla | v  |
        |            \     (i)       /     |
         \                                /
  */
  /* supg stabilisation: convective part ( L_conv_u) , convective term */
  /*
         /                                                     \
        |    /       n+1        \        /      n+1       \     |
        |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
        |    \       (i)        /        \      (i)       /     |
         \                                                     /
  */
  /* supg stabilisation: convective part ( L_conv_u) , reactive term if Newton */
  /*
         /                                                     \
        |    /       n+1        \        /     n+1        \     |
        |   |   rho*u    o nabla | Du , | rho*u    o nabla | v  |
        |    \       (i)        /        \     (i)        /     |
         \                                                     /
  */
  /* supg stabilisation: reaction if included */
  /*
         /                                  \
        |              /     n+1       \     |
        |  sigma*Du , | rho*u   o nabla | v  |
        |              \     (i)       /     |
         \                                  /
  */
  /* supg stabilisation: viscous part  (-L_visc_u) if is_higher_order_ele_ */
  /*
         /                                              \
        |               /  \    /       n+1        \     |
        |  nabla o eps | Du |, |   rho*u    o nabla | v  |
        |               \  /    \       (i)        /     |
         \                                              /
  */

  /* supg stabilisation: inertia, linearisation of testfunction if Newton */
  /*
              /                                       \
             |         n+1       /              \      |
             |    rho*u      ,  | rho*Du o nabla | v   |
             |         (i)       \              /      |
              \                                       /
  */
  /* supg stabilisation: reactive part of convection and linearisation of testfunction ( L_conv_u)
   * if Newton */
  /*
              /                                                       \
             |    /       n+1        \   n+1     /              \      |
             |   |   rho*u    o nabla | u    ,  | rho*Du o nabla | v   |
             |    \       (i)        /   (i)     \              /      |
              \                                                       /
  */
  /* supg stabilisation: reaction, linearisation of testfunction if Newton */
  /*
              /                                         \
             |           n+1       /              \      |
             |    sigma*u      ,  | rho*Du o nabla | v   |
             |           (i)       \              /      |
              \                                         /
  */
  /* supg stabilisation: pressure part, linearisation of test function if Newton ( L_pres_p) */
  /*
             /                                     \
            |         n+1    /                \     |
            |  nabla p    , |   rho*Du o nabla | v  |
            |         (i)    \                /     |
             \                                     /
  */
  /* supg stabilisation: viscous part, linearisation of test function if Newton (-L_visc_u) */
  /*
             /                                               \
            |               / n+1 \    /               \      |
            |  nabla o eps | u     |, |  rho*Du o nabla | v   |
            |               \ (i) /    \               /      |
             \                                               /
  */
  /* supg stabilisation: bodyforce part, linearisation of test function if Newton */
  /*
             /                                      \
            |                  /               \     |
            |  rho*rhsint   , |  rho*Du o nabla | v  |
            |                  \               /     |
             \                                      /
  */
  if (fldpara_->is_newton())
  {
    if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
    {
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        temp(jdim) = timefacfac * supgfac * momres_old_(jdim);
      }
    }
    else
    {
      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        temp(jdim) = -timefacfac * densaf_ * sgvelint_(jdim);
      }
    }
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          const double w = temp(idim) * funct_(ui);
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                lin_resM_Du(nsd_ * idim + jdim, ui) * supg_test(vi) + derxy_(jdim, vi) * w;
          }  // jdim
        }  // vi
      }  // ui
    }  // idim
  }  // end if (fldpara_->IsNewton())
  else if (is_higher_order_ele_)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            estif_u(nsd_ * vi + idim, nsd_ * ui + jdim) +=
                lin_resM_Du(nsd_ * idim + jdim, ui) * supg_test(vi);
          }
        }
      }
    }
  }  // end if (is_higher_order_ele_)
  else
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_ * vi + idim, nsd_ * ui + idim) +=
              lin_resM_Du(nsd_ * idim + idim, ui) * supg_test(vi);
        }  // ui
      }  // idim
    }  // vi
  }  // end if not (is_higher_order_ele_) nor (newton_)

  /* supg stabilisation: pressure part  ( L_pres_p) */
  /*
           /                                    \
          |              /       n+1       \     |
          |  nabla Dp , |   rho*u   o nabla | v  |
          |              \       (i)       /     |
           \                                    /
  */
  for (int vi = 0; vi < nen_; ++vi)
  {
    //       const double v = timefacfacpre*supg_test(vi);
    double v;
    if (not fldparatimint_->is_full_impl_pressure_and_cont())
    {
      v = timefacfacpre * supg_test(vi);
    }
    else
    {
      v = fldparatimint_->dt() * fac_ * supg_test(vi);
    }

    for (int ui = 0; ui < nen_; ++ui)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        estif_p_v(nsd_ * vi + idim, ui) += v * derxy_(idim, ui);
      }
    }
  }  // end for(idim)

  if (fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      temp(jdim) = rhsfac * momres_old_(jdim);
    }
  }
  else
  {
    for (int jdim = 0; jdim < nsd_; ++jdim)
    {
      temp(jdim) = -rhsfac * densaf_ * sgvelint_(jdim) / (fac3 * fldparatimint_->alpha_f());
    }
  }

  for (int idim = 0; idim < nsd_; ++idim)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      // supg stabilisation
      velforce(idim, vi) -= temp(idim) * supg_test(vi);
    }
  }  // end for(idim)

  // SUPG and Reynolds-stress term on right-hand side of
  // continuity equation for low-Mach-number flow
  if (fldpara_->physical_type() == Inpar::FLUID::loma and
      (fldpara_->conti_supg() or
          fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none))
  {
    const double temp_supg = rhsfac * scaconvfacaf_ * sgscaint_;

    if (fldpara_->conti_supg())
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        preforce(vi) -= temp_supg * conv_c_(vi);
      }
    }

    if (fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        preforce(vi) -= temp_supg * sgconv_c_(vi);
      }
    }
  }  // loma

  return;
}

/*---------------------------------------------------------------------------*
 | get ALE grid displacements and grid velocity for element     schott 11/14 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_grid_disp_vel_aleost_new(
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::Matrix<nsd_, nen_>& edispnp, Core::LinAlg::Matrix<nsd_, nen_>& egridvnp,
    Core::LinAlg::Matrix<nsd_, nen_>& egridvn)
{
  switch (fldpara_->physical_type())
  {
    case Inpar::FLUID::oseen:
    case Inpar::FLUID::stokes:
    {
      FOUR_C_THROW(
          "ALE with Oseen or Stokes seems to be a tricky combination. Think deep before removing "
          "FOUR_C_THROW!");
      break;
    }
    default:
    {
      get_grid_disp_ale(discretization, lm, edispnp);
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, &egridvnp, nullptr, "gridv");
      extract_values_from_global_vector(
          discretization, lm, *rotsymmpbc_, &egridvn, nullptr, "gridvn");
      break;
    }
  }
}

/*---------------------------------------------------------------------------*
 |  set the (relative) convective velocity at integration point schott 11/14 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::set_convective_velint_n(const bool isale)
{
  // get convective velocity at integration point
  switch (fldpara_->physical_type())
  {
    case Inpar::FLUID::incompressible:
    case Inpar::FLUID::artcomp:
    case Inpar::FLUID::varying_density:
    case Inpar::FLUID::loma:
    case Inpar::FLUID::tempdepwater:
    case Inpar::FLUID::boussinesq:
    {
      convvelintn_.update(velintn_);
      break;
    }
    case Inpar::FLUID::oseen:
    {
      FOUR_C_THROW("not supported for new ost up to now");
      break;
    }
    case Inpar::FLUID::stokes:
    {
      convvelintn_.clear();
      break;
    }
    default:
      FOUR_C_THROW(
          "Physical type not implemented here. For Poro-problems see derived class "
          "FluidEleCalcPoro.");
      break;
  }

  // (ALE case handled implicitly here using the (potential
  //  mesh-movement-dependent) convective velocity, avoiding
  //  various ALE terms used to be calculated before)
  if (isale)
  {
    convvelintn_.update(-1.0, gridvelintn_, 1.0);
  }
}


/*----------------------------------------------------------------------*
 |  compute turbulence parameters                       rasthofer 10/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::get_turbulence_params(
    Teuchos::ParameterList& turbmodelparams, double& Cs_delta_sq, double& Ci_delta_sq, int& nlayer,
    double CsDeltaSq, double CiDeltaSq)
{
  if (fldpara_->turb_mod_action() != Inpar::FLUID::no_model and nsd_ == 2)
    FOUR_C_THROW("turbulence and 2D flow does not make any sense");

  // classical smagorinsky does only have constant parameter
  if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky_with_van_Driest_damping)
  {
    // this will be the y-coordinate of a point in the element interior
    // we will determine the element layer in which he is contained to
    // be able to do the output of visceff etc.
    double center = 0.0;

    for (int inode = 0; inode < nen_; inode++)
    {
      center += xyze_(1, inode);
    }
    center /= nen_;

    // node coordinates of plane to the element layer
    std::shared_ptr<std::vector<double>> planecoords =
        turbmodelparams.get<std::shared_ptr<std::vector<double>>>("planecoords_");

    bool found = false;
    for (nlayer = 0; nlayer < (int)(*planecoords).size() - 1;)
    {
      if (center < (*planecoords)[nlayer + 1])
      {
        found = true;
        break;
      }
      nlayer++;
    }
    if (found == false)
    {
      FOUR_C_THROW("could not determine element layer");
    }
  }
  // --------------------------------------------------
  // Smagorinsky model with dynamic Computation of Cs
  // else if (physical_turbulence_model == "Dynamic_Smagorinsky")
  else if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
  {
    // turb_mod_action_ = Fluid::dynamic_smagorinsky;

    // for homogeneous flow, use averaged quantities
    if (fldpara_->cs_averaged() == true)
    {
      if (turbmodelparams.get<std::string>("HOMDIR", "not_specified") != "not_specified")
      {
        std::shared_ptr<std::vector<double>> averaged_LijMij =
            turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_LijMij_");
        std::shared_ptr<std::vector<double>> averaged_MijMij =
            turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_MijMij_");
        std::shared_ptr<std::vector<double>> averaged_CI_numerator = nullptr;
        std::shared_ptr<std::vector<double>> averaged_CI_denominator = nullptr;
        if (fldpara_->physical_type() == Inpar::FLUID::loma)
        {
          averaged_CI_numerator =
              turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_CI_numerator_");
          averaged_CI_denominator =
              turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_CI_denominator_");
        }

        // get homogeneous direction
        std::string homdir = turbmodelparams.get<std::string>("HOMDIR", "not_specified");

        // here, the layer is determined in order to get the correct
        // averaged value from the vector of averaged (M/L)ijMij
        double xcenter = 0.0;
        double ycenter = 0.0;
        double zcenter = 0.0;
        for (int inode = 0; inode < nen_; inode++)
        {
          xcenter += xyze_(0, inode);
          ycenter += xyze_(1, inode);
          zcenter += xyze_(2, inode);
        }
        xcenter /= nen_;
        ycenter /= nen_;
        zcenter /= nen_;

        if (homdir == "xyz")
        {
          nlayer = 0;
        }
        else if (homdir == "xy" or homdir == "xz" or homdir == "yz")
        {
          std::shared_ptr<std::vector<double>> planecoords =
              turbmodelparams.get<std::shared_ptr<std::vector<double>>>("planecoords_");
          // get center
          double center = 0.0;
          if (homdir == "xy")
            center = zcenter;
          else if (homdir == "xz")
            center = ycenter;
          else if (homdir == "yz")
            center = xcenter;

          bool found = false;
          for (nlayer = 0; nlayer < static_cast<int>((*planecoords).size() - 1);)
          {
            if (center < (*planecoords)[nlayer + 1])
            {
              found = true;
              break;
            }
            nlayer++;
          }
          if (found == false)
          {
            FOUR_C_THROW("could not determine element layer");
          }
        }
        else if (homdir == "x" or homdir == "y" or homdir == "z")
        {
          std::shared_ptr<std::vector<double>> dir1coords =
              turbmodelparams.get<std::shared_ptr<std::vector<double>>>("dir1coords_");
          std::shared_ptr<std::vector<double>> dir2coords =
              turbmodelparams.get<std::shared_ptr<std::vector<double>>>("dir2coords_");
          // get center
          double dim1_center = 0.0;
          double dim2_center = 0.0;
          if (homdir == "x")
          {
            dim1_center = ycenter;
            dim2_center = zcenter;
          }
          else if (homdir == "y")
          {
            dim1_center = xcenter;
            dim2_center = zcenter;
          }
          else if (homdir == "z")
          {
            dim1_center = xcenter;
            dim2_center = ycenter;
          }

          int n1layer = 0;
          int n2layer = 0;
          bool dir1found = false;
          bool dir2found = false;
          for (n1layer = 0; n1layer < (int)(*dir1coords).size() - 1;)
          {
            if (dim1_center < (*dir1coords)[n1layer + 1])
            {
              dir1found = true;
              break;
            }
            n1layer++;
          }
          if (dir1found == false)
          {
            FOUR_C_THROW("could not determine element layer");
          }
          for (n2layer = 0; n2layer < (int)(*dir2coords).size() - 1;)
          {
            if (dim2_center < (*dir2coords)[n2layer + 1])
            {
              dir2found = true;
              break;
            }
            n2layer++;
          }
          if (dir2found == false)
          {
            FOUR_C_THROW("could not determine element layer");
          }

          const int numdir1layer = (int)(*dir1coords).size() - 1;
          nlayer = numdir1layer * n2layer + n1layer;
        }
        else
          FOUR_C_THROW("Homogeneous directions not supported!");

        // Cs_delta_sq is set by the averaged quantities
        if ((*averaged_MijMij)[nlayer] > 1E-16)
          Cs_delta_sq = 0.5 * (*averaged_LijMij)[nlayer] / (*averaged_MijMij)[nlayer];
        else
          Cs_delta_sq = 0.0;

        // clipping to get algorithm stable
        if (Cs_delta_sq < 0) Cs_delta_sq = 0.0;

        // Ci_delta_sq is set by the averaged quantities
        if (fldpara_->physical_type() == Inpar::FLUID::loma and fldpara_->include_ci() == true)
        {
          if ((*averaged_CI_denominator)[nlayer] > 1E-16)
            Ci_delta_sq =
                0.5 * (*averaged_CI_numerator)[nlayer] / (*averaged_CI_denominator)[nlayer];
          else
            Ci_delta_sq = 0.0;

          // clipping to get algorithm stable
          if (Ci_delta_sq < 0.0) Ci_delta_sq = 0.0;
        }
      }
    }
    else
    {
      // when no averaging was done, we just keep the calculated (clipped) value
      Cs_delta_sq = CsDeltaSq;
      if (fldpara_->physical_type() == Inpar::FLUID::loma and fldpara_->include_ci() == true)
        Ci_delta_sq = CiDeltaSq;
    }
  }

  return;
}  // FluidEleCalc::get_turbulence_params


/*----------------------------------------------------------------------*
 |  calculation of (all-scale) subgrid viscosity               vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_subgr_visc(
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const double vol, double& Cs_delta_sq,
    double& Ci_delta_sq)
{
  // cast dimension to a double variable -> pow()
  const double dim = double(nsd_);
  //
  // SMAGORINSKY MODEL
  // -----------------
  //                                   +-                                 -+ 1
  //                               2   |          / h \           / h \    | -
  //    visc          = dens * lmix  * | 2 * eps | u   |   * eps | u   |   | 2
  //        turbulent           |      |          \   / ij        \   / ij |
  //                            |      +-                                 -+
  //                            |
  //                            |      |                                   |
  //                            |      +-----------------------------------+
  //                            |           'resolved' rate of strain
  //                    mixing length
  // -> either provided by dynamic modeling procedure and stored in Cs_delta_sq
  // -> or computed based on fixed Smagorinsky constant Cs:
  //             Cs = 0.17   (Lilly --- Determined from filter
  //                          analysis of Kolmogorov spectrum of
  //                          isotropic turbulence)
  //             0.1 < Cs < 0.24 (depending on the flow)
  //

  // compute (all-scale) rate of strain
  double rateofstrain = -1.0e30;
  rateofstrain = get_strain_rate(evelaf);

  if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
  {
    // subgrid viscosity
    sgvisc_ = densaf_ * Cs_delta_sq * rateofstrain;

    // calculate isotropic part of subgrid-stress tensor (loma only)
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
    {
      if (fldpara_->include_ci() and is_inflow_ele_ == false)
      {
        if (fldpara_->ci() < 0.0)
          q_sq_ = densaf_ * Ci_delta_sq * rateofstrain *
                  rateofstrain;  // remark: missing factor 2 is added in function ContStab()
        else
        {
          // get characteristic element length for Smagorinsky model for 2D and 3D
          // 3D: delta = V^1/3
          // 2D: delta = A^1/2
          const double delta = pow(vol, (1.0 / dim));
          q_sq_ = densaf_ * fldpara_->ci() * delta * delta * rateofstrain * rateofstrain;
        }
      }
      else
        q_sq_ = 0.0;
    }
  }
  // Vreman turbulence model according to
  //"An eddy-viscosity subgrid-scale model for turbulent shear flow: Algebraic theory and
  // applications", 2004
  else if (fldpara_->turb_mod_action() == Inpar::FLUID::vreman or
           fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_vreman)
  {
    if (nsd_ == 3)
    {
      const double cs = fldpara_->cs();
      double beta00;
      double beta11;
      double beta22;
      double beta01;
      double beta02;
      double beta12;
      double bbeta;
      double alphavreman;
      double hkxpow2;
      double hkypow2;
      double hkzpow2;

      Core::LinAlg::Matrix<nsd_, nsd_> velderxy;

      velderxy.multiply_nt(evelaf, derxy_);

      // calculate grid filter width: 3 options:
      //- direction dependent using the element length in x, y and z
      //- cube root of element volume
      //- minimum element length (of x, y, z)
      if (fldpara_->vrfi() == Inpar::FLUID::dir_dep)
      {
        double xmin = 0.0;
        double ymin = 0.0;
        double zmin = 0.0;
        double xmax = 0.0;
        double ymax = 0.0;
        double zmax = 0.0;
        for (int inen = 0; inen < nen_; inen++)
        {
          if (inen == 0)
          {
            xmin = xyze_(0, inen);
            xmax = xyze_(0, inen);
            ymin = xyze_(1, inen);
            ymax = xyze_(1, inen);
            zmin = xyze_(2, inen);
            zmax = xyze_(2, inen);
          }
          else
          {
            if (xyze_(0, inen) < xmin) xmin = xyze_(0, inen);
            if (xyze_(0, inen) > xmax) xmax = xyze_(0, inen);
            if (xyze_(1, inen) < ymin) ymin = xyze_(1, inen);
            if (xyze_(1, inen) > ymax) ymax = xyze_(1, inen);
            if (xyze_(2, inen) < zmin) zmin = xyze_(2, inen);
            if (xyze_(2, inen) > zmax) zmax = xyze_(2, inen);
          }
        }
        hkxpow2 = (xmax - xmin) * (xmax - xmin);  // filter width = 2 grid spacing?
        hkypow2 = (ymax - ymin) * (ymax - ymin);
        hkzpow2 = (zmax - zmin) * (zmax - zmin);
      }
      else if (fldpara_->vrfi() == Inpar::FLUID::cuberootvol)
      {
        hkxpow2 = pow(vol, (2.0 / 3.0));
        hkypow2 = hkxpow2;
        hkzpow2 = hkxpow2;
      }
      else  // minimum element length
      {
        double hk = 0.0;
        double xmin = 0.0;
        double ymin = 0.0;
        double zmin = 0.0;
        double xmax = 0.0;
        double ymax = 0.0;
        double zmax = 0.0;
        for (int inen = 0; inen < nen_; inen++)
        {
          if (inen == 0)
          {
            xmin = xyze_(0, inen);
            xmax = xyze_(0, inen);
            ymin = xyze_(1, inen);
            ymax = xyze_(1, inen);
            zmin = xyze_(2, inen);
            zmax = xyze_(2, inen);
          }
          else
          {
            if (xyze_(0, inen) < xmin) xmin = xyze_(0, inen);
            if (xyze_(0, inen) > xmax) xmax = xyze_(0, inen);
            if (xyze_(1, inen) < ymin) ymin = xyze_(1, inen);
            if (xyze_(1, inen) > ymax) ymax = xyze_(1, inen);
            if (xyze_(2, inen) < zmin) zmin = xyze_(2, inen);
            if (xyze_(2, inen) > zmax) zmax = xyze_(2, inen);
          }
        }
        if ((xmax - xmin) < (ymax - ymin))
        {
          if ((xmax - xmin) < (zmax - zmin)) hk = xmax - xmin;
        }
        else
        {
          if ((ymax - ymin) < (zmax - zmin))
            hk = ymax - ymin;
          else
            hk = zmax - zmin;
        }
        hkxpow2 = hk * hk;
        hkypow2 = hkxpow2;
        hkzpow2 = hkxpow2;
      }


      beta00 = hkxpow2 * velderxy(0, 0) * velderxy(0, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(0, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(0, 2);
      beta11 = hkxpow2 * velderxy(1, 0) * velderxy(1, 0) +
               hkypow2 * velderxy(1, 1) * velderxy(1, 1) +
               hkzpow2 * velderxy(1, 2) * velderxy(1, 2);
      beta22 = hkxpow2 * velderxy(2, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(2, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(2, 2) * velderxy(2, 2);
      beta01 = hkxpow2 * velderxy(0, 0) * velderxy(1, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(1, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(1, 2);
      beta02 = hkxpow2 * velderxy(0, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(2, 2);
      beta12 = hkxpow2 * velderxy(1, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(1, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(1, 2) * velderxy(2, 2);

      bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
              beta11 * beta22 - beta12 * beta12;

      alphavreman = velderxy(0, 0) * velderxy(0, 0) + velderxy(0, 1) * velderxy(0, 1) +
                    velderxy(0, 2) * velderxy(0, 2) + velderxy(1, 0) * velderxy(1, 0) +
                    velderxy(1, 1) * velderxy(1, 1) + velderxy(1, 2) * velderxy(1, 2) +
                    velderxy(2, 0) * velderxy(2, 0) + velderxy(2, 1) * velderxy(2, 1) +
                    velderxy(2, 2) * velderxy(2, 2);

      if (alphavreman < 1.0E-12)
        sgvisc_ = 0.0;
      else
      {
        if (fldpara_->turb_mod_action() == Inpar::FLUID::vreman)
          sgvisc_ = densaf_ * cs *
                    sqrt(bbeta / alphavreman);  // c_vreman=2.5*(c_smagorinsky*c_smagorinsky)
        else
        {
          double Cv = Cs_delta_sq;  // the variable of Cs_delta_sq has only been used to get the
                                    // Vreman constant here.

          sgvisc_ = densaf_ * Cv * sqrt(bbeta / alphavreman);
        }
      }
    }
    else
      FOUR_C_THROW("Vreman model only for nsd_==3");
  }
  else
  {
    double van_Driest_damping = 1.0;
    if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky_with_van_Driest_damping)
    {
      // since the Smagorinsky constant is only valid if hk is in the inertial
      // subrange of turbulent flows, the mixing length is damped in the
      // viscous near wall region using the van Driest damping function
      /*
                                     /         /   y+ \ \
                   lmix = Cs * hk * | 1 - exp | - ---- | |
                                     \         \   A+ / /
      */
      // A+ is a constant parameter, y+ the distance from the wall in wall
      // units
      const double A_plus = 26.0;
      double y_plus;

      // the integration point coordinate is defined by the isometric approach
      /*
                  +-----
                   \
              x =   +      N (x) * x
                   /        j       j
                  +-----
                  node j
      */

      Core::LinAlg::Matrix<nsd_, 1> centernodecoord;
      centernodecoord.multiply(xyze_, funct_);

      if (centernodecoord(1, 0) > 0)
        y_plus = (1.0 - centernodecoord(1, 0)) / fldpara_->ltau();
      else
        y_plus = (1.0 + centernodecoord(1, 0)) / fldpara_->ltau();

      // lmix *= (1.0-exp(-y_plus/A_plus));
      // multiply with van Driest damping function
      van_Driest_damping = (1.0 - exp(-y_plus / A_plus));
      fldpara_->setvan_driestdamping(van_Driest_damping);
    }

    // get characteristic element length for Smagorinsky model for 2D and 3D
    // 3D: hk = V^1/3
    // 2D: hk = A^1/2
    const double hk = std::pow(vol, (1.0 / dim));

    // mixing length set proportional to grid width: lmix = Cs * hk
    double lmix = fldpara_->cs() * van_Driest_damping * hk;

    Cs_delta_sq = lmix * lmix;

    // subgrid viscosity
    sgvisc_ = densaf_ * Cs_delta_sq * rateofstrain;

    // calculate isotropic part of subgrid-stress tensor (loma only)
    if (fldpara_->physical_type() == Inpar::FLUID::loma)
    {
      if (fldpara_->include_ci() and is_inflow_ele_ == false)
      {
        if (fldpara_->ci() < 0.0)
          FOUR_C_THROW("Ci expected!");
        else
        {
          // get characteristic element length for Smagorinsky model for 2D and 3D
          // 3D: delta = V^1/3
          // 2D: delta = A^1/2
          const double delta = pow(vol, (1.0 / dim));
          q_sq_ = densaf_ * fldpara_->ci() * delta * delta * rateofstrain * rateofstrain;
        }
      }
      else
        q_sq_ = 0.0;
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 |  calculation of fine-scale subgrid viscosity                vg 09/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_fine_scale_subgr_visc(
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const double vol)
{
  // cast dimension to a double variable -> pow()
  const double dim = double(nsd_);

  //     // get characteristic element length for Smagorinsky model for 2D and 3D
  // 3D: hk = V^1/3
  // 2D: hk = A^1/2
  const double hk = std::pow(vol, (1.0 / dim));

  if (fldpara_->fssgv() == Inpar::FLUID::smagorinsky_all)
  {
    //
    // ALL-SCALE SMAGORINSKY MODEL
    // ---------------------------
    //                                      +-                                 -+ 1
    //                                  2   |          / h \           / h \    | -
    //    visc          = dens * (C_S*h)  * | 2 * eps | u   |   * eps | u   |   | 2
    //        turbulent                     |          \   / ij        \   / ij |
    //                                      +-                                 -+
    //                                      |                                   |
    //                                      +-----------------------------------+
    //                                            'resolved' rate of strain
    //

    // compute (all-scale) rate of strain
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(evelaf);

    fssgvisc_ = densaf_ * fldpara_->cs() * fldpara_->cs() * hk * hk * rateofstrain;
  }
  else if (fldpara_->fssgv() == Inpar::FLUID::smagorinsky_small)
  {
    //
    // FINE-SCALE SMAGORINSKY MODEL
    // ----------------------------
    //                                      +-                                 -+ 1
    //                                  2   |          /    \          /   \    | -
    //    visc          = dens * (C_S*h)  * | 2 * eps | fsu |   * eps | fsu |   | 2
    //        turbulent                     |          \   / ij        \   / ij |
    //                                      +-                                 -+
    //                                      |                                   |
    //                                      +-----------------------------------+
    //                                            'resolved' rate of strain
    //

    // fine-scale rate of strain
    double fsrateofstrain = -1.0e30;
    fsrateofstrain = get_strain_rate(fsevelaf);

    fssgvisc_ = densaf_ * fldpara_->cs() * fldpara_->cs() * hk * hk * fsrateofstrain;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  compute multifractal subgrid scales parameters    rasthofer 04/2011 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::prepare_multifractal_subgr_scales(
    Core::LinAlg::Matrix<nsd_, 1>& B_mfs, double& D_mfs,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const double vol)
{
  // set input parameters
  double Csgs = fldpara_->csgs();
  double alpha = fldpara_->alpha();

  // allocate vector for parameter N
  // N may depend on the direction
  std::vector<double> Nvel(3);

  // potential calculation of Re to determine N
  double Re_ele = -1.0;
  // characteristic element length
  double hk = 1.0e+10;
  double strainnorm = 0.0;
  // ratio of viscous scale to element length
  double scale_ratio = 0.0;

  // get norm
  const double vel_norm = velint_.norm2();
  const double fsvel_norm = fsvelint_.norm2();

  // do we have a fixed parameter N
  if (not fldpara_->calc_n())
  {
    for (int rr = 1; rr < 3; rr++) Nvel[rr] = fldpara_->n();
  }
  else  // no, so we calculate N from Re
  {
    // calculate characteristic element length
    // cf. stabilization parameters
    switch (fldpara_->ref_length())
    {
      case FLUID::streamlength:
      {
        // a) streamlength due to Tezduyar et al. (1992)
        // normed velocity vector
        Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
        if (vel_norm >= 1e-6)
          velino.update(1.0 / vel_norm, velint_);
        else
        {
          velino.clear();
          velino(0, 0) = 1.0;
        }
        Core::LinAlg::Matrix<nen_, 1> tmp;
        // enriched dofs are not interpolatory with respect to geometry
        if (enrtype == Discret::Elements::Fluid::xwall)
        {
          Core::LinAlg::Matrix<nsd_, nen_> derxy_copy(derxy_);
          for (int inode = 1; inode < nen_; inode += 2)
          {
            for (int idim = 0; idim < nsd_; idim++) derxy_copy(idim, inode) = 0.0;
          }
          tmp.multiply_tn(derxy_copy, velino);
        }
        else
          tmp.multiply_tn(derxy_, velino);
        const double val = tmp.norm1();
        hk = 2.0 / val;

        break;
      }
      case FLUID::sphere_diameter:
      {
        // b) volume-equivalent diameter
        hk = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);

        break;
      }
      case FLUID::cube_edge:
      {
        // c) cubic element length
        hk = std::pow(vol, (1.0 / (double(nsd_))));
        break;
      }
      case FLUID::metric_tensor:
      {
        if (nsd_ != 3) FOUR_C_THROW("Turbulence is 3d!");
        /*          +-           -+   +-           -+   +-           -+
                    |             |   |             |   |             |
                    |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
              G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
               ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                    |    i     j  |   |    i     j  |   |    i     j  |
                    +-           -+   +-           -+   +-           -+
        */
        Core::LinAlg::Matrix<nsd_, nsd_> G(Core::LinAlg::Initialization::zero);

        for (int nn = 0; nn < nsd_; ++nn)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            G(nn, rr) = xji_(nn, 0) * xji_(rr, 0);
            for (int mm = 1; mm < nsd_; ++mm)
            {
              G(nn, rr) += xji_(nn, mm) * xji_(rr, mm);
            }
          }
        }

        /*          +----
                     \
            G : G =   +   G   * G
            -   -    /     ij    ij
            -   -   +----
                     i,j
        */
        double normG = 0.0;
        for (int nn = 0; nn < nsd_; ++nn)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            normG += G(nn, rr) * G(nn, rr);
          }
        }
        hk = std::pow(normG, -0.25);

        break;
      }
      case FLUID::gradient_based:
      {
        if (nsd_ != 3) FOUR_C_THROW("Turbulence is 3d!");
        Core::LinAlg::Matrix<nsd_, 1> normed_velgrad;

        for (int rr = 0; rr < nsd_; ++rr)
        {
          double val = 0.0;
          for (int idim = 0; idim < nsd_; idim++) val += vderxy_(idim, rr) * vderxy_(idim, rr);

          normed_velgrad(rr) = std::sqrt(val);

          // normed_velgrad(rr)=sqrt(vderxy_(0,rr)*vderxy_(0,rr)
          //                      +
          //                      vderxy_(1,rr)*vderxy_(1,rr)
          //                      +
          //                      vderxy_(2,rr)*vderxy_(2,rr));
        }
        double norm = normed_velgrad.norm2();

        // normed gradient
        if (norm > 1e-6)
        {
          for (int rr = 0; rr < nsd_; ++rr)
          {
            normed_velgrad(rr) /= norm;
          }
        }
        else
        {
          normed_velgrad(0) = 1.;
          for (int rr = 1; rr < nsd_; ++rr)
          {
            normed_velgrad(rr) = 0.0;
          }
        }

        // get length in this direction
        double val = 0.0;
        for (int rr = 0; rr < nen_; ++rr) /* loop element nodes */
        {
          double loc = 0.0;
          for (int idim = 0; idim < nsd_; idim++) loc += normed_velgrad(idim) * derxy_(idim, rr);

          val += std::abs(loc);

          // val += abs( normed_velgrad(0)*derxy_(0,rr)
          //            +normed_velgrad(1)*derxy_(1,rr)
          //            +normed_velgrad(2)*derxy_(2,rr));
        } /* end of loop over element nodes */

        hk = 2.0 / val;

        break;
      }
      default:
        FOUR_C_THROW("Unknown length");
    }

// alternative length for comparison, currently not used
#ifdef HMIN  // minimal element length
    double xmin = 0.0;
    double ymin = 0.0;
    double zmin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
    double zmax = 0.0;
    for (int inen = 0; inen < nen_; inen++)
    {
      if (inen == 0)
      {
        xmin = xyze_(0, inen);
        xmax = xyze_(0, inen);
        ymin = xyze_(1, inen);
        ymax = xyze_(1, inen);
        zmin = xyze_(2, inen);
        zmax = xyze_(2, inen);
      }
      else
      {
        if (xyze_(0, inen) < xmin) xmin = xyze_(0, inen);
        if (xyze_(0, inen) > xmax) xmax = xyze_(0, inen);
        if (xyze_(1, inen) < ymin) ymin = xyze_(1, inen);
        if (xyze_(1, inen) > ymax) ymax = xyze_(1, inen);
        if (xyze_(2, inen) < zmin) zmin = xyze_(2, inen);
        if (xyze_(2, inen) > zmax) zmax = xyze_(2, inen);
      }
    }
    if ((xmax - xmin) < (ymax - ymin))
    {
      if ((xmax - xmin) < (zmax - zmin)) hk = xmax - xmin;
    }
    else
    {
      if ((ymax - ymin) < (zmax - zmin))
        hk = ymax - ymin;
      else
        hk = zmax - zmin;
    }
#endif
#ifdef HMAX  // maximal element length
    double xmin = 0.0;
    double ymin = 0.0;
    double zmin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
    double zmax = 0.0;
    for (int inen = 0; inen < nen_; inen++)
    {
      if (inen == 0)
      {
        xmin = xyze_(0, inen);
        xmax = xyze_(0, inen);
        ymin = xyze_(1, inen);
        ymax = xyze_(1, inen);
        zmin = xyze_(2, inen);
        zmax = xyze_(2, inen);
      }
      else
      {
        if (xyze_(0, inen) < xmin) xmin = xyze_(0, inen);
        if (xyze_(0, inen) > xmax) xmax = xyze_(0, inen);
        if (xyze_(1, inen) < ymin) ymin = xyze_(1, inen);
        if (xyze_(1, inen) > ymax) ymax = xyze_(1, inen);
        if (xyze_(2, inen) < zmin) zmin = xyze_(2, inen);
        if (xyze_(2, inen) > zmax) zmax = xyze_(2, inen);
      }
    }
    if ((xmax - xmin) > (ymax - ymin))
    {
      if ((xmax - xmin) > (zmax - zmin)) hk = xmax - xmin;
    }
    else
    {
      if ((ymax - ymin) > (zmax - zmin))
        hk = ymax - ymin;
      else
        hk = zmax - zmin;
    }
#endif

    if (hk == 1.0e+10) FOUR_C_THROW("Something went wrong!");

    switch (fldpara_->ref_vel())
    {
      case FLUID::resolved:
      {
        Re_ele = vel_norm * hk * densaf_ / visc_;
        break;
      }
      case FLUID::fine_scale:
      {
        Re_ele = fsvel_norm * hk * densaf_ / visc_;
        break;
      }
      case FLUID::strainrate:
      {
        strainnorm = get_strain_rate(evelaf);
        strainnorm /= sqrt(2.0);  // cf. Burton & Dahm 2005
        Re_ele = strainnorm * hk * hk * densaf_ / visc_;
        break;
      }
      default:
        FOUR_C_THROW("Unknown velocity!");
    }
    if (Re_ele < 0.0) FOUR_C_THROW("Something went wrong!");

    // clip Re to prevent negative N
    if (Re_ele < 1.0) Re_ele = 1.0;

    //
    //   Delta
    //  ---------  ~ Re^(3/4)
    //  lambda_nu
    //
    scale_ratio = fldpara_->c_nu() * pow(Re_ele, 0.75);
    // scale_ratio < 1.0 leads to N < 0
    // therefore, we clip once more
    if (scale_ratio < 1.0) scale_ratio = 1.0;

    //         |   Delta     |
    //  N =log | ----------- |
    //        2|  lambda_nu  |
    double N_re = log(scale_ratio) / log(2.0);
    if (N_re < 0.0) FOUR_C_THROW("Something went wrong when calculating N!");

    // store calculated N
    for (int i = 0; i < nsd_; i++) Nvel[i] = N_re;
  }

  // calculate near-wall correction
  double Cai_phi = 0.0;
  if (fldpara_->near_wall_limit())
  {
    // if not yet calculated, estimate norm of strain rate
    if ((not fldpara_->calc_n()) or (fldpara_->ref_vel() != FLUID::strainrate))
    {
      // strainnorm = GetNormStrain(evelaf,derxy_,vderxy_);
      strainnorm = get_strain_rate(evelaf);
      strainnorm /= sqrt(2.0);  // cf. Burton & Dahm 2005
    }
    // and reference length
    if (not fldpara_->calc_n()) FOUR_C_THROW("hk not yet calculated");  // solution see scatra

    // get Re from strain rate
    double Re_ele_str = strainnorm * hk * hk * densaf_ / visc_;
    if (Re_ele_str < 0.0) FOUR_C_THROW("Something went wrong!");
    // ensure positive values
    if (Re_ele_str < 1.0) Re_ele_str = 1.0;

    // calculate corrected Csgs
    //           -3/16
    //  *(1 - (Re)   )
    //
    Csgs *= (1.0 - pow(Re_ele_str, -3.0 / 16.0));

    // store Cai for application to scalar field
    Cai_phi = (1.0 - pow(Re_ele_str, -3.0 / 16.0));
  }

  // call function to compute coefficient B
  calc_multi_frac_subgrid_vel_coef(Csgs, alpha, Nvel, B_mfs);

  // prepare calculation of subgrid-scalar coefficient for loma
  // required if further subgrid-scale terms of cross- and Reynolds-stress
  // type arising in the continuity equation should be included
  if (fldpara_->physical_type() == Inpar::FLUID::loma)
  {
    // set input parameters
    double Csgs_phi = fldpara_->csgs_phi();

    // calculate prandtl number
    double Pr = visc_ / diffus_;

    // since there are differences in the physical behavior between low and high
    // Prandtl/Schmidt number regime, we define a limit
    // to distinguish between the low and high Prandtl/Schmidt number regime
    // note: there is no clear definition of the ranges
    const double Pr_limit = 2.0;

    // allocate double for parameter N
    double Nphi = 0.0;
    // ratio of dissipation scale to element length
    double scale_ratio_phi = 0.0;

    if (fldpara_->calc_n())
    {
      //
      //   Delta
      //  ---------  ~ Re^(3/4)*Pr^(p)
      //  lambda_diff
      //
      // Pr <= 1: p=3/4
      // Pr >> 1: p=1/2
      double p = 0.75;
      if (Pr > Pr_limit) p = 0.5;

      scale_ratio_phi = fldpara_->c_diff() * pow(Re_ele, 0.75) * pow(Pr, p);
      // scale_ratio < 1.0 leads to N < 0
      // therefore, we clip again
      if (scale_ratio_phi < 1.0) scale_ratio_phi = 1.0;

      //         |   Delta     |
      //  N =log | ----------- |
      //        2|  lambda_nu  |
      Nphi = log(scale_ratio_phi) / log(2.0);
      if (Nphi < 0.0) FOUR_C_THROW("Something went wrong when calculating N!");
    }
    else
      FOUR_C_THROW("Multifractal subgrid-scales for loma with calculation of N, only!");

    // call function to compute coefficient D
    if (not fldpara_->near_wall_limit_scatra())
      calc_multi_frac_subgrid_sca_coef(Csgs_phi, alpha, Pr, Pr_limit, Nvel, Nphi, D_mfs);
    else
    {
      if (not fldpara_->near_wall_limit()) FOUR_C_THROW("Near-wall limit expected!");

      calc_multi_frac_subgrid_sca_coef(Csgs_phi * Cai_phi, alpha, Pr, Pr_limit, Nvel, Nphi, D_mfs);
    }
  }
}



/*-------------------------------------------------------------------------------*
 |calculation parameter for multifractal subgrid scale modeling  rasthofer 03/11 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_multi_frac_subgrid_vel_coef(
    const double Csgs, const double alpha, const std::vector<double> Nvel,
    Core::LinAlg::Matrix<nsd_, 1>& B_mfs)
{
  //
  //          |       1              |
  //  kappa = | -------------------- |
  //          |  1 - alpha ^ (-4/3)  |
  //
  double kappa = 1.0 / (1.0 - pow(alpha, -4.0 / 3.0));

  //                  1                                    1
  //                  2                 |                 |2
  //  B = Csgs * kappa * 2 ^ (-2*N/3) * | 2 ^ (4*N/3) - 1 |
  //                                    |                 |
  //
  for (int dim = 0; dim < nsd_; dim++)
  {
    B_mfs(dim, 0) = Csgs * sqrt(kappa) * pow(2.0, -2.0 * Nvel[dim] / 3.0) *
                    sqrt((pow(2.0, 4.0 * Nvel[dim] / 3.0) - 1.0));
  }

  //  if (eid_ == 100){
  //    std::cout << "B  " << std::setprecision(10) << B_mfs(0,0) << "  " << B_mfs(1,0) << "  " <<
  //    B_mfs(2,0) << "  " << std::endl; std::cout << "CsgsB  " << std::setprecision(10) << Csgs <<
  //    std::endl;
  //  }

  return;
}


/*-------------------------------------------------------------------------------*
 |calculation parameter for multifractal subgrid scale modeling  rasthofer 02/12 |
 |subgrid-scale scalar for loma                                                  |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::calc_multi_frac_subgrid_sca_coef(
    const double Csgs, const double alpha, const double Pr, const double Pr_limit,
    const std::vector<double> Nvel, double Nphi, double& D_mfs)
{
  // here, we have to distinguish tree different cases:
  // Pr ~ 1 : fluid and scalar field have the nearly the same cutoff (usual case)
  //          k^(-5/3) scaling -> gamma = 4/3
  // Pr >> 1: (i)  cutoff in the inertial-convective range (Nvel>0, tricky!)
  //               k^(-5/3) scaling in the inertial-convective range
  //               k^(-1) scaling in the viscous-convective range
  //          (ii) cutoff in the viscous-convective range (fluid field fully resolved, easier)
  //               k^(-1) scaling -> gamma = 2
  // rare:
  // Pr << 1: scatra field could be fully resolved, not necessary
  //          k^(-5/3) scaling -> gamma = 4/3
  // Remark: case 2.(i) not implemented, yet

  // caution: compared to the mfs-loma paper, gamma denotes gamma+1 here
  double gamma = 0.0;
  // special option for case 2 (i)
  bool two_ranges = false;
  if (Pr < Pr_limit)  // Pr <= 1, i.e., case 1 and 3
    gamma = 4.0 / 3.0;
  else  // Pr >> 1
  {
    FOUR_C_THROW("Loma with Pr>>1?");
    if (Nvel[0] < 1.0)  // Pr >> 1 and fluid fully resolved, i.e., case 2 (ii)
      gamma = 2.0;
    else  // Pr >> 1 and fluid not fully resolved, i.e., case 2 (i)
    {
      if (Nvel[0] > Nphi) FOUR_C_THROW("Nvel < Nphi expected!");
      // here different options are possible
      // 1) we assume k^(-5/3) for the complete range
      gamma = 4.0 / 3.0;
    }
  }

  //
  //   Phi    |       1                |
  //  kappa = | ---------------------- |
  //          |  1 - alpha ^ (-gamma)  |
  //
  double kappa_phi = 1.0 / (1.0 - pow(alpha, -gamma));

  //                                                             1
  //       Phi    Phi                       |                   |2
  //  D = Csgs * kappa * 2 ^ (-gamma*N/2) * | 2 ^ (gamma*N) - 1 |
  //                                        |                   |
  //
  if (not two_ranges)  // usual case
    D_mfs = Csgs * sqrt(kappa_phi) * pow(2.0, -gamma * Nphi / 2.0) *
            sqrt((pow(2.0, gamma * Nphi) - 1.0));
  else
  {
    FOUR_C_THROW("Special option for passive scalars only!");
  }

  return;
}


template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::fine_scale_sub_grid_viscosity_term(
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& fssgviscfac)
{
  if (nsd_ == 2)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* fine-scale subgrid-viscosity term on right hand side */
      /*
                          /                          \
                         |       /    \         / \   |
         - mu_art(fsu) * |  eps | Dfsu | , eps | v |  |
                         |       \    /         \ /   |
                          \                          /
      */
      velforce(0, vi) -=
          fssgviscfac * (2.0 * derxy_(0, vi) * fsvderxy_(0, 0) + derxy_(1, vi) * fsvderxy_(0, 1) +
                            derxy_(1, vi) * fsvderxy_(1, 0));
      velforce(1, vi) -=
          fssgviscfac * (derxy_(0, vi) * fsvderxy_(0, 1) + derxy_(0, vi) * fsvderxy_(1, 0) +
                            2.0 * derxy_(1, vi) * fsvderxy_(1, 1));
    }
  }
  else if (nsd_ == 3)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* fine-scale subgrid-viscosity term on right hand side */
      /*
                            /                          \
                           |       /    \         / \   |
           - mu_art(fsu) * |  eps | Dfsu | , eps | v |  |
                           |       \    /         \ /   |
                            \                          /
      */
      velforce(0, vi) -=
          fssgviscfac * (2.0 * derxy_(0, vi) * fsvderxy_(0, 0) + derxy_(1, vi) * fsvderxy_(0, 1) +
                            derxy_(1, vi) * fsvderxy_(1, 0) + derxy_(2, vi) * fsvderxy_(0, 2) +
                            derxy_(2, vi) * fsvderxy_(2, 0));
      velforce(1, vi) -=
          fssgviscfac * (derxy_(0, vi) * fsvderxy_(0, 1) + derxy_(0, vi) * fsvderxy_(1, 0) +
                            2.0 * derxy_(1, vi) * fsvderxy_(1, 1) +
                            derxy_(2, vi) * fsvderxy_(1, 2) + derxy_(2, vi) * fsvderxy_(2, 1));
      velforce(2, vi) -=
          fssgviscfac * (derxy_(0, vi) * fsvderxy_(0, 2) + derxy_(0, vi) * fsvderxy_(2, 0) +
                            derxy_(1, vi) * fsvderxy_(1, 2) + derxy_(1, vi) * fsvderxy_(2, 1) +
                            2.0 * derxy_(2, vi) * fsvderxy_(2, 2));
    }
  }
  else
    FOUR_C_THROW("fine-scale subgrid viscosity not implemented for 1-D problems!");

  return;
}


//----------------------------------------------------------------------
// Cross-stress terms: multifractal subgrid-scales       rasthofer 06/11
//----------------------------------------------------------------------
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::multfrac_sub_grid_scales_cross(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefacfac, const double& rhsfac)
{
  //--------------------------------------------------------------------
  // rhs contribution
  //--------------------------------------------------------------------
  if (nsd_ == 3)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* cross-stress term on right hand side */
      /*
               /                                      \
              |                                        |
              | ( du o nabla u + u o nabla du ) ,  v   |
              |                                        |
               \                                      /
      */
      velforce(0, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (convvelint_(0, 0) * mffsvderxy_(0, 0) + convvelint_(1, 0) * mffsvderxy_(0, 1) +
              convvelint_(2, 0) * mffsvderxy_(0, 2) + mffsvelint_(0, 0) * vderxy_(0, 0) +
              mffsvelint_(1, 0) * vderxy_(0, 1) + mffsvelint_(2, 0) * vderxy_(0, 2));
      velforce(1, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (convvelint_(0, 0) * mffsvderxy_(1, 0) + convvelint_(1, 0) * mffsvderxy_(1, 1) +
              convvelint_(2, 0) * mffsvderxy_(1, 2) + mffsvelint_(0, 0) * vderxy_(1, 0) +
              mffsvelint_(1, 0) * vderxy_(1, 1) + mffsvelint_(2, 0) * vderxy_(1, 2));
      velforce(2, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (convvelint_(0, 0) * mffsvderxy_(2, 0) + convvelint_(1, 0) * mffsvderxy_(2, 1) +
              convvelint_(2, 0) * mffsvderxy_(2, 2) + mffsvelint_(0, 0) * vderxy_(2, 0) +
              mffsvelint_(1, 0) * vderxy_(2, 1) + mffsvelint_(2, 0) * vderxy_(2, 2));

      /* cross-stress term on right hand side */
      /* additional terms conservative form */
      /*
               /                                         \
              |                                           |
              | ( du (nabla o u) + u (nabla o du ) ,  v   |
              |                                           |
               \                                         /
      */
      if (fldpara_->mfs_is_conservative() or fldpara_->is_conservative())
      {
        velforce(0, vi) -= rhsfac * densaf_ * funct_(vi, 0) *
                           (mffsvelint_(0, 0) * vdiv_ + convvelint_(0, 0) * mffsvdiv_);
        velforce(1, vi) -= rhsfac * densaf_ * funct_(vi, 0) *
                           (mffsvelint_(1, 0) * vdiv_ + convvelint_(1, 0) * mffsvdiv_);
        velforce(2, vi) -= rhsfac * densaf_ * funct_(vi, 0) *
                           (mffsvelint_(2, 0) * vdiv_ + convvelint_(2, 0) * mffsvdiv_);

        if (fldpara_->physical_type() == Inpar::FLUID::loma)
        {
          FOUR_C_THROW("Conservative formulation not supported for loma!");
        }
        if (gridvelint_.norm2() > 1.0e-9)
          FOUR_C_THROW("Conservative formulation not supported with ale!");
      }
    }
  }
  else
    FOUR_C_THROW("Multifractal subgrid-scale modeling model for 3D-problems only!");

  //--------------------------------------------------------------------
  // lhs contribution
  //--------------------------------------------------------------------
  // linearized as far as possible due to the filter

  // turn left-hand-side contribution on
  double beta = fldpara_->beta();

  if (beta > 1.0e-9)
  {
    if (gridvelint_.norm2() > 1.0e-9)
      FOUR_C_THROW("left hand side terms of MFS not supported with ale");
    Core::LinAlg::Matrix<nen_, 1> mfconv_c(Core::LinAlg::Initialization::zero);
    mfconv_c.multiply_tn(derxy_, mffsvelint_);
    // convective part
    for (int ui = 0; ui < nen_; ui++)
    {
      for (int idim = 0; idim < nsd_; idim++)
      {
        int fui = ui * nsd_ + idim;
        for (int vi = 0; vi < nen_; vi++)
        {
          for (int jdim = 0; jdim < nsd_; jdim++)
          {
            int fvi = vi * nsd_ + jdim;
            /*
                    /                             \
                   |  /                 \          |
                   | |   rho*Du  o nabla | du , v  |
                   |  \                 /          |
                    \                             /
            */
            estif_u(fvi, fui) +=
                beta * timefacfac * densaf_ * funct_(vi) * funct_(ui) * mffsvderxy_(jdim, idim);
            /*
                    /                             \
                   |  /                 \          |
                   | |   rho*du  o nabla | Du , v  |
                   |  \                 /          |
                    \                             /
            */
            if (jdim == idim)
            {
              estif_u(fvi, fui) += beta * timefacfac * densaf_ * funct_(vi) * mfconv_c(ui);
            }

            // additional terms conservative part
            if (fldpara_->mfs_is_conservative() or fldpara_->is_conservative())
            {
              /*
                   /                                     \
                   |      /               \       \      |
                   |  du | rho*nabla o Du  | , v   |     |
                   |      \               /       /      |
                   \                                     /
              */
              estif_u(fvi, fui) +=
                  beta * timefacfac * densaf_ * funct_(vi) * mffsvelint_(jdim) * derxy_(idim, ui);
              /*
                    /                                     \
                    |      /               \       \      |
                    |  Du | rho*nabla o du  | , v   |     |
                    |      \               /       /      |
                    \                                     /
              */
              if (jdim == idim)
              {
                estif_u(fvi, fui) +=
                    beta * timefacfac * densaf_ * funct_(vi) * funct_(ui) * mffsvdiv_;
              }

              if (fldpara_->physical_type() == Inpar::FLUID::loma)
              {
                FOUR_C_THROW("Conservative formulation not supported for loma!");
              }
            }
          }
        }
      }
    }
  }

  return;
}


//----------------------------------------------------------------------
// Reynolds-stress terms: multifractal subgrid-scales    rasthofer 06/11
//----------------------------------------------------------------------
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::multfrac_sub_grid_scales_reynolds(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefacfac, const double& rhsfac)
{
  //--------------------------------------------------------------------
  // rhs contribution
  //--------------------------------------------------------------------
  if (nsd_ == 3)
  {
    for (int vi = 0; vi < nen_; ++vi)
    {
      /* reynolds-stress term on right hand side */
      /*
               /                       \
              |                         |
              | ( du o nabla du) ,  v   |
              |                         |
               \                       /
      */
      velforce(0, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (mffsvelint_(0, 0) * mffsvderxy_(0, 0) + mffsvelint_(1, 0) * mffsvderxy_(0, 1) +
              mffsvelint_(2, 0) * mffsvderxy_(0, 2));
      velforce(1, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (mffsvelint_(0, 0) * mffsvderxy_(1, 0) + mffsvelint_(1, 0) * mffsvderxy_(1, 1) +
              mffsvelint_(2, 0) * mffsvderxy_(1, 2));
      velforce(2, vi) -=
          rhsfac * densaf_ * funct_(vi, 0) *
          (mffsvelint_(0, 0) * mffsvderxy_(2, 0) + mffsvelint_(1, 0) * mffsvderxy_(2, 1) +
              mffsvelint_(2, 0) * mffsvderxy_(2, 2));

      /* reynolds-stress term on right hand side */
      /* additional terms conservative form */
      /*
               /                       \
              |                         |
              |   du (nabla o du),  v   |
              |                         |
               \                       /
      */
      if (fldpara_->mfs_is_conservative() or fldpara_->is_conservative())
      {
        velforce(0, vi) -= rhsfac * densaf_ * funct_(vi, 0) * (mffsvelint_(0, 0) * mffsvdiv_);
        velforce(1, vi) -= rhsfac * densaf_ * funct_(vi, 0) * (mffsvelint_(1, 0) * mffsvdiv_);
        velforce(2, vi) -= rhsfac * densaf_ * funct_(vi, 0) * (mffsvelint_(2, 0) * mffsvdiv_);

        if (fldpara_->physical_type() == Inpar::FLUID::loma)
        {
          FOUR_C_THROW("Conservative formulation not supported for loma!");
        }
      }
    }
  }
  else
    FOUR_C_THROW("Multifractal subgrid-scale modeling for 3D-problems only!");

  //--------------------------------------------------------------------
  // lhs contribution
  //--------------------------------------------------------------------
  // no contribution, due to necessary linearization of filter

  return;
}


//----------------------------------------------------------------------
// concistency terms for residual-based stabilization           bk 10/14
//----------------------------------------------------------------------
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype,
    enrtype>::multfrac_sub_grid_scales_consistent_residual()
{
  if (fldpara_->turb_mod_action() == Inpar::FLUID::multifractal_subgrid_scales &&
      fldpara_->consistent_mfs_residual())
  {
    if (not fldparatimint_->is_genalpha())
      FOUR_C_THROW(
          "please check the implementation of the consistent residual for mfs for your "
          "time-integrator!");

    Core::LinAlg::Matrix<nsd_, 1> velforce(Core::LinAlg::Initialization::zero);

    /* cross-stress term of residual */
    /*
             /                                      \
            |                                        |
            | ( du o nabla u + u o nabla du )        |
            |                                        |
             \                                      /
    */
    velforce(0, 0) +=
        densaf_ * (convvelint_(0, 0) * mffsvderxy_(0, 0) + convvelint_(1, 0) * mffsvderxy_(0, 1) +
                      convvelint_(2, 0) * mffsvderxy_(0, 2) + mffsvelint_(0, 0) * vderxy_(0, 0) +
                      mffsvelint_(1, 0) * vderxy_(0, 1) + mffsvelint_(2, 0) * vderxy_(0, 2));
    velforce(1, 0) +=
        densaf_ * (convvelint_(0, 0) * mffsvderxy_(1, 0) + convvelint_(1, 0) * mffsvderxy_(1, 1) +
                      convvelint_(2, 0) * mffsvderxy_(1, 2) + mffsvelint_(0, 0) * vderxy_(1, 0) +
                      mffsvelint_(1, 0) * vderxy_(1, 1) + mffsvelint_(2, 0) * vderxy_(1, 2));
    velforce(2, 0) +=
        densaf_ * (convvelint_(0, 0) * mffsvderxy_(2, 0) + convvelint_(1, 0) * mffsvderxy_(2, 1) +
                      convvelint_(2, 0) * mffsvderxy_(2, 2) + mffsvelint_(0, 0) * vderxy_(2, 0) +
                      mffsvelint_(1, 0) * vderxy_(2, 1) + mffsvelint_(2, 0) * vderxy_(2, 2));


    /* reynolds-stress term of residual */
    /*
             /                       \
            |                         |
            | ( du o nabla du)        |
            |                         |
             \                       /
    */
    velforce(0, 0) +=
        densaf_ * (mffsvelint_(0, 0) * mffsvderxy_(0, 0) + mffsvelint_(1, 0) * mffsvderxy_(0, 1) +
                      mffsvelint_(2, 0) * mffsvderxy_(0, 2));
    velforce(1, 0) +=
        densaf_ * (mffsvelint_(0, 0) * mffsvderxy_(1, 0) + mffsvelint_(1, 0) * mffsvderxy_(1, 1) +
                      mffsvelint_(2, 0) * mffsvderxy_(1, 2));
    velforce(2, 0) +=
        densaf_ * (mffsvelint_(0, 0) * mffsvderxy_(2, 0) + mffsvelint_(1, 0) * mffsvderxy_(2, 1) +
                      mffsvelint_(2, 0) * mffsvderxy_(2, 2));

    for (int rr = 0; rr < nsd_; ++rr)
    {
      momres_old_(rr) += velforce(rr);
    }
  }
  return;
}

//----------------------------------------------------------------------
// outpu for statistics of dynamic Smagorinsky           rasthofer 09/12
//----------------------------------------------------------------------
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalc<distype, enrtype>::store_model_parameters_for_output(
    const double Cs_delta_sq, const double Ci_delta_sq, const int nlayer, const bool isowned,
    Teuchos::ParameterList& turbmodelparams)
{
  // do the fastest test first
  if (isowned)
  {
    if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky or
        fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky_with_van_Driest_damping)
    {
      if (turbmodelparams.get<std::string>("TURBULENCE_APPROACH", "none") == "CLASSICAL_LES")
      {
        if (turbmodelparams.get<std::string>("CANONICAL_FLOW", "no") ==
                "channel_flow_of_height_2" or
            turbmodelparams.get<std::string>("CANONICAL_FLOW", "no") ==
                "loma_channel_flow_of_height_2" or
            turbmodelparams.get<std::string>("CANONICAL_FLOW", "no") ==
                "scatra_channel_flow_of_height_2")
        {
          // recompute delta = pow((vol),(1.0/3.0))
          // evaluate shape functions and derivatives at element center
          eval_shape_func_and_derivs_at_ele_center();
          // set element volume
          const double vol = fac_;

          // to compare it with the standard Smagorinsky Cs
          if (fldpara_->turb_mod_action() == Inpar::FLUID::dynamic_smagorinsky)
            (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
                "local_Cs_sum")))[nlayer] += sqrt(Cs_delta_sq) / pow((vol), (1.0 / 3.0));
          else if (fldpara_->turb_mod_action() == Inpar::FLUID::smagorinsky_with_van_Driest_damping)
            (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
                "local_Cs_sum")))[nlayer] += fldpara_->cs() * fldpara_->van_driestdamping();
          else
            FOUR_C_THROW("Dynamic Smagorinsky or Smagorisnsky with van Driest damping expected!");
          (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
              "local_Cs_delta_sq_sum")))[nlayer] += Cs_delta_sq;
          (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
              "local_visceff_sum")))[nlayer] += visceff_;

          if (turbmodelparams.get<std::string>("CANONICAL_FLOW", "no") ==
              "loma_channel_flow_of_height_2")
          {
            (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
                "local_Ci_sum")))[nlayer] += sqrt(Ci_delta_sq) / pow((vol), (1.0 / 3.0));
            (*(turbmodelparams.get<std::shared_ptr<std::vector<double>>>(
                "local_Ci_delta_sq_sum")))[nlayer] += Ci_delta_sq;
          }
        }
      }
    }
  }

  return;
}

// template classes
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::Fluid::xwall>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::tet4,
    Discret::Elements::Fluid::xwall>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::hex20,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::hex27,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::tet4,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::tet10,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::wedge15,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::quad4,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::quad8,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::quad9,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::tri3,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::tri6,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::nurbs9,
    Discret::Elements::Fluid::none>;
template class Discret::Elements::FluidEleCalc<Core::FE::CellType::nurbs27,
    Discret::Elements::Fluid::none>;

FOUR_C_NAMESPACE_CLOSE
