// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_gder2.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_multiscale.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
    const int numdofpernode, const int numscal, const std::string& disname)
    : numdofpernode_(numdofpernode),
      numscal_(numscal),
      scatrapara_(Discret::Elements::ScaTraEleParameterStd::instance(disname)),
      turbparams_(Discret::Elements::ScaTraEleParameterTurbulence::instance(disname)),
      scatraparatimint_(Discret::Elements::ScaTraEleParameterTimInt::instance(disname)),
      diffmanager_(std::make_shared<ScaTraEleDiffManager>(numscal_)),
      reamanager_(std::make_shared<ScaTraEleReaManager>(numscal_)),
      ephin_(numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero)),
      ephinp_(numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero)),
      ehist_(numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero)),
      fsphinp_(numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero)),
      rotsymmpbc_(std::make_shared<FLD::RotationallySymmetricPeriodicBC<distype, nsd_ + 1,
              Discret::Elements::Fluid::none>>()),
      evelnp_(),
      econvelnp_(),
      efsvel_(),
      eaccnp_(),
      edispnp_(),
      eprenp_(),
      tpn_(0.0),
      xsi_(),
      xyze_(),
      funct_(),
      deriv_(),
      deriv2_(),
      derxy_(),
      derxy2_(),
      xjm_(),
      xij_(),
      xder2_(),
      bodyforce_(numdofpernode_),
      weights_(),
      myknots_(nsd_),
      eid_(0),
      ele_(nullptr),
      scatravarmanager_(std::make_shared<ScaTraEleInternalVariableManager<nsd_, nen_>>(numscal_))
{
  FOUR_C_ASSERT(
      nsd_ >= nsd_ele_, "problem dimension has to be equal or larger than the element dimension!");

  // safety checks related with turbulence
  if (scatrapara_->assgd() and turbparams_->fssgd())
  {
    FOUR_C_THROW(
        "No combination of all-scale and fine-scale subgrid-diffusivity approach currently "
        "possible!");
  }
  if (turbparams_->bd_gp() and not scatrapara_->mat_gp())
  {
    FOUR_C_THROW(
        "Evaluation of B and D at Gauss point should always be combined with material evaluation "
        "at Gauss point!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::setup_calc(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  // get element coordinates
  read_element_coordinates(ele);

  // Now do the nurbs specific stuff (for isogeometric elements)
  if (Core::FE::is_nurbs_celltype(distype))
  {
    // access knots and weights for this element
    bool zero_size =
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(discretization, ele, myknots_, weights_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return -1;
  }  // Nurbs specific stuff

  // set element id
  eid_ = ele->id();
  // set element
  ele_ = ele;

  // rotationally symmetric periodic bc's: do setup for current element
  rotsymmpbc_->setup(discretization, ele);

  std::shared_ptr<Core::Mat::Material> material = ele->material();
  if (material->material_type() == Core::Materials::m_scatra)
  {
    const std::shared_ptr<const Mat::ScatraMat> single_material =
        std::static_pointer_cast<const Mat::ScatraMat>(material);
    scatravarmanager_->set_reacts_to_force(single_material->reacts_to_external_force(), 0);
  }

  if (material->material_type() == Core::Materials::m_matlist or
      material->material_type() == Core::Materials::m_matlist_reactions)
  {
    const std::shared_ptr<const Mat::MatList>& material_list =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (material_list->num_mat() < numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < numscal_; ++k)
    {
      const int material_id = material_list->mat_id(k);

      if (material_list->material_by_id(material_id)->material_type() == Core::Materials::m_scatra)
      {
        std::shared_ptr<const Mat::ScatraMat> single_material =
            std::static_pointer_cast<const Mat::ScatraMat>(
                material_list->material_by_id(material_id));
        scatravarmanager_->set_reacts_to_force(single_material->reacts_to_external_force(), k);
      }
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::evaluate(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------

  if (setup_calc(ele, discretization) == -1) return 0;

  //--------------------------------------------------------------------------------
  // extract element based or nodal values
  //--------------------------------------------------------------------------------

  extract_element_and_node_values(ele, params, discretization, la);

  //--------------------------------------------------------------------------------
  // prepare turbulence models
  //--------------------------------------------------------------------------------

  int nlayer = 0;
  extract_turbulence_approach(ele, params, discretization, la, nlayer);

  //--------------------------------------------------------------------------------
  // calculate element coefficient matrix and rhs
  //--------------------------------------------------------------------------------

  sysmat(ele, elemat1, elevec1, elevec2);

  // perform finite difference check on element level
  if (scatrapara_->fd_check() == Inpar::ScaTra::fdcheck_local and
      ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
    fd_check(ele, elemat1, elevec1, elevec2);

  // ---------------------------------------------------------------------
  // output values of Prt, diffeff and Cs_delta_sq_Prt (channel flow only)
  // ---------------------------------------------------------------------

  if (turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky and turbparams_->cs_av())
  {
    Teuchos::ParameterList& turbulencelist = params.sublist("TURBULENCE MODEL");
    store_model_parameters_for_output(ele,
        ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()), turbulencelist,
        nlayer);
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatrapara_->nds_vel();

  // get convective (velocity - mesh displacement) velocity at nodes
  auto convel = discretization.get_state(ndsvel, "convective velocity field");
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (unsigned inode = 0; inode < nen_; ++inode)
    for (unsigned idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  if (scatrapara_->has_external_force())
  {
    auto force_velocity = discretization.get_state(ndsvel, "force_velocity");
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
        *force_velocity, eforcevelocity_, lmvel);
  }

  // extract local values of convective velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvelnp_, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->rotate_my_values_if_necessary(econvelnp_);

  // get additional state vector for ALE case: grid displacement
  if (scatrapara_->is_ale())
  {
    // get velocity at nodes
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
        discretization.get_state(ndsvel, "velocity field");
    if (vel == nullptr) FOUR_C_THROW("Cannot get state vector velocity");

    // extract local values of velocity field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*vel, evelnp_, lmvel);

    // rotate the vector field in the case of rotationally symmetric boundary conditions
    rotsymmpbc_->rotate_my_values_if_necessary(evelnp_);

    // get number of dofset associated with displacement related dofs
    const int ndsdisp = scatrapara_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      for (unsigned idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    update_node_coordinates();
  }
  else
  {
    edispnp_.clear();

    // velocity = convective velocity for the non-ale case
    evelnp_ = econvelnp_;
  }

  // get data required for subgrid-scale velocity: acceleration and pressure
  if (scatrapara_->rb_sub_gr_vel())
  {
    // get acceleration values at nodes
    const std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
        discretization.get_state(ndsvel, "acceleration field");
    if (acc == nullptr) FOUR_C_THROW("Cannot get state vector acceleration field");

    // extract local values of acceleration field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*acc, eaccnp_, lmvel);

    // rotate the vector field in the case of rotationally symmetric boundary conditions
    rotsymmpbc_->rotate_my_values_if_necessary(eaccnp_);

    // construct location vector for pressure dofs
    std::vector<int> lmpre(nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      lmpre[inode] = la[ndsvel].lm_[inode * numveldofpernode + nsd_];

    // extract local values of pressure field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*convel, eprenp_, lmpre);
  }

  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (hist == nullptr || phinp == nullptr)
    FOUR_C_THROW("Cannot get state vector 'hist' and/or 'phinp'");

  // values of scatra field are always in first dofset
  const std::vector<int>& lm = la[0].lm_;
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*hist, ehist_, lm);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, lm);

  if (scatraparatimint_->is_gen_alpha() and not scatraparatimint_->is_incremental())
  {
    // extract additional local values from global vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
    if (phin == nullptr) FOUR_C_THROW("Cannot get state vector 'phin'");
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phin, ephin_, lm);
  }

  // set reaction coefficient
  if (params.isParameter("rea_coeff"))
    reamanager_->set_rea_coeff(params.get<double>("rea_coeff"), 0);

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------
  body_force(ele);
  //--------------------------------------------------------------------------------
  // further node-based source terms not given via Neumann volume condition
  // i.e., set special body force for homogeneous isotropic turbulence
  //--------------------------------------------------------------------------------
  other_node_based_source_terms(lm, discretization, params);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::extract_turbulence_approach(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la, int& nlayer)
{
  if (turbparams_->turb_model() != Inpar::FLUID::no_model or
      (scatraparatimint_->is_incremental() and turbparams_->fssgd()))
  {
    // do some checks first
    if (numscal_ != 1 or numdofpernode_ != 1)
      FOUR_C_THROW("For the time being, turbulence approaches only support one scalar field!");
  }

  // set turbulent Prandt number to value given in parameterlist
  tpn_ = turbparams_->tpn();

  // if we have a dynamic model,we overwrite this value by a local element-based one here
  if (turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
  {
    Teuchos::ParameterList& turbulencelist = params.sublist("TURBULENCE MODEL");
    // remark: for dynamic estimation, this returns (Cs*h)^2 / Pr_t
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_prt =
        turbulencelist.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_ele_Prt");
    const int id = ele->lid();
    tpn_ = ele_prt->local_values_as_span()[id];

    // when no averaging was done, we just keep the calculated (clipped) value
    if (turbparams_->cs_av())
      get_mean_prt_of_homogeneous_direction(params.sublist("TURBULENCE MODEL"), nlayer);
  }

  // get fine-scale values
  if ((scatraparatimint_->is_incremental() and
          (turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_all or
              turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_small)) or
      turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    // get fine scale scalar field
    std::shared_ptr<const Core::LinAlg::Vector<double>> gfsphinp =
        discretization.get_state("fsphinp");
    if (gfsphinp == nullptr) FOUR_C_THROW("Cannot get state vector 'fsphinp'");

    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*gfsphinp, fsphinp_, la[0].lm_);

    if (turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_small or
        turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      // get number of dofset associated with velocity-related dofs
      const int ndsvel = scatrapara_->nds_vel();

      // get fine-scale velocity at nodes
      const std::shared_ptr<const Core::LinAlg::Vector<double>> fsvelocity =
          discretization.get_state(ndsvel, "fine-scale velocity field");
      if (fsvelocity == nullptr)
        FOUR_C_THROW("Cannot get fine-scale velocity field from scatra discretization!");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (unsigned inode = 0; inode < nen_; ++inode)
        for (unsigned idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // extract local values of fine-scale velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*fsvelocity, efsvel_, lmvel);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::sysmat(Core::Elements::Element* ele,
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs,
    Core::LinAlg::SerialDenseVector& subgrdiff)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  const double vol = eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // get material and stabilization parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n) (one per transported scalar)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F) (one per transported scalar)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M) (one per transported scalar)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // material parameter at the element center are also necessary
  // even if the stabilization parameter is evaluated at the element center
  if (not scatrapara_->mat_gp())
  {
    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  //----------------------------------------------------------------------
  // calculation of subgrid diffusivity and stabilization parameter(s)
  // at element center
  //----------------------------------------------------------------------

  // the stabilization parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);
  // stabilization parameters for the external force term (one per transported scalar)
  // std::vector<double> tau_force(numscal_, 0.0);
  // subgrid-scale diffusion coefficient
  double sgdiff(0.0);

  if (not scatrapara_->tau_gp())
  {
    for (int k = 0; k < numscal_; ++k)  // loop of each transported scalar
    {
      // get velocity at element center
      Core::LinAlg::Matrix<nsd_, 1> convelint = scatravarmanager_->con_vel(k);

      // calculation of all-scale subgrid diffusivity (by, e.g.,
      // Smagorinsky model) at element center
      if (turbparams_->turb_model() == Inpar::FLUID::smagorinsky or
          turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky or
          turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
      {
        calc_subgr_diff(visc, vol, k, densnp[k]);
      }

      // calculation of fine-scale artificial subgrid diffusivity at element center
      if (turbparams_->fssgd())
        calc_fine_scale_subgr_diff(sgdiff, subgrdiff, ele, vol, k, densnp[k],
            diffmanager_->get_isotropic_diff(k), convelint);

      // calculation of stabilization parameter at element center
      calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
          reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
          convelint, vol);
    }
  }

  // prepare multifractal subgrid-scale modeling
  // calculation of model coefficients B (velocity) and D (scalar)
  // at element center
  // coefficient B of fine-scale velocity
  Core::LinAlg::Matrix<nsd_, 1> B_mfs(Core::LinAlg::Initialization::zero);
  // coefficient D of fine-scale scalar
  double D_mfs = 0.0;
  if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    if (not turbparams_->bd_gp())
    {
      // make sure to get material parameters at element center
      // hence, determine them if not yet available
      if (scatrapara_->mat_gp())
      {
        set_internal_variables_for_mat_and_rhs();

        get_material_params(ele, densn, densnp, densam, visc);
      }

      // provide necessary velocities and gradients at element center
      // get velocity at element center
      Core::LinAlg::Matrix<nsd_, 1> fsvelint(Core::LinAlg::Initialization::zero);
      fsvelint.multiply(efsvel_, funct_);

      // calculate model coefficients
      for (int k = 0; k < numscal_; ++k)  // loop of each transported scalar
        calc_b_and_d_for_multifrac_subgrid_scales(B_mfs, D_mfs, vol, k, densnp[k],
            diffmanager_->get_isotropic_diff(k), visc, scatravarmanager_->con_vel(k), fsvelint);
    }
  }

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    // velocity divergence required for conservative form
    double vdiv(0.0);
    if (scatrapara_->is_conservative()) get_divergence(vdiv, evelnp_);

    // get fine-scale velocity and its derivatives at integration point
    Core::LinAlg::Matrix<nsd_, 1> fsvelint(Core::LinAlg::Initialization::zero);
    if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
      fsvelint.multiply(efsvel_, funct_);

    // loop all scalars
    for (int k = 0; k < numscal_; ++k)  // deal with a system of transported scalars
    {
      // reactive part of the form: (reaction coefficient)*phi
      double rea_phi(0.0);
      rea_phi = densnp[k] * scatravarmanager_->phinp(k) * reamanager_->get_rea_coeff(k);

      // compute gradient of fine-scale part of scalar value
      Core::LinAlg::Matrix<nsd_, 1> fsgradphi(Core::LinAlg::Initialization::zero);
      if (turbparams_->fssgd()) fsgradphi.multiply(derxy_, fsphinp_[k]);

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term
      double rhsint(0.0);
      get_rhs_int(rhsint, densnp[k], k);

      //--------------------------------------------------------------------
      // calculation of (fine-scale) subgrid diffusivity, subgrid-scale
      // velocity and stabilization parameter(s) at integration point
      //--------------------------------------------------------------------

      // subgrid-scale convective term
      Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::zero);
      // subgrid-scale velocity vector in gausspoint
      Core::LinAlg::Matrix<nsd_, 1> sgvelint(Core::LinAlg::Initialization::zero);

      double scatrares(0.0);
      // calculate strong residual
      calc_strong_residual(k, scatrares, densam[k], densnp[k], rea_phi, rhsint, tau[k]);

      if (scatrapara_->tau_gp())
      {
        // artificial diffusion / shock capturing: adaption of diffusion coefficient
        if (scatrapara_->assgd())
        {
          // pre-calculation of stabilization parameter at integration point need for some forms of
          // artificial diffusion
          calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
              reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
              scatravarmanager_->con_vel(k), vol);

          // compute artificial diffusion
          calc_artificial_diff(vol, k, densnp[k], scatravarmanager_->con_vel(k),
              scatravarmanager_->grad_phi(k), scatravarmanager_->conv_phi(k), scatrares, tau[k]);

          // recompute strong residual since now diffus_new = diffus_old + artdiff
          calc_strong_residual(k, scatrares, densam[k], densnp[k], rea_phi, rhsint,
              tau[k]);  // TODO:(Thon) do we really have to do this??
        }

        // calculation of all-scale subgrid diffusivity (by, e.g.,
        // Smagorinsky model) at element center
        if (turbparams_->turb_model() == Inpar::FLUID::smagorinsky or
            turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky or
            turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
        {
          calc_subgr_diff(visc, vol, k, densnp[k]);

          // recompute strong residual since now diffus_new = diffus_old + sgdiff
          calc_strong_residual(k, scatrares, densam[k], densnp[k], rea_phi, rhsint, tau[k]);
        }

        // calculation of fine-scale artificial subgrid diffusivity at element center
        if (turbparams_->fssgd())
          calc_fine_scale_subgr_diff(sgdiff, subgrdiff, ele, vol, k, densnp[k],
              diffmanager_->get_isotropic_diff(k), scatravarmanager_->con_vel(k));

        // calculation of subgrid-scale velocity at integration point if required
        if (scatrapara_->rb_sub_gr_vel())
        {
          // calculation of stabilization parameter related to fluid momentum
          // equation at integration point
          calc_tau(tau[k], visc, 0.0, densnp[k], scatravarmanager_->con_vel(k), vol);
          // calculation of residual-based subgrid-scale velocity
          calc_subgr_velocity(
              ele, sgvelint, densam[k], densnp[k], visc, scatravarmanager_->con_vel(k), tau[k]);

          // calculation of subgrid-scale convective part
          sgconv.multiply_tn(derxy_, sgvelint);
        }

        // (re)compute stabilization parameter at integration point, since diffusion may have
        // changed
        calc_tau(tau[k], diffmanager_->get_isotropic_diff(k),
            reamanager_->get_stabilization_coeff(k, scatravarmanager_->phinp(k)), densnp[k],
            scatravarmanager_->con_vel(k), vol);  // TODO:(Thon) do we really have to do this??
      }

      Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::zero);
      // diffusive term using current scalar value for higher-order elements
      if (use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        get_laplacian_strong_form(diff);
        diff.scale(diffmanager_->get_isotropic_diff(k));
      }

      // prepare multifractal subgrid-scale modeling
      // calculation of model coefficients B (velocity) and D (scalar)
      // at Gauss point as well as calculation
      // of multifractal subgrid-scale quantities
      Core::LinAlg::Matrix<nsd_, 1> mfsgvelint(Core::LinAlg::Initialization::zero);
      double mfsvdiv(0.0);
      double mfssgphi(0.0);
      Core::LinAlg::Matrix<nsd_, 1> mfsggradphi(Core::LinAlg::Initialization::zero);
      if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
      {
        if (turbparams_->bd_gp())
        {
          // calculate model coefficients
          calc_b_and_d_for_multifrac_subgrid_scales(B_mfs, D_mfs, vol, k, densnp[k],
              diffmanager_->get_isotropic_diff(k), visc, scatravarmanager_->con_vel(k), fsvelint);
        }

        // calculate fine-scale velocity, its derivative and divergence for multifractal
        // subgrid-scale modeling
        for (unsigned idim = 0; idim < nsd_; idim++)
          mfsgvelint(idim, 0) = fsvelint(idim, 0) * B_mfs(idim, 0);
        // required for conservative formulation in the context of passive scalar transport
        if (turbparams_->mfs_conservative() or scatrapara_->is_conservative())
        {
          // get divergence of subgrid-scale velocity
          Core::LinAlg::Matrix<nsd_, nsd_> mfsvderxy;
          mfsvderxy.multiply_nt(efsvel_, derxy_);
          for (unsigned idim = 0; idim < nsd_; idim++)
            mfsvdiv += mfsvderxy(idim, idim) * B_mfs(idim, 0);
        }

        // calculate fine-scale scalar and its derivative for multifractal subgrid-scale modeling
        mfssgphi = D_mfs * funct_.dot(fsphinp_[k]);
        mfsggradphi.multiply(derxy_, fsphinp_[k]);
        mfsggradphi.scale(D_mfs);
      }

      //----------------------------------------------------------------
      // standard Galerkin terms
      //----------------------------------------------------------------

      // stabilization parameter and integration factors
      const double taufac = tau[k] * fac;
      const double timefacfac = scatraparatimint_->time_fac() * fac;
      const double timetaufac = scatraparatimint_->time_fac() * taufac;

      //----------------------------------------------------------------
      // 1) element matrix: stationary terms
      //----------------------------------------------------------------

      // calculation of convective element matrix in convective form
      calc_mat_conv(emat, k, timefacfac, densnp[k], sgconv);

      // add conservative contributions
      if (scatrapara_->is_conservative())
        calc_mat_conv_add_cons(emat, k, timefacfac, vdiv, densnp[k]);

      // calculation of diffusive element matrix
      calc_mat_diff(emat, k, timefacfac);

      //----------------------------------------------------------------
      // convective stabilization term
      //----------------------------------------------------------------

      // convective stabilization of convective term (in convective form)
      // transient stabilization of convective term (in convective form)
      if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
        calc_mat_trans_conv_diff_stab(emat, k, timetaufac, densnp[k], sgconv, diff);

      //----------------------------------------------------------------
      // 2) element matrix: instationary terms
      //----------------------------------------------------------------

      if (not scatraparatimint_->is_stationary())
      {
        calc_mat_mass(emat, k, fac, densam[k]);

        if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
          calc_mat_mass_stab(emat, k, taufac, densam[k], densnp[k], sgconv, diff);
      }

      //----------------------------------------------------------------
      // 3) element matrix: reactive term
      //----------------------------------------------------------------

      // including stabilization
      if (reamanager_->active())
        calc_mat_react(emat, k, timefacfac, timetaufac, taufac, densnp[k], sgconv, diff);

      //----------------------------------------------------------------
      // 4) element matrix: chemotactic term
      //----------------------------------------------------------------

      // including stabilization
      calc_mat_chemo(emat, k, timefacfac, timetaufac, densnp[k], scatrares, sgconv, diff);

      //----------------------------------------------------------------
      // 5) element right hand side
      //----------------------------------------------------------------
      //----------------------------------------------------------------
      // computation of bodyforce (and potentially history) term,
      // residual, integration factors and standard Galerkin transient
      // term (if required) on right hand side depending on respective
      // (non-)incremental stationary or time-integration scheme
      //----------------------------------------------------------------
      double rhsfac = scatraparatimint_->time_fac_rhs() * fac;
      double rhstaufac = scatraparatimint_->time_fac_rhs_tau() * taufac;

      if (scatraparatimint_->is_incremental() and not scatraparatimint_->is_stationary())
        calc_rhs_lin_mass(erhs, k, rhsfac, fac, densam[k], densnp[k]);

      // the order of the following three functions is important
      // and must not be changed
      compute_rhs_int(rhsint, densam[k], densnp[k], scatravarmanager_->hist(k));

      recompute_scatra_res_for_rhs(scatrares, k, diff, densn[k], densnp[k], rea_phi, rhsint);

      recompute_conv_phi_for_rhs(k, sgvelint, densnp[k], densn[k], vdiv);

      //----------------------------------------------------------------
      // standard Galerkin transient, old part of rhs and bodyforce term
      //----------------------------------------------------------------
      calc_rhs_hist_and_source(erhs, k, fac, rhsint);

      //----------------------------------------------------------------
      // standard Galerkin terms on right hand side
      //----------------------------------------------------------------

      // convective term
      calc_rhs_conv(erhs, k, rhsfac);

      // diffusive term
      calc_rhs_diff(erhs, k, rhsfac);

      //----------------------------------------------------------------
      // stabilization terms
      //----------------------------------------------------------------
      if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
        calc_rhs_trans_conv_diff_stab(erhs, k, rhstaufac, densnp[k], scatrares, sgconv, diff);

      //----------------------------------------------------------------
      // reactive terms (standard Galerkin and stabilization) on rhs
      //----------------------------------------------------------------

      if (reamanager_->active())
        calc_rhs_react(erhs, k, rhsfac, rhstaufac, rea_phi, densnp[k], scatrares);

      //----------------------------------------------------------------
      // chemotactic terms (standard Galerkin and stabilization) on rhs
      //----------------------------------------------------------------

      calc_rhs_chemo(erhs, k, rhsfac, rhstaufac, scatrares, densnp[k]);

      //----------------------------------------------------------------
      // 6) advanced turbulence models
      //----------------------------------------------------------------

      //----------------------------------------------------------------
      // fine-scale subgrid-diffusivity term on right hand side
      //----------------------------------------------------------------
      if (scatraparatimint_->is_incremental() and turbparams_->fssgd())
        calc_rhsfssgd(erhs, k, rhsfac, sgdiff, fsgradphi);

      //---------------------------------------------------------------
      // multifractal subgrid-scale modeling on right hand side only
      //---------------------------------------------------------------
      if (scatraparatimint_->is_incremental() and
          turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
        calc_rhsmfs(erhs, k, rhsfac, densnp[k], mfsggradphi, mfsgvelint, mfssgphi, mfsvdiv);

      //----------------------------------------------------------------
      // 7) macro-scale matrix and vector contributions arising from
      //    macro-micro coupling in multi-scale simulations
      //----------------------------------------------------------------
      if (ele->material()->material_type() == Core::Materials::m_scatra_multiscale)
        calc_mat_and_rhs_multi_scale(ele, emat, erhs, k, iquad, timefacfac, rhsfac);

    }  // end loop all scalars
  }  // end loop Gauss points
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::body_force(
    const Core::Elements::Element* ele)
{
  std::vector<const Core::Conditions::Condition*> myneumcond;

  // check whether all nodes have a unique Neumann condition
  switch (nsd_ele_)
  {
    case 3:
      Core::Conditions::find_element_conditions(ele, "VolumeNeumann", myneumcond);
      break;
    case 2:
      Core::Conditions::find_element_conditions(ele, "SurfaceNeumann", myneumcond);
      break;
    case 1:
      Core::Conditions::find_element_conditions(ele, "LineNeumann", myneumcond);
      break;
    default:
      FOUR_C_THROW("Illegal number of spatial dimensions: {}", nsd_ele_);
      break;
  }

  if (myneumcond.size() > 1) FOUR_C_THROW("More than one Neumann condition on one node!");

  if (myneumcond.size() == 1)
  {
    // (SPATIAL) FUNCTION BUSINESS
    const auto funct = myneumcond[0]->parameters().get<std::vector<std::optional<int>>>("FUNCT");

    // get values and switches from the condition
    const auto onoff = myneumcond[0]->parameters().get<std::vector<int>>("ONOFF");
    const auto val = myneumcond[0]->parameters().get<std::vector<double>>("VAL");

    // set this condition to the bodyforce array
    for (int idof = 0; idof < numdofpernode_; idof++)
    {
      // function evaluation
      for (unsigned jnode = 0; jnode < nen_; jnode++)
      {
        double functfac = 1.0;

        if (funct[idof].has_value() && funct[idof].value() > 0)
        {
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[idof].value())
                         .evaluate((ele->nodes()[jnode])->x(), scatraparatimint_->time(), idof);
        }

        (bodyforce_[idof])(jnode) = onoff[idof] * val[idof] * functfac;
      }
    }
  }
  else
  {
    for (int idof = 0; idof < numdofpernode_; idof++)
    {
      // no bodyforce
      bodyforce_[idof].clear();
    }
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::other_node_based_source_terms(
    const std::vector<int>& lm, Core::FE::Discretization& discretization,
    Teuchos::ParameterList& params)
{
  // set externally calculated source term instead of body force by volume
  // Neumann boundary condition of input file
  if (turbparams_->scalar_forcing() == Inpar::FLUID::scalarforcing_isotropic)
  {
    // extract additional local values from global vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> source =
        discretization.get_state("forcing");
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*source, bodyforce_, lm);
  }
  // special forcing mean scalar gradient
  else if (turbparams_->scalar_forcing() == Inpar::FLUID::scalarforcing_mean_scalar_gradient)
  {
    // get mean-scalar gradient
    const double grad_phi = params.sublist("TURBULENCE MODEL").get<double>("MEAN_SCALAR_GRADIENT");

    // fill element array
    for (unsigned i = 0; i < nen_; ++i)
    {
      for (int k = 0; k < numdofpernode_; ++k)
      {
        // split for each transported scalar, insert into element arrays
        bodyforce_[k](i, 0) = -grad_phi * evelnp_(2, i);
      }
    }  // for i
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::read_element_coordinates(
    const Core::Elements::Element* ele)
{
  // Directly copy the coordinates since in 3D the transformation is just the identity
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double
Discret::Elements::ScaTraEleCalc<distype, probdim>::eval_shape_func_and_derivs_at_ele_center()
{
  // use one-point Gauss rule to do calculations at the element center
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = eval_shape_func_and_derivs_at_int_point(intpoints_tau, 0);

  return vol;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleCalc<distype, probdim>::eval_shape_func_and_derivs_at_int_point(
    const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (unsigned idim = 0; idim < nsd_ele_; idim++) xsi_(idim) = gpcoord[idim];

  const double det = eval_shape_func_and_derivs_in_parameter_space();

  if (det < 1E-16)
    FOUR_C_THROW("GLOBAL ELEMENT NO. {} \nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", eid_, det);

  // compute global spatial derivatives
  derxy_.multiply(xij_, deriv_);

  // set integration factor: fac = Gauss weight * det(J)
  const double fac = intpoints.ip().qwgt[iquad] * det;

  // compute second global derivatives (if needed)
  if (use2ndderiv_)
  {
    // get global second derivatives
    Core::FE::gder2<distype, nen_, probdim>(xjm_, derxy_, deriv2_, xyze_, derxy2_);
  }
  else
    derxy2_.clear();

  // return integration factor for current GP: fac = Gauss weight * det(J)
  return fac;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double
Discret::Elements::ScaTraEleCalc<distype, probdim>::eval_shape_func_and_derivs_in_parameter_space()
{
  double det = 0.0;

  if (nsd_ == nsd_ele_)  // standard case
  {
    if (not Core::FE::is_nurbs<distype>)
    {
      // shape functions and their first derivatives
      Core::FE::shape_function<distype>(xsi_, funct_);
      Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);
      if (use2ndderiv_)
      {
        // get the second derivatives of standard element at current GP
        Core::FE::shape_function_deriv2<distype>(xsi_, deriv2_);
      }
    }
    else  // nurbs elements are always somewhat special...
    {
      if (use2ndderiv_)
      {
        Core::FE::Nurbs::nurbs_get_funct_deriv_deriv2(
            funct_, deriv_, deriv2_, xsi_, myknots_, weights_, distype);
      }
      else
      {
        Core::FE::Nurbs::nurbs_get_funct_deriv(funct_, deriv_, xsi_, myknots_, weights_, distype);
      }
    }

    // compute Jacobian matrix and determinant
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
    det = xij_.invert(xjm_);
  }
  else  // element dimension is smaller than problem dimension -> manifold
  {
    static Core::LinAlg::Matrix<nsd_ele_, nen_> deriv_red;

    if (not Core::FE::is_nurbs<distype>)
    {
      // shape functions and their first derivatives
      Core::FE::shape_function<distype>(xsi_, funct_);
      Core::FE::shape_function_deriv1<distype>(xsi_, deriv_red);
      if (use2ndderiv_)
      {
        // get the second derivatives of standard element at current GP
        Core::FE::shape_function_deriv2<distype>(xsi_, deriv2_);
      }
    }
    else  // nurbs elements are always somewhat special...
    {
      if (use2ndderiv_)
      {
        Core::FE::Nurbs::nurbs_get_funct_deriv_deriv2(
            funct_, deriv_red, deriv2_, xsi_, myknots_, weights_, distype);
      }
      else
      {
        Core::FE::Nurbs::nurbs_get_funct_deriv(
            funct_, deriv_red, xsi_, myknots_, weights_, distype);
      }
    }

    //! metric tensor at integration point
    static Core::LinAlg::Matrix<nsd_ele_, nsd_ele_> metrictensor;
    static Core::LinAlg::Matrix<nsd_, 1> normalvec;

    // the metric tensor and the area of an infinitesimal surface/line element
    // optional: get unit normal at integration point as well
    const bool throw_error_if_negative_determinant(true);
    Core::FE::compute_metric_tensor_for_boundary_ele<distype, nsd_>(
        xyze_, deriv_red, metrictensor, det, throw_error_if_negative_determinant, &normalvec);

    if (det < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO. {} \nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", eid_, det);

    // transform the derivatives and Jacobians to the higher dimensional coordinates(problem
    // dimension)
    static Core::LinAlg::Matrix<nsd_ele_, nsd_> xjm_red;
    xjm_red.multiply_nt(deriv_red, xyze_);

    for (unsigned i = 0; i < nsd_; ++i)
    {
      for (unsigned j = 0; j < nsd_ele_; ++j) xjm_(j, i) = xjm_red(j, i);
      xjm_(nsd_ele_, i) = normalvec(i, 0);
    }

    for (unsigned i = 0; i < nen_; ++i)
    {
      for (unsigned j = 0; j < nsd_ele_; ++j) deriv_(j, i) = deriv_red(j, i);
      deriv_(nsd_ele_, i) = 0.0;
    }

    // special case: 1D element embedded in 3D problem
    if (nsd_ele_ == 1 and nsd_ == 3)
    {
      // compute second unit normal
      const double normalvec2_0 = xjm_red(0, 1) * normalvec(2, 0) - normalvec(1, 0) * xjm_red(0, 2);
      const double normalvec2_1 = xjm_red(0, 2) * normalvec(0, 0) - normalvec(2, 0) * xjm_red(0, 0);
      const double normalvec2_2 = xjm_red(0, 0) * normalvec(1, 0) - normalvec(0, 0) * xjm_red(0, 1);

      // norm
      const double norm2 = std::sqrt(
          normalvec2_0 * normalvec2_0 + normalvec2_1 * normalvec2_1 + normalvec2_2 * normalvec2_2);

      xjm_(2, 0) = normalvec2_0 / norm2;
      xjm_(2, 1) = normalvec2_1 / norm2;
      xjm_(2, 2) = normalvec2_2 / norm2;

      for (unsigned i = 0; i < nen_; i++) deriv_(2, i) = 0.0;
    }

    xij_.invert(xjm_);
  }

  // modify Jacobian determinant in case of spherical coordinates
  if (scatrapara_->spherical_coords())
  {
    static Core::LinAlg::Matrix<nsd_, 1> xyzint;

    // evaluate radial coordinate
    xyzint.multiply(xyze_, funct_);

    // multiply standard Jacobian determinant by square of radial coordinate and 4 pi
    constexpr double four_pi = 4.0 * std::numbers::pi;
    det *= xyzint(0) * xyzint(0) * four_pi;
  }

  return det;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele, std::vector<double>& densn, std::vector<double>& densnp,
    std::vector<double>& densam, double& visc, const int iquad)
{
  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() < numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
    materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::materials(
    const std::shared_ptr<const Core::Mat::Material> material, const int k, double& densn,
    double& densnp, double& densam, double& visc, const int iquad)
{
  switch (material->material_type())
  {
    case Core::Materials::m_electrode:
    {
      // safety check
      if (k != 0) FOUR_C_THROW("Invalid species ID!");

      mat_electrode(material);
      break;
    }

    case Core::Materials::m_scatra:
    {
      mat_scatra(material, k, densn, densnp, densam, visc, iquad);
      break;
    }

    case Core::Materials::m_scatra_multiscale:
    {
      mat_scatra_multi_scale(material, densn, densnp, densam);
      break;
    }

    default:
    {
      FOUR_C_THROW("Material type {} is not supported!", material->material_type());
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::mat_scatra(
    const std::shared_ptr<const Core::Mat::Material> material, const int k, double& densn,
    double& densnp, double& densam, double& visc, const int iquad)
{
  const std::shared_ptr<const Mat::ScatraMat>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMat>(material);

  // get constant diffusivity
  diffmanager_->set_isotropic_diff(actmat->diffusivity(), k);

  // get reaction coefficient
  reamanager_->set_rea_coeff(actmat->rea_coeff(), k);

  // in case of multifractal subgrid-scales, read Schmidt number
  if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales or
      scatrapara_->rb_sub_gr_vel() or
      turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
  {
    // access fluid discretization
    std::shared_ptr<Core::FE::Discretization> fluiddis = nullptr;
    fluiddis = Global::Problem::instance()->get_dis("fluid");
    // get corresponding fluid element (it has the same global ID as the scatra element)
    Core::Elements::Element* fluidele = fluiddis->g_element(eid_);
    if (fluidele == nullptr)
      FOUR_C_THROW("Fluid element {} not on local processor", eid_);
    else
    {
      // get fluid material
      std::shared_ptr<Core::Mat::Material> fluidmat = fluidele->material();
      if (fluidmat->material_type() != Core::Materials::m_fluid)
        FOUR_C_THROW("Invalid fluid material for passive scalar transport in turbulent flow!");

      const std::shared_ptr<const Mat::NewtonianFluid>& actfluidmat =
          std::dynamic_pointer_cast<const Mat::NewtonianFluid>(fluidmat);

      // get constant dynamic viscosity
      visc = actfluidmat->viscosity();
      densn = actfluidmat->density();
      densnp = actfluidmat->density();
      densam = actfluidmat->density();

      if (densam != 1.0 or densnp != 1.0 or densn != 1.0)
        FOUR_C_THROW("Check your parameters! Read comment!");
      // For all implementations, dens=1.0 is assumed, in particular for
      // multifractal_subgrid_scales. Hence, visc and diffus are kinematic quantities. Using
      // dens!=1.0 should basically work, but you should check it before application.
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::mat_scatra_multi_scale(
    const std::shared_ptr<const Core::Mat::Material> material, double& densn, double& densnp,
    double& densam) const
{
  // safety check
  if (numscal_ > 1)
    FOUR_C_THROW("Multi-scale scalar transport only implemented for one transported scalar!");

  // extract multi-scale scalar transport material
  const auto* matmultiscale = static_cast<const Mat::ScatraMultiScale*>(material.get());

  // set densities equal to porosity
  densn = densnp = densam = matmultiscale->porosity();

  // set effective diffusion coefficient in diffusion manager
  // effective diffusion coefficient = intrinsic diffusion coefficient * porosity / tortuosity
  diffmanager_->set_isotropic_diff(
      matmultiscale->diffusivity() * matmultiscale->porosity() / matmultiscale->tortuosity(), 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::mat_electrode(
    const std::shared_ptr<const Core::Mat::Material> material)
{
  // set constant diffusivity
  diffmanager_->set_isotropic_diff(
      std::static_pointer_cast<const Mat::Electrode>(material)
          ->compute_diffusion_coefficient_concentration_dependent(scatravarmanager_->phinp(0)),
      0);
}

/*---------------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::get_laplacian_strong_form(
    Core::LinAlg::Matrix<nen_, 1>& diff)
{
  diff.clear();
  // compute N,xx  +  N,yy +  N,zz for each shape function at integration point
  for (unsigned i = 0; i < nen_; ++i)
  {
    for (unsigned j = 0; j < nsd_; ++j)
    {
      diff(i) += derxy2_(j, i);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::get_divergence(
    double& vdiv, const Core::LinAlg::Matrix<nsd_, nen_>& evel)
{
  Core::LinAlg::Matrix<nsd_, nsd_> vderxy;
  vderxy.multiply_nt(evel, derxy_);

  vdiv = 0.0;
  // compute vel x,x  + vel y,y +  vel z,z at integration point
  for (unsigned j = 0; j < nsd_; ++j)
  {
    vdiv += vderxy(j, j);
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::get_rhs_int(
    double& rhsint, const double densnp, const int k)
{
  // compute rhs containing bodyforce (divided by specific heat capacity) and,
  // for temperature equation, the time derivative of thermodynamic pressure,
  // if not constant, and for temperature equation of a reactive
  // equation system, the reaction-rate term
  rhsint = bodyforce_[k].dot(funct_);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_conv(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double densnp, const Core::LinAlg::Matrix<nen_, 1>& sgconv)
{
  const Core::LinAlg::Matrix<nen_, 1>& conv = scatravarmanager_->conv(k);

  // convective term in convective form
  const double densfac = timefacfac * densnp;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = densfac * funct_(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * (conv(ui) + sgconv(ui));
    }
  }
}

/*------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_conv_add_cons(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac, const double vdiv,
    const double densnp)
{
  const double consfac = timefacfac * densnp * vdiv;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = consfac * funct_(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * funct_(ui);
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_diff(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac)
{
  // diffusive term
  const double fac_diffus = timefacfac * diffmanager_->get_isotropic_diff(k);
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;
      double laplawf(0.0);
      get_laplacian_weak_form(laplawf, ui, vi);
      emat(fvi, fui) += fac_diffus * laplawf;
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_trans_conv_diff_stab(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timetaufac,
    const double densnp, const Core::LinAlg::Matrix<nen_, 1>& sgconv,
    const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  const Core::LinAlg::Matrix<nen_, 1>& conv = scatravarmanager_->conv(k);

  const double dens2taufac = timetaufac * densnp * densnp;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = dens2taufac * (conv(vi) + sgconv(vi) +
                                       scatrapara_->usfemgls_fac() * 1.0 /
                                           scatraparatimint_->time_fac() * funct_(vi));
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * conv(ui);
    }
  }

  //----------------------------------------------------------------
  // stabilization terms for higher-order elements
  //----------------------------------------------------------------
  if (use2ndderiv_)
  {
    const double denstaufac = timetaufac * densnp;
    // convective stabilization of diffusive term (in convective form)
    // transient stabilization of diffusive term (in convective form)
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = denstaufac * (conv(vi) + sgconv(vi) +
                                        scatrapara_->usfemgls_fac() * 1.0 /
                                            scatraparatimint_->time_fac() * funct_(vi));
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) -= v * diff(ui);
      }
    }

    const double densdifftaufac = scatrapara_->usfemgls_fac() * denstaufac;
    // diffusive stabilization of convective term (in convective form)
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = densdifftaufac * diff(vi);
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) -= v * conv(ui);
      }
    }

    const double difftaufac = scatrapara_->usfemgls_fac() * timetaufac;
    // diffusive stabilization of diffusive term
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = difftaufac * diff(vi);
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) += v * diff(ui);
      }
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_mass(
    Core::LinAlg::SerialDenseMatrix& emat, const int& k, const double& fac, const double& densam)
{
  calc_mat_mass(emat, k, fac, densam, funct_, funct_);
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_mass(
    Core::LinAlg::SerialDenseMatrix& emat, const int& k, const double& fac, const double& densam,
    const Core::LinAlg::Matrix<nen_, 1>& sfunct, const Core::LinAlg::Matrix<nen_, 1>& tfunct) const
{
  const double densamfac = fac * densam;
  //----------------------------------------------------------------
  // standard Galerkin transient term
  //----------------------------------------------------------------
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = densamfac * tfunct(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * sfunct(ui);
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_mass_stab(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double taufac, const double densam,
    const double densnp, const Core::LinAlg::Matrix<nen_, 1>& sgconv,
    const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  const Core::LinAlg::Matrix<nen_, 1>& conv = scatravarmanager_->conv(k);
  const double densamnptaufac = taufac * densam * densnp;
  //----------------------------------------------------------------
  // stabilization of transient term
  //----------------------------------------------------------------
  // convective stabilization of transient term (in convective form)
  // transient stabilization of transient term
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = densamnptaufac * (conv(vi) + sgconv(vi) +
                                          scatrapara_->usfemgls_fac() * 1.0 /
                                              scatraparatimint_->time_fac() * funct_(vi));
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * funct_(ui);
    }
  }

  if (use2ndderiv_)
  {
    const double densamreataufac = scatrapara_->usfemgls_fac() * taufac * densam;
    // diffusive stabilization of transient term
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = densamreataufac * diff(vi);
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) -= v * funct_(ui);
      }
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_react(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double timetaufac, const double taufac, const double densnp,
    const Core::LinAlg::Matrix<nen_, 1>& sgconv, const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  const Core::LinAlg::Matrix<nen_, 1>& conv = scatravarmanager_->conv(k);

  // NOTE: it is important that the reaction coefficient reamanager_->GetReaCoeff(k) does not depend
  // on ANY concentrations.
  const double fac_reac = timefacfac * densnp * reamanager_->get_rea_coeff(k);
  const double timetaufac_reac = timetaufac * densnp * reamanager_->get_rea_coeff(k);

  //----------------------------------------------------------------
  // standard Galerkin reactive term
  //----------------------------------------------------------------
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = fac_reac * funct_(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * numdofpernode_ + k;

      emat(fvi, fui) += v * funct_(ui);
    }
  }

  //----------------------------------------------------------------
  // stabilization of reactive term
  //----------------------------------------------------------------
  if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
  {
    double densreataufac = timetaufac_reac * densnp;
    // convective stabilization of reactive term (in convective form)
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = densreataufac * (conv(vi) + sgconv(vi) +
                                           scatrapara_->usfemgls_fac() * 1.0 /
                                               scatraparatimint_->time_fac() * funct_(vi));
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) += v * funct_(ui);
      }
    }

    if (use2ndderiv_)
    {
      // diffusive stabilization of reactive term
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const double v = scatrapara_->usfemgls_fac() * timetaufac_reac * diff(vi);
        const int fvi = vi * numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * numdofpernode_ + k;

          emat(fvi, fui) -= v * funct_(ui);
        }
      }
    }

    //----------------------------------------------------------------
    // reactive stabilization
    //----------------------------------------------------------------
    densreataufac = scatrapara_->usfemgls_fac() * timetaufac_reac * densnp;

    // reactive stabilization of convective (in convective form) and reactive term
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = densreataufac * funct_(vi);
      const int fvi = vi * numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * numdofpernode_ + k;

        emat(fvi, fui) += v * (conv(ui) + reamanager_->get_rea_coeff(k) * funct_(ui));
      }
    }

    if (use2ndderiv_)
    {
      // reactive stabilization of diffusive term
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const double v = scatrapara_->usfemgls_fac() * timetaufac_reac * funct_(vi);
        const int fvi = vi * numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * numdofpernode_ + k;

          emat(fvi, fui) -= v * diff(ui);
        }
      }
    }


    if (not scatraparatimint_->is_stationary())
    {
      // reactive stabilization of transient term
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const double v = scatrapara_->usfemgls_fac() * taufac * densnp *
                         reamanager_->get_rea_coeff(k) * densnp * funct_(vi);
        const int fvi = vi * numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * numdofpernode_ + k;

          emat(fvi, fui) += v * funct_(ui);
        }
      }

      if (use2ndderiv_ and reamanager_->get_rea_coeff(k) != 0.0)
        FOUR_C_THROW("Second order reactive stabilization is not fully implemented!! ");
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_lin_mass(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac, const double fac,
    const double densam, const double densnp)
{
  const double& phinp = scatravarmanager_->phinp(k);
  const double& hist = scatravarmanager_->hist(k);

  double vtrans = 0.0;

  if (scatraparatimint_->is_gen_alpha())
    vtrans = rhsfac * densam * hist;
  else
  {
    // compute scalar at integration point
    vtrans = fac * densnp * phinp;
  }

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    erhs[fvi] -= vtrans * funct_(vi);
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::compute_rhs_int(
    double& rhsint, const double densam, const double densnp, const double hist)
{
  if (scatraparatimint_->is_gen_alpha())
  {
    if (not scatraparatimint_->is_incremental())
      rhsint += densam * hist * (scatraparatimint_->alpha_f() / scatraparatimint_->time_fac());

    rhsint *= (scatraparatimint_->time_fac() / scatraparatimint_->alpha_f());
  }
  else  // OST, BDF2, stationary
  {
    if (not scatraparatimint_->is_stationary())  // OST, BDF2
    {
      rhsint *= scatraparatimint_->time_fac();
      rhsint += densnp * hist;
    }
  }
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::recompute_scatra_res_for_rhs(
    double& scatrares, const int k, const Core::LinAlg::Matrix<nen_, 1>& diff, const double densn,
    const double densnp, double& rea_phi, const double rhsint)
{
  const Core::LinAlg::Matrix<nsd_, 1>& convelint = scatravarmanager_->con_vel(k);
  const double& phin = scatravarmanager_->phin(k);

  if (scatraparatimint_->is_gen_alpha())
  {
    if (not scatraparatimint_->is_incremental())
    {
      // for this case, gradphi_ (i.e. the gradient
      // at time n+1) is overwritten by the gradient at time n
      // analogously, conv_phi_ at time n+1 is replace by its
      // value at time n
      // gradient of scalar value at n
      Core::LinAlg::Matrix<nsd_, 1> gradphi;
      gradphi.multiply(derxy_, ephin_[k]);

      // convective term using scalar value at n
      double conv_phi = convelint.dot(gradphi);

      // diffusive term using current scalar value for higher-order elements
      double diff_phin = 0.0;
      if (use2ndderiv_) diff_phin = diff.dot(ephin_[k]);

      // reactive term using scalar value at n
      // if no reaction is chosen, GetReaCoeff(k) returns 0.0
      rea_phi = densnp * reamanager_->get_rea_coeff(k) * phin;
      // reacterm_[k] must be evaluated at t^n to be used in the line above!

      scatrares = (1.0 - scatraparatimint_->alpha_f()) * (densn * conv_phi - diff_phin + rea_phi) -
                  rhsint * scatraparatimint_->alpha_f() / scatraparatimint_->time_fac();

      scatravarmanager_->set_grad_phi(k, gradphi);
      scatravarmanager_->set_conv_phi(k, conv_phi);
    }
  }
  else if (scatraparatimint_->is_incremental() and not scatraparatimint_->is_gen_alpha())
  {
    if (not scatraparatimint_->is_stationary()) scatrares *= scatraparatimint_->dt();
  }
  else
    scatrares = -rhsint;
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::recompute_conv_phi_for_rhs(const int k,
    const Core::LinAlg::Matrix<nsd_, 1>& sgvelint, const double densnp, const double densn,
    const double vdiv)
{
  double conv_phi = 0.0;
  const double& phinp = scatravarmanager_->phinp(k);
  const double& phin = scatravarmanager_->phin(k);
  const Core::LinAlg::Matrix<nsd_, 1>& gradphi = scatravarmanager_->grad_phi(k);


  if (scatraparatimint_->is_incremental())
  {
    // addition to convective term due to subgrid-scale velocity
    // (not included in residual)
    double sgconv_phi = sgvelint.dot(gradphi);
    conv_phi += sgconv_phi;

    // addition to convective term for conservative form
    // (not included in residual)
    if (scatrapara_->is_conservative())
    {
      // convective term in conservative form
      conv_phi += phinp * vdiv;
    }

    // multiply convective term by density
    conv_phi *= densnp;

    // multiply convective term by density
    scatravarmanager_->scale_conv_phi(k, densnp);
    // addition to convective term due to subgrid-scale velocity
    scatravarmanager_->add_to_conv_phi(k, conv_phi);
  }
  else if (not scatraparatimint_->is_incremental() and scatraparatimint_->is_gen_alpha())
  {
    // addition to convective term due to subgrid-scale velocity
    // (not included in residual)
    double sgconv_phi = sgvelint.dot(gradphi);
    conv_phi += sgconv_phi;

    // addition to convective term for conservative form
    // (not included in residual)
    if (scatrapara_->is_conservative())
    {
      // convective term in conservative form
      // caution: velocity divergence is for n+1 and not for n!
      // -> hopefully, this inconsistency is of small amount
      conv_phi += phin * vdiv;
    }

    // multiply convective term by density
    conv_phi *= densn;

    // multiply convective term by density
    scatravarmanager_->scale_conv_phi(k, densn);
    // addition to convective term due to subgrid-scale velocity
    scatravarmanager_->add_to_conv_phi(k, conv_phi);
  }
}

/*-------------------------------------------------------------------------------------- *
 *---------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_hist_and_source(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double fac, const double rhsint)
{
  double vrhs = fac * rhsint;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    erhs[fvi] += vrhs * funct_(vi);
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_conv(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac)
{
  const double& conv_phi = scatravarmanager_->conv_phi(k);

  double vrhs = rhsfac * conv_phi;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    erhs[fvi] -= vrhs * funct_(vi);
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_diff(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac)
{
  const Core::LinAlg::Matrix<nsd_, 1>& gradphi = scatravarmanager_->grad_phi(k);

  double vrhs = rhsfac * diffmanager_->get_isotropic_diff(k);

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, gradphi, vi);
    erhs[fvi] -= vrhs * laplawf;
  }
}

/*--------------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_trans_conv_diff_stab(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhstaufac, const double densnp,
    const double scatrares, const Core::LinAlg::Matrix<nen_, 1>& sgconv,
    const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  const Core::LinAlg::Matrix<nen_, 1>& conv = scatravarmanager_->conv(k);

  // convective rhs stabilization (in convective form)
  double vrhs = rhstaufac * scatrares * densnp;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    erhs[fvi] -=
        vrhs * (conv(vi) + sgconv(vi) +
                   scatrapara_->usfemgls_fac() * 1.0 / scatraparatimint_->time_fac() * funct_(vi));
  }

  // diffusive rhs stabilization
  if (use2ndderiv_)
  {
    vrhs = rhstaufac * scatrares;
    // diffusive stabilization of convective temporal rhs term (in convective form)
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;

      erhs[fvi] += scatrapara_->usfemgls_fac() * vrhs * diff(vi);
    }
  }
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhs_react(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac, const double rhstaufac,
    const double rea_phi, const double densnp, const double scatrares)
{
  // standard Galerkin term
  double vrhs = rhsfac * rea_phi;

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    erhs[fvi] -= vrhs * funct_(vi);
  }

  // reactive rhs stabilization
  if (scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
  {
    vrhs = scatrapara_->usfemgls_fac() * rhstaufac * densnp * reamanager_->get_rea_coeff(k) *
           scatrares;
    // TODO: this is not totally correct since GetReaCoeff(k) can depend on phinp(k)...
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;

      erhs[fvi] -= vrhs * funct_(vi);
    }
  }
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhsfssgd(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac, const double sgdiff,
    const Core::LinAlg::Matrix<nsd_, 1> fsgradphi)
{
  const double vrhs = rhsfac * sgdiff;
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, fsgradphi, vi);
    erhs[fvi] -= (vrhs * laplawf);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_rhsmfs(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac, const double densnp,
    const Core::LinAlg::Matrix<nsd_, 1> mfsggradphi, const Core::LinAlg::Matrix<nsd_, 1> mfsgvelint,
    const double mfssgphi, const double mfsvdiv)
{
  const double& phinp = scatravarmanager_->phinp(k);
  const Core::LinAlg::Matrix<nsd_, 1>& gradphi = scatravarmanager_->grad_phi(k);
  const Core::LinAlg::Matrix<nsd_, 1>& convelint = scatravarmanager_->con_vel(k);

  if (nsd_ < 3) FOUR_C_THROW("Turbulence is 3D!");
  // fixed-point iteration only (i.e. beta=0.0 assumed), cf
  // turbulence part in evaluate()
  {
    double cross = convelint.dot(mfsggradphi) + mfsgvelint.dot(gradphi);
    double reynolds = mfsgvelint.dot(mfsggradphi);

    // conservative formulation
    double conserv = 0.0;
    if (turbparams_->mfs_conservative() or scatrapara_->is_conservative())
    {
      double convdiv = 0.0;
      get_divergence(convdiv, econvelnp_);

      conserv = mfssgphi * convdiv + phinp * mfsvdiv + mfssgphi * mfsvdiv;
    }

    const double vrhs = rhsfac * densnp * (cross + reynolds + conserv);

    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;
      erhs[fvi] -= vrhs * funct_(vi);
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_and_rhs_multi_scale(
    const Core::Elements::Element* const ele, Core::LinAlg::SerialDenseMatrix& emat,
    Core::LinAlg::SerialDenseVector& erhs, const int k, const int iquad, const double timefacfac,
    const double rhsfac)
{
  // extract multi-scale scalar transport material
  const Mat::ScatraMultiScale* matmultiscale =
      static_cast<const Mat::ScatraMultiScale*>(ele->material().get());

  // initialize variables for micro-scale coupling flux and derivative of micro-scale coupling flux
  // w.r.t. macro-scale state variable
  double q_micro(0.0);
  std::vector<double> dq_dphi_micro(1, 0.0);

  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  const double detF = eval_det_f_at_int_point(ele, intpoints, iquad);

  // evaluate multi-scale scalar transport material
  matmultiscale->evaluate(
      iquad, std::vector<double>(1, scatravarmanager_->phinp(k)), q_micro, dq_dphi_micro, detF);

  // macro-scale matrix contribution
  const double matrixterm =
      timefacfac * dq_dphi_micro[0] * matmultiscale->specific_micro_scale_surface_area(detF);
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = funct_(vi) * matrixterm;
    const int fvi = vi * numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui) emat(fvi, ui * numdofpernode_ + k) += v * funct_(ui);
  }

  // macro-scale vector contribution
  const double rhsterm = rhsfac * q_micro * matmultiscale->specific_micro_scale_surface_area(detF);
  for (unsigned vi = 0; vi < nen_; ++vi) erhs[vi * numdofpernode_ + k] -= funct_(vi) * rhsterm;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::set_internal_variables_for_mat_and_rhs()
{
  scatravarmanager_->set_internal_variables(
      funct_, derxy_, ephinp_, ephin_, econvelnp_, ehist_, eforcevelocity_);
}

FOUR_C_NAMESPACE_CLOSE

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
#include "4C_scatra_ele_calc.inst.hpp"
