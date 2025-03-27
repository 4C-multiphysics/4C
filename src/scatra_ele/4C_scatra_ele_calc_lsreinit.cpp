// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_lsreinit.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_geometry_integrationcell_coordtrafo.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_scatra_ele_parameter_lsreinit.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"
#include "4C_utils_singleton_owner.hpp"

#define USE_PHIN_FOR_VEL

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>*
Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcLsReinit<distype, prob_dim>>(
            new ScaTraEleCalcLsReinit<distype, prob_dim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::ScaTraEleCalcLsReinit(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype, prob_dim>::ScaTraEleCalc(
          numdofpernode, numscal, disname),
      ephizero_(my::numscal_),  // size of vector
      lsreinitparams_(
          Discret::Elements::ScaTraEleParameterLsReinit::instance(disname))  // parameter class
{
  // set appropriate diffusion manager
  my::diffmanager_ = std::make_shared<ScaTraEleDiffManagerLsReinit<nsd_>>(my::numscal_);
  // set appropriate internal variable manager
  my::scatravarmanager_ =
      std::make_shared<ScaTraEleInternalVariableManagerLsReinit<nsd_, nen_>>(my::numscal_);

  // safety checks
  if (my::scatrapara_->rb_sub_gr_vel())
    FOUR_C_THROW("CalcSubgrVelocityLevelSet not available anymore");
  if (lsreinitparams_->art_diff() != Inpar::ScaTra::artdiff_none)
  {
    if (not my::scatrapara_->mat_gp() or not my::scatrapara_->tau_gp())
      FOUR_C_THROW(
          "Evaluation of material and stabilization parameters need to be done at the integration "
          "points for reinitialization due to artificial diff!");
  }

  return;
}


/*----------------------------------------------------------------------*
 | Action type: Evaluate                                rasthofer 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
int Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::evaluate(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // setup
  if (setup_calc(ele, discretization) == -1) return 0;

  //(for now) only first dof set considered
  const std::vector<int>& lm = la[0].lm_;

  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);

  eval_reinitialization(*phinp, lm, ele, params, discretization, elemat1_epetra, elevec1_epetra);

  return 0;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::eval_reinitialization(
    const Core::LinAlg::Vector<double>& phinp, const std::vector<int>& lm,
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra)
{
  // --- standard case --------------------------------------------------------
  if (prob_dim == this->nsd_ele_)
  {
    eval_reinitialization_std(
        phinp, lm, ele, params, discretization, elemat1_epetra, elevec1_epetra);
  }
  // --- embedded case --------------------------------------------------------
  else
  {
    eval_reinitialization_embedded(lm, ele, params, discretization, elemat1_epetra, elevec1_epetra);
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::eval_reinitialization_embedded(
    const std::vector<int>& lm, Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra)
{
  // distinguish reinitialization
  switch (lsreinitparams_->reinit_type())
  {
    case Inpar::ScaTra::reinitaction_ellipticeq:
    {
      // ----------------------------------------------------------------------
      // extract boundary integration cells for elliptic reinitialization
      // ----------------------------------------------------------------------
      // define empty list
      Core::Geo::BoundaryIntCellPtrs boundaryIntCells = Core::Geo::BoundaryIntCellPtrs(0);

      // check the type: ToDo change to FOUR_C_ASSERT
      if (not params.INVALID_TEMPLATE_QUALIFIER
              isType<std::shared_ptr<const std::map<int, Core::Geo::BoundaryIntCellPtrs>>>(
                  "boundary cells"))
        FOUR_C_THROW("The given boundary cells have the wrong type!");

      const std::shared_ptr<const std::map<int, Core::Geo::BoundaryIntCellPtrs>>& allcells =
          params.get<std::shared_ptr<const std::map<int, Core::Geo::BoundaryIntCellPtrs>>>(
              "boundary cells", nullptr);

      // ----------------------------------------------------------------------
      // check if the current element is a cut element
      // ----------------------------------------------------------------------
      std::map<int, Core::Geo::BoundaryIntCellPtrs>::const_iterator cit = allcells->find(my::eid_);
      if (cit != allcells->end()) boundaryIntCells = cit->second;

      Core::LinAlg::Matrix<nen_, 1> el2sysmat_diag_inv(
          Core::LinAlg::Initialization::leave_uninitialized);
      if (lsreinitparams_->project())
      {
        FOUR_C_THROW(
            "Currently unsupported, since the variation of the "
            "l2-projected gradient is missing. -- hiermeier 12/2016");
        const std::shared_ptr<Core::LinAlg::MultiVector<double>>& gradphi =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("gradphi");
        Core::FE::extract_my_node_based_values(ele, my::econvelnp_, *gradphi, nsd_);
        std::shared_ptr<const Core::LinAlg::Vector<double>> l2_proj_sys_diag =
            discretization.get_state("l2_proj_system_mat_diag");
        if (!l2_proj_sys_diag) FOUR_C_THROW("Could not find the l2 projection system diagonal!");
        Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(
            *l2_proj_sys_diag, el2sysmat_diag_inv, lm);
        el2sysmat_diag_inv.reciprocal(el2sysmat_diag_inv);
      }
      else
      {
        std::fill(el2sysmat_diag_inv.data(), el2sysmat_diag_inv.data() + nen_, 1.0);
      }

      // get action
      const auto action = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");
      // ----------------------------------------------------------------------
      // calculate element coefficient matrix and/or rhs
      // ----------------------------------------------------------------------
      switch (action)
      {
        case ScaTra::Action::calc_mat_and_rhs:
        {
          elliptic_newton_system(
              &elemat1_epetra, &elevec1_epetra, el2sysmat_diag_inv, boundaryIntCells);
          break;
        }
        case ScaTra::Action::calc_rhs:
        {
          elliptic_newton_system(nullptr, &elevec1_epetra, el2sysmat_diag_inv, boundaryIntCells);
          break;
        }
        case ScaTra::Action::calc_mat:
        {
          elliptic_newton_system(&elemat1_epetra, nullptr, el2sysmat_diag_inv, boundaryIntCells);
          break;
        }
        default:
          FOUR_C_THROW("Unsupported action!");
          exit(EXIT_FAILURE);
      }
      break;
    }
    default:
      FOUR_C_THROW("Unsupported reinitialization equation for the embedded case!");
      exit(EXIT_FAILURE);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::elliptic_newton_system(
    Core::LinAlg::SerialDenseMatrix* emat, Core::LinAlg::SerialDenseVector* erhs,
    const Core::LinAlg::Matrix<nen_, 1>& el2sysmat_diag_inv,
    const Core::Geo::BoundaryIntCellPtrs& bcell)
{
  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    //--------------------------------------------------------------------
    // set all Gauss point quantities
    //--------------------------------------------------------------------

    // gradient of current scalar value at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphinp(Core::LinAlg::Initialization::set_zero);
    if (lsreinitparams_->project())
      gradphinp.multiply(my::econvelnp_, my::funct_);
    else
      gradphinp.multiply(my::derxy_, my::ephinp_[0]);

    double normgradphi = gradphinp.norm2();

    //----------------------------------------------------------------------
    // prepare diffusion manager
    //----------------------------------------------------------------------

    // calculate nonlinear diffusivity
    double diff_rhs = 0.0;
    double diff_mat = 0.0;
    switch (lsreinitparams_->diff_fct())
    {
      case Inpar::ScaTra::hyperbolic:
      {
        // the basic form: goes to -infinity, if norm(grad)=1
        // yields directly desired signed-distance field
        if (normgradphi < 1.0e-8)
          FOUR_C_THROW(
              " The gradient l2-norm is smaller than 1.0e-8! "
              "( value = {} )",
              normgradphi);

        double inv_normgradphi = 1.0 / normgradphi;
        diff_rhs = 1.0 - (inv_normgradphi);
        diff_mat = inv_normgradphi * inv_normgradphi * inv_normgradphi;

        break;
      }
      default:
      {
        FOUR_C_THROW("Unsupported diffusivity function!");
        break;
      }
    }

    //----------------------------------------------------------------
    // evaluation of matrix and rhs
    //----------------------------------------------------------------

    //----------------------------------------------------------------
    // 1) element matrix: diffusion tangential matrix
    //----------------------------------------------------------------
    if (emat)
    {
      // set diffusivity
      my::diffmanager_->set_isotropic_diff(1.0, 0);

      int k = 0;
      // calculate the outer product
      Core::LinAlg::Matrix<nsd_, nsd_> mat;
      mat.multiply_nt(gradphinp, gradphinp);
      mat.scale(diff_mat);
      // add a scalar value to the diagonal of mat
      for (unsigned d = 0; d < nsd_; ++d) mat(d, d) += diff_rhs;

      // diffusive term
      const double fac_diff = fac * my::diffmanager_->get_isotropic_diff(k);
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * my::numdofpernode_ + k;
          double laplawf(0.0);
          my::get_laplacian_weak_form(laplawf, mat, fui, fvi);

          (*emat)(fvi, fui) += fac_diff * laplawf;
        }
      }
    }

    //----------------------------------------------------------------
    // 2) element right hand side
    //----------------------------------------------------------------
    if (erhs)
    {
      // set diffusivity for rhs term
      my::diffmanager_->set_isotropic_diff(diff_rhs, 0);
      // set gradphi for rhs term
      my::scatravarmanager_->set_grad_phi(0, gradphinp);

      my::calc_rhs_diff(*erhs, 0, -fac);
    }
  }  // end: loop all Gauss points

  //----------------------------------------------------------------
  // 3) evaluation of penalty term at initial interface
  //----------------------------------------------------------------

  evaluate_interface_term(emat, erhs, bcell);

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::eval_reinitialization_std(
    const Core::LinAlg::Vector<double>& phinp, const std::vector<int>& lm,
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra)
{
  // distinguish reinitialization
  switch (lsreinitparams_->reinit_type())
  {
    case Inpar::ScaTra::reinitaction_sussman:
    {
      // extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
      std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
      std::shared_ptr<const Core::LinAlg::Vector<double>> phizero =
          discretization.get_state("phizero");
      if (hist == nullptr || phin == nullptr || phizero == nullptr)
        FOUR_C_THROW("Cannot get state vector 'hist' and/or 'phin' and/or 'phizero'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phin, my::ephin_, lm);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phizero, ephizero_, lm);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*hist, my::ehist_, lm);

      if (lsreinitparams_->use_projected_vel())
      {
        // get velocity at nodes (pre-computed via L2 projection)
        const std::shared_ptr<Core::LinAlg::MultiVector<double>> velocity =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "reinitialization velocity field");
        Core::FE::extract_my_node_based_values(ele, my::econvelnp_, *velocity, nsd_);
      }

      // calculate element coefficient matrix and rhs
      sysmat_hyperbolic(elemat1_epetra, elevec1_epetra);
      break;
    }
    case Inpar::ScaTra::reinitaction_ellipticeq:
    {
      // extract boundary integration cells for elliptic reinitialization
      // define empty list
      Core::Geo::BoundaryIntCellPtrs boundaryIntCells = Core::Geo::BoundaryIntCellPtrs(0);

      std::shared_ptr<std::map<int, Core::Geo::BoundaryIntCells>> allcells =
          params.get<std::shared_ptr<std::map<int, Core::Geo::BoundaryIntCells>>>(
              "boundary cells", nullptr);

      std::map<int, Core::Geo::BoundaryIntCells>::iterator cit_map = allcells->find(my::eid_);
      if (cit_map != allcells->end())
      {
        boundaryIntCells.reserve(cit_map->second.size());
        Core::Geo::BoundaryIntCells::iterator cit_vec;
        for (cit_vec = cit_map->second.begin(); cit_vec != cit_map->second.end(); ++cit_vec)
        {
          boundaryIntCells.push_back(Core::Utils::shared_ptr_from_ref((*cit_vec)));
        }
      }

      if (lsreinitparams_->project())
      {
        const std::shared_ptr<Core::LinAlg::MultiVector<double>> gradphi =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("gradphi");
        Core::FE::extract_my_node_based_values(ele, my::econvelnp_, *gradphi, nsd_);
      }

      // calculate element coefficient matrix and rhs
      sysmat_elliptic(elemat1_epetra, elevec1_epetra, boundaryIntCells);
      break;
    }
    default:
      FOUR_C_THROW("Unknown reinitialization equation");
      break;
  }
}


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)             rasthofer 12/13 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::sysmat_hyperbolic(
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs   ///< element rhs to calculate
)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = my::eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // calculation of characteristic element length
  //----------------------------------------------------------------------

  // get gradient of initial phi at element center
  Core::LinAlg::Matrix<nsd_, 1> gradphizero(Core::LinAlg::Initialization::set_zero);
  gradphizero.multiply(my::derxy_, ephizero_[0]);

  // get characteristic element length
  const double charelelength = calc_char_ele_length_reinit(vol, gradphizero);

  //----------------------------------------------------------------------
  // prepare diffusion manager
  //----------------------------------------------------------------------

  // set diffusion coefficient of scalar 0 to 0.0
  if (not my::scatrapara_->mat_gp()) my::diffmanager_->set_isotropic_diff(0.0, 0);

  //----------------------------------------------------------------------
  // calculation of stabilization parameter at element center
  //----------------------------------------------------------------------

  // the stabilization parameter
  double tau = 0.0;

  if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
  {
    if (not my::scatrapara_->tau_gp())
    {
      // get velocity at element center
      Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::set_zero);

      // switch type for velocity field
      if (not lsreinitparams_->use_projected_vel())
      {
#ifndef USE_PHIN_FOR_VEL
        // gradient of current scalar value at element center
        Core::LinAlg::Matrix<nsd_, 1> gradphinp(Core::LinAlg::Initialization::set_zero);
        gradphinp.multiply(my::derxy_, my::ephinp_[0]);
        // get norm
        const double gradphinp_norm = gradphinp.norm2();
        // get sign function
        double signphi = 0.0;
        // initial phi at element center
        double phizero = 0.0;
        phizero = my::funct_.dot(ephizero_[0]);
        // current phi at element center
        double phinp = 0.0;
        phinp = my::funct_.dot(my::ephinp_[0]);
        sign_function(signphi, charelelength, phizero, gradphizero, phinp, gradphinp);

        if (gradphinp_norm > 1e-8) convelint.update(signphi / gradphinp_norm, gradphinp);
        // otherwise gradphi is almost zero and we keep a zero velocity
#else
        // gradient of scalar value at t_n at element center
        Core::LinAlg::Matrix<nsd_, 1> gradphin(Core::LinAlg::Initialization::set_zero);
        gradphin.multiply(my::derxy_, my::ephin_[0]);
        // get norm
        const double gradphin_norm = gradphin.norm2();
        // get sign function
        double signphi = 0.0;
        // initial phi at element center
        double phizero = 0.0;
        phizero = my::funct_.dot(ephizero_[0]);
        // phi at element center
        double phin = 0.0;
        phin = my::funct_.dot(my::ephin_[0]);
        sign_function(signphi, charelelength, phizero, gradphizero, phin, gradphin);

        if (gradphin_norm > 1e-8) convelint.update(signphi / gradphin_norm, gradphin);
        // otherwise gradphi is almost zero and we keep a zero velocity
#endif
      }
      else
      {
        convelint.multiply(my::econvelnp_, my::funct_);
      }

      // calculation of stabilization parameter at element center
      // here, second argument is isoptropic diffusion, which is zero!
      my::calc_tau(tau, 0.0, 0.0, 1.0, convelint, vol);
    }
  }

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------

    // set diffusion coefficient of scalar 0 to 0.0
    if (my::scatrapara_->mat_gp()) my::diffmanager_->set_isotropic_diff(0.0, 0);

    //--------------------------------------------------------------------
    // set all Gauss point quantities
    //--------------------------------------------------------------------

    // gradient of current scalar value at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphinp(Core::LinAlg::Initialization::set_zero);
    gradphinp.multiply(my::derxy_, my::ephinp_[0]);
    // scalar at integration point at time step n+1
    const double phinp = my::funct_.dot(my::ephinp_[0]);

    // initial phi at integration point
    double phizero = 0.0;
    phizero = my::funct_.dot(ephizero_[0]);
    // and corresponding gradient
    gradphizero.clear();
    gradphizero.multiply(my::derxy_, ephizero_[0]);

    // scalar at integration point at time step n
    const double phin = my::funct_.dot(my::ephin_[0]);

    // also store values in variable manager
    var_manager()->set_phinp(0, phinp);
    var_manager()->set_phin(0, phin);
    var_manager()->set_grad_phi(0, gradphinp);

    // get velocity at element center
    Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::set_zero);

    // get sign function
    double signphi = 0.0;
#ifndef USE_PHIN_FOR_VEL
    sign_function(signphi, charelelength, phizero, gradphizero, phinp, gradphinp);
#else
    // gradient of scalar value at t_n at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphin(Core::LinAlg::Initialization::set_zero);
    gradphin.multiply(my::derxy_, my::ephin_[0]);

    sign_function(signphi, charelelength, phizero, gradphizero, phin, gradphin);
#endif

    // switch type for velocity field
    if (not lsreinitparams_->use_projected_vel())
    {
#ifndef USE_PHIN_FOR_VEL
      // get norm
      const double gradphinp_norm = gradphinp.norm2();

      if (gradphinp_norm > 1e-8) convelint.update(signphi / gradphinp_norm, gradphinp);
      // otherwise gradphi is almost zero and we keep a zero velocity
#else
      // get norm
      const double gradphin_norm = gradphin.norm2();

      if (gradphin_norm > 1e-8) convelint.update(signphi / gradphin_norm, gradphin);
      // otherwise gradphi is almost zero and we keep a zero velocity
#endif
    }
    else
    {
      convelint.multiply(my::econvelnp_, my::funct_);
    }

    // convective part in convective form: u_x*N,x+ u_y*N,y
    Core::LinAlg::Matrix<nen_, 1> conv(Core::LinAlg::Initialization::set_zero);
    conv.multiply_tn(my::derxy_, convelint);

    // convective term using current scalar value
    double conv_phi(0.0);
    conv_phi = convelint.dot(gradphinp);

    // set changed values in variable manager
    var_manager()->set_conv(0, conv);
    var_manager()->set_con_vel(0, convelint);
    var_manager()->set_conv_phi(0, conv_phi);

    Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::set_zero);
    // diffusive term using current scalar value for higher-order elements
    if (my::use2ndderiv_)
    {
      // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
      my::get_laplacian_strong_form(diff);
      diff.put_scalar(0.0);
    }

    // get history data (or acceleration)
    double hist(0.0);
    // use history vector of global level
    hist = my::funct_.dot(my::ehist_[0]);
    // set changed values in variable manager
    var_manager()->set_hist(0, hist);

    //--------------------------------------------------------------------
    // calculation of stabilization parameter at integration point
    //--------------------------------------------------------------------

    // subgrid-scale velocity vector in gausspoint
    // Core::LinAlg::Matrix<nsd_,1> sgvelint(true);

    if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
    {
      if (my::scatrapara_->tau_gp())
        // calculation of stabilization parameter at integration point
        // here, second argument is isoptropic diffusion, which is zero!
        my::calc_tau(tau, 0.0, 0.0, 1.0, convelint, vol);
    }

    //--------------------------------------------------------------------
    // calculation of artificial diffusion
    //--------------------------------------------------------------------

    if (lsreinitparams_->art_diff() != Inpar::ScaTra::artdiff_none)
    {
      // residual of reinit eq
      double scatrares = 0.0;

      my::calc_strong_residual(0, scatrares, 1.0, 1.0, 0.0, signphi, tau);

      // compute artificial diffusion
      // diffusion coefficient has been explicitly set to zero
      // additionally stored in subgrid diffusion coefficient
      my::calc_artificial_diff(vol, 0, 1.0, convelint, gradphinp, conv_phi, scatrares, tau);

      if (lsreinitparams_->art_diff() == Inpar::ScaTra::artdiff_crosswind)
        diff_manager()->set_velocity_for_cross_wind_diff(convelint);

#ifdef MODIFIED_EQ
      // recompute tau to get adaption to artificial diffusion
      if (my::scatrapara_->StabType() != Inpar::ScaTra::stabtype_no_stabilization)
      {
        if (my::scatrapara_->TauGP())
          // calculation of stabilization parameter at integration point
          // here, second argument is isoptropic diffusion, which is zero!
          my::CalcTau(tau, my::diffmanager_->GetIsotropicDiff(0), 0.0, 1.0, convelint, vol, 0);
      }

      // recompute diff operator
      if (my::use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        my::get_laplacian_strong_form(diff);
        diff.scale(my::diffmanager_->GetIsotropicDiff(0));
        diff_phi = diff.dot(my::ephinp_[0]);
      }
#endif
    }

    // residual of convection-diffusion-reaction eq
    double scatrares(0.0);
    // compute residual of scalar transport equation and
    // subgrid-scale part of scalar
    my::calc_strong_residual(0, scatrares, 1.0, 1.0, 0.0, signphi, tau);

    //----------------------------------------------------------------
    // evaluation of matrix and rhs
    //----------------------------------------------------------------

    // stabilization parameter and integration factors
    const double taufac = tau * fac;
    const double timefacfac = my::scatraparatimint_->time_fac() * fac;
    const double dtfac = my::scatraparatimint_->dt() * fac;
    const double timetaufac = my::scatraparatimint_->time_fac() * taufac;

    //----------------------------------------------------------------
    // 1) element matrix: instationary terms
    //----------------------------------------------------------------

    my::calc_mat_mass(emat, 0, fac, 1.0);

    // subgrid-scale velocity (dummy)
    Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::set_zero);
    if (lsreinitparams_->lin_form() == Inpar::ScaTra::newton)
    {
      if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
        my::calc_mat_mass_stab(emat, 0, taufac, 1.0, 1.0, sgconv, diff);
    }

    //----------------------------------------------------------------
    // 2) element matrix: convective term in convective form
    //----------------------------------------------------------------

    if (lsreinitparams_->lin_form() == Inpar::ScaTra::newton)
      my::calc_mat_conv(emat, 0, timefacfac, 1.0, sgconv);

    // convective stabilization of convective term (in convective form)
    // transient stabilization of convective term (in convective form)
    if (lsreinitparams_->lin_form() == Inpar::ScaTra::newton)
    {
      if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
        my::calc_mat_trans_conv_diff_stab(emat, 0, timetaufac, 1.0, sgconv, diff);
    }


    // calculation of diffusive element matrix
    if (lsreinitparams_->art_diff() != Inpar::ScaTra::artdiff_none)
#ifndef MODIFIED_EQ
      calc_mat_diff(emat, 0, dtfac);  // implicit treatment
#else
      CalcMatDiff(emat, 0, timefacfac);
#endif

    //----------------------------------------------------------------
    // 3) element right hand side
    //----------------------------------------------------------------

    double rhsint = signphi;
    double rhsfac = my::scatraparatimint_->time_fac_rhs() * fac;
    double rhstaufac = my::scatraparatimint_->time_fac_rhs_tau() * taufac;

    // linearization of transient term
    my::calc_rhs_lin_mass(erhs, 0, rhsfac, fac, 1.0, 1.0);

    // the order of the following three functions is important
    // and must not be changed
    my::compute_rhs_int(rhsint, 1.0, 1.0, hist);
    double rea_phi(0.0);  // dummy
    my::recompute_scatra_res_for_rhs(scatrares, 0, diff, 1.0, 1.0, rea_phi, rhsint);
    // note: the third function is not required here, since we neither have a subgrid velocity
    //       nor a conservative form

    // standard Galerkin transient, old part of rhs and bodyforce term
    my::calc_rhs_hist_and_source(erhs, 0, fac, rhsint);

    // linearization of convective term
    my::calc_rhs_conv(erhs, 0, rhsfac);

    // linearization of diffusive term
    if (lsreinitparams_->art_diff() != Inpar::ScaTra::artdiff_none)
#ifndef MODIFIED_EQ
      calc_rhs_diff(erhs, 0, dtfac, gradphinp);  // implicit treatment
#else
      calc_rhs_diff(erhs, 0, rhsfac, gradphinp);
#endif

    // linearization of stabilization terms
    if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
      my::calc_rhs_trans_conv_diff_stab(erhs, 0, rhstaufac, 1.0, scatrares, sgconv, diff);

  }  // end: loop all Gauss points

  return;
}


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)             rasthofer 09/14 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::sysmat_elliptic(
    Core::LinAlg::SerialDenseMatrix& emat,       ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs,       ///< element rhs to calculate
    const Core::Geo::BoundaryIntCellPtrs& bcell  ///< interface for penalty term
)
{
  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    //--------------------------------------------------------------------
    // set all Gauss point quantities
    //--------------------------------------------------------------------

    // gradient of current scalar value at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphinp(Core::LinAlg::Initialization::set_zero);
    if (lsreinitparams_->project())
      gradphinp.multiply(my::econvelnp_, my::funct_);
    else
      gradphinp.multiply(my::derxy_, my::ephinp_[0]);

    // TODO: remove
    //    if (std::abs(gradphinp(2,0))>1.0e-8)
    //    {
    //        std::cout << gradphinp << std::setprecision(8) << my::ephinp_[0] << std::endl;
    //        FOUR_C_THROW("END");
    //    }

    double normgradphi = gradphinp.norm2();

    //----------------------------------------------------------------------
    // prepare diffusion manager
    //----------------------------------------------------------------------

    // calculate nonlinear diffusivity
    double diff = 0.0;
    switch (lsreinitparams_->diff_fct())
    {
      case Inpar::ScaTra::hyperbolic:
      {
        // the basic form: goes to -infinity, if norm(grad)=1
        // yields directly desired signed-distance field
        if (normgradphi > 1.0e-8)
          diff = 1.0 - (1.0 / normgradphi);
        else
          diff = 1.0 - (1.0 / 1.0e-8);

        break;
      }
      case Inpar::ScaTra::hyperbolic_smoothed_positive:
      {
        // version as suggested by Basting and Kuzmin 2013
        // returns to positive values for norm(grad)<0.5
        if (normgradphi > 1.0)
          diff = 1.0 - (1.0 / normgradphi);
        else
          diff = 2.0 * normgradphi * normgradphi - 3.0 * normgradphi + 1.0;

        break;
      }
      case Inpar::ScaTra::hyperbolic_clipped_05:
      {
        if (normgradphi > (2.0 / 3.0))
          diff = 1.0 - (1.0 / normgradphi);
        else
          diff = -0.5;

        break;
      }
      case Inpar::ScaTra::hyperbolic_clipped_1:
      {
        if (normgradphi > 0.5)
          diff = 1.0 - (1.0 / normgradphi);
        else
          diff = -1.0;

        break;
      }
      default:
      {
        FOUR_C_THROW("Unknown diffusivity function!");
        break;
      }
    }

    // set diffusivity of scalar 0 to 1.0 for lhs term
    my::diffmanager_->set_isotropic_diff(1.0, 0);

    //----------------------------------------------------------------
    // evaluation of matrix and rhs
    //----------------------------------------------------------------

    //----------------------------------------------------------------
    // 1) element matrix: diffusion matrix
    //----------------------------------------------------------------

    my::calc_mat_diff(emat, 0, fac);

    //----------------------------------------------------------------
    // 2) element right hand side
    //----------------------------------------------------------------

    // set diffusivity for rhs term
    my::diffmanager_->set_isotropic_diff((-diff + 1.0), 0);
    // set gradphi for rhs term
    my::scatravarmanager_->set_grad_phi(0, gradphinp);

    my::calc_rhs_diff(erhs, 0, -fac);

  }  // end: loop all Gauss points

  //----------------------------------------------------------------
  // 3) evaluation of penalty term at initial interface
  //----------------------------------------------------------------

  evaluate_interface_term(&emat, &erhs, bcell);

  return;
}


/*----------------------------------------------------------------------*
 |  sign function                                       rasthofer 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::sign_function(double& sign_phi,
    const double charelelength, const double phizero,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphizero, const double phi,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi)
{
  // compute interface thickness
  const double epsilon = lsreinitparams_->interface_thickness_fac() * charelelength;

  if (lsreinitparams_->sign_type() == Inpar::ScaTra::signtype_nonsmoothed)
  {
    if (phizero < 0.0)
      sign_phi = -1.0;
    else if (phizero > 0.0)
      sign_phi = +1.0;
    else
      sign_phi = 0.0;
  }
  else if (lsreinitparams_->sign_type() == Inpar::ScaTra::signtype_SussmanSmerekaOsher1994)
  {
    sign_phi = phizero / sqrt(phizero * phizero + epsilon * epsilon);
  }
  else if (lsreinitparams_->sign_type() == Inpar::ScaTra::signtype_PengEtAl1999)
  {
    const double grad_norm_phi = gradphi.norm2();
    sign_phi = phi / sqrt(phi * phi + epsilon * epsilon * grad_norm_phi * grad_norm_phi);
  }
  else if (lsreinitparams_->sign_type() == Inpar::ScaTra::signtype_SussmanFatemi1999)
  {
    if (fabs(epsilon) < 1e-15)
      FOUR_C_THROW("divide by zero in evaluate for smoothed sign function");

    if (phizero < -epsilon)
      sign_phi = -1.0;
    else if (phizero > epsilon)
      sign_phi = +1.0;
    else
      sign_phi = phizero / epsilon + 1.0 / M_PI * sin(M_PI * phizero / epsilon);
  }
  else
    FOUR_C_THROW("unknown type of sign function!");
  return;
}


/*----------------------------------------------------------------------*
 | derivative of sign function                          rasthofer 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::deriv_sign_function(
    double& deriv_sign, const double charelelength, const double phizero)
{
  // compute interface thickness
  const double epsilon = lsreinitparams_->interface_thickness_fac() * charelelength;

  if (phizero < -epsilon)
    deriv_sign = 0.0;
  else if (phizero > epsilon)
    deriv_sign = 0.0;
  else
    deriv_sign = 1.0 / (2.0 * epsilon) * (1.0 + cos(M_PI * phizero / epsilon));

  return;
}


/*----------------------------------------------------------------------*
 |  calculation of characteristic element length        rasthofer 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
double Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_char_ele_length_reinit(
    const double vol, const Core::LinAlg::Matrix<nsd_, 1>& gradphizero)
{
  // define and initialize length
  double h = 0.0;

  //---------------------------------------------------------------------
  // select from various definitions for characteristic element length
  //---------------------------------------------------------------------
  switch (lsreinitparams_->char_ele_length_reinit())
  {
    // a) streamlength due to Kees et al. (2011)
    case Inpar::ScaTra::streamlength_reinit:
    {
      // get norm of gradient of phi
      double gradphi_norm = gradphizero.norm2();
      Core::LinAlg::Matrix<nsd_, 1> gradphi_scaled(Core::LinAlg::Initialization::set_zero);
      if (gradphi_norm >= 1e-8)
        gradphi_scaled.update(1.0 / gradphi_norm, gradphizero);
      else
      {
        // TODO: clarify this
        FOUR_C_THROW("gradphi_norm=0: cannot compute characteristic element length");
        gradphi_scaled.update(1.0, gradphizero);
        // gradphi_scaled.clear();
        // gradphi_scaled(0,0) = 1.0;
      }

      // computation of covariant metric tensor
      double G;
      double Gnormgradphi(0.0);
      for (unsigned nn = 0; nn < nsd_; ++nn)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          G = my::xij_(nn, 0) * my::xij_(rr, 0);
          for (unsigned tt = 1; tt < nsd_; ++tt)
          {
            G += my::xij_(nn, tt) * my::xij_(rr, tt);
          }
          Gnormgradphi += gradphi_scaled(nn, 0) * G * gradphi_scaled(rr, 0);
        }
      }

      h = 1.0 / std::sqrt(Gnormgradphi);
    }
    break;

    // b) cubic/square root of element volume/area or element length (3-/2-/1-D)
    case Inpar::ScaTra::root_of_volume_reinit:
    {
      // cast dimension to a double variable -> pow()
      const double dim = static_cast<double>(nsd_);
      h = std::pow(vol, 1.0 / dim);
    }
    break;

    default:
      FOUR_C_THROW("unknown characteristic element length\n");
      break;
  }

  return h;
}


/*------------------------------------------------------------------- *
 | calculation of diffusive element matrix            rasthofer 12/13 |
 | here we consider both isotropic and crosswind artificial diffusion |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_mat_diff(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac)
{
  // flag for anisotropic diffusion
  const bool crosswind = diff_manager()->have_cross_wind_diff();

  // set scalar diffusion: isotropic or factor for anisotropic tensor in case of
  // crosswind diffusion
  double diffus = diff_manager()->get_isotropic_diff(k);

  // diffusive factor
  const double fac_diffus = timefacfac * diffus;

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * my::numdofpernode_ + k;
      double laplawf(0.0);

      // in case of anisotropic or crosswind diffusion, multiply 'derxy_' with diffusion tensor
      // inside 'get_laplacian_weak_form'
      if (crosswind)
        my::get_laplacian_weak_form(laplawf, diff_manager()->get_crosswind_tensor(), ui, vi);
      else
        my::get_laplacian_weak_form(laplawf, ui, vi);

      emat(fvi, fui) += fac_diffus * laplawf;
    }
  }
  return;
}


/*-------------------------------------------------------------------- *
 |  standard Galerkin diffusive term on right hand side rasthofer 12/13 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_rhs_diff(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi)
{
  // flag for anisotropic diffusion
  const bool crosswind = diff_manager()->have_cross_wind_diff();

  // set scalar diffusion: isotropic or factor for anisotropic tensor in case of
  // crosswind diffusion
  double vrhs = rhsfac * diff_manager()->get_isotropic_diff(k);

  Core::LinAlg::Matrix<nsd_, 1> gradphirhs(Core::LinAlg::Initialization::set_zero);
  if (crosswind)
  {
    // in case of anisotropic or crosswind diffusion, multiply 'gradphi' with diffusion tensor
    gradphirhs.multiply(diff_manager()->get_crosswind_tensor(), gradphi);
  }
  else
  {
    gradphirhs.update(1.0, gradphi, 0.0);
  }

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    double laplawf(0.0);

    this->get_laplacian_weak_form_rhs(laplawf, gradphirhs, vi);

    erhs[fvi] -= vrhs * laplawf;
  }

  return;
}


/*-------------------------------------------------------------------- *
 | calculation of interface penalty term for elliptic reinitialization |
 |                                                     rasthofer 09/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::evaluate_interface_term(
    Core::LinAlg::SerialDenseMatrix* emat,       //!< element matrix to calculate
    Core::LinAlg::SerialDenseVector* erhs,       //!< element vector to calculate
    const Core::Geo::BoundaryIntCellPtrs& bcell  //!< interface for penalty term
)
{
  //---------------------------------------------------------------------------
  // loop over boundary integration cells
  //---------------------------------------------------------------------------
  for (Core::Geo::BoundaryIntCellPtrs::const_iterator cell = bcell.begin(); cell != bcell.end();
      ++cell)
  {
    // get shape of boundary cell
    Core::FE::CellType celldistype = (*cell)->shape();

    Core::Geo::BoundaryIntCell& actcell = **cell;

    switch (celldistype)
    {
      case Core::FE::CellType::point1:
      {
        calc_penalty_term_0_d(emat, erhs, actcell);
        break;
      }
      case Core::FE::CellType::tri3:
      {
        calc_penalty_term<Core::FE::CellType::tri3>(*emat, *erhs, actcell);
        break;
      }
      case Core::FE::CellType::quad4:
      {
        calc_penalty_term<Core::FE::CellType::quad4>(*emat, *erhs, actcell);
        break;
      }
      default:
        FOUR_C_THROW("cell distype not implemented yet ( cellType = {} )",
            Core::FE::cell_type_to_string(celldistype).c_str());
        break;
    }
  }

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_penalty_term_0_d(
    Core::LinAlg::SerialDenseMatrix* emat, Core::LinAlg::SerialDenseVector* erhs,
    const Core::Geo::BoundaryIntCell& cell)
{
  // get the cut position ( local parent element coordinates )
  const Core::LinAlg::Matrix<nsd_ele_, 1> posXiDomain(
      cell.cell_nodal_pos_xi_domain().values(), true);

  // --------------------------------------------------------------------------
  // evaluate shape functions at the cut position
  // --------------------------------------------------------------------------
  my::funct_.clear();
  Core::FE::shape_function<distype>(posXiDomain, my::funct_);

  //--------------------------------------------------------------------------
  // evaluate element matrix (mass matrix-like)
  //--------------------------------------------------------------------------
  // caution density of original function is replaced by penalty parameter here
  if (emat)
  {
    my::calc_mat_mass(*emat, 0, 1.0, lsreinitparams_->penalty_para());
  }

  //--------------------------------------------------------------------------
  // evaluate rhs contributions
  //--------------------------------------------------------------------------
  if (erhs)
  {
    my::scatravarmanager_->set_conv_phi(0, my::ephinp_[0].dot(my::funct_));
    my::calc_rhs_conv(*erhs, 0, -lsreinitparams_->penalty_para());
  }
}

/*-------------------------------------------------------------------- *
 | calculation of interface penalty term for elliptic reinitialization |
 | gauss loop                                          rasthofer 09/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
template <Core::FE::CellType celldistype>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_penalty_term(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to calculate
    const Core::Geo::BoundaryIntCell& cell  //!< interface cell
)
{
  // get number of vertices of cell
  const unsigned numvertices = Core::FE::num_nodes<celldistype>;
  const unsigned nsd = 3;
  if (nsd_ != 3) FOUR_C_THROW("Extend for other dimensions");
  const size_t nsd_cell = 2;  // nsd_-1;
  // get coordinates of vertices of boundary integration cell in element coordinates \xi^domain
  Core::LinAlg::SerialDenseMatrix cellXiDomaintmp = cell.cell_nodal_pos_xi_domain();
  // cellXiDomaintmp.print(std::cout);
  // transform to fixed size format
  const Core::LinAlg::Matrix<nsd, numvertices> cellXiDomain(cellXiDomaintmp);

  //----------------------------------------------------------------------------------------------
  // integration loop over Gaussian points
  //----------------------------------------------------------------------------------------------
  // integrations points and weights
  Core::FE::IntegrationPoints2D intpoints(ScaTra::CellTypeToOptGaussRule<celldistype>::rule);

  for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
  {
    // new transformation for boundary integrals
    // 1. define a coupled transformation x_3D(xi_3D(eta_2D)): transformation from 2D->3D
    // 2. compute the corresponding Jacobian J_eta2D->x_3D and
    // 3. the corresponding surface integral factor sqrt(det(J_eta2D->x_3D^T * J_eta2D->x_3D))
    // 4. approximate integral with Gauss rule in eta coordinate system
    // 5. evaluate the transformed integrand f(x(xi(eta)))

    const Core::LinAlg::Matrix<nsd_cell, 1> gpinEta2D(intpoints.qxg[iquad]);

    // Jacobian for coupled transformation
    // get derivatives dxi_3D/deta_2D
    static Core::LinAlg::Matrix<nsd_cell, numvertices> deriv_eta2D;
    Core::FE::shape_function_2d_deriv1(deriv_eta2D, gpinEta2D(0, 0), gpinEta2D(1, 0), celldistype);

    // calculate dxi3Ddeta2D
    static Core::LinAlg::Matrix<nsd, nsd_cell> dXi3Ddeta2D;
    dXi3Ddeta2D.clear();

    for (unsigned i = 0; i < nsd; i++)         // dimensions
      for (unsigned j = 0; j < nsd_cell; j++)  // derivatives
        for (unsigned k = 0; k < numvertices; k++)
        {
          dXi3Ddeta2D(i, j) += cellXiDomain(i, k) * deriv_eta2D(j, k);
        }

    // transform Gauss point to xi3D space (element parameter space)
    static Core::LinAlg::Matrix<nsd, 1> gpinXi3D;
    gpinXi3D.clear();

    // coordinates of this integration point in element coordinates \xi^domain
    Core::Geo::map_eta_b_to_xi_d(cell, gpinEta2D, gpinXi3D);

    static Core::LinAlg::Matrix<nsd, nen_> deriv_xi3D;
    Core::FE::shape_function_3d_deriv1(
        deriv_xi3D, gpinXi3D(0, 0), gpinXi3D(1, 0), gpinXi3D(2, 0), distype);

    // calculate dx3Ddxi3D
    static Core::LinAlg::Matrix<nsd, nsd> dX3DdXi3D;
    dX3DdXi3D.clear();
    for (unsigned i = 0; i < nsd; i++)    // dimensions
      for (unsigned j = 0; j < nsd; j++)  // derivatives
        for (unsigned k = 0; k < nen_; k++) dX3DdXi3D(i, j) += my::xyze_(i, k) * deriv_xi3D(j, k);

    // get the coupled Jacobian dx3Ddeta2D
    static Core::LinAlg::Matrix<3, 2> dx3Ddeta2D;
    dx3Ddeta2D.clear();
    for (unsigned i = 0; i < nsd; i++)         // dimensions
      for (unsigned j = 0; j < nsd_cell; j++)  // derivatives
        for (unsigned k = 0; k < nsd; k++) dx3Ddeta2D(i, j) += dX3DdXi3D(i, k) * dXi3Ddeta2D(k, j);

    // get deformation factor
    static Core::LinAlg::Matrix<nsd_cell, nsd_cell> Jac_tmp;  // J^T*J
    Jac_tmp.clear();
    Jac_tmp.multiply_tn(dx3Ddeta2D, dx3Ddeta2D);

    if (Jac_tmp.determinant() == 0.0)
      FOUR_C_THROW("deformation factor for boundary integration is zero");
    const double deform_factor = sqrt(Jac_tmp.determinant());  // sqrt(det(J^T*J))

    const double fac = intpoints.qwgt[iquad] * deform_factor;

    Core::LinAlg::Matrix<nsd_cell, 1> posEtaBoundary;
    posEtaBoundary.clear();
    for (unsigned i = 0; i < nsd_cell; i++) posEtaBoundary(i, 0) = gpinEta2D(i, 0);

    Core::LinAlg::Matrix<nsd, 1> posXiDomain;
    posXiDomain.clear();
    for (unsigned i = 0; i < nsd; i++) posXiDomain(i, 0) = gpinXi3D(i, 0);

    //--------------------------------------------------------------------------------------------
    // evaluate shape functions and their first derivatives at this Gaussian point
    //--------------------------------------------------------------------------------------------
    my::funct_.clear();
    Core::FE::shape_function_3d(
        my::funct_, posXiDomain(0), posXiDomain(1), posXiDomain(2), distype);

    //--------------------------------------------------------------------------------------------
    // evaluate mat and rhs
    //--------------------------------------------------------------------------------------------
    // caution density of original function is replaced by penalty parameter here
    my::calc_mat_mass(emat, 0, fac, lsreinitparams_->penalty_para());

  }  // loop Gaussian points

  return;
}

FOUR_C_NAMESPACE_CLOSE

// template classes

#include "4C_scatra_ele_calc.inst.hpp"

FOUR_C_NAMESPACE_OPEN

// 1D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad8,2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex20,3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::wedge6,3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::nurbs27,3>;

FOUR_C_NAMESPACE_CLOSE
