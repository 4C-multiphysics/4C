// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_lubrication_ele_calc.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_lubrication.hpp"
#include "4C_lubrication_ele_action.hpp"
#include "4C_lubrication_ele_calc_utils.hpp"
#include "4C_lubrication_ele_parameter.hpp"
#include "4C_mat_lubrication_mat.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * Constructor
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::LubricationEleCalc<distype, probdim>::LubricationEleCalc(
    const std::string& disname)
    : lubricationpara_(Discret::Elements::LubricationEleParameter::instance(
          disname)),  // standard parameter list
      eprenp_(true),  // initialized to zero
      xsi_(true),     // initialized to zero
      xyze_(true),    // initialized to zero
      funct_(true),   // initialized to zero
      deriv_(true),   // initialized to zero
      derxy_(true),   // initialized to zero
      xjm_(true),     // initialized to zero
      xij_(true),     // initialized to zero
      eheinp_(true),
      eheidotnp_(true),
      edispnp_(true),
      viscmanager_(
          std::make_shared<LubricationEleViscManager>()),  // viscosity manager for viscosity
      lubricationvarmanager_(std::shared_ptr<LubricationEleInternalVariableManager<nsd_, nen_>>(
          new LubricationEleInternalVariableManager<nsd_, nen_>())),  // internal variable manager
      eid_(0),
      ele_(nullptr),
      Dt_(0.0),

      // heightint_(0.0),
      // heightdotint_(0.0),
      pflowfac_(true),
      pflowfacderiv_(true),
      sflowfac_(0.0),
      sflowfacderiv_(0.0)
{
  FOUR_C_ASSERT(
      nsd_ >= nsd_ele_, "problem dimension has to be equal or larger than the element dimension!");

  return;
}


/*----------------------------------------------------------------------*
 | singleton access method                                  wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::LubricationEleCalc<distype, probdim>*
Discret::Elements::LubricationEleCalc<distype, probdim>::instance(const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const std::string& disname)
      {
        return std::unique_ptr<LubricationEleCalc<distype, probdim>>(
            new LubricationEleCalc<distype, probdim>(disname));
      });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, disname);
}


/*----------------------------------------------------------------------*
 * Action type: Evaluate                                    wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::LubricationEleCalc<distype, probdim>::evaluate(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
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
  // calculate element coefficient matrix and rhs
  //--------------------------------------------------------------------------------

  sysmat(ele, elemat1_epetra, elevec1_epetra);

  return 0;
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate element matrix associated to the linearization
 * of the residual wrt. the discrete film height (only used for monolithic
 * EHL problems)
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::LubricationEleCalc<distype, probdim>::evaluate_ehl_mon(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------

  if (setup_calc(ele, discretization) == -1) return 0;

  // set time step as a class member
  double dt = params.get<double>("delta time");
  Dt_ = dt;

  //--------------------------------------------------------------------------------
  // extract element based or nodal values
  //--------------------------------------------------------------------------------

  extract_element_and_node_values(ele, params, discretization, la);

  //--------------------------------------------------------------------------------
  // calculate element off-diagonal-matrix for height linearization in monolithic EHL
  //--------------------------------------------------------------------------------

  matrixfor_ehl_mon(ele, elemat1_epetra, elemat2_epetra);

  return 0;
}

/*----------------------------------------------------------------------*
 | setup element evaluation                                 wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::LubricationEleCalc<distype, probdim>::setup_calc(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  // get element coordinates
  read_element_coordinates(ele);

  // set element id
  eid_ = ele->id();
  // set element
  ele_ = ele;

  return 0;
}

/*----------------------------------------------------------------------*
 | read element coordinates                                 wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::read_element_coordinates(
    const Core::Elements::Element* ele)
{
  // Directly copy the coordinates since in 3D the transformation is just the identity
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  return;
}  // LubricationEleCalc::read_element_coordinates

/*----------------------------------------------------------------------*
 | extract element based or nodal values
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // 1. Extract the average tangential velocity

  // nodeset the velocity is defined on
  const int ndsvel = 1;

  // get the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
      discretization.get_state(ndsvel, "av_tang_vel");
  if (!vel) FOUR_C_THROW("got nullptr pointer for \"av_tang_vel\"");

  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  // extract local vel from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*vel, eAvTangVel_, lmvel);

  if (lubricationpara_->modified_reynolds())
  {
    // 1.1 Extract the relative tangential velocity
    // get the global vector
    auto velrel = discretization.get_state(ndsvel, "rel_tang_vel");
    if (!velrel) FOUR_C_THROW("got nullptr pointer for \"rel_tang_vel\"");

    const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

    // construct location vector for velocity related dofs
    std::vector<int> lmvel(nsd_ * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

    // extract local vel from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*velrel, eRelTangVel_, lmvel);
  }

  // 2. In case of ale, extract the displacements of the element nodes and update the nodal
  // coordinates

  // Only required, in case of ale:
  if (params.get<bool>("isale"))
  {
    // get number of dofset associated with displacement related dofs
    const int ndsdisp = 1;  // needs further implementation: params.get<int>("ndsdisp");
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    xyze_ += edispnp_;
  }

  // 3. Extract the film height at the element nodes
  // get number of dofset associated with height dofs
  const int ndsheight = 1;  // needs further implementation: params.get<int>("ndsheight");

  // get the global vector containing the heights
  std::shared_ptr<const Core::LinAlg::Vector<double>> height =
      discretization.get_state(ndsheight, "height");
  if (height == nullptr) FOUR_C_THROW("Cannot get state vector height");

  // determine number of height related dofs per node
  const int numheightdofpernode = la[ndsheight].lm_.size() / nen_;

  // construct location vector for height related dofs
  std::vector<int> lmheight(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmheight[inode * nsd_ + idim] = la[ndsheight].lm_[inode * numheightdofpernode + idim];

  // extract local height from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
      *height, eheinp_, la[ndsheight].lm_);

  // 3.1. Extract the film height time derivative at the element node
  if (lubricationpara_->add_sqz())
  {
    const int ndsheightdot = 1;

    // get the global vector containing the heightdots
    auto heightdot = discretization.get_state(ndsheightdot, "heightdot");
    if (heightdot == nullptr) FOUR_C_THROW("Cannot get state vector heightdot");

    // determine number of heightdot related dofs per node
    const int numheightdotdofpernode = la[ndsheightdot].lm_.size() / nen_;

    // construct location vector for heightdot related dofs
    std::vector<int> lmheightdot(nsd_ * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        lmheightdot[inode * nsd_ + idim] =
            la[ndsheightdot].lm_[inode * numheightdotdofpernode + idim];

    // extract local height from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
        *heightdot, eheidotnp_, la[ndsheightdot].lm_);
  }
  // 4. Extract the pressure field at the element nodes

  // get the global vector containing the pressure
  std::shared_ptr<const Core::LinAlg::Vector<double>> prenp = discretization.get_state("prenp");
  if (prenp == nullptr) FOUR_C_THROW("Cannot get state vector 'prenp'");

  // values of pressure field are always in first dofset
  const std::vector<int>& lm = la[0].lm_;

  // extract the local values at the element nodes
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*prenp, eprenp_, lm);

  return;
}

/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 wirtz 10/15 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::sysmat(
    Core::Elements::Element* ele,           ///< the element whose matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs   ///< element rhs to calculate
)
{
  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n)
  double densn(1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  double densnp(1.0);
  // density at t_(n+alpha_M)
  double densam(1.0);

  // fluid viscosity
  double visc(0.0);
  double dvisc(0.0);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      FourC::Lubrication::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    // calculate height (i.e. the distance of the contacting bodies) at Integration point
    double heightint(0.0);
    calc_height_at_int_point(heightint);

    // calculate heightDot (i.e. the distance of the contacting bodies) at Integration point
    double heightdotint(0.0);
    calc_height_dot_at_int_point(heightdotint);

    // calculate average surface velocity of the contacting bodies at Integration point
    Core::LinAlg::Matrix<nsd_, 1> avrvel(true);  // average surface velocity, initialized to zero
    calc_avr_vel_at_int_point(avrvel);

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    get_material_params(ele, densn, densnp, densam, visc, dvisc, iquad);

    // integration factors
    const double timefacfac = fac;  // works only for stationary problems!

    double rhsfac = fac;  // works only for stationary problems!

    // bool modifiedreynolds = lubricationpara_->ModifiedReynolds();
    if (lubricationpara_->modified_reynolds())
    {
      if (lubricationpara_->pure_lub())
        FOUR_C_THROW("pure lubrication is not implemented for modified Reynolds equation");
      // calculate relative surface velocity of the contacting bodies at Integration point
      Core::LinAlg::Matrix<nsd_, 1> relvel(true);  // relative surface velocity, initialized to zero
      calc_rel_vel_at_int_point(relvel);

      // calculate pressure flow factor at Integration point
      // Core::LinAlg::Matrix<nsd_, 1> pflowfac(true);  // Pressure flow factor, initialized to zero
      // Core::LinAlg::Matrix<nsd_, 1> pflowfacderiv(true);
      calc_p_flow_fac_at_int_point(pflowfac_, pflowfacderiv_, heightint);

      // calculate shear flow factor at Integration point
      // double sflowfac(0.0);
      // double sflowfacderiv(0.0);
      calc_s_flow_fac_at_int_point(sflowfac_, sflowfacderiv_, heightint);

      // 1) element matrix
      // 1.1) calculation of Poiseuille contribution of element matrix -> Kpp
      calc_mat_psl(emat, timefacfac, visc, heightint, pflowfac_);
      // CalcMatPDVis(emat,timefacfac,visc, heightint_);
      // 2) rhs matrix
      // 2.0) calculation of shear contribution to RHS matrix
      calc_rhs_shear(erhs, rhsfac, relvel, sflowfac_);

      // 2.1) calculation of Poiseuille contribution of rhs matrix

      calc_rhs_psl(erhs, rhsfac, visc, heightint, pflowfac_);
    }
    else
    {
      // 1) element matrix
      // 1.1) calculation of Poiseuille contribution of element matrix

      calc_mat_psl(emat, timefacfac, visc, heightint);

      // 1.2) calculation of Poiseuille-Pressure-dependent-viscosity contribution of element matrix

      calc_mat_psl_vis(emat, timefacfac, visc, heightint, dvisc);

      // 2) rhs matrix
      // 2.1) calculation of Poiseuille contribution of rhs matrix

      calc_rhs_psl(erhs, rhsfac, visc, heightint);
    }
    // 2.2) calculation of Wedge contribution of rhs matrix

    calc_rhs_wdg(erhs, rhsfac, heightint, avrvel);

    // 2.3) calculation of squeeze contribution to RHS matrix
    if (lubricationpara_->add_sqz()) calc_rhs_sqz(erhs, rhsfac, heightdotint);

  }  // end loop Gauss points
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::matrixfor_ehl_mon(
    Core::Elements::Element* ele,  ///< the element whose matrix is calculated
    Core::LinAlg::SerialDenseMatrix& ematheight, Core::LinAlg::SerialDenseMatrix& ematvel)
{
  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n)
  double densn(1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  double densnp(1.0);
  // density at t_(n+alpha_M)
  double densam(1.0);

  // fluid viscosity
  double visc(0.0);
  double dvisc(0.0);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      FourC::Lubrication::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // set gauss point variables needed for evaluation of mat and rhs
    set_internal_variables_for_mat_and_rhs();

    // calculate height (i.e. the distance of the contacting bodies) at Integration point
    double heightint(0.0);
    calc_height_at_int_point(heightint);

    // calculate average surface velocity of the contacting bodies at Integration point
    Core::LinAlg::Matrix<nsd_, 1> avrvel(true);  // average surface velocity, initialized to zero
    calc_avr_vel_at_int_point(avrvel);

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    get_material_params(ele, densn, densnp, densam, visc, dvisc, iquad);

    const Core::LinAlg::Matrix<nsd_, 1>& gradpre = lubricationvarmanager_->grad_pre();
    const double roughness = lubricationpara_->roughness_deviation();
    // std::cout << "roughness is equal to check::" << roughness << std::endl;

    if (lubricationpara_->modified_reynolds())
    {
      // FOUR_C_THROW("we should not be here");
      // calculate relative surface velocity of the contacting bodies at Integration point
      Core::LinAlg::Matrix<nsd_, 1> relvel(true);  // relative surface velocity, initialized to zero
      calc_rel_vel_at_int_point(relvel);

      // calculate pressure flow factor at Integration point
      calc_p_flow_fac_at_int_point(pflowfac_, pflowfacderiv_, heightint);

      // calculate shear flow factor at Integration point
      calc_s_flow_fac_at_int_point(sflowfac_, sflowfacderiv_, heightint);

      // Linearization of Poiseuille term wrt the film height (first part)
      for (int vi = 0; vi < nen_; vi++)
      {
        double laplawf(0.0);
        get_laplacian_weak_form_rhs(laplawf, gradpre, vi, pflowfac_);
        for (int ui = 0; ui < nen_; ui++)
        {
          double val = fac * (1 / (12 * visc)) * 3 * heightint * heightint * laplawf * funct_(ui);
          ematheight(vi, (ui * nsd_)) -= val;
        }
      }  // end loop for linearization of Poiseuille term wrt the film height (first part)

      // Linearization of Poiseuille term wrt the film height (second part)
      for (int vi = 0; vi < nen_; vi++)
      {
        double laplawf(0.0);
        get_laplacian_weak_form_rhs(laplawf, gradpre, vi, pflowfacderiv_);
        for (int ui = 0; ui < nen_; ui++)
        {
          double val =
              fac * (1 / (12 * visc)) * heightint * heightint * heightint * laplawf * funct_(ui);
          ematheight(vi, (ui * nsd_)) -= val;
        }
      }  // end loop for linearization of Poiseuille term wrt the film height (second part)

      // Linearization of Shear term wrt the film height
      for (int vi = 0; vi < nen_; vi++)
      {
        double val(0.0);

        for (int idim = 0; idim < nsd_; idim++)
        {
          val += derxy_(idim, vi) * relvel(idim);
        }

        for (int ui = 0; ui < nen_; ui++)
        {
          ematheight(vi, (ui * nsd_)) += fac * roughness * sflowfacderiv_ * val * funct_(ui);
        }
      }  // end loop for linearization of Shear term wrt the film height

      // Linearization of Shear term wrt the velocities
      for (int vi = 0; vi < nen_; vi++)
      {
        for (int ui = 0; ui < nen_; ui++)
        {
          for (int idim = 0; idim < nsd_; idim++)
          {
            ematvel(vi, (ui * nsd_ + idim)) -=
                fac * sflowfac_ * roughness * derxy_(idim, vi) * funct_(ui);
          }
        }
      }  // end loop for linearization of Shear term wrt the velocities
    }
    else
    {
      // Linearization of Poiseuille term wrt the film height
      for (int vi = 0; vi < nen_; vi++)
      {
        double laplawf(0.0);
        get_laplacian_weak_form_rhs(laplawf, gradpre, vi);
        for (int ui = 0; ui < nen_; ui++)
        {
          double val = fac * (1 / (12 * visc)) * 3 * heightint * heightint * laplawf * funct_(ui);
          ematheight(vi, (ui * nsd_)) -= val;
        }
      }  // end loop for linearization of Poiseuille term wrt the film height
    }


    // Linearization of Couette term wrt the film height
    for (int vi = 0; vi < nen_; vi++)
    {
      double val(0.0);

      for (int idim = 0; idim < nsd_; idim++)
      {
        val += derxy_(idim, vi) * avrvel(idim);
      }

      for (int ui = 0; ui < nen_; ui++)
      {
        ematheight(vi, (ui * nsd_)) += fac * val * funct_(ui);
      }
    }  // end loop for linearization of Couette term wrt the film height

    if (lubricationpara_->add_sqz())
    {
      // Linearization of Squeeze term wrt the film height
      for (int vi = 0; vi < nen_; vi++)
      {
        for (int ui = 0; ui < nen_; ui++)
        {
          ematheight(vi, (ui * nsd_)) -= fac * (1.0 / Dt_) * funct_(ui) * funct_(vi);
        }
      }  // end loop for linearization of Squeeze term wrt the film height
    }

    // Linearization of Couette term wrt the velocities
    for (int vi = 0; vi < nen_; vi++)
    {
      for (int ui = 0; ui < nen_; ui++)
      {
        for (int idim = 0; idim < nsd_; idim++)
        {
          ematvel(vi, (ui * nsd_ + idim)) -= fac * heightint * derxy_(idim, vi) * funct_(ui);
        }
      }
    }  // end loop for linearization of Couette term wrt the velocities

  }  // end gauss point loop
}


/*------------------------------------------------------------------------------*
 | set internal variables                                           wirtz 10/15 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype,
    probdim>::set_internal_variables_for_mat_and_rhs()
{
  lubricationvarmanager_->set_internal_variables(funct_, derxy_, eprenp_);
  return;
}

/*------------------------------------------------------------------------------*
 | set internal variables (pressure flow factor)                   Faraji 02/19 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_p_flow_fac_at_int_point(
    Core::LinAlg::Matrix<nsd_, 1>& pflowfac, Core::LinAlg::Matrix<nsd_, 1>& pflowfacderiv,
    const double heightint)
{
  if (!(lubricationpara_->modified_reynolds()))
    FOUR_C_THROW("Classical Reynolds Equ. does NOT need flow factors!");
  // lubricationvarmanager_->PressureFlowFactor(funct_, derxy_, eprenp_);
  const double roughness = lubricationpara_->roughness_deviation();
  const double r = (roughness / heightint);
  for (int i = 0; i < nsd_; ++i)
  {
    pflowfac(i) = 1 + 3 * r * r;
    pflowfacderiv(i) = (-6) * r * r / heightint;
  }
  return;
}

/*------------------------------------------------------------------------------*
 | set internal variables  (shear flow factor)                     Faraji 02/19 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_s_flow_fac_at_int_point(
    double& sflowfac, double& sflowfacderiv, const double heightint)
{
  if (!(lubricationpara_->modified_reynolds()))
    FOUR_C_THROW("Classical Reynolds Equ. does NOT need flow factors!");
  // lubricationvarmanager_->ShearFlowFactor(sflowfac, sflowfacderiv, heightint);
  const double roughness = lubricationpara_->roughness_deviation();
  const double r = (roughness / heightint);
  sflowfac = (-3 * r - 30 * r * r * r) / (1 + 6 * r * r);
  sflowfacderiv = (3 * r + 54 * r * r * r - 360 * r * r * r * r * r) /
                  (heightint * (1 + 6 * r * r) * (1 + 6 * r * r));
  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (private)                     wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    double& densn,                       //!< density at t_(n)
    double& densnp,                      //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,                      //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    double& dvisc,                       //!< derivative of the fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  materials(material, densn, densnp, densam, visc, dvisc, iquad);

  return;
}  // LubricationEleCalc::get_material_params

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                   wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::materials(
    const std::shared_ptr<Core::Mat::Material> material,  //!< pointer to current material
    double& densn,                                        //!< density at t_(n)
    double& densnp,                                       //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,                                       //!< density at t_(n+alpha_M)
    double& visc,                                         //!< fluid viscosity
    double& dvisc,                                        //!< derivative of the fluid viscosity
    const int iquad                                       //!< id of current gauss point

)
{
  switch (material->material_type())
  {
    case Core::Materials::m_lubrication:
      mat_lubrication(material, densn, densnp, densam, visc, dvisc, iquad);
      break;
    default:
      FOUR_C_THROW("Material type %s is not supported",
          Core::Materials::to_string(material->material_type()).data());
      break;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Material Lubrication                                    wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::mat_lubrication(
    const std::shared_ptr<Core::Mat::Material> material,  //!< pointer to current material
    double& densn,                                        //!< density at t_(n)
    double& densnp,                                       //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,                                       //!< density at t_(n+alpha_M)
    double& visc,                                         //!< fluid viscosity
    double& dvisc,                                        //!< derivative of the fluid viscosity
    const int iquad  //!< id of current gauss point (default = -1)
)
{
  const std::shared_ptr<Mat::LubricationMat>& actmat =
      std::dynamic_pointer_cast<Mat::LubricationMat>(material);
  // get constant viscosity

  // double pressure = 0.0;
  //  const double pres = my::eprenp_.Dot(my::funct_);
  const double pre = lubricationvarmanager_->prenp();
  //  const double p = eprenp_.Dot(funct_);

  visc = actmat->compute_viscosity(pre);
  dvisc = actmat->compute_viscosity_deriv(pre, visc);


  // viscmanager_->SetIsotropicVisc(visc);
  return;
}  // LubricationEleCalc<distype>::mat_lubrication

/*------------------------------------------------------------------- *
 |  calculate linearization of the Laplacian (weak form)
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::get_laplacian_weak_form(
    double& val,   //!< value of linearization of weak laplacian
    const int vi,  //!< node index for the weighting function
    const int ui   //!< node index for the shape function
)
{
  val = 0.0;
  for (int j = 0; j < nsd_; j++)
  {
    val += derxy_(j, vi) * derxy_(j, ui);
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculate linearization of the Laplacian (weak form)
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::get_laplacian_weak_form(
    double& val,   //!< value of linearization of weak laplacian
    const int vi,  //!< node index for the weighting function
    const int ui,  //!< node index for the shape function
    const Core::LinAlg::Matrix<nsd_, 1> pflowfac)
{
  // FOUR_C_THROW("we should not be here");
  val = 0.0;
  for (int j = 0; j < nsd_; j++)
  {
    val += derxy_(j, vi) * derxy_(j, ui) * pflowfac(j);
  }

  return;
}

/*------------------------------------------------------------------- *
 |  calculate the Laplacian (weak form)
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::get_laplacian_weak_form_rhs(
    double& val,                                   //!< value of weak laplacian
    const Core::LinAlg::Matrix<nsd_, 1>& gradpre,  //!< pressure gradient at gauss point
    const int vi                                   //!< node index for the weighting function
)
{
  val = 0.0;
  for (int j = 0; j < nsd_; j++)
  {
    val += derxy_(j, vi) * gradpre(j);
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculate the Laplacian (weak form)
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::get_laplacian_weak_form_rhs(
    double& val,                                   //!< value of weak laplacian
    const Core::LinAlg::Matrix<nsd_, 1>& gradpre,  //!< pressure gradient at gauss point
    const int vi,                                  //!< node index for the weighting function
    const Core::LinAlg::Matrix<nsd_, 1> pflowfac)
{
  // FOUR_C_THROW("we should not be here");
  val = 0.0;
  for (int j = 0; j < nsd_; j++)
  {
    val += derxy_(j, vi) * gradpre(j) * pflowfac(j);
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of Poiseuille element matrix
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_mat_psl(
    Core::LinAlg::SerialDenseMatrix& emat, const double timefacfac, const double viscosity,
    const double height)
{
  // Poiseuille term
  const double fac_psl = timefacfac * (1 / (12 * viscosity)) * height * height * height;
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      double laplawf(0.0);
      get_laplacian_weak_form(laplawf, ui, vi);
      emat(vi, ui) -= fac_psl * laplawf;
    }
  }
  return;
}

/*-----------------------------------------------------------------*
 | contribution of pressure dependent viscosity                    |
 *-----------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_mat_psl_vis(
    Core::LinAlg::SerialDenseMatrix& emat, const double timefacfac, const double viscosity,
    const double height, const double dviscosity_dp)
{
  // Poiseuille term (pressure dependent viscosity)

  const double fac_pslvisc =
      timefacfac * (1 / (12 * viscosity * viscosity)) * height * height * height * dviscosity_dp;
  const Core::LinAlg::Matrix<nsd_, 1>& gradpre = lubricationvarmanager_->grad_pre();

  for (int vi = 0; vi < nen_; ++vi)
  {
    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, gradpre, vi);

    for (int ui = 0; ui < nen_; ++ui)
    {
      emat(vi, ui) += fac_pslvisc * laplawf * funct_(ui);
    }
  }
  return;
}


/*------------------------------------------------------------------- *
 |  calculation of Poiseuille element matrix           Faraji  02/19  |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_mat_psl(
    Core::LinAlg::SerialDenseMatrix& emat, const double timefacfac, const double viscosity,
    const double height, const Core::LinAlg::Matrix<nsd_, 1> pflowfac)
{
  if (!(lubricationpara_->modified_reynolds()))
    FOUR_C_THROW("There is no pressure flow factor in classical reynolds Equ.");

  // Poiseuille term
  const double fac_psl = timefacfac * (1 / (12 * viscosity)) * height * height * height;
  for (int vi = 0; vi < nen_; ++vi)
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      double laplawf(0.0);
      get_laplacian_weak_form(laplawf, ui, vi, pflowfac);
      emat(vi, ui) -= fac_psl * laplawf;
    }
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of Poiseuille rhs matrix
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rhs_psl(
    Core::LinAlg::SerialDenseVector& erhs, const double rhsfac, const double viscosity,
    const double height)
{
  // Poiseuille rhs term
  const double fac_rhs_psl = rhsfac * (1 / (12 * viscosity)) * height * height * height;

  const Core::LinAlg::Matrix<nsd_, 1>& gradpre = lubricationvarmanager_->grad_pre();

  for (int vi = 0; vi < nen_; ++vi)
  {
    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, gradpre, vi);
    erhs[vi] += fac_rhs_psl * laplawf;
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of Poiseuille rhs matrix              Faraji   02/19  |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rhs_psl(
    Core::LinAlg::SerialDenseVector& erhs, const double rhsfac, const double viscosity,
    const double height, const Core::LinAlg::Matrix<nsd_, 1> pflowfac)
{
  if (!(lubricationpara_->modified_reynolds()))
    FOUR_C_THROW("There is no pressure flow factor in classical reynolds Equ.");
  // Poiseuille rhs term
  const double fac_rhs_psl = rhsfac * (1 / (12 * viscosity)) * height * height * height;

  const Core::LinAlg::Matrix<nsd_, 1>& gradpre = lubricationvarmanager_->grad_pre();

  for (int vi = 0; vi < nen_; ++vi)
  {
    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, gradpre, vi, pflowfac);
    erhs[vi] += fac_rhs_psl * laplawf;
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of Wedge rhs matrix
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rhs_wdg(
    Core::LinAlg::SerialDenseVector& erhs, const double rhsfac, const double height,
    const Core::LinAlg::Matrix<nsd_, 1> velocity)
{
  // Wedge rhs term
  const double fac_rhs_wdg = rhsfac * height;

  for (int vi = 0; vi < nen_; ++vi)
  {
    double val(0.0);

    for (int i = 0; i < nsd_; ++i)
    {
      val += derxy_(i, vi) * velocity(i);
    }
    erhs[vi] -= fac_rhs_wdg * val;
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of Squeeze rhs matrix                         Faraji  |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rhs_sqz(
    Core::LinAlg::SerialDenseVector& erhs, const double rhsfac, const double heightdot)
{
  if (!lubricationpara_->add_sqz())
    FOUR_C_THROW("You chosed NOT to add the squeeze term! WATCHOUT");
  // Squeeze rhs term
  const double fac_rhs_sqz = rhsfac * heightdot;

  for (int vi = 0; vi < nen_; ++vi)
  {
    erhs[vi] += fac_rhs_sqz * funct_(vi);
  }
  return;
}

/*------------------------------------------------------------------- *
 |  calculation of shear rhs matrix                     Faraji 02/19  |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rhs_shear(
    Core::LinAlg::SerialDenseVector& erhs, const double rhsfac,
    const Core::LinAlg::Matrix<nsd_, 1> velocity, const double sflowfac)
{
  if (!(lubricationpara_->modified_reynolds()))
    FOUR_C_THROW("There is no shear term in classical reynolds Equ.");
  // shear rhs term
  const double roughness = lubricationpara_->roughness_deviation();
  const double fac_rhs_shear = rhsfac * roughness;

  for (int vi = 0; vi < nen_; ++vi)
  {
    double val(0.0);

    for (int i = 0; i < nsd_; ++i)
    {
      val += derxy_(i, vi) * velocity(i) * sflowfac;
    }
    erhs[vi] -= fac_rhs_shear * val;
  }
  return;
}

/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives at int. point   wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double
Discret::Elements::LubricationEleCalc<distype, probdim>::eval_shape_func_and_derivs_at_int_point(
    const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints,  ///< integration points
    const int iquad                                            ///< id of current Gauss point
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (int idim = 0; idim < nsd_ele_; idim++) xsi_(idim) = gpcoord[idim];

  const double det = eval_shape_func_and_derivs_in_parameter_space();

  if (det < 1E-16)
    FOUR_C_THROW("GLOBAL ELEMENT NO. %d \nZERO OR NEGATIVE JACOBIAN DETERMINANT: %lf", eid_, det);

  // compute global spatial derivatives
  derxy_.multiply(xij_, deriv_);

  // set integration factor: fac = Gauss weight * det(J)
  const double fac = intpoints.ip().qwgt[iquad] * det;

  // return integration factor for current GP: fac = Gauss weight * det(J)
  return fac;

}  // LubricationImpl::eval_shape_func_and_derivs_at_int_point

/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives in parameter space wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::LubricationEleCalc<distype,
    probdim>::eval_shape_func_and_derivs_in_parameter_space()
{
  double det = 0.0;

  if (nsd_ == nsd_ele_)  // standard case
  {
    // shape functions and their first derivatives
    Core::FE::shape_function<distype>(xsi_, funct_);
    Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);


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
  else  // element dimension is smaller than problem dimension -> mannifold
  {
    static Core::LinAlg::Matrix<nsd_ele_, nen_> deriv_red;

    // shape functions and their first derivatives
    Core::FE::shape_function<distype>(xsi_, funct_);
    Core::FE::shape_function_deriv1<distype>(xsi_, deriv_red);

    //! metric tensor at integration point
    static Core::LinAlg::Matrix<nsd_ele_, nsd_ele_> metrictensor;
    static Core::LinAlg::Matrix<nsd_, 1> normalvec;

    // the metric tensor and the area of an infinitesimal surface/line element
    // optional: get unit normal at integration point as well
    const bool throw_error_if_negative_determinant(true);
    Core::FE::compute_metric_tensor_for_boundary_ele<distype, nsd_>(
        xyze_, deriv_red, metrictensor, det, throw_error_if_negative_determinant, &normalvec);

    if (det < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO. %d \nZERO OR NEGATIVE JACOBIAN DETERMINANT: %lf", eid_, det);

    // transform the derivatives and Jacobians to the higher dimensional coordinates(problem
    // dimension)
    static Core::LinAlg::Matrix<nsd_ele_, nsd_> xjm_red;
    xjm_red.multiply_nt(deriv_red, xyze_);

    for (int i = 0; i < nsd_; i++)
    {
      for (int j = 0; j < nsd_ele_; j++) xjm_(j, i) = xjm_red(j, i);
      xjm_(nsd_ele_, i) = normalvec(i, 0);
    }

    for (int i = 0; i < nen_; i++)
    {
      for (int j = 0; j < nsd_ele_; j++) deriv_(j, i) = deriv_red(j, i);
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

      for (int i = 0; i < nen_; i++) deriv_(2, i) = 0.0;
    }

    xij_.invert(xjm_);
  }

  return det;
}

/*-----------------------------------------------------------------------*
  |  get the lubrication height interpolated at the Int Point
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_height_at_int_point(
    double& heightint  //!< lubrication height at Int point
)
{
  // interpolate the height at the integration point
  for (int j = 0; j < nen_; j++)
  {
    heightint +=
        funct_(j) * eheinp_(0, j);  // Note that the same value is stored for all space dimensions
  }

  return;
}  // ReynoldsEleCalc::calc_height_at_int_point

/*-----------------------------------------------------------------------*
  |  get the lubrication heightDot interpolated at the Int Point    Faraji|
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_height_dot_at_int_point(
    double& heightdotint  //!< lubrication heightDot at Int point
)
{
  // interpolate the heightDot at the integration point
  for (int j = 0; j < nen_; j++)
  {
    heightdotint +=
        funct_(j) *
        eheidotnp_(0, j);  // Note that the same value is stored for all space dimensions
  }

  return;
}

/*-----------------------------------------------------------------------*
  |  get the average velocity of the contacting bodies interpolated at   |
  |  the Int Point
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_avr_vel_at_int_point(
    Core::LinAlg::Matrix<nsd_, 1>& avrvel  //!< average surface velocity at Int point
)
{
  // interpolate the velocities at the integration point
  avrvel.multiply(1., eAvTangVel_, funct_, 0.);

  return;
}

/*-----------------------------------------------------------------------*
  |  get the relative velocity of the contacting bodies interpolated at   |
  |  the Int Point
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calc_rel_vel_at_int_point(
    Core::LinAlg::Matrix<nsd_, 1>& relvel  //!< relative surface velocity at Int point
)
{
  // interpolate the velocities at the integration point
  relvel.multiply(1., eRelTangVel_, funct_, 0.);

  return;
}

/*----------------------------------------------------------------------*
 | evaluate service routine                                 wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::LubricationEleCalc<distype, probdim>::evaluate_service(
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

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<FourC::Lubrication::Action>(params, "action");

  // evaluate action
  evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate action                                          wirtz 10/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::LubricationEleCalc<distype, probdim>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const FourC::Lubrication::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  //(for now) only first dof set considered
  const std::vector<int>& lm = la[0].lm_;
  // determine and evaluate action
  switch (action)
  {
    case FourC::Lubrication::calc_error:
    {
      // check if length suffices
      if (elevec1_epetra.length() < 1) FOUR_C_THROW("Result vector too short");

      // need current solution
      std::shared_ptr<const Core::LinAlg::Vector<double>> prenp = discretization.get_state("prenp");
      if (prenp == nullptr) FOUR_C_THROW("Cannot get state vector 'prenp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*prenp, eprenp_, lm);

      cal_error_compared_to_analyt_solution(ele, params, elevec1_epetra);

      break;
    }

    case FourC::Lubrication::calc_mean_pressures:
    {
      // get flag for inverting
      bool inverting = params.get<bool>("inverting");

      // need current pressure vector
      // -> extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> prenp = discretization.get_state("prenp");
      if (prenp == nullptr) FOUR_C_THROW("Cannot get state vector 'prenp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*prenp, eprenp_, lm);

      // calculate pressures and domain integral
      calculate_pressures(ele, elevec1_epetra, inverting);

      break;
    }

    default:
    {
      FOUR_C_THROW("Not acting on this action. Forgot implementation?");
      break;
    }
  }  // switch(action)

  return 0;
}

/*----------------------------------------------------------------------*
  |  calculate error compared to analytical solution        wirtz 10/15 |
  *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::cal_error_compared_to_analyt_solution(
    const Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& errors)
{
  if (Teuchos::getIntegralValue<FourC::Lubrication::Action>(params, "action") !=
      FourC::Lubrication::calc_error)
    FOUR_C_THROW("How did you get here?");

  // -------------- prepare common things first ! -----------------------
  // set constants for analytical solution
  const double t = lubricationpara_->time();

  // integration points and weights
  // more GP than usual due to (possible) cos/exp fcts in analytical solutions
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      FourC::Lubrication::DisTypeToGaussRuleForExactSol<distype>::rule);

  const auto errortype =
      Teuchos::getIntegralValue<Inpar::Lubrication::CalcError>(params, "calcerrorflag");
  switch (errortype)
  {
    case Inpar::Lubrication::calcerror_byfunction:
    {
      const int errorfunctno = params.get<int>("error function number");

      // analytical solution
      double pre_exact(0.0);
      double deltapre(0.0);
      //! spatial gradient of current pressure value
      Core::LinAlg::Matrix<nsd_, 1> gradpre(true);
      Core::LinAlg::Matrix<nsd_, 1> gradpre_exact(true);
      Core::LinAlg::Matrix<nsd_, 1> deltagradpre(true);

      // start loop over integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; iquad++)
      {
        const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

        // get coordinates at integration point
        // gp reference coordinates
        Core::LinAlg::Matrix<nsd_, 1> xyzint(true);
        xyzint.multiply(xyze_, funct_);

        // function evaluation requires a 3D position vector!!
        double position[3] = {0.0, 0.0, 0.0};

        for (int dim = 0; dim < nsd_; ++dim) position[dim] = xyzint(dim);

        // pressure at integration point at time step n+1
        const double prenp = funct_.dot(eprenp_);
        // spatial gradient of current pressure value
        gradpre.multiply(derxy_, eprenp_);

        pre_exact = Global::Problem::instance()
                        ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno - 1)
                        .evaluate(position, t, 0);

        std::vector<double> gradpre_exact_vec =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno - 1)
                .evaluate_spatial_derivative(position, t, 0);

        if (gradpre_exact_vec.size())
        {
          if (nsd_ == nsd_ele_)
            for (int dim = 0; dim < nsd_; ++dim) gradpre_exact(dim) = gradpre_exact_vec[dim];
          else
          {
            std::cout << "Warning: Gradient of analytical solution cannot be evaluated correctly "
                         "for lubrication on curved surfaces!"
                      << std::endl;
            gradpre_exact.clear();
          }
        }
        else
        {
          std::cout << "Warning: Gradient of analytical solution was not evaluated!" << std::endl;
          gradpre_exact.clear();
        }

        // error at gauss point
        deltapre = prenp - pre_exact;
        deltagradpre.update(1.0, gradpre, -1.0, gradpre_exact);

        // 0: delta pressure for L2-error norm
        // 1: delta pressure for H1-error norm
        // 2: analytical pressure for L2 norm
        // 3: analytical pressure for H1 norm

        // the error for the L2 and H1 norms are evaluated at the Gauss point

        // integrate delta pressure for L2-error norm
        errors(0) += deltapre * deltapre * fac;
        // integrate delta pressure for H1-error norm
        errors(1) += deltapre * deltapre * fac;
        // integrate analytical pressure for L2 norm
        errors(2) += pre_exact * pre_exact * fac;
        // integrate analytical pressure for H1 norm
        errors(3) += pre_exact * pre_exact * fac;

        // integrate delta pressure derivative for H1-error norm
        errors(1) += deltagradpre.dot(deltagradpre) * fac;
        // integrate analytical pressure derivative for H1 norm
        errors(3) += gradpre_exact.dot(gradpre_exact) * fac;
      }  // loop over integration points
    }
    break;
    default:
      FOUR_C_THROW("Unknown analytical solution!");
      break;
  }  // switch(errortype)

  return;
}  // Discret::Elements::LubricationEleCalc<distype,probdim>::cal_error_compared_to_analyt_solution

/*---------------------------------------------------------------------*
|  calculate pressure(s) and domain integral               wirtz 10/15 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::LubricationEleCalc<distype, probdim>::calculate_pressures(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& pressures,
    const bool inverting)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      FourC::Lubrication::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // calculate integrals of (inverted) pressure(s) and domain
    if (inverting)
    {
      for (int i = 0; i < nen_; i++)
      {
        const double fac_funct_i = fac * funct_(i);
        if (std::abs(eprenp_(i, 0)) > 1e-14)
          pressures[0] += fac_funct_i / eprenp_(i, 0);
        else
          FOUR_C_THROW("Division by zero");
        // for domain volume
        pressures[1] += fac_funct_i;
      }
    }
    else
    {
      for (int i = 0; i < nen_; i++)
      {
        const double fac_funct_i = fac * funct_(i);
        pressures[0] += fac_funct_i * eprenp_(i, 0);
        // for domain volume
        pressures[1] += fac_funct_i;
      }
    }
  }  // loop over integration points

  return;
}  // LubricationEleCalc::calculate_pressures

// template classes

// 1D elements
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line2, 1>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line2, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line2, 3>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line3, 1>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line3, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::line3, 3>;

// 2D elements
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad8, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::LubricationEleCalc<Core::FE::CellType::quad9, 3>;

FOUR_C_NAMESPACE_CLOSE
