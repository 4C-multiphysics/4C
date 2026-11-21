// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_reaction_coupling.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * factory method                                            vuong 09/16
 *----------------------------------------------------------------------*/
std::shared_ptr<Mat::PAR::REACTIONCOUPLING::ReactionInterface>
Mat::PAR::REACTIONCOUPLING::ReactionInterface::create_reaction(
    Mat::PAR::ReactionCoupling couplingtype,  //!< coupling type defining reaction
    bool isreacstart,                         //!< flag for reaction start feature
    const std::vector<double>& reacstart      //!< reaction start vector
)
{
  std::shared_ptr<Mat::PAR::REACTIONCOUPLING::ReactionInterface> tmpreaction = nullptr;

  switch (couplingtype)
  {
    case Mat::PAR::reac_coup_simple_multiplicative:  // reaction of type A*B*C:
    {
      tmpreaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::SimpleMultiplicative>();
      break;
    }
    case Mat::PAR::reac_coup_power_multiplicative:  // reaction of type A^2*B^-1.5*C:
    {
      tmpreaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::PowerMultiplicative>();
      break;
    }
    case Mat::PAR::reac_coup_constant:  // constant source term:
    {
      tmpreaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::Constant>();
      break;
    }
    case Mat::PAR::reac_coup_michaelis_menten:  // reaction of type A*B/(B+4)
    {
      tmpreaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::MichaelisMenten>();
      break;
    }
    case Mat::PAR::reac_coup_byfunction:  // reaction by function
    {
      tmpreaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::ByFunction>();
      break;
    }
    case Mat::PAR::reac_coup_none:
      FOUR_C_THROW("reac_coup_none is not a valid coupling");
      break;
    default:
      FOUR_C_THROW("The couplingtype {} is not a valid coupling type.", couplingtype);
      break;
  }

  // we always use potentially scaled phis for the reactions (for reference concentrations)
  std::shared_ptr<Mat::PAR::REACTIONCOUPLING::ReactionInterface> scaledreaction =
      std::make_shared<Mat::PAR::REACTIONCOUPLING::ReactionWithPhiScaling>(tmpreaction);

  std::shared_ptr<Mat::PAR::REACTIONCOUPLING::ReactionInterface> reaction = nullptr;
  // in case of reac start feature, wrap it one more time
  if (isreacstart)
    reaction = std::make_shared<Mat::PAR::REACTIONCOUPLING::ReacStart>(scaledreaction, reacstart);
  else
    reaction = scaledreaction;

  return reaction;
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  access method for calculating advanced reaction terms    vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::calc_rea_body_force_term(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if necessary (e.g. for reference concentrations)
  std::vector<double> phinp_mod(modify_phi(phinp, scale_phi));

  // call the real evaluation
  return reaction_->calc_rea_body_force_term(
      k, numscal, phinp_mod, constants, couprole, scale_reac, scale_phi);
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::calc_rea_body_force_deriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if necessary (e.g. for reference concentrations)
  std::vector<double> phinp_mod(modify_phi(phinp, scale_phi));

  // call the real evaluation
  reaction_->calc_rea_body_force_deriv(
      k, numscal, derivs, phinp_mod, constants, couprole, scale_reac, scale_phi);

  return;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives after additional    |
 |  variables (e.g. for monolithic coupling)                     kremheller 07/17 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::calc_rea_body_force_deriv_add_variables(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector if necessary (e.g. for reference concentrations)
  // std::vector<double> phinp_mod(modify_phi(phinp,scale_phi));
  if (fabs(scale_phi - 1.0) > 1.0e-14)
  {
    FOUR_C_THROW("scale_phi is not equal to 1.0, you should make your own modify phi function");
  }

  // call the real evaluation
  reaction_->calc_rea_body_force_deriv_add_variables(
      k, derivs, variables, constants, couprole, scale_reac, scale_phi);

  return;
}

/*----------------------------------------------------------------------------------*
 |  Modify concentrations according to scaling                         vuong 09/16 |
 *----------------------------------------------------------------------------------*/
std::vector<double> Mat::PAR::REACTIONCOUPLING::ReactionWithPhiScaling::modify_phi(
    const std::vector<double>& phinp, double scale_phi)
{
  // copy the vector
  std::vector<double> phinp_mod(phinp);

  if (scale_phi != 1.0)
    // scale the vector and save the result in phinp_mod
    std::transform(phinp.begin(), phinp.end(), phinp_mod.begin(),
        [&](const double& ele) { return ele * scale_phi; });

  return phinp_mod;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ReacStart::initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                              //!< coupling role vector
)
{
  reaction_->initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  access method for calculating advanced reaction terms    vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::ReacStart::calc_rea_body_force_term(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector for reaction start feature
  std::vector<double> phinp_mod(modify_phi(phinp));

  // call the real evaluation
  return reaction_->calc_rea_body_force_term(
      k, numscal, phinp_mod, constants, couprole, scale_reac, scale_phi);
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ReacStart::calc_rea_body_force_deriv(int k,  //!< current scalar id
    int numscal,                                                              //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
)
{
  // modify the phinp vector for reaction start feature
  std::vector<double> phinp_mod(modify_phi(phinp));

  // temporary vector of same size as derivs
  std::vector<double> myderivs(derivs.size(), 0.0);

  // call reaction evaluation
  reaction_->calc_rea_body_force_deriv(
      k, numscal, myderivs, phinp_mod, constants, couprole, scale_reac, scale_phi);

  for (int toderive = 0; toderive < numscal; toderive++)
  {
    // only copy the value if reaction has already started
    if (not(reacstart_[toderive] > 0 and phinp_mod[toderive] == 0.0))
      derivs[toderive] += myderivs[toderive];
  }

  return;
}

/*----------------------------------------------------------------------------------*
 |  Modify concentrations according to reacstart vector                  vuong 09/16 |
 *----------------------------------------------------------------------------------*/
std::vector<double> Mat::PAR::REACTIONCOUPLING::ReacStart::modify_phi(
    const std::vector<double>& phinp)
{
  // copy the vector
  std::vector<double> phinp_mod(phinp);

  for (unsigned int ii = 0; ii < phinp_mod.size(); ii++)
  {
    phinp_mod[ii] -= reacstart_[ii];
    if (phinp_mod[ii] < 0.0) phinp_mod[ii] = 0.0;
  }

  return phinp_mod;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::SimpleMultiplicative::initialize(
    int numscal,                         //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  ReactionBase::initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::SimpleMultiplicative::calc_rea_body_force_term(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] != 0)
    {
      bftfac *= phinp[ii];
    }
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::SimpleMultiplicative::calc_rea_body_force_deriv(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    const double
        scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    if (couprole[toderive] != 0)
    {
      for (int ii = 0; ii < numscal; ii++)
      {
        if (couprole[ii] != 0 and ii != toderive) bfdmfac *= phinp[ii];
      }
    }
    else
      bfdmfac = 0.0;

    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::PowerMultiplicative::initialize(
    int numscal,                         //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  ReactionBase::initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::PowerMultiplicative::calc_rea_body_force_term(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] != 0)
    {
      bftfac *= std::pow(phinp[ii], couprole[ii]);
    }
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::PowerMultiplicative::calc_rea_body_force_deriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    if (couprole[toderive] != 0)
    {
      for (int ii = 0; ii < numscal; ii++)
      {
        if (couprole[ii] != 0 and ii != toderive)
          bfdmfac *= std::pow(phinp[ii], couprole[ii]);
        else if (couprole[ii] != 0 and ii == toderive)
          bfdmfac *= couprole[ii] * std::pow(phinp[ii], couprole[ii] - 1.0);
      }
    }
    else
      bfdmfac = 0.0;

    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::Constant::initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                             //!< coupling role vector
)
{
  ReactionBase::initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::Constant::calc_rea_body_force_term(int k,  //!< current scalar id
    int numscal,                                                              //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  return scale_reac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::Constant::calc_rea_body_force_deriv(int k,  //!< current scalar id
    int numscal,                                                             //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // zero derivative -> do nothing
  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::MichaelisMenten::initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                                    //!< coupling role vector
)
{
  ReactionBase::initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           vuong 09/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::MichaelisMenten::calc_rea_body_force_term(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  double bftfac = 1.0;

  for (int ii = 0; ii < numscal; ii++)
  {
    if (couprole[ii] > 0.0)  // and (ii!=k))
      bftfac *= phinp[ii] / (couprole[ii] + phinp[ii]);
    else if (couprole[ii] < 0.0)  // and (ii!=k))
      bftfac *= phinp[ii];
  }

  return scale_reac * bftfac;
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          vuong 09/16 |
 *--------------------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::MichaelisMenten::calc_rea_body_force_deriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  for (int toderive = 0; toderive < numscal; toderive++)
  {
    double bfdmfac = 1.0;
    for (int ii = 0; ii < numscal; ii++)
    {
      if (ii != toderive)
      {
        if (couprole[ii] > 0.0)
          bfdmfac *= phinp[ii] / (couprole[ii] + phinp[ii]);
        else if (couprole[ii] < 0.0)
          bfdmfac *= phinp[ii];
        else
          bfdmfac *= 1;
      }
      else
      {
        if (couprole[ii] > 0.0)
          bfdmfac *= couprole[ii] / (std::pow((phinp[ii] + couprole[ii]), 2));
        else if (couprole[ii] < 0.0)
          bfdmfac *= 1;
        else
          bfdmfac = 0;
      }
    }
    derivs[toderive] += scale_reac * bfdmfac;
  }

  return;
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ByFunction::initialize(int numscal,  //!< number of scalars
    const std::vector<double>& couprole                               //!< coupling role vector
)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return initialize_internal<1>(numscal, couprole);
    case 2:
      return initialize_internal<2>(numscal, couprole);

    case 3:
      return initialize_internal<3>(numscal, couprole);

    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::REACTIONCOUPLING::ByFunction::initialize_internal(
    int numscal,                         //!< number of scalars
    const std::vector<double>& couprole  //!< coupling role vector
)
{
  if (not is_init())
  {
    variables_.reserve(numscal);
    for (int ii = 0; ii < numscal; ii++)
    {
      // we take the value in couprole list as function ID
      const int functID = round(couprole[ii]);
      if (functID != 0)
      {
        if (Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfAnything>(functID)
                .number_components() != 1)
          FOUR_C_THROW("expected only one component for the reaction evaluation");

        for (int k = 0; k < numscal; k++)
        {
          // construct the strings for scalar
          std::string name = "phi" + std::to_string(k + 1);

          // save the phi values with correct name
          variables_.emplace_back(name, 0.0);
        }
      }
    }
  }


  // call base class
  ReactionBase::initialize(numscal, couprole);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_term(
    const int k,                       //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                       //!< stoichometry)
)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return calc_rea_body_force_term_internal<1>(
          k, numscal, phinp, constants, couprole, scale_reac);
    case 2:
      return calc_rea_body_force_term_internal<2>(
          k, numscal, phinp, constants, couprole, scale_reac);

    case 3:
      return calc_rea_body_force_term_internal<3>(
          k, numscal, phinp, constants, couprole, scale_reac);

    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
      return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_term_internal(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // copy phi vector in different format to be read by the function
  build_phi_vector_for_function(phinp, numscal);

  // evaluate reaction term
  double bftfac = Global::Problem::instance()
                      ->function_by_id<Core::Utils::FunctionOfAnything>(round(couprole[k]))
                      .evaluate(variables_, constants, 0);

  return scale_reac * bftfac;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_deriv(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return calc_rea_body_force_deriv_internal<1>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
    case 2:
      return calc_rea_body_force_deriv_internal<2>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
    case 3:
      return calc_rea_body_force_deriv_internal<3>(
          k, numscal, derivs, phinp, constants, couprole, scale_reac);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_deriv_internal(
    int k,                             //!< current scalar id
    int numscal,                       //!< number of scalars
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars (e.g.
                    //!< t,x,y,z)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // copy phi vector in different format to be read by the function
  build_phi_vector_for_function(phinp, numscal);

  // evaluate the derivatives of the reaction term
  std::vector<double> myderivs =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(round(couprole[k]))
          .evaluate_derivative(variables_, constants, 0);

  // add it to derivs
  for (int toderive = 0; toderive < numscal; toderive++)
    derivs[toderive] += scale_reac * myderivs[toderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_deriv_add_variables(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return calc_rea_body_force_deriv_add_variables_internal<1>(
          k, derivs, variables, constants, couprole, scale_reac);
    case 2:
      return calc_rea_body_force_deriv_add_variables_internal<2>(
          k, derivs, variables, constants, couprole, scale_reac);
    case 3:
      return calc_rea_body_force_deriv_add_variables_internal<3>(
          k, derivs, variables, constants, couprole, scale_reac);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::REACTIONCOUPLING::ByFunction::calc_rea_body_force_deriv_add_variables_internal(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,                        //!< constants (including scalar values phinp)
    const std::vector<double>& couprole,  //!< coupling role vector
    double scale_reac  //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
)
{
  // evaluate the derivatives of the reaction term
  std::vector<double> myderivs =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(round(couprole[k]))
          .evaluate_derivative(variables, constants, 0);

  if (myderivs.size() != derivs.size())
  {
    FOUR_C_THROW("mismatch in dimensions, Input {}, Output {}", derivs.size(), myderivs.size());
  }

  // add it to derivs
  for (unsigned toderive = 0; toderive < variables.size(); toderive++)
    derivs[toderive] += scale_reac * myderivs[toderive];

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::REACTIONCOUPLING::ByFunction::add_additional_variables(
    const int k,                                                   //!< current scalar id
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<double>& couprole                            //!< coupling role vector
)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return add_additional_variables_internal<1>(k, variables, couprole);
    case 2:
      return add_additional_variables_internal<2>(k, variables, couprole);
    case 3:
      return add_additional_variables_internal<3>(k, variables, couprole);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::REACTIONCOUPLING::ByFunction::add_additional_variables_internal(
    const int k,                                                   //!< current scalar id
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<double>& couprole                            //!< coupling role vector
)
{
  // nothing to do

  return;
}

/*---------------------------------------------------------------------------------/
 | helper for evaluation by function                                     vuong 08/16 |
/--------------------------------------------------------------------------------- */
void Mat::PAR::REACTIONCOUPLING::ByFunction::build_phi_vector_for_function(
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    int numscal                        //!< number of scalars
)
{
  // note: we use the fact that the 'variables_' vector is ordered in the same way
  //       as the phi vector!
  for (int ii = 0; ii < numscal; ii++)
  {
    variables_[ii].second = phinp[ii];
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE
