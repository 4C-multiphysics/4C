// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_reaction.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra_reaction_coupling.hpp"
#include "4C_utils_function.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScatraReactionMat::ScatraReactionMat(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numscal_(matdata.parameters.get<int>("NUMSCAL")),
      stoich_(matdata.parameters.get<std::vector<int>>("STOICH")),
      reaccoeff_(matdata.parameters.get<double>("REACCOEFF")),
      distrfunctreaccoeffid_(matdata.parameters.get<int>("DISTRFUNCT")),
      coupling_(set_coupling_type(matdata)),
      couprole_(matdata.parameters.get<std::vector<double>>("ROLE")),
      reacstart_(matdata.parameters.get_or<std::vector<double>>(
          "REACSTART", std::vector<double>(numscal_))),
      isdistrfunctreaccoeff_(distrfunctreaccoeffid_ != 0),
      isreacstart_(false),
      isinit_(false)
{
  // Some checks for more safety
  if (coupling_ == reac_coup_none)
    FOUR_C_THROW(
        "The coupling '%s' is not a valid reaction coupling. Valid couplings are "
        "'simple_multiplicative', 'constant' and 'michaelis_menten'.",
        (matdata.parameters.get<std::string>("COUPLING")).c_str());

  if (numscal_ != (int)stoich_.size())
    FOUR_C_THROW("number of scalars %d does not fit to size of the STOICH vector %d", numscal_,
        stoich_.size());

  if (numscal_ != (int)couprole_.size())
    FOUR_C_THROW("number of scalars %d does not fit to size of the ROLE vector %d", numscal_,
        couprole_.size());

  if (numscal_ != (int)reacstart_.size())
    FOUR_C_THROW("number of scalars %d does not fit to size of the REACSTART vector %d", numscal_,
        reacstart_.size());

  for (int ii = 0; ii < numscal_; ii++)
  {
    if (reacstart_.at(ii) < 0)
    {
      FOUR_C_THROW("In the REACSTART vector only non-negative values are allowed!");
    }
    else if (reacstart_.at(ii) > 0)
    {
      isreacstart_ = true;
    }
  }

  // do some more input checks depending on coupling type
  {
    switch (coupling_)
    {
      case Mat::PAR::reac_coup_simple_multiplicative:  // reaction of type A*B*C:
      {
        bool stoichallzero = true;
        bool roleallzero = true;
        for (int ii = 0; ii < numscal_; ii++)
        {
          if (stoich_.at(ii) != 0) stoichallzero = false;
          if (couprole_.at(ii) != 0.0) roleallzero = false;
        }
        if (roleallzero)
          FOUR_C_THROW(
              "reac_coup_simple_multiplicative must contain at least one non-zero entry in the "
              "ROLE list");
        if (stoichallzero)
          FOUR_C_THROW(
              "reac_coup_simple_multiplicative must contain at least one non-zero entry in the "
              "STOICH list");

        break;
      }

      case Mat::PAR::reac_coup_power_multiplicative:  // reaction of type A^2*B^-1.5*C:
      {
        bool stoichallzero = true;
        bool roleallzero = true;
        for (int ii = 0; ii < numscal_; ii++)
        {
          if (stoich_.at(ii) != 0) stoichallzero = false;
          if (couprole_.at(ii) != 0.0) roleallzero = false;
        }
        if (roleallzero)
          FOUR_C_THROW(
              "reac_coup_power_multiplicative must contain at least one positive entry in the ROLE "
              "list");
        if (stoichallzero)
          FOUR_C_THROW(
              "reac_coup_michaelis_menten must contain at least one non-zero entry in the STOICH "
              "list");

        break;
      }

      case Mat::PAR::reac_coup_constant:  // constant source term:
      {
        bool issomepositiv = false;
        for (int ii = 0; ii < numscal_; ii++)
        {
          if (stoich_.at(ii) < 0)
            FOUR_C_THROW(
                "reac_coup_constant must only contain positive entries in the STOICH list");
          if (couprole_.at(ii) != 0.0)
            FOUR_C_THROW("reac_coup_constant must only contain zero entries in the ROLE list");
          if (stoich_.at(ii) > 0) issomepositiv = true;
        }
        if (not issomepositiv)
          FOUR_C_THROW(
              "reac_coup_constant must contain at least one positive entry in the STOICH list");

        break;
      }

      case Mat::PAR::reac_coup_michaelis_menten:  // reaction of type A*B/(B+4)
      {
        bool stoichallzero = true;
        bool roleallzero = true;
        for (int ii = 0; ii < numscal_; ii++)
        {
          if (stoich_.at(ii) != 0) stoichallzero = false;
          if (couprole_.at(ii) != 0) roleallzero = false;
        }
        if (roleallzero)
          FOUR_C_THROW(
              "reac_coup_michaelis_menten must contain at least one non-zero entry in the ROLE "
              "list");
        if (stoichallzero)
          FOUR_C_THROW(
              "reac_coup_michaelis_menten must contain at least one non-zero entry in the STOICH "
              "list");

        break;
      }

      case Mat::PAR::reac_coup_byfunction:  // reaction by function
      {
        int functID = -1;
        for (int ii = 0; ii < numscal_; ii++)
        {
          if (stoich_.at(ii) != 0)
          {
            if (round(couprole_.at(ii)) < 1)
              FOUR_C_THROW(
                  "reac_coup_byfunction: no function defined in the ROLE list for scalar with "
                  "positive entry in the STOICH list");
            if (functID == -1)
              functID = round(couprole_.at(ii));
            else if (functID != round(couprole_.at(ii)))
              FOUR_C_THROW("The FUNC IDs defined in the ROLE list should all match");
          }
        }
        if (functID == -1)
          FOUR_C_THROW(
              "reac_coup_byfunction must contain at least one positive entry in the STOICH list");

        break;
      }

      case Mat::PAR::reac_coup_none:
        FOUR_C_THROW("reac_coup_none is not a valid coupling");
        break;

      default:
        FOUR_C_THROW("The couplingtype %i is not a valid coupling type.", coupling_);
        break;
    }
  }

  // if all checks are passed, we can build the reaction class
  reaction_ = Mat::PAR::REACTIONCOUPLING::ReactionInterface::create_reaction(
      coupling_, isreacstart_, reacstart_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::ScatraReactionMat::initialize() { reaction_->initialize(numscal_, couprole_); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraReactionMat::create_material()
{
  return std::make_shared<Mat::ScatraReactionMat>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::ScatraReactionMatType Mat::ScatraReactionMatType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::ReactionCoupling Mat::PAR::ScatraReactionMat::set_coupling_type(
    const Core::Mat::PAR::Parameter::Data& matdata)
{
  if ((matdata.parameters.get<std::string>("COUPLING")) == "simple_multiplicative")
  {
    return reac_coup_simple_multiplicative;
  }
  else if ((matdata.parameters.get<std::string>("COUPLING")) == "power_multiplicative")
  {
    return reac_coup_power_multiplicative;
  }
  else if ((matdata.parameters.get<std::string>("COUPLING")) == "constant")
  {
    return reac_coup_constant;
  }
  else if ((matdata.parameters.get<std::string>("COUPLING")) == "michaelis_menten")
  {
    return reac_coup_michaelis_menten;
  }
  else if ((matdata.parameters.get<std::string>("COUPLING")) == "by_function")
  {
    return reac_coup_byfunction;
  }
  else if ((matdata.parameters.get<std::string>("COUPLING")) == "no_coupling")
  {
    return reac_coup_none;
  }
  else
  {
    return reac_coup_none;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ScatraReactionMatType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraReactionMat* scatra_reaction_mat = new Mat::ScatraReactionMat();
  scatra_reaction_mat->unpack(buffer);
  return scatra_reaction_mat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraReactionMat::ScatraReactionMat() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraReactionMat::ScatraReactionMat(Mat::PAR::ScatraReactionMat* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraReactionMat::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraReactionMat::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::ScatraReactionMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }
}

/*-------------------------------------------------------------------------------/
 | return reaction coefficient at Gauss-point                 Brandstaeter 11/16 |
/-------------------------------------------------------------------------------*/
double Mat::ScatraReactionMat::reac_coeff(const std::vector<std::pair<std::string, double>>&
        constants  //!< vector containing values which are independent of the scalars
) const
{
  double reaccoeff = params_->reaccoeff_;

  if (get_is_distr_funct_reac_coeff())
  {
    // get time and coordinates
    // Note: we get them counting from the back, since we have added them last (and in exactly this
    // order!)
    const unsigned size = constants.size();

    const double time = constants[size - 4].second;

    double gpcoord[3] = {0.0, 0.0, 0.0};
    gpcoord[0] = constants[size - 3].second;
    gpcoord[1] = constants[size - 2].second;
    gpcoord[2] = constants[size - 1].second;

    reaccoeff *=
        (Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(dis_funct_reac_coeff_id() - 1)
                .evaluate(gpcoord, time, 0));
  }

  return reaccoeff;
}

/*----------------------------------------------------------------------/
 | calculate advanced reaction terms                        Thon 08/16 |
/----------------------------------------------------------------------*/
double Mat::ScatraReactionMat::calc_rea_body_force_term(const int k,  //!< current scalar id
    const std::vector<double>& phinp,                                 //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,    //!< vector containing values which are independent of the scalars
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  const double reaccoeff = reac_coeff(constants);

  if (stoich()->at(k) != 0 and fabs(reaccoeff) > 1.0e-14)
  {
    return calc_rea_body_force_term(k, phinp, constants, reaccoeff * stoich()->at(k),
        scale_phi);  // scalar at integration point np
  }
  else
    return 0.0;
}

/*----------------------------------------------------------------------/
 | calculate advanced reaction term derivatives             Thon 08/16 |
/----------------------------------------------------------------------*/
void Mat::ScatraReactionMat::calc_rea_body_force_deriv_matrix(const int k,  //!< current scalar id
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,    //!< vector containing values which are independent of the scalars
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  const double reaccoeff = reac_coeff(constants);

  if (stoich()->at(k) != 0 and fabs(reaccoeff) > 1.0e-14)
  {
    calc_rea_body_force_deriv(k, derivs, phinp, constants, reaccoeff * stoich()->at(k), scale_phi);
  }

  return;
}

/*--------------------------------------------------------------------------------*
 |  calculate advanced reaction term derivatives after additional variables       |
 |  (e.g. for monolithic coupling with by-function reaction)     kremheller 07/17 |
 *--------------------------------------------------------------------------------*/
void Mat::ScatraReactionMat::calc_rea_body_force_deriv_matrix_add_variables(
    const int k,                  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,    //!< constants (including the scalar values phinp)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  const double reaccoeff = reac_coeff(constants);

  if (stoich()->at(k) != 0 and fabs(reaccoeff) > 1.0e-14)
  {
    calc_rea_body_force_deriv_add_variables(
        k, derivs, variables, constants, reaccoeff * stoich()->at(k), scale_phi);
  }

  return;
}

/*----------------------------------------------------------------------/
 | add variables to the reaction (only by-function)    kremheller 07/17 |
/----------------------------------------------------------------------*/
void Mat::ScatraReactionMat::add_additional_variables(const int k,  //!< current scalar id
    const std::vector<std::pair<std::string, double>>& variables    //!< variables
) const
{
  if (stoich()->at(k) != 0)
  {
    params_->reaction_->add_additional_variables(k, variables, *couprole());
  }

  return;
}

/*----------------------------------------------------------------------*
 |  helper for calculating advanced reaction terms           thon 08/16 |
 *----------------------------------------------------------------------*/
double Mat::ScatraReactionMat::calc_rea_body_force_term(int k,  //!< current scalar id
    const std::vector<double>& phinp,                           //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  return params_->reaction_->calc_rea_body_force_term(
      k, num_scal(), phinp, constants, *couprole(), scale_reac, scale_phi);
}

/*--------------------------------------------------------------------------------*
 |  helper for calculating advanced reaction term derivatives          thon 08/16 |
 *--------------------------------------------------------------------------------*/
void Mat::ScatraReactionMat::calc_rea_body_force_deriv(int k,  //!< current scalar id
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< vector containing values which are independent of the scalars
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  params_->reaction_->calc_rea_body_force_deriv(
      k, num_scal(), derivs, phinp, constants, *couprole(), scale_reac, scale_phi);

  return;
}

/*--------------------------------------------------------------------------------*
 |  calculate advanced reaction term derivatives after additional variables       |
 |  (e.g. for monolithic coupling with by-function reaction)     kremheller 07/17 |
 *--------------------------------------------------------------------------------*/
void Mat::ScatraReactionMat::calc_rea_body_force_deriv_add_variables(int k,  //!< current scalar id
    std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
    const std::vector<std::pair<std::string, double>>& variables,  //!< variables
    const std::vector<std::pair<std::string, double>>&
        constants,  //!< constants (including the scalar values phinp)
    double
        scale_reac,   //!< scaling factor for reaction term (= reaction coefficient * stoichometry)
    double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
) const
{
  params_->reaction_->calc_rea_body_force_deriv_add_variables(
      k, derivs, variables, constants, *couprole(), scale_reac, scale_phi);

  return;
}

/*---------------------------------------------------------------------------------/
 | Calculate influence factor for scalar dependent membrane transport   Thon 08/16 |
/--------------------------------------------------------------------------------- */
double Mat::ScatraReactionMat::calc_perm_influence(const int k,  //!< current scalar id
    const std::vector<double>& phinp,                            //!< scalar values at t_(n+1)
    const double time,                                           //!< current time
    const double* gpcoord,                                       //!< Gauss-point coordinates
    const double scale  //!< scaling factor for reference concentrations
) const
{
  // set time and space coordinates
  std::vector<std::pair<std::string, double>> constants;
  constants.push_back(std::pair<std::string, double>("t", time));
  constants.push_back(std::pair<std::string, double>("x", gpcoord[0]));
  constants.push_back(std::pair<std::string, double>("y", gpcoord[1]));
  constants.push_back(std::pair<std::string, double>("z", gpcoord[2]));

  if (not(stoich()->at(k) > 0))
    FOUR_C_THROW("You need to specify a positive STOICH entry for scalar %i", k);
  if (fabs(reac_coeff(constants)) > 1.0e-14) FOUR_C_THROW("You need to set REACOEFF to 0.0!");

  return (calc_rea_body_force_term(k, phinp, constants, stoich()->at(k), scale));
}

/*---------------------------------------------------------------------------------/
 | Calculate influence factor for scalar dependent membrane transport   Thon 08/16 |
/--------------------------------------------------------------------------------- */
void Mat::ScatraReactionMat::calc_perm_influence_deriv(const int k,  //!< current scalar id
    std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
    const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
    const double time,                 //!< current time
    const double* gpcoord,             //!< Gauss-point coordinates
    const double scale                 //!< scaling factor for reference concentrations
) const
{
  // set time and space coordinates
  std::vector<std::pair<std::string, double>> constants;
  constants.push_back(std::pair<std::string, double>("t", time));
  constants.push_back(std::pair<std::string, double>("x", gpcoord[0]));
  constants.push_back(std::pair<std::string, double>("y", gpcoord[1]));
  constants.push_back(std::pair<std::string, double>("z", gpcoord[2]));

  calc_rea_body_force_deriv(k, derivs, phinp, constants, stoich()->at(k), scale);
}

FOUR_C_NAMESPACE_CLOSE
