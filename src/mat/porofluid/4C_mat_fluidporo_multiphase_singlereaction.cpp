// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *  constructor (public)                               vuong 08/16      |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroSingleReaction::FluidPoroSingleReaction(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numscal_(matdata.parameters.get<int>("NUMSCAL")),
      numvolfrac_(matdata.parameters.get<int>("NUMVOLFRAC")),
      totalnummultiphasedof_(matdata.parameters.get<int>("TOTALNUMDOF")),
      numfluidphases_(
          get_numfluidphases_from_closing_relation_of_additional_porous_network_type(matdata)),
      scale_(matdata.parameters.get<std::vector<int>>("SCALE")),
      coupling_(set_coupling_type(matdata)),
      functID_(matdata.parameters.get<int>("FUNCTID")),
      isinit_(false),
      scalarnames_(numscal_),
      pressurenames_(numfluidphases_),
      saturationnames_(numfluidphases_),
      porosityname_("porosity"),
      volfracnames_(numvolfrac_),
      volfracpressurenames_(numvolfrac_)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroSingleReaction::initialize()
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return initialize_internal<1>();
    case 2:
      return initialize_internal<2>();
    case 3:
      return initialize_internal<3>();
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::FluidPoroSingleReaction::initialize_internal()
{
  // safety check
  if (Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functID_)
          .number_components() != 1)
    FOUR_C_THROW("expected only one component for single phase reaction!");

  for (int k = 0; k < numscal_; k++)
  {
    // add scalar names
    {
      scalarnames_[k] = "phi" + std::to_string(k + 1);
    }
  }

  for (int k = 0; k < numfluidphases_; k++)
  {
    // add pressure names
    pressurenames_[k] = "p" + std::to_string(k + 1);

    // add saturation names
    saturationnames_[k] = "S" + std::to_string(k + 1);
  }


  // add additional volume fractions
  for (int k = 0; k < numvolfrac_; k++)
  {
    // add volume fraction names
    volfracnames_[k] = "VF" + std::to_string(k + 1);

    // add volume fraction pressure names
    volfracpressurenames_[k] = "VFP" + std::to_string(k + 1);
  }

  isinit_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroSingleReaction::evaluate_function(std::vector<double>& reacval,
    std::vector<std::vector<double>>& reacderivspressure,
    std::vector<std::vector<double>>& reacderivssaturation, std::vector<double>& reacderivsporosity,
    std::vector<std::vector<double>>& reacderivsvolfrac,
    std::vector<std::vector<double>>& reacderivsvolfracpressure,
    std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
    const std::vector<double>& saturation, const double& porosity,
    const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
    const std::vector<double>& scalar)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return evaluate_function_internal<1>(reacval, reacderivspressure, reacderivssaturation,
          reacderivsporosity, reacderivsvolfrac, reacderivsvolfracpressure, reacderivsscalar,
          pressure, saturation, porosity, volfracs, volfracpressures, scalar);
    case 2:
      return evaluate_function_internal<2>(reacval, reacderivspressure, reacderivssaturation,
          reacderivsporosity, reacderivsvolfrac, reacderivsvolfracpressure, reacderivsscalar,
          pressure, saturation, porosity, volfracs, volfracpressures, scalar);
    case 3:
      return evaluate_function_internal<3>(reacval, reacderivspressure, reacderivssaturation,
          reacderivsporosity, reacderivsvolfrac, reacderivsvolfracpressure, reacderivsscalar,
          pressure, saturation, porosity, volfracs, volfracpressures, scalar);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::FluidPoroSingleReaction::evaluate_function_internal(std::vector<double>& reacval,
    std::vector<std::vector<double>>& reacderivspressure,
    std::vector<std::vector<double>>& reacderivssaturation, std::vector<double>& reacderivsporosity,
    std::vector<std::vector<double>>& reacderivsvolfrac,
    std::vector<std::vector<double>>& reacderivsvolfracpressure,
    std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
    const std::vector<double>& saturation, const double& porosity,
    const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
    const std::vector<double>& scalar)
{
  // safety check if sizes fit
  check_sizes(reacval, reacderivspressure, reacderivssaturation, reacderivsporosity,
      reacderivsvolfrac, reacderivsvolfracpressure, reacderivsscalar, pressure, saturation,
      porosity, volfracs, volfracpressures, scalar);

  std::vector<std::pair<std::string, double>> variables;
  variables.reserve(numfluidphases_ + numfluidphases_ + 1 + numscal_ + numvolfrac_ + numvolfrac_);

  std::vector<std::pair<std::string, double>> constants;
  constants.reserve(0);

  // set pressure values as variable
  for (int k = 0; k < numfluidphases_; k++)
    variables.push_back(std::pair<std::string, double>(pressurenames_[k], pressure[k]));

  // set saturation values as variable
  for (int k = 0; k < numfluidphases_; k++)
    variables.push_back(std::pair<std::string, double>(saturationnames_[k], saturation[k]));

  // set porosity value as variable
  variables.push_back(std::pair<std::string, double>(porosityname_, porosity));

  // set scalar values as variables
  for (int k = 0; k < numscal_; k++)
    variables.push_back(std::pair<std::string, double>(scalarnames_[k], scalar[k]));

  // set volfrac values as variables
  for (int k = 0; k < numvolfrac_; k++)
    variables.push_back(std::pair<std::string, double>(volfracnames_[k], volfracs[k]));

  // set volfrac pressure values as variables
  for (int k = 0; k < numvolfrac_; k++)
    variables.push_back(
        std::pair<std::string, double>(volfracpressurenames_[k], volfracpressures[k]));

  // evaluate the reaction term
  double curval = Global::Problem::instance()
                      ->function_by_id<Core::Utils::FunctionOfAnything>(functID_)
                      .evaluate(variables, constants, 0);
  // evaluate derivatives
  std::vector<double> curderivs(Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functID_)
          .evaluate_derivative(variables, constants, 0));

  // fill the output vector
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    const int scale = scale_[k];
    if (scale != 0)
    {
      // add the values from the reaction terms
      reacval[k] += scale * curval;

      // derivatives
      std::vector<double>& presk = reacderivspressure[k];
      std::vector<double>& satk = reacderivssaturation[k];
      for (int j = 0; j < numfluidphases_; j++)
      {
        presk[j] += scale * curderivs[j];
        satk[j] += scale * curderivs[numfluidphases_ + j];
      }
      reacderivsporosity[k] += scale * curderivs[2 * numfluidphases_];
      std::vector<double>& scalark = reacderivsscalar[k];
      for (int j = 0; j < numscal_; j++)
      {
        scalark[j] += scale * curderivs[2 * numfluidphases_ + 1 + j];
      }
      std::vector<double>& volfrack = reacderivsvolfrac[k];
      std::vector<double>& volfracpressurek = reacderivsvolfracpressure[k];
      for (int j = 0; j < numvolfrac_; j++)
      {
        volfrack[j] += scale * curderivs[2 * numfluidphases_ + 1 + numscal_ + j];
        volfracpressurek[j] +=
            scale * curderivs[2 * numfluidphases_ + 1 + numscal_ + numvolfrac_ + j];
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 *  check sizes of vectors                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroSingleReaction::check_sizes(std::vector<double>& reacval,
    std::vector<std::vector<double>>& reacderivspressure,
    std::vector<std::vector<double>>& reacderivssaturation, std::vector<double>& reacderivsporosity,
    std::vector<std::vector<double>>& reacderivsvolfrac,
    std::vector<std::vector<double>>& reacderivsvolfracpressure,
    std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
    const std::vector<double>& saturation, const double& porosity,
    const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
    const std::vector<double>& scalar)
{
  //  FOUR_C_ASSERT(pressurenames_.size()==pressure.size(),"Invalid number of pressure values for
  //  this the fluid poro reaction material!");
  //  FOUR_C_ASSERT(pressurenames_.size()==saturation.size(),"Invalid number of pressure values for
  //  this the fluid poro reaction material!");
  //  FOUR_C_ASSERT(scalarnames_.size()==scalar.size(),"Invalid number of pressure values for this
  //  the fluid poro reaction material!");

  if (numfluidphases_ != (int)pressure.size())
    FOUR_C_THROW("Invalid number of pressure values for this fluid poro reaction material!");

  if (numfluidphases_ != (int)saturation.size())
    FOUR_C_THROW("Invalid number of saturation values for this fluid poro reaction material!");
  if (numscal_ != (int)scalar.size())
    FOUR_C_THROW("Invalid number of scalar values for this fluid poro reaction material!");
  if (numvolfrac_ != (int)volfracs.size())
    FOUR_C_THROW("Invalid number of volfrac values for this fluid poro reaction material!");
  if (numvolfrac_ != (int)volfracpressures.size())
    FOUR_C_THROW("Invalid number of volfrac values for this fluid poro reaction material!");

  if (totalnummultiphasedof_ != (int)reacderivsporosity.size())
    FOUR_C_THROW(
        "Invalid length of vector for porosity derivatives for this fluid poro reaction material!");
  if (totalnummultiphasedof_ != (int)reacderivspressure.size())
    FOUR_C_THROW(
        "Invalid length of vector for pressure derivatives for this fluid poro reaction material!");
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    if (numfluidphases_ != (int)reacderivspressure[k].size())
      FOUR_C_THROW(
          "Invalid length of vector for pressure derivatives for this fluid poro reaction "
          "material!");
  }
  if (totalnummultiphasedof_ != (int)reacderivssaturation.size())
    FOUR_C_THROW(
        "Invalid length of vector for pressure derivatives for this fluid poro reaction material!");
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    if (numfluidphases_ != (int)reacderivssaturation[k].size())
      FOUR_C_THROW(
          "Invalid length of vector for pressure derivatives for this fluid poro reaction "
          "material!");
  }
  if (totalnummultiphasedof_ != (int)reacderivsscalar.size())
    FOUR_C_THROW(
        "Invalid length of vector for scalar derivatives for this fluid poro reaction material!");
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    if (numscal_ != (int)reacderivsscalar[k].size())
      FOUR_C_THROW(
          "Invalid length of vector for scalar derivatives for this fluid poro reaction material!");
  }
  if (totalnummultiphasedof_ != (int)reacderivsvolfrac.size())
    FOUR_C_THROW(
        "Invalid length of vector for vol frac derivatives for this fluid poro reaction material!");
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    if (numvolfrac_ != (int)reacderivsvolfrac[k].size())
      FOUR_C_THROW(
          "Invalid length of vector for scalar derivatives for this fluid poro reaction material!");
  }
  if (totalnummultiphasedof_ != (int)reacderivsvolfracpressure.size())
    FOUR_C_THROW(
        "Invalid length of vector for vol frac derivatives for this fluid poro reaction material!");
  for (int k = 0; k < totalnummultiphasedof_; k++)
  {
    if (numvolfrac_ != (int)reacderivsvolfracpressure[k].size())
      FOUR_C_THROW(
          "Invalid length of vector for scalar derivatives for this fluid poro reaction material!");
  }

  return;
}

/*----------------------------------------------------------------------*
 *  Create Material (public)                             vuong 08/16      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::FluidPoroSingleReaction::create_material()
{
  return std::make_shared<Mat::FluidPoroSingleReaction>(this);
}

/*----------------------------------------------------------------------*
 *  translate coupling type                             vuong 08/16      |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroSingleReaction::PorofluidReactionCoupling
Mat::PAR::FluidPoroSingleReaction::set_coupling_type(const Core::Mat::PAR::Parameter::Data& matdata)
{
  if ((matdata.parameters.get<PoroPressureBased::FluidporoReactionCoupling>("COUPLING")) ==
      PoroPressureBased::FluidporoReactionCoupling::scalar_by_function)
  {
    return porofluid_reac_coup_scalarsbyfunction;
  }
  else if ((matdata.parameters.get<PoroPressureBased::FluidporoReactionCoupling>("COUPLING")) ==
           PoroPressureBased::FluidporoReactionCoupling::no_coupling)
  {
    return porofluid_reac_coup_none;
  }
  else
  {
    return porofluid_reac_coup_none;
  }
}

int Mat::PAR::FluidPoroSingleReaction::
    get_numfluidphases_from_closing_relation_of_additional_porous_network_type(
        const Core::Mat::PAR::Parameter::Data& matdata)
{
  switch (matdata.parameters.get<Mat::PAR::PoroFluidPressureBased::ClosingRelation>(
      "VOLFRAC_CLOSING_RELATION"))
  {
    case Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung:
      return (
          matdata.parameters.get<int>("TOTALNUMDOF") - matdata.parameters.get<int>("NUMVOLFRAC"));
    case Mat::PAR::PoroFluidPressureBased::ClosingRelation::
        evolutionequation_homogenized_vasculature_tumor:
      return (matdata.parameters.get<int>("TOTALNUMDOF") -
              2 * matdata.parameters.get<int>("NUMVOLFRAC"));
    default:
      FOUR_C_THROW(
          "Your number of volfracs and number of fluidphases in multiphase porespace does not "
          "match!");
  }
}

/*----------------------------------------------------------------------*
  global instance of parameter class                   vuong 08/16     |
*----------------------------------------------------------------------*/
Mat::FluidPoroSingleReactionType Mat::FluidPoroSingleReactionType::instance_;

/*----------------------------------------------------------------------*
 *  Create material from given data                          vuong 08/16 |
 *----------------------------------------------------------------------*/

Core::Communication::ParObject* Mat::FluidPoroSingleReactionType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::FluidPoroSingleReaction* fluid_poro = new Mat::FluidPoroSingleReaction();
  fluid_poro->unpack(buffer);
  return fluid_poro;
}

/*----------------------------------------------------------------------*
 *   Create empty material                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
Mat::FluidPoroSingleReaction::FluidPoroSingleReaction() : params_(nullptr) {}

/*----------------------------------------------------------------------*
 *   Create material with parameters                         vuong 08/16 |
 *----------------------------------------------------------------------*/
Mat::FluidPoroSingleReaction::FluidPoroSingleReaction(Mat::PAR::FluidPoroSingleReaction* params)
    : params_(params)
{
}

/*----------------------------------------------------------------------*
 * pack material for communication                           vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroSingleReaction::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
}

/*----------------------------------------------------------------------*
 * unpack material                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroSingleReaction::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::FluidPoroSingleReaction*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

/*----------------------------------------------------------------------*
 *  initialize                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroSingleReaction::initialize()
{
  params_->initialize();
  return;
}

/*----------------------------------------------------------------------*
 *  set values in function                                 vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroSingleReaction::evaluate_reaction(std::vector<double>& reacval,
    std::vector<std::vector<double>>& reacderivspressure,
    std::vector<std::vector<double>>& reacderivssaturation, std::vector<double>& reacderivsporosity,
    std::vector<std::vector<double>>& reacderivsvolfrac,
    std::vector<std::vector<double>>& reacderivsvolfracpressure,
    std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
    const std::vector<double>& saturation, const double& porosity,
    const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
    const std::vector<double>& scalar)
{
  params_->evaluate_function(reacval, reacderivspressure, reacderivssaturation, reacderivsporosity,
      reacderivsvolfrac, reacderivsvolfracpressure, reacderivsscalar, pressure, saturation,
      porosity, volfracs, volfracpressures, scalar);

  return;
}

FOUR_C_NAMESPACE_CLOSE
