// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_beam_elasthyper_parameter.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_beam_elasthyper.hpp"
#include "4C_material_parameter_base.hpp"

#include <Sacado.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double Mat::PAR::determine_shear_modulus(const Core::Mat::PAR::Parameter::Data& matdata)
{
  double shearmodulus = 0.0;
  double poissonratio = 0.0;

  // We want the flexibility to either specify the shear modulus or the Poisson's ratio.
  // Therefore, both parameters are defined as optional in the definition of the input file line

  shearmodulus = matdata.parameters.get<double>("SHEARMOD");
  poissonratio = matdata.parameters.get<double>("POISSONRATIO");

  if (shearmodulus != -1.0 and poissonratio == -1.0)
  {
    // all good, only a value for shear modulus was given directly
  }
  else if (shearmodulus == -1.0 and poissonratio != -1.0)
  {
    // compute shear modulus from Young's modulus and given Poisson's ratio
    shearmodulus = matdata.parameters.get<double>("YOUNG") / (2.0 * (1.0 + poissonratio));
  }
  else if (shearmodulus != -1.0 and poissonratio != -1.0)
  {
    FOUR_C_THROW(
        "You specified both of the redundant material parameters SHEARMOD and POISSONRATIO! "
        "Specify exactly one of them in the material definition of your input file!");
  }
  else
  {
    FOUR_C_THROW(
        "You specified none of the material parameters SHEARMOD and POISSONRATIO! "
        "Specify exactly one of them in the material definition of your input file!");
  }

  return shearmodulus;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double Mat::PAR::determine_default_interaction_radius(
    const Core::Mat::PAR::Parameter::Data& matdata)
{
  double radius = matdata.parameters.get<double>("INTERACTIONRADIUS");

  double Iyy = matdata.parameters.get<double>("MOMIN2");
  double Izz = matdata.parameters.get<double>("MOMIN3");

  // determine default value for interaction radius if no value was given:
  // assume circular cross-section and compute from the area moment of inertia
  if (radius == -1.0 and Iyy == Izz) radius = std::pow(4.0 * Iyy / std::numbers::pi, 0.25);

  return radius;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double Mat::PAR::determine_default_interaction_radius_isotropic(
    const Core::Mat::PAR::Parameter::Data& matdata)
{
  double radius = matdata.parameters.get<double>("INTERACTIONRADIUS");

  double Iyy = matdata.parameters.get<double>("MOMIN");

  // determine default value for interaction radius if no value was given:
  // assume circular cross-section and compute from the area moment of inertia
  if (radius == -1.0) radius = std::pow(4.0 * Iyy / std::numbers::pi, 0.25);

  return radius;
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamElastHyperMaterialParameterGeneric::BeamElastHyperMaterialParameterGeneric(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), use_fad_(matdata.parameters.get<bool>("FAD"))
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material>
Mat::PAR::BeamElastHyperMaterialParameterGeneric::create_material()
{
  /* all the different parameter sets (Reissner/Kirchhoff/..., 'classic'/'by modes') are used to
   * parameterize the same constitutive relations based on a hyperelastic stored energy function
   * formulated for cross-section resultants which are implemented in BeamElastHyperMaterial */
  std::shared_ptr<Core::Mat::Material> matobject;

  if (uses_fad())
  {
    matobject = std::make_shared<Mat::BeamElastHyperMaterial<Sacado::Fad::DFad<double>>>(this);
  }
  else
    matobject = std::make_shared<Mat::BeamElastHyperMaterial<double>>(this);
  return matobject;
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamReissnerElastHyperMaterialParams::BeamReissnerElastHyperMaterialParams(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      youngs_modulus_(matdata.parameters.get<double>("YOUNG")),
      shear_modulus_(determine_shear_modulus(matdata)),
      density_(matdata.parameters.get<double>("DENS")),
      cross_section_area_(matdata.parameters.get<double>("CROSSAREA")),
      shear_correction_factor_(matdata.parameters.get<double>("SHEARCORR")),
      area_moment_inertia_polar_(matdata.parameters.get<double>("MOMINPOL")),
      area_moment_inertia_2_(matdata.parameters.get<double>("MOMIN2")),
      area_moment_inertia_3_(matdata.parameters.get<double>("MOMIN3")),
      radius_interaction_(determine_default_interaction_radius(matdata))
{
  if (youngs_modulus_ <= 0.0) FOUR_C_THROW("Young's modulus must be positive value");

  if (shear_modulus_ <= 0.0) FOUR_C_THROW("shear modulus must be positive value");

  if (density_ < 0.0) FOUR_C_THROW("density must not be negative value");

  if (cross_section_area_ <= 0.0) FOUR_C_THROW("cross-section area must be positive value");

  if (shear_correction_factor_ <= 0.0)
    FOUR_C_THROW("shear correction factor must be positive value");

  if (area_moment_inertia_polar_ <= 0.0)
    FOUR_C_THROW("polar/axial area moment of inertia must be positive value");

  if (area_moment_inertia_2_ <= 0.0) FOUR_C_THROW("area moment of inertia must be positive value");

  if (area_moment_inertia_3_ <= 0.0) FOUR_C_THROW("area moment of inertia must be positive value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered and you "
        "don't want to use the default radius computed from the area moment of inertia), the "
        "given interaction radius must be a positive value");
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamReissnerElastHyperMaterialParamsByMode::BeamReissnerElastHyperMaterialParamsByMode(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      axial_rigidity_(matdata.parameters.get<double>("EA")),
      shear_rigidity_2_(matdata.parameters.get<double>("GA2")),
      shear_rigidity_3_(matdata.parameters.get<double>("GA3")),
      torsional_rigidity_(matdata.parameters.get<double>("GI_T")),
      bending_rigidity_2_(matdata.parameters.get<double>("EI2")),
      bending_rigidity_3_(matdata.parameters.get<double>("EI3")),
      translational_mass_inertia_(matdata.parameters.get<double>("RhoA")),
      mass_moment_inertia_polar_(matdata.parameters.get<double>("MASSMOMINPOL")),
      mass_moment_inertia_2_(matdata.parameters.get<double>("MASSMOMIN2")),
      mass_moment_inertia_3_(matdata.parameters.get<double>("MASSMOMIN3")),
      radius_interaction_(matdata.parameters.get<double>("INTERACTIONRADIUS"))
{
  if (axial_rigidity_ <= 0.0) FOUR_C_THROW("axial rigidity must be positive value");

  if (shear_rigidity_2_ <= 0.0 or shear_rigidity_3_ <= 0.0)
    FOUR_C_THROW("shear rigidity must be positive value");

  if (torsional_rigidity_ <= 0.0) FOUR_C_THROW("torsional rigidity must be positive value");

  if (bending_rigidity_2_ <= 0.0 or bending_rigidity_3_ <= 0.0)
    FOUR_C_THROW("bending rigidity must be positive value");

  if (translational_mass_inertia_ < 0.0)
    FOUR_C_THROW("translational mass inertia must not be negative value");

  if (mass_moment_inertia_polar_ < 0.0)
    FOUR_C_THROW("polar mass moment of inertia must not be negative value");

  if (mass_moment_inertia_2_ < 0.0 or mass_moment_inertia_3_ < 0.0)
    FOUR_C_THROW("mass moment of inertia must not be negative value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered), the "
        "given interaction radius must be a positive value");
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamKirchhoffElastHyperMaterialParams::BeamKirchhoffElastHyperMaterialParams(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      youngs_modulus_(matdata.parameters.get<double>("YOUNG")),
      shear_modulus_(determine_shear_modulus(matdata)),
      density_(matdata.parameters.get<double>("DENS")),
      cross_section_area_(matdata.parameters.get<double>("CROSSAREA")),
      area_moment_inertia_polar_(matdata.parameters.get<double>("MOMINPOL")),
      area_moment_inertia_2_(matdata.parameters.get<double>("MOMIN2")),
      area_moment_inertia_3_(matdata.parameters.get<double>("MOMIN3")),
      radius_interaction_(determine_default_interaction_radius(matdata))
{
  if (youngs_modulus_ <= 0.0) FOUR_C_THROW("Young's modulus must be positive value");

  if (shear_modulus_ <= 0.0) FOUR_C_THROW("shear modulus must be positive value");

  if (density_ < 0.0) FOUR_C_THROW("density must not be negative value");

  if (cross_section_area_ <= 0.0) FOUR_C_THROW("cross-section area must be positive value");

  if (area_moment_inertia_polar_ <= 0.0)
    FOUR_C_THROW("polar/axial area moment of inertia must be positive value");

  if (area_moment_inertia_2_ <= 0.0) FOUR_C_THROW("area moment of inertia must be positive value");

  if (area_moment_inertia_3_ <= 0.0) FOUR_C_THROW("area moment of inertia must be positive value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered and you "
        "don't want to use the default radius computed from the area moment of inertia), the "
        "given interaction radius must be a positive value");
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamKirchhoffElastHyperMaterialParamsByMode::BeamKirchhoffElastHyperMaterialParamsByMode(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      axial_rigidity_(matdata.parameters.get<double>("EA")),
      torsional_rigidity_(matdata.parameters.get<double>("GI_T")),
      bending_rigidity_2_(matdata.parameters.get<double>("EI2")),
      bending_rigidity_3_(matdata.parameters.get<double>("EI3")),
      translational_mass_inertia_(matdata.parameters.get<double>("RhoA")),
      mass_moment_inertia_polar_(matdata.parameters.get<double>("MASSMOMINPOL")),
      mass_moment_inertia_2_(matdata.parameters.get<double>("MASSMOMIN2")),
      mass_moment_inertia_3_(matdata.parameters.get<double>("MASSMOMIN3")),
      radius_interaction_(matdata.parameters.get<double>("INTERACTIONRADIUS"))
{
  if (axial_rigidity_ <= 0.0) FOUR_C_THROW("axial rigidity must be positive value");

  if (torsional_rigidity_ <= 0.0) FOUR_C_THROW("torsional rigidity must be positive value");

  if (bending_rigidity_2_ <= 0.0 or bending_rigidity_3_ <= 0.0)
    FOUR_C_THROW("bending rigidity must be positive value");

  if (translational_mass_inertia_ < 0.0)
    FOUR_C_THROW("translational mass inertia must not be negative value");

  if (mass_moment_inertia_polar_ < 0.0)
    FOUR_C_THROW("polar mass moment of inertia must not be negative value");

  if (mass_moment_inertia_2_ < 0.0 or mass_moment_inertia_3_ < 0.0)
    FOUR_C_THROW("mass moment of inertia must not be negative value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered), the "
        "given interaction radius must be a positive value");
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamKirchhoffTorsionFreeElastHyperMaterialParams::
    BeamKirchhoffTorsionFreeElastHyperMaterialParams(const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      youngs_modulus_(matdata.parameters.get<double>("YOUNG")),
      density_(matdata.parameters.get<double>("DENS")),
      cross_section_area_(matdata.parameters.get<double>("CROSSAREA")),
      area_moment_inertia_(matdata.parameters.get<double>("MOMIN")),
      radius_interaction_(determine_default_interaction_radius_isotropic(matdata))
{
  if (youngs_modulus_ <= 0.0) FOUR_C_THROW("Young's modulus must be positive value");

  if (density_ < 0.0) FOUR_C_THROW("density must not be negative value");

  if (cross_section_area_ <= 0.0) FOUR_C_THROW("cross-section area must be positive value");

  if (area_moment_inertia_ <= 0.0) FOUR_C_THROW("area moment of inertia must be positive value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered and you "
        "don't want to use the default radius computed from the area moment of inertia), the "
        "given interaction radius must be a positive value");
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamKirchhoffTorsionFreeElastHyperMaterialParamsByMode::
    BeamKirchhoffTorsionFreeElastHyperMaterialParamsByMode(
        const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamElastHyperMaterialParameterGeneric(matdata),
      axial_rigidity_(matdata.parameters.get<double>("EA")),
      bending_rigidity_(matdata.parameters.get<double>("EI")),
      translational_mass_inertia_(matdata.parameters.get<double>("RhoA")),
      radius_interaction_(matdata.parameters.get<double>("INTERACTIONRADIUS"))
{
  if (axial_rigidity_ <= 0.0) FOUR_C_THROW("axial rigidity must be positive value");

  if (bending_rigidity_ <= 0.0) FOUR_C_THROW("bending rigidity must be positive value");

  if (translational_mass_inertia_ < 0.0)
    FOUR_C_THROW("translational mass inertia must not be negative value");


  /* the radius of an assumed circular cross-section is only used for the evaluation
   * of all kinds of interactions. it can hence be ignored if no interaction are considered. */
  if (radius_interaction_ != -1.0 and radius_interaction_ <= 0.0)
    FOUR_C_THROW(
        "if specified (only required if any kind of beam interactions are considered), the "
        "given interaction radius must be a positive value");
}

FOUR_C_NAMESPACE_CLOSE
