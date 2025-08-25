// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"

#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
 * get the single phase material from the element multiphase reaction material   vuong 08/16 |
 *-------------------------------------------------------------------------------------------*/
Mat::FluidPoroSingleReaction&
PoroPressureBased::ElementUtils::get_single_reaction_mat_from_multi_reactions_material(
    const Mat::FluidPoroMultiPhaseReactions& multiphasereacmat, int phasenum)
{
  // get the single phase material by its ID
  const int matid = multiphasereacmat.reac_id(phasenum);
  std::shared_ptr<Core::Mat::Material> singlemat = multiphasereacmat.material_by_id(matid);

  // safety check and cast
  if (singlemat->material_type() != Core::Materials::m_fluidporo_singlereaction)
    FOUR_C_THROW("only poro singleraction material valid");

  return static_cast<Mat::FluidPoroSingleReaction&>(*singlemat);
}

/*----------------------------------------------------------------------------------*
 * get the single phase material from the element multiphase material    vuong 08/16 |
 *-----------------------------------------------------------------------------------*/
const Mat::FluidPoroSinglePhase&
PoroPressureBased::ElementUtils::get_single_phase_mat_from_multi_material(
    const Mat::FluidPoroMultiPhase& multiphasemat, int phasenum)
{
  // get the single phase material by its ID
  const int matid = multiphasemat.mat_id(phasenum);
  std::shared_ptr<Core::Mat::Material> singlemat = multiphasemat.material_by_id(matid);

  // safety check and cast
  if (singlemat->material_type() != Core::Materials::m_fluidporo_singlephase)
    FOUR_C_THROW("check at position {}/{} failed, only poro singlephase material valid",
        phasenum + 1, multiphasemat.num_mat());

  return static_cast<const Mat::FluidPoroSinglePhase&>(*singlemat);
}

/*------------------------------------------------------------------------*
 *  get the single phase material from the element material   vuong 08/16 |
 *-------------------------------------------------------------------------*/
const Mat::FluidPoroSinglePhase&
PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(
    const Core::Mat::Material& material, int phasenum)
{
  // safety check
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  return get_single_phase_mat_from_multi_material(multiphasemat, phasenum);
}

/*---------------------------------------------------------------------------------------*
 * get the single volfrac material from the element multiphase material kremheller 08/17 |
 *----------------------------------------------------------------------------------------*/
const Mat::FluidPoroSingleVolFrac&
PoroPressureBased::ElementUtils::get_single_vol_frac_mat_from_multi_material(
    const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum)
{
  // get the single phase material by its ID
  const int matid = multiphasemat.mat_id(volfracnum);
  std::shared_ptr<Core::Mat::Material> singlemat = multiphasemat.material_by_id(matid);

  // safety check and cast
  if (singlemat->material_type() != Core::Materials::m_fluidporo_singlevolfrac)
    FOUR_C_THROW("check at position {}/{} failed, only poro single vol fraction material valid",
        volfracnum + 1, multiphasemat.num_mat());

  return static_cast<const Mat::FluidPoroSingleVolFrac&>(*singlemat);
}

/*---------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
const Mat::FluidPoroVolFracPressureBloodLung&
PoroPressureBased::ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
    const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum)
{
  // get the single phase material by its ID
  const int matid = multiphasemat.mat_id(volfracnum);
  std::shared_ptr<Core::Mat::Material> singlemat = multiphasemat.material_by_id(matid);

  // safety check and cast
  if (singlemat->material_type() != Core::Materials::m_fluidporo_volfrac_pressure_blood_lung)
    FOUR_C_THROW("check at position {}/{} failed, only poro single vol fraction material valid",
        volfracnum + 1, multiphasemat.num_mat());

  return static_cast<const Mat::FluidPoroVolFracPressureBloodLung&>(*singlemat);
}


/*-------------------------------------------------------------------------------*
 *  get the single volfrac material from the element material   kremheller 08/17 |
 *--------------------------------------------------------------------------------*/
const Mat::FluidPoroSingleVolFrac&
PoroPressureBased::ElementUtils::get_single_vol_frac_mat_from_material(
    const Core::Mat::Material& material, int volfracnum)
{
  // safety check
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  return get_single_vol_frac_mat_from_multi_material(multiphasemat, volfracnum);
}

/*-------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
const Mat::FluidPoroVolFracPressureBloodLung&
PoroPressureBased::ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_material(
    const Core::Mat::Material& material, int volfracnum)
{
  // safety check
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  return get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(multiphasemat, volfracnum);
}

/*-------------------------------------------------------------------------------------------------*
 * get the volume fraction pressure material from the element multiphase material kremheller 02/18 |
 *--------------------------------------------------------------------------------------------------*/
const Mat::FluidPoroVolFracPressure&
PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_multi_material(
    const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum)
{
  // get the single phase material by its ID
  const int matid = multiphasemat.mat_id(volfracnum);
  std::shared_ptr<Core::Mat::Material> singlemat = multiphasemat.material_by_id(matid);

  // safety check and cast
  if (singlemat->material_type() != Core::Materials::m_fluidporo_volfracpressure)
    FOUR_C_THROW("check at position {}/{} failed, only poro single vol fraction material valid",
        volfracnum + 1, multiphasemat.num_mat());

  return static_cast<const Mat::FluidPoroVolFracPressure&>(*singlemat);
}

/*-----------------------------------------------------------------------------------------*
 *  get the volume fraction pressure material from the element material   kremheller 02/18 |
 *------------------------------------------------------------------------------------------*/
const Mat::FluidPoroVolFracPressure&
PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_material(
    const Core::Mat::Material& material, int volfracnum)
{
  // safety check
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  return get_vol_frac_pressure_mat_from_multi_material(multiphasemat, volfracnum);
}

FOUR_C_NAMESPACE_CLOSE
