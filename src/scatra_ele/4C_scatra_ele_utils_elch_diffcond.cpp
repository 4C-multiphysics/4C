// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_utils_elch_diffcond.hpp"

#include "4C_mat_elchmat.hpp"
#include "4C_mat_elchphase.hpp"
#include "4C_mat_newman.hpp"
#include "4C_mat_newman_multiscale.hpp"
#include "4C_scatra_ele_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_calc_elch_diffcond_multiscale.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>*
Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleUtilsElchDiffCond<distype>>(
            new ScaTraEleUtilsElchDiffCond<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::ScaTraEleUtilsElchDiffCond(
    const int numdofpernode, const int numscal, const std::string& disname)
    : myelectrode::ScaTraEleUtilsElchElectrode(numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::mat_elch_mat(
    std::shared_ptr<const Core::Mat::Material> material, const std::vector<double>& concentrations,
    const double temperature, const ElCh::EquPot equpot, const double ffrt,
    std::shared_ptr<ScaTraEleDiffManagerElchDiffCond> diffmanager, ElCh::DiffCondMat& diffcondmat)
{
  // cast material to electrolyte material
  const std::shared_ptr<const Mat::ElchMat> elchmat =
      std::static_pointer_cast<const Mat::ElchMat>(material);

  // safety check
  if (elchmat->num_phase() != 1)
    FOUR_C_THROW("Can only have a single electrolyte phase at the moment!");

  // extract electrolyte phase
  const std::shared_ptr<const Core::Mat::Material> elchphase =
      elchmat->phase_by_id(elchmat->phase_id(0));

  if (elchphase->material_type() == Core::Materials::m_elchphase)
  {
    // evaluate electrolyte phase
    mat_elch_phase(elchphase, concentrations, temperature, equpot, ffrt, diffmanager, diffcondmat);
  }
  else
    FOUR_C_THROW("Invalid material type!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::mat_elch_phase(
    std::shared_ptr<const Core::Mat::Material> material, const std::vector<double>& concentrations,
    const double temperature, const ElCh::EquPot& equpot, const double& ffrt,
    std::shared_ptr<ScaTraEleDiffManagerElchDiffCond> diffmanager, ElCh::DiffCondMat& diffcondmat)
{
  // cast material to electrolyte phase
  const std::shared_ptr<const Mat::ElchPhase> matelchphase =
      std::static_pointer_cast<const Mat::ElchPhase>(material);

  // set porosity
  diffmanager->set_phase_poro(matelchphase->epsilon(), 0);

  // set tortuosity
  diffmanager->set_phase_tort(matelchphase->tortuosity(), 0);

  // loop over materials within electrolyte phase
  for (int imat = 0; imat < matelchphase->num_mat(); ++imat)
  {
    const std::shared_ptr<const Core::Mat::Material> elchPhaseMaterial =
        matelchphase->mat_by_id(matelchphase->mat_id(imat));

    switch (elchPhaseMaterial->material_type())
    {
      case Core::Materials::m_newman:
      case Core::Materials::m_newman_multiscale:
      {
        // safety check
        if (matelchphase->num_mat() != 1)
          FOUR_C_THROW("Newman material must be the only transported species!");

        // set ion type
        diffcondmat = ElCh::diffcondmat_newman;

        // evaluate standard Newman material
        mat_newman(elchPhaseMaterial, concentrations[0], temperature, diffmanager);

        break;
      }

      case Core::Materials::m_ion:
      {
        // set ion type
        diffcondmat = ElCh::diffcondmat_ion;

        myelch::mat_ion(elchPhaseMaterial, imat, equpot, diffmanager);

        // calculation of conductivity and transference number based on diffusion coefficient and
        // valence
        if (imat == matelchphase->num_mat() - 1)
        {
          diffmanager->calc_conductivity(matelchphase->num_mat(), ffrt, concentrations);
          diffmanager->calc_trans_num(matelchphase->num_mat(), concentrations);
        }

        break;
      }

      default:
      {
        FOUR_C_THROW("Invalid material type!");
        break;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::mat_newman(
    std::shared_ptr<const Core::Mat::Material> material, const double concentration,
    const double temperature, std::shared_ptr<ScaTraEleDiffManagerElchDiffCond> diffmanager)
{
  // cast material to Newman material
  const std::shared_ptr<const Mat::Newman> matnewman =
      std::static_pointer_cast<const Mat::Newman>(material);

  // valence of ionic species
  diffmanager->set_valence(matnewman->valence(), 0);

  // concentration depending diffusion coefficient
  diffmanager->set_isotropic_diff(
      matnewman->compute_diffusion_coefficient(concentration, temperature), 0);
  // derivation of concentration depending diffusion coefficient wrt concentration
  diffmanager->set_conc_deriv_iso_diff_coef(
      matnewman->compute_concentration_derivative_of_diffusion_coefficient(
          concentration, temperature),
      0, 0);

  // derivation of concentration depending diffusion coefficient wrt temperature
  diffmanager->set_temp_deriv_iso_diff_coef(
      matnewman->compute_temperature_derivative_of_diffusion_coefficient(
          concentration, temperature),
      0, 0);

  // concentration depending transference number
  diffmanager->set_trans_num(matnewman->compute_transference_number(concentration), 0);
  // derivation of concentration depending transference number wrt all ionic species
  diffmanager->set_deriv_trans_num(
      matnewman->compute_concentration_derivative_of_transference_number(concentration), 0, 0);

  // thermodynamic factor of electrolyte solution
  diffmanager->set_therm_fac(matnewman->compute_thermodynamic_factor(concentration));
  // derivative of conductivity with respect to concentrations
  diffmanager->set_deriv_therm_fac(
      matnewman->compute_concentration_derivative_of_thermodynamic_factor(concentration), 0);

  // conductivity and first derivative can maximally depend on one concentration
  // since time curve is used as input routine
  // conductivity of electrolyte solution
  diffmanager->set_cond(matnewman->compute_conductivity(concentration, temperature));

  // derivative of electronic conductivity w.r.t. concentration
  diffmanager->set_conc_deriv_cond(
      matnewman->compute_concentration_derivative_of_conductivity(concentration, temperature), 0);

  // derivative of electronic conductivity w.r.t. temperature
  diffmanager->set_temp_deriv_cond(
      matnewman->compute_temperature_derivative_of_conductivity(concentration, temperature), 0);
}


// template classes
// 1D elements
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::quad4>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::nurbs3>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::hex8>;
// template class
// Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::tet10>;
// template class
// Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::pyramid5>;
// template class
// Discret::Elements::ScaTraEleUtilsElchDiffCond<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
