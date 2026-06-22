// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_utils.hpp"

#include "4C_fem_condition_utils.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  bool is_binary_electrolyte(const std::vector<double>& valence)
  {
    int numions(0);
    for (size_t k = 0; k < valence.size(); k++)
    {
      if (abs(valence[k]) > 1e-10) numions++;
    }
    return (numions == 2);
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  std::vector<int> get_indices_binary_electrolyte(const std::vector<double>& valence)
  {
    // indices of the two charged species to be determined
    std::vector<int> indices;
    for (size_t k = 0; k < valence.size(); k++)
    {
      // is there some charge?
      if (abs(valence[k]) > 1e-10) indices.push_back(k);
    }
    if (indices.size() != 2) FOUR_C_THROW("Found no binary electrolyte!");

    return indices;
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  double cal_res_diff_coeff(const std::vector<double>& valence, const std::vector<double>& diffus,
      const std::vector<int>& indices)
  {
    if (indices.size() != 2) FOUR_C_THROW("Non-matching number of indices!");
    const int first = indices[0];
    const int second = indices[1];
    if ((valence[first] * valence[second]) > 1e-10)
      FOUR_C_THROW("Binary electrolyte has no opposite charges.");
    const double n = ((diffus[first] * valence[first]) - (diffus[second] * valence[second]));
    if (abs(n) < 1e-12)
      FOUR_C_THROW("denominator in resulting diffusion coefficient is nearly zero");

    return diffus[first] * diffus[second] * (valence[first] - valence[second]) / n;
  }


  /*-------------------------------------------------------------------------------*
   |find elements of inflow section                                rasthofer 01/12 |
   |for turbulent low Mach number flows with turbulent inflow condition            |
   *-------------------------------------------------------------------------------*/
  bool inflow_element(const Core::Elements::Element* ele)
  {
    bool inflow_ele = false;

    std::vector<const Core::Conditions::Condition*> myinflowcond;

    // check whether all nodes have a unique inflow condition
    Core::Conditions::find_element_conditions(ele, "TurbulentInflowSection", myinflowcond);
    if (myinflowcond.size() > 1) FOUR_C_THROW("More than one inflow condition on one node!");

    if (myinflowcond.size() == 1) inflow_ele = true;

    return inflow_ele;
  }


  /*---------------------------------------------------------------------------------------------------------------------*
   | convert implementation type of scalar transport elements into corresponding string for output
   purposes   fang 02/15 |
   *---------------------------------------------------------------------------------------------------------------------*/
  std::string impl_type_to_string(const ScaTra::ImplType impltype)
  {
    // determine implementation type
    std::string impltypestring;
    switch (impltype)
    {
      case ScaTra::impltype_std:
      {
        impltypestring = "Standard scalar transport";
        break;
      }
      case ScaTra::impltype_thermo_elch_electrode:
      {
        impltypestring = "Heat transport within electrodes";
        break;
      }
      case ScaTra::impltype_thermo_elch_diffcond:
      {
        impltypestring = "Heat transport within concentrated electrolytes";
        break;
      }
      case ScaTra::impltype_advreac:
      {
        impltypestring = "Advanced reactions";
        break;
      }
      case ScaTra::impltype_chemo:
      {
        impltypestring = "Chemotaxis";
        break;
      }
      case ScaTra::impltype_chemoreac:
      {
        impltypestring = "Advanced reactions AND chemotaxis";
        break;
      }
      case ScaTra::impltype_aniso:
      {
        impltypestring = "Anisotropic scalar transport";
        break;
      }
      case ScaTra::impltype_cardiac_monodomain:
      {
        impltypestring = "Cardiac monodomain";
        break;
      }
      case ScaTra::impltype_gr:
      {
        impltypestring = "Growth and remodeling";
        break;
      }
      case ScaTra::impltype_nl_stimulus:
      {
        impltypestring = "Non-local stimulus";
        break;
      }
      case ScaTra::impltype_elch_diffcond:
      {
        impltypestring = "Electrochemistry for diffusion-conduction formulation";
        break;
      }
      case ScaTra::impltype_elch_diffcond_multiscale:
      {
        impltypestring =
            "Electrochemistry for diffusion-conduction formulation within a multi-scale framework";
        break;
      }
      case ScaTra::impltype_elch_diffcond_thermo:
      {
        impltypestring =
            "Electrochemistry for diffusion-conduction formulation with thermal effects";
        break;
      }
      case ScaTra::impltype_elch_electrode:
      {
        impltypestring = "Electrochemistry for electrodes";
        break;
      }
      case ScaTra::impltype_elch_electrode_growth:
      {
        impltypestring = "Electrochemistry for electrodes exhibiting lithium plating and stripping";
        break;
      }
      case ScaTra::impltype_elch_electrode_thermo:
      {
        impltypestring = "Electrochemistry for electrodes with thermal effects";
        break;
      }
      case ScaTra::impltype_elch_NP:
      {
        impltypestring = "Electrochemistry for Nernst-Planck formulation";
        break;
      }
      case ScaTra::impltype_loma:
      {
        impltypestring = "Low Mach number flow";
        break;
      }
      case ScaTra::impltype_levelset:
      {
        impltypestring = "Levelset without reinitialization";
        break;
      }
      case ScaTra::impltype_lsreinit:
      {
        impltypestring = "Levelset with reinitialization";
        break;
      }
      case ScaTra::impltype_poro:
      {
        impltypestring = "Scalar transport in porous media";
        break;
      }
      case ScaTra::impltype_pororeac:
      {
        impltypestring = "Reactive scalar transport in porous media";
        break;
      }
      case ScaTra::impltype_one_d_artery:
      {
        impltypestring = "Scalar Transport in 1D artery";
        break;
      }
      case ScaTra::impltype_no_physics:
      {
        impltypestring = "Dummy with no physics";
        break;
      }
      case ScaTra::impltype_undefined:
      {
        impltypestring = "Undefined";
        break;
      }
      default:
      {
        FOUR_C_THROW("Invalid implementation type!");
        break;
      }
    }

    return impltypestring;
  }
}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE
