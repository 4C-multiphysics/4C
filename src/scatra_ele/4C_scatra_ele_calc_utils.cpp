/*----------------------------------------------------------------------*/
/*! \file

\brief Utility methods for scatra

\level 2

*/
/*----------------------------------------------------------------------*/


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

    std::vector<Core::Conditions::Condition*> myinflowcond;

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
  std::string impl_type_to_string(const Inpar::ScaTra::ImplType impltype)
  {
    // determine implementation type
    std::string impltypestring;
    switch (impltype)
    {
      case Inpar::ScaTra::impltype_std:
      {
        impltypestring = "Standard scalar transport";
        break;
      }
      case Inpar::ScaTra::impltype_thermo_elch_electrode:
      {
        impltypestring = "Heat transport within electrodes";
        break;
      }
      case Inpar::ScaTra::impltype_thermo_elch_diffcond:
      {
        impltypestring = "Heat transport within concentrated electrolytes";
        break;
      }
      case Inpar::ScaTra::impltype_advreac:
      {
        impltypestring = "Advanced reactions";
        break;
      }
      case Inpar::ScaTra::impltype_refconcreac:
      {
        impltypestring = "Reference concentrations AND reactions";
        break;
      }
      case Inpar::ScaTra::impltype_chemo:
      {
        impltypestring = "Chemotaxis";
        break;
      }
      case Inpar::ScaTra::impltype_chemoreac:
      {
        impltypestring = "Advanced reactions AND chemotaxis";
        break;
      }
      case Inpar::ScaTra::impltype_aniso:
      {
        impltypestring = "Anisotropic scalar transport";
        break;
      }
      case Inpar::ScaTra::impltype_cardiac_monodomain:
      {
        impltypestring = "Cardiac monodomain";
        break;
      }
      case Inpar::ScaTra::impltype_elch_diffcond:
      {
        impltypestring = "Electrochemistry for diffusion-conduction formulation";
        break;
      }
      case Inpar::ScaTra::impltype_elch_diffcond_multiscale:
      {
        impltypestring =
            "Electrochemistry for diffusion-conduction formulation within a multi-scale framework";
        break;
      }
      case Inpar::ScaTra::impltype_elch_diffcond_thermo:
      {
        impltypestring =
            "Electrochemistry for diffusion-conduction formulation with thermal effects";
        break;
      }
      case Inpar::ScaTra::impltype_elch_electrode:
      {
        impltypestring = "Electrochemistry for electrodes";
        break;
      }
      case Inpar::ScaTra::impltype_elch_electrode_growth:
      {
        impltypestring = "Electrochemistry for electrodes exhibiting lithium plating and stripping";
        break;
      }
      case Inpar::ScaTra::impltype_elch_electrode_thermo:
      {
        impltypestring = "Electrochemistry for electrodes with thermal effects";
        break;
      }
      case Inpar::ScaTra::impltype_elch_NP:
      {
        impltypestring = "Electrochemistry for Nernst-Planck formulation";
        break;
      }
      case Inpar::ScaTra::impltype_loma:
      {
        impltypestring = "Low Mach number flow";
        break;
      }
      case Inpar::ScaTra::impltype_levelset:
      {
        impltypestring = "Levelset without reinitialization";
        break;
      }
      case Inpar::ScaTra::impltype_lsreinit:
      {
        impltypestring = "Levelset with reinitialization";
        break;
      }
      case Inpar::ScaTra::impltype_poro:
      {
        impltypestring = "Scalar transport in porous media";
        break;
      }
      case Inpar::ScaTra::impltype_pororeac:
      {
        impltypestring = "Reactive scalar transport in porous media";
        break;
      }
      case Inpar::ScaTra::impltype_one_d_artery:
      {
        impltypestring = "Scalar Transport in 1D artery";
        break;
      }
      case Inpar::ScaTra::impltype_no_physics:
      {
        impltypestring = "Dummy with no physics";
        break;
      }
      case Inpar::ScaTra::impltype_undefined:
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
