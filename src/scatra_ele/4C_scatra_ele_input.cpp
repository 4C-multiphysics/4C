// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_scatra_ele.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | read element input                                        fang 02/15 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::Transport::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read implementation type
  auto impltype = container.get<std::string>("TYPE");
  if (impltype == "Std")
    impltype_ = Inpar::ScaTra::impltype_std;
  else if (impltype == "AdvReac")
    impltype_ = Inpar::ScaTra::impltype_advreac;
  else if (impltype == "RefConcReac")
    impltype_ = Inpar::ScaTra::impltype_refconcreac;
  else if (impltype == "Chemo")
    impltype_ = Inpar::ScaTra::impltype_chemo;
  else if (impltype == "ChemoReac")
    impltype_ = Inpar::ScaTra::impltype_chemoreac;
  else if (impltype == "Aniso")
    impltype_ = Inpar::ScaTra::impltype_aniso;
  else if (impltype == "CardMono")
    impltype_ = Inpar::ScaTra::impltype_cardiac_monodomain;
  else if (impltype == "ElchDiffCond")
    impltype_ = Inpar::ScaTra::impltype_elch_diffcond;
  else if (impltype == "ElchDiffCondMultiScale")
    impltype_ = Inpar::ScaTra::impltype_elch_diffcond_multiscale;
  else if (impltype == "ElchDiffCondThermo")
    impltype_ = Inpar::ScaTra::impltype_elch_diffcond_thermo;
  else if (impltype == "ElchScl")
    impltype_ = Inpar::ScaTra::impltype_elch_scl;
  else if (impltype == "ElchElectrode")
    impltype_ = Inpar::ScaTra::impltype_elch_electrode;
  else if (impltype == "ElchElectrodeGrowth")
    impltype_ = Inpar::ScaTra::impltype_elch_electrode_growth;
  else if (impltype == "ElchElectrodeThermo")
    impltype_ = Inpar::ScaTra::impltype_elch_electrode_thermo;
  else if (impltype == "ElchNP")
    impltype_ = Inpar::ScaTra::impltype_elch_NP;
  else if (impltype == "Loma")
    impltype_ = Inpar::ScaTra::impltype_loma;
  else if (impltype == "Ls")
    impltype_ = Inpar::ScaTra::impltype_levelset;
  else if (impltype == "LsReinit")
    impltype_ = Inpar::ScaTra::impltype_lsreinit;
  else if (impltype == "Poro")
    impltype_ = Inpar::ScaTra::impltype_poro;
  else if (impltype == "PoroReac")
    impltype_ = Inpar::ScaTra::impltype_pororeac;
  else if (impltype == "PoroReacECM")
    impltype_ = Inpar::ScaTra::impltype_pororeacECM;
  else if (impltype == "Hdg")
    impltype_ = Inpar::ScaTra::impltype_std_hdg;
  else if (impltype == "HdgCardMono")
    impltype_ = Inpar::ScaTra::impltype_cardiac_monodomain_hdg;
  else
    FOUR_C_THROW("Transport element received invalid implementation type!");

  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // set discretization type
  set_dis_type(Core::FE::string_to_cell_type(distype));

  if (material()->material_type() == Core::Materials::m_myocard)
  {
    std::shared_ptr<Mat::Myocard> myocard = std::dynamic_pointer_cast<Mat::Myocard>(material());
    myocard->setup(container);
  }

  return true;
}

FOUR_C_NAMESPACE_CLOSE
