// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_resulttest.hpp"

#include "4C_cardiovascular0d.hpp"
#include "4C_cardiovascular0d_manager.hpp"
#include "4C_fem_discretization.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Cardiovascular0DResultTest::Cardiovascular0DResultTest(
    Utils::Cardiovascular0DManager& cardvasc0dman, std::shared_ptr<Core::FE::Discretization> discr)
    : Core::Utils::ResultTest("CARDIOVASCULAR0D"),
      actdisc_(discr),
      cardvasc0d_dof_(cardvasc0dman
              .get0_d_dof_m()),  // cardiovascular 0D dofs at generalized mid-point t_{n+\theta}
      havecardio_4elementwindkessel_(
          cardvasc0dman.get_cardvasc0_d4_element_windkessel()->have_cardiovascular0_d()),
      havecardio_arterialproxdist_(
          cardvasc0dman.get_cardvasc0_d_arterial_prox_dist()->have_cardiovascular0_d()),
      havecardio_syspulcirculation_(
          cardvasc0dman.get_cardvasc0_d_sys_pul_circulation()->have_cardiovascular0_d()),
      havecardiorespir_syspulperiphcirculation_(
          cardvasc0dman.get_cardvasc_respir0_d_sys_pul_periph_circulation()
              ->have_cardiovascular0_d())
{
  // empty
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Cardiovascular0DResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  std::string quantity = container.get<std::string>("QUANTITY");
  bool unknownquantity = true;  // make sure the result value std::string can be handled
  double result = 0.0;          // will hold the actual result of run


  const Epetra_BlockMap& cardvasc0dmap = cardvasc0d_dof_->Map();
  const int offset = cardvasc0dmap.MinAllGID();

  bool havegid = false;

  if (havecardio_4elementwindkessel_)
    FOUR_C_THROW("Testing not implemented for 4ElementWindkessel model!");

  if (havecardio_arterialproxdist_)
    FOUR_C_THROW("Testing not implemented for ArterialProxDist model!");

  if (havecardio_syspulcirculation_)
  {
    // test for left atrial pressure
    if (quantity == "p_at_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 0);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 0)];
    }
    // test for left ventricular in-flux
    if (quantity == "q_vin_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 1);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 1)];
    }
    // test for left ventricular out-flux
    if (quantity == "q_vout_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 2);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 2)];
    }
    // test for left ventricular pressure
    if (quantity == "p_v_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 3);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 3)];
    }
    // test for systemic arterial pressure
    if (quantity == "p_ar_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 4);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 4)];
    }
    // test for systemic arterial flux
    if (quantity == "q_ar_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 5);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 5)];
    }
    // test for systemic venous pressure
    if (quantity == "p_ven_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 6);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 6)];
    }
    // test for systemic venous flux
    if (quantity == "q_ven_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 7);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 7)];
    }
    // test for right atrial pressure
    if (quantity == "p_at_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 8);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 8)];
    }
    // test for right ventricular in-flux
    if (quantity == "q_vin_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 9);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 9)];
    }
    // test for right ventricular out-flux
    if (quantity == "q_vout_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 10);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 10)];
    }
    // test for right ventricular pressure
    if (quantity == "p_v_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 11);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 11)];
    }
    // test for pulmonary arterial pressure
    if (quantity == "p_ar_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 12);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 12)];
    }
    // test for pulmonary arterial flux
    if (quantity == "q_ar_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 13);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 13)];
    }
    // test for pulmonary venous pressure
    if (quantity == "p_ven_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 14);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 14)];
    }
    // test for pulmonary venous flux
    if (quantity == "q_ven_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 15);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 15)];
    }

    // catch quantity strings, which are not handled by cardiovascular 0D result test
    if (unknownquantity)
      FOUR_C_THROW("Quantity '%s' not supported in cardiovascular 0D testing", quantity.c_str());

    if (havegid)
    {
      // compare values
      const int err = compare_values(result, "SPECIAL", container);
      nerr += err;
      test_count++;
    }
  }



  if (havecardiorespir_syspulperiphcirculation_)
  {
    // test for left atrial pressure
    if (quantity == "p_at_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 0);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 0)];
    }
    // test for left ventricular in-flux
    if (quantity == "q_vin_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 1);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 1)];
    }
    // test for left ventricular out-flux
    if (quantity == "q_vout_l")
    {
      havegid = cardvasc0dmap.MyGID(offset + 2);
      unknownquantity = false;
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 2)];
    }
    // test for left ventricular pressure
    if (quantity == "p_v_l")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 3);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 3)];
    }
    // test for systemic arterial pressure
    if (quantity == "p_ar_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 4);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 4)];
    }
    // test for systemic arterial flux
    if (quantity == "q_ar_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 5);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 5)];
    }
    // test for systemic arterial peripheral pressure
    if (quantity == "p_arperi_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 6);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 6)];
    }
    // test for systemic arterial splanchnic flux
    if (quantity == "q_arspl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 7);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 7)];
    }
    // test for systemic arterial extra-splanchnic flux
    if (quantity == "q_arespl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 8);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 8)];
    }
    // test for systemic arterial muscular flux
    if (quantity == "q_armsc_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 9);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 9)];
    }
    // test for systemic arterial cerebral flux
    if (quantity == "q_arcer_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 10);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 10)];
    }
    // test for systemic arterial coronary flux
    if (quantity == "q_arcor_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 11);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 11)];
    }
    // test for systemic venous splanchnic pressure
    if (quantity == "p_venspl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 12);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 12)];
    }
    // test for systemic venous splanchnic flux
    if (quantity == "q_venspl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 13);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 13)];
    }
    // test for systemic venous extra-splanchnic pressure
    if (quantity == "p_venespl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 14);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 14)];
    }
    // test for systemic venous extra-splanchnic flux
    if (quantity == "q_venespl_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 15);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 15)];
    }
    // test for systemic venous muscular pressure
    if (quantity == "p_venmsc_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 16);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 16)];
    }
    // test for systemic venous muscular flux
    if (quantity == "q_venmsc_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 17);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 17)];
    }
    // test for systemic venous cerebral pressure
    if (quantity == "p_vencer_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 18);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 18)];
    }
    // test for systemic venous cerebral flux
    if (quantity == "q_vencer_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 19);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 19)];
    }
    // test for systemic venous coronary pressure
    if (quantity == "p_vencor_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 20);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 20)];
    }
    // test for systemic venous coronary flux
    if (quantity == "q_vencor_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 21);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 21)];
    }
    // test for systemic venous pressure
    if (quantity == "p_ven_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 22);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 22)];
    }
    // test for systemic venous flux
    if (quantity == "q_ven_sys")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 23);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 23)];
    }
    // test for right atrial pressure
    if (quantity == "p_at_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 24);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 24)];
    }
    // test for right ventricular in-flux
    if (quantity == "q_vin_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 25);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 25)];
    }
    // test for right ventricular out-flux
    if (quantity == "q_vout_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 26);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 26)];
    }
    // test for right ventricular pressure
    if (quantity == "p_v_r")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 27);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 27)];
    }
    // test for pulmonary arterial pressure
    if (quantity == "p_ar_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 28);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 28)];
    }
    // test for pulmonary arterial flux
    if (quantity == "q_ar_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 29);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 29)];
    }
    // test for pulmonary capillary pressure
    if (quantity == "p_cap_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 30);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 30)];
    }
    // test for pulmonary capillary flux
    if (quantity == "q_cap_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 31);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 31)];
    }
    // test for pulmonary venous pressure
    if (quantity == "p_ven_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 32);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 32)];
    }
    // test for pulmonary venous flux
    if (quantity == "q_ven_pul")
    {
      unknownquantity = false;
      havegid = cardvasc0dmap.MyGID(offset + 33);
      if (havegid) result = (*cardvasc0d_dof_)[cardvasc0dmap.LID(offset + 33)];
    }

    // catch quantity strings, which are not handled by cardiovascular 0D result test
    if (unknownquantity)
      FOUR_C_THROW("Quantity '%s' not supported in cardiovascular 0D testing", quantity.c_str());

    if (havegid)
    {
      // compare values
      const int err = compare_values(result, "SPECIAL", container);
      nerr += err;
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
