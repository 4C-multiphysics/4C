// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_strategy_factory.hpp"

#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_contact_constitutivelaw_interface.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_lagrange_strategy.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_lagrange_strategy_wear.hpp"
#include "4C_contact_nitsche_strategy.hpp"
#include "4C_contact_nitsche_strategy_poro.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_contact_nitsche_strategy_ssi_elch.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_contact_penalty_strategy.hpp"
#include "4C_contact_rough_node.hpp"
#include "4C_contact_tsi_interface.hpp"
#include "4C_contact_utils.hpp"
#include "4C_contact_wear_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_input.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_utils.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::setup(const int dim)
{
  check_init();
  Mortar::STRATEGY::Factory::setup(dim);

  set_is_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::read_and_check_input(Teuchos::ParameterList& params) const
{
  check_init();
  // console output at the beginning
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Checking contact input parameters...........";
    fflush(stdout);
  }

  // read parameter lists from Global::Problem
  const Teuchos::ParameterList& mortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& contact = Global::Problem::instance()->contact_dynamic_params();
  const Teuchos::ParameterList& wearlist = Global::Problem::instance()->wear_params();
  const Teuchos::ParameterList& tsic = Global::Problem::instance()->tsi_contact_params();

  // read Problem Type and Problem Dimension from Global::Problem
  const Core::ProblemType problemtype = Global::Problem::instance()->get_problem_type();
  Core::FE::ShapeFunctionType distype = Global::Problem::instance()->spatial_approximation_type();
  const int dim = Global::Problem::instance()->n_dim();

  // in case just System type system_condensed_lagmult
  if (Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") ==
      CONTACT::SystemType::condensed_lagmult)
  {
    FOUR_C_THROW(
        "For Contact anyway just the lagrange multiplier can be condensed, "
        "choose SYSTEM = Condensed.");
  }

  // ---------------------------------------------------------------------
  // invalid parallel strategies
  // ---------------------------------------------------------------------
  const Teuchos::ParameterList& mortarParallelRedistParams =
      mortar.sublist("PARALLEL REDISTRIBUTION");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none &&
      mortarParallelRedistParams.get<int>("MIN_ELEPROC") < 0)
    FOUR_C_THROW(
        "Minimum number of elements per processor for parallel redistribution must be >= 0");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == Inpar::Mortar::ParallelRedist::redist_dynamic &&
      mortarParallelRedistParams.get<double>("MAX_BALANCE_EVAL_TIME") < 1.0)
  {
    FOUR_C_THROW(
        "Maximum allowed value of load balance for dynamic parallel redistribution must be "
        ">= 1.0");
  }

  if (problemtype == Core::ProblemType::tsi &&
      Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none &&
      Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
          CONTACT::SolvingStrategy::nitsche)
    FOUR_C_THROW("Parallel redistribution not yet implemented for TSI problems");

  // ---------------------------------------------------------------------
  // adhesive contact
  // ---------------------------------------------------------------------
  if (Teuchos::getIntegralValue<CONTACT::AdhesionType>(contact, "ADHESION") !=
          CONTACT::AdhesionType::none and
      Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
          Inpar::Wear::wear_none)
    FOUR_C_THROW("Adhesion combined with wear not yet tested!");

  if (Teuchos::getIntegralValue<CONTACT::AdhesionType>(contact, "ADHESION") !=
          CONTACT::AdhesionType::none and
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none)
    FOUR_C_THROW("Adhesion combined with friction not yet tested!");

  // ---------------------------------------------------------------------
  // generally invalid combinations (nts/mortar)
  // ---------------------------------------------------------------------
  if ((Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::penalty ||
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::nitsche) &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

  if ((Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::penalty ||
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::nitsche) &&
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    FOUR_C_THROW("Tangential penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    FOUR_C_THROW("Tangential penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<int>("UZAWAMAXSTEPS") < 2)
    FOUR_C_THROW("Maximum number of Uzawa / Augmentation steps must be at least 2");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<double>("UZAWACONSTRTOL") <= 0.0)
    FOUR_C_THROW("Constraint tolerance for Uzawa / Augmentation scheme must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("SEMI_SMOOTH_CT") == 0.0)
    FOUR_C_THROW("Parameter ct = 0, must be greater than 0 for frictional contact");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
          CONTACT::FrictionType::tresca &&
      dim == 3 &&
      Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
          CONTACT::SolvingStrategy::nitsche)
    FOUR_C_THROW(
        "3D frictional contact with Tresca's law only implemented for nitsche formulation");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      not contact.get<bool>("SEMI_SMOOTH_NEWTON") && dim == 3)
    FOUR_C_THROW("3D frictional contact only implemented with Semi-smooth Newton");

  if (mortar.get<bool>("CROSSPOINTS") && dim == 3)
    FOUR_C_THROW("Crosspoints / edge node modification not yet implemented for 3D");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
          CONTACT::FrictionType::tresca &&
      contact.get<bool>("FRLESS_FIRST"))
    // hopefully coming soon, when Coulomb and Tresca are combined
    FOUR_C_THROW("Frictionless first contact step with Tresca's law not yet implemented");

  // ---------------------------------------------------------------------
  // warnings
  // ---------------------------------------------------------------------
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (mortar.get<double>("SEARCH_PARAM") == 0.0)
      std::cout << ("Warning: Contact search called without inflation of bounding volumes\n")
                << std::endl;

    if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(wearlist, "WEAR_SIDE") !=
        Inpar::Wear::wear_slave)
      std::cout << ("\n \n Warning: Contact with both-sided wear is still experimental !")
                << std::endl;
  }

  // ---------------------------------------------------------------------
  //                       MORTAR-SPECIFIC CHECKS
  // ---------------------------------------------------------------------
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
      Inpar::Mortar::algorithm_mortar)
  {
    // ---------------------------------------------------------------------
    // invalid parameter combinations
    // ---------------------------------------------------------------------
    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Petrov-Galerkin approach for LM only with Lagrange multiplier strategy");

    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
            CONTACT::SolvingStrategy::lagmult &&
        (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
                Inpar::Mortar::shape_standard &&
            Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") !=
                Inpar::Mortar::lagmult_const) &&
        Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") ==
            CONTACT::SystemType::condensed)
      FOUR_C_THROW("Condensation of linear system only possible for dual Lagrange multipliers");

    if (Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
            mortar, "LM_DUAL_CONSISTENT") != Inpar::Mortar::ConsistentDualType::consistent_none &&
        Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
            Inpar::Mortar::shape_standard)
    {
      FOUR_C_THROW(
          "Consistent dual shape functions in boundary elements only for Lagrange "
          "multiplier strategy.");
    }

    if (Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
            mortar, "LM_DUAL_CONSISTENT") != Inpar::Mortar::ConsistentDualType::consistent_none &&
        Teuchos::getIntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE") ==
            Inpar::Mortar::inttype_elements &&
        (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_dual))
    {
      FOUR_C_THROW(
          "Consistent dual shape functions in boundary elements not for purely "
          "element-based integration.");
    }

    // ---------------------------------------------------------------------
    // not (yet) implemented combinations
    // ---------------------------------------------------------------------

    if (mortar.get<bool>("CROSSPOINTS") && Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                                               mortar, "LM_QUAD") == Inpar::Mortar::lagmult_lin)
      FOUR_C_THROW("Crosspoints and linear LM interpolation for quadratic FE not yet compatible");

    // check for self contact
    std::vector<const Core::Conditions::Condition*> contactConditions;
    discret().get_condition("Mortar", contactConditions);
    bool self = false;

    for (const auto& condition : contactConditions)
    {
      const auto& side = condition->parameters().get<std::string>("Side");
      if (side == "Selfcontact") self = true;
    }

    if (self && Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
                    "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
      FOUR_C_THROW("Self contact and parallel redistribution not yet compatible");

    if (contact.get<bool>("INITCONTACTBYGAP") && contact.get<double>("INITCONTACTGAPVALUE") == 0.0)
      FOUR_C_THROW(
          "For initialization of init contact with gap, the INITCONTACTGAPVALUE is needed.");

    if (Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
            mortar, "LM_DUAL_CONSISTENT") != Inpar::Mortar::ConsistentDualType::consistent_none &&
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") !=
            Inpar::Mortar::lagmult_undefined &&
        distype != Core::FE::ShapeFunctionType::nurbs)
    {
      FOUR_C_THROW(
          "Consistent dual shape functions in boundary elements only for linear shape "
          "functions or NURBS.");
    }

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none &&
        contact.get<bool>("FRLESS_FIRST"))
      FOUR_C_THROW("Frictionless first contact step with wear not yet implemented");

    if (problemtype != Core::ProblemType::ehl && contact.get<bool>("REGULARIZED_NORMAL_CONTACT"))
      FOUR_C_THROW("Regularized normal contact only implemented for EHL");


    // ---------------------------------------------------------------------
    // thermal-structure-interaction contact
    // ---------------------------------------------------------------------
    if (problemtype == Core::ProblemType::tsi &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_standard &&
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") !=
            Inpar::Mortar::lagmult_const)
      FOUR_C_THROW("Thermal contact only for dual shape functions");

    if (problemtype == Core::ProblemType::tsi &&
        Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") !=
            CONTACT::SystemType::condensed)
      FOUR_C_THROW("Thermal contact only for dual shape functions with condensed system");

    // no nodal scaling in for thermal-structure-interaction
    if (problemtype == Core::ProblemType::tsi &&
        tsic.get<double>("TEMP_DAMAGE") <= tsic.get<double>("TEMP_REF"))
      FOUR_C_THROW("damage temperature must be greater than reference temperature");

    // ---------------------------------------------------------------------
    // contact with wear
    // ---------------------------------------------------------------------
    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") ==
            Inpar::Wear::wear_none &&
        wearlist.get<double>("WEARCOEFF") != 0.0)
      FOUR_C_THROW("Wear coefficient only necessary in the context of wear.");

    if (problemtype == Core::ProblemType::structure and
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none and
        Teuchos::getIntegralValue<Inpar::Wear::WearTimInt>(wearlist, "WEARTIMINT") !=
            Inpar::Wear::wear_expl)
    {
      FOUR_C_THROW(
          "Wear calculation for pure structure problems only with explicit internal state "
          "variable approach reasonable!");
    }

    if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
            CONTACT::FrictionType::none &&
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none)
      FOUR_C_THROW("Wear models only applicable to frictional contact.");

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none &&
        wearlist.get<double>("WEARCOEFF") <= 0.0)
      FOUR_C_THROW("No valid wear coefficient provided, must be equal or greater 0.0");

    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult &&
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none)
      FOUR_C_THROW("Wear model only applicable in combination with Lagrange multiplier strategy.");

    if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
            CONTACT::FrictionType::tresca &&
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none)
      FOUR_C_THROW("Wear only for Coulomb friction!");

    // ---------------------------------------------------------------------
    // 3D quadratic mortar (choice of interpolation and testing fcts.)
    // ---------------------------------------------------------------------
    if (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") ==
            Inpar::Mortar::lagmult_pwlin &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_dual)
    {
      FOUR_C_THROW(
          "No piecewise linear approach (for LM) implemented for quadratic contact with "
          "DUAL shape fct.");
    }

    // ---------------------------------------------------------------------
    // poroelastic contact
    // ---------------------------------------------------------------------
    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                Inpar::Mortar::shape_dual &&
            Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                Inpar::Mortar::shape_petrovgalerkin))
      FOUR_C_THROW("POROCONTACT: Only dual and petrovgalerkin shape functions implemented yet!");

    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
            "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
      FOUR_C_THROW(
          "POROCONTACT: Parallel Redistribution not implemented yet!");  // Since we use Pointers to
                                                                         // Parent Elements, which
                                                                         // are not copied to other
                                                                         // procs!

    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult)
      FOUR_C_THROW("POROCONTACT: Use Lagrangean Strategy for poro contact!");

    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
            CONTACT::FrictionType::none)
      FOUR_C_THROW("POROCONTACT: Friction for poro contact not implemented!");

    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") !=
            CONTACT::SystemType::condensed)
      FOUR_C_THROW("POROCONTACT: System has to be condensed for poro contact!");

    if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
            problemtype == Core::ProblemType::fpsi_xfem) &&
        (dim != 3) && (dim != 2))
    {
      const Teuchos::ParameterList& porodyn =
          Global::Problem::instance()->poroelast_dynamic_params();
      if (porodyn.get<bool>("CONTACT_NO_PENETRATION"))
        FOUR_C_THROW("POROCONTACT: PoroContact with no penetration just tested for 3d (and 2d)!");
    }

    // ---------------------------------------------------------------------
    // element-based vs. segment-based mortar integration
    // ---------------------------------------------------------------------
    auto inttype = Teuchos::getIntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE");

    if (inttype == Inpar::Mortar::inttype_elements && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

    if (inttype == Inpar::Mortar::inttype_elements_BS && mortar.get<int>("NUMGP_PER_DIM") <= 0)
    {
      FOUR_C_THROW(
          "Invalid Gauss point number NUMGP_PER_DIM for element-based integration with "
          "boundary segmentation."
          "\nPlease note that the value you have to provide only applies to the element-based "
          "integration"
          "\ndomain, while pre-defined default values will be used in the segment-based boundary "
          "domain.");
    }

    if ((inttype == Inpar::Mortar::inttype_elements ||
            inttype == Inpar::Mortar::inttype_elements_BS) &&
        mortar.get<int>("NUMGP_PER_DIM") <= 1)
      FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");
  }  // END MORTAR CHECKS

  // ---------------------------------------------------------------------
  //                       NTS-SPECIFIC CHECKS
  // ---------------------------------------------------------------------
  else if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
           Inpar::Mortar::algorithm_nts)
  {
    if (problemtype == Core::ProblemType::poroelast or problemtype == Core::ProblemType::fpsi or
        problemtype == Core::ProblemType::tsi)
      FOUR_C_THROW("NTS only for problem type: structure");
  }  // END NTS CHECKS

  // ---------------------------------------------------------------------
  //                       GPTS-SPECIFIC CHECKS
  // ---------------------------------------------------------------------
  else if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
           Inpar::Mortar::algorithm_gpts)
  {
    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::penalty &&
        Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::nitsche)
      FOUR_C_THROW("GPTS-Algorithm only with penalty or nitsche strategy");

    if (contact.get<double>("PENALTYPARAM") <= 0.0)
      FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
        Inpar::Wear::wear_none)
      FOUR_C_THROW("GPTS algorithm not implemented for wear");

    if (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") !=
        Inpar::Mortar::lagmult_undefined)
      FOUR_C_THROW("GPTS algorithm only implemented for first order interpolation");

    if (dim != 3) FOUR_C_THROW("GPTS algorithm only implemented for 3D contact");
  }  // END GPTS CHECKS

  // ---------------------------------------------------------------------
  // store contents of BOTH ParameterLists in local parameter list
  // ---------------------------------------------------------------------
  params.setParameters(mortar);
  params.setParameters(contact);
  params.setParameters(wearlist);
  params.setParameters(tsic);

  switch (problemtype)
  {
    case Core::ProblemType::tsi:
    {
      double timestep = Global::Problem::instance()->tsi_dynamic_params().get<double>("TIMESTEP");
      // rauch 01/16
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "\n \n  Warning: CONTACT::STRATEGY::Factory::read_and_check_input() reads "
                     "TIMESTEP = "
                  << timestep << " from Global::Problem::instance()->TSIDynamicParams().  \n"
                  << "Anyway, you should not use the \"TIMESTEP\" variable inside of "
                  << "the new structural/contact framework!" << std::endl;
      }
      params.set<double>("TIMESTEP", timestep);
      break;
    }
    case Core::ProblemType::structure:
    {
      params.set<double>("TIMESTEP",
          Global::Problem::instance()->structural_dynamic_params().get<double>("TIMESTEP"));
      break;
    }
    case Core::ProblemType::poroelast:
    case Core::ProblemType::poroscatra:
    {
      const Teuchos::ParameterList& stru = Global::Problem::instance()->structural_dynamic_params();
      const Teuchos::ParameterList& porodyn =
          Global::Problem::instance()->poroelast_dynamic_params();
      // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
      double porotimefac =
          1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
      params.set<double>("porotimefac", porotimefac);
      params.set<bool>("CONTACT_NO_PENETRATION",
          porodyn.get<bool>("CONTACT_NO_PENETRATION"));  // used in the integrator
      break;
    }
    default:
      /* Do nothing, all the time integration related stuff is supposed to be handled outside
       * of the contact strategy. */
      break;
  }

  // ---------------------------------------------------------------------
  // NURBS contact
  // ---------------------------------------------------------------------
  switch (distype)
  {
    case Core::FE::ShapeFunctionType::nurbs:
    {
      params.set<bool>("NURBS", true);
      break;
    }
    default:
    {
      params.set<bool>("NURBS", false);
      break;
    }
  }

  // ---------------------------------------------------------------------
  params.setName("CONTACT DYNAMIC / MORTAR COUPLING");

  // store relevant problem types
  if (problemtype == Core::ProblemType::tsi)
  {
    params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::tsi);
  }
  else if (problemtype == Core::ProblemType::ssi)
  {
    if (Teuchos::getIntegralValue<SSI::ScaTraTimIntType>(
            Global::Problem::instance()->ssi_control_params(), "SCATRATIMINTTYPE") ==
        SSI::ScaTraTimIntType::elch)
    {
      params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::ssi_elch);
    }
    else
    {
      params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::ssi);
    }
  }
  else if (problemtype == Core::ProblemType::poroelast or problemtype == Core::ProblemType::fpsi or
           problemtype == Core::ProblemType::fpsi_xfem)
  {
    params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::poroelast);
  }
  else if (problemtype == Core::ProblemType::fsi_xfem)
  {
    params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::fsi);
  }
  else if (problemtype == Core::ProblemType::fpsi_xfem)
  {
    FOUR_C_THROW(
        "Everything which is related to a special time integration scheme has to be moved to the"
        " related scheme. Don't do it here! -- hiermeier 02/2016");
    const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();
    params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::fpi);
    //    //porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    //    double porotimefac = 1/(stru.sublist("ONESTEPTHETA").get<double>("THETA") *
    //    stru.get<double>("TIMESTEP")); params.set<double> ("porotimefac", porotimefac);
    params.set<bool>("CONTACT_NO_PENETRATION",
        porodyn.get<bool>("CONTACT_NO_PENETRATION"));  // used in the integrator
  }
  else
  {
    params.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::other);
  }

  // no parallel redistribution in the serial case
  if (Core::Communication::num_mpi_ranks(get_comm()) == 1)
    params.sublist("PARALLEL REDISTRIBUTION")
        .set<Inpar::Mortar::ParallelRedist>(
            "PARALLEL_REDIST", Inpar::Mortar::ParallelRedist::redist_none);

  // console output at the end
  if (Core::Communication::my_mpi_rank(get_comm()) == 0) std::cout << "done!" << std::endl;

  // set dimension
  params.set<int>("DIMENSION", dim);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::build_interfaces(const Teuchos::ParameterList& params,
    std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces, bool& poroslave,
    bool& poromaster) const
{
  // start building interfaces
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Building contact interface(s)..............." << std::endl;
    fflush(stdout);
  }

  // Vector that solely contains solid-to-solid contact pairs
  std::vector<std::vector<const Core::Conditions::Condition*>> ccond_grps;
  CONTACT::Utils::get_contact_condition_groups(ccond_grps, discret());

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with displacement dofs
  int maxdof = discret().dof_row_map()->max_all_gid();

  // get input par.
  auto stype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params, "STRATEGY");
  auto wlaw = Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(params, "WEARLAW");
  auto constr_direction =
      Teuchos::getIntegralValue<CONTACT::ConstraintDirection>(params, "CONSTRAINT_DIRECTIONS");
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(params, "FRICTION");
  auto ad = Teuchos::getIntegralValue<CONTACT::AdhesionType>(params, "ADHESION");
  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(params, "ALGORITHM");

  bool friplus = false;
  if ((wlaw != Inpar::Wear::wear_none) ||
      (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi))
    friplus = true;

  // only for poro
  bool isporo = (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast) ||
                (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra);
  bool structmaster = false;
  bool structslave = false;
  bool isanyselfcontact = false;
  enum Mortar::Element::PhysicalType slavetype = Mortar::Element::other;
  enum Mortar::Element::PhysicalType mastertype = Mortar::Element::other;

  // loop over all contact condition groups
  for (auto& currentgroup : ccond_grps)
  {
    // initialize a reference to the i-th contact condition group
    const auto groupid1 = currentgroup[0]->parameters().get<int>("InterfaceID");

    // In case of MultiScale contact this is the id of the interface's constitutive contact law
    auto contactconstitutivelaw_id =
        currentgroup[0]->parameters().get<std::optional<int>>("ConstitutiveLawID");

    // Initialize a flag to check for MIRCO contact consitutive law
    bool mircolaw = false;

    // Initialize the variables to create a RoughNode for MIRCO
    int resolution = 0;
    bool randomtopologyflag = false;
    bool randomseedflag = false;
    int randomgeneratorseed = 0;
    int hurstexponentfunction = 0;
    int initialtopologystddeviationfunction = 0;

    if (contactconstitutivelaw_id.has_value() && contactconstitutivelaw_id.value() > 0)
    {
      const int probinst =
          Global::Problem::instance()->contact_constitutive_laws()->get_read_from_problem();
      auto coconstlaw = Global::Problem::instance(probinst)->contact_constitutive_laws()->by_id(
          contactconstitutivelaw_id.value());
      // Set the variables if MIRCO contact constitutive law is found
      if (coconstlaw.get<CONTACT::CONSTITUTIVELAW::ConstitutiveLawType>("LAW_TYPE") ==
          CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_mirco)
      {
        mircolaw = true;
        const auto& mirco_data = coconstlaw.group("CoConstLaw_mirco");
        resolution = mirco_data.get<int>("Resolution");
        randomtopologyflag = mirco_data.get<bool>("RandomTopologyFlag");
        randomseedflag = mirco_data.get<bool>("RandomSeedFlag");
        randomgeneratorseed = mirco_data.get<int>("RandomGeneratorSeed");
        hurstexponentfunction = mirco_data.get<int>("HurstExponentFunct");
        initialtopologystddeviationfunction =
            mirco_data.get<int>("InitialTopologyStdDeviationFunct");
      }
    }

    // find out which sides are Master and Slave
    std::vector<bool> isslave;
    std::vector<bool> isself;
    CONTACT::Utils::get_master_slave_side_info(isslave, isself, currentgroup);
    for (const bool is : isself)
    {
      if (is)
      {
        isanyselfcontact = true;
        break;
      }
    }

    // find out which sides are initialized as In/Active and other initialization data
    std::vector<bool> isactive(currentgroup.size());
    bool Two_half_pass(false);
    bool Check_nonsmooth_selfcontactsurface(false);
    bool Searchele_AllProc(false);

    CONTACT::Utils::get_initialization_info(Two_half_pass, Check_nonsmooth_selfcontactsurface,
        Searchele_AllProc, isactive, isslave, isself, currentgroup);

    // create interface local parameter list (copy)
    Teuchos::ParameterList icparams = params;

    // find out if interface-specific coefficients of friction are given
    if (ftype == CONTACT::FrictionType::tresca or ftype == CONTACT::FrictionType::coulomb or
        ftype == CONTACT::FrictionType::stick)
    {
      // read interface COFs
      std::vector<double> frcoeff(currentgroup.size());
      for (std::size_t j = 0; j < currentgroup.size(); ++j)
        frcoeff[j] = currentgroup[j]->parameters().get<double>("FrCoeffOrBound");

      // check consistency of interface COFs
      for (std::size_t j = 1; j < currentgroup.size(); ++j)
        if (frcoeff[j] != frcoeff[0])
          FOUR_C_THROW("Inconsistency in friction coefficients of interface {}", groupid1);

      // check for infeasible value of COF
      if (frcoeff[0] < 0.0) FOUR_C_THROW("Negative FrCoeff / FrBound on interface {}", groupid1);

      // add COF locally to contact parameter list of this interface
      if (ftype == CONTACT::FrictionType::tresca)
      {
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      // dummy values for FRCOEFF and FRBOUND have to be set,
      // since entries are accessed regardless of the friction law
      else if (ftype == CONTACT::FrictionType::stick)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
    }

    // find out if interface-specific coefficients of adhesion are given
    if (ad == CONTACT::AdhesionType::bounded)
    {
      // read interface COFs
      std::vector<double> ad_bound(currentgroup.size());
      for (std::size_t j = 0; j < currentgroup.size(); ++j)
        ad_bound[j] = currentgroup[j]->parameters().get<double>("AdhesionBound");

      // check consistency of interface COFs
      for (std::size_t j = 1; j < currentgroup.size(); ++j)
        if (ad_bound[j] != ad_bound[0])
          FOUR_C_THROW("Inconsistency in adhesion bounds of interface {}", groupid1);

      // check for infeasible value of COF
      if (ad_bound[0] < 0.0) FOUR_C_THROW("Negative adhesion bound on interface {}", groupid1);

      // add COF locally to contact parameter list of this interface
      icparams.setEntry("ADHESION_BOUND", static_cast<Teuchos::ParameterEntry>(ad_bound[0]));
    }

    // add information to contact parameter list of this interface
    icparams.set<bool>("Two_half_pass", Two_half_pass);
    icparams.set<bool>("Check_nonsmooth_selfcontactsurface", Check_nonsmooth_selfcontactsurface);
    icparams.set<bool>("Searchele_AllProc", Searchele_AllProc);

    set_parameters_for_contact_condition(groupid1, icparams);

    // for structural contact we currently choose redundant master storage
    // the only exception is self contact where a redundant slave is needed, too
    auto redundant = Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(
        icparams.sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");
    if (isanyselfcontact && redundant != Inpar::Mortar::ExtendGhosting::redundant_all)
      FOUR_C_THROW("Self contact requires fully redundant slave and master storage");

    // ------------------------------------------------------------------------
    // create the desired interface object
    // ------------------------------------------------------------------------
    const auto& non_owning_discret = Core::Utils::shared_ptr_from_ref(discret());

    std::shared_ptr<CONTACT::Interface> newinterface = create_interface(groupid1, get_comm(),
        n_dim(), icparams, isself[0], nullptr, contactconstitutivelaw_id.value_or(-1));
    interfaces.push_back(newinterface);

    // get it again
    const std::shared_ptr<CONTACT::Interface>& interface = interfaces.back();

    /* note that the nodal ids are unique because they come from
     * one global problem discretization containing all nodes of the
     * contact interface.
     * We rely on this fact, therefore it is not possible to
     * do contact between two distinct discretizations here. */

    // collect all initial active nodes
    std::vector<int> initialactive_nodeids;

    //-------------------------------------------------- process nodes
    for (int j = 0; j < (int)currentgroup.size(); ++j)
    {
      // get all nodes and add them
      const std::vector<int>* nodeids = currentgroup[j]->get_nodes();
      if (!nodeids) FOUR_C_THROW("Condition does not have Node Ids");
      for (int gid : *nodeids)
      {
        // do only nodes that I have in my discretization
        if (!discret().have_global_node(gid)) continue;
        Core::Nodes::Node* node = discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

        if (node->num_element() == 0)
        {
          FOUR_C_THROW(
              "surface node without adjacent element detected! "
              "(node-id = {})",
              node->id());
        }

        const bool nurbs =
            Core::FE::is_nurbs_celltype(node->adjacent_elements()[0].user_element()->shape());
        for (auto ele : node->adjacent_elements())
        {
          const Core::Elements::Element* adj_ele = ele.user_element();
          if (nurbs != Core::FE::is_nurbs_celltype(adj_ele->shape()))
          {
            FOUR_C_THROW(
                "There are NURBS and non-NURBS adjacent elements to this "
                "node. What shall be done?");
          }
        }

        // store initial active node gids
        if (isactive[j]) initialactive_nodeids.push_back(gid);

        /* find out if this node is initial active on another Condition
         * and do NOT overwrite this status then! */
        bool foundinitialactive = false;
        if (!isactive[j])
        {
          for (int initialactive_nodeid : initialactive_nodeids)
          {
            if (gid == initialactive_nodeid)
            {
              foundinitialactive = true;
              break;
            }
          }
        }

        /* create Node object or FriNode object in the frictional case
         * for the boolean variable initactive we use isactive[j]+foundinitialactive,
         * as this is true for BOTH initial active nodes found for the first time
         * and found for the second, third, ... time! */
        if (ftype != CONTACT::FrictionType::none)
        {
          std::shared_ptr<CONTACT::FriNode> cnode =
              std::make_shared<CONTACT::FriNode>(node->id(), node->x(), node->owner(),
                  discret().dof(0, node), isslave[j], isactive[j] + foundinitialactive, friplus);
          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            prepare_nurbs_node(node, *cnode);
          }

          // get edge and corner information:
          std::vector<const Core::Conditions::Condition*> contactCornerConditions;
          discret().get_condition("mrtrcorner", contactCornerConditions);
          for (auto& condition : contactCornerConditions)
          {
            if (condition->contains_node(node->id()))
            {
              cnode->set_on_corner() = true;
            }
          }
          std::vector<const Core::Conditions::Condition*> contactEdgeConditions;
          discret().get_condition("mrtredge", contactEdgeConditions);
          for (auto& condition : contactEdgeConditions)
          {
            if (condition->contains_node(node->id()))
            {
              cnode->set_on_edge() = true;
            }
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<const Core::Conditions::Condition*> contactSymConditions;
          discret().get_condition("mrtrsym", contactSymConditions);

          for (auto& condition : contactSymConditions)
          {
            if (condition->contains_node(node->id()))
            {
              const auto& onoff = condition->parameters().get<std::vector<int>>("ONOFF");
              for (unsigned k = 0; k < onoff.size(); k++)
                if (onoff.at(k) == 1) cnode->dbc_dofs()[k] = true;
              if (stype == CONTACT::SolvingStrategy::lagmult &&
                  constr_direction != CONTACT::ConstraintDirection::xyz)
              {
                FOUR_C_THROW(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
              }
            }
          }

          /* note that we do not have to worry about double entries
           * as the add_node function can deal with this case!
           * the only problem would have occurred for the initial active nodes,
           * as their status could have been overwritten, but is prevented
           * by the "foundinitialactive" block above! */
          interface->add_node(cnode);
        }
        else
        {
          std::shared_ptr<CONTACT::Node> cnode;
          if (mircolaw == true)
          {
            cnode = std::make_shared<CONTACT::RoughNode>(node->id(), node->x(), node->owner(),
                discret().dof(0, node), isslave[j], isactive[j] + foundinitialactive,
                hurstexponentfunction, initialtopologystddeviationfunction, resolution,
                randomtopologyflag, randomseedflag, randomgeneratorseed);
          }
          else
          {
            cnode = std::make_shared<CONTACT::Node>(node->id(), node->x(), node->owner(),
                discret().dof(0, node), isslave[j], isactive[j] + foundinitialactive);
          }

          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            prepare_nurbs_node(node, *cnode);
          }

          // get edge and corner information:
          std::vector<const Core::Conditions::Condition*> contactCornerConditions;
          discret().get_condition("mrtrcorner", contactCornerConditions);
          for (auto& condition : contactCornerConditions)
          {
            if (condition->contains_node(node->id()))
            {
              cnode->set_on_corner() = true;
            }
          }
          std::vector<const Core::Conditions::Condition*> contactEdgeConditions;
          discret().get_condition("mrtredge", contactEdgeConditions);
          for (auto& condition : contactEdgeConditions)
          {
            if (condition->contains_node(node->id()))
            {
              cnode->set_on_edge() = true;
            }
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<const Core::Conditions::Condition*> contactSymConditions;
          discret().get_condition("mrtrsym", contactSymConditions);

          for (auto& condition : contactSymConditions)
          {
            if (condition->contains_node(node->id()))
            {
              const auto& onoff = condition->parameters().get<std::vector<int>>("ONOFF");
              for (unsigned k = 0; k < onoff.size(); k++)
              {
                if (onoff.at(k) == 1)
                {
                  cnode->dbc_dofs()[k] = true;
                  if (stype == CONTACT::SolvingStrategy::lagmult &&
                      constr_direction != CONTACT::ConstraintDirection::xyz)
                  {
                    FOUR_C_THROW(
                        "Contact symmetry with Lagrange multiplier method"
                        " only with contact constraints in xyz direction.\n"
                        "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
                  }
                }
              }
            }
          }

          /* note that we do not have to worry about double entries
           * as the add_node function can deal with this case!
           * the only problem would have occurred for the initial active nodes,
           * as their status could have been overwritten, but is prevented
           * by the "foundinitialactive" block above! */
          interface->add_node(cnode);
        }
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (std::size_t j = 0; j < currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      const std::map<int, std::shared_ptr<Core::Elements::Element>>& currele =
          currentgroup[j]->geometry();

      /* elements in a boundary condition have a unique id
       * but ids are not unique among 2 distinct conditions
       * due to the way elements in conditions are build.
       * We therefore have to give the second, third,... set of elements
       * different ids. ids do not have to be continuous, we just add a large
       * enough number ggsize to all elements of cond2, cond3,... so they are
       * different from those in cond1!!!
       * note that elements in ele1/ele2 already are in column (overlapping) map */

      /* We count only elements, which are owned by the processor. In this way
       * the element ids stay the same for more than one processor in use.
       * hiermeier 02/2016 */
      int lsize = 0;
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator fool;
      for (fool = currele.begin(); fool != currele.end(); ++fool)
        if (fool->second->owner() == Core::Communication::my_mpi_rank(get_comm())) ++lsize;

      int gsize = 0;
      gsize = Core::Communication::sum_all(lsize, get_comm());

      bool nurbs = false;
      if (currele.size() > 0) nurbs = Core::FE::is_nurbs_celltype(currele.begin()->second->shape());

      for (fool = currele.begin(); fool != currele.end(); ++fool)
      {
        std::shared_ptr<Core::Elements::Element> ele = fool->second;
        if (Core::FE::is_nurbs_celltype(ele->shape()) != nurbs)
        {
          FOUR_C_THROW(
              "All elements of one interface side (i.e. slave or master) "
              "must be NURBS or LAGRANGE elements. A mixed NURBS/Lagrange "
              "discretizations on one side of the interface is currently "
              "unsupported.");
        }

        std::shared_ptr<CONTACT::Element> cele =
            std::make_shared<CONTACT::Element>(ele->id() + ggsize, ele->owner(), ele->shape(),
                ele->num_node(), ele->node_ids(), isslave[j], nurbs);

        if (isporo) set_poro_parent_element(slavetype, mastertype, *cele, ele, discret());

        if (algo == Inpar::Mortar::algorithm_gpts)
        {
          std::shared_ptr<Core::Elements::FaceElement> faceele =
              std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
          if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
          if (faceele->parent_element() == nullptr) FOUR_C_THROW("face parent does not exist");
          if (discret().element_col_map()->lid(faceele->parent_element()->id()) == -1)
            FOUR_C_THROW("vol dis does not have parent ele");
          cele->set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());
        }

        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (nurbs)
        {
          prepare_nurbs_element(discret(), ele, *cele);
        }

        interface->add_element(cele);
      }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)

      ggsize += gsize;  // update global element counter
    }

    //-------------------- finalize the contact interface construction
    interface->fill_complete(Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true, maxdof);

    if (isporo)
      find_poro_interface_types(
          poromaster, poroslave, structmaster, structslave, slavetype, mastertype);

  }  // for (int i=0; i<(int)contactconditions.size(); ++i)

  if (not isanyselfcontact) fully_overlapping_interfaces(interfaces);

  // finish the interface creation
  if (Core::Communication::my_mpi_rank(get_comm()) == 0) std::cout << "done!" << std::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::fully_overlapping_interfaces(
    std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces) const
{
  int ocount = 0;
  for (auto it = interfaces.begin(); it != interfaces.end(); ++it, ++ocount)
  {
    Interface& interface = **it;

    const Core::LinAlg::Map& srownodes_i = *interface.slave_row_nodes();
    const Core::LinAlg::Map& mrownodes_i = *interface.master_row_nodes();

    for (auto it2 = (it + 1); it2 != interfaces.end(); ++it2)
    {
      Interface& iinterface = **it2;

      const Core::LinAlg::Map& srownodes_ii = *iinterface.slave_row_nodes();
      const Core::LinAlg::Map& mrownodes_ii = *iinterface.master_row_nodes();

      const int sl_fullsubset_id = identify_full_subset(srownodes_i, srownodes_ii);
      if (sl_fullsubset_id != -1)
        FOUR_C_THROW("Currently the slave element maps are not allowed to overlap!");

      const int ma_fullsubset_id = identify_full_subset(mrownodes_i, mrownodes_ii);

      // handle fully overlapping master interfaces
      if (ma_fullsubset_id == 0)
        interface.add_ma_sharing_ref_interface(&iinterface);
      else if (ma_fullsubset_id == 1)
        iinterface.add_ma_sharing_ref_interface(&interface);
    }
  }

  for (const auto& inter : interfaces)
  {
    if (inter->has_ma_sharing_ref_interface())
    {
      Core::IO::cout << "master side of mortar interface #" << inter->id()
                     << " is fully overlapping with the master side of interface #"
                     << inter->get_ma_sharing_ref_interface().id() << Core::IO::endl;
    }
  }

  for (auto& interface : interfaces)
    Core::IO::cout << interface->id() << " has_ma_sharing_ref_interface  = "
                   << (interface->has_ma_sharing_ref_interface() ? "TRUE" : "FALSE")
                   << Core::IO::endl;

  // resort the interface vector via a short bubble sort:
  /* Move all interfaces with a shared reference interface to the end of the
   * vector */
  for (auto it = interfaces.begin(); it != interfaces.end(); ++it)
  {
    if ((*it)->has_ma_sharing_ref_interface())
    {
      for (auto it2 = it + 1; it2 != interfaces.end(); ++it2)
      {
        if (not(*it2)->has_ma_sharing_ref_interface())
        {
          std::swap(*it, *it2);
          break;
        }
      }
    }
  }

  Core::IO::cout << "After sorting:\n";
  for (auto& interface : interfaces)
    Core::IO::cout << interface->id() << " has_ma_sharing_ref_interface  = "
                   << (interface->has_ma_sharing_ref_interface() ? "TRUE" : "FALSE")
                   << Core::IO::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int CONTACT::STRATEGY::Factory::identify_full_subset(const Core::LinAlg::Map& map_0,
    const Core::LinAlg::Map& map_1, bool throw_if_partial_subset_on_proc) const
{
  const Core::LinAlg::Map* ref_map = nullptr;
  const Core::LinAlg::Map* sub_map = nullptr;

  int sub_id = -1;

  if (map_0.num_global_elements() >= map_1.num_global_elements())
  {
    ref_map = &map_0;

    sub_id = 1;
    sub_map = &map_1;
  }
  else
  {
    ref_map = &map_1;

    sub_id = 0;
    sub_map = &map_0;
  }

  const unsigned nummysubentries = sub_map->num_my_elements();
  const int* mysubgids = sub_map->my_global_elements();

  bool is_fullsubmap = false;
  for (unsigned i = 0; i < nummysubentries; ++i)
  {
    if (i == 0 and ref_map->my_gid(mysubgids[i]))
      is_fullsubmap = true;
    else if (is_fullsubmap != ref_map->my_gid(mysubgids[i]))
    {
      if (throw_if_partial_subset_on_proc)
        FOUR_C_THROW(
            "Partial sub-map detected on proc #{}!", Core::Communication::my_mpi_rank(get_comm()));
      is_fullsubmap = false;
    }
  }

  if (nummysubentries == 0) is_fullsubmap = true;

  int lfullsubmap = static_cast<int>(is_fullsubmap);
  int gfullsubmap = 0;

  gfullsubmap = Core::Communication::sum_all(lfullsubmap, get_comm());

  return (gfullsubmap == Core::Communication::num_mpi_ranks(get_comm()) ? sub_id : -1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<CONTACT::Interface> CONTACT::STRATEGY::Factory::create_interface(const int id,
    MPI_Comm comm, const int dim, Teuchos::ParameterList& icparams, const bool selfcontact,
    std::shared_ptr<CONTACT::InterfaceDataContainer> interfaceData_ptr,
    const int contactconstitutivelaw_id)
{
  auto stype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(icparams, "STRATEGY");

  return create_interface(
      stype, id, comm, dim, icparams, selfcontact, interfaceData_ptr, contactconstitutivelaw_id);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<CONTACT::Interface> CONTACT::STRATEGY::Factory::create_interface(
    const enum CONTACT::SolvingStrategy stype, const int id, MPI_Comm comm, const int dim,
    Teuchos::ParameterList& icparams, const bool selfcontact,
    std::shared_ptr<CONTACT::InterfaceDataContainer> interface_data_ptr,
    const int contactconstitutivelaw_id)
{
  std::shared_ptr<CONTACT::Interface> newinterface = nullptr;

  auto wlaw = Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(icparams, "WEARLAW");

  switch (stype)
  {
    case CONTACT::SolvingStrategy::multiscale:
      interface_data_ptr = std::make_shared<CONTACT::InterfaceDataContainer>();
      newinterface = std::make_shared<CONTACT::ConstitutivelawInterface>(
          interface_data_ptr, id, comm, dim, icparams, selfcontact, contactconstitutivelaw_id);
      break;
    // ------------------------------------------------------------------------
    // Default case for the wear, TSI and standard Lagrangian case
    // ------------------------------------------------------------------------
    default:
    {
      interface_data_ptr = std::make_shared<CONTACT::InterfaceDataContainer>();

      if (wlaw != Inpar::Wear::wear_none)
      {
        newinterface = std::make_shared<Wear::WearInterface>(
            interface_data_ptr, id, comm, dim, icparams, selfcontact);
      }
      else if (icparams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi &&
               stype == CONTACT::SolvingStrategy::lagmult)
      {
        newinterface = std::make_shared<CONTACT::TSIInterface>(
            interface_data_ptr, id, comm, dim, icparams, selfcontact);
      }
      else
        newinterface = std::make_shared<CONTACT::Interface>(
            interface_data_ptr, id, comm, dim, icparams, selfcontact);
      break;
    }
  }

  return newinterface;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::set_poro_parent_element(
    enum Mortar::Element::PhysicalType& slavetype, enum Mortar::Element::PhysicalType& mastertype,
    CONTACT::Element& cele, std::shared_ptr<Core::Elements::Element>& ele,
    const Core::FE::Discretization& discret) const
{
  // ints to communicate decision over poro bools between processors on every interface
  // safety check - because there may not be mixed interfaces and structural slave elements
  std::shared_ptr<Core::Elements::FaceElement> faceele =
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
  if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
  cele.phys_type() = Mortar::Element::other;
  std::vector<const Core::Conditions::Condition*> poroCondVec;
  discret.get_condition("PoroCoupling", poroCondVec);
  if (!cele.is_slave())  // treat an element as a master element if it is no slave element
  {
    for (auto& poroCond : poroCondVec)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = poroCond->geometry().begin();
          eleitergeometry != poroCond->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (mastertype == Mortar::Element::poro)
          {
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          }
          cele.phys_type() = Mortar::Element::poro;
          mastertype = Mortar::Element::poro;
          break;
        }
      }
    }
    if (cele.phys_type() == Mortar::Element::other)
    {
      if (mastertype == Mortar::Element::structure)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele.phys_type() = Mortar::Element::structure;
      mastertype = Mortar::Element::structure;
    }
  }
  else if (cele.is_slave())  // treat an element as slave element if it is one
  {
    for (auto& poroCond : poroCondVec)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = poroCond->geometry().begin();
          eleitergeometry != poroCond->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (slavetype == Mortar::Element::structure)
          {
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          }
          cele.phys_type() = Mortar::Element::poro;
          slavetype = Mortar::Element::poro;
          break;
        }
      }
    }
    if (cele.phys_type() == Mortar::Element::other)
    {
      if (slavetype == Mortar::Element::poro)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele.phys_type() = Mortar::Element::structure;
      slavetype = Mortar::Element::structure;
    }
  }
  // store information about parent for porous contact (required for calculation of deformation
  // gradient!) in every contact element although only really needed for phystype poro
  cele.set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::find_poro_interface_types(bool& poromaster, bool& poroslave,
    bool& structmaster, bool& structslave, enum Mortar::Element::PhysicalType& slavetype,
    enum Mortar::Element::PhysicalType& mastertype) const
{
  // find poro and structure elements when a poro coupling condition is applied on an element
  // and restrict to pure poroelastic or pure structural interfaces' sides.
  //(only poro slave elements AND (only poro master elements or only structure master elements)
  // Tell the contact element which physical type it is to extract PhysType in contact integrator
  // bools to decide which side is structural and which side is poroelastic to manage all 4
  // constellations
  // s-s, p-s, s-p, p-p
  // wait for all processors to determine if they have poro or structural master or slave elements
  Core::Communication::barrier(get_comm());
  /* FixMe Should become possible for scoped enumeration with C++11,
   * till then we use the shown workaround.
   *  enum Mortar::Element::PhysicalType slaveTypeList[Comm().NumProc()];
   *  enum Mortar::Element::PhysicalType masterTypeList[Comm().NumProc()];
   *  Comm().gather_all(static_cast<int*>(&slavetype),static_cast<int*>(slaveTypeList.data()),1);
   *  Comm().gather_all(static_cast<int*>(&mastertype),static_cast<int*>(masterTypeList.data()),1);
   */
  std::vector<int> slaveTypeList(Core::Communication::num_mpi_ranks(get_comm()));
  std::vector<int> masterTypeList(Core::Communication::num_mpi_ranks(get_comm()));
  int int_slavetype = static_cast<int>(slavetype);
  int int_mastertype = static_cast<int>(mastertype);
  Core::Communication::gather_all(&int_slavetype, slaveTypeList.data(), 1, get_comm());
  Core::Communication::gather_all(&int_mastertype, masterTypeList.data(), 1, get_comm());
  Core::Communication::barrier(get_comm());

  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i)
  {
    switch (slaveTypeList[i])
    {
      case static_cast<int>(Mortar::Element::other):
        break;
      case static_cast<int>(Mortar::Element::poro):
      {
        if (structslave)
        {
          FOUR_C_THROW(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        }
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poroslave = true;
        break;
      }
      case static_cast<int>(Mortar::Element::structure):
      {
        if (poroslave)
        {
          FOUR_C_THROW(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        }
        structslave = true;
        break;
      }
      default:
      {
        FOUR_C_THROW("this cannot happen");
        break;
      }
    }
  }

  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i)
  {
    switch (masterTypeList[i])
    {
      case static_cast<int>(Mortar::Element::other):
        break;
      case static_cast<int>(Mortar::Element::poro):
      {
        if (structmaster)
        {
          FOUR_C_THROW(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        }
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poromaster = true;
        break;
      }
      case static_cast<int>(Mortar::Element::structure):
      {
        if (poromaster)
        {
          FOUR_C_THROW(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        }
        structmaster = true;
        break;
      }
      default:
      {
        FOUR_C_THROW("this cannot happen");
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<CONTACT::AbstractStrategy> CONTACT::STRATEGY::Factory::build_strategy(
    const Teuchos::ParameterList& params, const bool& poroslave, const bool& poromaster,
    const int& dof_offset, std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
    CONTACT::ParamsInterface* cparams_interface) const
{
  const auto stype = Teuchos::getIntegralValue<enum CONTACT::SolvingStrategy>(params, "STRATEGY");
  std::shared_ptr<CONTACT::AbstractStrategyDataContainer> data_ptr = nullptr;

  return build_strategy(stype, params, poroslave, poromaster, dof_offset, interfaces,
      discret().dof_row_map(), discret().node_row_map(), n_dim(), get_comm(), data_ptr,
      cparams_interface);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<CONTACT::AbstractStrategy> CONTACT::STRATEGY::Factory::build_strategy(
    const CONTACT::SolvingStrategy stype, const Teuchos::ParameterList& params,
    const bool& poroslave, const bool& poromaster, const int& dof_offset,
    std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* node_row_map, const int dim,
    const MPI_Comm& comm_ptr, std::shared_ptr<CONTACT::AbstractStrategyDataContainer> data_ptr,
    CONTACT::ParamsInterface* cparams_interface)
{
  if (Core::Communication::my_mpi_rank(comm_ptr) == 0)
  {
    std::cout << "Building contact strategy object............";
    fflush(stdout);
  }
  std::shared_ptr<CONTACT::AbstractStrategy> strategy_ptr = nullptr;

  // get input par.
  auto wlaw = Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(params, "WEARLAW");
  auto wtype = Teuchos::getIntegralValue<Inpar::Wear::WearType>(params, "WEARTYPE");
  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(params, "ALGORITHM");

  // Set dummy parameter. The correct parameter will be read directly from time integrator. We still
  // need to pass an argument as long as we want to support the same strategy constructor as the old
  // time integration.
  double dummy = -1.0;

  // create LagrangeStrategyWear for wear as non-distinct quantity
  if (stype == CONTACT::SolvingStrategy::lagmult && wlaw != Inpar::Wear::wear_none &&
      (wtype == Inpar::Wear::wear_intstate || wtype == Inpar::Wear::wear_primvar))
  {
    data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
    strategy_ptr = std::make_shared<Wear::LagrangeStrategyWear>(
        data_ptr, dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, dummy, dof_offset);
  }
  else if (stype == CONTACT::SolvingStrategy::lagmult)
  {
    if (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
        params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra)
    {
      FOUR_C_THROW("This contact strategy is not yet considered!");
      //      strategy_ptr = Teuchos::rcp(new LagrangeStrategyPoro(
      //          dof_row_map,
      //          node_row_map,
      //          params,
      //          interfaces,
      //          dim,
      //          comm_ptr,
      //          maxdof,
      //          poroslave,
      //          poromaster));
    }
    else if (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi)
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<LagrangeStrategyTsi>(data_ptr, dof_row_map, node_row_map,
          params, interfaces, dim, comm_ptr, dummy, dof_offset);
    }
    else
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<LagrangeStrategy>(data_ptr, dof_row_map, node_row_map, params,
          interfaces, dim, comm_ptr, dummy, dof_offset);
    }
  }
  else if (((stype == CONTACT::SolvingStrategy::penalty or
                stype == CONTACT::SolvingStrategy::multiscale) &&
               algo != Inpar::Mortar::algorithm_gpts) &&
           stype != CONTACT::SolvingStrategy::uzawa)
  {
    strategy_ptr = std::make_shared<PenaltyStrategy>(
        dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, dummy, dof_offset);
  }
  else if (stype == CONTACT::SolvingStrategy::uzawa)
  {
    FOUR_C_THROW("This contact strategy is not yet considered!");
    //    strategy_ptr = Teuchos::rcp(new PenaltyStrategy(
    //        dof_row_map,
    //        node_row_map,
    //        params,
    //        interfaces,
    //        dim,
    //        comm_ptr,
    //        maxdof));
  }
  else if (algo == Inpar::Mortar::algorithm_gpts && (stype == CONTACT::SolvingStrategy::nitsche ||
                                                        stype == CONTACT::SolvingStrategy::penalty))
  {
    if (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::ssi)
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<NitscheStrategySsi>(
          data_ptr, dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, 0, dof_offset);
    }
    else if (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::ssi_elch)
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<NitscheStrategySsiElch>(
          data_ptr, dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, 0, dof_offset);
    }
    else if (params.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast)
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<NitscheStrategyPoro>(
          data_ptr, dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, 0, dof_offset);
    }
    else
    {
      data_ptr = std::make_shared<CONTACT::AbstractStrategyDataContainer>();
      strategy_ptr = std::make_shared<NitscheStrategy>(
          data_ptr, dof_row_map, node_row_map, params, interfaces, dim, comm_ptr, 0, dof_offset);
    }
  }
  else
  {
    FOUR_C_THROW("Unrecognized strategy: \"{}\"", CONTACT::solving_strategy_to_string(stype));
  }

  // setup the strategy object
  strategy_ptr->setup(false, true);

  if (Core::Communication::my_mpi_rank(comm_ptr) == 0) std::cout << "done!" << std::endl;

  return strategy_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::build_search_tree(
    const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces) const
{
  // create binary search tree
  for (const auto& interface : interfaces) interface->create_search_tree();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::print(
    const std::vector<std::shared_ptr<CONTACT::Interface>>& interfaces,
    CONTACT::AbstractStrategy& strategy_ptr, const Teuchos::ParameterList& params) const
{
  // print friction information of interfaces
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    // get input parameter
    auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(params, "FRICTION");

    for (unsigned i = 0; i < interfaces.size(); ++i)
    {
      double checkfrcoeff = 0.0;
      if (ftype == CONTACT::FrictionType::tresca)
      {
        checkfrcoeff = interfaces[i]->interface_params().get<double>("FRBOUND");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrBound (Tresca)  " << checkfrcoeff << std::endl;
      }
      else if (ftype == CONTACT::FrictionType::coulomb)
      {
        checkfrcoeff = interfaces[i]->interface_params().get<double>("FRCOEFF");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrCoeff (Coulomb) " << checkfrcoeff << std::endl;
      }
    }
  }

  // print initial parallel redistribution
  for (const auto& interface : interfaces) interface->print_parallel_distribution();

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    print_strategy_banner(strategy_ptr.type());
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::print_strategy_banner(const enum CONTACT::SolvingStrategy soltype)
{
  // some parameters
  const Teuchos::ParameterList& smortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& scontact = Global::Problem::instance()->contact_dynamic_params();
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(smortar, "LM_SHAPEFCN");
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(scontact, "SYSTEM");
  auto algorithm = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(smortar, "ALGORITHM");
  bool nonSmoothGeometries = scontact.get<bool>("NONSMOOTH_GEOMETRIES");

  if (nonSmoothGeometries)
  {
    if (soltype == CONTACT::SolvingStrategy::lagmult)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Lagrange Multiplier Strategy =============================\n";
      Core::IO::cout << "===== NONSMOOTH - GEOMETRIES ===================================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    else if (soltype == CONTACT::SolvingStrategy::nitsche and
             algorithm == Inpar::Mortar::algorithm_gpts)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Gauss-Point-To-Segment approach ==========================\n";
      Core::IO::cout << "===== using Nitsche formulation ================================\n";
      Core::IO::cout << "===== NONSMOOTH - GEOMETRIES ===================================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    else
      FOUR_C_THROW("Invalid system type for contact/meshtying interface smoothing");
  }
  else
  {
    if (algorithm == Inpar::Mortar::algorithm_mortar)
    {
      // saddle point formulation
      if (systype == CONTACT::SystemType::saddlepoint)
      {
        if (soltype == CONTACT::SolvingStrategy::lagmult &&
            shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Standard Lagrange multiplier strategy ====================\n";
          Core::IO::cout << "===== (Saddle point formulation) ===============================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Lagrange multiplier strategy ========================\n";
          Core::IO::cout << "===== (Saddle point formulation) ===============================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                 shapefcn == Inpar::Mortar::shape_petrovgalerkin)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Petrov-Galerkin Lagrange multiplier strategy =============\n";
          Core::IO::cout << "===== (Saddle point formulation) ===============================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::penalty &&
                 shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Standard Penalty strategy ================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::penalty &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Penalty strategy ====================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::uzawa &&
                 shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Uzawa Augmented Lagrange strategy ========================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::uzawa &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Uzawa Augmented Lagrange strategy ===================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else
          FOUR_C_THROW("Invalid strategy or shape function type for contact/meshtying");
      }

      // condensed formulation
      else if (systype == CONTACT::SystemType::condensed ||
               systype == CONTACT::SystemType::condensed_lagmult)
      {
        if (soltype == CONTACT::SolvingStrategy::lagmult && shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Lagrange multiplier strategy ========================\n";
          Core::IO::cout << "===== (Condensed formulation) ==================================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                 shapefcn == Inpar::Mortar::shape_standard &&
                 Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(smortar, "LM_QUAD") ==
                     Inpar::Mortar::lagmult_const)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== const Lagrange multiplier strategy =======================\n";
          Core::IO::cout << "===== (Condensed formulation) ==================================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                 shapefcn == Inpar::Mortar::shape_petrovgalerkin)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Petrov-Galerkin Lagrange multiplier strategy =============\n";
          Core::IO::cout << "===== (Condensed formulation) ==================================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::penalty &&
                 shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Standard Penalty strategy ================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::penalty &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Penalty strategy ====================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::multiscale &&
                 shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Multi Scale strategy ================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::multiscale &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout
              << "===== Dual Multi Scale strategy ====================================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::uzawa &&
                 shapefcn == Inpar::Mortar::shape_standard)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Uzawa Augmented Lagrange strategy ========================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else if (soltype == CONTACT::SolvingStrategy::uzawa &&
                 shapefcn == Inpar::Mortar::shape_dual)
        {
          Core::IO::cout << "================================================================\n";
          Core::IO::cout << "===== Dual Uzawa Augmented Lagrange strategy ===================\n";
          Core::IO::cout << "===== (Pure displacement formulation) ==========================\n";
          Core::IO::cout << "================================================================\n\n";
        }
        else
          FOUR_C_THROW("Invalid strategy or shape function type for contact/meshtying");
      }
    }
    else if (algorithm == Inpar::Mortar::algorithm_nts)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Node-To-Segment approach =================================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    else if (algorithm == Inpar::Mortar::algorithm_lts)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Line-To-Segment approach =================================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    else if (algorithm == Inpar::Mortar::algorithm_stl)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Segment-To-Line approach =================================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    else if (algorithm == Inpar::Mortar::algorithm_gpts)
    {
      Core::IO::cout << "================================================================\n";
      Core::IO::cout << "===== Gauss-Point-To-Segment approach ==========================\n";
      Core::IO::cout << "================================================================\n\n";
    }
    // invalid system type
    else
      FOUR_C_THROW("Invalid system type for contact/meshtying");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::STRATEGY::Factory::set_parameters_for_contact_condition(
    const int conditiongroupid, Teuchos::ParameterList& contactinterfaceparameters) const
{
  // add parameters if we have SSI contact
  if (discret().has_condition("SSIInterfaceContact"))
  {
    // get the scatra-scatra interface coupling condition
    std::vector<const Core::Conditions::Condition*> s2ikinetics_conditions;
    discret().get_condition("S2IKinetics", s2ikinetics_conditions);

    // create a sublist which is filled and added to the contact interface parameters
    auto& s2icouplingparameters = contactinterfaceparameters.sublist("ContactS2ICoupling");

    // loop over all s2i conditions and get the one with the same condition id (that they have to
    // match is assured within the setup of the SSI framework) at the slave-side, as only this
    // stores all the information
    for (const auto& s2ikinetics_cond : s2ikinetics_conditions)
    {
      // only add to parameters if condition ID's match
      if (s2ikinetics_cond->parameters().get<int>("ConditionID") == conditiongroupid)
      {
        // only the slave-side stores the parameters
        if (s2ikinetics_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
            Inpar::S2I::side_slave)
        {
          // fill the parameters from the s2i condition
          ScaTra::MeshtyingStrategyS2I::
              write_s2_i_kinetics_specific_scatra_parameters_to_parameter_list(
                  *s2ikinetics_cond, s2icouplingparameters);

          // add the sublist to the contact interface parameter list
          contactinterfaceparameters.setParameters(s2icouplingparameters);
        }
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
