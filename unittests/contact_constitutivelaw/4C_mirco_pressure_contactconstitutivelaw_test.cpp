// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_constitutivelaw_mirco_contactconstitutivelaw.hpp"
#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_function.hpp"

#ifdef FOUR_C_WITH_MIRCO

namespace
{
  using namespace FourC;

  class MircoConstitutiveLawPressureTest : public ::testing::Test
  {
   public:
    MircoConstitutiveLawPressureTest()
    {
      const int problemid(0);
      Global::Problem& problem = (*Global::Problem::instance());
      problem.materials()->set_read_from_problem(problemid);

      std::vector<Teuchos::RCP<Core::Utils::FunctionVariable>> variables;
      Teuchos::RCP<Core::Utils::SymbolicFunctionOfSpaceTime> FFUNCT1 =
          Teuchos::make_rcp<Core::Utils::SymbolicFunctionOfSpaceTime>(
              std::vector<std::string>{"0.7"}, variables);
      Teuchos::RCP<Core::Utils::SymbolicFunctionOfSpaceTime> FFUNCT2 =
          Teuchos::make_rcp<Core::Utils::SymbolicFunctionOfSpaceTime>(
              std::vector<std::string>{"20.0"}, variables);

      Teuchos::RCP<Core::Utils::FunctionOfSpaceTime> FUNCT1 = FFUNCT1;
      Teuchos::RCP<Core::Utils::FunctionOfSpaceTime> FUNCT2 = FFUNCT2;

      Core::Utils::FunctionManager functionmanager_;
      functionmanager_.set_functions<std::any>({FUNCT1, FUNCT2});
      problem.set_function_manager(std::move(functionmanager_));

      // set up material to be added to problem instance
      Core::IO::InputParameterContainer mat_stvenant;
      mat_stvenant.add("YOUNG", 1.0);
      mat_stvenant.add("NUE", 0.3);
      mat_stvenant.add("DENS", 1.0);

      problem.materials()->insert(
          1, Mat::make_parameter(1, Core::Materials::MaterialType::m_stvenant, mat_stvenant));

      // initialize container for material parameters
      const Teuchos::RCP<CONTACT::CONSTITUTIVELAW::Container> container =
          Teuchos::make_rcp<CONTACT::CONSTITUTIVELAW::Container>(
              1, Inpar::CONTACT::ConstitutiveLawType::colaw_mirco, "Mirco Constitutivelaw");

      // add parameters to container
      container->add("FirstMatID", 1);
      container->add("SecondMatID", 1);
      container->add("LateralLength", 1000.0);
      container->add("Resolution", 6);
      container->add("PressureGreenFunFlag", true);
      container->add("InitialTopologyStdDeviationFunct", 2);
      container->add("HurstExponentFunct", 1);
      container->add("RandomTopologyFlag", true);
      container->add("RandomSeedFlag", false);
      container->add("RandomGeneratorSeed", 95);
      container->add("Tolerance", 0.01);
      container->add("MaxIteration", 100);
      container->add("WarmStartingFlag", true);
      container->add("Offset", 2.0);
      container->add("FiniteDifferenceFraction", 0.001);
      container->add("ActiveGapTolerance", 1e-6);
      container->add("TopologyFilePath", std::string("sup6.dat"));

      const Teuchos::RCP<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> mircococonstlaw =
          CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::factory(container);
      coconstlaw_ = mircococonstlaw;

      std::vector<double> x(3, 0.0);
      std::vector<int> dofs(3);

      double hurstexponentfunct = container->get<int>("HurstExponentFunct");
      double initialtopologystddevfunct = container->get<int>("InitialTopologyStdDeviationFunct");
      int resolution = container->get<int>("Resolution");
      bool randomtopologyflag = container->get<bool>("RandomTopologyFlag");
      bool randomseedflag = container->get<bool>("RandomSeedFlag");
      int randomgeneratorseed = container->get<int>("RandomGeneratorSeed");

      cnode = Teuchos::make_rcp<CONTACT::RoughNode>(1, x, 1, dofs, true, true, hurstexponentfunct,
          initialtopologystddevfunct, resolution, randomtopologyflag, randomseedflag,
          randomgeneratorseed);
    }
    Teuchos::RCP<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw_;

    Teuchos::RCP<CONTACT::Node> cnode;
  };

  //! test member function Evaluate
  TEST_F(MircoConstitutiveLawPressureTest, TestEvaluate)
  {
    // gap < 0
    EXPECT_ANY_THROW(coconstlaw_->evaluate(1.0, cnode.get()));
    // 0< gap < offset
    EXPECT_ANY_THROW(coconstlaw_->evaluate(-0.25, cnode.get()));
    // offset < gap
    EXPECT_NEAR(coconstlaw_->evaluate(-12.0, cnode.get()), -0.0005861475487657709, 1.e-10);
  }

  //! test member function EvaluateDeriv
  TEST_F(MircoConstitutiveLawPressureTest, TestEvaluateDeriv)
  {
    EXPECT_NEAR(coconstlaw_->evaluate_deriv(-12.0, cnode.get()), 1.56329102801896e-04, 1.e-10);
    EXPECT_ANY_THROW(coconstlaw_->evaluate_deriv(-0.25, cnode.get()));
  }
}  // namespace

#endif
