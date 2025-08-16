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
#include "4C_utils_singleton_owner.hpp"

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

      std::vector<std::shared_ptr<Core::Utils::FunctionVariable>> variables;
      std::shared_ptr<Core::Utils::SymbolicFunctionOfSpaceTime> FFUNCT1 =
          std::make_shared<Core::Utils::SymbolicFunctionOfSpaceTime>(
              std::vector<std::string>{"0.7"}, variables);
      std::shared_ptr<Core::Utils::SymbolicFunctionOfSpaceTime> FFUNCT2 =
          std::make_shared<Core::Utils::SymbolicFunctionOfSpaceTime>(
              std::vector<std::string>{"20.0"}, variables);

      std::shared_ptr<Core::Utils::FunctionOfSpaceTime> FUNCT1 = FFUNCT1;
      std::shared_ptr<Core::Utils::FunctionOfSpaceTime> FUNCT2 = FFUNCT2;

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
      Core::IO::InputParameterContainer container;
      // add parameters to container
      container.add("FirstMatID", 1);
      container.add("SecondMatID", 1);
      container.add("LateralLength", 1000.0);
      container.add("Resolution", 6);
      container.add("PressureGreenFunFlag", true);
      container.add("InitialTopologyStdDeviationFunct", 2);
      container.add("HurstExponentFunct", 1);
      container.add("RandomTopologyFlag", true);
      container.add("RandomSeedFlag", false);
      container.add("RandomGeneratorSeed", 95);
      container.add("Tolerance", 0.01);
      container.add("MaxIteration", 100);
      container.add("WarmStartingFlag", true);
      container.add("Offset", 2.0);
      container.add("FiniteDifferenceFraction", 0.001);
      container.add("ActiveGapTolerance", 1e-6);
      container.add("TopologyFilePath", std::string("sup6.dat"));

      CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams mircococonstlaw(container);
      coconstlaw_ =
          std::make_shared<CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw>(mircococonstlaw);

      std::vector<double> x(3, 0.0);
      std::vector<int> dofs(3);

      double hurstexponentfunct = container.get<int>("HurstExponentFunct");
      double initialtopologystddevfunct = container.get<int>("InitialTopologyStdDeviationFunct");
      int resolution = container.get<int>("Resolution");
      bool randomtopologyflag = container.get<bool>("RandomTopologyFlag");
      bool randomseedflag = container.get<bool>("RandomSeedFlag");
      int randomgeneratorseed = container.get<int>("RandomGeneratorSeed");

      cnode = std::make_shared<CONTACT::RoughNode>(1, x, 1, dofs, true, true, hurstexponentfunct,
          initialtopologystddevfunct, resolution, randomtopologyflag, randomseedflag,
          randomgeneratorseed);
    }
    std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw_;

    std::shared_ptr<CONTACT::Node> cnode;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  //! test member function Evaluate
  TEST_F(MircoConstitutiveLawPressureTest, TestEvaluate)
  {
    // gap < 0
    EXPECT_ANY_THROW(coconstlaw_->evaluate(1.0, cnode.get()));
    // 0< gap < offset
    EXPECT_ANY_THROW(coconstlaw_->evaluate(-0.25, cnode.get()));
    // offset < gap
    EXPECT_NEAR(coconstlaw_->evaluate(-12.0, cnode.get()), -0.0004784628885090747, 1.e-10);
  }

  //! test member function EvaluateDeriv
  TEST_F(MircoConstitutiveLawPressureTest, TestEvaluateDeriv)
  {
    EXPECT_NEAR(coconstlaw_->evaluate_derivative(-12.0, cnode.get()), 1.17161352338802e-04, 1.e-10);
    EXPECT_ANY_THROW(coconstlaw_->evaluate_derivative(-0.25, cnode.get()));
  }
}  // namespace

#endif
