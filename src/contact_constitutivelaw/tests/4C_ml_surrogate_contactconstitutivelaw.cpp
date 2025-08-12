// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_constitutivelaw_ml_surrogate_contactconstitutivelaw.hpp"
#include "4C_contact_node.hpp"

// #ifdef FOUR_C_WITH_ROUGH_SURFACE_ML_SURROGATE

namespace
{
  using namespace FourC;

  class MLSurrogateConstitutiveLawTest : public ::testing::Test
  {
   public:
    MLSurrogateConstitutiveLawTest()
    {
      /// initialize container for material parameters
      Core::IO::InputParameterContainer container;
      container.add("A", 1.5);
      container.add("B", 0.0);
      container.add("Offset", 0.5);

      CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLawParams params(container);
      coconstlaw_ = std::make_shared<CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLaw>(params);
    }

    std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw_;

    std::shared_ptr<CONTACT::Node> cnode;
  };

  //! test member function Evaluate
  TEST_F(MLSurrogateConstitutiveLawTest, TestEvaluate)
  {
    // gap < 0
    EXPECT_ANY_THROW(coconstlaw_->evaluate(1.0, cnode.get()));
    // 0< gap < offset
    EXPECT_ANY_THROW(coconstlaw_->evaluate(-0.25, cnode.get()));
    // offset < gap
    EXPECT_NEAR(coconstlaw_->evaluate(-0.75, cnode.get()), -0.375, 1.e-15);
  }

  //! test member function EvaluateDeriv
  TEST_F(MLSurrogateConstitutiveLawTest, TestEvaluateDeriv)
  {
    EXPECT_NEAR(coconstlaw_->evaluate_derivative(-0.75, cnode.get()), 1.5, 1.e-15);
    EXPECT_ANY_THROW(coconstlaw_->evaluate_derivative(-0.25, cnode.get()));
  }
}  // namespace

// #endif
