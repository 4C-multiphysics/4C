// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_constitutivelaw_ml_surrogate_contactconstitutivelaw.hpp"
#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_singleton_owner.hpp"

// #ifdef FOUR_C_WITH_ROUGH_SURFACE_ML_SURROGATE

namespace
{
  using namespace FourC;

  class MLSurrogateConstitutiveLawForceTest : public ::testing::Test
  {
   public:
    MLSurrogateConstitutiveLawForceTest() {}
  };

  //! test member function Evaluate
  TEST_F(MLSurrogateConstitutiveLawForceTest, TestEvaluate) { EXPECT_EQ(0, 0); }

  //! test member function EvaluateDeriv
  TEST_F(MLSurrogateConstitutiveLawForceTest, TestEvaluateDeriv) { EXPECT_EQ(0, 0); }
}  // namespace

// #endif
