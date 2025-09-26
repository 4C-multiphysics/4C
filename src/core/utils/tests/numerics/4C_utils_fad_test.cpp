// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_fad.hpp"

#include <Sacado_Fad_SLFad.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <typename T>
  T compute_function(const T& x, const T& y)
  {
    return sin(x * x * y) * y;
  }

  TEST(CoreUtilsFadTest, IndicesRemapping)
  {
    using Inner = Sacado::Fad::DFad<double>;
    using Middle = Sacado::Fad::DFad<Inner>;
    using Outer = Sacado::Fad::DFad<Middle>;

    const auto x = Core::FADUtils::HigherOrderFadValue<Outer>::apply(2, 0, 2.0);
    const auto y = Core::FADUtils::HigherOrderFadValue<Outer>::apply(2, 1, 3.0);
    const auto f = compute_function(x, y);

    const auto x_full = Core::FADUtils::HigherOrderFadValue<Outer>::apply(4, 1, 2.0);
    const auto y_full = Core::FADUtils::HigherOrderFadValue<Outer>::apply(4, 3, 3.0);
    const auto f_full = compute_function(x_full, y_full);

    const auto remapping = Core::FADUtils::RemapInformation{
        .old_local_id_to_new_local_id_{1, 3}, .n_dependent_variables_ = 4};
    const auto f_lifted = Core::FADUtils::remap_fad_ordering<Outer>(f, remapping);

    // Compare the "full" ad result and the lifted one.
    EXPECT_EQ(f_full.size(), f_lifted.size());
    EXPECT_NEAR(f_full.val().val().val(), f_lifted.val().val().val(), 1e-12);
    for (unsigned int i = 0; i < 4; i++)
    {
      EXPECT_NEAR(f_full.dx(i).val().val(), f_lifted.dx(i).val().val(), 1e-12);
      EXPECT_NEAR(f_full.val().dx(i).val(), f_lifted.val().dx(i).val(), 1e-12);
      EXPECT_NEAR(f_full.val().val().dx(i), f_lifted.val().val().dx(i), 1e-12);

      for (unsigned int j = 0; j < 4; j++)
      {
        EXPECT_NEAR(f_full.dx(i).dx(j).val(), f_lifted.dx(i).dx(j).val(), 1e-12);
        EXPECT_NEAR(f_full.val().dx(i).dx(j), f_lifted.val().dx(i).dx(j), 1e-12);
        EXPECT_NEAR(f_full.dx(j).val().dx(i), f_lifted.dx(j).val().dx(i), 1e-12);

        for (unsigned int k = 0; k < 4; k++)
        {
          EXPECT_NEAR(f_full.dx(i).dx(j).dx(k), f_lifted.dx(i).dx(j).dx(k), 1e-12);
        }
      }
    }
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE
