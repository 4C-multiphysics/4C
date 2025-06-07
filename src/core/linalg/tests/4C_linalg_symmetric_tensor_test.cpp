// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <cstddef>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <typename T>
  class SymmetricTensorTemplatedTest : public ::testing::Test
  {
   public:
    static constexpr std::size_t size = T::value;
  };

  using test_sizes = ::testing::Types<std::integral_constant<std::size_t, 2>,
      std::integral_constant<std::size_t, 3>, std::integral_constant<std::size_t, 4>,
      std::integral_constant<std::size_t, 5>>;

  TYPED_TEST_SUITE(SymmetricTensorTemplatedTest, test_sizes);

  TYPED_TEST(SymmetricTensorTemplatedTest, TestConstructability)
  {
    constexpr std::size_t size = TypeParam::value;
    // Tensor should be default constructible
    static_assert(
        std::is_default_constructible_v<Core::LinAlg::SymmetricTensor<double, size, size>>);
  }

  TYPED_TEST(SymmetricTensorTemplatedTest, DefaultConstruct2Tensor)
  {
    constexpr std::size_t size = TypeParam::value;
    Core::LinAlg::SymmetricTensor<double, size, size> t{};

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size; ++j)
      {
        EXPECT_DOUBLE_EQ(t(i, j), 0.0);
      }
    }
  }

  TYPED_TEST(SymmetricTensorTemplatedTest, DefaultConstruct4Tensor)
  {
    constexpr std::size_t size = TypeParam::value;
    Core::LinAlg::SymmetricTensor<double, size, size, size, size> t{};

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size; ++j)
      {
        for (std::size_t k = 0; k < size; ++k)
        {
          for (std::size_t l = 0; l < size; ++l)
          {
            EXPECT_DOUBLE_EQ(t(i, j, k, l), 0.0);
          }
        }
      }
    }
  }

  TYPED_TEST(SymmetricTensorTemplatedTest, SymmetryRemainsAfterAssignmentFor2Tensor)
  {
    constexpr std::size_t size = TypeParam::value;
    Core::LinAlg::SymmetricTensor<double, size, size> t{};

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size - i; ++j)
      {
        t(i, j) = i * 10 + j;
      }
    }

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size - 1; ++j)
      {
        t(i, j) = i * 10 + j;
        t(j, i) = i * 10 + j;
      }
    }
  }

  TYPED_TEST(SymmetricTensorTemplatedTest, SymmetryRemainsAfterAssignmentFor4Tensor)
  {
    constexpr std::size_t size = TypeParam::value;
    Core::LinAlg::SymmetricTensor<double, size, size, size, size> t{};

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size - i; ++j)
      {
        for (std::size_t k = 0; k < size; ++k)
        {
          for (std::size_t l = 0; l < size - k; ++l)
          {
            t(i, j, k, l) = i * 10 + j + k * 0.1 + l * 0.01;
          }
        }
      }
    }

    for (std::size_t i = 0; i < size; ++i)
    {
      for (std::size_t j = 0; j < size - 1; ++j)
      {
        for (std::size_t k = 0; k < size; ++k)
        {
          for (std::size_t l = 0; l < size - k; ++l)
          {
            t(i, j, k, l) = i * 10 + j + k * 0.1 + l * 0.01;
            t(j, i, k, l) = i * 10 + j + k * 0.1 + l * 0.01;
            t(i, j, l, k) = i * 10 + j + k * 0.1 + l * 0.01;
            t(j, i, l, k) = i * 10 + j + k * 0.1 + l * 0.01;
          }
        }
      }
    }
  }

  TEST(SymmetricTensorTest, InitializationAndIndexing2Tensor)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> t = Core::LinAlg::assert_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{0.0, 0.1, 0.2}, {0.1, 1.1, 1.2}, {0.2, 1.2, 2.2}}});


    EXPECT_DOUBLE_EQ(t(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 0.1);
    EXPECT_DOUBLE_EQ(t(0, 2), 0.2);
    EXPECT_DOUBLE_EQ(t(1, 0), 0.1);
    EXPECT_DOUBLE_EQ(t(1, 1), 1.1);
    EXPECT_DOUBLE_EQ(t(1, 2), 1.2);
    EXPECT_DOUBLE_EQ(t(2, 0), 0.2);
    EXPECT_DOUBLE_EQ(t(2, 1), 1.2);
    EXPECT_DOUBLE_EQ(t(2, 2), 2.2);


    Core::LinAlg::SymmetricTensor<double, 2, 2> t2 =
        Core::LinAlg::assert_symmetry(Core::LinAlg::Tensor<double, 2, 2>{{{0.0, 0.1}, {0.1, 1.1}}});

    EXPECT_DOUBLE_EQ(t2(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(t2(0, 1), 0.1);
    EXPECT_DOUBLE_EQ(t2(1, 0), 0.1);
    EXPECT_DOUBLE_EQ(t2(1, 1), 1.1);
  }

  TEST(SymmetricTensorTest, AlignmentSameAsVoigtMatrix)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> t = Core::LinAlg::assert_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{0.0, 0.1, 0.2}, {0.1, 1.1, 1.2}, {0.2, 1.2, 2.2}}});

    Core::LinAlg::Matrix<6, 1> m_voigt(t.data());
    Core::LinAlg::Matrix<3, 3> m;
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(m_voigt, m);

    EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 0.1);
    EXPECT_DOUBLE_EQ(m(0, 2), 0.2);
    EXPECT_DOUBLE_EQ(m(1, 0), 0.1);
    EXPECT_DOUBLE_EQ(m(1, 1), 1.1);
    EXPECT_DOUBLE_EQ(m(1, 2), 1.2);
    EXPECT_DOUBLE_EQ(m(2, 0), 0.2);
    EXPECT_DOUBLE_EQ(m(2, 1), 1.2);
    EXPECT_DOUBLE_EQ(m(2, 2), 2.2);
  }


  TEST(SymmetricTensorTest, InitializationAndIndexing4Tensor)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2, 2, 2> t =
        Core::LinAlg::assert_symmetry(Core::LinAlg::Tensor<double, 2, 2, 2, 2>{
            {{{{0.000, 0.001}, {0.001, 0.011}}, {{0.100, 0.101}, {0.101, 0.111}}},
                {{{0.100, 0.101}, {0.101, 0.111}}, {{1.100, 1.101}, {1.101, 1.111}}}}});

    for (std::size_t i = 0; i < 2; ++i)
    {
      for (std::size_t j = 0; j < i + 1; ++j)
      {
        for (std::size_t k = 0; k < 2; ++k)
        {
          for (std::size_t l = 0; l < k + 1; ++l)
          {
            EXPECT_DOUBLE_EQ(t(i, j, k, l), i * 0.1 + j + k * 0.001 + l * 0.01);
            EXPECT_DOUBLE_EQ(t(i, j, l, k), i * 0.1 + j + k * 0.001 + l * 0.01);
            EXPECT_DOUBLE_EQ(t(j, i, k, l), i * 0.1 + j + k * 0.001 + l * 0.01);
            EXPECT_DOUBLE_EQ(t(j, i, l, k), i * 0.1 + j + k * 0.001 + l * 0.01);
          }
        }
      }
    }
  }

  TEST(SymmetricTensorTest, IsSymmetric)
  {
    {
      Core::LinAlg::Tensor<double, 2, 2> t2_nonsym = {{{1.0, 1.1}, {2.0, 2.1}}};
      EXPECT_FALSE(Core::LinAlg::is_symmetric(t2_nonsym));

      Core::LinAlg::Tensor<double, 2, 2> t2_sym = {{{1.0, 1.1}, {1.1, 2.1}}};
      EXPECT_TRUE(Core::LinAlg::is_symmetric(t2_sym));
    }

    {
      Core::LinAlg::Tensor<double, 3, 3> t2_nonsym = {
          {{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}};
      EXPECT_FALSE(Core::LinAlg::is_symmetric(t2_nonsym));

      Core::LinAlg::Tensor<double, 3, 3> t2_sym = {
          {{1.1, 1.2, 1.3}, {1.2, 2.2, 2.3}, {1.3, 2.3, 3.3}}};
      EXPECT_TRUE(Core::LinAlg::is_symmetric(t2_sym));
    }

    {
      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_nonsym = {{
          {
              {{0.0, 0.01}, {0.1, 0.11}},
              {{1.0, 1.01}, {1.1, 1.11}},
          },
          {
              {{10.0, 10.01}, {10.1, 10.11}},
              {{11.0, 11.01}, {11.1, 11.11}},
          },
      }};
      EXPECT_FALSE(Core::LinAlg::is_symmetric(t4_nonsym));

      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_sym = {{
          {
              {{0.0, 0.01}, {0.01, 0.11}},
              {{1.0, 1.01}, {1.01, 1.11}},
          },
          {
              {{1.0, 1.01}, {1.01, 1.11}},
              {{11.0, 11.01}, {11.01, 11.11}},
          },
      }};
      EXPECT_TRUE(Core::LinAlg::is_symmetric(t4_sym));
    }
  }

  TEST(SymmetricTensorTest, AssumeSymmetry)
  {
    {
      Core::LinAlg::Tensor<double, 2, 2> t2_sym = {{{1.0, 1.1}, {1.1, 2.1}}};
      FOUR_C_EXPECT_NEAR(Core::LinAlg::assume_symmetry(t2_sym), t2_sym, 1e-20);
    }

    {
      Core::LinAlg::Tensor<double, 3, 3> t2_sym = {
          {{1.1, 1.2, 1.3}, {1.2, 2.2, 2.3}, {1.3, 2.3, 3.3}}};
      FOUR_C_EXPECT_NEAR(Core::LinAlg::assume_symmetry(t2_sym), t2_sym, 1e-20);
    }

    {
      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_sym = {{
          {
              {{0.0, 0.01}, {0.01, 0.11}},
              {{1.0, 1.01}, {1.01, 1.11}},
          },
          {
              {{1.0, 1.01}, {1.01, 1.11}},
              {{11.0, 11.01}, {11.01, 11.11}},
          },
      }};
      FOUR_C_EXPECT_NEAR(Core::LinAlg::assume_symmetry(t4_sym), t4_sym, 1e-20);
    }
  }

  TEST(SymmetricTensorTest, GetFull)
  {
    {
      Core::LinAlg::Tensor<double, 2, 2> t2_sym = {{{1.0, 1.1}, {1.1, 2.1}}};
      FOUR_C_EXPECT_NEAR(
          Core::LinAlg::get_full(Core::LinAlg::assume_symmetry(t2_sym)), t2_sym, 1e-20);
    }

    {
      Core::LinAlg::Tensor<double, 3, 3> t2_sym = {
          {{1.1, 1.2, 1.3}, {1.2, 2.2, 2.3}, {1.3, 2.3, 3.3}}};
      FOUR_C_EXPECT_NEAR(
          Core::LinAlg::get_full(Core::LinAlg::assume_symmetry(t2_sym)), t2_sym, 1e-20);
    }

    {
      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_sym = {{
          {
              {{0.0, 0.01}, {0.01, 0.11}},
              {{1.0, 1.01}, {1.01, 1.11}},
          },
          {
              {{1.0, 1.01}, {1.01, 1.11}},
              {{11.0, 11.01}, {11.01, 11.11}},
          },
      }};
      FOUR_C_EXPECT_NEAR(
          Core::LinAlg::get_full(Core::LinAlg::assume_symmetry(t4_sym)), t4_sym, 1e-20);
    }
  }

  TEST(SymmetricTensorTest, AssertSymmetry)
  {
    {
      Core::LinAlg::Tensor<double, 2, 2> t2_nonsym = {{{1.0, 1.1}, {2.0, 2.1}}};
      EXPECT_ANY_THROW(Core::LinAlg::assert_symmetry(t2_nonsym));

      Core::LinAlg::Tensor<double, 2, 2> t2_sym = {{{1.0, 1.1}, {1.1, 2.1}}};
      EXPECT_NO_THROW(Core::LinAlg::assert_symmetry(t2_sym));
    }

    {
      Core::LinAlg::Tensor<double, 3, 3> t2_nonsym = {
          {{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}};
      EXPECT_ANY_THROW(Core::LinAlg::assert_symmetry(t2_nonsym));

      Core::LinAlg::Tensor<double, 3, 3> t2_sym = {
          {{1.1, 1.2, 1.3}, {1.2, 2.2, 2.3}, {1.3, 2.3, 3.3}}};
      EXPECT_NO_THROW(Core::LinAlg::assert_symmetry(t2_sym));
    }

    {
      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_nonsym = {{
          {
              {{0.0, 0.01}, {0.1, 0.11}},
              {{1.0, 1.01}, {1.1, 1.11}},
          },
          {
              {{10.0, 10.01}, {10.1, 10.11}},
              {{11.0, 11.01}, {11.1, 11.11}},
          },
      }};
      EXPECT_ANY_THROW(Core::LinAlg::assert_symmetry(t4_nonsym));

      Core::LinAlg::Tensor<double, 2, 2, 2, 2> t4_sym = {{
          {
              {{0.0, 0.01}, {0.01, 0.11}},
              {{1.0, 1.01}, {1.01, 1.11}},
          },
          {
              {{1.0, 1.01}, {1.01, 1.11}},
              {{11.0, 11.01}, {11.01, 11.11}},
          },
      }};
      EXPECT_NO_THROW(Core::LinAlg::assert_symmetry(t4_sym));
    }
  }

  TEST(SymmetricTensorTest, PrintTensor)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2 =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 2, 2>{{{0.0, 0.1}, {1.1, 1.1}}});
    std::stringstream ss;
    ss << t2;
    EXPECT_EQ(ss.str(), R"(SymmetricTensor<double, 2, 2>[
 [0, 0.1],
 [0.1, 1.1]
])");
  }

  TEST(SymmetricTensorViewTest, PrintTensor)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2 =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 2, 2>{{{0.0, 0.1}, {1.1, 1.1}}});


    Core::LinAlg::SymmetricTensorView<double, 2, 2> t2_view = t2;
    std::stringstream ss;
    ss << t2_view;
    EXPECT_EQ(ss.str(), R"(SymmetricTensorView<double, 2, 2>[
 [0, 0.1],
 [0.1, 1.1]
])");
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE