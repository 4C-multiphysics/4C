// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <std::size_t n>
  Core::LinAlg::SymmetricTensor<double, n, n> create_symmetric_tensor(
      double base_value = 1.0, double base_value2 = 0.1)
  {
    Core::LinAlg::SymmetricTensor<double, n, n> t{};
    for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
      {
        t(i, j) = base_value + i * i + j * j + (i + 1) * (j + 1) * base_value2;
      }
    }
    return t;
  }

  template <std::size_t n>
  Core::LinAlg::SymmetricTensor<double, n, n, n, n> create_symmetric_four_tensor(
      double base_value = 1.0)
  {
    Core::LinAlg::SymmetricTensor<double, n, n, n, n> t{};
    for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
      {
        for (std::size_t k = 0; k < n; ++k)
        {
          for (std::size_t l = 0; l < n; ++l)
          {
            t(i, j, k, l) =
                base_value + i * i + j * j + (i + 1) * (j + 1) * 0.1 + i * j * 0.01 + k * l * 0.01;
          }
        }
      }
    }
    return t;
  }

  template <std::size_t... n>
  Core::LinAlg::Tensor<double, n...> create_tensor()
  {
    Core::LinAlg::Tensor<double, n...> t{};

    double previous_entry = 0.0;
    for (auto& entry : t.container())
    {
      previous_entry += 1.0;
      entry = previous_entry;
    }
    return t;
  }

  TEST(SymmetricTensorOperationsTest, det)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    EXPECT_NEAR(Core::LinAlg::det(t2_sym), -1.1, 1e-10);


    Core::LinAlg::SymmetricTensor<double, 3, 3> t3_sym = create_symmetric_tensor<3>();
    EXPECT_NEAR(Core::LinAlg::det(t3_sym), -0.3999999999999977, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, trace)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    EXPECT_NEAR(Core::LinAlg::trace(t2_sym), 4.5, 1e-10);

    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym = create_symmetric_tensor<3>();
    EXPECT_NEAR(Core::LinAlg::trace(t_sym), 14.4, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, inv)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_inv = Core::LinAlg::inv(t2_sym);

    EXPECT_NEAR(t2_sym_inv(0, 0), -3.090909090909090909, 1e-10);
    EXPECT_NEAR(t2_sym_inv(0, 1), 2, 1e-10);

    EXPECT_NEAR(t2_sym_inv(1, 0), 2, 1e-10);
    EXPECT_NEAR(t2_sym_inv(1, 1), -1, 1e-10);


    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym = create_symmetric_tensor<3>();
    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym_inv = Core::LinAlg::inv(t_sym);

    EXPECT_NEAR(t_sym_inv(0, 0), 24.75, 1e-10);
    EXPECT_NEAR(t_sym_inv(0, 1), -33., 1e-10);
    EXPECT_NEAR(t_sym_inv(0, 2), 8.75, 1e-10);

    EXPECT_NEAR(t_sym_inv(1, 0), -33., 1e-10);
    EXPECT_NEAR(t_sym_inv(1, 1), 43., 1e-10);
    EXPECT_NEAR(t_sym_inv(1, 2), -11., 1e-10);

    EXPECT_NEAR(t_sym_inv(2, 0), 8.75, 1e-10);
    EXPECT_NEAR(t_sym_inv(2, 1), -11., 1e-10);
    EXPECT_NEAR(t_sym_inv(2, 2), 2.75, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, transpose)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_t = Core::LinAlg::transpose(t2_sym);
    FOUR_C_EXPECT_NEAR(t2_sym_t, t2_sym, 1e-20);


    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym = create_symmetric_tensor<3>();
    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym_t = Core::LinAlg::transpose(t_sym);
    FOUR_C_EXPECT_NEAR(t_sym_t, t_sym, 1e-20);
  }

  TEST(SymmetricTensorOperationsTest, dot)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    {
      Core::LinAlg::Tensor<double, 2, 3> t2 = create_tensor<2, 3>();
      Core::LinAlg::Tensor<double, 2, 3> t2_sym_dot_t2 = Core::LinAlg::dot(t2_sym, t2);
      EXPECT_NEAR(t2_sym_dot_t2(0, 0), 5.5, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2(0, 1), 12.1, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2(0, 2), 18.7, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2(1, 0), 9, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2(1, 1), 20.2, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2(1, 2), 31.4, 1e-10);
    }

    {
      Core::LinAlg::Tensor<double, 3, 2> t2 = create_tensor<3, 2>();
      Core::LinAlg::Tensor<double, 3, 2> t2_dot_t2_sym = Core::LinAlg::dot(t2, t2_sym);
      EXPECT_NEAR(t2_dot_t2_sym(0, 0), 9.9, 1e-10);
      EXPECT_NEAR(t2_dot_t2_sym(0, 1), 15.8, 1e-10);
      EXPECT_NEAR(t2_dot_t2_sym(1, 0), 13.2, 1e-10);
      EXPECT_NEAR(t2_dot_t2_sym(1, 1), 21.4, 1e-10);
      EXPECT_NEAR(t2_dot_t2_sym(2, 0), 16.5, 1e-10);
      EXPECT_NEAR(t2_dot_t2_sym(2, 1), 27, 1e-10);
    }

    {
      Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_2 = create_symmetric_tensor<2>(1.1);
      Core::LinAlg::Tensor<double, 2, 2> t2_sym_dot_t2_sym_2 = Core::LinAlg::dot(t2_sym, t2_sym_2);
      EXPECT_NEAR(t2_sym_dot_t2_sym_2(0, 0), 6.38, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2_sym_2(0, 1), 10.23, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2_sym_2(1, 0), 10.46, 1e-10);
      EXPECT_NEAR(t2_sym_dot_t2_sym_2(1, 1), 16.96, 1e-10);
    }
  }

  TEST(SymmetricTensorOperationsTest, ddot)
  {
    {
      Core::LinAlg::SymmetricTensor<double, 2, 2, 2, 2> t4_sym = create_symmetric_four_tensor<2>();
      Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
      Core::LinAlg::SymmetricTensor<double, 2, 2> t4_sym_ddot_t2_sym =
          Core::LinAlg::ddot(t4_sym, t2_sym);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(0, 0), 9.824, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(0, 1), 19.614, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(1, 0), 19.614, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(1, 1), 30.383, 1e-10);
    }

    {
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> t4_sym =
          assume_symmetry(Core::LinAlg::Tensor<double, 3, 3, 3, 3>{{
              {
                  {{0.771321, 0.0207519, 0.633648}, {0.0207519, 0.498507, 0.224797},
                      {0.633648, 0.224797, 0.169111}},
                  {{0.0883398, 0.68536, 0.953393}, {0.68536, 0.512192, 0.812621},
                      {0.953393, 0.812621, 0.291876}},
                  {{0.917774, 0.714576, 0.542544}, {0.714576, 0.373341, 0.674134},
                      {0.542544, 0.674134, 0.617767}},
              },
              {
                  {{0.0883398, 0.68536, 0.953393}, {0.68536, 0.512192, 0.812621},
                      {0.953393, 0.812621, 0.291876}},
                  {{0.113984, 0.828681, 0.0468963}, {0.828681, 0.547586, 0.819287},
                      {0.0468963, 0.819287, 0.351653}},
                  {{0.754648, 0.295962, 0.883936}, {0.295962, 0.165016, 0.392529},
                      {0.883936, 0.392529, 0.151152}},
              },
              {
                  {{0.917774, 0.714576, 0.542544}, {0.714576, 0.373341, 0.674134},
                      {0.542544, 0.674134, 0.617767}},
                  {{0.754648, 0.295962, 0.883936}, {0.295962, 0.165016, 0.392529},
                      {0.883936, 0.392529, 0.151152}},
                  {{0.314927, 0.636491, 0.346347}, {0.636491, 0.879915, 0.763241},
                      {0.346347, 0.763241, 0.605578}},
              },
          }});

      Core::LinAlg::SymmetricTensor<double, 3, 3> t2_sym = Core::LinAlg::assume_symmetry(
          Core::LinAlg::Tensor<double, 3, 3>{{{0.513467, 0.597837, 0.262216},
              {0.597837, 0.0253998, 0.303063}, {0.262216, 0.303063, 0.565507}}});

      Core::LinAlg::SymmetricTensor<double, 3, 3> t4_sym_ddot_t2_sym =
          Core::LinAlg::ddot(t4_sym, t2_sym);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(0, 0), 0.9977164139212, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(0, 1), 2.0354347142422, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(0, 2), 2.3776185361748, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(1, 0), 2.0354347142422, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(1, 1), 1.7833152290394, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(1, 2), 1.5325161574708, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(2, 0), 2.3776185361748, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(2, 1), 1.5325161574708, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2_sym(2, 2), 1.931804289176, 1e-10);



      Core::LinAlg::SymmetricTensor<double, 3, 3> t2_sym_ddot_t4_sym =
          Core::LinAlg::ddot(t2_sym, t4_sym);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(0, 0), 1.6213839037404, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(0, 1), 1.7652477801221, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(0, 2), 2.48266139601174, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(1, 0), 1.7652477801221, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(1, 1), 1.6757006732928, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(1, 2), 2.1309429714246, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(2, 0), 2.48266139601174, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(2, 1), 2.1309429714246, 1e-10);
      EXPECT_NEAR(t2_sym_ddot_t4_sym(2, 2), 1.2028059166724, 1e-10);
    }

    {
      Core::LinAlg::SymmetricTensor<double, 2, 2, 2, 2> t4_sym = create_symmetric_four_tensor<2>();
      Core::LinAlg::Tensor<double, 2, 2> t2 = create_tensor<2, 2>();
      Core::LinAlg::SymmetricTensor<double, 2, 2> t4_sym_ddot_t2 = Core::LinAlg::ddot(t4_sym, t2);
      EXPECT_NEAR(t4_sym_ddot_t2(0, 0), 11.04, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2(0, 1), 22.04, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2(1, 0), 22.04, 1e-10);
      EXPECT_NEAR(t4_sym_ddot_t2(1, 1), 34.14, 1e-10);
    }
  }

  TEST(SymmetricTensorOperationsTest, scale)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_inv = Core::LinAlg::scale(t2_sym, 2.0);

    EXPECT_NEAR(t2_sym_inv(0, 0), 2.2, 1e-10);
    EXPECT_NEAR(t2_sym_inv(0, 1), 4.4, 1e-10);

    EXPECT_NEAR(t2_sym_inv(1, 0), 4.4, 1e-10);
    EXPECT_NEAR(t2_sym_inv(1, 1), 6.8, 1e-10);


    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym = create_symmetric_tensor<3>();
    Core::LinAlg::SymmetricTensor<double, 3, 3> t_sym_inv = Core::LinAlg::scale(t_sym, 2.0);

    EXPECT_NEAR(t_sym_inv(0, 0), 2.2, 1e-10);
    EXPECT_NEAR(t_sym_inv(0, 1), 4.4, 1e-10);
    EXPECT_NEAR(t_sym_inv(0, 2), 10.6, 1e-10);

    EXPECT_NEAR(t_sym_inv(1, 0), 4.4, 1e-10);
    EXPECT_NEAR(t_sym_inv(1, 1), 6.8, 1e-10);
    EXPECT_NEAR(t_sym_inv(1, 2), 13.2, 1e-10);

    EXPECT_NEAR(t_sym_inv(2, 0), 10.6, 1e-10);
    EXPECT_NEAR(t_sym_inv(2, 1), 13.2, 1e-10);
    EXPECT_NEAR(t_sym_inv(2, 2), 19.8, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, add)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_2 = create_symmetric_tensor<2>(1.1);
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_addition =
        Core::LinAlg::add(t2_sym, t2_sym_2);

    EXPECT_NEAR(t2_sym_addition(0, 0), 2.3, 1e-10);
    EXPECT_NEAR(t2_sym_addition(0, 1), 4.5, 1e-10);

    EXPECT_NEAR(t2_sym_addition(1, 0), 4.5, 1e-10);
    EXPECT_NEAR(t2_sym_addition(1, 1), 6.9, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, subtract)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_2 = create_symmetric_tensor<2>(1.1, 0.2);
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_subtraction =
        Core::LinAlg::subtract(t2_sym, t2_sym_2);

    EXPECT_NEAR(t2_sym_subtraction(0, 0), -0.2, 1e-10);
    EXPECT_NEAR(t2_sym_subtraction(0, 1), -0.3, 1e-10);

    EXPECT_NEAR(t2_sym_subtraction(1, 0), -0.3, 1e-10);
    EXPECT_NEAR(t2_sym_subtraction(1, 1), -0.5, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, dyadic)
  {
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym = create_symmetric_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t2_sym_2 = create_symmetric_tensor<2>(1.1, 0.2);
    Core::LinAlg::SymmetricTensor<double, 2, 2, 2, 2> t2_sym_dyad =
        Core::LinAlg::dyadic(t2_sym, t2_sym_2);

    EXPECT_NEAR(t2_sym_dyad(0, 0, 0, 0), 1.43, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(0, 0, 0, 1), 2.75, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(0, 0, 1, 0), 2.75, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(0, 0, 1, 1), 4.29, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(0, 1, 0, 0), 2.86, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(0, 1, 0, 1), 5.5, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(0, 1, 1, 0), 5.5, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(0, 1, 1, 1), 8.58, 1e-10);


    EXPECT_NEAR(t2_sym_dyad(1, 0, 0, 0), 2.86, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(1, 0, 0, 1), 5.5, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(1, 0, 1, 0), 5.5, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(1, 0, 1, 1), 8.58, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(1, 1, 0, 0), 4.42, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(1, 1, 0, 1), 8.5, 1e-10);

    EXPECT_NEAR(t2_sym_dyad(1, 1, 1, 0), 8.5, 1e-10);
    EXPECT_NEAR(t2_sym_dyad(1, 1, 1, 1), 13.26, 1e-10);
  }

  TEST(SymmetricTensorOperationsTest, self_dyadic)
  {
    Core::LinAlg::Tensor<double, 2> t1 = create_tensor<2>();
    Core::LinAlg::SymmetricTensor<double, 2, 2> t1_self_dyad = Core::LinAlg::self_dyadic(t1);

    std::cout << t1 << std::endl;

    EXPECT_NEAR(t1_self_dyad(0, 0), 1, 1e-10);
    EXPECT_NEAR(t1_self_dyad(0, 1), 2, 1e-10);

    EXPECT_NEAR(t1_self_dyad(1, 0), 2, 1e-10);
    EXPECT_NEAR(t1_self_dyad(1, 1), 4, 1e-10);


    Core::LinAlg::SymmetricTensor<double, 2, 2, 2, 2> t1_self_dyad4 =
        Core::LinAlg::self_dyadic<4>(t1);

    EXPECT_NEAR(t1_self_dyad4(0, 0, 0, 0), 1, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(0, 0, 0, 1), 2, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(0, 0, 1, 0), 2, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(0, 0, 1, 1), 4, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(0, 1, 0, 0), 2, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(0, 1, 0, 1), 4, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(0, 1, 1, 0), 4, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(0, 1, 1, 1), 8, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(1, 0, 0, 0), 2, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(1, 0, 0, 1), 4, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(1, 0, 1, 0), 4, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(1, 0, 1, 1), 8, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(1, 1, 0, 0), 4, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(1, 1, 0, 1), 8, 1e-10);

    EXPECT_NEAR(t1_self_dyad4(1, 1, 1, 0), 8, 1e-10);
    EXPECT_NEAR(t1_self_dyad4(1, 1, 1, 1), 16, 1e-10);
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE