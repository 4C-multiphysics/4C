// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_tensor.hpp"

#include <benchmark/benchmark.h>

FOUR_C_NAMESPACE_OPEN
template <std::size_t... size>
Core::LinAlg::Tensor<double, size...> get_random_tensor()
{
  Core::LinAlg::Tensor<double, size...> tensor;
  std::generate(tensor.container().begin(), tensor.container().end(),
      []() { return static_cast<double>(std::rand()) / RAND_MAX; });
  return tensor;
}

template <std::size_t size>
static void tensor_det(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    double result = Core::LinAlg::det(A);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(tensor_det<2>);
BENCHMARK(tensor_det<3>);
BENCHMARK(tensor_det<4>);

template <std::size_t size>
static void tensor_inv(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    auto result = Core::LinAlg::inv(A);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(tensor_inv<2>);
BENCHMARK(tensor_inv<3>);


template <std::size_t size>
static void matrix_dot_vector(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size> b = get_random_tensor<size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size> result = A * b;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_dot_vector<2>);
BENCHMARK(matrix_dot_vector<3>);
BENCHMARK(matrix_dot_vector<4>);
BENCHMARK(matrix_dot_vector<5>);
BENCHMARK(matrix_dot_vector<6>);
BENCHMARK(matrix_dot_vector<9>);

template <std::size_t size>
static void matrix_dot_matrix(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> B = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = A * B;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_dot_matrix<2>);
BENCHMARK(matrix_dot_matrix<3>);
BENCHMARK(matrix_dot_matrix<4>);
BENCHMARK(matrix_dot_matrix<5>);
BENCHMARK(matrix_dot_matrix<6>);

template <std::size_t size>
static void matrix_ddot_matrix(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> B = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    double result = Core::LinAlg::ddot(A, B);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_ddot_matrix<2>);
BENCHMARK(matrix_ddot_matrix<3>);
BENCHMARK(matrix_ddot_matrix<4>);

template <std::size_t size>
static void four_tensor_ddot_matrix(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size, size, size> A_sym =
      get_random_tensor<size, size, size, size>();
  Core::LinAlg::Tensor<double, size, size> B_sym = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = Core::LinAlg::ddot(A_sym, B_sym);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(four_tensor_ddot_matrix<2>);
BENCHMARK(four_tensor_ddot_matrix<3>);
BENCHMARK(four_tensor_ddot_matrix<4>);
BENCHMARK(four_tensor_ddot_matrix<5>);
BENCHMARK(four_tensor_ddot_matrix<6>);

template <std::size_t size>
static void fourtensor_ddot_fourtensor(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size, size, size> A =
      get_random_tensor<size, size, size, size>();
  Core::LinAlg::Tensor<double, size, size, size, size> B =
      get_random_tensor<size, size, size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size, size, size> result = Core::LinAlg::ddot(A, B);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(fourtensor_ddot_fourtensor<2>);
BENCHMARK(fourtensor_ddot_fourtensor<3>);
BENCHMARK(fourtensor_ddot_fourtensor<4>);
BENCHMARK(fourtensor_ddot_fourtensor<5>);
BENCHMARK(fourtensor_ddot_fourtensor<6>);


template <std::size_t size>
static void matrix_dot_matrix_t(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> B = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = A * Core::LinAlg::transpose(B);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_dot_matrix_t<2>);
BENCHMARK(matrix_dot_matrix_t<3>);
BENCHMARK(matrix_dot_matrix_t<4>);
BENCHMARK(matrix_dot_matrix_t<5>);
BENCHMARK(matrix_dot_matrix_t<6>);


template <std::size_t size>
static void scaled_matrix_matrix_product(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> A = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> B = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = 3.0 * A * B;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(scaled_matrix_matrix_product<2>);
BENCHMARK(scaled_matrix_matrix_product<3>);
BENCHMARK(scaled_matrix_matrix_product<4>);
BENCHMARK(scaled_matrix_matrix_product<5>);
BENCHMARK(scaled_matrix_matrix_product<6>);

template <std::size_t size>
static void matrix_plus_matrix(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> a = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> b = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = a + b;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_plus_matrix<2>);
BENCHMARK(matrix_plus_matrix<3>);
BENCHMARK(matrix_plus_matrix<4>);
BENCHMARK(matrix_plus_matrix<5>);
BENCHMARK(matrix_plus_matrix<6>);

template <std::size_t size>
static void matrix_dyad_matrix(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size, size> a = get_random_tensor<size, size>();
  Core::LinAlg::Tensor<double, size, size> b = get_random_tensor<size, size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size, size, size> result = Core::LinAlg::dyadic(a, b);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(matrix_dyad_matrix<2>);
BENCHMARK(matrix_dyad_matrix<3>);
BENCHMARK(matrix_dyad_matrix<4>);
BENCHMARK(matrix_dyad_matrix<5>);
BENCHMARK(matrix_dyad_matrix<6>);

FOUR_C_NAMESPACE_CLOSE