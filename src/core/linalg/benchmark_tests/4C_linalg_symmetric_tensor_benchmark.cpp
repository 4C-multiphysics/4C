// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_symmetric_tensor.hpp"

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
static void symmetric_tensor_det(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    double result = Core::LinAlg::det(A_sym);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_tensor_det<2>);
BENCHMARK(symmetric_tensor_det<3>);
BENCHMARK(symmetric_tensor_det<4>);

template <std::size_t size>
static void symmetric_tensor_inv(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    auto result = Core::LinAlg::inv(A_sym);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_tensor_inv<2>);
BENCHMARK(symmetric_tensor_inv<3>);

template <std::size_t size>
static void symmetric_matrix_dot_vector(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  Core::LinAlg::Tensor<double, size> b = get_random_tensor<size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size> result = A_sym * b;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_matrix_dot_vector<2>);
BENCHMARK(symmetric_matrix_dot_vector<3>);
BENCHMARK(symmetric_matrix_dot_vector<4>);
BENCHMARK(symmetric_matrix_dot_vector<5>);
BENCHMARK(symmetric_matrix_dot_vector<6>);
BENCHMARK(symmetric_matrix_dot_vector<9>);

template <std::size_t size>
static void symmetric_matrix_dot_symmetric_matrix(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  Core::LinAlg::SymmetricTensor<double, size, size> B_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = A_sym * B_sym;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_matrix_dot_symmetric_matrix<2>);
BENCHMARK(symmetric_matrix_dot_symmetric_matrix<3>);
BENCHMARK(symmetric_matrix_dot_symmetric_matrix<4>);
BENCHMARK(symmetric_matrix_dot_symmetric_matrix<5>);
BENCHMARK(symmetric_matrix_dot_symmetric_matrix<6>);

template <std::size_t size>
static void symmetric_matrix_ddot_symmetric_matrix(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  Core::LinAlg::SymmetricTensor<double, size, size> B_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    double result = Core::LinAlg::ddot(A_sym, B_sym);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_matrix_ddot_symmetric_matrix<2>);
BENCHMARK(symmetric_matrix_ddot_symmetric_matrix<3>);
BENCHMARK(symmetric_matrix_ddot_symmetric_matrix<4>);

template <std::size_t size>
static void symmetric_four_tensor_ddot_symmetric_matrix(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size, size, size> A_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size, size, size>());
  Core::LinAlg::SymmetricTensor<double, size, size> B_sym =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    Core::LinAlg::SymmetricTensor<double, size, size> result = Core::LinAlg::ddot(A_sym, B_sym);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_four_tensor_ddot_symmetric_matrix<2>);
BENCHMARK(symmetric_four_tensor_ddot_symmetric_matrix<3>);
BENCHMARK(symmetric_four_tensor_ddot_symmetric_matrix<4>);
BENCHMARK(symmetric_four_tensor_ddot_symmetric_matrix<5>);
BENCHMARK(symmetric_four_tensor_ddot_symmetric_matrix<6>);

template <std::size_t size>
static void self_dyadic_symmetric(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size> a = get_random_tensor<size>();
  for (auto _ : state)
  {
    Core::LinAlg::SymmetricTensor<double, size, size> result = Core::LinAlg::self_dyadic(a);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(self_dyadic_symmetric<2>);
BENCHMARK(self_dyadic_symmetric<3>);
BENCHMARK(self_dyadic_symmetric<4>);
BENCHMARK(self_dyadic_symmetric<5>);
BENCHMARK(self_dyadic_symmetric<6>);

template <std::size_t size>
static void self_dyadic(benchmark::State& state)
{
  Core::LinAlg::Tensor<double, size> a = get_random_tensor<size>();
  for (auto _ : state)
  {
    Core::LinAlg::Tensor<double, size, size> result = Core::LinAlg::dyadic(a, a);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(self_dyadic<2>);
BENCHMARK(self_dyadic<3>);
BENCHMARK(self_dyadic<4>);
BENCHMARK(self_dyadic<5>);
BENCHMARK(self_dyadic<6>);

template <std::size_t size>
static void symmetric_matrix_plus_symmetric_matrix(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> a =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  Core::LinAlg::SymmetricTensor<double, size, size> b =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    Core::LinAlg::SymmetricTensor<double, size, size> result = a + b;
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_matrix_plus_symmetric_matrix<2>);
BENCHMARK(symmetric_matrix_plus_symmetric_matrix<3>);
BENCHMARK(symmetric_matrix_plus_symmetric_matrix<4>);
BENCHMARK(symmetric_matrix_plus_symmetric_matrix<5>);
BENCHMARK(symmetric_matrix_plus_symmetric_matrix<6>);

template <std::size_t size>
static void symmetric_matrix_dyad_symmetric_matrix(benchmark::State& state)
{
  Core::LinAlg::SymmetricTensor<double, size, size> a =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  Core::LinAlg::SymmetricTensor<double, size, size> b =
      Core::LinAlg::assume_symmetry(get_random_tensor<size, size>());
  for (auto _ : state)
  {
    Core::LinAlg::SymmetricTensor<double, size, size, size, size> result =
        Core::LinAlg::dyadic(a, b);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(symmetric_matrix_dyad_symmetric_matrix<2>);
BENCHMARK(symmetric_matrix_dyad_symmetric_matrix<3>);
BENCHMARK(symmetric_matrix_dyad_symmetric_matrix<4>);
BENCHMARK(symmetric_matrix_dyad_symmetric_matrix<5>);
BENCHMARK(symmetric_matrix_dyad_symmetric_matrix<6>);

FOUR_C_NAMESPACE_CLOSE