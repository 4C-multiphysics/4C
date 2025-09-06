// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_utils.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_solid_3D_ele_properties.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN


Core::LinAlg::SymmetricTensor<double, 3, 3> Solid::Utils::pk2_to_cauchy(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& pk2,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  return Core::LinAlg::assume_symmetry(defgrd * pk2 * Core::LinAlg::transpose(defgrd)) /
         Core::LinAlg::det(defgrd);
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Solid::Utils::green_lagrange_to_euler_almansi(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  Core::LinAlg::Tensor<double, 3, 3> invdefgrd = Core::LinAlg::inv(defgrd);

  return Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(invdefgrd) * gl * invdefgrd);
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Solid::Utils::green_lagrange_to_log_strain(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl)
{
  auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(gl);

  // compute principal logarithmic strains
  std::ranges::for_each(
      eigenvalues, [](double& value) { value = std::log(std::sqrt(2 * value + 1.0)); });

  const auto eig = Core::LinAlg::TensorGenerators::diagonal(eigenvalues);
  return Core::LinAlg::assume_symmetry(eigenvectors * eig * Core::LinAlg::transpose(eigenvectors));
}

int Solid::Utils::ReadElement::read_element_material(
    const Core::IO::InputParameterContainer& container)
{
  int material = container.get<int>("MAT");
  return material;
}


Discret::Elements::SolidElementProperties Solid::Utils::ReadElement::read_solid_element_properties(
    const Core::IO::InputParameterContainer& container)
{
  Discret::Elements::SolidElementProperties solid_properties{};

  // element technology
  solid_properties.element_technology = container.get_or<Discret::Elements::ElementTechnology>(
      "TECH", Discret::Elements::ElementTechnology::none);

  // prestress technology
  solid_properties.prestress_technology = container.get_or<Discret::Elements::PrestressTechnology>(
      "PRESTRESS_TECH", Discret::Elements::PrestressTechnology::none);

  // kinematic type
  solid_properties.kintype = container.get<Inpar::Solid::KinemType>("KINEM");

  return solid_properties;
}

void Solid::Utils::nodal_block_information_solid(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 6;
}

FOUR_C_NAMESPACE_CLOSE
