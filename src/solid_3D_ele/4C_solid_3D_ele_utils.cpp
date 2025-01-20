// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_utils.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_solid_3D_ele_properties.hpp"

FOUR_C_NAMESPACE_OPEN


void Solid::Utils::pk2_to_cauchy(const Core::LinAlg::Matrix<6, 1>& pk2,
    const Core::LinAlg::Matrix<3, 3>& defgrd, Core::LinAlg::Matrix<6, 1>& cauchy)
{
  Core::LinAlg::Matrix<3, 3> S_matrix;
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(pk2, S_matrix);

  Core::LinAlg::Matrix<3, 3> FS;
  FS.multiply_nn(defgrd, S_matrix);

  Core::LinAlg::Matrix<3, 3> cauchy_matrix;
  cauchy_matrix.multiply_nt(1.0 / defgrd.determinant(), FS, defgrd, 0.0);

  Core::LinAlg::Voigt::Stresses::matrix_to_vector(cauchy_matrix, cauchy);
}

Core::LinAlg::Matrix<6, 1> Solid::Utils::green_lagrange_to_euler_almansi(
    const Core::LinAlg::Matrix<6, 1>& gl, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  Core::LinAlg::Matrix<3, 3> invdefgrd(defgrd);
  invdefgrd.invert();

  Core::LinAlg::Matrix<3, 3> E_matrix;
  Core::LinAlg::Voigt::Strains::vector_to_matrix(gl, E_matrix);

  Core::LinAlg::Matrix<3, 3> iFTE;
  iFTE.multiply_tn(invdefgrd, E_matrix);

  Core::LinAlg::Matrix<3, 3> ea_matrix;
  ea_matrix.multiply_nn(iFTE, invdefgrd);

  Core::LinAlg::Matrix<6, 1> ea;
  Core::LinAlg::Voigt::Strains::matrix_to_vector(ea_matrix, ea);
  return ea;
}

Core::LinAlg::Matrix<6, 1> Solid::Utils::green_lagrange_to_log_strain(
    const Core::LinAlg::Matrix<6, 1>& gl)
{
  Core::LinAlg::Matrix<3, 3> E_matrix;
  Core::LinAlg::Voigt::Strains::vector_to_matrix(gl, E_matrix);

  Core::LinAlg::Matrix<3, 3> pr_strain(true);  // squared principal strains
  Core::LinAlg::Matrix<3, 3> pr_dir(true);     // principal directions
  Core::LinAlg::syev(E_matrix, pr_strain, pr_dir);

  // compute principal logarithmic strains
  Core::LinAlg::Matrix<3, 3> pr_log_strain(true);
  for (int i = 0; i < 3; ++i) pr_log_strain(i, i) = std::log(std::sqrt(2 * pr_strain(i, i) + 1.0));

  // create logarithmic strain matrix
  Core::LinAlg::Matrix<3, 3> log_strain_matrix(true);
  Core::LinAlg::Matrix<3, 3> VH(false);
  VH.multiply_nn(pr_dir, pr_log_strain);
  log_strain_matrix.multiply_nt(VH, pr_dir);

  // convert to strain-like voigt notation
  Core::LinAlg::Matrix<6, 1> log_strain_voigt(true);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(log_strain_matrix, log_strain_voigt);
  return log_strain_voigt;
}

int Solid::Utils::ReadElement::read_element_material(
    const Core::IO::InputParameterContainer& container)
{
  int material = container.get<int>("MAT");
  return material;
}

Inpar::Solid::KinemType Solid::Utils::ReadElement::read_element_kinematic_type(
    const Core::IO::InputParameterContainer& container)
{
  auto kinem = container.get<std::string>("KINEM");

  if (kinem == "nonlinear")
    return Inpar::Solid::KinemType::nonlinearTotLag;
  else if (kinem == "linear")
    return Inpar::Solid::KinemType::linear;
  else
  {
    FOUR_C_THROW("unknown kinematic type %s", kinem.c_str());
    return Inpar::Solid::KinemType::vague;
  }
}

Discret::Elements::ElementTechnology Solid::Utils::ReadElement::read_element_technology(
    const Core::IO::InputParameterContainer& container)
{
  auto type = container.get_or<std::string>("TECH", "none");

  if (type == "fbar")
  {
    return Discret::Elements::ElementTechnology::fbar;
  }
  else if (type == "eas_full")
  {
    return Discret::Elements::ElementTechnology::eas_full;
  }
  else if (type == "eas_mild")
  {
    return Discret::Elements::ElementTechnology::eas_mild;
  }
  else if (type == "shell_ans")
  {
    return Discret::Elements::ElementTechnology::shell_ans;
  }
  else if (type == "shell_eas")
  {
    return Discret::Elements::ElementTechnology::shell_eas;
  }
  else if (type == "shell_eas_ans")
  {
    return Discret::Elements::ElementTechnology::shell_eas_ans;
  }
  else if (type == "none")
  {
    return Discret::Elements::ElementTechnology::none;
  }
  else
    FOUR_C_THROW("unrecognized element technology type %s", type.c_str());
}

Discret::Elements::PrestressTechnology Solid::Utils::ReadElement::read_prestress_technology(
    const Core::IO::InputParameterContainer& container)
{
  auto type = container.get_or<std::string>("PRESTRESS_TECH", "none");

  if (type == "none")
  {
    return Discret::Elements::PrestressTechnology::none;
  }
  else if (type == "mulf")
  {
    return Discret::Elements::PrestressTechnology::mulf;
  }

  FOUR_C_THROW("unrecognized prestress technology type %s", type.c_str());
}

Discret::Elements::SolidElementProperties Solid::Utils::ReadElement::read_solid_element_properties(
    const Core::IO::InputParameterContainer& container)
{
  Discret::Elements::SolidElementProperties solid_properties{};

  // element technology
  solid_properties.element_technology =
      Solid::Utils::ReadElement::read_element_technology(container);

  // prestress technology
  solid_properties.prestress_technology =
      Solid::Utils::ReadElement::read_prestress_technology(container);

  // kinematic type
  solid_properties.kintype = Solid::Utils::ReadElement::read_element_kinematic_type(container);

  return solid_properties;
}

void Solid::Utils::nodal_block_information_solid(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;

  nv = 3;
}

FOUR_C_NAMESPACE_CLOSE
