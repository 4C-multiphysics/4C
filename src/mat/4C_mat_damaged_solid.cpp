// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_damaged_solid.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_mat_material_factory.hpp"

#include <algorithm>
#include <limits>

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::shared_ptr<Mat::So3Material> create_wrapped_material(const int matid)
  {
    auto material = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(matid));
    FOUR_C_ASSERT_ALWAYS(material != nullptr,
        "The material {} wrapped by the damage material is no solid material.", matid);
    return material;
  }
}  // namespace

Mat::DamagedSolid::DamagedSolid(const int matid) : wrapped_material_(create_wrapped_material(matid))
{
}

Mat::DamagedSolid::DamagedSolid(const DamagedSolid& other)
    : wrapped_material_(other.wrapped_material_ == nullptr ? nullptr
                                                           : std::dynamic_pointer_cast<So3Material>(
                                                                 other.wrapped_material_->clone()))
{
}

Mat::DamagedSolid& Mat::DamagedSolid::operator=(const DamagedSolid& other)
{
  if (this != &other)
  {
    wrapped_material_ = other.wrapped_material_ == nullptr ? nullptr
                                                           : std::dynamic_pointer_cast<So3Material>(
                                                                 other.wrapped_material_->clone());
  }
  return *this;
}

void Mat::DamagedSolid::pack_wrapped_material(Core::Communication::PackBuffer& data) const
{
  wrapped_material_->pack(data);
}

void Mat::DamagedSolid::unpack_wrapped_material(
    Core::Communication::UnpackBuffer& buffer, const int matid)
{
  wrapped_material_ = create_wrapped_material(matid);
  wrapped_material_->unpack(buffer);
}

double Mat::DamagedSolid::strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext<3>& context, const int gp, const int eleGID) const
{
  return (1.0 - damage(gp)) * wrapped_material_->strain_energy(glstrain, context, gp, eleGID);
}

void Mat::DamagedSolid::setup(const int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  wrapped_material_->setup(numgp, fibers, coord_system);
  setup_damage(numgp);
}

void Mat::DamagedSolid::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  wrapped_material_->post_setup(params, eleGID);
}

void Mat::DamagedSolid::update()
{
  wrapped_material_->update();
  update_damage();
}

void Mat::DamagedSolid::update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, const int gp,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, const int eleGID)
{
  wrapped_material_->update(defgrd, gp, params, context, eleGID);
}

void Mat::DamagedSolid::reset_step()
{
  wrapped_material_->reset_step();
  reset_damage();
}

void Mat::DamagedSolid::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  wrapped_material_->register_output_data_names(names_and_size);
  names_and_size["damage"] = 1;
}

bool Mat::DamagedSolid::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "damage")
  {
    for (int gp = 0; gp < data.numRows(); ++gp) data(gp, 0) = damage(gp);
    return true;
  }
  return wrapped_material_->evaluate_output_data(name, data);
}

Mat::Trait::ThermoSolid& Mat::DamagedSolid::wrapped_thermo_solid_material() const
{
  auto* thermo_solid_material = dynamic_cast<Trait::ThermoSolid*>(wrapped_material_.get());
  FOUR_C_ASSERT_ALWAYS(thermo_solid_material != nullptr,
      "The material wrapped by the damage material is no thermo-solid material.");
  return *thermo_solid_material;
}

void Mat::DamagedSolid::reinit(const double temperature, const unsigned gp)
{
  wrapped_thermo_solid_material().reinit(temperature, gp);
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::DamagedSolid::evaluate_d_stress_d_scalar(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, const int gp,
    const int eleGID)
{
  // the damage only depends on the strain
  return (1.0 - damage(gp)) * wrapped_thermo_solid_material().evaluate_d_stress_d_scalar(
                                  defgrad, glstrain, params, context, gp, eleGID);
}

std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
Mat::DamagedSolid::evaluate_d_stress_d_scalars(const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    const int num_scalars, const int gp, const int eleGID)
{
  std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> d_stress_d_scalars =
      wrapped_thermo_solid_material().evaluate_d_stress_d_scalars(
          defgrad, glstrain, params, context, num_scalars, gp, eleGID);
  for (auto& d_stress_d_scalar : d_stress_d_scalars) d_stress_d_scalar *= 1.0 - damage(gp);
  return d_stress_d_scalars;
}

void Mat::DamagedSolid::stress_temperature_modulus_and_deriv(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, const int gp)
{
  wrapped_thermo_solid_material().stress_temperature_modulus_and_deriv(stm, stm_dT, gp);
  stm *= 1.0 - damage(gp);
  stm_dT *= 1.0 - damage(gp);
}

double Mat::DamageUtils::evaluate_extent(
    std::span<const double> nodal_coordinates, const Core::LinAlg::Tensor<double, 3>& direction)
{
  FOUR_C_ASSERT_ALWAYS(!nodal_coordinates.empty() && nodal_coordinates.size() % 3 == 0,
      "The damage material needs the reference coordinates of the element nodes.");
  double min_projection = std::numeric_limits<double>::max();
  double max_projection = std::numeric_limits<double>::lowest();
  for (std::size_t i = 0; i < nodal_coordinates.size(); i += 3)
  {
    const double projection = nodal_coordinates[i] * direction(0) +
                              nodal_coordinates[i + 1] * direction(1) +
                              nodal_coordinates[i + 2] * direction(2);
    min_projection = std::min(min_projection, projection);
    max_projection = std::max(max_projection, projection);
  }
  return max_projection - min_projection;
}

Mat::DamageUtils::MaxPrincipalValue Mat::DamageUtils::evaluate_max_principal_value(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& tensor)
{
  // the eigenvalues are sorted in ascending order, the eigenvectors are the columns
  const auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(tensor);
  Core::LinAlg::Tensor<double, 3> direction;
  for (int i = 0; i < 3; ++i) direction(i) = eigenvectors(i, 2);
  return {.value = eigenvalues[2], .direction = direction};
}

FOUR_C_NAMESPACE_CLOSE
