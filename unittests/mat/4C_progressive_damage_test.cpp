// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_progressive_damage.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_ParameterList.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>

namespace
{
  using namespace FourC;

  constexpr int wrapped_matid = 1;
  constexpr int damage_matid = 2;

  struct StressAndTangent
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress;
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat;
  };

  class ProgressiveDamageTest : public ::testing::Test
  {
   protected:
    //! Add the wrapped material and the damage material to the global problem and create the latter
    std::shared_ptr<Mat::ProgressiveDamage> create_material(const double poisson_ratio,
        const double thermal_expansion, const double fracture_energy,
        const Core::Materials::MaterialType wrapped_type = Core::Materials::m_thermostvenant)
    {
      Global::Problem& problem = *Global::Problem::instance();
      problem.materials()->set_read_from_problem(0);

      Core::IO::InputParameterContainer wrapped_data;
      if (wrapped_type == Core::Materials::m_thermostvenant)
      {
        wrapped_data.add("YOUNGNUM", 1);
        wrapped_data.add("YOUNG", std::vector<double>{young_});
        wrapped_data.add("THEXPANS", thermal_expansion);
        wrapped_data.add("INITTEMP", reference_temperature_);
      }
      else
      {
        wrapped_data.add("YOUNG", young_);
      }
      wrapped_data.add("NUE", poisson_ratio);
      wrapped_data.add("DENS", 1.0);
      problem.materials()->insert(
          wrapped_matid, Mat::make_parameter(wrapped_matid, wrapped_type, wrapped_data));

      Core::IO::InputParameterContainer damage_data;
      damage_data.add("MATID", wrapped_matid);
      damage_data.add("TENSILE_STRENGTH", tensile_strength_);
      damage_data.add("FRACTURE_ENERGY", fracture_energy);
      damage_data.add("RESIDUAL_STRENGTH_RATIO", residual_strength_ratio_);
      problem.materials()->insert(damage_matid,
          Mat::make_parameter(damage_matid, Core::Materials::m_progressive_damage, damage_data));

      auto material = std::dynamic_pointer_cast<Mat::ProgressiveDamage>(Mat::factory(damage_matid));
      material->setup(1, {}, {});
      return material;
    }

    [[nodiscard]] StressAndTangent evaluate(Mat::So3Material& material,
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const double temperature) const
    {
      Teuchos::ParameterList params;
      params.set("temperature", temperature);
      StressAndTangent result{};
      material.evaluate(&defgrad, glstrain, params, context_, result.stress, result.cmat, 0, 0);
      return result;
    }

    [[nodiscard]] StressAndTangent evaluate(
        Mat::So3Material& material, const Core::LinAlg::Tensor<double, 3, 3>& defgrad) const
    {
      return evaluate(material, defgrad, green_lagrange_strain(defgrad), reference_temperature_);
    }

    static Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_strain(
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad)
    {
      return 0.5 * (Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrad) * defgrad) -
                       Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    }

    //! A deformation gradient with stretch @p stretch in x-direction and some shear
    static Core::LinAlg::Tensor<double, 3, 3> sheared_defgrad(const double stretch)
    {
      return Core::LinAlg::Tensor<double, 3, 3>{{{stretch, 0.3 * (stretch - 1.0), 0.0},
          {0.0, 1.0, 0.1 * (stretch - 1.0)}, {0.0, 0.0, 1.0}}};
    }

    //! Compare the tangent with central finite differences of the stress w.r.t. the strain
    void expect_tangent_matches_finite_differences(
        Mat::So3Material& material, const Core::LinAlg::Tensor<double, 3, 3>& defgrad) const
    {
      const Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain = green_lagrange_strain(defgrad);
      const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat =
          evaluate(material, defgrad).cmat;

      constexpr double perturbation = 1e-8;
      for (int i = 0; i < 3; ++i)
      {
        for (int j = i; j < 3; ++j)
        {
          Core::LinAlg::SymmetricTensor<double, 3, 3> direction{};
          direction(i, j) = 1.0;

          const auto stress_plus = evaluate(
              material, defgrad, glstrain + perturbation * direction, reference_temperature_)
                                       .stress;
          const auto stress_minus = evaluate(
              material, defgrad, glstrain - perturbation * direction, reference_temperature_)
                                        .stress;
          const Core::LinAlg::SymmetricTensor<double, 3, 3> d_stress =
              (stress_plus - stress_minus) / (2.0 * perturbation);

          FOUR_C_EXPECT_NEAR(Core::LinAlg::ddot(cmat, direction), d_stress, 1e-5 * young_);
        }
      }

      // restore the state of the evaluation at the given deformation
      (void)evaluate(material, defgrad);
    }

    //! Set the reference coordinates of the element nodes to a box with the given extents
    void set_box(const double length_x, const double length_y, const double length_z)
    {
      for (int node = 0; node < 8; ++node)
      {
        nodal_coordinates_[3 * node] = (node & 1) ? length_x : 0.0;
        nodal_coordinates_[3 * node + 1] = (node & 2) ? length_y : 0.0;
        nodal_coordinates_[3 * node + 2] = (node & 4) ? length_z : 0.0;
      }
    }

    //! Stretch uniaxially in @p direction until the material fails and return the work per volume
    double work_until_failure(Mat::ProgressiveDamage& material, const int direction,
        const double fracture_energy, const double length) const
    {
      const double failure_strain = fracture_energy / (tensile_strength_ * length);
      const double strain_increment = 1e-4 * failure_strain;
      double work = 0.0;
      double strain = 0.0;
      double stress = 0.0;
      while (!material.is_failed(0))
      {
        Core::LinAlg::Tensor<double, 3, 3> defgrad =
            Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
        defgrad(direction, direction) = std::sqrt(1.0 + 2.0 * (strain + strain_increment));
        const double next_stress = evaluate(material, defgrad).stress(direction, direction);
        material.update();

        work += 0.5 * (stress + next_stress) * strain_increment;
        stress = next_stress;
        strain += strain_increment;
        if (strain > 20.0 * failure_strain) return std::numeric_limits<double>::infinity();
      }
      EXPECT_LT(stress, 2.0 * residual_strength_ratio_ * tensile_strength_);
      return work;
    }

    const double young_ = 1.0e4;
    const double tensile_strength_ = 1.0;
    const double residual_strength_ratio_ = 1e-3;
    const double reference_temperature_ = 293.0;

    //! reference coordinates of the nodes of a box-shaped element
    std::array<double, 24> nodal_coordinates_ = []
    {
      std::array<double, 24> coordinates{};
      for (int node = 0; node < 8; ++node)
      {
        coordinates[3 * node] = (node & 1) ? 0.5 : 0.0;
        coordinates[3 * node + 1] = (node & 2) ? 1.0 : 0.0;
        coordinates[3 * node + 2] = (node & 4) ? 1.0 : 0.0;
      }
      return coordinates;
    }();

    const Mat::EvaluationContext<3> context_{.total_time = nullptr,
        .time_step_size = nullptr,
        .xi = nullptr,
        .ref_coords = nullptr,
        .reference_nodal_coordinates = nodal_coordinates_};

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  TEST_F(ProgressiveDamageTest, NoDamageBelowTheTensileStrength)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);
    auto wrapped_material =
        std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(wrapped_matid));

    const auto defgrad = sheared_defgrad(1.00005);
    const StressAndTangent damaged = evaluate(*material, defgrad);
    const StressAndTangent undamaged = evaluate(*wrapped_material, defgrad);

    EXPECT_EQ(material->damage(0), 0.0);
    EXPECT_FALSE(material->is_failed(0));
    FOUR_C_EXPECT_NEAR(damaged.stress, undamaged.stress, 1e-12);
    FOUR_C_EXPECT_NEAR(damaged.cmat, undamaged.cmat, 1e-12);
  }

  TEST_F(ProgressiveDamageTest, TangentDuringSofteningMatchesFiniteDifferences)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);

    // initiate the damage
    (void)evaluate(*material, sheared_defgrad(1.0002));
    material->update();
    ASSERT_EQ(material->damage(0), 0.0);

    const auto defgrad = sheared_defgrad(1.0004);
    (void)evaluate(*material, defgrad);
    EXPECT_GT(material->damage(0), 0.1);
    EXPECT_LT(material->damage(0), 0.9);

    expect_tangent_matches_finite_differences(*material, defgrad);
  }

  TEST_F(ProgressiveDamageTest, UnloadingKeepsTheDamage)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);
    auto wrapped_material =
        std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(wrapped_matid));

    (void)evaluate(*material, sheared_defgrad(1.0002));
    material->update();
    (void)evaluate(*material, sheared_defgrad(1.0004));
    material->update();
    const double damage = material->damage(0);

    const auto defgrad = sheared_defgrad(1.0003);
    const StressAndTangent damaged = evaluate(*material, defgrad);
    const StressAndTangent undamaged = evaluate(*wrapped_material, defgrad);

    EXPECT_EQ(material->damage(0), damage);
    FOUR_C_EXPECT_NEAR(damaged.stress, (1.0 - damage) * undamaged.stress, 1e-12);
    FOUR_C_EXPECT_NEAR(damaged.cmat, (1.0 - damage) * undamaged.cmat, 1e-9);
    expect_tangent_matches_finite_differences(*material, defgrad);
  }

  TEST_F(ProgressiveDamageTest, FailsAtTheResidualStrength)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);

    (void)evaluate(*material, sheared_defgrad(1.0002));
    material->update();
    (void)evaluate(*material, sheared_defgrad(1.5));

    // the remaining stiffness is below the residual strength ratio
    const double damage = material->damage(0);
    EXPECT_TRUE(material->is_failed(0));
    EXPECT_GT(damage, 1.0 - residual_strength_ratio_);
    EXPECT_LT(damage, 1.0);

    // the damage does not grow anymore
    material->update();
    const auto defgrad = sheared_defgrad(2.0);
    (void)evaluate(*material, defgrad);
    EXPECT_EQ(material->damage(0), damage);
    expect_tangent_matches_finite_differences(*material, defgrad);
  }

  TEST_F(ProgressiveDamageTest, DriverRotatesButLengthIsFixedAtInitiation)
  {
    // a flat element, initiated in x-direction where it is long
    set_box(0.5, 0.05, 1.0);
    const double fracture_energy = 1e-2;
    auto material = create_material(0.0, 0.0, fracture_energy);

    const double initiation_stretch = 1.0002;
    (void)evaluate(*material, Core::LinAlg::Tensor<double, 3, 3>{{{initiation_stretch, 0.0, 0.0},
                                  {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}});
    material->update();
    ASSERT_EQ(material->damage(0), 0.0);

    // a larger stretch in y-direction drives the damage with the length in x-direction
    const double stretch = 1.0006;
    (void)evaluate(*material, Core::LinAlg::Tensor<double, 3, 3>{
                                  {{1.0, 0.0, 0.0}, {0.0, stretch, 0.0}, {0.0, 0.0, 1.0}}});

    const double initiation_strain = std::log(initiation_stretch);
    const double strain = std::log(stretch);
    const double failure_strain =
        fracture_energy / (tensile_strength_ * 0.5) + 0.5 * initiation_strain;
    const double expected_damage =
        1.0 - initiation_strain / strain *
                  std::exp(-(strain - initiation_strain) / (failure_strain - initiation_strain));
    EXPECT_NEAR(material->damage(0), expected_damage, 1e-10);
  }

  TEST_F(ProgressiveDamageTest, CharacteristicLengthIsTheExtentAlongTheOpeningDirection)
  {
    // a flat element: the crack opens in y-direction, where it is thin
    set_box(0.5, 0.05, 1.0);
    const double fracture_energy = 1e-4;
    auto material = create_material(0.0, 0.0, fracture_energy);

    const double work = work_until_failure(*material, 1, fracture_energy, 0.05);
    EXPECT_NEAR(work * 0.05, fracture_energy, 1e-2 * fracture_energy);
  }

  class ProgressiveDamageDissipationTest : public ProgressiveDamageTest,
                                           public ::testing::WithParamInterface<double>
  {
  };

  TEST_P(ProgressiveDamageDissipationTest, DissipatesTheFractureEnergyUntilFailure)
  {
    // uniaxial strain without lateral contraction, i.e., a one-dimensional bar
    const double length = GetParam();
    set_box(length, 1.0, 1.0);
    const double fracture_energy = 1e-4;
    auto material = create_material(0.0, 0.0, fracture_energy);

    // until failure, a band of width h takes the fracture energy, independent of h
    const double work = work_until_failure(*material, 0, fracture_energy, length);
    EXPECT_NEAR(work * length, fracture_energy, 1e-2 * fracture_energy);
  }

  // ductility ratios of the failure to the initiation strain from about 3 to 21
  INSTANTIATE_TEST_SUITE_P(
      ElementLengths, ProgressiveDamageDissipationTest, ::testing::Values(0.5, 0.15, 0.05));

  TEST_F(ProgressiveDamageTest, TemperatureDerivativeMatchesFiniteDifferences)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);

    (void)evaluate(*material, sheared_defgrad(1.0002));
    material->update();

    const auto defgrad = sheared_defgrad(1.0004);
    const auto glstrain = green_lagrange_strain(defgrad);
    const double temperature = reference_temperature_ + 10.0;
    (void)evaluate(*material, defgrad, glstrain, temperature);
    ASSERT_GT(material->damage(0), 0.1);

    Teuchos::ParameterList params;
    params.set("temperature", temperature);
    const Core::LinAlg::SymmetricTensor<double, 3, 3> d_stress_d_temperature =
        material->evaluate_d_stress_d_scalar(defgrad, glstrain, params, context_, 0, 0);

    constexpr double perturbation = 1e-4;
    const auto stress_plus =
        evaluate(*material, defgrad, glstrain, temperature + perturbation).stress;
    const auto stress_minus =
        evaluate(*material, defgrad, glstrain, temperature - perturbation).stress;

    FOUR_C_EXPECT_NEAR(
        d_stress_d_temperature, (stress_plus - stress_minus) / (2.0 * perturbation), 1e-8);
  }

  TEST_F(ProgressiveDamageTest, ThermalCouplingNeedsAThermoSolidMaterial)
  {
    auto material = create_material(0.3, 0.0, 1e-2, Core::Materials::m_stvenant);

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->reinit(reference_temperature_, 0), Core::Exception, "no thermo-solid material");
  }

  TEST_F(ProgressiveDamageTest, PackAndUnpackKeepTheDamage)
  {
    auto material = create_material(0.3, 1e-4, 1e-2);

    (void)evaluate(*material, sheared_defgrad(1.0002));
    material->update();
    const auto defgrad = sheared_defgrad(1.0004);
    const StressAndTangent original = evaluate(*material, defgrad);
    material->update();

    Core::Communication::PackBuffer data;
    material->pack(data);
    Core::Communication::UnpackBuffer buffer(data());
    Mat::ProgressiveDamage unpacked_material;
    unpacked_material.unpack(buffer);

    EXPECT_EQ(unpacked_material.damage(0), material->damage(0));
    const StressAndTangent unpacked = evaluate(unpacked_material, defgrad);
    FOUR_C_EXPECT_NEAR(unpacked.stress, original.stress, 1e-12);
  }
}  // namespace
