// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_ductile_damage.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_trait_plastic.hpp"
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

  constexpr int plastic_matid = 1;
  constexpr int damage_matid = 2;
  constexpr int elastic_matid = 3;

  struct StressAndTangent
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress;
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat;
  };

  class DuctileDamageTest : public ::testing::Test
  {
   protected:
    //! Add the wrapped plastic material (perfectly plastic for @p hardening = 0)
    void add_plastic_material(const double hardening) const
    {
      if (Global::Problem::instance()->materials()->id_exists(plastic_matid)) return;
      Core::IO::InputParameterContainer data;
      data.add("YOUNG", young_);
      data.add("NUE", 0.3);
      data.add("DENS", 1.0);
      data.add("CTE", 0.0);
      data.add("INITTEMP", temperature_);
      data.add("YIELD", yield_stress_);
      data.add("ISOHARD", hardening);
      // no saturation hardening, i.e., linear isotropic hardening
      data.add("SATHARDENING", yield_stress_);
      data.add("HARDEXPO", 0.0);
      data.add("YIELDSOFT", 0.0);
      data.add("HARDSOFT", 0.0);
      data.add("TOL", 1e-12);
      Global::Problem::instance()->materials()->insert(plastic_matid,
          Mat::make_parameter(plastic_matid, Core::Materials::m_thermoplhyperelast, data));
    }

    std::shared_ptr<Mat::DuctileDamage> create_material(const double hardening,
        const double fracture_strain, const double fracture_energy, const int wrapped_matid)
    {
      Global::Problem& problem = *Global::Problem::instance();
      problem.materials()->set_read_from_problem(0);
      add_plastic_material(hardening);

      if (!problem.materials()->id_exists(damage_matid))
      {
        Core::IO::InputParameterContainer data;
        data.add("MATID", wrapped_matid);
        data.add("D1", fracture_strain);
        data.add("D2", 0.0);
        data.add("D3", 0.0);
        data.add("FRACTURE_ENERGY", fracture_energy);
        data.add("RESIDUAL_STRENGTH_RATIO", residual_strength_ratio_);
        problem.materials()->insert(damage_matid,
            Mat::make_parameter(damage_matid, Core::Materials::m_ductile_damage, data));
      }

      auto material = std::dynamic_pointer_cast<Mat::DuctileDamage>(Mat::factory(damage_matid));
      material->setup(1, {}, {});
      return material;
    }

    //! The stretch tensor of a Green-Lagrange strain, i.e., a deformation gradient without rotation
    static Core::LinAlg::Tensor<double, 3, 3> stretch(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain)
    {
      const auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(
          2.0 * glstrain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
      Core::LinAlg::Tensor<double, 3, 3> stretch{};
      for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            stretch(i, j) += std::sqrt(eigenvalues[k]) * eigenvectors(i, k) * eigenvectors(j, k);
      return stretch;
    }

    //! Green-Lagrange strain of a uniaxial strain in x-direction with some shear
    static Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain(
        const double strain, const double shear = 0.2)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain{};
      glstrain(0, 0) = strain;
      glstrain(0, 1) = shear * strain;
      return glstrain;
    }

    StressAndTangent evaluate(
        Mat::So3Material& material, const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain)
    {
      Teuchos::ParameterList params;
      params.set("temperature", temperature_);
      const Core::LinAlg::Tensor<double, 3, 3> defgrad = stretch(glstrain);
      StressAndTangent result{};
      material.evaluate(&defgrad, glstrain, params, context_, result.stress, result.cmat, 0, 0);
      return result;
    }

    //! Load in steps up to @p strain and accept each step
    void load(Mat::So3Material& material, const double strain, const int steps)
    {
      for (int step = 1; step <= steps; ++step)
      {
        (void)evaluate(material, glstrain(strain * step / steps));
        material.update();
      }
    }

    void expect_tangent_matches_finite_differences(
        Mat::So3Material& material, const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain)
    {
      const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat =
          evaluate(material, glstrain).cmat;

      constexpr double perturbation = 1e-7;
      for (int i = 0; i < 3; ++i)
      {
        for (int j = i; j < 3; ++j)
        {
          Core::LinAlg::SymmetricTensor<double, 3, 3> direction{};
          direction(i, j) = 1.0;
          const auto stress_plus = evaluate(material, glstrain + perturbation * direction).stress;
          const auto stress_minus = evaluate(material, glstrain - perturbation * direction).stress;
          FOUR_C_EXPECT_NEAR(Core::LinAlg::ddot(cmat, direction),
              (stress_plus - stress_minus) / (2.0 * perturbation), 1e-4 * young_);
        }
      }

      // restore the state of the evaluation at the given strain
      (void)evaluate(material, glstrain);
    }

    void set_box(const double length)
    {
      for (int node = 0; node < 8; ++node)
      {
        nodal_coordinates_[3 * node] = (node & 1) ? length : 0.0;
        nodal_coordinates_[3 * node + 1] = (node & 2) ? 1.0 : 0.0;
        nodal_coordinates_[3 * node + 2] = (node & 4) ? 1.0 : 0.0;
      }
    }

    const double young_ = 2.0e5;
    const double yield_stress_ = 400.0;
    const double temperature_ = 293.0;
    const double residual_strength_ratio_ = 0.01;

    //! reference coordinates of the nodes of a box-shaped element, 0.5 long in x-direction
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

  TEST_F(
      DuctileDamageTest, PlasticStrainDerivativeOfThermoPlasticHyperElastMatchesFiniteDifferences)
  {
    Global::Problem::instance()->materials()->set_read_from_problem(0);
    add_plastic_material(1000.0);
    auto material = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(plastic_matid));
    material->setup(1, {}, {});
    const auto& plastic = dynamic_cast<const Mat::Trait::Plastic&>(*material);

    load(*material, 0.004, 4);
    const auto strain = glstrain(0.006);
    (void)evaluate(*material, strain);
    ASSERT_GT(plastic.equivalent_plastic_strain(0), 0.0);
    const Core::LinAlg::SymmetricTensor<double, 3, 3> derivative =
        plastic.d_equivalent_plastic_strain_d_glstrain(0);

    constexpr double perturbation = 1e-7;
    for (int i = 0; i < 3; ++i)
    {
      for (int j = i; j < 3; ++j)
      {
        Core::LinAlg::SymmetricTensor<double, 3, 3> direction{};
        direction(i, j) = 1.0;
        (void)evaluate(*material, strain + perturbation * direction);
        const double plus = plastic.equivalent_plastic_strain(0);
        (void)evaluate(*material, strain - perturbation * direction);
        const double minus = plastic.equivalent_plastic_strain(0);
        EXPECT_NEAR(
            Core::LinAlg::ddot(derivative, direction), (plus - minus) / (2.0 * perturbation), 1e-5);
      }
    }
  }

  TEST_F(DuctileDamageTest, InitiatesAtTheFractureStrain)
  {
    const double fracture_strain = 0.02;
    auto material = create_material(0.0, fracture_strain, 10.0, plastic_matid);

    // elastic and early plastic loading: the indicator grows, the damage does not
    load(*material, 0.01, 10);
    EXPECT_GT(material->initiation_indicator(0), 0.0);
    EXPECT_LT(material->initiation_indicator(0), 1.0);
    EXPECT_EQ(material->damage(0), 0.0);

    // well beyond the fracture strain the damage grows
    load(*material, 0.06, 60);
    EXPECT_GE(material->initiation_indicator(0), 1.0);
    EXPECT_GT(material->damage(0), 0.0);
    EXPECT_FALSE(material->is_failed(0));
  }

  TEST_F(DuctileDamageTest, TangentMatchesFiniteDifferencesWhileTheDamageGrows)
  {
    auto material = create_material(1000.0, 0.005, 10.0, plastic_matid);
    load(*material, 0.02, 20);
    ASSERT_GT(material->damage(0), 0.0);

    const auto strain = glstrain(0.021);
    (void)evaluate(*material, strain);
    ASSERT_GT(material->damage(0), 0.0);
    ASSERT_FALSE(material->is_failed(0));
    expect_tangent_matches_finite_differences(*material, strain);
  }

  TEST_F(DuctileDamageTest, StrainUntilFailureScalesWithTheInverseElementLength)
  {
    // logarithmic strain from the initiation to the failure of a perfectly plastic material
    const auto strain_from_initiation_to_failure = [&](const double length)
    {
      set_box(length);
      auto material = create_material(0.0, 0.005, 1.0, plastic_matid);
      const double increment = 1e-5;
      double strain = 0.0;
      double initiation = -1.0;
      while (!material->is_failed(0))
      {
        strain += increment;
        // without shear, the opening direction is the x-direction, along which the box is long
        (void)evaluate(*material, glstrain(strain, 0.0));
        material->update();
        if (initiation < 0.0 && material->initiation_indicator(0) >= 1.0) initiation = strain;
        if (strain > 1.0) return std::numeric_limits<double>::infinity();
      }
      // the plastic flow follows the logarithmic strain
      const auto logarithmic = [](const double green_lagrange)
      { return 0.5 * std::log(1.0 + 2.0 * green_lagrange); };
      return logarithmic(strain) - logarithmic(initiation);
    };

    const double long_element = strain_from_initiation_to_failure(0.5);
    const double short_element = strain_from_initiation_to_failure(0.25);
    EXPECT_NEAR(short_element / long_element, 2.0, 0.1);
  }

  TEST_F(DuctileDamageTest, NeedsAPlasticMaterial)
  {
    Global::Problem& problem = *Global::Problem::instance();
    Core::IO::InputParameterContainer elastic_data;
    elastic_data.add("YOUNG", young_);
    elastic_data.add("NUE", 0.3);
    elastic_data.add("DENS", 1.0);
    problem.materials()->insert(elastic_matid,
        Mat::make_parameter(elastic_matid, Core::Materials::m_stvenant, elastic_data));

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_material(0.0, 0.1, 1.0, elastic_matid), Core::Exception, "no plastic material");
  }
}  // namespace
