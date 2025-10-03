// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_interaction_sph_kernel.hpp"

#include "4C_particle_interaction_utils.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace
{
  using namespace FourC;

  class SPHKernelCubicSplineTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<ParticleInteraction::SPHKernelCubicSpline> kernel_1D_;
    std::unique_ptr<ParticleInteraction::SPHKernelCubicSpline> kernel_2D_;
    std::unique_ptr<ParticleInteraction::SPHKernelCubicSpline> kernel_3D_;

    SPHKernelCubicSplineTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_sph_1D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel1D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel1D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel1D), &params_sph_1D);

      Teuchos::ParameterList params_sph_2D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel2D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel2D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel2D), &params_sph_2D);

      Teuchos::ParameterList params_sph_3D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel3D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel3D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel3D), &params_sph_3D);

      // create kernel handler
      kernel_1D_ = std::make_unique<ParticleInteraction::SPHKernelCubicSpline>(params_sph_1D);
      kernel_2D_ = std::make_unique<ParticleInteraction::SPHKernelCubicSpline>(params_sph_2D);
      kernel_3D_ = std::make_unique<ParticleInteraction::SPHKernelCubicSpline>(params_sph_3D);

      // init kernel handler
      kernel_1D_->init();
      kernel_2D_->init();
      kernel_3D_->init();

      // setup kernel handler
      kernel_1D_->setup();
      kernel_2D_->setup();
      kernel_3D_->setup();
    }
    // note: the public functions init() and setup() of class SPHKernelCubicSpline are called in
    // setup() and thus implicitly tested by all following unittests
  };

  TEST_F(SPHKernelCubicSplineTest, kernel_space_dimension)
  {
    int dim = 0;

    kernel_1D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 1);

    kernel_2D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 2);

    kernel_3D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 3);
  }

  TEST_F(SPHKernelCubicSplineTest, SmoothingLength)
  {
    const double support = 0.8;
    const double h = 0.4;

    EXPECT_NEAR(kernel_1D_->smoothing_length(support), h, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->smoothing_length(support), h, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->smoothing_length(support), h, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, normalization_constant)
  {
    const double h = 0.4;
    const double inv_h = 1.0 / h;

    const double normalizationconstant_1D = 2.0 / (3.0 * h);
    const double normalizationconstant_2D =
        10.0 * std::numbers::inv_pi / (7.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);

    EXPECT_NEAR(kernel_1D_->normalization_constant(inv_h), normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->normalization_constant(inv_h), normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->normalization_constant(inv_h), normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, W0)
  {
    const double support = 0.8;
    const double h = 0.4;

    const double normalizationconstant_1D = 2.0 / (3.0 * h);
    const double normalizationconstant_2D =
        10.0 * std::numbers::inv_pi / (7.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);

    double w_unnormalized = 1.0;
    EXPECT_NEAR(kernel_1D_->w0(support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w0(support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w0(support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, W)
  {
    const double support = 0.8;
    const double h = 0.4;

    const double normalizationconstant_1D = 2.0 / (3.0 * h);
    const double normalizationconstant_2D =
        10.0 * std::numbers::inv_pi / (7.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = 1.0;
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = 1.0 - 1.5 * ParticleInteraction::Utils::pow<2>(q) +
                     0.75 * ParticleInteraction::Utils::pow<3>(q);
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.6;
    q = rij / h;
    w_unnormalized = ParticleInteraction::Utils::pow<3>(2.0 - q) / 4.0;
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, dWdrij)
  {
    const double support = 0.8;
    const double h = 0.4;

    const double normalizationconstant_1D = 2.0 / (3.0 * h);
    const double normalizationconstant_2D =
        10.0 * std::numbers::inv_pi / (7.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = (-3.0 * q + 2.25 * ParticleInteraction::Utils::pow<2>(q)) * (1.0 / h);
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.6;
    q = rij / h;
    w_unnormalized = (-0.75 * ParticleInteraction::Utils::pow<2>(2.0 - q)) * (1.0 / h);
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, d2Wdrij2)
  {
    const double support = 0.8;
    const double h = 0.4;

    const double normalizationconstant_1D = 2.0 / (3.0 * h);
    const double normalizationconstant_2D =
        10.0 * std::numbers::inv_pi / (7.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = -3.0 * (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = (-3.0 + 4.5 * q) * (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.6;
    q = rij / h;
    w_unnormalized = (1.5 * (2.0 - q)) * (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelCubicSplineTest, GradWij)
  {
    const double rij = 0.2;
    const double support = 0.8;
    double eij[3];
    eij[0] = 0.5;
    eij[1] = std::sqrt(3.0) / 2.0;
    eij[2] = 0.0;

    const double h = 0.4;
    const double normalizationconstant_3D =
        std::numbers::inv_pi / ParticleInteraction::Utils::pow<3>(h);
    const double q = rij / h;
    const double w_unnormalized =
        (-3.0 * q + 2.25 * ParticleInteraction::Utils::pow<2>(q)) * (1.0 / h);

    double gradWij_reference[3];
    for (int i = 0; i < 3; ++i)
      gradWij_reference[i] = w_unnormalized * normalizationconstant_3D * eij[i];


    double gradWij[3];
    kernel_3D_->grad_wij(rij, support, eij, gradWij);

    FOUR_C_EXPECT_ITERABLE_NEAR(gradWij, gradWij_reference, 3, 1.0e-10);
  }


  class SPHKernelQuinticSplineTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<ParticleInteraction::SPHKernelQuinticSpline> kernel_1D_;
    std::unique_ptr<ParticleInteraction::SPHKernelQuinticSpline> kernel_2D_;
    std::unique_ptr<ParticleInteraction::SPHKernelQuinticSpline> kernel_3D_;

    SPHKernelQuinticSplineTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_sph_1D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel1D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel1D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel1D), &params_sph_1D);

      Teuchos::ParameterList params_sph_2D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel2D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel2D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel2D), &params_sph_2D);

      Teuchos::ParameterList params_sph_3D;
      Teuchos::setStringToIntegralParameter<PARTICLE::KernelSpaceDimension>("KERNEL_SPACE_DIM",
          "Kernel3D", "kernel space dimension number", Teuchos::tuple<std::string>("Kernel3D"),
          Teuchos::tuple<PARTICLE::KernelSpaceDimension>(PARTICLE::Kernel3D), &params_sph_3D);

      // create kernel handler
      kernel_1D_ = std::make_unique<ParticleInteraction::SPHKernelQuinticSpline>(params_sph_1D);
      kernel_2D_ = std::make_unique<ParticleInteraction::SPHKernelQuinticSpline>(params_sph_2D);
      kernel_3D_ = std::make_unique<ParticleInteraction::SPHKernelQuinticSpline>(params_sph_3D);

      // init kernel handler
      kernel_1D_->init();
      kernel_2D_->init();
      kernel_3D_->init();

      // setup kernel handler
      kernel_1D_->setup();
      kernel_2D_->setup();
      kernel_3D_->setup();
    }
    // note: the public functions init() and setup() of class SPHKernelQuinticSpline are called in
    // setup() and thus implicitly tested by all following unittests
  };

  TEST_F(SPHKernelQuinticSplineTest, kernel_space_dimension)
  {
    int dim = 0;

    kernel_1D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 1);

    kernel_2D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 2);

    kernel_3D_->kernel_space_dimension(dim);
    EXPECT_EQ(dim, 3);
  }

  TEST_F(SPHKernelQuinticSplineTest, SmoothingLength)
  {
    const double support = 0.9;
    const double h = 0.3;

    EXPECT_NEAR(kernel_1D_->smoothing_length(support), h, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->smoothing_length(support), h, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->smoothing_length(support), h, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, normalization_constant)
  {
    const double h = 0.3;
    const double inv_h = 1.0 / h;

    const double normalizationconstant_1D = 1.0 / (120.0 * h);
    const double normalizationconstant_2D =
        7.0 * std::numbers::inv_pi / (478.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));

    EXPECT_NEAR(kernel_1D_->normalization_constant(inv_h), normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->normalization_constant(inv_h), normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->normalization_constant(inv_h), normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, W0)
  {
    const double support = 0.9;
    const double h = 0.3;

    const double normalizationconstant_1D = 1.0 / (120.0 * h);
    const double normalizationconstant_2D =
        7.0 * std::numbers::inv_pi / (478.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));

    double w_unnormalized = 66.0;
    EXPECT_NEAR(kernel_1D_->w0(support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w0(support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w0(support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, W)
  {
    const double support = 0.9;
    const double h = 0.3;

    const double normalizationconstant_1D = 1.0 / (120.0 * h);
    const double normalizationconstant_2D =
        7.0 * std::numbers::inv_pi / (478.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = 66.0;
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = ParticleInteraction::Utils::pow<5>(3.0 - q) -
                     6.0 * ParticleInteraction::Utils::pow<5>(2.0 - q) +
                     15.0 * ParticleInteraction::Utils::pow<5>(1.0 - q);
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.5;
    q = rij / h;
    w_unnormalized = ParticleInteraction::Utils::pow<5>(3.0 - q) -
                     6.0 * ParticleInteraction::Utils::pow<5>(2.0 - q);
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = ParticleInteraction::Utils::pow<5>(3.0 - q);
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.9;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(kernel_1D_->w(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(kernel_2D_->w(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(kernel_3D_->w(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, dWdrij)
  {
    const double support = 0.9;
    const double h = 0.3;

    const double normalizationconstant_1D = 1.0 / (120.0 * h);
    const double normalizationconstant_2D =
        7.0 * std::numbers::inv_pi / (478.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = (-5.0 * ParticleInteraction::Utils::pow<4>(3.0 - q) +
                         30.0 * ParticleInteraction::Utils::pow<4>(2.0 - q) -
                         75.0 * ParticleInteraction::Utils::pow<4>(1.0 - q)) *
                     (1.0 / h);
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.5;
    q = rij / h;
    w_unnormalized = (-5.0 * ParticleInteraction::Utils::pow<4>(3.0 - q) +
                         30.0 * ParticleInteraction::Utils::pow<4>(2.0 - q)) *
                     (1.0 / h);
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = (-5.0 * ParticleInteraction::Utils::pow<4>(3.0 - q)) * (1.0 / h);
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.9;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d_wdrij(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, d2Wdrij2)
  {
    const double support = 0.9;
    const double h = 0.3;

    const double normalizationconstant_1D = 1.0 / (120.0 * h);
    const double normalizationconstant_2D =
        7.0 * std::numbers::inv_pi / (478.0 * ParticleInteraction::Utils::pow<2>(h));
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));

    double rij = 0.0;
    double q = rij / h;
    double w_unnormalized = -4000.0 / 3.0;
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.2;
    q = rij / h;
    w_unnormalized = (20.0 * ParticleInteraction::Utils::pow<3>(3.0 - q) -
                         120.0 * ParticleInteraction::Utils::pow<3>(2.0 - q) +
                         300.0 * ParticleInteraction::Utils::pow<3>(1.0 - q)) *
                     (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.5;
    q = rij / h;
    w_unnormalized = (20.0 * ParticleInteraction::Utils::pow<3>(3.0 - q) -
                         120.0 * ParticleInteraction::Utils::pow<3>(2.0 - q)) *
                     (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.8;
    q = rij / h;
    w_unnormalized = (20.0 * ParticleInteraction::Utils::pow<3>(3.0 - q)) *
                     (1.0 / ParticleInteraction::Utils::pow<2>(h));
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);

    rij = 0.9;
    q = rij / h;
    w_unnormalized = 0.0;
    EXPECT_NEAR(
        kernel_1D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_1D, 1.0e-10);
    EXPECT_NEAR(
        kernel_2D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_2D, 1.0e-10);
    EXPECT_NEAR(
        kernel_3D_->d2_wdrij2(rij, support), w_unnormalized * normalizationconstant_3D, 1.0e-10);
  }

  TEST_F(SPHKernelQuinticSplineTest, GradWij)
  {
    const double rij = 0.2;
    const double support = 0.9;
    double eij[3];
    eij[0] = 0.5;
    eij[1] = std::sqrt(3.0) / 2.0;
    eij[2] = 0.0;

    const double h = 0.3;
    const double normalizationconstant_3D =
        3.0 * std::numbers::inv_pi / (359.0 * ParticleInteraction::Utils::pow<3>(h));
    const double q = rij / h;
    const double w_unnormalized = (-5.0 * ParticleInteraction::Utils::pow<4>(3.0 - q) +
                                      30.0 * ParticleInteraction::Utils::pow<4>(2.0 - q) -
                                      75.0 * ParticleInteraction::Utils::pow<4>(1.0 - q)) *
                                  (1.0 / h);

    double gradWij_reference[3];
    for (int i = 0; i < 3; ++i)
      gradWij_reference[i] = w_unnormalized * normalizationconstant_3D * eij[i];


    double gradWij[3];
    kernel_3D_->grad_wij(rij, support, eij, gradWij);

    FOUR_C_EXPECT_ITERABLE_NEAR(gradWij, gradWij_reference, 3, 1.0e-10);
  }
}  // namespace
