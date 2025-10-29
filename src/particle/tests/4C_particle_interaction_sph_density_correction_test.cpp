// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_interaction_sph_density_correction.hpp"


namespace
{
  using namespace FourC;

  class SPHDensityCorrectionInteriorTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::SPHDensityCorrectionInterior> densitycorrection_;

   public:
    SPHDensityCorrectionInteriorTest()
    {
      // create density correction handler
      densitycorrection_ = std::make_unique<Particle::SPHDensityCorrectionInterior>();

      // init density correction handler
      densitycorrection_->init();

      // setup density correction handler
      densitycorrection_->setup();
    }
    // note: the public functions init() and setup() of class SPHEquationOfStateGenTait are called
    // in setup() and thus implicitly tested by all following unittests
  };

  TEST_F(SPHDensityCorrectionInteriorTest, ComputeDensityBC)
  {
    EXPECT_FALSE(densitycorrection_->compute_density_bc());
  }

  TEST_F(SPHDensityCorrectionInteriorTest, corrected_density_interior)
  {
    const double denssum = 1.07;
    double dens = 0.98;

    densitycorrection_->corrected_density_interior(&denssum, &dens);

    EXPECT_NEAR(dens, denssum, 1e-14);
  }

  TEST_F(SPHDensityCorrectionInteriorTest, corrected_density_free_surface)
  {
    const double denssum = 1.07;
    const double colorfield = 0.82;
    const double dens_bc = 1.05;
    double dens = 0.78;

    const double dens_ref = dens;

    densitycorrection_->corrected_density_free_surface(&denssum, &colorfield, &dens_bc, &dens);

    EXPECT_NEAR(dens, dens_ref, 1e-14);
  }


  class SPHDensityCorrectionNormalizedTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::SPHDensityCorrectionNormalized> densitycorrection_;

    SPHDensityCorrectionNormalizedTest()
    {
      // create density correction handler
      densitycorrection_ = std::make_unique<Particle::SPHDensityCorrectionNormalized>();

      // init density correction handler
      densitycorrection_->init();

      // setup density correction handler
      densitycorrection_->setup();
    }
    // note: the public functions init() and setup() of class SPHEquationOfStateGenTait are called
    // in SetUp() and thus implicitly tested by all following unittests
  };

  TEST_F(SPHDensityCorrectionNormalizedTest, ComputeDensityBC)
  {
    EXPECT_FALSE(densitycorrection_->compute_density_bc());
  }
  TEST_F(SPHDensityCorrectionNormalizedTest, corrected_density_interior)
  {
    const double denssum = 1.07;
    double dens = 0.98;

    densitycorrection_->corrected_density_interior(&denssum, &dens);

    EXPECT_NEAR(dens, denssum, 1e-14);
  }

  TEST_F(SPHDensityCorrectionNormalizedTest, corrected_density_free_surface)
  {
    const double denssum = 1.07;
    const double colorfield = 0.82;
    const double dens_bc = 1.05;
    double dens = 0.78;

    const double dens_ref = denssum / colorfield;

    densitycorrection_->corrected_density_free_surface(&denssum, &colorfield, &dens_bc, &dens);

    EXPECT_NEAR(dens, dens_ref, 1e-14);
  }


  class SPHDensityCorrectionRandlesTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::SPHDensityCorrectionRandles> densitycorrection_;

    SPHDensityCorrectionRandlesTest()
    {
      // create density correction handler
      densitycorrection_ = std::make_unique<Particle::SPHDensityCorrectionRandles>();

      // init density correction handler
      densitycorrection_->init();

      // setup density correction handler
      densitycorrection_->setup();
    }
    // note: the public functions init() and setup() of class SPHEquationOfStateGenTait are called
    // in SetUp() and thus implicitly tested by all following unittests
  };
  TEST_F(SPHDensityCorrectionRandlesTest, ComputeDensityBC)
  {
    EXPECT_TRUE(densitycorrection_->compute_density_bc());
  }

  TEST_F(SPHDensityCorrectionRandlesTest, corrected_density_interior)
  {
    const double denssum = 1.07;
    double dens = 0.98;

    densitycorrection_->corrected_density_interior(&denssum, &dens);

    EXPECT_NEAR(dens, denssum, 1e-14);
  }

  TEST_F(SPHDensityCorrectionRandlesTest, corrected_density_free_surface)
  {
    const double denssum = 1.07;
    const double colorfield = 0.82;
    const double dens_bc = 1.05;
    double dens = 0.78;

    const double dens_ref = denssum + dens_bc * (1.0 - colorfield);

    densitycorrection_->corrected_density_free_surface(&denssum, &colorfield, &dens_bc, &dens);

    EXPECT_NEAR(dens, dens_ref, 1e-14);
  }
}  // namespace
