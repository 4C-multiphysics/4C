// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_interaction_dem_contact_normal.hpp"

#include "4C_particle_interaction_utils.hpp"

#include <Teuchos_ParameterList.hpp>

#include <numbers>


namespace
{
  using namespace FourC;

  class DEMContactNormalLinearSpringTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalLinearSpring> contactnormal_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;

    const double dens_max_ = 1.0;

    DEMContactNormalLinearSpringTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalLinearSpring>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
    }

    // note: the public functions setup() of class DEMContactNormalLinearSpring are
    // called in the constructor and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalLinearSpringTest, get_normal_contact_stiffness)
  {
    const double k_normal = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) /
                            Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalLinearSpringTest, GetTimeCriticalStiffness)
  {
    const double k_normal = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) /
                            Particle::ParticleUtils::pow<2>(c_);

    const double k_normal_crit = k_normal;

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalLinearSpringTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double normalcontactforce_ref = contactnormal_->get_normal_contact_stiffness() * gap;

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);
  }

  TEST_F(DEMContactNormalLinearSpringTest, normal_potential_energy)
  {
    const double gap = -0.15;

    const double normalpotentialenergy_ref =
        0.5 * contactnormal_->get_normal_contact_stiffness() * Particle::ParticleUtils::pow<2>(gap);

    double normalpotentialenergy = 0.0;
    contactnormal_->normal_potential_energy(gap, normalpotentialenergy);

    EXPECT_NEAR(normalpotentialenergy, normalpotentialenergy_ref, 1.0e-12);
  }


  class DEMContactNormalLinearSpringDampTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalLinearSpringDamp> contactnormal_;
    std::unique_ptr<Particle::DEMContactNormalLinearSpringDamp> contactnormal_ezero_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;
    const double e_ = 0.8;

    const double dens_max_ = 1.0;

    DEMContactNormalLinearSpringDampTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);
      params_dem.set("COEFF_RESTITUTION", e_);
      params_dem.set("DAMP_REG_FAC", -1.0);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalLinearSpringDamp>(params_dem);

      params_dem.set("COEFF_RESTITUTION", 0.0);
      contactnormal_ezero_ =
          std::make_unique<Particle::DEMContactNormalLinearSpringDamp>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
      contactnormal_ezero_->setup(dens_max_);
    }
    // note: the public function setup() of class DEMContactNormalLinearSpringDamp are
    // called in the constructor and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalLinearSpringDampTest, get_normal_contact_stiffness)
  {
    const double k_normal = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) /
                            Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalLinearSpringDampTest, GetTimeCriticalStiffness)
  {
    const double k_normal = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) /
                            Particle::ParticleUtils::pow<2>(c_);

    const double k_normal_crit = k_normal;

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalLinearSpringDampTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double k_normal = contactnormal_->get_normal_contact_stiffness();
    const double lne = std::log(e_);
    const double d_normal_fac =
        2.0 * std::abs(lne) *
        std::sqrt(k_normal / (Particle::ParticleUtils::pow<2>(lne) +
                                 Particle::ParticleUtils::pow<2>(std::numbers::pi)));
    const double d_normal = d_normal_fac * std::sqrt(m_eff);
    const double normalcontactforce_ref = k_normal * gap - d_normal * v_rel_normal;

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);



    const double d_ezero_normal = 2.0 * std::sqrt(k_normal) * std::sqrt(m_eff);
    const double normalcontactforce_ezero_ref = k_normal * gap - d_ezero_normal * v_rel_normal;

    double normalcontactforce_ezero = 0.0;
    contactnormal_ezero_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce_ezero);

    EXPECT_NEAR(normalcontactforce_ezero, normalcontactforce_ezero_ref, 1.0e-12);
  }

  class DEMContactNormalHertzTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalHertz> contactnormal_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;

    const double dens_max_ = 1.0;

    DEMContactNormalHertzTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalHertz>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
    }

    // note: the public functions setup() of class DEMContactNormalHertz are called in
    // setup() and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalHertzTest, get_normal_contact_stiffness)
  {
    const double k_normal = 10.0 / 3.0 * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) * std::pow(r_max_, 0.5) /
                            std::pow(2.0 * c_, 2.5);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalHertzTest, GetTimeCriticalStiffness)
  {
    const double k_normal_crit = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                                 Particle::ParticleUtils::pow<2>(v_max_) /
                                 Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalHertzTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double normalcontactforce_ref =
        -contactnormal_->get_normal_contact_stiffness() * std::pow(-gap, 1.5);

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);
  }

  TEST_F(DEMContactNormalHertzTest, normal_potential_energy)
  {
    const double gap = -0.15;

    const double normalpotentialenergy_ref = 0.4 * contactnormal_->get_normal_contact_stiffness() *
                                             Particle::ParticleUtils::pow<2>(gap) * std::sqrt(-gap);

    double normalpotentialenergy = 0.0;
    contactnormal_->normal_potential_energy(gap, normalpotentialenergy);

    EXPECT_NEAR(normalpotentialenergy, normalpotentialenergy_ref, 1.0e-12);
  }

  class DEMContactNormalLeeHerrmannTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalLeeHerrmann> contactnormal_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;
    const double d_normal_ = 6.0;

    const double dens_max_ = 1.0;

    DEMContactNormalLeeHerrmannTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);
      params_dem.set("NORMAL_DAMP", d_normal_);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalLeeHerrmann>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
    }
    // note: the public functions setup() of class DEMContactNormalLeeHerrmann are called
    // in setup() and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalLeeHerrmannTest, get_normal_contact_stiffness)
  {
    const double k_normal = 10.0 / 3.0 * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) * std::pow(r_max_, 0.5) /
                            std::pow(2.0 * c_, 2.5);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalLeeHerrmannTest, GetTimeCriticalStiffness)
  {
    const double k_normal_crit = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                                 Particle::ParticleUtils::pow<2>(v_max_) /
                                 Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalLeeHerrmannTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double normalcontactforce_ref =
        -contactnormal_->get_normal_contact_stiffness() * std::pow(-gap, 1.5) -
        m_eff * d_normal_ * v_rel_normal;

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);
  }

  class DEMContactNormalKuwabaraKonoTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalKuwabaraKono> contactnormal_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;
    const double d_normal_ = 6.0;

    const double dens_max_ = 1.0;

    DEMContactNormalKuwabaraKonoTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);
      params_dem.set("NORMAL_DAMP", d_normal_);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalKuwabaraKono>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
    }
    // note: the public functions setup() of class DEMContactNormalKuwabaraKono are
    // called in the constructor and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalKuwabaraKonoTest, get_normal_contact_stiffness)
  {
    const double k_normal = 10.0 / 3.0 * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) * std::pow(r_max_, 0.5) /
                            std::pow(2.0 * c_, 2.5);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalKuwabaraKonoTest, GetTimeCriticalStiffness)
  {
    const double k_normal_crit = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                                 Particle::ParticleUtils::pow<2>(v_max_) /
                                 Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalKuwabaraKonoTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double normalcontactforce_ref =
        -contactnormal_->get_normal_contact_stiffness() * std::pow(-gap, 1.5) -
        d_normal_ * v_rel_normal * std::pow(-gap, 0.5);

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);
  }

  class DEMContactNormalTsujiTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::DEMContactNormalTsuji> contactnormal_;

    const double r_max_ = 1.5;
    const double v_max_ = 1.25;
    const double c_ = 0.05;
    const double d_normal_ = 6.0;

    const double dens_max_ = 1.0;

    DEMContactNormalTsujiTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("MAX_RADIUS", r_max_);
      params_dem.set("MAX_VELOCITY", v_max_);
      params_dem.set("REL_PENETRATION", c_);
      params_dem.set("NORMAL_STIFF", -1.0);
      params_dem.set("NORMAL_DAMP", d_normal_);

      // create normal contact handler
      contactnormal_ = std::make_unique<Particle::DEMContactNormalTsuji>(params_dem);

      // setup normal contact handler
      contactnormal_->setup(dens_max_);
    }
    // note: the public functions setup() of class DEMContactNormalTsuji are called in
    // setup() and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactNormalTsujiTest, get_normal_contact_stiffness)
  {
    const double k_normal = 10.0 / 3.0 * std::numbers::pi * dens_max_ *
                            Particle::ParticleUtils::pow<2>(v_max_) * std::pow(r_max_, 0.5) /
                            std::pow(2.0 * c_, 2.5);

    EXPECT_NEAR(contactnormal_->get_normal_contact_stiffness(), k_normal, 1.0e-12);
  }

  TEST_F(DEMContactNormalTsujiTest, GetTimeCriticalStiffness)
  {
    const double k_normal_crit = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max_ *
                                 Particle::ParticleUtils::pow<2>(v_max_) /
                                 Particle::ParticleUtils::pow<2>(c_);

    EXPECT_NEAR(contactnormal_->get_critical_normal_contact_stiffness(), k_normal_crit, 1.0e-12);
  }

  TEST_F(DEMContactNormalTsujiTest, NormalContactForce)
  {
    const double gap = -0.15;
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double v_rel_normal = 0.3;
    const double m_eff = 1.0;

    const double normalcontactforce_ref =
        -contactnormal_->get_normal_contact_stiffness() * std::pow(-gap, 1.5) -
        d_normal_ * v_rel_normal * std::pow(-gap, 0.25);

    double normalcontactforce = 0.0;
    contactnormal_->normal_contact_force(
        gap, &rad_i, &rad_j, v_rel_normal, m_eff, normalcontactforce);

    EXPECT_NEAR(normalcontactforce, normalcontactforce_ref, 1.0e-12);
  }
}  // namespace
