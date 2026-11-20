// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_rigidbody_utils.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

#include <numbers>

namespace
{
  using namespace FourC;

  TEST(QuaternionTest, Clear)
  {
    const double q_ref[4] = {0.0, 0.0, 0.0, 1.0};
    double q[4] = {1.0, 2.0, 3.0, 4.0};
    Particle::ParticleUtils::quaternion_clear(q);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, Set)
  {
    double q1[4] = {0.0, 0.0, 0.0, 1.0};
    const double q2[4] = {1.0, 2.0, 3.0, 4.0};

    Particle::ParticleUtils::quaternion_set(q1, q2);

    FOUR_C_EXPECT_ITERABLE_NEAR(q1, q2, 4, 1.0e-14);
  }

  TEST(QuaternionTest, invert)
  {
    double q1[4] = {0.0, 0.0, 0.0, 1.0};
    const double q2[4] = {1.0, 2.0, 3.0, 4.0};
    const double q_ref[4] = {-q2[0], -q2[1], -q2[2], q2[3]};

    Particle::ParticleUtils::quaternion_invert(q1, q2);

    FOUR_C_EXPECT_ITERABLE_NEAR(q1, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, Product)
  {
    const double q12_ref[4] = {
        -0.01121419126499877, 0.9977058985744629, -0.05089858263600289, 0.04320319605818204};
    const double phi1[3] = {0.1, -2.0, 0.3};
    const double phi2[3] = {-0.8, 5.0, 0.0};

    double q1[4];
    Particle::ParticleUtils::quaternion_from_angle(q1, phi1);

    double q2[4];
    Particle::ParticleUtils::quaternion_from_angle(q2, phi2);

    double q12[4];
    Particle::ParticleUtils::quaternion_product(q12, q2, q1);

    FOUR_C_EXPECT_ITERABLE_NEAR(q12, q12_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, FromAngleZero)
  {
    const double q_ref[4] = {0.0, 0.0, 0.0, 1.0};
    const double phi[3] = {0.0, 0.0, 0.0};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, FromAngleXAxis)
  {
    const double q_ref[4] = {
        std::sin(std::numbers::pi / 4), 0.0, 0.0, std::cos(std::numbers::pi / 4)};
    const double phi[3] = {std::numbers::pi / 2, 0.0, 0.0};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, FromAngleYAxis)
  {
    const double q_ref[4] = {
        0.0, std::sin(std::numbers::pi / 4), 0.0, std::cos(std::numbers::pi / 4)};
    const double phi[3] = {0.0, std::numbers::pi / 2, 0.0};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, FromAngleZAxis)
  {
    const double q_ref[4] = {
        0.0, 0.0, std::sin(std::numbers::pi / 4), std::cos(std::numbers::pi / 4)};
    const double phi[3] = {0.0, 0.0, std::numbers::pi / 2};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, FromAngleGeneral)
  {
    const double q_ref[4] = {
        -0.2759788075111623, 0.8279364225334871, 0.4139682112667435, -0.2588190451025209};
    const double phi[3] = {-std::numbers::pi / 3, std::numbers::pi, std::numbers::pi / 2};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    FOUR_C_EXPECT_ITERABLE_NEAR(q, q_ref, 4, 1.0e-14);
  }

  TEST(QuaternionTest, RotateVectorXUnitAroundZAxis)
  {
    const double w_ref[3] = {0.0, 1.0, 0.0};
    double v[3] = {1.0, 0.0, 0.0};
    double w[3] = {0.0, 0.0, 0.0};

    const double q[4] = {0.0, 0.0, std::sin(std::numbers::pi / 4), std::cos(std::numbers::pi / 4)};

    Particle::ParticleUtils::quaternion_rotate_vector(w, q, v);

    FOUR_C_EXPECT_ITERABLE_NEAR(w, w_ref, 3, 1.0e-14);
  }

  TEST(QuaternionTest, RotateVectorZUnitAroundYAxis)
  {
    const double w_ref[3] = {1.0, 0.0, 0.0};
    double v[3] = {0.0, 0.0, 1.0};
    double w[3] = {0.0, 0.0, 0.0};

    const double q[4] = {0.0, std::sin(std::numbers::pi / 4), 0.0, std::cos(std::numbers::pi / 4)};

    Particle::ParticleUtils::quaternion_rotate_vector(w, q, v);

    FOUR_C_EXPECT_ITERABLE_NEAR(w, w_ref, 3, 1.0e-14);
  }

  TEST(QuaternionTest, RotateVectorYUnitAroundXAxis)
  {
    const double w_ref[3] = {0.0, 0.0, 1.0};
    double v[3] = {0.0, 1.0, 0.0};
    double w[3] = {0.0, 0.0, 0.0};

    const double q[4] = {std::sin(std::numbers::pi / 4), 0.0, 0.0, std::cos(std::numbers::pi / 4)};

    Particle::ParticleUtils::quaternion_rotate_vector(w, q, v);

    FOUR_C_EXPECT_ITERABLE_NEAR(w, w_ref, 3, 1.0e-14);
  }

  TEST(QuaternionTest, RotateVectorGeneral)
  {
    const double w_ref[3] = {0.7145801717316358, -0.9159468817988596, 1.97494721141881};
    double v[3] = {0.5, 1.0, -2.0};
    double w[3] = {0.0, 0.0, 0.0};

    const double phi[3] = {-std::numbers::pi / 3, std::numbers::pi, std::numbers::pi / 2};

    double q[4];
    Particle::ParticleUtils::quaternion_from_angle(q, phi);

    Particle::ParticleUtils::quaternion_rotate_vector(w, q, v);

    FOUR_C_EXPECT_ITERABLE_NEAR(w, w_ref, 3, 1.0e-14);
  }
}  // namespace
