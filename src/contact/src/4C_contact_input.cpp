// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_input.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_structure_new_input.hpp"

FOUR_C_NAMESPACE_OPEN



Core::IO::InputSpec CONTACT::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for structural meshtying and contact */
  Core::IO::InputSpec spec = group("CONTACT DYNAMIC",
      {

          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for meshtying and contact",
                  .default_value = -1}),

          parameter<bool>("RESTART_WITH_CONTACT",
              {.description =
                      "Must be chosen if a non-contact simulation is to be restarted with contact",
                  .default_value = false}),

          parameter<CONTACT::AdhesionType>(
              "ADHESION", {.description = "Type of adhesion law",
                              .default_value = CONTACT::AdhesionType::none}),

          deprecated_selection<CONTACT::FrictionType>("FRICTION",
              {
                  {"None", CONTACT::FrictionType::none},
                  {"Stick", CONTACT::FrictionType::stick},
                  {"Tresca", CONTACT::FrictionType::tresca},
                  {"Coulomb", CONTACT::FrictionType::coulomb},
              },
              {.description = "Type of friction law",
                  .default_value = CONTACT::FrictionType::none}),

          parameter<bool>("FRLESS_FIRST",
              {.description = "If chosen the first time step of a newly in contact source "
                              "node is regarded as frictionless",
                  .default_value = false}),

          parameter<bool>("GP_SLIP_INCR",
              {.description =
                      "If chosen the slip increment is computed gp-wise which results to a "
                      "non-objective "
                      "quantity, but this would be consistent to wear and tsi calculations.",
                  .default_value = false}),

          deprecated_selection<CONTACT::SolvingStrategy>("STRATEGY",
              {
                  {"LagrangianMultipliers", CONTACT::SolvingStrategy::lagmult},
                  {"lagrange", CONTACT::SolvingStrategy::lagmult},
                  {"Lagrange", CONTACT::SolvingStrategy::lagmult},
                  {"augmented", CONTACT::SolvingStrategy::augmented},
                  {"penalty", CONTACT::SolvingStrategy::penalty},
                  {"Penalty", CONTACT::SolvingStrategy::penalty},
                  {"Uzawa", CONTACT::SolvingStrategy::uzawa},
                  {"Nitsche", CONTACT::SolvingStrategy::nitsche},
                  {"Ehl", CONTACT::SolvingStrategy::ehl},
                  {"MultiScale", CONTACT::SolvingStrategy::multiscale},
              },
              {.description = "Type of employed solving strategy",
                  .default_value = CONTACT::SolvingStrategy::lagmult}),

          deprecated_selection<CONTACT::SystemType>("SYSTEM",
              {
                  {"Condensed", CONTACT::SystemType::condensed},
                  {"condensed", CONTACT::SystemType::condensed},
                  {"cond", CONTACT::SystemType::condensed},
                  {"Condensedlagmult", CONTACT::SystemType::condensed_lagmult},
                  {"condensedlagmult", CONTACT::SystemType::condensed_lagmult},
                  {"condlm", CONTACT::SystemType::condensed_lagmult},
                  {"SaddlePoint", CONTACT::SystemType::saddlepoint},
                  {"Saddlepoint", CONTACT::SystemType::saddlepoint},
                  {"saddlepoint", CONTACT::SystemType::saddlepoint},
                  {"sp", CONTACT::SystemType::saddlepoint},
                  {"none", CONTACT::SystemType::none},
              },
              {.description = "Type of linear system setup / solution",
                  .default_value = CONTACT::SystemType::condensed}),

          parameter<double>("PENALTYPARAM",
              {.description = "Penalty parameter for penalty / Uzawa augmented solution strategy",
                  .default_value = 0.0}),
          parameter<double>(
              "PENALTYPARAMTAN", {.description = "Tangential penalty parameter for penalty / Uzawa "
                                                 "augmented solution strategy",
                                     .default_value = 0.0}),
          parameter<int>("UZAWAMAXSTEPS",
              {.description = "Maximum no. of Uzawa steps for Uzawa solution strategy",
                  .default_value = 10}),
          parameter<double>("UZAWACONSTRTOL",
              {.description = "Tolerance of constraint norm for Uzawa solution strategy",
                  .default_value = 1.0e-8}),

          parameter<bool>("SEMI_SMOOTH_NEWTON",
              {.description = "If chosen semi-smooth Newton concept is applied",
                  .default_value = true}),

          parameter<double>("SEMI_SMOOTH_CN",
              {.description = "Weighting factor cn for semi-smooth PDASS", .default_value = 1.0}),
          parameter<double>("SEMI_SMOOTH_CT",
              {.description = "Weighting factor ct for semi-smooth PDASS", .default_value = 1.0}),

          parameter<bool>("CONTACTFORCE_ENDTIME",
              {.description =
                      "If chosen, the contact force is not evaluated at the generalized midpoint, "
                      "but at the end of the time step",
                  .default_value = false}),

          parameter<bool>(
              "VELOCITY_UPDATE", {.description = "If chosen, velocity update method is applied",
                                     .default_value = false}),

          parameter<bool>(
              "INITCONTACTBYGAP", {.description = "Initialize init contact by weighted gap vector",
                                      .default_value = false}),

          parameter<double>("INITCONTACTGAPVALUE",
              {.description = "Value for initialization of init contact set with gap vector",
                  .default_value = 0.0}),

          // solver convergence test parameters for contact/meshtying in saddlepoint formulation
          deprecated_selection<Solid::BinaryOp>("NORMCOMBI_RESFCONTCONSTR",
              {
                  {"And", Solid::bop_and},
                  {"Or", Solid::bop_or},
              },
              {.description =
                      "binary operator to combine contact constraints and residual force values",
                  .default_value = Solid::bop_and}),

          deprecated_selection<Solid::BinaryOp>("NORMCOMBI_DISPLAGR",
              {
                  {"And", Solid::bop_and},
                  {"Or", Solid::bop_or},
              },
              {.description =
                      "binary operator to combine displacement increments and Lagrange multiplier "
                      "increment values",
                  .default_value = Solid::bop_and}),

          parameter<double>("TOLCONTCONSTR",
              {.description = "tolerance in the contact constraint norm for the newton "
                              "iteration (saddlepoint formulation only)",
                  .default_value = 1.0E-6}),
          parameter<double>("TOLLAGR", {.description = "tolerance in the LM norm for the newton "
                                                       "iteration (saddlepoint formulation only)",
                                           .default_value = 1.0E-6}),

          parameter<CONTACT::ConstraintDirection>("CONSTRAINT_DIRECTIONS",
              {.description = "formulation of constraints in normal/tangential or xyz-direction",
                  .default_value = CONTACT::ConstraintDirection::ntt}),

          parameter<bool>("NONSMOOTH_GEOMETRIES",
              {.description =
                      "If chosen the contact algorithm combines mortar and nts formulations. This "
                      "is needed if contact between entities of different geometric dimension "
                      "(such as contact between surfaces and lines, or lines and nodes) can occur",
                  .default_value = false}),

          parameter<bool>("NONSMOOTH_CONTACT_SURFACE",
              {.description =
                      "This flag is used to alter the criterion for the evaluation of the "
                      "so-called "
                      "qualified vectors in the case of a self contact scenario. This is needed as "
                      "the "
                      "standard criterion is only valid for smooth surfaces and thus has to be "
                      "altered, if "
                      "the surface that is defined to be a self contact surface is non-smooth!",
                  .default_value = false}),

          parameter<double>("HYBRID_ANGLE_MIN",
              {.description = "Non-smooth contact: angle between cpp normal and "
                              "element normal: begin transition (Mortar)",
                  .default_value = -1.0}),
          parameter<double>("HYBRID_ANGLE_MAX",
              {.description = "Non-smooth contact: angle between cpp normal and "
                              "element normal: end transition (NTS)",
                  .default_value = -1.0}),

          parameter<bool>("CPP_NORMALS",
              {.description =
                      "If chosen the nodal normal field is created as averaged CPP normal field.",
                  .default_value = false}),

          parameter<bool>("TIMING_DETAILS",
              {.description = "Enable and print detailed contact timings to screen.",
                  .default_value = false}),

          // --------------------------------------------------------------------------
          parameter<double>("NITSCHE_THETA",
              {.description = "+1: symmetric, 0: non-symmetric, -1: skew-symmetric",
                  .default_value = 0.0}),
          parameter<double>("NITSCHE_THETA_2",
              {.description = "+1: Chouly-type, 0: Burman penalty-free (only with theta=-1)",
                  .default_value = 1.0}),

          parameter<CONTACT::NitscheWeighting>("NITSCHE_WEIGHTING",
              {.description = "how to weight consistency terms in Nitsche contact formulation",
                  .default_value = CONTACT::NitscheWeighting::harmonic}),

          parameter<bool>("NITSCHE_PENALTY_ADAPTIVE",
              {.description = "adapt penalty parameter after each converged time step",
                  .default_value = true}),

          parameter<bool>("REGULARIZED_NORMAL_CONTACT",
              {.description = "add a regularized normal contact formulation",
                  .default_value = false}),
          parameter<double>("REGULARIZATION_THICKNESS",
              {.description = "maximum contact penetration", .default_value = -1.}),
          parameter<double>("REGULARIZATION_STIFFNESS",
              {.description = "initial contact stiffness (i.e. initial \"penalty parameter\")",
                  .default_value = -1.})},
      {.required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE