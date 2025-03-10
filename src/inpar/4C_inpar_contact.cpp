// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_contact.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::CONTACT::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for structural meshtying and contact */
  Core::Utils::SectionSpecs scontact{"CONTACT DYNAMIC"};

  scontact.specs.emplace_back(parameter<int>(
      "LINEAR_SOLVER", {.description = "number of linear solver used for meshtying and contact",
                           .default_value = -1}));

  scontact.specs.emplace_back(parameter<bool>("RESTART_WITH_CONTACT",
      {.description = "Must be chosen if a non-contact simulation is to be restarted with contact",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::AdhesionType>("ADHESION", "None",
      "Type of adhesion law", tuple<std::string>("None", "none", "bounded", "b"),
      tuple<Inpar::CONTACT::AdhesionType>(
          adhesion_none, adhesion_none, adhesion_bound, adhesion_bound),
      scontact);

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::FrictionType>("FRICTION", "None",
      "Type of friction law", tuple<std::string>("None", "Stick", "Tresca", "Coulomb"),
      tuple<Inpar::CONTACT::FrictionType>(
          friction_none, friction_stick, friction_tresca, friction_coulomb),
      scontact);

  scontact.specs.emplace_back(parameter<bool>(
      "FRLESS_FIRST", {.description = "If chosen the first time step of a newly in contact slave "
                                      "node is regarded as frictionless",
                          .default_value = false}));

  scontact.specs.emplace_back(parameter<bool>("GP_SLIP_INCR",
      {.description =
              "If chosen the slip increment is computed gp-wise which results to a non-objective "
              "quantity, but this would be consistent to wear and tsi calculations.",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::SolvingStrategy>("STRATEGY",
      "LagrangianMultipliers", "Type of employed solving strategy",
      tuple<std::string>("LagrangianMultipliers", "lagrange", "Lagrange", "penalty", "Penalty",
          "Uzawa", "Nitsche", "Ehl", "MultiScale"),
      tuple<Inpar::CONTACT::SolvingStrategy>(solution_lagmult, solution_lagmult, solution_lagmult,
          solution_penalty, solution_penalty, solution_uzawa, solution_nitsche, solution_ehl,
          solution_multiscale),
      scontact);

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::SystemType>("SYSTEM", "Condensed",
      "Type of linear system setup / solution",
      tuple<std::string>("Condensed", "condensed", "cond", "Condensedlagmult", "condensedlagmult",
          "condlm", "SaddlePoint", "Saddlepoint", "saddlepoint", "sp", "none"),
      tuple<Inpar::CONTACT::SystemType>(system_condensed, system_condensed, system_condensed,
          system_condensed_lagmult, system_condensed_lagmult, system_condensed_lagmult,
          system_saddlepoint, system_saddlepoint, system_saddlepoint, system_saddlepoint,
          system_none),
      scontact);

  scontact.specs.emplace_back(parameter<double>("PENALTYPARAM",
      {.description = "Penalty parameter for penalty / Uzawa augmented solution strategy",
          .default_value = 0.0}));
  scontact.specs.emplace_back(parameter<double>("PENALTYPARAMTAN",
      {.description =
              "Tangential penalty parameter for penalty / Uzawa augmented solution strategy",
          .default_value = 0.0}));
  scontact.specs.emplace_back(parameter<int>(
      "UZAWAMAXSTEPS", {.description = "Maximum no. of Uzawa steps for Uzawa solution strategy",
                           .default_value = 10}));
  scontact.specs.emplace_back(parameter<double>(
      "UZAWACONSTRTOL", {.description = "Tolerance of constraint norm for Uzawa solution strategy",
                            .default_value = 1.0e-8}));

  scontact.specs.emplace_back(parameter<bool>("SEMI_SMOOTH_NEWTON",
      {.description = "If chosen semi-smooth Newton concept is applied", .default_value = true}));

  scontact.specs.emplace_back(parameter<double>("SEMI_SMOOTH_CN",
      {.description = "Weighting factor cn for semi-smooth PDASS", .default_value = 1.0}));
  scontact.specs.emplace_back(parameter<double>("SEMI_SMOOTH_CT",
      {.description = "Weighting factor ct for semi-smooth PDASS", .default_value = 1.0}));

  scontact.specs.emplace_back(parameter<bool>("CONTACTFORCE_ENDTIME",
      {.description = "If chosen, the contact force is not evaluated at the generalized midpoint, "
                      "but at the end of the time step",
          .default_value = false}));

  scontact.specs.emplace_back(parameter<bool>("VELOCITY_UPDATE",
      {.description = "If chosen, velocity update method is applied", .default_value = false}));

  scontact.specs.emplace_back(parameter<bool>("INITCONTACTBYGAP",
      {.description = "Initialize init contact by weighted gap vector", .default_value = false}));

  scontact.specs.emplace_back(parameter<double>("INITCONTACTGAPVALUE",
      {.description = "Value for initialization of init contact set with gap vector",
          .default_value = 0.0}));

  // solver convergence test parameters for contact/meshtying in saddlepoint formulation
  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_RESFCONTCONSTR",
      "And", "binary operator to combine contact constraints and residual force values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), scontact);

  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_DISPLAGR", "And",
      "binary operator to combine displacement increments and Lagrange multiplier increment values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), scontact);

  scontact.specs.emplace_back(parameter<double>(
      "TOLCONTCONSTR", {.description = "tolerance in the contact constraint norm for the newton "
                                       "iteration (saddlepoint formulation only)",
                           .default_value = 1.0E-6}));
  scontact.specs.emplace_back(parameter<double>("TOLLAGR",
      {.description =
              "tolerance in the LM norm for the newton iteration (saddlepoint formulation only)",
          .default_value = 1.0E-6}));

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::ConstraintDirection>(
      "CONSTRAINT_DIRECTIONS", "ntt",
      "formulation of constraints in normal/tangential or xyz-direction",
      tuple<std::string>("ntt", "xyz"),
      tuple<Inpar::CONTACT::ConstraintDirection>(constr_ntt, constr_xyz), scontact);

  scontact.specs.emplace_back(parameter<bool>("NONSMOOTH_GEOMETRIES",
      {.description = "If chosen the contact algorithm combines mortar and nts formulations. This "
                      "is needed if contact between entities of different geometric dimension "
                      "(such as contact between surfaces and lines, or lines and nodes) can occur",
          .default_value = false}));

  scontact.specs.emplace_back(parameter<bool>("NONSMOOTH_CONTACT_SURFACE",
      {.description =
              "This flag is used to alter the criterion for the evaluation of the so-called "
              "qualified vectors in the case of a self contact scenario. This is needed as the "
              "standard criterion is only valid for smooth surfaces and thus has to be altered, if "
              "the surface that is defined to be a self contact surface is non-smooth!",
          .default_value = false}));

  scontact.specs.emplace_back(parameter<double>(
      "HYBRID_ANGLE_MIN", {.description = "Non-smooth contact: angle between cpp normal and "
                                          "element normal: begin transition (Mortar)",
                              .default_value = -1.0}));
  scontact.specs.emplace_back(parameter<double>(
      "HYBRID_ANGLE_MAX", {.description = "Non-smooth contact: angle between cpp normal and "
                                          "element normal: end transition (NTS)",
                              .default_value = -1.0}));

  scontact.specs.emplace_back(parameter<bool>("CPP_NORMALS",
      {.description = "If chosen the nodal normal field is created as averaged CPP normal field.",
          .default_value = false}));

  scontact.specs.emplace_back(parameter<bool>(
      "TIMING_DETAILS", {.description = "Enable and print detailed contact timings to screen.",
                            .default_value = false}));

  // --------------------------------------------------------------------------
  scontact.specs.emplace_back(parameter<double>(
      "NITSCHE_THETA", {.description = "+1: symmetric, 0: non-symmetric, -1: skew-symmetric",
                           .default_value = 0.0}));
  scontact.specs.emplace_back(parameter<double>("NITSCHE_THETA_2",
      {.description = "+1: Chouly-type, 0: Burman penalty-free (only with theta=-1)",
          .default_value = 1.0}));

  Core::Utils::string_to_integral_parameter<Inpar::CONTACT::NitscheWeighting>("NITSCHE_WEIGHTING",
      "harmonic", "how to weight consistency terms in Nitsche contact formulation",
      tuple<std::string>("slave", "master", "harmonic"),
      tuple<Inpar::CONTACT::NitscheWeighting>(NitWgt_slave, NitWgt_master, NitWgt_harmonic),
      scontact);

  scontact.specs.emplace_back(parameter<bool>("NITSCHE_PENALTY_ADAPTIVE",
      {.description = "adapt penalty parameter after each converged time step",
          .default_value = true}));

  scontact.specs.emplace_back(parameter<bool>("REGULARIZED_NORMAL_CONTACT",
      {.description = "add a regularized normal contact formulation", .default_value = false}));
  scontact.specs.emplace_back(parameter<double>("REGULARIZATION_THICKNESS",
      {.description = "maximum contact penetration", .default_value = -1.}));
  Core::Utils::double_parameter("REGULARIZATION_STIFFNESS", -1.,
      "initial contact stiffness (i.e. initial \"penalty parameter\")", scontact);

  scontact.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
