// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_mirco_contactconstitutivelaw.hpp"

#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_par_bundle.hpp"

#ifdef FOUR_C_WITH_MIRCO

#include <mirco_evaluate.h>
#include <mirco_kokkostypes.h>
#include <mirco_topology.h>
#include <mirco_topologyutilities.h>

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams::MircoConstitutiveLawParams(
    const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container),
      firstmatid_(container.get<int>("FirstMatID")),
      secondmatid_(container.get<int>("SecondMatID")),
      lateral_length_(container.get<double>("LateralLength")),
      resolution_(container.get<int>("Resolution")),
      pressure_green_fun_flag_(container.get<bool>("PressureGreenFunFlag")),
      random_topology_flag_(container.get<bool>("RandomTopologyFlag")),
      random_seed_flag_(container.get<bool>("RandomSeedFlag")),
      random_generator_seed_(container.get<int>("RandomGeneratorSeed")),
      tolerance_(container.get<double>("Tolerance")),
      max_iteration_(container.get<int>("MaxIteration")),
      warm_starting_flag_(container.get<bool>("WarmStartingFlag")),
      finite_difference_fraction_(container.get<double>("FiniteDifferenceFraction")),
      active_gap_tolerance_(container.get<double>("ActiveGapTolerance")),
      topology_file_path_((container.get<std::string>("TopologyFilePath")))
{
  this->set_parameters();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::MircoConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params)
    : params_(std::move(params))
{
}

void CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams::set_parameters()
{
  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW(
        "An attempt to access the list of materials in the instance of the global problem returned "
        "a null pointer.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  const auto& firstmat = Global::Problem::instance(probinst)
                             ->materials()
                             ->parameter_by_id(get_first_mat_id())
                             ->raw_parameters();
  const auto& secondmat = Global::Problem::instance(probinst)
                              ->materials()
                              ->parameter_by_id(get_second_mat_id())
                              ->raw_parameters();

  const double E1 = firstmat.get<double>("YOUNG");
  const double E2 = secondmat.get<double>("YOUNG");
  const double nu1 = firstmat.get<double>("NUE");
  const double nu2 = secondmat.get<double>("NUE");

  // Composite Young's modulus
  composite_youngs_ = pow(((1 - pow(nu1, 2)) / E1 + (1 - pow(nu2, 2)) / E2), -1);

  double ngrid = (pow(2, resolution_) + 1);
  grid_size_ = lateral_length_ / ngrid;

  // Shape factors (See section 3.3 of https://doi.org/10.1007/s00466-019-01791-3)
  // These are the shape factors to calculate the elastic compliance correction of the micro-scale
  // contact constitutive law for various resolutions.
  // NOTE: Currently MIRCO works for resouluion of 1 to 8. The following map store the shape
  // factors for resolution of 1 to 8.

  // The following pressure based constants are calculated by solving a flat indentor problem in
  // MIRCO using the pressure based Green function described in Pohrt and Li (2014).
  // http://dx.doi.org/10.1134/s1029959914040109
  const std::map<int, double> shape_factors_pressure{{1, 0.961389237917602}, {2, 0.924715342432435},
      {3, 0.899837531880697}, {4, 0.884976751041942}, {5, 0.876753783192863},
      {6, 0.872397956576882}, {7, 0.8701463093314326}, {8, 0.8689982669426167}};

  // The following force based constants are taken from Table 1 of Bonari et al. (2020).
  // https://doi.org/10.1007/s00466-019-01791-3
  const std::map<int, double> shape_factors_force{{1, 0.778958541513360}, {2, 0.805513388666376},
      {3, 0.826126871395416}, {4, 0.841369158110513}, {5, 0.851733020725652},
      {6, 0.858342234203154}, {7, 0.862368243479785}, {8, 0.864741597831785}};

  const double ShapeFactor = pressure_green_fun_flag_ ? shape_factors_pressure.at(resolution_)
                                                      : shape_factors_force.at(resolution_);

  elastic_compliance_correction_ = lateral_length_ * composite_youngs_ / ShapeFactor;

  meshgrid_ = MIRCO::CreateMeshgrid(ngrid, grid_size_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes. ");
  }
  if (-(gap + params_.get_offset()) < params_.get_active_gap_tolerance())
  {
    return 0.0;
  }

  const RoughNode* roughNode = dynamic_cast<const RoughNode*>(cnode);
  auto topology = *roughNode->get_topology();

  double pressure = 0.0;
  double contact_area_fraction = 0.0;
  MIRCO::Evaluate(pressure, contact_area_fraction, -(gap + params_.get_offset()),
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());

  return (-1 * pressure);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes.");
  }
  if (-(gap + params_.get_offset()) < params_.get_active_gap_tolerance())
  {
    return 0.0;
  }

  const RoughNode* roughNode = dynamic_cast<const RoughNode*>(cnode);
  auto topology = *roughNode->get_topology();

  double pressure1 = 0.0;
  double pressure2 = 0.0;
  double contact_area_fraction = 0.0;
  // using backward difference approach
  MIRCO::Evaluate(pressure1, contact_area_fraction, -1.0 * (gap + params_.get_offset()),
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());
  MIRCO::Evaluate(pressure2, contact_area_fraction,
      -(1 - params_.get_finite_difference_fraction()) * (gap + params_.get_offset()),
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());
  return ((pressure1 - pressure2) /
          (-(params_.get_finite_difference_fraction()) * (gap + params_.get_offset())));
}

FOUR_C_NAMESPACE_CLOSE

#endif
