// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_loma.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 12/13 |
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntLoma::ScaTraTimIntLoma(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(dis, solver, sctratimintparams, extraparams, output),
      lomaparams_(params),
      initialmass_(0.0),
      thermpressn_(0.0),
      thermpressnp_(0.0),
      thermpressdtn_(0.0),
      thermpressdtnp_(0.0)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 | initialize algorithm                                     rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::init()
{
  // safety check
  if (lomaparams_->get<bool>("SGS_MATERIAL_UPDATE"))
    FOUR_C_THROW(
        "Material update using subgrid-scale temperature currently not supported for loMa "
        "problems. Read remark in file 'scatra_ele_calc_loma.H'!");

  return;
}


/*----------------------------------------------------------------------*
 | setup algorithm                                          rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::setup()
{
  setup_splitter();
  return;
}

/*----------------------------------------------------------------------*
 | setup splitter                                          deanda 11/17 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::setup_splitter()
{
  // set up a species-temperature splitter (if more than one scalar)
  if (num_scal() > 1)
  {
    splitter_ = std::make_shared<Core::LinAlg::MapExtractor>();
    Core::LinAlg::create_map_extractor_from_discretization(*discret_, num_scal() - 1, *splitter_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | set initial thermodynamic pressure                          vg 07/09 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::set_initial_therm_pressure()
{
  // get thermodynamic pressure from material parameters
  int id = problem_->materials()->first_id_by_type(Core::Materials::m_sutherland);
  if (id != -1)  // i.e., Sutherland material found
  {
    const Core::Mat::PAR::Parameter* mat = problem_->materials()->parameter_by_id(id);
    const Mat::PAR::Sutherland* actmat = static_cast<const Mat::PAR::Sutherland*>(mat);

    thermpressn_ = actmat->thermpress_;
  }
  else
  {
    FOUR_C_THROW(
        "No Sutherland material found for initial setting of "
        "thermodynamic pressure!");
  }

  // initialize also value at n+1
  // (computed if not constant, otherwise prescribed value remaining)
  thermpressnp_ = thermpressn_;

  // initialize time derivative of thermodynamic pressure at n+1 and n
  // (computed if not constant, otherwise remaining zero)
  thermpressdtnp_ = 0.0;
  thermpressdtn_ = 0.0;

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  // -> For constant thermodynamic pressure, this is done here once and
  // for all simulation time.
  compute_therm_pressure_intermediate_values();

  return;
}  // ScaTra::ScaTraTimIntLoma::set_initial_therm_pressure


/*----------------------------------------------------------------------*
 | compute initial total mass in domain                        vg 01/09 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::compute_initial_mass()
{
  // set scalar values needed by elements
  discret_->clear_state();
  discret_->set_state("phinp", *phin_);
  // set action for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  // inverted scalar values are required here
  eleparams.set("inverting", true);
  eleparams.set("calc_grad_phi", false);

  // evaluate integral of inverse temperature
  std::shared_ptr<Core::LinAlg::SerialDenseVector> scalars =
      std::make_shared<Core::LinAlg::SerialDenseVector>(num_scal() + 1);
  discret_->evaluate_scalars(eleparams, *scalars);
  discret_->clear_state();  // clean up

  // compute initial mass times gas constant: R*M_0 = int(1/T_0)*tp
  initialmass_ = (*scalars)[0] * thermpressn_;

  // print out initial total mass
  if (myrank_ == 0)
  {
    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
    std::cout << "Initial total mass in domain (times gas constant): " << initialmass_ << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
  }

  return;
}  // ScaTra::ScaTraTimIntLoma::ComputeInitialMass


/*----------------------------------------------------------------------*
 | compute thermodynamic pressure from mass conservation       vg 01/09 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::compute_therm_pressure_from_mass_cons()
{
  // set scalar values needed by elements
  discret_->clear_state();
  discret_->set_state("phinp", *phinp_);
  // set action for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  // inverted scalar values are required here
  eleparams.set("inverting", true);
  eleparams.set("calc_grad_phi", false);

  // evaluate integral of inverse temperature
  std::shared_ptr<Core::LinAlg::SerialDenseVector> scalars =
      std::make_shared<Core::LinAlg::SerialDenseVector>(num_scal() + 1);
  discret_->evaluate_scalars(eleparams, *scalars);
  discret_->clear_state();  // clean up

  // compute thermodynamic pressure: tp = R*M_0/int(1/T)
  thermpressnp_ = initialmass_ / (*scalars)[0];

  // print out thermodynamic pressure
  if (myrank_ == 0)
  {
    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
    std::cout << "Thermodynamic pressure from mass conservation: " << thermpressnp_ << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
  }

  // compute time derivative of thermodynamic pressure at time step n+1
  compute_therm_pressure_time_derivative();

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  compute_therm_pressure_intermediate_values();

  return;
}  // ScaTra::ScaTraTimIntLoma::compute_therm_pressure_from_mass_cons


/*----------------------------------------------------------------------*
 | add parameters depending on the problem              rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntLoma::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params  //!< parameter list
)
{
  add_therm_press_to_parameter_list(params);
  return;
}

FOUR_C_NAMESPACE_CLOSE
