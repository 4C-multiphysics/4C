// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_fld_fluid_ale.hpp"

#include "4C_adapter_ale_fluid.hpp"
#include "4C_ale_input.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_dirichletneumann_volcoupl.hpp"
#include "4C_fsi_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Adapter::FluidAle::FluidAle(
    Global::Problem& problem, const Teuchos::ParameterList& prbdyn, std::string condname)
    : problem_(problem), timeparams_(prbdyn)
{
  Adapter::FluidBaseAlgorithm fluid(
      problem_, prbdyn, problem_.fluid_dynamic_params(), "fluid", true, false);
  fluid_ = fluid.fluid_field();
  std::shared_ptr<Adapter::AleBaseAlgorithm> ale =
      std::make_shared<Adapter::AleBaseAlgorithm>(problem_, prbdyn, problem_.get_dis("ale"));
  ale_ = std::dynamic_pointer_cast<Adapter::AleFluidWrapper>(ale->ale_field());

  if (ale_ == nullptr) FOUR_C_THROW("Failed to cast to problem-specific ALE-wrapper");

  const int ndim = problem_.n_dim();

  // default parameters for coupling
  double tolerance = 1.e-3;
  int target_dofset_number = 0;
  int source_dofset_number = 0;

  // set target_dofset_number = 2 in case of HDG discretization
  // (nds = 0 used for trace values, nds = 1 used for interior values)
  if (problem_.spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg)
  {
    target_dofset_number = 2;
  }

  // check for matching fluid and ale meshes (==true in default case)
  if (problem_.fsi_dynamic_params().get<bool>("MATCHGRID_FLUIDALE"))
  {
    // the fluid-ale coupling matches
    const Core::LinAlg::Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
    const Core::LinAlg::Map* alenodemap = ale_field()->discretization()->node_row_map();

    /* Setup coupling adapter
     *
     * Since ALE has been cloned form fluid discretization, nodes reside at the
     * exact same location. Thus, we specify a very tight tolerance for the
     * octree search.
     */
    std::shared_ptr<Coupling::Adapter::Coupling> coupfa_matching =
        std::make_shared<Coupling::Adapter::Coupling>();
    coupfa_matching->setup_coupling(*fluid_field()->discretization(),
        *ale_field()->discretization(), *fluidnodemap, *alenodemap, ndim,
        problem_.fsi_dynamic_params().get<bool>("MATCHALL"), tolerance, target_dofset_number,
        source_dofset_number);
    coupfa_ = coupfa_matching;
  }
  else
  {
    // non matching volume meshes of fluid and ale
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> coupfa_volmortar =
        std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    // couple displacement dofs of ale and velocity dofs of fluid

    // projection ale -> fluid : all ndim dofs (displacements)
    std::vector<int> coupleddof12 = std::vector<int>(ndim, 1);

    // projection fluid -> ale : ndim dofs (only velocity, no pressure)
    std::vector<int> coupleddof21 = std::vector<int>(ndim + 1, 1);
    // unmark pressure dof
    coupleddof21[ndim] = 0;

    // define dof sets to be coupled for both projections
    std::pair<int, int> dofsets12(0, 0);
    std::pair<int, int> dofsets21(0, 0);

    // initialize coupling adapter
    coupfa_volmortar->init(ndim, fluid_field()->discretization(),
        ale_field()->write_access_discretization(), &coupleddof12, &coupleddof21, &dofsets12,
        &dofsets21, nullptr, false);

    // setup coupling adapter
    coupfa_volmortar->setup(problem_.volmortar_params(), problem_.cut_general_params());

    // set pointer to coupling adapter
    coupfa_ = coupfa_volmortar;
  }

  // Apply initial ALE mesh displacement
  if (Teuchos::getIntegralValue<ALE::InitialDisp>(problem_.ale_dynamic_params(), "INITIALDISP") !=
      ALE::initdisp_zero_disp)
  {
    fluid_field()->set_mesh_map(coupfa_->target_dof_map(), target_dofset_number);
    std::shared_ptr<Core::LinAlg::Vector<double>> initfluiddisp =
        ale_to_fluid_field(ale_field()->dispn());
    fluid_field()->apply_initial_mesh_displacement(initfluiddisp);
  }

  // initializing the fluid is done later as for xfluids the first cut is done
  // there (coupfa_ cannot be build anymore!!!)
  fluid_field()->init();
  fluid.set_initial_flow_field(problem_.fluid_dynamic_params());  // call from base algorithm


  if (problem_.fsi_dynamic_params().get<bool>("MATCHGRID_STRUCTALE"))
  {
    std::shared_ptr<Coupling::Adapter::Coupling> icoupfa =
        std::make_shared<Coupling::Adapter::Coupling>();
    icoupfa->setup_condition_coupling(*fluid_field()->discretization(),
        fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
        ale_field()->interface()->fsi_cond_map(), condname, ndim, true, target_dofset_number,
        source_dofset_number);
    icoupfa_ = icoupfa;
  }
  else
  {
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> icoupfa =
        std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    // couple displacement dofs of ale and velocity dofs of fluid

    // projection ale -> fluid : all ndim dofs (displacements)
    std::vector<int> coupleddof12 = std::vector<int>(ndim, 1);

    // projection fluid -> ale : ndim dofs (only velocity, no pressure)
    std::vector<int> coupleddof21 = std::vector<int>(ndim + 1, 1);
    // unmark pressure dof
    coupleddof21[ndim] = 0;

    // define dof sets to be coupled for both projections
    std::pair<int, int> dofsets12(0, 0);
    std::pair<int, int> dofsets21(0, 0);

    icoupfa->init(ndim, problem_.get_dis("fluid"), problem_.get_dis("ale"), &coupleddof12,
        &coupleddof21, &dofsets12, &dofsets21, nullptr, false);

    icoupfa->setup(problem_.volmortar_params(), problem_.cut_general_params());

    icoupfa_ = icoupfa;
  }

  aucoupfa_ = std::make_shared<Coupling::Adapter::Coupling>();
  aucoupfa_->setup_condition_coupling(*fluid_field()->discretization(),
      fluid_field()->interface()->au_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->au_cond_map(), "ALEUPDATECoupling", ndim, true,
      target_dofset_number, source_dofset_number);

  fluid_field()->set_mesh_map(coupfa_->target_dof_map(), target_dofset_number);

  // the ale matrix might be build just once
  ale_field()->create_system_matrix();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization> Adapter::FluidAle::discretization()
{
  return fluid_field()->discretization();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::prepare_time_step()
{
  fluid_field()->prepare_time_step();
  ale_field()->prepare_time_step();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::update()
{
  fluid_field()->update();
  ale_field()->update();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::output()
{
  fluid_field()->statistics_and_output();

  // Note: We want to write the fsi interface tractions in order to restart
  // monolithically from an partitioned fsi scheme (e.g. fsi prestress simulation).
  // TODO (Thon): this is not the nice way, but fluid-ale and xfem problems may have now FSI
  // interface, so we can not do this in general :(
  if (problem_.get_problem_type() == Core::ProblemType::fsi)
  {
    // we want to be able to restart monolithically from an partitioned fsi scheme
    const int uprestart = timeparams_.get<int>("RESTARTEVERY");
    const int upres = timeparams_.get<int>("RESULTSEVERY");

    if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
        fluid_field()->step() % upres == 0)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
          fluid_field()->extract_interface_forces();
      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
          fluid_field()->interface()->insert_fsi_cond_vector(*lambda);
      fluid_field()->disc_writer()->write_vector("fsilambda", lambdafull);
    }
  }

  ale_field()->output();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double Adapter::FluidAle::read_restart(int step)
{
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);
  return fluid_field()->time();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::nonlinear_solve(std::shared_ptr<Core::LinAlg::Vector<double>> idisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel)
{
  if (idisp != nullptr)
  {
    // if we have values at the interface we need to apply them
    ale_field()->apply_interface_displacements(fluid_to_ale(idisp));
    fluid_field()->apply_interface_velocities(ivel);
  }

  // Update the ale update part
  if (fluid_field()->interface()->au_cond_relevant())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = fluid_field()->dispnp();
    std::shared_ptr<Core::LinAlg::Vector<double>> audispnp =
        fluid_field()->interface()->extract_au_cond_vector(*dispnp);
    ale_field()->apply_ale_update_displacements(aucoupfa_->target_to_source(*audispnp));
  }

  // Note: We do not look for moving ale boundaries (outside the coupling
  // interface) on the fluid side. Thus if you prescribe time variable ale
  // Dirichlet conditions the according fluid Dirichlet conditions will not
  // notice.

  ale_field()->solve();
  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp =
      ale_to_fluid_field(ale_field()->dispnp());
  fluid_field()->apply_mesh_displacement(fluiddisp);
  fluid_field()->solve();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::nonlinear_solve_vol_coupl(
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel,
    std::shared_ptr<FSI::InterfaceCorrector> icorrector)
{
  if (idisp != nullptr)
  {
    ale_field()->apply_interface_displacements(
        ale_field()->interface()->extract_fsi_cond_vector(*idisp));
    fluid_field()->apply_interface_velocities(ivel);
  }

  // Update the ale update part
  if (fluid_field()->interface()->au_cond_relevant())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = fluid_field()->dispnp();
    std::shared_ptr<Core::LinAlg::Vector<double>> audispnp =
        fluid_field()->interface()->extract_au_cond_vector(*dispnp);
    ale_field()->apply_ale_update_displacements(aucoupfa_->target_to_source(*audispnp));
  }

  // Note: We do not look for moving ale boundaries (outside the coupling
  // interface) on the fluid side. Thus if you prescribe time variable ale
  // Dirichlet conditions the according fluid Dirichlet conditions will not
  // notice.

  ale_field()->solve();
  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp =
      ale_to_fluid_field(ale_field()->dispnp());

  icorrector->correct_interface_displacements(fluiddisp, fluid_field()->interface());
  fluid_field()->apply_mesh_displacement(fluiddisp);
  fluid_field()->solve();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::FluidAle::apply_interface_values(std::shared_ptr<Core::LinAlg::Vector<double>> idisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel)
{
  if (idisp != nullptr)
  {
    // if we have values at the interface we need to apply them
    ale_field()->apply_interface_displacements(fluid_to_ale(idisp));
    fluid_field()->apply_interface_velocities(ivel);
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp =
      ale_to_fluid_field(ale_field()->dispnp());
  fluid_field()->apply_mesh_displacement(fluiddisp);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::relaxation_solve(
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp, double dt)
{
  // Here we have a mesh position independent of the
  // given trial vector, but still the grid velocity depends on the
  // trial vector only.

  // grid velocity
  ale_field()->apply_interface_displacements(fluid_to_ale(idisp));

  ale_field()->solve();
  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp =
      ale_to_fluid_field(ale_field()->dispnp());
  fluiddisp->scale(1. / dt);

  fluid_field()->apply_mesh_velocity(fluiddisp);

  // grid position is done inside RelaxationSolve

  // the displacement -> velocity conversion at the interface
  idisp->scale(1. / dt);

  return fluid_field()->relaxation_solve(idisp);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::extract_interface_forces()
{
  return fluid_field()->extract_interface_forces();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::extract_interface_velnp()
{
  return fluid_field()->extract_interface_velnp();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::extract_interface_veln()
{
  return fluid_field()->extract_interface_veln();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::integrate_interface_shape()
{
  return fluid_field()->integrate_interface_shape();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Adapter::FluidAle::create_field_test()
{
  return fluid_field()->create_field_test();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::ale_to_fluid_field(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_->source_to_target(*iv);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::ale_to_fluid_field(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_->source_to_target(*iv);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::fluid_to_ale(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv) const
{
  return icoupfa_->target_to_source(*iv);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidAle::fluid_to_ale(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return icoupfa_->target_to_source(*iv);
}

FOUR_C_NAMESPACE_CLOSE
