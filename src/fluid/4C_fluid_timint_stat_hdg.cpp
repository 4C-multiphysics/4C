// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_stat_hdg.hpp"

#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_hdg.hpp"
#include "4C_fluid_ele_hdg_weak_comp.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_volumetric_surfaceFlow_condition.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                      also 01/18 |    // TODO also fix
 fluid_timint_stat_hdg because it is not working
 *----------------------------------------------------------------------*/
FLD::TimIntStationaryHDG::TimIntStationaryHDG(
    const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      TimIntStationary(actdis, solver, params, output, alefluid),
      first_assembly_(false)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                      also 01/18 |
 *----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::init()
{
  Core::FE::DiscretizationHDG* hdgdis = dynamic_cast<Core::FE::DiscretizationHDG*>(discret_.get());
  if (hdgdis == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  int elementndof = hdgdis->num_my_row_elements() > 0
                        ? dynamic_cast<Discret::Elements::FluidHDG*>(hdgdis->l_row_element(0))
                              ->num_dof_per_element_auxiliary()
                        : 0;

  // set degrees of freedom in the discretization
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, elementndof, 0, false);
  discret_->add_dof_set(dofsetaux);
  discret_->fill_complete();

  // build velocity/pressure splitting
  std::set<int> conddofset;
  std::set<int> otherdofset;

  for (int j = 0; j < hdgdis->num_my_row_elements(); ++j)
  {
    std::vector<int> dof = hdgdis->dof(0, hdgdis->l_row_element(j));
    FOUR_C_ASSERT(dof.size() >= 1, "Internal error: could not find HDG pressure dof");
    for (unsigned int i = 0; i < dof.size(); ++i) conddofset.insert(dof[i]);
  }
  for (int i = 0; i < hdgdis->num_my_row_faces(); ++i)
  {
    std::vector<int> dof = hdgdis->dof(0, hdgdis->l_row_face(i));
    for (unsigned int j = 0; j < dof.size(); ++j) otherdofset.insert(dof[j]);
  }

  std::vector<int> conddofmapvec;
  conddofmapvec.reserve(conddofset.size());
  conddofmapvec.assign(conddofset.begin(), conddofset.end());
  conddofset.clear();
  std::shared_ptr<Core::LinAlg::Map> conddofmap = std::make_shared<Core::LinAlg::Map>(
      -1, conddofmapvec.size(), conddofmapvec.data(), 0, hdgdis->get_comm());
  std::vector<int> otherdofmapvec;
  otherdofmapvec.reserve(otherdofset.size());
  otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
  otherdofset.clear();
  std::shared_ptr<Core::LinAlg::Map> otherdofmap = std::make_shared<Core::LinAlg::Map>(
      -1, otherdofmapvec.size(), otherdofmapvec.data(), 0, hdgdis->get_comm());
  velpressplitter_->setup(*hdgdis->dof_row_map(), conddofmap, otherdofmap);

  // call init()-functions of base classes
  // note: this order is important
  FLD::TimIntStationary::init();
}


void FLD::TimIntStationaryHDG::reset(int numsteps, int iter)
{
  FluidImplicitTimeInt::reset(numsteps, iter);
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(1);
  intvelnp_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "Number of degrees of freedom in HDG system: "
              << discret_->dof_row_map(0)->num_global_elements() << std::endl;
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                       also 01/18 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::set_custom_ele_params_assemble_mat_and_rhs(
    Teuchos::ParameterList& eleparams)
{
  eleparams.set<bool>("needslocalupdate", !first_assembly_);
}


/*----------------------------------------------------------------------*
| set old part of right hand side                             also 01/18 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::set_old_part_of_righthandside()
{
  /*
     Stationary:

                   mom: hist_ = 0.0
                  (con: hist_ = 0.0)
  */

  hist_->put_scalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  first_assembly_ = true;
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                       also 01/18 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::set_state_tim_int()
{
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(1);
  Core::LinAlg::Vector<double> zerovec(*intdofrowmap, true);

  discret_->set_state(0, "velaf", *velnp_);
  discret_->set_state(1, "intvelaf", *intvelnp_);  // TODO also fill in intvelnp_!
  discret_->set_state(1, "intaccam", zerovec);     // TODO memory management working?
  discret_->set_state(1, "intvelnp", *intvelnp_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                       also 01/18 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::clear_state_assemble_mat_and_rhs()
{
  if (!first_assembly_)
  {
    // Wrote into the state vector during element calls, need to transfer the
    // data back before it disappears when clearing the state (at least for nproc>1)
    const Core::LinAlg::Vector<double>& intvelnpGhosted = *discret_->get_state(1, "intvelnp");
    for (int i = 0; i < intvelnp_->local_length(); ++i)
      (*intvelnp_).get_values()[i] =
          intvelnpGhosted
              .local_values_as_span()[intvelnpGhosted.get_map().lid(intvelnp_->get_map().gid(i))];
  }
  first_assembly_ = false;
  FluidImplicitTimeInt::clear_state_assemble_mat_and_rhs();
}

/*----------------------------------------------------------------------*
 |  set initial flow field for test cases              kronbichler 05/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntStationaryHDG::set_initial_flow_field(
    const Inpar::FLUID::InitialField initfield, const int startfuncno)
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  Core::LinAlg::SerialDenseVector elevec1, elevec2, elevec3;
  Core::LinAlg::SerialDenseMatrix elemat1, elemat2;
  Teuchos::ParameterList initParams;
  initParams.set<FLD::Action>("action", FLD::project_fluid_field);
  initParams.set("startfuncno", startfuncno);
  initParams.set<Inpar::FLUID::InitialField>("initfield", initfield);
  // loop over all elements on the processor
  Core::Elements::LocationArray la(2);
  double error = 0;
  for (int el = 0; el < discret_->num_my_col_elements(); ++el)
  {
    Core::Elements::Element* ele = discret_->l_col_element(el);

    ele->location_vector(*discret_, la);
    if (static_cast<std::size_t>(elevec1.numRows()) != la[0].lm_.size())
      elevec1.size(la[0].lm_.size());
    if (elevec2.numRows() != discret_->num_dof(1, ele)) elevec2.size(discret_->num_dof(1, ele));

    ele->evaluate(initParams, *discret_, la[0].lm_, elemat1, elemat2, elevec1, elevec2, elevec3);

    // now fill the ele vector into the discretization
    for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
    {
      const int lid = dofrowmap->lid(la[0].lm_[i]);
      if (lid >= 0)
      {
        if (velnp_->local_values_as_span()[lid] != 0)
          error += std::abs(velnp_->local_values_as_span()[lid] - elevec1(i));
        (*velnp_).get_values()[lid] = elevec1(i);
        (*veln_).get_values()[lid] = elevec1(i);
        (*velnm_).get_values()[lid] = elevec1(i);
      }
    }

    if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
    {
      std::vector<int> globalDofs = discret_->dof(1, ele);
      FOUR_C_ASSERT(
          globalDofs.size() == static_cast<std::size_t>(elevec2.numRows()), "Internal error");
      intvelnp_->replace_global_values(globalDofs.size(), elevec2.values(), globalDofs.data());
    }
  }
  double globerror = 0;
  globerror = Core::Communication::sum_all(error, discret_->get_comm());
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "Error project when setting face twice: " << globerror << std::endl;
}


// -------------------------------------------------------------------
// set general time parameter (AE 01/2011)
// -------------------------------------------------------------------
void FLD::TimIntStationaryHDG::set_element_time_parameter()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // set time integration scheme
  eleparams.set<Inpar::FLUID::TimeIntegrationScheme>("TimeIntegrationScheme", timealgo_);

  // set general element parameters
  eleparams.set("dt", dta_);
  eleparams.set("theta", theta_);
  eleparams.set("omtheta", 0.0);

  // set scheme-specific element parameters and vector values
  eleparams.set("total time", time_);

  Discret::Elements::FluidEleParameterTimInt::instance()->set_element_time_parameter(eleparams);
}

FOUR_C_NAMESPACE_CLOSE
