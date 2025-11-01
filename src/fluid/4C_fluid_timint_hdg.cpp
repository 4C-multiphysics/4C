// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_hdg.hpp"

#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_hdg.hpp"
#include "4C_fluid_ele_hdg_weak_comp.hpp"
#include "4C_fluid_turbulence_hit_forcing.hpp"
#include "4C_fluid_turbulence_hit_initial_field.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
FLD::TimIntHDG::TimIntHDG(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      TimIntGenAlpha(actdis, solver, params, output, alefluid),
      timealgoset_(Inpar::FLUID::timeint_afgenalpha),
      first_assembly_(false)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::init()
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

  // implement ost and bdf2 through gen-alpha facilities
  if (timealgo_ == Inpar::FLUID::timeint_bdf2)
  {
    alphaM_ = 1.5;
    alphaF_ = 1.0;
    gamma_ = 1.0;
  }
  else if (timealgo_ == Inpar::FLUID::timeint_one_step_theta)
  {
    alphaM_ = 1.0;
    alphaF_ = 1.0;
    gamma_ = params_->get<double>("theta");
  }
  else if (timealgo_ == Inpar::FLUID::timeint_stationary)
    FOUR_C_THROW("Stationary case not implemented for HDG");

  timealgoset_ = timealgo_;
  timealgo_ = Inpar::FLUID::timeint_afgenalpha;

  // call init()-functions of base classes
  // note: this order is important
  FLD::TimIntGenAlpha::init();
}



/*----------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_, modified for HDG  kronbi 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::set_theta()
{
  // -------------------------------------------------------------------
  //  For af-generalized-alpha time-integration scheme:
  //  set "pseudo-theta", calculate initial accelerations according to
  //  prescribed Dirichlet values for generalized-alpha time
  //  integration and values at intermediate time steps
  // -------------------------------------------------------------------
  // starting algorithm
  if (startalgo_ || (step_ <= 2 && timealgoset_ == Inpar::FLUID::timeint_bdf2))
  {
    // use backward-Euler-type parameter combination
    if (step_ <= numstasteps_ || (step_ <= 1 && timealgoset_ == Inpar::FLUID::timeint_bdf2))
    {
      if (myrank_ == 0)
      {
        std::cout << "Starting algorithm for Af_GenAlpha active. "
                  << "Performing step " << step_ << " of " << numstasteps_
                  << " Backward Euler starting steps" << std::endl;
      }
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_ = 1.0;
    }
    else
    {
      // recall original user wish
      if (timealgoset_ == Inpar::FLUID::timeint_one_step_theta)
      {
        alphaM_ = alphaF_ = 1.0;
        gamma_ = params_->get<double>("theta");
      }
      else if (timealgoset_ == Inpar::FLUID::timeint_bdf2)
      {
        alphaF_ = gamma_ = 1.0;
        alphaM_ = 3. / 2.;
      }
      else
      {
        alphaM_ = params_->get<double>("alpha_M");
        alphaF_ = params_->get<double>("alpha_F");
        gamma_ = params_->get<double>("gamma");
      }

      // do not enter starting algorithm section in the future
      startalgo_ = false;
    }
  }

  // compute "pseudo-theta" for af-generalized-alpha scheme
  theta_ = alphaF_ * gamma_ / alphaM_;
}


/*----------------------------------------------------------------------*
 * Explicit predictor                                 kronbichler 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::explicit_predictor()
{
  if (predictor_ == "steady_state")
  {
    // this has already been done in TimeUpdate()
  }
  else if (predictor_ == "zero_acceleration")
  {
    velnp_->update(1.0, *veln_, (1.0 - theta_) * dta_, *accn_, 0.0);
    intvelnp_->update(1.0, *intveln_, (1.0 - theta_) * dta_, *intaccn_, 0.0);
  }
  else if (predictor_ == "constant_acceleration")
  {
    velnp_->update(1.0, *veln_, dta_, *accn_, 0.0);
    intvelnp_->update(1.0, *intveln_, dta_, *intaccn_, 0.0);
  }
  else if (predictor_ == "constant_increment")
  {
    velnp_->update(2.0, *veln_, -1.0, *velnm_, 0.0);
    intvelnp_->update(2.0, *intveln_, -1.0, *intvelnm_, 0.0);
  }
  else if (predictor_ == "explicit_second_order_midpoint")
  {
    velnp_->update(1.0, *velnm_, 2.0 * dta_, *accn_, 0.0);
    intvelnp_->update(1.0, *intvelnm_, 2.0 * dta_, *intaccn_, 0.0);
  }
  else
    FOUR_C_THROW("Unknown fluid predictor {}", predictor_);
}

/*----------------------------------------------------------------------*
| set old part of right hand side                     kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::set_old_part_of_righthandside()
{
  hist_->put_scalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  first_assembly_ = true;
}



/*----------------------------------------------------------------------*
 | update acceleration for generalized-alpha time integration kro 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::gen_alpha_update_acceleration()
{
  //                                  n+1     n
  //                               vel   - vel
  //       n+1      n  gamma-1.0      (i)
  //    acc    = acc * --------- + ------------
  //       (i)           gamma      gamma * dt
  //

  // as opposed to standard fluid, update all variables including pressure

  // compute factors
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);

  accnp_->update(fact2, *accn_, 0.0);
  accnp_->update(fact1, *velnp_, -fact1, *veln_, 1.0);

  intaccnp_->update(fact2, *intaccn_, 0.0);
  intaccnp_->update(fact1, *intvelnp_, -fact1, *intveln_, 1.0);
}



/*----------------------------------------------------------------------*
 | compute values at intermediate time steps for gen.-alpha  kron 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::gen_alpha_intermediate_values()
{
  // set intermediate values for accelerations
  //
  //       n+alphaM                n+1                      n
  //    acc         = alpha_M * acc     + (1-alpha_M) *  acc
  //       (i)                     (i)
  accam_->update((alphaM_), *accnp_, (1.0 - alphaM_), *accn_, 0.0);
  intaccam_->update((alphaM_), *intaccnp_, (1.0 - alphaM_), *intaccn_, 0.0);

  // set intermediate values for velocity, pressure, velocity gradient
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  //
  // note that its af-genalpha with mid-point treatment of the pressure,
  // not implicit treatment as for the genalpha according to Whiting
  velaf_->update((alphaF_), *velnp_, (1.0 - alphaF_), *veln_, 0.0);
  intvelaf_->update((alphaF_), *intvelnp_, (1.0 - alphaF_), *intveln_, 0.0);
}


/*----------------------------------------------------------------------*
| set HDG state vectors                               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::set_state_tim_int()
{
  discret_->set_state(0, "velaf", *velaf_);
  discret_->set_state(1, "intvelaf", *intvelaf_);
  discret_->set_state(1, "intaccam", *intaccam_);
  discret_->set_state(1, "intvelnp", *intvelnp_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::set_custom_ele_params_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams)
{
  eleparams.set<bool>("needslocalupdate", !first_assembly_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::clear_state_assemble_mat_and_rhs()
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
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::time_update()
{
  FluidImplicitTimeInt::time_update();

  // velocities/pressures of this step become most recent
  // velocities/pressures of the last step
  intvelnm_->update(1.0, *intveln_, 0.0);
  intveln_->update(1.0, *intvelnp_, 0.0);

  intaccnm_->update(1.0, *intaccn_, 0.0);
  intaccn_->update(1.0, *intaccnp_, 0.0);
}



/*----------------------------------------------------------------------*
 |  set initial flow field for test cases              kronbichler 05/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::set_initial_flow_field(
    const Inpar::FLUID::InitialField initfield, const int startfuncno)
{
  if (initfield == Inpar::FLUID::initfield_hit_comte_bellot_corrsin or
      initfield == Inpar::FLUID::initfield_forced_hit_simple_algebraic_spectrum or
      initfield == Inpar::FLUID::initfield_forced_hit_numeric_spectrum or
      initfield == Inpar::FLUID::initfield_passive_hit_const_input)
  {
    // initialize calculation of initial field based on fast Fourier transformation
    HomoIsoTurbInitialFieldHDG HitInitialFieldHDG(*this, initfield);
    // calculate initial field
    HitInitialFieldHDG.calculate_initial_field();

    // get statistics of initial field
    call_statistics_manager();

    // initialize  forcing depending on initial field
    forcing_interface_->set_initial_spectrum(initfield);
  }
  else
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
        intveln_->replace_global_values(globalDofs.size(), elevec2.values(), globalDofs.data());
        intvelnm_->replace_global_values(globalDofs.size(), elevec2.values(), globalDofs.data());
      }
    }
    double globerror = 0;
    globerror = Core::Communication::sum_all(error, discret_->get_comm());
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      std::cout << "Error project when setting face twice: " << globerror << std::endl;
  }
}



/*----------------------------------------------------------------------*
 | evaluate error for test cases with analytical solutions  kronbi 05/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<double>> FLD::TimIntHDG::evaluate_error_compared_to_analytical_sol()
{
  // HDG needs one more state vector for the interior solution (i.e., the actual solution)
  const auto calcerr =
      Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(*params_, "calculate error");

  switch (calcerr)
  {
    case Inpar::FLUID::beltrami_flow:
    case Inpar::FLUID::channel2D:
    case Inpar::FLUID::gravitation:
    case Inpar::FLUID::shear_flow:
    case Inpar::FLUID::fsi_fluid_pusher:
    case Inpar::FLUID::byfunct:
      discret_->set_state(1, "intvelnp", *intvelnp_);
      break;
    default:
      break;
  };

  return FluidImplicitTimeInt::evaluate_error_compared_to_analytical_sol();
}



/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::TimIntHDG::reset(int numsteps, int iter)
{
  FluidImplicitTimeInt::reset(numsteps, iter);
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(1);
  intvelnp_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intvelaf_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intvelnm_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intveln_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intaccnp_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intaccam_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intaccnm_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intaccn_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "Number of degrees of freedom in HDG system: "
              << discret_->dof_row_map(0)->num_global_elements() << std::endl;
}



namespace
{
  // internal helper function for output
  void get_node_vectors_hdg(Core::FE::Discretization& dis,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& interiorValues,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& traceValues, const int ndim,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& velocity,
      std::shared_ptr<Core::LinAlg::Vector<double>>& pressure,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& tracevel,
      std::shared_ptr<Core::LinAlg::Vector<double>>& cellPres)
  {
    // create dofsets for velocity and pressure at nodes
    if (pressure.get() == nullptr || pressure->global_length() != dis.num_global_nodes())
    {
      velocity = std::make_shared<Core::LinAlg::MultiVector<double>>(*dis.node_row_map(), ndim);
      pressure = std::make_shared<Core::LinAlg::Vector<double>>(*dis.node_row_map());
    }
    tracevel = std::make_shared<Core::LinAlg::MultiVector<double>>(velocity->get_map(), ndim);
    cellPres = std::make_shared<Core::LinAlg::Vector<double>>(*dis.element_row_map());

    // call element routine for interpolate HDG to elements
    Teuchos::ParameterList params;
    params.set<FLD::Action>("action", FLD::interpolate_hdg_to_node);
    dis.set_state(1, "intvelnp", *interiorValues);
    dis.set_state(0, "velnp", *traceValues);
    std::vector<int> dummy;
    Core::LinAlg::SerialDenseMatrix dummyMat;
    Core::LinAlg::SerialDenseVector dummyVec;
    Core::LinAlg::SerialDenseVector interpolVec;
    std::vector<unsigned char> touchCount(dis.num_my_row_nodes());
    velocity->put_scalar(0.);
    pressure->put_scalar(0.);
    for (int el = 0; el < dis.num_my_col_elements(); ++el)
    {
      Core::Elements::Element* ele = dis.l_col_element(el);
      if (interpolVec.numRows() == 0) interpolVec.resize(ele->num_node() * (2 * ndim + 1) + 1);

      ele->evaluate(params, dis, dummy, dummyMat, dummyMat, interpolVec, dummyVec, dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i = 0; i < ele->num_node(); ++i)
      {
        Core::Nodes::Node* node = ele->nodes()[i];
        const int localIndex = dis.node_row_map()->lid(node->id());
        if (localIndex < 0) continue;
        touchCount[localIndex]++;
        for (int d = 0; d < ndim; ++d)
          (*velocity)(d).get_values()[localIndex] += interpolVec(i + d * ele->num_node());
        (*pressure).get_values()[localIndex] += interpolVec(i + ndim * ele->num_node());
        for (int d = 0; d < ndim; ++d)
          (*tracevel)(d).get_values()[localIndex] +=
              interpolVec(i + (ndim + 1 + d) * ele->num_node());
      }
      const int eleIndex = dis.element_row_map()->lid(ele->id());
      if (eleIndex >= 0)
        (*cellPres).get_values()[eleIndex] += interpolVec((2 * ndim + 1) * ele->num_node());
    }

    for (int i = 0; i < pressure->local_length(); ++i)
    {
      (*pressure).get_values()[i] /= touchCount[i];
      for (int d = 0; d < ndim; ++d) (*velocity)(d).get_values()[i] /= touchCount[i];
      for (int d = 0; d < ndim; ++d) (*tracevel)(d).get_values()[i] /= touchCount[i];
    }
    dis.clear_state();
  }
}  // namespace



/*----------------------------------------------------------------------*
 | output of solution vector to binio                  kronbichler 05/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::output()
{
  // output of solution, currently only small subset of functionality
  if (step_ % upres_ == 0)
  {
    // step number and time
    output_->new_step(step_, time_);

    std::shared_ptr<Core::LinAlg::Vector<double>> cellPres;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> traceVel;
    get_node_vectors_hdg(*discret_, intvelnp_, velnp_,
        params_->get<int>("number of velocity degrees of freedom"), interpolatedVelocity_,
        interpolatedPressure_, traceVel, cellPres);
    output_->write_multi_vector("velnp_hdg", *interpolatedVelocity_, Core::IO::nodevector);
    output_->write_vector("pressure_hdg", interpolatedPressure_, Core::IO::nodevector);
    output_->write_multi_vector("tracevel_hdg", *traceVel, Core::IO::nodevector);
    output_->write_vector("pressure_avg", cellPres, Core::IO::elementvector);

    if (step_ == upres_ or step_ == 0) output_->write_element_data(true);

    if (uprestart_ != 0 && step_ % uprestart_ == 0)  // add restart data
    {
      // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
      // output_->write_vector("accnp",intaccnp_);
      // output_->write_vector("accn", intaccn_);
      // output_->write_vector("veln", intveln_);
      // output_->write_vector("velnm",intvelnm_);
    }
  }
}

/*----------------------------------------------------------------------*
 | calculate intermediate solution                              bk 04/15|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::calc_intermediate_solution()
{
  if ((special_flow_ == "forced_homogeneous_isotropic_turbulence" or
          special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
          special_flow_ == "decaying_homogeneous_isotropic_turbulence") and
      Teuchos::getIntegralValue<Inpar::FLUID::ForcingType>(params_->sublist("TURBULENCE MODEL"),
          "FORCING_TYPE") == Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> inttmp =
        Core::LinAlg::create_vector(*discret_->dof_row_map(1), true);
    inttmp->update(1.0, *intvelnp_, 0.0);

    FLD::FluidImplicitTimeInt::calc_intermediate_solution();

    intvelnp_->update(1.0, *inttmp, 0.0);

    // This code is entered at the beginning of the nonlinear iteration, so
    // store that the assembly to be done next is going to be the first one
    // (without combined vector update) for HDG.
    first_assembly_ = true;


    // recompute intermediate values, since they have been likewise overwritten
    // --------------------------------------------------
    // adjust accnp according to Dirichlet values of velnp
    //
    //                                  n+1     n
    //                               vel   - vel
    //       n+1      n  gamma-1.0      (0)
    //    acc    = acc * --------- + ------------
    //       (0)           gamma      gamma * dt
    //
    gen_alpha_update_acceleration();

    // ----------------------------------------------------------------
    // compute values at intermediate time steps
    // ----------------------------------------------------------------
    gen_alpha_intermediate_values();
  }
  return;
}

/*----------------------------------------------------------------------*
 | Initialize forcing for HIT and periodic hill                  bk 04/15|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::init_forcing()
{
  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  if (special_flow_ == "forced_homogeneous_isotropic_turbulence" or
      special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
      special_flow_ == "decaying_homogeneous_isotropic_turbulence" or
      special_flow_ == "periodic_hill")
  {
    forcing_ = Core::LinAlg::create_vector(*(discret_->dof_row_map(1)), true);

    if (special_flow_ == "forced_homogeneous_isotropic_turbulence" or
        special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
        special_flow_ == "decaying_homogeneous_isotropic_turbulence")
    {
      forcing_interface_ = std::make_shared<FLD::HomoIsoTurbForcingHDG>(*this);
    }
    else
      FOUR_C_THROW("forcing interface doesn't know this flow");
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE
