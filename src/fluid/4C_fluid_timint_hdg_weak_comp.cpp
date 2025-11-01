// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_hdg_weak_comp.hpp"

#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_hdg_weak_comp.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_fluid_weakly_compressible.hpp"
#include "4C_mat_par_bundle.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                  laspina 08/19 |
 *----------------------------------------------------------------------*/
FLD::TimIntHDGWeakComp::TimIntHDGWeakComp(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      TimIntGenAlpha(actdis, solver, params, output, alefluid),
      timealgoset_(Inpar::FLUID::timeint_afgenalpha),
      first_assembly_(false)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                  laspina 08/19 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::init()
{
  Core::FE::DiscretizationHDG* hdgdis = dynamic_cast<Core::FE::DiscretizationHDG*>(discret_.get());
  if (hdgdis == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  // get number of spatial dimensions
  const unsigned int nsd = params_->get<int>("number of velocity degrees of freedom");

  // initialize density/momentum splitting
  std::vector<int> dof_all;
  std::set<int> dofset_r;
  std::set<int> dofset_w;

  // fill dofset
  for (int i = 0; i < hdgdis->num_my_row_faces(); ++i)
  {
    dof_all = hdgdis->dof(0, hdgdis->l_row_face(i));

    for (unsigned int j_r = 0; j_r < (dof_all.size() / (1 + nsd)); ++j_r)
      dofset_r.insert(dof_all[j_r]);

    for (unsigned int j_w = (dof_all.size() / (1 + nsd)); j_w < dof_all.size(); ++j_w)
      dofset_w.insert(dof_all[j_w]);
  }

  // define density dof map
  std::vector<int> dofmapvec_r;
  dofmapvec_r.reserve(dofset_r.size());
  dofmapvec_r.assign(dofset_r.begin(), dofset_r.end());
  dofset_r.clear();
  std::shared_ptr<Core::LinAlg::Map> dofmap_r = std::make_shared<Core::LinAlg::Map>(
      -1, dofmapvec_r.size(), dofmapvec_r.data(), 0, hdgdis->get_comm());

  // define momentum dof map
  std::vector<int> dofmapvec_w;
  dofmapvec_w.reserve(dofset_w.size());
  dofmapvec_w.assign(dofset_w.begin(), dofset_w.end());
  dofset_w.clear();
  std::shared_ptr<Core::LinAlg::Map> dofmap_w = std::make_shared<Core::LinAlg::Map>(
      -1, dofmapvec_w.size(), dofmapvec_w.data(), 0, hdgdis->get_comm());

  // build density/momentum (actually velocity/pressure) splitter
  velpressplitter_->setup(*hdgdis->dof_row_map(), dofmap_r, dofmap_w);

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
  {
    // mimic backward Euler neglecting inertial terms
    alphaM_ = 1.0;
    alphaF_ = 1.0;
    gamma_ = 1.0;
  }

  timealgoset_ = timealgo_;
  timealgo_ = Inpar::FLUID::timeint_afgenalpha;

  // call init()-functions of base classes
  // note: this order is important
  FLD::TimIntGenAlpha::init();
}



/*----------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_, modified for HDG laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::set_theta()
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
 * Explicit predictor                                     laspina 08/19 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::explicit_predictor()
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
| set old part of right hand side                         laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::set_old_part_of_righthandside()
{
  hist_->put_scalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  first_assembly_ = true;
}



/*----------------------------------------------------------------------*
 | update acceleration for generalized-alpha              laspina 08/19 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::gen_alpha_update_acceleration()
{
  //                                  n+1     n
  //                               vel   - vel
  //       n+1      n  gamma-1.0      (i)
  //    acc    = acc * --------- + ------------
  //       (i)           gamma      gamma * dt
  //

  // as opposed to standard fluid, update all variables

  // compute factors
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);

  accnp_->update(fact2, *accn_, 0.0);
  accnp_->update(fact1, *velnp_, -fact1, *veln_, 1.0);

  intaccnp_->update(fact2, *intaccn_, 0.0);
  intaccnp_->update(fact1, *intvelnp_, -fact1, *intveln_, 1.0);
}



/*----------------------------------------------------------------------*
 | compute values at intermediate time steps              laspina 08/19 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::gen_alpha_intermediate_values()
{
  // set intermediate values for accelerations
  //
  //       n+alphaM                n+1                      n
  //    acc         = alpha_M * acc     + (1-alpha_M) *  acc
  //       (i)                     (i)
  accam_->update((alphaM_), *accnp_, (1.0 - alphaM_), *accn_, 0.0);
  intaccam_->update((alphaM_), *intaccnp_, (1.0 - alphaM_), *intaccn_, 0.0);

  // set intermediate values for mixed variable, density and momentum
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  velaf_->update((alphaF_), *velnp_, (1.0 - alphaF_), *veln_, 0.0);
  intvelaf_->update((alphaF_), *intvelnp_, (1.0 - alphaF_), *intveln_, 0.0);
}


/*----------------------------------------------------------------------*
| set HDG state vectors                                   laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::set_state_tim_int()
{
  discret_->set_state(0, "velaf", *velaf_);
  discret_->set_state(1, "intvelaf", *intvelaf_);
  discret_->set_state(1, "intaccam", *intaccam_);
  discret_->set_state(1, "intvelnp", *intvelnp_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                   laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::set_custom_ele_params_assemble_mat_and_rhs(
    Teuchos::ParameterList& eleparams)
{
  eleparams.set<bool>("needslocalupdate", !first_assembly_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                   laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::clear_state_assemble_mat_and_rhs()
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
| update within iteration                                 laspina 08/19 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::iter_update(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> increment)
{
  // call element routine to update local solution
  Teuchos::ParameterList params;
  params.set<FLD::Action>("action", FLD::update_local_solution);

  // location array
  Core::Elements::LocationArray la(2);

  // interior dofs map
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(1);

  // dummy variables
  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;

  // initialize elemental local increments
  Core::LinAlg::SerialDenseVector elemintinc;

  // initialize increments of local variables
  std::shared_ptr<Core::LinAlg::Vector<double>> intvelincnp =
      Core::LinAlg::create_vector(*intdofrowmap, true);

  // set state
  set_state_tim_int();
  discret_->set_state(0, "globaltraceinc", *increment);

  for (int el = 0; el < discret_->num_my_col_elements(); ++el)
  {
    // get element
    Core::Elements::Element* ele = discret_->l_col_element(el);
    ele->location_vector(*discret_, la);

    // evaluate interior local increments
    ele->evaluate(params, *discret_, la[0].lm_, dummyMat, dummyMat, elemintinc, dummyVec, dummyVec);

    // fill the interior increment vector for all the discretization
    if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
    {
      std::vector<int> globalDofs = discret_->dof(1, ele);
      intvelincnp->replace_global_values(globalDofs.size(), elemintinc.values(), globalDofs.data());
    }
  }

  // update interior values by adding increments
  intvelnp_->update(1.0, *intvelincnp, 1.0);

  // set new state
  set_state_tim_int();

  // call base function
  FluidImplicitTimeInt::iter_update(increment);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::time_update()
{
  FluidImplicitTimeInt::time_update();

  // local solution of this step become most recent
  // local solution of the last step
  intvelnm_->update(1.0, *intveln_, 0.0);
  intveln_->update(1.0, *intvelnp_, 0.0);

  intaccnm_->update(1.0, *intaccn_, 0.0);
  intaccn_->update(1.0, *intaccnp_, 0.0);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::update_gridv()
{
  if (timealgoset_ == Inpar::FLUID::timeint_afgenalpha ||
      timealgoset_ == Inpar::FLUID::timeint_npgenalpha ||
      timealgoset_ == Inpar::FLUID::timeint_bdf2)  // 2nd order methods
  {
    if (step_ <= 1)
    {
      // use BDF1 in 1st step
      gridv_->update(1.0 / dta_, *dispnp_, -1.0 / dta_, *dispn_, 0.0);
    }
    else
    {
      // use BDF2 after 1st step
      gridv_->update(1.5 / dta_, *dispnp_, -2.0 / dta_, *dispn_, 0.0);
      gridv_->update(0.5 / dta_, *dispnm_, 1.0);
    }
  }
  else if (timealgoset_ == Inpar::FLUID::timeint_one_step_theta)  // 1st order methods
  {
    // use BDF1
    gridv_->update(1.0 / dta_, *dispnp_, -1.0 / dta_, *dispn_, 0.0);
  }
  else if (timealgoset_ == Inpar::FLUID::timeint_stationary)
  {
    gridv_->put_scalar(0.0);
  }
}



/*----------------------------------------------------------------------*
 |  set initial flow field                                 laspina 08/19|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::set_initial_flow_field(
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
      intveln_->replace_global_values(globalDofs.size(), elevec2.values(), globalDofs.data());
      intvelnm_->replace_global_values(globalDofs.size(), elevec2.values(), globalDofs.data());
    }
  }

  double globerror = 0;
  globerror = Core::Communication::sum_all(error, discret_->get_comm());
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "Error project when setting face twice: " << globerror << std::endl;
}



/*----------------------------------------------------------------------*
 | evaluate error for test cases with analytical solutions laspina 08/19|
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<double>>
FLD::TimIntHDGWeakComp::evaluate_error_compared_to_analytical_sol()
{
  // HDG needs one more state vector for the interior solution (i.e., the actual solution)
  const auto calcerr =
      Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(*params_, "calculate error");

  switch (calcerr)
  {
    case Inpar::FLUID::no:
    {
      return nullptr;
      break;
    }
    case Inpar::FLUID::byfunct:
    {
      discret_->set_state(1, "intvelnp", *intvelnp_);

      // std::vector containing
      // [0]: absolute L2 mixed variable error
      // [1]: absolute L2 density error
      // [2]: absolute L2 momentum error
      std::shared_ptr<std::vector<double>> abserror = std::make_shared<std::vector<double>>(3);

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;

      // action for elements
      eleparams.set<FLD::Action>("action", FLD::calc_fluid_error);
      eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);
      eleparams.set<Inpar::FLUID::CalcError>("calculate error", calcerr);

      // get function number
      const int errorfunctno = params_->get<int>("error function number", -1);
      eleparams.set<int>("error function number", errorfunctno);

      // set scheme-specific element parameters and vector values
      set_state_tim_int();

      if (alefluid_) discret_->set_state(2, "dispnp", *dispnp_);

      // get (squared) error values
      // 0: delta mixed variable for L2-error norm
      // 1: delta density for L2-error norm
      // 2: delta momentum for L2-error norm
      // (3: analytical mixed variable for L2 norm)
      // (4: analytical density for L2 norm)
      // (5: analytical momentum for L2 norm)
      Core::LinAlg::SerialDenseVector errors(3 + 3);

      // call loop over elements (assemble nothing)
      discret_->evaluate_scalars(eleparams, errors);
      discret_->clear_state();

      // evaluate absolute L2 error
      (*abserror)[0] = sqrt((errors)[0]);
      (*abserror)[1] = sqrt((errors)[1]);
      (*abserror)[2] = sqrt((errors)[2]);

      if (myrank_ == 0)
      {
        {
          std::cout.precision(8);
          std::cout << std::endl;
          std::cout << "---- Error norm for analytical solution -------------------" << std::endl;
          std::cout << "| absolute L_2 mixed variable error norm:   " << (*abserror)[0]
                    << std::endl;
          std::cout << "| absolute L_2 density        error norm:   " << (*abserror)[1]
                    << std::endl;
          std::cout << "| absolute L_2 momentum       error norm:   " << (*abserror)[2]
                    << std::endl;
          std::cout << "-----------------------------------------------------------" << std::endl;
          std::cout << std::endl;
        }

        // print last error in a separate file

        // append error of the last time step to the error file
        if ((step_ == stepmax_) or (time_ == maxtime_))  // write results to file
        {
          std::ostringstream temp;
          const std::string simulation =
              Global::Problem::instance()->output_control_file()->file_name();
          const std::string fname = simulation + ".abserror";

          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "#| " << simulation << "\n";
          f << "#| Step | Time | abs. L2-error mixed variable | abs. L2-error density | abs. "
               "L2-error momentum |\n";
          f << step_ << " " << time_ << " " << (*abserror)[0] << " " << (*abserror)[1] << " "
            << (*abserror)[2] << "\n";
          f.flush();
          f.close();
        }

        std::ostringstream temp;
        const std::string simulation =
            Global::Problem::instance()->output_control_file()->file_name();
        const std::string fname = simulation + "_time.abserror";

        if (step_ == 1)
        {
          std::ofstream f;
          f.open(fname.c_str());
          f << "#| Step | Time | abs. L2-error mixed variable | abs. L2-error density | abs. "
               "L2-error momentum |\n";
          f << std::setprecision(10) << step_ << " " << std::setw(1) << std::setprecision(5)
            << time_ << std::setw(1) << std::setprecision(6) << " " << (*abserror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*abserror)[1] << std::setprecision(6)
            << " " << (*abserror)[2] << "\n";
          f.flush();
          f.close();
        }
        else
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << std::setprecision(10) << step_ << " " << std::setw(3) << std::setprecision(5)
            << time_ << std::setw(1) << std::setprecision(6) << " " << (*abserror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*abserror)[1] << std::setprecision(6)
            << " " << (*abserror)[2] << "\n";
          f.flush();
          f.close();
        }
      }
      return abserror;
    }
    break;
    default:
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem");
      break;
  }

  return nullptr;
}



/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::reset(int numsteps, int iter)
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
  void get_node_vectors_hdg_weak_comp(Core::FE::Discretization& dis,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& interiorValues,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& traceValues, const int ndim,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& mixedvar,
      std::shared_ptr<Core::LinAlg::Vector<double>>& density,
      std::shared_ptr<Core::LinAlg::Vector<double>>& traceden)
  {
    const int msd = ndim * (ndim + 1.0) / 2.0;

    // create dofsets for mixed variable, density and momentum at nodes
    if (density.get() == nullptr || density->global_length() != dis.num_global_nodes())
    {
      mixedvar = std::make_shared<Core::LinAlg::MultiVector<double>>(*dis.node_row_map(), msd);
      density = std::make_shared<Core::LinAlg::Vector<double>>(*dis.node_row_map());
    }
    traceden = std::make_shared<Core::LinAlg::Vector<double>>(density->get_map());

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
    mixedvar->put_scalar(0.);
    density->put_scalar(0.);

    for (int el = 0; el < dis.num_my_col_elements(); ++el)
    {
      Core::Elements::Element* ele = dis.l_col_element(el);
      if (interpolVec.numRows() == 0)
        interpolVec.resize(ele->num_node() * (msd + 1 + ndim + 1 + ndim));

      ele->evaluate(params, dis, dummy, dummyMat, dummyMat, interpolVec, dummyVec, dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i = 0; i < ele->num_node(); ++i)
      {
        Core::Nodes::Node* node = ele->nodes()[i];
        const int localIndex = dis.node_row_map()->lid(node->id());
        if (localIndex < 0) continue;
        touchCount[localIndex]++;
        for (int m = 0; m < msd; ++m)
          (*mixedvar)(m).get_values()[localIndex] += interpolVec(i + m * ele->num_node());
        (*density).get_values()[localIndex] += interpolVec(i + msd * ele->num_node());
        (*traceden).get_values()[localIndex] += interpolVec(i + (msd + 1 + ndim) * ele->num_node());
      }
    }

    for (int i = 0; i < density->local_length(); ++i)
    {
      for (int m = 0; m < msd; ++m) (*mixedvar)(m).get_values()[i] /= touchCount[i];
      (*density).get_values()[i] /= touchCount[i];
      (*traceden).get_values()[i] /= touchCount[i];
    }
    dis.clear_state();
  }
}  // namespace



/*----------------------------------------------------------------------*
 | output of solution vector to binio                      laspina 08/19|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDGWeakComp::output()
{
  // output of solution, currently only small subset of functionality
  if (step_ % upres_ == 0)
  {
    // step number and time
    output_->new_step(step_, time_);

    // get number of spatial dimensions
    const unsigned int nsd = params_->get<int>("number of velocity degrees of freedom");

    // initialize trace variables
    std::shared_ptr<Core::LinAlg::Vector<double>> traceDen;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> traceMom;

    // get node vectors
    get_node_vectors_hdg_weak_comp(
        *discret_, intvelnp_, velnp_, nsd, interpolatedMixedVar_, interpolatedDensity_, traceDen);

    // get weakly compressible material
    int id = Global::Problem::instance()->materials()->first_id_by_type(
        Core::Materials::m_fluid_weakly_compressible);
    const Core::Mat::PAR::Parameter* mat =
        Global::Problem::instance()->materials()->parameter_by_id(id);
    const Mat::PAR::WeaklyCompressibleFluid* actmat =
        static_cast<const Mat::PAR::WeaklyCompressibleFluid*>(mat);

    // evaluate derived variables
    std::shared_ptr<Core::LinAlg::MultiVector<double>> interpolatedVelocity;
    std::shared_ptr<Core::LinAlg::Vector<double>> interpolatedPressure;
    interpolatedPressure =
        std::make_shared<Core::LinAlg::Vector<double>>(interpolatedDensity_->get_map());
    for (int i = 0; i < interpolatedDensity_->local_length(); ++i)
    {
      (*interpolatedPressure).get_values()[i] =
          actmat->refpressure_ +
          1.0 / actmat->comprcoeff_ *
              (interpolatedDensity_->local_values_as_span()[i] - actmat->refdensity_);
    }

    // write solution variables
    output_->write_multi_vector("Mixedvar", *interpolatedMixedVar_, Core::IO::nodevector);
    output_->write_vector("Density", interpolatedDensity_, Core::IO::nodevector);
    output_->write_vector("Trace_density", traceDen, Core::IO::nodevector);

    // write derived variables
    output_->write_vector("Pressure", interpolatedPressure, Core::IO::nodevector);

    // write ALE variables
    if (alefluid_)
    {
      Core::LinAlg::MultiVector<double> AleDisplacement(*discret_->node_row_map(), nsd);
      for (int i = 0; i < interpolatedDensity_->local_length(); ++i)
        for (unsigned int d = 0; d < nsd; ++d)
          AleDisplacement(d).get_values()[i] = dispnp_->local_values_as_span()[(i * nsd) + d];

      output_->write_multi_vector("Ale_displacement", AleDisplacement, Core::IO::nodevector);
    }

    if (step_ == upres_ or step_ == 0) output_->write_element_data(true);
  }
}

FOUR_C_NAMESPACE_CLOSE
