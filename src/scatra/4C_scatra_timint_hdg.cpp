// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_hdg.hpp"

#include "4C_binstrategy.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_hdg.hpp"
#include "4C_scatra_resulttest_hdg.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntHDG::TimIntHDG(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Teuchos::ParameterList>& extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output),
      TimIntGenAlpha(actdis, solver, params, extraparams, output),
      nds_intvar_(2),
      intphinp_(nullptr),
      intphin_(nullptr),
      interpolatedPhinp_(nullptr),
      timealgoset_(Inpar::ScaTra::timeint_gen_alpha),
      startalgo_(true),
      theta_(-1),
      hdgdis_(nullptr),
      padaptivity_(params->get<bool>("PADAPTIVITY")),
      padapterrortol_(params->get<double>("PADAPTERRORTOL")),
      padapterrorbase_(params->get<double>("PADAPTERRORBASE")),
      padaptdegreemax_(params->get<int>("PADAPTDEGREEMAX")),
      elementdegree_(nullptr)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::setup()
{
  hdgdis_ = dynamic_cast<Core::FE::DiscretizationHDG*>(discret_.get());
  if (hdgdis_ == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  // vector to store the dofs per element
  const std::shared_ptr<Core::LinAlg::Vector<int>> eledofs =
      std::make_shared<Core::LinAlg::Vector<int>>(*discret_->element_col_map());

  // loop over elements
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    auto* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(discret_->l_col_element(iele));
    (*eledofs).get_local_values()[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // add proxy for interior degrees of freedom to scatra discretization
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, eledofs, 0, false);
  if (discret_->add_dof_set(dofsetaux) != 2)
    FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
  discret_->fill_complete();

  // HDG vectors passed to the element
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(nds_intvar_);
  intphinp_ = Core::LinAlg::create_vector(*intdofrowmap, true);
  intphin_ = Core::LinAlg::create_vector(*intdofrowmap, true);

  // write number of degrees of freedom for hdg and interior variables to screen output
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::cout << "Number of degrees of freedom in HDG system: "
              << discret_->dof_row_map(0)->num_global_elements() << std::endl;
    std::cout << "Number of degrees of freedom of interior variables: "
              << discret_->dof_row_map(nds_intvar_)->num_global_elements() << std::endl;
  }

  // implement ost and bdf2 through gen-alpha facilities
  // TO DO: implement other time integration schemes, at the moment only one-step-theta and
  // stationary are implemented
  switch (timealgo_)
  {
    case Inpar::ScaTra::timeint_bdf2:
    {
      FOUR_C_THROW("At the moment only one step theta implemented");
      alphaM_ = 1.5;
      alphaF_ = 1.0;
      gamma_ = 1.0;
      break;
    }
    case Inpar::ScaTra::timeint_one_step_theta:
    {
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_ = params_->get<double>("THETA");
      break;
    }
    case Inpar::ScaTra::timeint_stationary:
    {
      break;
    }
    default:
      FOUR_C_THROW("At the moment only one step theta implemented");
  }

  timealgoset_ = timealgo_;
  if (timealgo_ != Inpar::ScaTra::timeint_stationary) timealgo_ = Inpar::ScaTra::timeint_gen_alpha;

  // call init()-functions of base classes
  // note: this order is important
  ScaTra::TimIntGenAlpha::setup();

  // create vector for concentration at nodes for output
  interpolatedPhinp_ = Core::LinAlg::create_vector(*discret_->node_row_map(), true);

  // vector to store the elementdegree at each time step
  elementdegree_ = Core::LinAlg::create_vector(*(discret_->element_row_map()), true);
}


/*------------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_, modified for HDG  hoermann 09/15 |
*-------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::set_theta()
{
  // -------------------------------------------------------------------
  //  For af-generalized-alpha time-integration scheme:
  //  set "pseudo-theta", calculate initial accelerations according to
  //  prescribed Dirichlet values for generalized-alpha time
  //  integration and values at intermediate time steps
  // -------------------------------------------------------------------
  // starting algorithm
  if (startalgo_ || (step_ <= 2 && timealgoset_ == Inpar::ScaTra::timeint_bdf2))
  {
    // use backward-Euler-type parameter combination
    if (step_ <= 1 && timealgoset_ == Inpar::ScaTra::timeint_bdf2)
    {
      if (myrank_ == 0)
      {
        std::cout << "Starting algorithm for Af_GenAlpha active. "
                  //<<"Performing step "<<step_ <<" of "<<numstasteps_
                  << " Backward Euler starting steps" << std::endl;
      }
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_ = 1.0;
    }
    else
    {
      // recall original user wish
      switch (timealgoset_)
      {
        case Inpar::ScaTra::timeint_one_step_theta:
        {
          alphaM_ = alphaF_ = 1.0;
          gamma_ = params_->get<double>("THETA");
          break;
        }
        case Inpar::ScaTra::timeint_bdf2:
        {
          alphaF_ = gamma_ = 1.0;
          alphaM_ = 3. / 2.;
          break;
        }
        case Inpar::ScaTra::timeint_stationary:
        {
          // Setting the parameters as for Inpar::ScaTra::timeint_one_step_theta with theta = 1
          // (basically BDF1 and therefore we only compute the RHS and Dirich at t+1)
          alphaM_ = alphaF_ = gamma_ = 1.0;
          // The time step can not be set to zero because there is plenty of divisions by dt.
          // Dt is therefore to 1.0
          dta_ = 1.0;
          // Set time equal -dta, this way the steady state is given as the solution at t=0.0.
          // This is necessary otherwise the solver would recognise that we are at the end of a
          // simulation and skip the solution altogether.
          time_ = -dta_;
          maxtime_ = dta_;
          // stepmax is 1 to avoid waste computation (it's stationary after all)
          stepmax_ = 1;
          break;
        }
        default:
        {
          alphaM_ = params_->get<double>("alpha_M");
          alphaF_ = params_->get<double>("alpha_F");
          gamma_ = params_->get<double>("gamma");
          break;
        }
      }

      // do not enter starting algorithm section in the future
      startalgo_ = false;
    }
  }

  // compute "pseudo-theta" for af-generalized-alpha scheme
  theta_ = alphaF_ * gamma_ / alphaM_;
}


/*----------------------------------------------------------------------*
| set HDG state vectors                                  hoermann 09/15 |
*-----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // set hdg vector and interior variables vector
  discret_->set_state(0, "phin", *phin_);
  discret_->set_state(0, "phiaf", *phinp_);
  discret_->set_state(nds_intvar_, "intphinp", *intphinp_);
  discret_->set_state(nds_intvar_, "intphin", *intphin_);
}  // add_time_integration_specific_vectors

/*----------------------------------------------------------------------*
 | compute values at intermediate time steps for gen.-alpha  hoer 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::gen_alpha_intermediate_values()
{
  // set intermediate values for concentration derivatives
  //
  //       n+alphaM                n+1                      n
  //  dtphi       = alpha_M * dtphi    + (1-alpha_M) * dtphi
  //       (i)                     (i)
  phidtam_->update((alphaM_), *phidtnp_, (1.0 - alphaM_), *phidtn_, 0.0);

  // set intermediate values for concentration, concentration gradient
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  //
  // note that its af-genalpha with mid-point treatment of the pressure,
  // not implicit treatment as for the genalpha according to Whiting
  phiaf_->update((alphaF_), *phinp_, (1.0 - alphaF_), *phin_, 0.0);

  phiam_->update(alphaM_, *phinp_, (1.0 - alphaM_), *phin_, 0.0);

}  // gen_alpha_intermediate_values

/*----------------------------------------------------------------------*
| set old part of right hand side                        hoermann 09/15 |
*-----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::set_old_part_of_righthandside()
{
  set_theta();
  hist_->put_scalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  //  ScaTra::TimIntGenAlpha::set_old_part_of_righthandside();
}

/*----------------------------------------------------------------------*
 * Update
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::update()
{
  ScaTra::TimIntGenAlpha::update();

  // concentrations of this step become most recent
  // concentrations of the last step
  intphin_->update(1.0, *intphinp_, 0.0);

  if (padaptivity_) adapt_degree();

}  // Update

namespace
{
  // internal helper function for output
  void get_node_vectors_hdg(Core::FE::Discretization& dis,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& interiorValues,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& traceValues, const int ndim,
      std::shared_ptr<Core::LinAlg::Vector<double>>& phi,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& gradphi,
      std::shared_ptr<Core::LinAlg::Vector<double>>& tracephi, int nds_intvar_, int ndofs)
  {
    dis.clear_state(true);

    // create dofsets for concentration at nodes
    tracephi = std::make_shared<Core::LinAlg::Vector<double>>(phi->get_map());
    gradphi = std::make_shared<Core::LinAlg::MultiVector<double>>(*dis.node_row_map(), ndim);

    // call element routine to interpolate HDG to elements
    Teuchos::ParameterList eleparams;
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::interpolate_hdg_to_node, eleparams);
    dis.set_state(0, "phiaf", *traceValues);
    dis.set_state(nds_intvar_, "intphinp", *interiorValues);
    Core::Elements::LocationArray la(ndofs);
    Core::LinAlg::SerialDenseMatrix dummyMat;
    Core::LinAlg::SerialDenseVector dummyVec;
    Core::LinAlg::SerialDenseVector interpolVec;
    std::vector<unsigned char> touchCount(dis.num_my_row_nodes());

    phi->put_scalar(0.);

    for (int el = 0; el < dis.num_my_col_elements(); ++el)
    {
      Core::Elements::Element* ele = dis.l_col_element(el);
      ele->location_vector(dis, la);
      interpolVec.size(ele->num_node() * (2 + ndim));

      ele->evaluate(eleparams, dis, la, dummyMat, dummyMat, interpolVec, dummyVec, dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i = 0; i < ele->num_node(); ++i)
      {
        Core::Nodes::Node* node = ele->nodes()[i];
        const int localIndex = dis.node_row_map()->lid(node->id());
        if (localIndex < 0) continue;
        touchCount[localIndex]++;
        (*phi).get_values()[localIndex] += interpolVec(i);
        (*tracephi).get_values()[localIndex] += interpolVec(i + ele->num_node());
        for (int d = 0; d < ndim; ++d)
          (*gradphi)(d).get_values()[localIndex] += interpolVec(i + (d + 2) * ele->num_node());
      }
    }

    // build average of values
    for (int i = 0; i < phi->local_length(); ++i)
    {
      (*phi).get_values()[i] /= touchCount[i];
      (*tracephi).get_values()[i] /= touchCount[i];
      for (int d = 0; d < ndim; ++d) (*gradphi)(d).get_values()[i] /= touchCount[i];
    }
    dis.clear_state(true);
  }
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::write_restart() const
{
  ScaTra::TimIntGenAlpha::write_restart();
  output_->write_vector("intphinp", intphinp_);
  output_->write_vector("phinp_trace", phinp_);
  output_->write_vector("intphin", intphin_);

  output_->write_mesh(
      step_, time_);  // add info to control file for reading all variables in restart
}

/*----------------------------------------------------------------------*
 -----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::collect_runtime_output_data()
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> interpolatedGradPhi;
  std::shared_ptr<Core::LinAlg::Vector<double>> interpolatedtracePhi;
  // get (averaged) values at element nodes
  get_node_vectors_hdg(*discret_, intphinp_, phinp_, Global::Problem::instance()->n_dim(),
      interpolatedPhinp_, interpolatedGradPhi, interpolatedtracePhi, nds_intvar_,
      discret_->num_dof_sets());

  // write vector to output file
  std::vector<std::optional<std::string>> context(interpolatedPhinp_->num_vectors(), "phi");
  visualization_writer().append_result_data_vector_with_context(
      *interpolatedPhinp_, Core::IO::OutputEntity::node, context);

  context.assign(interpolatedGradPhi->num_vectors(), "grad_phi");
  visualization_writer().append_result_data_vector_with_context(
      *interpolatedGradPhi, Core::IO::OutputEntity::node, context);

  context.assign(interpolatedtracePhi->num_vectors(), "trace_phi");
  visualization_writer().append_result_data_vector_with_context(
      *interpolatedtracePhi, Core::IO::OutputEntity::node, context);

  collect_problem_specific_runtime_output_data(interpolatedPhinp_);

  visualization_writer().append_result_data_vector_with_context(
      *elementdegree_, Core::IO::OutputEntity::element, {"elementdegree"});
}

/*----------------------------------------------------------------------*
 | read restart                                          hoermann 09/15 |
 -----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::read_restart(const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);

  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  reader.read_history_data(step);  // Read all saved data in nodes and elements and call nodal and
                                   // element Unpacking each global variable has to be read

  if (padaptivity_)
  {
    // redistribute discr. with help of binning strategy
    if (Core::Communication::num_mpi_ranks(discret_->get_comm()) > 1)
    {
      // create vector of discr.
      std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
      dis.push_back(discret_);

      // binning strategy for parallel redistribution
      std::shared_ptr<Core::Binstrategy::BinningStrategy> binningstrategy;

      std::vector<std::shared_ptr<Core::LinAlg::Map>> stdelecolmap;
      std::vector<std::shared_ptr<Core::LinAlg::Map>> stdnodecolmap;

      // binning strategy is created and parallel redistribution is performed
      Teuchos::ParameterList binning_params =
          Global::Problem::instance()->binning_strategy_params();
      Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
          "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
          binning_params);
      binningstrategy = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
          Global::Problem::instance()->output_control_file(), discret_->get_comm(),
          Core::Communication::my_mpi_rank(discret_->get_comm()), nullptr, nullptr, dis);
      binningstrategy
          ->do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
              dis, stdelecolmap, stdnodecolmap);
    }
  }

  // vector to store the dofs per element
  const std::shared_ptr<Core::LinAlg::Vector<int>> eledofs =
      std::make_shared<Core::LinAlg::Vector<int>>(*discret_->element_col_map());

  // build new maps for face dofs with adapted element order
  hdgdis_->build_faces();
  hdgdis_->build_face_row_map();
  hdgdis_->build_face_col_map();

  // assign the degrees of freedom to the adapted dofsets
  hdgdis_->assign_degrees_of_freedom(0);

  // replace all ghosted element with the original thus the correct polynomial degree is used
  discret_->export_column_elements(*discret_->element_col_map(), false, false);

  hdgdis_->fill_complete();

  // store the number of dofs per element on vector
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    auto* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(discret_->l_col_element(iele));
    // store the number of dofs for the element
    (*eledofs).get_local_values()[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // create new local dofset for the new interior element dofs with adapted element order
  std::shared_ptr<Core::DOFSets::DofSetPredefinedDoFNumber> eledofs_new =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, eledofs, 0, false);
  // replace old interior element dofs with the new created dofset
  discret_->replace_dof_set(nds_intvar_, eledofs_new, false);

  hdgdis_->assign_degrees_of_freedom(0);

  // clear map cache since after every fill_complete() / assign_degrees_of_freedom() old maps are
  // stored in the mapstack
  output_->clear_map_cache();

  // reset the residual, increment and sysmat to the size
  residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  increment_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  neumann_loads_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  sysmat_ = nullptr;
  sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27);

  // reset the state vectors
  intphinp_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map(nds_intvar_)), true);
  intphin_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map(nds_intvar_)), true);
  phinp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  phin_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));

  // read state vectors that are needed for hdg
  reader.read_vector(phinp_, "phinp_trace");
  reader.read_vector(intphinp_, "intphinp");

  intphin_->update(1.0, *intphinp_, 0.0);
  phin_->update(1.0, *phinp_, 0.0);

  // reset vector
  interpolatedPhinp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_row_map()));
  elementdegree_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->element_row_map()));
}

/*----------------------------------------------------------------------*
 |  set initial field for phi                            hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::set_initial_field(
    const Inpar::ScaTra::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case Inpar::ScaTra::initfield_zero_field:
    {
      // set initial field to zero
      phin_->put_scalar(0.0);
      phinp_->put_scalar(0.0);
      intphin_->put_scalar(0.0);
      intphinp_->put_scalar(0.0);
      break;
    }
    case Inpar::ScaTra::initfield_field_by_function:
    {
      // set initial field defined by function
      Teuchos::ParameterList eleparams;
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
          "action", ScaTra::Action::set_initial_field, eleparams);
      eleparams.set<int>("funct", startfuncno);

      discret_->set_state("phiaf", *phinp_);
      discret_->set_state("phin", *phin_);
      discret_->set_state(nds_intvar_, "intphin", *intphin_);
      discret_->set_state(nds_intvar_, "intphinp", *intphinp_);

      Core::LinAlg::SerialDenseMatrix dummyMat;
      Core::LinAlg::SerialDenseVector updateVec1, updateVec2, dummyVec;
      Core::Elements::LocationArray la(discret_->num_dof_sets());

      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
      double error = 0;

      for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
      {
        Core::Elements::Element* ele = discret_->l_col_element(iele);
        ele->location_vector(*discret_, la);
        if (static_cast<std::size_t>(updateVec1.numRows()) != la[0].lm_.size())
          updateVec1.size(la[0].lm_.size());
        else
          updateVec1.putScalar(0.0);
        if (updateVec2.numRows() != discret_->num_dof(nds_intvar_, ele))
          updateVec2.size(discret_->num_dof(nds_intvar_, ele));
        else
          updateVec2.putScalar(0.0);
        ele->evaluate(
            eleparams, *discret_, la, dummyMat, dummyMat, updateVec1, updateVec2, dummyVec);

        if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
        {
          std::vector<int> globalDofs = discret_->dof(nds_intvar_, ele);
          FOUR_C_ASSERT(globalDofs.size() == static_cast<std::size_t>(updateVec2.numRows()),
              "Internal error");
          intphinp_->replace_global_values(
              globalDofs.size(), updateVec2.values(), globalDofs.data());
        }

        // now fill the element vector into the discretization
        for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
        {
          const int lid = dofrowmap->lid(la[0].lm_[i]);
          if (lid >= 0)
          {
            // safety check if initial value for trace dof is set for all elements the same
            // (interior face)
            if ((*phinp_).get_values()[lid] != 0)
              error += std::abs((*phinp_).get_values()[lid] - updateVec1(i));
            (*phinp_).get_values()[lid] = updateVec1(i);
            (*phin_).get_values()[lid] = updateVec1(i);
          }
        }
      }

      double globerror = 0;
      globerror = Core::Communication::sum_all(error, discret_->get_comm());
      if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        std::cout << "Error project when setting face twice: " << globerror << std::endl;

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      intphin_->update(1.0, *intphinp_, 0.0);

      break;
    }

    default:
      FOUR_C_THROW("Option for initial field not implemented: {}", init);
      break;
  }  // switch(init)

}  // SetInitialField


/*----------------------------------------------------------------------*
 | calculate intermediate solution                        hoermann 09/15|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::compute_intermediate_values()
{
  // time derivatives are not independent but rather have to be computed
  // from phinp_, phin_ and phidtn_
  gen_alpha_compute_time_derivative();
  // compute values at intermediate time steps
  gen_alpha_intermediate_values();
}

/*----------------------------------------------------------------------*
 | compute values at the interior of the elements         hoermann 09/15|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::compute_interior_values()
{
  // Update the interior variables
  update_interior_variables(intphinp_);
}

/*----------------------------------------------------------------------*
 | update time derivative for gen-alpha time integration hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::gen_alpha_compute_time_derivative()
{
  //                                  n+1     n
  //                               phi   - phi
  //       n+1      n  gamma-1.0      (i)
  // phidt    = phidt * --------- + ------------
  //       (i)           gamma      gamma * dt
  //

  // compute factors
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);

  phidtnp_->update(fact2, *phidtn_, 0.0);
  phidtnp_->update(fact1, *phinp_, -fact1, *phin_, 1.0);

}  // gen_alpha_compute_time_derivative


/*----------------------------------------------------------------------*
 | update interior variables                             hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::update_interior_variables(
    std::shared_ptr<Core::LinAlg::Vector<double>> updatevector)
{
  discret_->clear_state(true);
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::update_interior_variables, eleparams);
  discret_->set_state("phiaf", *phinp_);
  discret_->set_state("phin", *phin_);
  discret_->set_state(nds_intvar_, "intphin", *intphin_);
  discret_->set_state(nds_intvar_, "intphinp", *intphinp_);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;
  Core::LinAlg::SerialDenseVector updateVec;
  Core::Elements::LocationArray la(discret_->num_dof_sets());

  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Core::Elements::Element* ele = discret_->l_col_element(iele);
    if (ele->owner() != Core::Communication::my_mpi_rank(discret_->get_comm())) continue;

    ele->location_vector(*discret_, la);
    updateVec.size(discret_->num_dof(nds_intvar_, ele));

    ele->evaluate(eleparams, *discret_, la, dummyMat, dummyMat, updateVec, dummyVec, dummyVec);

    std::vector<int> globalDofs = discret_->dof(nds_intvar_, ele);
    FOUR_C_ASSERT(
        globalDofs.size() == static_cast<std::size_t>(updateVec.numRows()), "Internal error");
    updatevector->replace_global_values(globalDofs.size(), updateVec.values(), globalDofs.data());
  }

  discret_->clear_state(true);
}


/*-------------------------------------------------------------------------------------*
 | finite difference check for system matrix (for debugging only)       hoermann 09/15 |
 *-------------------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::fd_check()
{
  // make a copy of state variables to undo perturbations later
  Core::LinAlg::Vector<double> phinp_original(*phinp_);

  discret_->clear_state(true);

  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map(0);
  const Core::LinAlg::Map* intdofrowmap = discret_->dof_row_map(nds_intvar_);

  std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1, systemmatrix2;
  std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1, systemvector2, systemvector3;

  // create matrix and vector for calculation of sysmat and assemble
  systemmatrix1 = std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27);
  systemvector1 = Core::LinAlg::create_vector(*dofrowmap, true);
  Core::FE::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);

  // fill state vector with original state variables
  phinp_->update(1., phinp_original, 0.);

  // make temporary vector for interior variables for the update of the last step and use this
  // vector for the calculation of the original residual the temporary vector is necessary because
  // afterwards we need to calculate also the interior vectors with the state vector with
  // perturbation without influence for this update
  std::shared_ptr<Core::LinAlg::Vector<double>> intphitemp;
  intphitemp = Core::LinAlg::create_vector(*intdofrowmap, true);

  strategy.zero();

  // calculate of residual vector
  update_interior_variables(intphitemp);

  discret_->clear_state(true);
  discret_->set_state("phiaf", *phinp_);
  discret_->set_state(nds_intvar_, "intphin", *intphin_);
  discret_->set_state(0, "phin", *phin_);
  discret_->set_state(nds_intvar_, "intphinp", *intphitemp);
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs, eleparams);
  Core::Elements::LocationArray la(discret_->num_dof_sets());

  // loop over elements
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Core::Elements::Element* ele = discret_->l_col_element(iele);
    ele->location_vector(*discret_, la);

    strategy.clear_element_storage(la[0].size(), la[0].size());

    // evaluate
    ele->evaluate(eleparams, *discret_, la, strategy.elematrix1(), strategy.elematrix2(),
        strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
    int eid = ele->id();
    strategy.assemble_matrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
    strategy.assemble_vector1(la[0].lm_, la[0].lmowner_);
  }
  strategy.complete();

  Core::LinAlg::SparseMatrix sysmatcopy(
      *(std::static_pointer_cast<Core::LinAlg::SparseMatrix>(systemmatrix1)));
  sysmatcopy.complete();

  // make a copy of system right-hand side vector
  Core::LinAlg::Vector<double> residualVec(*systemvector1);
  std::shared_ptr<Core::LinAlg::Vector<double>> fdvec =
      Core::LinAlg::create_vector(*dofrowmap, true);

  for (int k = 0; k < 16; ++k)
  {
    double eps = 1000;
    for (int j = 0; j < k; ++j) eps *= 0.1;

    // initialize tracking variable for maximum absolute and relative errors
    double maxabserr(0.);
    double maxrelerr(0.);

    // calculate fd matrix
    for (int colgid = 0; colgid <= sysmatcopy.col_map().max_all_gid(); ++colgid)
    {
      // check whether current column index is a valid global column index and continue loop if not
      int collid(sysmatcopy.col_map().lid(colgid));
      int maxcollid(-1);
      maxcollid = Core::Communication::max_all(collid, discret_->get_comm());
      if (maxcollid < 0) continue;

      strategy.zero();

      // fill state vector with original state variables
      phinp_->update(1., phinp_original, 0.);

      // impose perturbation and update interior variables
      if (phinp_->get_map().my_gid(colgid))
      {
        if (phinp_->sum_into_global_value(colgid, eps))
          FOUR_C_THROW(
              "Perturbation could not be imposed on state vector for finite difference check!");
      }
      update_interior_variables(intphitemp);

      discret_->clear_state(true);

      discret_->set_state("phiaf", *phinp_);
      discret_->set_state(nds_intvar_, "intphin", *intphin_);
      discret_->set_state(0, "phin", *phin_);
      discret_->set_state(nds_intvar_, "intphinp", *intphitemp);

      Teuchos::ParameterList eleparams;
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
          "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

      Core::Elements::LocationArray la(discret_->num_dof_sets());

      for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
      {
        Core::Elements::Element* ele = discret_->l_col_element(iele);
        ele->location_vector(*discret_, la);

        strategy.clear_element_storage(la[0].size(), la[0].size());
        ele->evaluate(eleparams, *discret_, la, strategy.elematrix1(), strategy.elematrix2(),
            strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
        int eid = ele->id();
        strategy.assemble_matrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
        strategy.assemble_vector1(la[0].lm_, la[0].lmowner_);
      }
      strategy.complete();
      fdvec->put_scalar(0.0);

      // finite difference suggestion (first divide by epsilon and then subtract for better
      // conditioning)
      for (int j = 0; j < phinp_->local_length(); ++j)
        (*fdvec).get_values()[j] = -(*systemvector1)[j] / eps + (residualVec)[j] / eps;

      for (int rowlid = 0; rowlid < discret_->dof_row_map()->num_my_elements(); ++rowlid)
      {
        // get global index of current matrix row
        const int rowgid = sysmatcopy.row_map().gid(rowlid);
        if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

        // get current entry in original system matrix
        double entry(0.);
        int length = sysmatcopy.num_my_entries(rowlid);
        int numentries;
        std::vector<double> values(length);
        std::vector<int> indices(length);
        sysmatcopy.extract_my_row_copy(rowlid, length, numentries, values.data(), indices.data());

        for (int ientry = 0; ientry < length; ++ientry)
        {
          if (sysmatcopy.col_map().gid(indices[ientry]) == colgid)
          {
            entry = values[ientry];
            break;
          }
        }

        // absolute and relative errors in first comparison
        const double abserr1 = entry - (*fdvec)[rowlid];
        double relerr1 = 0;
        if (abs(entry) > 1.e-17)
          relerr1 = abserr1 / abs(entry);
        else if (abs((*fdvec)[rowlid]) > 1.e-17)
          relerr1 = abserr1 / abs((*fdvec)[rowlid]);
        // store max abs and rel error
        if (abs(abserr1) > maxabserr) maxabserr = abs(abserr1);
        if (abs(relerr1) > maxrelerr) maxrelerr = abs(relerr1);
      }
    }
    // end calculate fd matrix

    // screen output
    if (myrank_ == 0)
    {
      std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR SCATRA HDG SYSTEM MATRIX" << std::endl;
      std::cout << "EPS:        " << eps << std::endl;
      std::cout << "ABSOLUTE: " << maxabserr << std::endl;
      std::cout << "RELATIVE: " << maxrelerr << std::endl;
    }
  }
  FOUR_C_THROW("FD check END");
}

/*----------------------------------------------------------------------------------*
 | compute relative error with reference to analytical solution    berardocco 05/20 |
 *----------------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::evaluate_error_compared_to_analytical_sol()
{
  switch (calcerror_)
  {
    case Inpar::ScaTra::calcerror_byfunction:
    case Inpar::ScaTra::calcerror_spherediffusion:
    {
      std::shared_ptr<Core::LinAlg::SerialDenseVector> errors = compute_error();
      if (errors == nullptr)
        FOUR_C_THROW("It was not possible to compute error. Check the error function number.");

      if (std::abs((*errors)[1]) > 1e-14)
        (*relerrors_)[0] = std::sqrt((*errors)[0]) / std::sqrt((*errors)[1]);
      else
        FOUR_C_THROW(
            "Can't compute scalar's relative L2 error due to numerical roundoff sensitivity!");
      if (std::abs((*errors)[3]) > 1e-14)
        (*relerrors_)[1] = std::sqrt((*errors)[2]) / std::sqrt((*errors)[3]);
      else
        FOUR_C_THROW(
            "Can't compute gradient's relative L2 error due to numerical roundoff sensitivity!");

      if (myrank_ == 0)
      {
        // print last error in a separate file
        const std::string simulation = problem_->output_control_file()->file_name();
        const std::string fname = simulation + "_time.relerror";
        std::ofstream f;

        // create new error file and write initial error
        if (step_ == 0)
        {
          f.open(fname.c_str());
          f << "| Step | Time | abs. L2-error (phi) | rel. L2-error (phi)| abs. L2-error "
               "(gradPhi)| rel. L2-error (gradPhi)|"
            << std::endl;
        }

        // append error of the last time step to the error file
        else
        {
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);

          f << step_ << " " << time_ << " " << std::setprecision(6) << std::sqrt((*errors)[0])
            << " " << (*relerrors_)[0] << " " << std::sqrt((*errors)[2]) << " " << (*relerrors_)[1]
            << std::endl;
        }

        f.flush();
        f.close();
      }

      break;
    }
    case Inpar::ScaTra::calcerror_no:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem!");
      break;
    }
  }
}  // ScaTra::TimIntHDG::evaluate_error_compared_to_analytical_sol

/*----------------------------------------------------------------------------------*
 | compute relative error with reference to analytical solution    berardocco 08/20 |
 *----------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SerialDenseVector> ScaTra::TimIntHDG::compute_error() const
{
  // If we are here it means that we either arrived at the end and we are checking the results or
  // that we specified that we want to compute the error. In any case, if the error function was not
  // specified, we just return a null pointer.
  const int errorfunctnumber = params_->get<int>("CALCERRORNO", -1);
  if (errorfunctnumber < 1) return nullptr;

  // create the parameters for the error calculation
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_error, eleparams);

  eleparams.set<int>("error function number", errorfunctnumber);
  eleparams.set<double>("time", time_);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("phiaf", *phinp_);
  discret_->set_state(nds_intvar_, "intphinp", *intphinp_);
  // get (squared) error values
  // The error is computed for the transported scalar and its gradient. Notice that so far only
  // the L2 error is computed, feel free to extend the calculations to any error measure needed
  unsigned int NumErrorEntries = 4;
  std::shared_ptr<Core::LinAlg::SerialDenseVector> errors =
      std::make_shared<Core::LinAlg::SerialDenseVector>(NumErrorEntries);

  discret_->evaluate_scalars(eleparams, *errors);
  discret_->clear_state();

  return errors;
}

/*----------------------------------------------------------------------*
 | prepare time loop                                     hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::prepare_time_loop()
{
  // calculate matrices on element
  calc_mat_initial();

  // call base class routine
  ScaTraTimIntImpl::prepare_time_loop();

}  // ScaTra::TimIntHDG::prepare_time_loop


/*----------------------------------------------------------------------*
 | calculate matrices on element                        hoermann 07/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::calc_mat_initial()
{
  TEUCHOS_FUNC_TIME_MONITOR("ScaTra::TimIntHDG::CalcMat");

  discret_->clear_state(true);

  // check validity of material and element formulation
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_initial, eleparams);

  discret_->set_state("phiaf", *phinp_);
  discret_->set_state("phin", *phin_);
  discret_->set_state(nds_intvar_, "intphin", *intphin_);
  discret_->set_state(nds_intvar_, "intphinp", *intphinp_);

  std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1, systemmatrix2;
  std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1, systemvector2, systemvector3;

  // create matrix and vector for calculation of sysmat and assemble
  systemmatrix1 = std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27);
  Core::FE::AssembleStrategy strategy(0, 0, sysmat_, nullptr, nullptr, nullptr, nullptr);

  strategy.zero();
  Core::Elements::LocationArray la(discret_->num_dof_sets());

  //    // get cpu time
  //    const double tcmatinit = Teuchos::Time::wallTime();

  // loop over elements
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Core::Elements::Element* ele = discret_->l_col_element(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->has_only_ghost_nodes(Core::Communication::my_mpi_rank(discret_->get_comm()))) continue;
    ele->location_vector(*discret_, la);

    strategy.clear_element_storage(la[0].size(), la[0].size());

    // evaluate
    int err = ele->evaluate(eleparams, *discret_, la, strategy.elematrix1(), strategy.elematrix2(),
        strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
    if (err)
      FOUR_C_THROW("Proc {}: Element {} returned err={}",
          Core::Communication::my_mpi_rank(discret_->get_comm()), ele->id(), err);

    int eid = ele->id();
    strategy.assemble_matrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
  }
  sysmat_->complete();

  //    // end time measurement for element
  //    double dtmatinit=Teuchos::Time::wallTime()-tcmatinit;
  //    std::cout << "Time measurement evaluate: " << dtmatinit << std::endl;

  // Output of non-zeros in system matrix
  if (step_ == 0 and Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    int numglobalnonzeros = system_matrix()->num_global_nonzeros();
    std::cout << "Number of non-zeros in system matrix: " << numglobalnonzeros << std::endl;
  }

  return;
}  // ScaTra::TimIntHDG::CalcMatInitial


/*----------------------------------------------------------------------*
 | adapt degree of test function on element               hoermann 07/16|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::adapt_degree()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + adapt degree");

  // get cpu time
  const double tcadapt = Teuchos::Time::wallTime();

  // cast and check if hdg discretization is provided
  Core::FE::DiscretizationHDG* hdgdis = dynamic_cast<Core::FE::DiscretizationHDG*>(discret_.get());
  if (hdgdis == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  // vector to store the dofs per single element
  const std::shared_ptr<Core::LinAlg::Vector<int>> eledofs =
      std::make_shared<Core::LinAlg::Vector<int>>(*discret_->element_col_map());

  // vector to store the location array of the dofsets before the adaption with the new order
  std::vector<Core::Elements::LocationArray> la_old;

  // copy the old face dof map and the old interior element dof map
  Core::LinAlg::Map facedofs_old(*discret_->dof_col_map(0));
  Core::LinAlg::Map eledofs_old(*discret_->dof_col_map(nds_intvar_));

  // set action
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_padaptivity, eleparams);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;

  discret_->set_state("phiaf", *phinp_);
  discret_->set_state(nds_intvar_, "intphinp", *intphinp_);

  // get cpu time
  //  const double tccalcerr = Teuchos::Time::wallTime();


  // store if degree changes
  int degchange(0);

  // loop over elements
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    // add new location array in vector for each element
    la_old.push_back(Core::Elements::LocationArray(discret_->num_dof_sets()));

    Core::Elements::Element* ele = discret_->l_col_element(iele);

    // fill location array and store it for later use
    ele->location_vector(*discret_, la_old[iele]);

    if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
    {
      Discret::Elements::ScaTraHDG* hdgele =
          dynamic_cast<Discret::Elements::ScaTraHDG*>(discret_->l_col_element(iele));

      // call routine on elements to calculate error on element
      ele->evaluate(
          eleparams, *discret_, la_old[iele], dummyMat, dummyMat, dummyVec, dummyVec, dummyVec);

      double error = eleparams.get<double>("error");
      double errorlog = 0;

      if (error < 0) FOUR_C_THROW("Error is negative!");

      if (error > 0)
        errorlog = log(error / padapterrortol_);
      else
        errorlog = 0.;

      int deg = hdgele->degree() + ceil(errorlog / padapterrorbase_);

      if (deg < 0)
        deg = 0;
      else if (deg > padaptdegreemax_)
        deg = padaptdegreemax_;

      if (hdgele->degree() != deg) degchange = 1;

      // set degree on element
      hdgele->set_degree(deg);

      // store element degree (only for output)
      const int eleIndex = discret_->element_row_map()->lid(ele->id());
      if (eleIndex >= 0) (*elementdegree_).get_values()[eleIndex] = deg;
    }
  }

  int degchangeall;
  degchangeall = Core::Communication::sum_all(degchange, discret_->get_comm());

  if (!degchangeall) return;

  pack_material();

  //  // end time measurement for element
  //  double dtcalcerr=Teuchos::Time::wallTime()-tccalcerr;
  //  std::cout << "Time measurement for error calculation: " << dtcalcerr << std::endl;

  //  // get cpu time
  //  const double tcfillcomplete = Teuchos::Time::wallTime();

  // number of dofset in location array
  int nds_intvar_old(discret_->num_dof_sets() + 2);
  int nds_var_old(discret_->num_dof_sets());

  // build new maps for face dofs with adapted element order
  hdgdis_->build_faces();
  hdgdis_->build_face_row_map();
  hdgdis_->build_face_col_map();

  // assign the degrees of freedom to the adapted dofsets
  hdgdis_->assign_degrees_of_freedom(0);

  // replace all ghosted element with the original thus the correct polynomial degree is used
  discret_->export_column_elements(*discret_->element_col_map(), false, false);

  hdgdis_->fill_complete();

  // store the number of dofs per element on vector
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Discret::Elements::ScaTraHDG* hdgele =
        dynamic_cast<Discret::Elements::ScaTraHDG*>(discret_->l_col_element(iele));
    // store the number of dofs for the element
    (*eledofs).get_local_values()[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // create new local dofset for the new interior element dofs with adapted element order
  std::shared_ptr<Core::DOFSets::DofSetPredefinedDoFNumber> eledofs_new =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, eledofs, 0, false);
  // replace old interior element dofs with the new created dofset
  discret_->replace_dof_set(nds_intvar_, eledofs_new, false);

  hdgdis_->assign_degrees_of_freedom(0);

  // clear map cache since after every fill_complete() / assign_degrees_of_freedom() old maps are
  // stored in the mapstack
  output_->clear_map_cache();

  // copy old values of the state vectors phi and intphi into vectors, which are then used for the
  // projection
  std::shared_ptr<Core::LinAlg::Vector<double>> phinp_old =
      Core::LinAlg::create_vector(facedofs_old, true);
  Core::LinAlg::export_to(*phinp_, *phinp_old);

  std::shared_ptr<Core::LinAlg::Vector<double>> intphinp_old =
      Core::LinAlg::create_vector(eledofs_old, true);
  Core::LinAlg::export_to(*intphinp_, *intphinp_old);

  // reset the residual, increment and sysmat to the size of the adapted new dofset
  residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  increment_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  neumann_loads_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  sysmat_ = nullptr;
  sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27);

  // reset the state vectors
  intphinp_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map(nds_intvar_)), true);
  intphin_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map(nds_intvar_)), true);
  phinp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));
  phin_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()));

  //  // end time measurement for element
  //  double dtfillcomplete=Teuchos::Time::wallTime()-tcfillcomplete;
  //  std::cout << "Time measurement fill complete: " << dtfillcomplete << std::endl;

  //  // get cpu time
  //  const double tcproject = Teuchos::Time::wallTime();

  // unpack material data
  unpack_material();

  adapt_variable_vector(
      phinp_, phinp_old, intphinp_, intphinp_old, nds_var_old, nds_intvar_old, la_old);

  intphin_->update(1.0, *intphinp_, 0.0);
  phin_->update(1.0, *phinp_, 0.0);

  project_material();

  //  // end time measurement for element
  //  double dtproject=Teuchos::Time::wallTime()-tcproject;
  //  std::cout << "Time measurement for projection: " << dtproject << std::endl;
  //
  //  // get cpu time
  //  const double tcmatinit = Teuchos::Time::wallTime();

  calc_mat_initial();

  //  // end time measurement for element
  //  double dtmatinit=Teuchos::Time::wallTime()-tcmatinit;
  //  std::cout << "Time measurement calc mat initial: " << dtmatinit << std::endl;

  // end time measurement for element
  double dtadapt = Teuchos::Time::wallTime() - tcadapt;

  if (myrank_ == 0)
    std::cout << "Time measurement for adaption of element degree: " << dtadapt << std::endl;


  return;
}

/*----------------------------------------------------------------------*
 | adapt trace vector and interior variables when adapting element      |
 | degrees                                                hoermann 07/16|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::adapt_variable_vector(std::shared_ptr<Core::LinAlg::Vector<double>> phi_new,
    std::shared_ptr<Core::LinAlg::Vector<double>> phi_old,
    std::shared_ptr<Core::LinAlg::Vector<double>> intphi_new,
    std::shared_ptr<Core::LinAlg::Vector<double>> intphi_old, int nds_var_old, int nds_intvar_old,
    std::vector<Core::Elements::LocationArray> la_old)
{
  // set action
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::project_field, eleparams);

  // set number of dofset for the old dofsets on the parameter list to extract the correct location
  // array
  eleparams.set<int>("nds_var_old", nds_var_old);
  eleparams.set<int>("nds_intvar_old", nds_intvar_old);

  // dof row map for adapted dofset
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map(0);


  // set old state vector on parameter list
  eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("phi", phi_old);
  eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("intphi", intphi_old);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector intphi_ele, phi_ele, dummyVec;

  // create location array for new and old dofsets (old ones are already filled and only copied to
  // the location array)
  Core::Elements::LocationArray la(2 * discret_->num_dof_sets());
  // create location array for new dofsets
  Core::Elements::LocationArray la_temp(discret_->num_dof_sets());

  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Core::Elements::Element* ele = discret_->l_col_element(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->has_only_ghost_nodes(Core::Communication::my_mpi_rank(discret_->get_comm()))) continue;

    // fill location array for adapted dofsets
    ele->location_vector(*discret_, la_temp);

    for (int i = 0; i < discret_->num_dof_sets(); i++)
    {
      // copy old and new location arrays to global location array la
      la[i] = la_temp[i];
      la[discret_->num_dof_sets() + i] = la_old[iele][i];
    }

    const unsigned size = la_temp[0].lm_.size();

    if (static_cast<std::size_t>(phi_ele.numRows()) != size)
      phi_ele.size(la[0].lm_.size());
    else
      phi_ele.putScalar(0.0);
    if (intphi_ele.numRows() != discret_->num_dof(nds_intvar_, ele))
      intphi_ele.size(discret_->num_dof(nds_intvar_, ele));
    else
      intphi_ele.putScalar(0.0);

    // call routine on elements to project values from old to new element vector
    ele->evaluate(eleparams, *discret_, la, dummyMat, dummyMat, phi_ele, intphi_ele, dummyVec);

    // store projected values of the element on the new state vector for the interior variables
    if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
    {
      std::vector<int> globalDofs = discret_->dof(nds_intvar_, ele);
      FOUR_C_ASSERT(
          globalDofs.size() == static_cast<std::size_t>(intphi_ele.numRows()), "Internal error");
      (intphi_new)
          ->replace_global_values(globalDofs.size(), intphi_ele.values(), globalDofs.data());
    }

    // now fill the element vector into the new state vector for the trace values
    for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
    {
      const int lid = dofrowmap->lid(la[0].lm_[i]);

      if (lid >= 0) (*phi_new).get_values()[lid] = phi_ele(i);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | chooses the assembly process for matrix and rhs       hoermann 06/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::assemble_mat_and_rhs()
{
  if (not params_->get<bool>("SEMIIMPLICIT"))
    ScaTra::ScaTraTimIntImpl::assemble_mat_and_rhs();
  else  // in semi-implicit evaluation matrix does not change, thus only rhs is assembled in every
        // step
    assemble_rhs();

  return;
}  // TimIntHDG::assemble_mat_and_rhs

/*----------------------------------------------------------------------*
 | contains the assembly process only for rhs            hoermann 06/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::assemble_rhs()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // reset the residual vector
  residual_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

  // set vector values needed by elements
  discret_->clear_state();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // add problem specific time-integration parameters
  add_problem_specific_parameters_and_vectors(eleparams);

  Core::FE::AssembleStrategy strategy(0, 0, nullptr, nullptr, residual_, nullptr, nullptr);

  strategy.zero();

  Core::Elements::LocationArray la(discret_->num_dof_sets());

  // loop over elements
  for (int iele = 0; iele < discret_->num_my_col_elements(); ++iele)
  {
    Core::Elements::Element* ele = discret_->l_col_element(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->has_only_ghost_nodes(Core::Communication::my_mpi_rank(discret_->get_comm()))) continue;

    ele->location_vector(*discret_, la);

    strategy.clear_element_storage(la[0].size(), la[0].size());

    // evaluate
    ele->evaluate(eleparams, *discret_, la, strategy.elematrix1(), strategy.elematrix2(),
        strategy.elevector1(), strategy.elevector2(), strategy.elevector3());
    strategy.assemble_vector1(la[0].lm_, la[0].lmowner_);
  }

  discret_->clear_state();

  // potential residual scaling and potential addition of Neumann terms
  scaling_and_neumann();

  // end time measurement for element
  dtele_ = Teuchos::Time::wallTime() - tcpuele;

  return;
}  // TimIntHDG::AssembleRHS

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> ScaTra::TimIntHDG::create_scatra_field_test()
{
  return std::make_shared<ScaTra::HDGResultTest>(Core::Utils::shared_ptr_from_ref(*this));
}

FOUR_C_NAMESPACE_CLOSE
