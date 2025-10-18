// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_levelset_algorithm.hpp"
#include "4C_levelset_intersection_utils.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc_utils.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | initialize or update velocity field                  rasthofer 03/14 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::set_velocity_field(bool init)
{
  // call function of base class
  ScaTraTimIntImpl::set_velocity_field_from_function();

  // note: This function is only called from the level-set dyn. This is ok, since
  //       we only want to initialize conveln_ at the beginning of the simulation.
  //       for the remainder, it is updated as usual. For the dependent velocity fields
  //       the base class function is called in prepare_time_step().
}


/*----------------------------------------------------------------------*
 | add problem depended params for assemble_mat_and_rhs    rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params)
{
  const auto add_multi_vector =
      [&](const std::string& name, std::shared_ptr<const Core::LinAlg::MultiVector<double>> vec)
  {
    auto tmp = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *discret_->node_col_map(), vec->num_vectors());
    Core::LinAlg::export_to(*vec, *tmp);
    params.set(name, tmp);
  };

  // set only special parameters of the solution of the reinitialization equation
  // otherwise we take the standard parameters only
  if (switchreinit_)
  {
    // action for elements
    params.set<bool>("solve reinit eq", true);

    if (reinitaction_ == Inpar::ScaTra::reinitaction_sussman)
    {
      // set initial phi, i.e., solution of level-set equation
      discret_->set_state("phizero", *initialphireinit_);
      // TODO: RM if not needed
      discret_->set_state("phin", *phin_);

#ifndef USE_PHIN_FOR_VEL
      if (useprojectedreinitvel_ == Inpar::ScaTra::vel_reinit_node_based)
        calc_node_based_reinit_vel();
#endif

      // add nodal velocity field, if required
      if (useprojectedreinitvel_ == Inpar::ScaTra::vel_reinit_node_based)
        add_multi_vector("reinitialization velocity field", nb_grad_val_);
    }
    else if (reinitaction_ == Inpar::ScaTra::reinitaction_ellipticeq)
    {
      // add node-based gradient, if required
      if (projection_ == true) add_multi_vector("gradphi", nb_grad_val_);

      // add interface integration cells
      params.set<std::shared_ptr<std::map<int, Core::Geo::BoundaryIntCells>>>(
          "boundary cells", interface_eleq_);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | capture interface                                    rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::capture_interface(
    std::map<int, Core::Geo::BoundaryIntCells>& interface, const bool writetofile)
{
  double volminus = 0.0;
  double volplus = 0.0;
  double surf = 0.0;
  // reconstruct interface and calculate volumes, etc ...
  ScaTra::LevelSet::Intersection intersect;
  intersect.capture_zero_level_set(*phinp_, *discret_, volminus, volplus, surf, interface);

  // do mass conservation check
  mass_conservation_check(volminus, writetofile);

  return;
}


/*----------------------------------------------------------------------*
 | mass conservation check                              rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::mass_conservation_check(
    const double actvolminus, const bool writetofile)
{
  if (myrank_ == 0)
  {
    // compute mass loss
    if (initvolminus_ != 0.0)
    {
      double massloss = -(1. - actvolminus / initvolminus_) * 100.0;

      // 'isnan' seems to work not reliably; error occurs in line above
      if (std::isnan(massloss)) FOUR_C_THROW("NaN detected in mass conservation check");

      Core::IO::cout << "---------------------------------------" << Core::IO::endl;
      Core::IO::cout << "           mass conservation" << Core::IO::endl;
      Core::IO::cout << " initial mass: " << std::setprecision(5) << initvolminus_
                     << Core::IO::endl;
      Core::IO::cout << " final mass:   " << std::setprecision(5) << actvolminus << Core::IO::endl;
      Core::IO::cout << " mass loss:    " << std::setprecision(5) << massloss << "%"
                     << Core::IO::endl;
      Core::IO::cout << "---------------------------------------" << Core::IO::endl;

      if (writetofile)
      {
        const std::string simulation = problem_->output_control_file()->file_name();
        const std::string fname = simulation + "_massconservation.relerror";

        if (step_ == 0)
        {
          std::ofstream f;
          f.open(fname.c_str());
          f << "#| Step | Time | mass loss w.r.t. minus domain |\n";
          f << "  " << std::setw(2) << std::setprecision(10) << step_ << "    " << std::setw(3)
            << std::setprecision(5) << time_ << std::setw(8) << std::setprecision(10) << "    "
            << massloss << " "
            << "\n";

          f.flush();
          f.close();
        }
        else
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "  " << std::setw(2) << std::setprecision(10) << step_ << "    " << std::setw(3)
            << std::setprecision(5) << time_ << std::setw(8) << std::setprecision(10) << "    "
            << massloss << " "
            << "\n";

          f.flush();
          f.close();
        }
      }
    }
    else
    {
      if (step_ > 0)
        Core::IO::cout
            << " there is no 'minus domain'! -> division by zero checking mass conservation"
            << Core::IO::endl;
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | calculate error compared to analytical solution      rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::evaluate_error_compared_to_analytical_sol()
{
  const auto calcerr =
      Teuchos::getIntegralValue<Inpar::ScaTra::CalcErrorLevelSet>(*levelsetparams_, "CALCERROR");

  switch (calcerr)
  {
    case Inpar::ScaTra::calcerror_no_ls:  // do nothing (the usual case)
      break;
    case Inpar::ScaTra::calcerror_initial_field:
    {
      if (myrank_ == 0)
      {
        if (step_ == 0)
        {
          const std::string simulation = problem_->output_control_file()->file_name();
          const std::string fname = simulation + "_shape.error";

          std::ofstream f;
          f.open(fname.c_str());
          f << "#| Step | Time | L1-err        | Linf-err        |\n";

          f.flush();
          f.close();
        }
      }

      if (step_ == stepmax_)  // do only at the end of the simulation
      {
        // create the parameters for the error calculation
        Teuchos::ParameterList eleparams;
        Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
            "action", ScaTra::Action::calc_error, eleparams);
        eleparams.set<Inpar::ScaTra::CalcErrorLevelSet>("calcerrorflag", calcerr);

        // get initial field
        const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
        Core::LinAlg::Vector<double> phiref(*dofrowmap, true);

        // get function
        int startfuncno = params_->get<int>("INITFUNCNO");
        if (startfuncno < 1) FOUR_C_THROW("No initial field defined!");

        for (auto lnode : discret_->my_row_node_range())
        {
          // the set of degrees of freedom associated with the node
          std::vector<int> nodedofset = discret_->dof(0, lnode);

          int numdofs = nodedofset.size();
          for (int k = 0; k < numdofs; ++k)
          {
            const int dofgid = nodedofset[k];
            int doflid = dofrowmap->lid(dofgid);
            // evaluate component k of spatial function
            double initialval =
                problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                    .evaluate(lnode.x(), time_, k);
            phiref.replace_local_value(doflid, initialval);
          }
        }

        // set vector values needed by elements
        discret_->clear_state();
        discret_->set_state("phinp", *phinp_);
        discret_->set_state("phiref", phiref);

        // get error and volume
        Core::LinAlg::SerialDenseVector errors(2);
        discret_->evaluate_scalars(eleparams, errors);
        discret_->clear_state();

        // division by thickness of element layer for 2D problems with domain size 1
        double errL1 = (errors)[0] / (errors)[1];
        Core::LinAlg::Vector<double> phidiff(*phinp_);
        phidiff.update(-1.0, phiref, 1.0);
        double errLinf = 0.0;
        phidiff.norm_inf(&errLinf);

        const std::string simulation = problem_->output_control_file()->file_name();
        const std::string fname = simulation + "_shape.error";

        if (myrank_ == 0)
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "  " << std::setw(2) << std::setprecision(10) << step_ << "    " << std::setw(3)
            << std::setprecision(5) << time_ << std::setw(8) << std::setprecision(10) << "    "
            << errL1 << std::setw(8) << std::setprecision(10) << "    " << errLinf << " "
            << "\n";

          f.flush();
          f.close();
        }
      }
    }
    break;
    default:
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem");
      break;
  }

  return;
}


/*------------------------------------------------------------------------------------------------*
 | compute convective velocity for contact points no-slip wall and interface      rasthofer 12/13 |
 *------------------------------------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::apply_contact_point_boundary_condition()
{
  // get condition
  std::vector<const Core::Conditions::Condition*> lscontactpoint;
  discret_->get_condition("LsContact", lscontactpoint);

  // map to store node gid and corrected values
  std::map<int, std::vector<double>> nodal_correction;

  // extract convective velocity field
  std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discret_->get_state(nds_vel(), "convective velocity field");
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  std::shared_ptr<Core::LinAlg::Vector<double>> convel_new =
      std::make_shared<Core::LinAlg::Vector<double>>(*convel);

  // loop all conditions
  for (std::size_t icond = 0; icond < lscontactpoint.size(); icond++)
  {
    const Core::Conditions::Condition* mycondition = lscontactpoint[icond];

    // loop all nodes belonging to this condition
    // for these nodes, new values have to be set
    const std::vector<int>* mynodes = mycondition->get_nodes();
    for (std::size_t inode = 0; inode < mynodes->size(); inode++)
    {
      // for all nodes on this proc: is this check really necessary here
      if (discret_->have_global_node((*mynodes)[inode]))
      {
        // get node
        Core::Nodes::Node* actnode = discret_->g_node((*mynodes)[inode]);

        // exclude ghosted nodes, since we need all adjacent elements here,
        // which are only available for nodes belonging to the this proc
        if (actnode->owner() == myrank_)
        {
          // initialize vector for averaged center velocity
          // note: velocity in scatra algorithm has three components (see also basic constructor)
          std::vector<double> averagedvel(3);
          for (int rr = 0; rr < 3; rr++) averagedvel[rr] = 0.0;

          // loop all adjacent elements
          for (auto adj_ele : actnode->adjacent_elements())
          {
            // get discretization type
            if (adj_ele.user_element()->shape() != Core::FE::CellType::hex8)
              FOUR_C_THROW("Currently only hex8 supported");
            const Core::FE::CellType distype = Core::FE::CellType::hex8;

            // in case of further distypes, move the following block to a templated function
            {
              // get number of element nodes
              const int nen = Core::FE::num_nodes(distype);
              // get number of space dimensions
              const int nsd = Core::FE::dim<distype>;

              // get nodal values of velocity field from secondary dofset
              Core::Elements::LocationArray la(discret_->num_dof_sets());
              adj_ele.user_element()->location_vector(*discret_, la);
              const std::vector<int>& lmvel = la[nds_vel()].lm_;
              std::vector<double> myconvel(lmvel.size());

              // extract local values from global vector
              myconvel = Core::FE::extract_values(*convel, lmvel);

              // determine number of velocity related dofs per node
              const int numveldofpernode = lmvel.size() / nen;

              Core::LinAlg::Matrix<nsd, nen> evel(Core::LinAlg::Initialization::zero);

              // loop over number of nodes
              for (int inode = 0; inode < nen; ++inode)
                // loop over number of dimensions
                for (int idim = 0; idim < nsd; ++idim)
                  evel(idim, inode) = myconvel[idim + (inode * numveldofpernode)];

              // use one-point Gauss rule to do calculations at the element center
              // used here to get center coordinates
              Core::FE::IntPointsAndWeights<nsd> centercoord(
                  ScaTra::DisTypeToStabGaussRule<distype>::rule);
              Core::LinAlg::Matrix<nsd, 1> xsi(Core::LinAlg::Initialization::zero);
              const double* gpcoord = (centercoord.ip().qxg)[0];
              for (int idim = 0; idim < nsd; idim++) xsi(idim, 0) = gpcoord[idim];

              // compute shape functions at element center
              Core::LinAlg::Matrix<nen, 1> funct(Core::LinAlg::Initialization::zero);
              Core::FE::shape_function<distype>(xsi, funct);

              // get velocity at integration point
              Core::LinAlg::Matrix<nsd, 1> velint(Core::LinAlg::Initialization::zero);
              velint.multiply(evel, funct);

              // add to averaged velocity vector
              for (int idim = 0; idim < nsd; idim++) averagedvel[idim] += velint(idim, 0);
            }
          }  // end loop elements

          // computed averaged value
          for (int rr = 0; rr < 3; rr++) averagedvel[rr] /= (double)(actnode->num_element());

          // store value in map, if not yet stored (i.e., multiple conditions intersect at one
          // point)
          if (nodal_correction.find(actnode->id()) == nodal_correction.end())
            nodal_correction.insert(
                std::pair<int, std::vector<double>>(actnode->id(), averagedvel));
        }
      }
    }  // end loop nodes
  }  // end loop conditions

  // replace values in velocity vector
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
  for (std::map<int, std::vector<double>>::iterator iter = nodal_correction.begin();
      iter != nodal_correction.end(); iter++)
  {
    const int gnodeid = iter->first;
    const int lnodeid = noderowmap->lid(gnodeid);
    Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);

    std::vector<int> nodedofs = discret_->dof(nds_vel(), lnode);

    std::vector<double> myvel = iter->second;
    for (int index = 0; index < 3; ++index)
    {
      // get global and local dof IDs
      const int gid = nodedofs[index];
      const int lid = convel_new->get_map().lid(gid);
      if (lid < 0) FOUR_C_THROW("Local ID not found in map for given global ID!");
      const double convelocity = myvel[index];
      convel_new->replace_local_value(lid, convelocity);
    }
  }

  // update velocity vectors
  discret_->set_state(nds_vel(), "convective velocity field", *convel_new);
  discret_->set_state(nds_vel(), "velocity field", *convel_new);

  return;
}



/*----------------------------------------------------------------------------*
 | Get Mass Center, using the smoothing function                 winter 06/14 |
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::mass_center_using_smoothing()
{
  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("phinp", *phinp_);

  // create the parameters for the error calculation
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mass_center_smoothingfunct, eleparams);

  // give access to interface thickness from smoothing function (TPF module) in element calculations
  eleparams.set<double>(
      "INTERFACE_THICKNESS_TPF", levelsetparams_->get<double>("INTERFACE_THICKNESS_TPF"));

  // get masscenter and volume, last entry of vector is total volume of minus domain.
  Core::LinAlg::SerialDenseVector masscenter_and_volume(nsd_ + 1);
  discret_->evaluate_scalars(eleparams, masscenter_and_volume);
  discret_->clear_state();

  std::vector<double> center(nsd_);

  for (int idim = 0; idim < nsd_; idim++)
  {
    center[idim] = masscenter_and_volume.values()[idim] / (masscenter_and_volume.values()[nsd_]);
  }

  if (nsd_ != 3)
    FOUR_C_THROW("Writing the mass center only available for 3 dimensional problems currently.");

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    // write to file
    const std::string simulation = problem_->output_control_file()->file_name();
    const std::string fname = simulation + "_center_of_mass.txt";

    if (step() == 0)
    {
      std::ofstream f;
      f.open(fname.c_str());
      f << "#| Step | Time |       x       |       y       |       z       |\n";
      f << "  " << std::setw(2) << std::setprecision(10) << step() << "    " << std::setw(3)
        << std::setprecision(5) << time() << std::setw(4) << std::setprecision(8) << "  "
        << center[0] << "    " << std::setprecision(8) << center[1] << "    "
        << std::setprecision(8) << center[2] << " "
        << "\n";

      f.flush();
      f.close();
    }
    else
    {
      std::ofstream f;
      f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
      f << "  " << std::setw(2) << std::setprecision(10) << step() << "    " << std::setw(3)
        << std::setprecision(5) << time() << std::setw(4) << std::setprecision(8) << "  "
        << center[0] << "    " << std::setprecision(8) << center[1] << "    "
        << std::setprecision(8) << center[2] << " "
        << "\n";

      f.flush();
      f.close();
    }
  }

  return;

}  // ScaTra::LevelSetAlgorithm::mass_center_using_smoothing


/*----------------------------------------------------------------------------*
 | redistribute the scatra discretization and vectors         rasthofer 07/11 |
 | according to nodegraph according to nodegraph              DA wichmann     |
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::redistribute(Core::LinAlg::Graph& nodegraph)
{
  // TODO: works if and only if discretization has already been redistributed
  //      change this and use unused nodegraph
  // TODO: does not work for gen-alpha, since time-integration dependent vectors have
  //      to be redistributed, too
  FOUR_C_THROW("Fix Redistribution!");
  //--------------------------------------------------------------------
  // Now update all vectors and matrices to the new dofmap
  //--------------------------------------------------------------------

  Core::FE::compute_null_space_if_necessary(*discret_, solver_->params(), true);

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // initialize standard (stabilized) system matrix (and save its graph!)
  // in standard case, but do not save the graph if fine-scale subgrid
  // diffusivity is used in non-incremental case
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no and not incremental_)
    // sysmat_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dofrowmap,27));
    // cf constructor
    sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 27, false, true);
  else
    sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 27, false, true);

  // -------------------------------------------------------------------
  // create vectors containing problem variables
  // -------------------------------------------------------------------

  // solutions at time n+1 and n
  std::shared_ptr<Core::LinAlg::Vector<double>> old;
  std::shared_ptr<Core::LinAlg::MultiVector<double>> oldMulti;

  if (phinp_ != nullptr)
  {
    old = phinp_;
    phinp_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *phinp_);
  }

  if (phin_ != nullptr)
  {
    old = phin_;
    phin_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *phin_);
  }

  // temporal solution derivative at time n+1
  if (phidtnp_ != nullptr)
  {
    old = phidtnp_;
    phidtnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *phidtnp_);
  }

  // temporal solution derivative at time n
  if (phidtn_ != nullptr)
  {
    old = phidtn_;
    phidtn_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *phidtn_);
  }

  // history vector (a linear combination of phinm, phin (BDF)
  // or phin, phidtn (One-Step-Theta, Generalized-alpha))
  if (hist_ != nullptr)
  {
    old = hist_;
    hist_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *hist_);
  }

  // -------------------------------------------------------------------
  // create vectors associated to boundary conditions
  // -------------------------------------------------------------------
  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  if (zeros_ != nullptr)
  {
    old = zeros_;
    zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *zeros_);
  }

  // -------------------------------------------------------------------
  // create vectors associated to solution process
  // -------------------------------------------------------------------
  // the vector containing body and surface forces
  if (neumann_loads_ != nullptr)
  {
    old = neumann_loads_;
    neumann_loads_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *neumann_loads_);
  }

  // the residual vector --- more or less the rhs
  if (residual_ != nullptr)
  {
    old = residual_;
    residual_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *residual_);
  }

  // residual vector containing the normal boundary fluxes
  if (trueresidual_ != nullptr)
  {
    old = trueresidual_;
    trueresidual_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *trueresidual_);
  }

  // incremental solution vector
  if (increment_ != nullptr)
  {
    old = increment_;
    increment_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *increment_);
  }

  // subgrid-diffusivity(-scaling) vector
  // (used either for AVM3 approach or temperature equation
  //  with all-scale subgrid-diffusivity model)
  if (subgrdiff_ != nullptr)
  {
    old = subgrdiff_;
    subgrdiff_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *subgrdiff_);
  }

  if (initialphireinit_ != nullptr)
  {
    old = initialphireinit_;
    initialphireinit_ = Core::LinAlg::create_vector(*dofrowmap, true);
    Core::LinAlg::export_to(*old, *initialphireinit_);
  }

  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no)
  {
    FOUR_C_THROW("No redistribution for AVM3 subgrid stuff.");
  }

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0) std::cout << "done" << std::endl;

  return;
}  // ScaTra::ScaTraTimIntImpl::redistribute

FOUR_C_NAMESPACE_CLOSE
