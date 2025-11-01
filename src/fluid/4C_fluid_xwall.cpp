// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_xwall.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_xwall.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_turbulence_transfer_turb_inflow.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_rebalance_graph_based.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 04/14 |
 *----------------------------------------------------------------------*/
FLD::XWall::XWall(std::shared_ptr<Core::FE::Discretization> dis, int nsd,
    std::shared_ptr<Teuchos::ParameterList>& params,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
    std::shared_ptr<FLD::Utils::StressManager> wssmanager)
    : discret_(dis), params_(params), mystressmanager_(wssmanager), iter_(0)
{
  // get the processor ID from the communicator
  myrank_ = Core::Communication::my_mpi_rank(discret_->get_comm());

  if (myrank_ == 0)
  {
    std::cout << "\nWall modeling with a Spalding's law enrichment" << std::endl;
  }

  // some exclusions and safety checks:
  if (nsd != 3) FOUR_C_THROW("Only 3D problems considered in xwall modelling!");
  if (Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(*params_, "time int algo") !=
          Inpar::FLUID::timeint_afgenalpha &&
      (Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(*params_, "time int algo") !=
          Inpar::FLUID::timeint_npgenalpha))
    FOUR_C_THROW(
        "Use Af-Genalpha for time integration in combination with xwall wall modeling. There would "
        "be additional updates necessary otherwise");

  if (params_->get<std::string>("predictor", "steady_state_predictor") != "steady_state")
    FOUR_C_THROW("The meshtying framework does only support a steady-state predictor");

  std::string tauwtype = params_->sublist("WALL MODEL").get<std::string>("Tauw_Type", "constant");

  if (tauwtype == "constant")
    tauwtype_ = Inpar::FLUID::constant;
  else if (tauwtype == "between_steps")
    tauwtype_ = Inpar::FLUID::between_steps;
  else
    FOUR_C_THROW("unknown Tauw_Type");

  std::string tauwcalctype =
      params_->sublist("WALL MODEL").get<std::string>("Tauw_Calc_Type", "residual");

  if (tauwcalctype == "residual")
    tauwcalctype_ = Inpar::FLUID::residual;
  else if (tauwcalctype == "gradient")
    tauwcalctype_ = Inpar::FLUID::gradient;
  else if (tauwcalctype == "gradient_to_residual")
    tauwcalctype_ = Inpar::FLUID::gradient_to_residual;
  else
    FOUR_C_THROW("unknown Tauw_Calc_Type");

  constant_tauw_ = params_->sublist("WALL MODEL").get<double>("C_Tauw");

  min_tauw_ = params_->sublist("WALL MODEL").get<double>("Min_Tauw");

  fac_ = params_->sublist("WALL MODEL").get<double>("Inc_Tauw");

  // get gauss points
  gp_norm_ = params_->sublist("WALL MODEL").get<int>("GP_Wall_Normal");
  gp_norm_ow_ = params_->sublist("WALL MODEL").get<int>("GP_Wall_Normal_Off_Wall");
  gp_par_ = params_->sublist("WALL MODEL").get<int>("GP_Wall_Parallel");

  // compute initial pressure
  int id = Global::Problem::instance()->materials()->first_id_by_type(Core::Materials::m_fluid);
  if (id == -1) FOUR_C_THROW("Newtonian fluid material could not be found");
  const Core::Mat::PAR::Parameter* mat =
      Global::Problem::instance()->materials()->parameter_by_id(id);
  const Mat::PAR::NewtonianFluid* actmat = static_cast<const Mat::PAR::NewtonianFluid*>(mat);
  dens_ = actmat->density_;
  visc_ = actmat->viscosity_ / dens_;  // here I want to have the kinematic viscosity


  std::string projectiontype = params_->sublist("WALL MODEL").get<std::string>("Projection", "No");

  if (projectiontype == "only_l2_projection")
    proj_ = true;
  else if (projectiontype == "No")
    proj_ = false;
  else
    FOUR_C_THROW("unknown projection type");

  std::string blendingtype =
      params_->sublist("WALL MODEL").get<std::string>("Blending_Type", "none");

  if (blendingtype == "none")
    blendingtype_ = Inpar::FLUID::none;
  else if (blendingtype == "ramp_function")
    blendingtype_ = Inpar::FLUID::ramp_function;
  else
    FOUR_C_THROW("unknown Blending_Type");

  inctauwnorm_ = 0.0;

  const int mlsmooth =
      (Global::Problem::instance()->fluid_dynamic_params()).get<int>("WSS_ML_AGR_SOLVER");
  if (mlsmooth != -1)
    smooth_res_aggregation_ = true;
  else
    smooth_res_aggregation_ = false;

  switch_step_ = params_->sublist("WALL MODEL").get<int>("Switch_Step");
  if (tauwcalctype_ == Inpar::FLUID::gradient_to_residual && switch_step_ < 2)
    FOUR_C_THROW("provide reasonable Switch_Step if you want to use gradient_to_residual");

  if (smooth_res_aggregation_ && tauwcalctype_ == Inpar::FLUID::gradient)
    FOUR_C_THROW(
        "smoothing of tauw works only for residual-based tauw, as the residual is smoothed");

  fix_residual_on_inflow_ =
      params_->sublist("WALL MODEL").get<bool>("Treat_Tauw_on_Dirichlet_Inflow");

  // output:
  if (myrank_ == 0)
  {
    std::cout << "\nXWall settings: " << std::endl;
    std::cout << "Tau_w is updated with:        " << tauwtype << std::endl;
    std::cout << "Tau_w is calculated with:     " << tauwcalctype << std::endl;
    std::cout << "Switching from grad to res:   " << switch_step_ << std::endl;
    std::cout << "Constant tau_w:               " << constant_tauw_ << std::endl;
    std::cout << "Minimum tau_w (clipping):     " << min_tauw_ << std::endl;
    std::cout << "Increment of tau_w:           " << fac_ << std::endl;
    std::cout << "Gauss rule:                   normal:  " << gp_norm_ << "  parallel:  " << gp_par_
              << "  overall:  " << gp_norm_ * gp_par_ * gp_par_ << std::endl;
    std::cout << "Gauss rule (off-wall):        normal:  " << gp_norm_ow_
              << "  parallel:  " << gp_par_ << "  overall:  " << gp_norm_ow_ * gp_par_ * gp_par_
              << std::endl;
    std::cout << "Enriched DOFs l2-projected:   " << projectiontype << std::endl;
    std::cout << "Blending method:              " << blendingtype << std::endl;
    std::cout << "Smooth tau_w:                 " << smooth_res_aggregation_ << std::endl;
    std::cout << "Solver for tau_w smoothing:   "
              << (Global::Problem::instance()->fluid_dynamic_params()).get<int>("WSS_ML_AGR_SOLVER")
              << std::endl;
    std::cout << "Solver for projection:        "
              << params_->sublist("WALL MODEL").get<int>("PROJECTION_SOLVER") << std::endl;
    std::cout << "Fix Dirichlet inflow:         " << fix_residual_on_inflow_ << std::endl;
    std::cout << std::endl;
    std::cout << "WARNING: ramp functions are used to treat fluid Mortar coupling conditions"
              << std::endl;
    std::cout << "WARNING: face element with enrichment not implemented" << std::endl;
    std::cout << "WARNING: stabilization terms for Neumann inflow only on standard space"
              << std::endl;
  }

  setup();

  turbulent_inflow_condition_ =
      std::make_shared<TransferTurbulentInflowConditionNodal>(discret_, dbcmaps);

  if (turbulent_inflow_condition_->is_active())
  {
    oldtauw_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_row_map()), true);
    oldinctauw_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_row_map()), true);
  }
  else
  {
    oldtauw_ = nullptr;
    oldinctauw_ = nullptr;
  }
}

/*----------------------------------------------------------------------*
 |  Set params required to build the shape functions           bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::set_x_wall_params(Teuchos::ParameterList& eleparams)
{
  // params required for the shape functions
  eleparams.set("walldist", wdist_);
  eleparams.set("tauw", tauw_);
  eleparams.set("inctauw", inctauw_);
  eleparams.set("xwalltoggle", xwalltoggle_);

  eleparams.set("mk", mkstate_);

  eleparams.set("gpnorm", gp_norm_);
  eleparams.set("gpnormow", gp_norm_ow_);
  eleparams.set("gppar", gp_par_);

  return;
}

/*----------------------------------------------------------------------*
 |  Set params required to build the shape functions           bk 08/14 |
 |  Used for the xwdiscret_, which is redistributed                     |
 *----------------------------------------------------------------------*/
void FLD::XWall::set_x_wall_params_xw_dis(Teuchos::ParameterList& eleparams)
{
  // params required for the shape functions
  eleparams.set("walldist", wdistxwdis_);
  eleparams.set("tauw", tauwxwdis_);
  eleparams.set("inctauw", inctauwxwdis_);
  eleparams.set("xwalltoggle", xwalltogglexwdis_);

  eleparams.set("mk", mkxwstate_);

  eleparams.set("gpnorm", gp_norm_);
  eleparams.set("gpnormow", gp_norm_ow_);
  eleparams.set("gppar", gp_par_);
  return;
}

/*----------------------------------------------------------------------*
 |  Setup XWall                                                bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::setup()
{
  if (myrank_ == 0) std::cout << "Setup: " << std::endl;

  init_x_wall_maps();

  init_wall_dist();

  setup_x_wall_dis();

  tauw_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_col_map()), true);
  inctauw_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_col_map()), true);

  // initialize for first call or for constant tauw setting
  tauw_->put_scalar(constant_tauw_);

  wdistxwdis_ = std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->node_col_map()), true);
  Core::LinAlg::export_to(*walldist_, *wdistxwdis_);

  tauwxwdis_ = std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->node_col_map()), true);
  inctauwxwdis_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->node_col_map()), true);

  // initialize for first call or for constant tauw setting
  tauwxwdis_->put_scalar(constant_tauw_);

  init_toggle_vector();

  setup_l2_projection();

  {
    // the value for linear elements is 1/3
    // initialize just in case
    mkxwstate_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->element_col_map()), true);
    mkxwstate_->put_scalar(0.33333333333);
    mkstate_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->element_col_map()), true);
    mkstate_->put_scalar(0.33333333333);

    restart_wss_ = nullptr;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Setup basic xwall map and dirichlet map                    bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::init_x_wall_maps()
{
  if (myrank_ == 0) std::cout << "- build xwall maps...                                   ";

  // build row vector of all xwall nodes
  {
    std::vector<int> rowvec;  // node row map
    for (int i = 0; i < discret_->node_row_map()->num_my_elements(); ++i)
    {
      int xwallgid = discret_->node_row_map()->gid(i);
      Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
      if (!xwallnode) FOUR_C_THROW("Cannot find node");

      bool enriched = false;

      // check if one of the surrounding elements is xwall element
      for (auto ele : xwallnode->adjacent_elements())
      {
        Discret::Elements::FluidXWall* xwallele =
            dynamic_cast<Discret::Elements::FluidXWall*>(ele.user_element());

        if (xwallele) enriched = true;
      }

      if (enriched) rowvec.push_back(xwallgid);
    }

    xwallrownodemap_ = std::make_shared<Core::LinAlg::Map>(
        -1, (int)rowvec.size(), rowvec.data(), 0, discret_->get_comm());
  }

  // get Dirichlet conditions
  std::vector<const Core::Conditions::Condition*> dircond;
  discret_->get_condition("FluidStressCalc", dircond);

  if (not dircond.empty())
  {
    std::vector<int> testcollect;
    int count = 0;
    for (unsigned numcond = 0; numcond < dircond.size(); ++numcond)
    {
      const std::vector<int>* test = dircond[numcond]->get_nodes();
      int j = 0;
      for (std::vector<int>::const_iterator i = (*test).begin(); i != (*test).end(); ++i)
      {
        ++count;
        testcollect.push_back((*test)[j]);
        ++j;
      }
    }

    int gcount;
    gcount = Core::Communication::sum_all(count, (discret_->get_comm()));
    dircolnodemap_ = std::make_shared<Core::LinAlg::Map>(
        gcount, count, testcollect.data(), 0, discret_->get_comm());
  }  // end loop this conditions
  else
    FOUR_C_THROW("You need DESIGN FLUID STRESS CALC SURF CONDITIONS for xwall");


  // map is of course not unique as it is a column map
  //  if(dircolnodemap_->UniqueGIDs())
  //    FOUR_C_THROW("Map resulting from DESIGN FLUID STRESS CALC SURF CONDITIONS not unique,
  //    probably node specified on two conditions?");

  if (myrank_ == 0)
    std::cout << xwallrownodemap_->num_global_elements() << " XWall nodes initialized!"
              << std::endl;
  if (xwallrownodemap_->num_global_elements() == 0) FOUR_C_THROW("No XWall elements found");
  return;
}


/*----------------------------------------------------------------------*
 |  Calculate wall distance and coupling matrix of for tauw    bk 04/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::init_wall_dist()
{
  if (myrank_ == 0) std::cout << "- calculate wall distance...                            ";

  tauwcouplingmattrans_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*xwallrownodemap_, 2, false, false);
  wdist_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_col_map()), true);
  walldist_ = std::make_shared<Core::LinAlg::Vector<double>>(*xwallrownodemap_, true);

  // build a new discretization which lies on all procs
  MPI_Comm newcomm(discret_->get_comm());

  // this is very expensive in terms of memory
  // we will delete it as soon as we are ready here
  std::shared_ptr<Core::FE::Discretization> commondis = std::make_shared<Core::FE::Discretization>(
      (std::string) "Commondis", newcomm, Global::Problem::instance()->n_dim());

  // loop over all column nodes of underlying problem discret and add
  for (int i = 0; i < (discret_->node_col_map())->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = discret_->l_col_node(i);
    if (!node) FOUR_C_THROW("Cannot find node with local id {}", i);
    std::shared_ptr<Core::Nodes::Node> newnode(node->clone());
    commondis->add_node(newnode->x(), newnode->id(), newnode);
  }
  // loop over all column elements of underlying problem discret and add
  for (int i = 0; i < (discret_->element_col_map())->num_my_elements(); ++i)
  {
    Core::Elements::Element* node = discret_->l_col_element(i);
    if (!node) FOUR_C_THROW("Cannot find ele with local id {}", i);
    std::shared_ptr<Core::Elements::Element> newnode(node->clone());
    commondis->add_element(newnode);
  }

  std::shared_ptr<Core::LinAlg::Map> testrednodecolmap =
      Core::LinAlg::allreduce_e_map(*(discret_->node_row_map()));
  commondis->export_column_nodes(*testrednodecolmap);

  // do not assign any dofs to save memory
  // only the nodes are needed in this discretization
  commondis->fill_complete(Core::FE::OptionsFillComplete::none());

  // also build a fully overlapping map of enriched nodes here:
  std::vector<int> colvec;  // node col map
  for (int i = 0; i < (commondis->node_col_map())->num_my_elements(); ++i)
  {
    int gid = commondis->node_col_map()->gid(i);
    Core::Nodes::Node* xwallnode = commondis->l_col_node(i);
    if (!xwallnode) FOUR_C_THROW("Cannot find node with local id {}", i);
    int enriched = 0;

    for (auto ele : xwallnode->adjacent_elements())
    {
      Discret::Elements::FluidXWall* xwallele =
          dynamic_cast<Discret::Elements::FluidXWall*>(ele.user_element());

      if (xwallele) enriched = 1;
    }
    int genriched = 0;
    genriched = Core::Communication::sum_all(enriched, (commondis->get_comm()));
    if (genriched > 0) colvec.push_back(gid);
  }
  int count = (int)colvec.size();

  xwallcolnodemap_ =
      std::make_shared<Core::LinAlg::Map>(count, count, colvec.data(), 0, discret_->get_comm());

  for (int j = 0; j < xwallcolnodemap_->num_my_elements(); ++j)
  {
    int xwallgid = xwallcolnodemap_->gid(j);

    Core::Nodes::Node* xwallnode = commondis->g_node(xwallgid);
    if (!xwallnode) FOUR_C_THROW("Cannot find node");

    double mydist = 1.0E10;
    double gdist = 1.0E10;
    int mygid = 0;


    for (int i = 0; i < dircolnodemap_->num_my_elements(); ++i)
    {
      int gid = dircolnodemap_->gid(i);

      if (discret_->node_row_map()->my_gid(gid))
      {
        Core::Nodes::Node* node = discret_->g_node(gid);

        if (!node) FOUR_C_THROW("ERROR: Cannot find wall node with gid %", gid);

        double newdist =
            sqrt(((xwallnode->x())[0] - (node->x())[0]) * ((xwallnode->x())[0] - (node->x())[0]) +
                 ((xwallnode->x())[1] - (node->x())[1]) * ((xwallnode->x())[1] - (node->x())[1]) +
                 ((xwallnode->x())[2] - (node->x())[2]) * ((xwallnode->x())[2] - (node->x())[2]));
        if (newdist < mydist)
        {
          mydist = newdist;
          mygid = gid;
        }
      }
    }

    gdist = Core::Communication::min_all(mydist, discret_->get_comm());

    // now write this value in the node based vector
    if (xwallrownodemap_->my_gid(xwallgid))
    {
      walldist_->replace_global_values(1, &gdist, &xwallgid);
    }

    // this is the processor that knows the respective node
    if (mydist == gdist)
    {
      tauwcouplingmattrans_->assemble(1.0, mygid, xwallgid);
    }
  }

  Core::LinAlg::export_to(*walldist_, *wdist_);
  tauwcouplingmattrans_->complete();

  double mean = 0.0;
  walldist_->mean_value(&mean);

  if (myrank_ == 0)
    std::cout << "the mean distance from the wall of all XWall nodes is: " << mean << "... ";

  if (myrank_ == 0) std::cout << "done!  " << std::endl;

  commondis = nullptr;
  return;
}

/*----------------------------------------------------------------------*
 |  Build a node-based toggle vector (on/off=1.0/0.0, 0.7)     bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::init_toggle_vector()
{
  if (myrank_ == 0) std::cout << "- build enriched/blending toggle vector for elements... ";

  xwalltoggle_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_col_map()), true);
  xwalltogglexwdis_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->node_col_map()), true);
  xtoggleloc_ = std::make_shared<Core::LinAlg::Vector<double>>(*xwallrownodemap_, true);
  int count = 0;
  for (int j = 0; j < xwallrownodemap_->num_my_elements(); ++j)
  {
    int xwallgid = xwallrownodemap_->gid(j);

    if (discret_->node_row_map()->my_gid(xwallgid))  // just in case
    {
      Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
      if (!xwallnode) FOUR_C_THROW("Cannot find node");

      bool fullyenriched = true;

      // Mortar interface
      std::vector<const Core::Conditions::Condition*> mortarcond =
          discret_->get_conditions_on_node("Mortar", xwallnode);
      if (not mortarcond.empty()) fullyenriched = false;

      // get all surrounding elements
      for (auto ele : xwallnode->adjacent_elements())
      {
        Discret::Elements::FluidXWall* xwallele =
            dynamic_cast<Discret::Elements::FluidXWall*>(ele.user_element());

        if (!xwallele) fullyenriched = false;
      }

      if (fullyenriched == true)
      {
        xtoggleloc_->replace_local_value(j, 1.0);
      }
      else
      {
        if (blendingtype_ != Inpar::FLUID::ramp_function)
        {
          xtoggleloc_->replace_local_value(j, 0.7);
        }
        count++;
      }
    }
  }

  Core::LinAlg::export_to(*xtoggleloc_, *xwalltoggle_);
  Core::LinAlg::export_to(*xtoggleloc_, *xwalltogglexwdis_);

  int gcount;
  gcount = Core::Communication::sum_all(count, (discret_->get_comm()));
  if (myrank_ == 0) std::cout << gcount << " blending nodes identified... ";

  if (myrank_ == 0) std::cout << "done!  " << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 |  Build a discretization only including xwall nodes/elements bk 07/14 |
 |  and redistribute                                                    |
 *----------------------------------------------------------------------*/
void FLD::XWall::setup_x_wall_dis()
{
  // build a new discretization
  MPI_Comm newcomm(discret_->get_comm());

  xwdiscret_ = std::make_shared<Core::FE::Discretization>(
      (std::string) "xwalldis", newcomm, Global::Problem::instance()->n_dim());

  // loop over all xwall row nodes and add
  for (int i = 0; i < (discret_->node_col_map())->num_my_elements(); ++i)
  {
    int gid = (discret_->node_col_map())->gid(i);

    if (xwallcolnodemap_->my_gid(gid))
    {
      Core::Nodes::Node* node = discret_->l_col_node(i);
      if (!node) FOUR_C_THROW("Cannot find node with local id {}", i);
      std::shared_ptr<Core::Nodes::Node> newnode(node->clone());
      xwdiscret_->add_node(newnode->x(), newnode->id(), newnode);
    }
  }
  // loop over all column elements of underlying problem discret and add
  for (int i = 0; i < discret_->num_my_col_elements(); ++i)
  {
    Core::Elements::Element* ele = discret_->l_col_element(i);
    if (!ele) FOUR_C_THROW("Cannot find ele with local id {}", i);

    Discret::Elements::FluidXWall* xwallele = dynamic_cast<Discret::Elements::FluidXWall*>(ele);

    if (!xwallele) continue;

    std::shared_ptr<Core::Elements::Element> newele(ele->clone());
    xwdiscret_->add_element(newele);
  }

  // make all conditions known to the child discretization
  // i.e. periodic boundary conditions, dirichlet conditions, ...
  {
    // get all conditions types prescribed in the input file
    std::vector<std::string> allcond;
    discret_->get_condition_names(allcond);
    // loop all conditions types
    for (unsigned numcond = 0; numcond < allcond.size(); ++numcond)
    {
      // get condition
      std::vector<const Core::Conditions::Condition*> actcond;
      discret_->get_condition(allcond[numcond], actcond);
      // loop all condition of the current type
      for (unsigned numactcond = 0; numactcond < actcond.size(); ++numactcond)
      {
        // finally set condition
        xwdiscret_->set_condition(allcond[numcond], actcond[numactcond]->copy_without_geometry());
      }
    }
  }


  // find out if we are in parallel; needed for TransparentDofSet
  bool parallel = Core::Communication::num_mpi_ranks(discret_->get_comm()) != 1;

  // dofs of the original discretization are used to set same dofs for the new discretization
  std::shared_ptr<Core::DOFSets::DofSet> newdofset =
      std::make_shared<Core::DOFSets::TransparentDofSet>(discret_, parallel);

  xwdiscret_->replace_dof_set(newdofset);
  xwdiscret_->fill_complete();

  // redistribute and treat periodic bc if parallel
  if (parallel)
  {
    // redistribute
    Core::LinAlg::Map elemap(*xwdiscret_->element_row_map());
    MPI_Comm comm(discret_->get_comm());

    std::shared_ptr<const Core::LinAlg::Graph> nodegraph =
        Core::Rebalance::build_graph(*xwdiscret_, elemap);

    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>(
        "num parts", std::to_string(Core::Communication::num_mpi_ranks(comm)));

    const auto& [rownodes, colnodes] =
        Core::Rebalance::rebalance_node_maps(*nodegraph, rebalanceParams);

    // rebuild of the system with new maps
    xwdiscret_->redistribute(
        {*rownodes, *colnodes}, {.fill_complete = Core::FE::OptionsFillComplete{
                                     .assign_degrees_of_freedom = false, .init_elements = false}});

    Core::Conditions::PeriodicBoundaryConditions pbc(xwdiscret_, false);
    pbc.update_dofs_for_periodic_boundary_conditions();
    xwdiscret_->replace_dof_set(newdofset);
    xwdiscret_->fill_complete();
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Setup matrix, vectors and solver for projection            bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::setup_l2_projection()
{
  // create matrix for projection
  if (proj_)
  {
    // build the dof maps of p and u_y

    std::vector<int> enrdf;  // enriched dofs

    for (int i = 0; i < xwdiscret_->node_row_map()->num_my_elements(); ++i)
    {
      int gid = xwdiscret_->node_row_map()->gid(i);
      // continue only if on this proc
      if (xwdiscret_->node_row_map()->my_gid(gid))
      {
        Core::Nodes::Node* node = xwdiscret_->g_node(gid);
        if (!node) FOUR_C_THROW("ERROR: Cannot find off wall node with gid %", gid);
        // make sure that periodic nodes are not assembled twice

        std::vector<const Core::Conditions::Condition*> periodiccond =
            xwdiscret_->get_conditions_on_node("SurfacePeriodic", node);
        // make sure that slave periodic bc are not included
        bool includedofs = true;
        if (not periodiccond.empty())
        {
          for (unsigned numcondper = 0; numcondper < periodiccond.size(); ++numcondper)
          {
            const std::string& mymasterslavetoggle =
                periodiccond[numcondper]->parameters().get<std::string>("MASTER_OR_SLAVE");
            if (mymasterslavetoggle == "Slave")
            {
              includedofs = false;
            }
          }
        }
        if (includedofs)
        {
          int firstglobaldofid = xwdiscret_->dof(0, node, 0);
          enrdf.push_back(firstglobaldofid + 4);
          enrdf.push_back(firstglobaldofid + 5);
          enrdf.push_back(firstglobaldofid + 6);
        }
      }
    }

    enrdofrowmap_ = std::make_shared<Core::LinAlg::Map>(
        -1, (int)enrdf.size(), enrdf.data(), 0, xwdiscret_->get_comm());

    massmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*enrdofrowmap_, 108, false, true);

    incveln_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
    incvelnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
    incaccn_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

    stateveln_ = std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->dof_row_map()), true);
    statevelnp_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->dof_row_map()), true);
    stateaccn_ = std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->dof_row_map()), true);

    mergedmap_ = nullptr;
    lagrdofrowmap_ = nullptr;
  }
  else
  {
    massmatrix_ = nullptr;
    incveln_ = nullptr;
    incvelnp_ = nullptr;
    incaccn_ = nullptr;
    stateveln_ = nullptr;
    statevelnp_ = nullptr;
    stateaccn_ = nullptr;
    enrdofrowmap_ = nullptr;
    lagrdofrowmap_ = nullptr;
  }

  // setup solver
  if (proj_)
  {
    const int solvernumber = params_->sublist("WALL MODEL").get<int>("PROJECTION_SOLVER");
    if (solvernumber < 0) FOUR_C_THROW("provide a solver number for l2-projection");
    // get solver parameter list of linear solver
    const Teuchos::ParameterList& solverparams =
        Global::Problem::instance()->solver_params(solvernumber);
    const auto solvertype =
        Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, xwdiscret_->get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));

    if (solvertype != Core::LinearSolver::SolverType::umfpack)
    {
      if (solvertype != Core::LinearSolver::SolverType::belos && myrank_ == 0)
        std::cout
            << "\nUse Belos as solver because it can handle several right hand sides at once!\n"
            << std::endl;
      const auto prectype =
          Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(solverparams, "AZPREC");
      // watch out: only ILU might work right now because of compute nullspace might not work...?
      // ... test!
      switch (prectype)
      {
        case Core::LinearSolver::PreconditionerType::multigrid_muelu:
        {
          if (proj_)
          {  // has 3 dofs, velocity dofs
            // BUT: enriched nodes have 8 dofs, so we have to calculate our own nullspace for 3 dofs
            // store nv and np at unique location in solver parameter list
            solver_->params().sublist("nodal_block_information").set("number of momentum dofs", 3);
            solver_->params()
                .sublist("nodal_block_information")
                .set("number of constraint dofs", 0);
            solver_->params().sublist("nodal_block_information").set("number of dofs per node", 3);
            solver_->params().sublist("nodal_block_information").set("nullspace dimension", 3);

            Teuchos::ParameterList* mllist_ptr = nullptr;
            mllist_ptr = &((solver_->params()).sublist("ML Parameters"));
            Teuchos::ParameterList& mllist = *mllist_ptr;  // solveparams.sublist("ML Parameters");

            std::shared_ptr<std::vector<double>> ns =
                mllist.get<std::shared_ptr<std::vector<double>>>("nullspace", nullptr);
            if (ns != nullptr) break;

            // no, we have not previously computed the nullspace
            // -> compute nullspace
            ns = nullptr;

            mllist.set<std::shared_ptr<std::vector<double>>>("nullspace", nullptr);
            // ML would not tolerate this Teuchos::rcp-ptr in its list otherwise
            mllist.set<bool>("ML validate parameter list", false);

            mllist.set("PDE equations", 3);
            mllist.set("null space: dimension", 3);
            mllist.set("null space: type", "pre-computed");
            mllist.set("null space: add default vectors", false);

            // allocate dimns times the local length of the rowmap
            const int lrows = enrdofrowmap_->num_my_elements();
            ns = std::make_shared<std::vector<double>>(3 * lrows);
            double* nullsp = ns->data();
            mllist.set<std::shared_ptr<std::vector<double>>>("nullspace", ns);
            mllist.set("null space: vectors", nullsp);

            // now compute the nullspace
            double* mode[6];
            for (int i = 0; i < 3; ++i) mode[i] = &((*ns)[i * lrows]);

            for (int i = 0; i < xwdiscret_->num_my_row_nodes(); ++i)
            {
              const unsigned int ndof = 3;
              Core::Nodes::Node* actnode = xwdiscret_->l_row_node(i);
              if (!actnode) FOUR_C_THROW("cannot find node");
              std::vector<int> dofs = xwdiscret_->dof(0, actnode);
              std::vector<int> actdofs;
              // only dof 4...6 (enriched dofs)
              actdofs.push_back(dofs.at(4));
              actdofs.push_back(dofs.at(5));
              actdofs.push_back(dofs.at(6));

              for (unsigned j = 0; j < ndof; ++j)
              {
                const int dof = actdofs.at(j);

                const int lid = enrdofrowmap_->lid(dof);
                if (lid < 0) FOUR_C_THROW("Cannot find dof {}", dof);

                for (unsigned k = 0; k < ndof; ++k)
                {
                  if (k % ndof == j % ndof)
                    mode[k % ndof][lid] = 1.0;
                  else
                    mode[k % ndof][lid] = 0.0;
                }
              }  // for (int j=0; j<actnode->Dof().NumDof(); ++j)
            }  // for (int i=0; i<NumMyRowNodes(); ++i)
          }
        }
        break;
        case Core::LinearSolver::PreconditionerType::ilu:
          // do nothing
          break;
        default:
          FOUR_C_THROW("You have to choose ML, MueLu or ILU preconditioning");
          break;
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Routine to update Tauw                                     bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::update_tau_w(int step, std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual,
    int itnum, std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln)
{
  iter_ = 0;

  transfer_and_save_tauw();

  Core::LinAlg::Vector<double> newtauw(*xwallrownodemap_, true);
  // calculate wall stress
  std::shared_ptr<Core::LinAlg::Vector<double>> wss;
  if (restart_wss_ != nullptr)
  {
    wss = restart_wss_;
  }
  else if (((tauwcalctype_ == Inpar::FLUID::gradient_to_residual && step >= switch_step_) ||
               (tauwcalctype_ != Inpar::FLUID::gradient_to_residual && step > 1)))
  {
    if (mystressmanager_ == nullptr) FOUR_C_THROW("wssmanager not available in xwall");
    // fix nodal forces on dirichlet inflow surfaces if desired
    wss = mystressmanager_->get_pre_calc_wall_shear_stresses(*fix_dirichlet_inflow(*trueresidual));
  }

  switch (tauwtype_)
  {
    case Inpar::FLUID::constant:  // works
    {
      if (step == 1) calc_mk();
      // tauw_ is constant and inctauw_ is zero
      return;
    }
    break;
    case Inpar::FLUID::between_steps:
    {
      inctauw_->put_scalar(0.0);

      if (itnum == 0)  // in between steps
        calc_tau_w(step, *velnp, wss.get());
      else
        return;
    }
    break;
    default:
      FOUR_C_THROW("unknown tauwtype_");
      break;
  }

  Core::LinAlg::export_to(*tauw_, newtauw);

  double actmean = -1.0;
  newtauw.mean_value(&actmean);

  double min = -1.0;
  newtauw.min_value(&min);
  if (min < 1e-10) FOUR_C_THROW("tauw is zero");
  double max = -1.0;
  newtauw.max_value(&max);

  // convergence check, works only if we don't take the mean
  newtauw.put_scalar(0.0);
  Core::LinAlg::export_to(*inctauw_, newtauw);
  newtauw.norm_2(&inctauwnorm_);
  // rescale inctauw to full increment
  if (fac_ > 1.0e-8) inctauwnorm_ /= fac_;

  if (myrank_ == 0)
    std::cout << "  min:  " << min << "  max:  " << max << "  mean-applied:  " << actmean
              << "  inc norm2:  " << inctauwnorm_;

  // also project the other vectors
  // later I could also run all three at the same time (with the same matrix)

  if (proj_)
  {
    if (myrank_ == 0) std::cout << "  L2-project... ";
    if (tauwtype_ == Inpar::FLUID::between_steps)
    {
      l2_project_vector(*veln, nullptr, accn);

      // at the beginning of this time step they are equal -> calculate only one of them
      velnp->update(1.0, *veln, 0.0);
    }
    else
      l2_project_vector(*veln, velnp, accn);

    if (myrank_ == 0) std::cout << "done!" << std::endl;
  }
  else
    std::cout << std::endl;

  calc_mk();

  // destruct vector so that we don't use it next time
  if (restart_wss_ != nullptr) restart_wss_ = nullptr;

  return;
}

/*----------------------------------------------------------------------*
 |  Routines to calculate Tauw                                 bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::calc_tau_w(
    int step, Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>* wss)
{
  Core::LinAlg::Vector<double> newtauw(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> newtauw2(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> tauw(*(discret_->node_col_map()), true);

  if (tauwcalctype_ == Inpar::FLUID::gradient_to_residual && switch_step_ == step && myrank_ == 0)
    std::cout << "\n switching from gradient to residual \n" << std::endl;


  if (tauwcalctype_ == Inpar::FLUID::residual ||
      (tauwcalctype_ == Inpar::FLUID::gradient_to_residual && step >= switch_step_))
  {
    FOUR_C_ASSERT_ALWAYS(wss, "No wall shear stress given.");
    for (int lnodeid = 0; lnodeid < dircolnodemap_->num_my_elements(); lnodeid++)
    {
      int gid = dircolnodemap_->gid(lnodeid);
      // continue only if on this proc
      if (discret_->node_row_map()->my_gid(gid))
      {
        Core::Nodes::Node* node = discret_->g_node(gid);
        if (!node) FOUR_C_THROW("ERROR: Cannot find off wall node with gid %", gid);

        int firstglobaldofid = discret_->dof(0, node, 0);
        int firstlocaldofid = wss->get_map().lid(firstglobaldofid);

        if (firstlocaldofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");
        double forcex = wss->local_values_as_span()[firstlocaldofid];
        double forcey = wss->local_values_as_span()[firstlocaldofid + 1];
        double forcez = wss->local_values_as_span()[firstlocaldofid + 2];

        double tauw = sqrt(forcex * forcex + forcey * forcey + forcez * forcez);

        // this is necessary since we are dividing by tauw on element level
        // also, the shape functions become singular, if tauw==0
        if (tauw < min_tauw_) tauw = min_tauw_;
        // store in vector
        newtauw.replace_global_value(gid, tauw);
      }
    }
  }
  else if (tauwcalctype_ == Inpar::FLUID::gradient ||
           (tauwcalctype_ == Inpar::FLUID::gradient_to_residual && step < switch_step_))
  {
    // necessary to set right state (the maps of the state vector and discretization have to be
    // equal)
    Core::LinAlg::Vector<double> statevel(*(xwdiscret_->dof_row_map()), true);
    Core::LinAlg::Vector<double> newtauwxwdis(*(xwdiscret_->node_row_map()), true);
    Core::LinAlg::export_to(velnp, statevel);

    xwdiscret_->set_state("vel", statevel);

    Core::LinAlg::Vector<double> timesvec(*(xwdiscret_->node_row_map()), true);

    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    // get number of elements
    const int numele = xwdiscret_->num_my_col_elements();

    // set action in order to project element void fraction to nodal void fraction
    Teuchos::ParameterList params;

    set_x_wall_params_xw_dis(params);

    params.set<FLD::Action>("action", FLD::tauw_via_gradient);

    // loop column elements: vector
    for (int i = 0; i < numele; ++i)
    {
      Core::Elements::Element* actele = xwdiscret_->l_col_element(i);

      const int numnode = actele->num_node();

      // get element location vector and ownerships
      actele->location_vector(*xwdiscret_, lm, lmowner, lmstride);

      elevector1.size(numnode);
      elevector2.size(numnode);

      // call the element specific evaluate method (elemat1 = mass matrix, elemat2 = rhs)
      // elevector1 has to be nullptr here, because I am assuming a dof-based vector otherwise
      actele->evaluate(
          params, *xwdiscret_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);

      // get element location vector for nodes
      lm.resize(numnode);
      lmowner.resize(numnode);

      Core::Nodes::Node** nodes = actele->nodes();
      for (int n = 0; n < numnode; ++n)
      {
        lm[n] = nodes[n]->id();
        lmowner[n] = nodes[n]->owner();
      }

      // assembling into node maps
      Core::LinAlg::assemble(newtauwxwdis, elevector1, lm, lmowner);
      Core::LinAlg::assemble(timesvec, elevector2, lm, lmowner);
    }  // end element loop

    xwdiscret_->clear_state();

    // scale with times:
    for (int l = 0; l < (xwdiscret_->node_row_map())->num_my_elements(); l++)
    {
      double sumnewtauw = newtauwxwdis.local_values_as_span()[l];
      double timesfac = timesvec.local_values_as_span()[l];
      double newtauwsc = 1.0;
      // we only have to do something, if we assembled at least once the same value
      if (timesfac > 0.5)
      {
        newtauwsc = sumnewtauw / timesfac;

        if (newtauwsc < min_tauw_) newtauwsc = min_tauw_;
        newtauwxwdis.replace_local_value(l, newtauwsc);
      }
    }
    Core::LinAlg::export_to(newtauwxwdis, newtauw);
  }
  else
    FOUR_C_THROW("unknown tauwcalctype_");

  tauw.update(1.0, *tauw_, 0.0);


  inctauw_->update(1.0, tauw, 0.0);
  tauw.put_scalar(0.0);

  tauwcouplingmattrans_->multiply(true, newtauw, newtauw2);
  double meansp = 0.0;
  newtauw2.mean_value(&meansp);

  Core::LinAlg::export_to(newtauw2, tauw);
  inctauw_->update(fac_, tauw, -fac_);  // now this is the increment (new-old)

  tauw_->update(1.0, *inctauw_, 1.0);

  overwrite_transferred_values();

  Core::LinAlg::export_to(*inctauw_, newtauw2);
  Core::LinAlg::export_to(newtauw2, *inctauwxwdis_);
  Core::LinAlg::export_to(*tauw_, newtauw2);
  Core::LinAlg::export_to(newtauw2, *tauwxwdis_);

  if (meansp < 2.0e-9)
    FOUR_C_THROW(
        "Average wall shear stress is zero. You probably forgot to specify appropriate DESIGN "
        "FLUID "
        "STRESS CALC SURF CONDITIONS where the stress should be calculated.");

  if (myrank_ == 0) std::cout << "tauw mean:  " << meansp;

  return;
}

/*----------------------------------------------------------------------*
 |  L2-project enriched dofs of vector                         bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::l2_project_vector(Core::LinAlg::Vector<double>& veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn)
{
  if (not veln.get_map().same_as(*discret_->dof_row_map()))
    FOUR_C_THROW("input map is not the dof row map of the fluid discretization");

  massmatrix_->zero();

  incveln_->put_scalar(0.0);
  if (accn != nullptr) incaccn_->put_scalar(0.0);
  if (velnp != nullptr) incvelnp_->put_scalar(0.0);

  Core::LinAlg::export_to(veln, *stateveln_);
  if (accn != nullptr) Core::LinAlg::export_to(*accn, *stateaccn_);
  if (velnp != nullptr) Core::LinAlg::export_to(*velnp, *statevelnp_);

  // number of right hand sides during solving
  // is the number of velocity components that is solved for
  // 3 since we are in 3D
  int numberofrhs = 0;
  if (velnp == nullptr && accn == nullptr)
    numberofrhs = 1;
  else if (velnp == nullptr || accn == nullptr)
    numberofrhs = 2;
  else
    numberofrhs = 3;

  xwdiscret_->set_state("veln", *stateveln_);
  if (accn != nullptr) xwdiscret_->set_state("accn", *stateaccn_);
  if (velnp != nullptr) xwdiscret_->set_state("velnp", *statevelnp_);

  // set action in order to project nodal enriched values to new shape functions
  Teuchos::ParameterList params;
  set_x_wall_params_xw_dis(params);
  params.set<FLD::Action>("action", FLD::xwall_l2_projection);

  // create empty right hand side
  std::shared_ptr<Core::LinAlg::MultiVector<double>> rhsassemble =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*enrdofrowmap_, numberofrhs);

  std::vector<int> lm;
  std::vector<int> lmowner;
  std::vector<int> lmstride;

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevectordummy;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // get number of elements
  const int numele = xwdiscret_->num_my_col_elements();

  // loop column elements
  for (int i = 0; i < numele; ++i)
  {
    Core::Elements::Element* actele = xwdiscret_->l_col_element(i);

    const int numnode = actele->num_node();
    const int numdf = 3;

    // get element location vector and ownerships
    actele->location_vector(*xwdiscret_, lm, lmowner, lmstride);

    // Reshape element matrices and vectors and initialize to zero
    elematrix1.shape(numnode * numdf, numnode * numdf);
    // Reshape element matrices and vectors and initialize to zero
    elematrix2.shape(
        numnode * numdf, numberofrhs);  // we have 3 right hand sides for now: 3 velocity components

    // call the element specific evaluate method (elemat1 = mass matrix, elemat2 = rhs)
    actele->evaluate(
        params, *xwdiscret_, lm, elematrix1, elematrix2, elevectordummy, elevector2, elevector3);

    // get element location vector for enriched dofs
    std::vector<int> lmassemble;
    std::vector<int> lmownerassemble;
    lmassemble.resize(numnode * numdf);
    lmownerassemble.resize(numnode * numdf);

    for (int n = 0; n < numnode; ++n)
    {
      for (int df = 4; df < 7; ++df)
      {
        lmassemble[n * numdf + df - 4] = lm[n * 8 + df];
        lmownerassemble[n * numdf + df - 4] = lmowner[n * 8 + df];
      }
    }

    // assembling into node maps
    massmatrix_->assemble(actele->id(), elematrix1, lmassemble, lmownerassemble);

    // assembling into node maps
    // assemble numberofrhs entries in rhs vector sequentially
    elevector1.size(numnode * numdf);
    for (int n = 0; n < numberofrhs; ++n)
    {
      // copy results into Serial_DenseVector for assembling
      for (int idf = 0; idf < numnode * numdf; ++idf) elevector1(idf) = elematrix2(idf, n);
      // assemble into nth vector of MultiVector
      Core::LinAlg::assemble(*rhsassemble, n, elevector1, lmassemble, lmownerassemble);
    }
  }  // end element loop

  xwdiscret_->clear_state();
  // finalize the matrix
  massmatrix_->complete();

  // solution vector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> resultvec =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*enrdofrowmap_, numberofrhs);

  // solve for 1, 2 or 3 right hand sides at the same time --> thanks to Belos
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve_with_multi_vector(massmatrix_, resultvec, rhsassemble, solver_params);

  // now copy result in original vector: the result is an increment of the velocity/ acceleration
  Core::LinAlg::export_to(((*resultvec)(0)), *incveln_);
  if (numberofrhs > 1) Core::LinAlg::export_to(((*resultvec)(1)), *incaccn_);
  if (numberofrhs > 2) Core::LinAlg::export_to(((*resultvec)(2)), *incvelnp_);

  veln.update(1.0, *incveln_, 1.0);
  if (accn != nullptr) accn->update(1.0, *incaccn_, 1.0);
  if (velnp != nullptr) velnp->update(1.0, *incvelnp_, 1.0);


  return;
}

/*----------------------------------------------------------------------*
 |  Adapt ML Nullspace for MFS aggregation                     bk 09/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::adapt_ml_nullspace(Core::LinAlg::Solver& solver)
{
  // extract the ML parameters:
  Teuchos::ParameterList& mlparams = solver.params().sublist("ML Parameters");

  // get nullspace parameters
  double* nullspace = mlparams.get("null space: vectors", (double*)nullptr);
  if (!nullspace) FOUR_C_THROW("No nullspace supplied in parameter list");
  int nsdim = mlparams.get("null space: dimension", 1);
  if (nsdim != 4) FOUR_C_THROW("Wrong Nullspace dimension for XWall");
  int lrowdofs = discret_->dof_row_map()->num_my_elements();
  //  std::cout << "lrowdofs  " << lrowdofs << std::endl;
  // std::cout << "check the nullspace for mfs" << std::endl;
  for (int j = 0; j < xwallrownodemap_->num_my_elements(); ++j)
  {
    int xwallgid = xwallrownodemap_->gid(j);

    if (not discret_->node_row_map()->my_gid(xwallgid))  // just in case
      FOUR_C_THROW("not on proc");
    {
      Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
      if (!xwallnode) FOUR_C_THROW("Cannot find node");

      int firstglobaldofid = discret_->dof(xwallnode, 0);
      int firstlocaldofid = discret_->dof_row_map()->lid(firstglobaldofid);

      nullspace[firstlocaldofid + 4] = 0.0;
      nullspace[lrowdofs + firstlocaldofid + 5] = 0.0;
      nullspace[lrowdofs * 2 + firstlocaldofid + 6] = 0.0;
      nullspace[lrowdofs * 3 + firstlocaldofid + 7] = 0.0;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Calculate MK for residual-based stabilization parameter    bk 09/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::calc_mk()
{
  Core::LinAlg::MultiVector<double> mkxw(*(xwdiscret_->element_row_map()), 1, true);

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<FLD::Action>("action", FLD::xwall_calc_mk);

  set_x_wall_params_xw_dis(eleparams);

  xwdiscret_->evaluate_scalars(eleparams, mkxw);


  Core::LinAlg::Vector<double> mkxwv(*(xwdiscret_->element_row_map()), true);
  Core::LinAlg::Vector<double> mkv(*(discret_->element_row_map()), true);

  // export
  Core::LinAlg::export_to(mkxw(0), mkxwv);
  Core::LinAlg::export_to(mkxwv, *mkxwstate_);
  Core::LinAlg::export_to(mkxwv, mkv);
  Core::LinAlg::export_to(mkv, *mkstate_);


  return;
}  // end calc_mk

/*----------------------------------------------------------------------*
 |  Write enriched dofs in standard dofs for output            bk 09/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::XWall::get_output_vector(
    Core::LinAlg::Vector<double>& vel)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> velenr =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  for (int i = 0; i < xwallrownodemap_->num_my_elements(); ++i)
  {
    int xwallgid = xwallrownodemap_->gid(i);
    Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
    if (!xwallnode) FOUR_C_THROW("Cannot find node");

    int firstglobaldofid = discret_->dof(xwallnode, 0);
    int firstlocaldofid = discret_->dof_row_map()->lid(firstglobaldofid);

    velenr->replace_local_value(firstlocaldofid, vel.local_values_as_span()[firstlocaldofid + 4]);
    velenr->replace_local_value(
        firstlocaldofid + 1, vel.local_values_as_span()[firstlocaldofid + 5]);
    velenr->replace_local_value(
        firstlocaldofid + 2, vel.local_values_as_span()[firstlocaldofid + 6]);
    velenr->replace_local_value(
        firstlocaldofid + 3, vel.local_values_as_span()[firstlocaldofid + 7]);
  }
  return velenr;
}

/*----------------------------------------------------------------------*
 |  transfer tauw for turbulent inflow channel                 bk 09/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::XWall::get_tauw() { return tauw_; }

/*----------------------------------------------------------------------*
 |  transfer tauw for turbulent inflow channel                 bk 09/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::transfer_and_save_tauw()
{
  if (turbulent_inflow_condition_->is_active())
  {
    Core::LinAlg::export_to(*tauw_, *oldtauw_);
    Core::LinAlg::export_to(*inctauw_, *oldinctauw_);
    turbulent_inflow_condition_->transfer(oldtauw_, oldtauw_, 0.0);
    turbulent_inflow_condition_->transfer(oldinctauw_, oldinctauw_, 0.0);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  transfer tauw for turbulent inflow channel                 bk 09/14 |
 *----------------------------------------------------------------------*/
void FLD::XWall::overwrite_transferred_values()
{
  if (turbulent_inflow_condition_->is_active())
  {
    Core::LinAlg::Vector<double> inctauwtmp(*(discret_->node_row_map()), true);
    Core::LinAlg::export_to(*inctauw_, inctauwtmp);
    Core::LinAlg::Vector<double> tauwtmp(*(discret_->node_row_map()), true);
    Core::LinAlg::export_to(*tauw_, tauwtmp);

    for (int i = 0; i < discret_->node_row_map()->num_my_elements(); ++i)
    {
      int xwallgid = discret_->node_row_map()->gid(i);
      Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
      if (!xwallnode) FOUR_C_THROW("Cannot find node");
      std::vector<const Core::Conditions::Condition*> nodecloudstocouple =
          discret_->get_conditions_on_node("TransferTurbulentInflow", xwallnode);
      if (not nodecloudstocouple.empty())
      {
        // usually we will only have one condition in nodecloudstocouple
        // but it doesn't hurt if there are several ones
        for (auto cond = nodecloudstocouple.begin(); cond != nodecloudstocouple.end(); ++cond)
        {
          const std::string& mytoggle = (*cond)->parameters().get<std::string>("toggle");
          if (mytoggle == "slave")
          {
            inctauwtmp.replace_local_value(i, oldinctauw_->local_values_as_span()[i]);
            tauwtmp.replace_local_value(i, oldtauw_->local_values_as_span()[i]);
          }
        }
      }
    }

    Core::LinAlg::export_to(inctauwtmp, *inctauw_);
    Core::LinAlg::export_to(tauwtmp, *tauw_);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Read Restart                                               bk 01/15 |
 *----------------------------------------------------------------------*/
void FLD::XWall::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> tauw =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->node_row_map()), true);
  reader.read_vector(tauw, "xwall_tauw");
  Core::LinAlg::export_to(*tauw, *tauw_);
  Core::LinAlg::export_to(*tauw, *tauwxwdis_);

  restart_wss_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  reader.read_vector(restart_wss_, "wss");
  return;
}


/*----------------------------------------------------------------------*
 |  treat Dirichlet inflow                                     bk 04/15 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::XWall::fix_dirichlet_inflow(
    Core::LinAlg::Vector<double>& trueresidual)
{
  // copy for safety reasons
  std::shared_ptr<Core::LinAlg::Vector<double>> fixedtrueresidual =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  fixedtrueresidual->update(1.0, trueresidual, 0.0);

  // fix nodal forces on dirichlet inflow surfaces
  if (fix_residual_on_inflow_)
  {
    Core::LinAlg::Vector<double> res(*(discret_->dof_col_map()), true);
    Core::LinAlg::export_to(trueresidual, res);
    for (int j = 0; j < xwallrownodemap_->num_my_elements(); ++j)
    {
      int xwallgid = xwallrownodemap_->gid(j);

      Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
      if (!xwallnode) FOUR_C_THROW("Cannot find node");
      std::vector<const Core::Conditions::Condition*> periodiccond =
          discret_->get_conditions_on_node("SurfacePeriodic", xwallnode);

      bool includedofs = true;
      if (not periodiccond.empty())
      {
        for (unsigned numcondper = 0; numcondper < periodiccond.size(); ++numcondper)
        {
          const std::string& mymasterslavetoggle =
              periodiccond[numcondper]->parameters().get<std::string>("MASTER_OR_SLAVE");
          if (mymasterslavetoggle == "Slave")
          {
            includedofs = false;
          }
        }
      }
      if (includedofs)
      {
        if (discret_->node_row_map()->my_gid(xwallgid))
        {
          std::vector<const Core::Conditions::Condition*> dircond =
              discret_->get_conditions_on_node("Dirichlet", xwallnode);

          std::vector<const Core::Conditions::Condition*> stresscond =
              discret_->get_conditions_on_node("FluidStressCalc", xwallnode);

          int numdf = discret_->num_dof(xwallnode);

          if ((not dircond.empty()) && (not stresscond.empty()) && numdf > 5)
          {
            bool isuglydirnode = false;
            for (unsigned numcond = 0; numcond < dircond.size(); ++numcond)
            {
              const auto& flag = dircond[numcond]->parameters().get<std::vector<int>>("ONOFF");

              if (flag[4] or flag[5] or flag[6]) isuglydirnode = true;
            }

            if (isuglydirnode)
            {
              //
              // the new node has to be on these as well
              //  std::vector<Core::Conditions::Condition*> dircond;
              //    discret_->GetCondition("FluidStressCalc",dircond);
              auto surrele = xwallnode->adjacent_elements();

              // loop over all surrounding elements and find indices of node k, l which is closes
              // while fulfilling all criteria
              double founddist = 1e9;
              int foundk = -1;
              int foundl = -1;
              for (int k = 0; k < (xwallnode->num_element()); ++k)  // loop over elements
              {
                auto test = surrele[k].nodes();
                for (size_t l = 0; l < surrele[k].num_nodes(); ++l)  // loop over nodes of element
                {
                  auto* current_node = test[l].user_node();
                  // it has to be on fluidstresscalc
                  // it may not be a dirichlet inflow node
                  // get Dirichlet conditions
                  std::vector<const Core::Conditions::Condition*> stresscond =
                      current_node->discretization()->get_conditions_on_node(
                          "FluidStressCalc", current_node);
                  int numdf = discret_->num_dof(current_node);
                  if (not stresscond.empty() and numdf > 5)
                  {
                    std::vector<const Core::Conditions::Condition*> dircond =
                        current_node->discretization()->get_conditions_on_node(
                            "Dirichlet", current_node);
                    bool isuglydirnode = false;
                    if (dircond.empty())
                    {
                      dircond = current_node->discretization()->get_conditions_on_node(
                          "FSICoupling", current_node);
                      if (dircond.empty())
                        FOUR_C_THROW("this should be a Dirichlet or fsi coupling node");
                    }
                    else
                    {
                      for (auto& numcond : dircond)
                      {
                        const auto& flag = numcond->parameters().get<std::vector<int>>("ONOFF");
                        if (flag[4] or flag[5] or flag[6]) isuglydirnode = true;
                      }
                    }

                    if (not isuglydirnode)
                    {
                      const auto& x = current_node->x();
                      double dist = abs(x[0] - xwallnode->x()[0]) + abs(x[1] - xwallnode->x()[1]) +
                                    abs(x[2] - xwallnode->x()[2]);
                      if (founddist > dist)
                      {
                        founddist = dist;
                        foundk = k;
                        foundl = l;
                      }
                    }
                  }
                }
              }

              if (foundk < 0 or foundl < 0) FOUR_C_THROW("haven't found required node");

              auto test = surrele[foundk].nodes();

              int firstglobaldofidtoreplace = discret_->dof(xwallnode, 0);
              int secondglobaldofidtoreplace = discret_->dof(xwallnode, 0) + 1;
              int thirdglobaldofidtoreplace = discret_->dof(xwallnode, 0) + 2;
              int firstglobaldofidnewvalue = discret_->dof(test[foundl].user_node(), 0);

              int firstlocaldofidnewvalue = discret_->dof_col_map()->lid(firstglobaldofidnewvalue);
              // half because the area is half on a boundary node compared to an inner node
              double newvalue1 = 0.5 * res.local_values_as_span()[firstlocaldofidnewvalue];
              double newvalue2 = 0.5 * res.local_values_as_span()[firstlocaldofidnewvalue + 1];
              double newvalue3 = 0.5 * res.local_values_as_span()[firstlocaldofidnewvalue + 2];

              fixedtrueresidual->replace_global_values(1, &newvalue1, &firstglobaldofidtoreplace);
              fixedtrueresidual->replace_global_values(1, &newvalue2, &secondglobaldofidtoreplace);
              fixedtrueresidual->replace_global_values(1, &newvalue3, &thirdglobaldofidtoreplace);
            }
          }
        }
      }
    }
  }
  return fixedtrueresidual;
}


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 01/15 |
 *----------------------------------------------------------------------*/
FLD::XWallAleFSI::XWallAleFSI(std::shared_ptr<Core::FE::Discretization> dis, int nsd,
    std::shared_ptr<Teuchos::ParameterList>& params,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
    std::shared_ptr<FLD::Utils::StressManager> wssmanager,
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> gridv)
    : XWall(dis, nsd, params, dbcmaps, wssmanager), mydispnp_(dispnp), mygridv_(gridv)
{
  incwdistxwdis_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*(xwdiscret_->node_col_map()), true);
  return;
}

void FLD::XWallAleFSI::update_w_dist_wale()
{
  // save old one for projection
  incwdistxwdis_->update(1.0, *wdistxwdis_, 0.0);

  Core::LinAlg::Vector<double> x(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> y(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> z(*xwallrownodemap_, true);

  // fill vectors with coords
  for (int j = 0; j < xwallrownodemap_->num_my_elements(); ++j)
  {
    int xwallgid = xwallrownodemap_->gid(j);

    if (not discret_->node_row_map()->my_gid(xwallgid))  // just in case
      FOUR_C_THROW("not on proc");
    Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
    if (!xwallnode) FOUR_C_THROW("Cannot find node");

    int firstglobaldofid = discret_->dof(xwallnode, 0);
    int firstlocaldofid = discret_->dof_row_map()->lid(firstglobaldofid);

    x.replace_local_value(
        j, xwallnode->x()[0] + mydispnp_->local_values_as_span()[firstlocaldofid]);
    y.replace_local_value(
        j, xwallnode->x()[1] + mydispnp_->local_values_as_span()[firstlocaldofid + 1]);
    z.replace_local_value(
        j, xwallnode->x()[2] + mydispnp_->local_values_as_span()[firstlocaldofid + 2]);
  }

  Core::LinAlg::Vector<double> wdistx(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> wdisty(*xwallrownodemap_, true);
  Core::LinAlg::Vector<double> wdistz(*xwallrownodemap_, true);

  // project coordinates of the closest wall node to the node itself
  tauwcouplingmattrans_->multiply(true, x, wdistx);
  tauwcouplingmattrans_->multiply(true, y, wdisty);
  tauwcouplingmattrans_->multiply(true, z, wdistz);

  // get delta
  wdistx.update(-1.0, x, 1.0);
  wdisty.update(-1.0, y, 1.0);
  wdistz.update(-1.0, z, 1.0);

  // fill vectors with coords
  for (int j = 0; j < xwallrownodemap_->num_my_elements(); ++j)
  {
    int xwallgid = xwallrownodemap_->gid(j);

    if (not discret_->node_row_map()->my_gid(xwallgid))  // just in case
      FOUR_C_THROW("not on proc");
    Core::Nodes::Node* xwallnode = discret_->g_node(xwallgid);
    if (!xwallnode) FOUR_C_THROW("Cannot find node");
    double x = wdistx.local_values_as_span()[j];
    double y = wdisty.local_values_as_span()[j];
    double z = wdistz.local_values_as_span()[j];
    double newwdist = sqrt(x * x + y * y + z * z);
    walldist_->replace_local_value(j, newwdist);
  }

  Core::LinAlg::export_to(*walldist_, *wdist_);
  Core::LinAlg::export_to(*walldist_, *wdistxwdis_);
  // save old one for projection
  incwdistxwdis_->update(1.0, *wdistxwdis_, -1.0);

  double mean = 0.0;
  walldist_->mean_value(&mean);

  if (myrank_ == 0)
    std::cout << "the new mean distance from the wall of all XWall nodes is: " << mean << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 |  Set params required to build the shape functions           bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWallAleFSI::set_x_wall_params(Teuchos::ParameterList& eleparams)
{
  XWall::set_x_wall_params(eleparams);
  return;
}

/*----------------------------------------------------------------------*
 |  Set params required to build the shape functions           bk 08/14 |
 |  Used for the xwdiscret_, which is redistributed                     |
 *----------------------------------------------------------------------*/
void FLD::XWallAleFSI::set_x_wall_params_xw_dis(Teuchos::ParameterList& eleparams)
{
  XWall::set_x_wall_params_xw_dis(eleparams);
  // params required for the shape functions
  eleparams.set("incwalldist", incwdistxwdis_);
  Core::LinAlg::Vector<double> xwdisdispnp(*(xwdiscret_->dof_row_map()), true);
  Core::LinAlg::export_to(*mydispnp_, xwdisdispnp);
  Core::LinAlg::Vector<double> xwdisgridv(*(xwdiscret_->dof_row_map()), true);
  Core::LinAlg::export_to(*mygridv_, xwdisgridv);

  xwdiscret_->set_state("dispnp", xwdisdispnp);
  xwdiscret_->set_state("gridv", xwdisgridv);

  return;
}

/*----------------------------------------------------------------------*
 |  Routine to update Tauw                                     bk 07/14 |
 *----------------------------------------------------------------------*/
void FLD::XWallAleFSI::update_tau_w(int step,
    std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual, int itnum,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln)
{
  update_w_dist_wale();
  FLD::XWall::update_tau_w(step, trueresidual, itnum, accn, velnp, veln);
  if (tauwtype_ == Inpar::FLUID::constant)
  {
    if (proj_)
    {
      if (myrank_ == 0) std::cout << "  L2-project... ";

      l2_project_vector(*veln, nullptr, accn);

      // at the beginning of this time step they are equal -> calculate only one of them
      velnp->update(1.0, *veln, 0.0);

      if (myrank_ == 0) std::cout << "done!" << std::endl;
    }
    else
      FOUR_C_THROW(
          "projection required for ale case even with constant tauw, since wdist is updating");
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
