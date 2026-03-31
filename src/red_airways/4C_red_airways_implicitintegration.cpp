// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_implicitintegration.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_maxwell_0d_acinus_Ogden.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_red_airways_evaluation_data.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_red_airways_resulttest.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <stdio.h>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 01/10|
 *----------------------------------------------------------------------*/
Airway::RedAirwayImplicitTimeInt::RedAirwayImplicitTimeInt(
    std::shared_ptr<Core::FE::Discretization> actdis, std::unique_ptr<Core::LinAlg::Solver> solver,
    Teuchos::ParameterList& params,
    Core::IO::DiscretizationWriter& output)
    :  // Call constructor for "nontrivial" objects
      discret_(actdis),
      solver_(std::move(solver)),
      params_(params),
      output_(output),
      time_(0.0),
      step_(0),
      uprestart_(params.get("write restart every", -1)),
      upres_(params.get("write solution every", -1)),
      coupledTo3D_(false)
{
  // Get the processor ID from the communicator
  myrank_ = Core::Communication::my_mpi_rank(discret_->get_comm());

  // Time measurement: initialization
  if (!coupledTo3D_)
  {
    // Time measurement: initialization
    TEUCHOS_FUNC_TIME_MONITOR(" + initialization");
  }

  // Get the basic parameters first
  // time-step size
  dtp_ = dta_ = params_.get<double>("time step size");
  // maximum number of timesteps
  stepmax_ = params_.get<int>("max number timesteps");
  // maximum simulation time
  maxtime_ = dtp_ * double(stepmax_);
  // maximum iteration steps
  maxiter_ = params_.get<int>("maximum iteration steps");
  // tolerance of nonlinear solution
  non_lin_tol_ = params_.get<double>("tolerance");
  // solve Aw-AC-Interdependency
  compAwAcInter_ = params_.get<bool>("CompAwAcInter");

  // calculate acini volume0 flag; option for acini volume adjustment via prestress
  calcV0PreStress_ = params_.get<bool>("CalcV0PreStress");
  // transpulmonary pressure, only needed in case of prestressing
  if (calcV0PreStress_) transpulmpress_ = params_.get<double>("transpulmpress");

  // ensure that degrees of freedom in the discretization have been set
  if (!discret_->filled() || !actdis->have_dofs()) discret_->fill_complete();

  airway_acinus_dep_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret_->element_col_map(), true);

  // extend ghosting of discretization to ensure correct neighbor search
  if (compAwAcInter_)
  {
    // to be filled with additional elements to be ghosted (needs to be done before
    // making discret fully overlapping)
    std::set<int> elecolset;
    const Core::LinAlg::Map* elecolmap = discret_->element_col_map();
    for (int lid = 0; lid < elecolmap->num_my_elements(); ++lid)
    {
      int gid = elecolmap->gid(lid);
      elecolset.insert(gid);
    }

    // to be filled with additional nodes to be ghosted
    std::set<int> nodecolset;
    const Core::LinAlg::Map* nodecolmap = discret_->node_col_map();
    for (int lid = 0; lid < nodecolmap->num_my_elements(); ++lid)
    {
      int gid = nodecolmap->gid(lid);
      nodecolset.insert(gid);
    }

    // make search discret fully overlapping on all procs
    Core::Rebalance::ghost_discretization_on_all_procs(*discret_);
    discret_->fill_complete(Core::FE::OptionsFillComplete::none());

    // Get elements and nodes that need to be ghosted to have correct neighbor search
    // independent of number of procs
    compute_nearest_acinus(*discret_, &elecolset, &nodecolset, nullptr);

    // extended ghosting for elements (also revert fully overlapping here)
    std::vector<int> coleles(elecolset.begin(), elecolset.end());
    const Core::LinAlg::Map extendedelecolmap(
        -1, coleles.size(), coleles.data(), 0, discret_->get_comm());

    discret_->export_column_elements(extendedelecolmap);

    // extended ghosting for nodes
    std::vector<int> colnodes(nodecolset.begin(), nodecolset.end());
    const Core::LinAlg::Map extendednodecolmap(
        -1, colnodes.size(), colnodes.data(), 0, discret_->get_comm());

    discret_->export_column_nodes(extendednodecolmap);

    // fill and inform user (not fully overlapping anymore at this point
    discret_->fill_complete();
    Core::Rebalance::print_parallel_distribution(*discret_);

    // Neighbouring acinus
    airway_acinus_dep_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->element_col_map(), true);
    compute_nearest_acinus(*discret_, nullptr, nullptr, airway_acinus_dep_);
  }

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  const Core::LinAlg::Map* dofcolmap = discret_->dof_col_map();
  const Core::LinAlg::Map* elementcolmap = discret_->element_col_map();
  const Core::LinAlg::Map* elementrowmap = discret_->element_row_map();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization for a vector which only
  // contains the volumetric flow rate dofs and for one vector which only
  // contains cross-sectional area degrees of freedom.
  // -------------------------------------------------------------------


  // This is a first estimate for the number of non zeros in a row of
  // the matrix. Each node has 3 adjacent nodes (including itself), each
  // with 1 dofs. (3*1=3)
  // We do not need the exact number here, just for performance reasons
  // a 'good' estimate

  // initialize standard (stabilized) system matrix
  sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 3, false, true);

  // Vectors passed to the element
  // Pressures at time n+1, n and n-1
  pnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  on_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  pnm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  p_nonlin_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  n_intr_ac_ln_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // Inlet volumetric flow rates at time n+1, n and n-1
  qin_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  qin_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  qin_nm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // Trajectory vector x at time n+1 and n
  x_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  x_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // State of airway
  open_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // External pressure
  p_extnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  p_extn_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  pnp_colmap_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  on_colmap_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // Outlet volumetric flow rates at time n+1, n and n-1
  qout_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  qout_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  qout_nm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // This vector will be used for exportation and restart reasons
  qexp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementrowmap, true);
  qexp2_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementrowmap, true);
  pexp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // Element volume at time n+1, n and n-1
  elemVolumenp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  elemVolumen_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  elemVolumenm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  elemVolume0_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  elemArea0_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // Element radius at time n+1
  elemRadiusnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // This vector will be used to test convergence
  residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  bc_residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofcolmap, true);

  // Vectors for postprocessing, Element Node Ids, radii, generations, etc ...
  nodeIds_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  radii_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  generations_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // A vector of zeros to be used to enforce zero dirichlet boundary conditions
  // This part might be optimized later
  bcval_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  dbctog_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  acini_bc_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_e_volume0_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_e_volumenm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_e_volumen_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_e_volumenp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_e_volume_strain_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  acini_max_strain_location_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // Vectors used for solution process
  // right hand side vector and right hand side corrector
  rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // ---------------------------------------------------------------------------------------
  // Initialize all the arteries' cross-sectional areas to the initial crossectional area Ao
  // and the volumetric flow rate to 0
  // ---------------------------------------------------------------------------------------
  Teuchos::ParameterList eleparams;

  // loop all elements and initialize all of the values

  // note: We use an RCP because ParameterList wants something printable and comparable
  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  evaluation_data.p0np = pnp_;
  evaluation_data.p0n = on_;
  evaluation_data.p0nm = pnm_;

  evaluation_data.generations = generations_;
  evaluation_data.acini_bc = acini_bc_;
  evaluation_data.acini_e_volume = acini_e_volumenp_;
  evaluation_data.elemVolume = elemVolumenp_;
  evaluation_data.elemArea0 = elemArea0_;
  eleparams.set("action", "get_initial_state");

  std::shared_ptr<Core::LinAlg::Vector<double>> radii_in =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> radii_out =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  discret_->evaluate(eleparams, nullptr, nullptr, radii_in, radii_out, n_intr_ac_ln_);

  for (int i = 0; i < radii_->local_length(); i++)
  {
    if ((radii_in->local_values_as_span()[i]) == 0.0)
    {
      (*radii_).get_values()[i] = radii_out->local_values_as_span()[i];
    }
    else if (radii_out->local_values_as_span()[i] == 0.0)
    {
      (*radii_).get_values()[i] = radii_in->local_values_as_span()[i];
    }
    else
    {
      (*radii_).get_values()[i] =
          0.5 * (radii_in->local_values_as_span()[i] + radii_out->local_values_as_span()[i]);
    }
  }

  acini_e_volumen_->update(1.0, *acini_e_volumenp_, 0.0);
  acini_e_volumenm_->update(1.0, *acini_e_volumenp_, 0.0);
  acini_e_volume0_->update(1.0, *acini_e_volumenp_, 0.0);
  elemVolumen_->update(1.0, *elemVolumenp_, 0.0);
  elemVolumenm_->update(1.0, *elemVolumenp_, 0.0);
  elemVolume0_->update(1.0, *elemVolumenp_, 0.0);

  // Fill the NodeId vector
  for (int nele = 0; nele < discret_->num_my_col_elements(); ++nele)
  {
    // get the element
    Core::Elements::Element* ele = discret_->l_col_element(nele);

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmstride;
    // vector<int> lmowner;
    std::vector<int> lmowner;
    ele->location_vector(*discret_, lm, lmowner, lmstride);

    // loop all nodes of this element, add values to the global vectors

    if (myrank_ == (lmowner)[0])
    {
      int gid = lm[0];
      double val = gid;
      nodeIds_->replace_global_values(1, &val, &gid);
    }
    if (myrank_ == (lmowner)[1])
    {
      int gid = lm[1];
      double val = gid;
      nodeIds_->replace_global_values(1, &val, &gid);
    }
  }

  // Set initial open/close value for an airway
  for (int j = 0; j < (discret_->num_my_col_elements()); j++)
  {
    // check if element is an airway
    if ((generations_->local_values_as_span()[j] != -1) and
        (generations_->local_values_as_span()[j] != -2))
    {
      int GID = discret_->element_col_map()->gid(j);  // global element ID
      const Core::Elements::ElementType& ele_type = discret_->g_element(GID)->element_type();
      if (ele_type == Discret::Elements::RedAirwayType::instance())
      {
        // dynamic cast to airway element, since Elements base class does not have the functions
        // getParams and setParams
        Discret::Elements::RedAirway* ele =
            dynamic_cast<Discret::Elements::RedAirway*>(discret_->g_element(GID));
        const auto airway_params = ele->get_airway_params();
        // check if airway is collapsible
        const double airwayColl = airway_params.airway_coll;
        if (airwayColl == 1)
        {
          const double val = airway_params.open_init;

          // adjust airway states
          (*x_np_).get_values()[j] = val;
          (*x_n_).get_values()[j] = val;
          //(*open_)[j] = val;
        }
        const double val = airway_params.open_init;
        (*open_).get_values()[j] = val;
      }
    }
  }

}  // RedAirwayImplicitTimeInt::RedAirwayImplicitTimeInt


/*----------------------------------------------------------------------*
 | Integrate () routine to start the time integration.                  |
 |                                                          ismail 01/10|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::integrate()
{
  std::shared_ptr<Teuchos::ParameterList> param;
  integrate(false, param);
}  // RedAirwayImplicitTimeInt::Integrate()


/*----------------------------------------------------------------------*
 | Integrate () routine to start the time integration.                  |
 |                                                          ismail 01/10|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::integrate(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingParams)
{
  // Do prestressing if required
  if (calcV0PreStress_)
  {
    compute_vol0_for_pre_stress();
  }

  // Get coupling parameters in case of 3D/0D coupling
  coupledTo3D_ = CoupledTo3D;
  if (CoupledTo3D && CouplingParams.get() == nullptr)
  {
    FOUR_C_THROW(
        "Coupling parameter list is not allowed to be empty, If a 3-D/reduced-D coupling is "
        "defined\n");
  }

  // Start time loop
  time_loop(CoupledTo3D, CouplingParams);
}  // RedAirwayImplicitTimeInt::Integrate


/*-----------------------------------------------------------------------------*
 | Prestress the lung to a given transpulmonary pressure given in the input file|
 | This will shink the lung before the first timestep in such a way that the    |
 | volume of each acinus reaches the given volume in the input file when p_tp is|
 | applied                                                                      |
 *-----------------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::compute_vol0_for_pre_stress()
{
  double p = transpulmpress_;

  // loop over number of elements (on processor)
  for (int i = 0; i < (discret_->num_my_col_elements()); i++)
  {
    // check if element is an acinus
    if ((generations_->local_values_as_span()[i]) == -1)
    {
      int GID = discret_->element_col_map()->gid(i);  // global element ID

      // check if aciuns is ogden type material
      if (discret_->g_element(GID)->material(0)->material_type() ==
          Core::Materials::m_0d_maxwell_acinus_ogden)
      {
        // get material parameters kappa and beta
        std::shared_ptr<Mat::Maxwell0dAcinusOgden> mymat =
            std::dynamic_pointer_cast<Mat::Maxwell0dAcinusOgden>(
                discret_->g_element(GID)->material(0));
        double kappa = mymat->get_params("kappa");
        double beta = mymat->get_params("beta");

        // calculate alpha
        double alpha = 0.1;  // starting value for approximation
        double f;
        double f_der;

        // get approximation for alpha with newton-raphson method
        while (std::abs(p - kappa / beta / alpha * (1 - std::pow(alpha, -beta))) > 1e-06)
        {
          f = p - kappa / beta / alpha * (1 - std::pow(alpha, -beta));
          f_der = kappa / beta * std::pow(alpha, -2.0) +
                  kappa / beta * (1 - beta) * std::pow(alpha, -beta);
          // get new alpha
          alpha = alpha - f / f_der;
        }

        // adjust acinus volume 0 in the elements parameters
        // additional check whether element is RedAcinusType
        const Core::Elements::ElementType& ele_type = discret_->g_element(GID)->element_type();
        if (ele_type == Discret::Elements::RedAcinusType::instance())
        {
          // dynamic cast to aciunus element, since Elements base class does not have the functions
          // getParams and setParams
          auto* acini_ele = dynamic_cast<Discret::Elements::RedAcinus*>(discret_->g_element(GID));
          const auto acinus_params = acini_ele->get_acinus_params();
          // get original value for aciuns volume (entered in input file)
          double val = acinus_params.volume_init;
          // calculate new value for aciuns volume with alpha and set in element parameters
          val = val / alpha;
          acini_ele->update_relaxed_volume(val);

          // adjust acini volumes in the vectors used in this function
          if (not Global::Problem::instance()->restart())
          {
            (*acini_e_volumenp_).get_values()[i] = val;
            (*acini_e_volumen_).get_values()[i] = val;
            (*acini_e_volumenm_).get_values()[i] = val;
            (*acini_e_volume0_).get_values()[i] = val;
          }
        }
      }
      else
      {
        std::cout << "Warning! Acinus " << GID
                  << " is not Ogden type material! Initial volume cannot be adjusted!" << std::endl;
      }
    }
  }
}

void Airway::RedAirwayImplicitTimeInt::compute_nearest_acinus(
    const Core::FE::Discretization& search_discret, std::set<int>* elecolset,
    std::set<int>* nodecolset, std::shared_ptr<Core::LinAlg::Vector<double>> airway_acinus_dep)
{
  // Loop over all airways contained on this proc
  for (int j = 0; j < (search_discret.num_my_col_elements()); j++)
  {
    // global element ID airway
    int GID1 = search_discret.element_col_map()->gid(j);

    // check if element is airway element
    Discret::Elements::RedAirway* ele_aw =
        dynamic_cast<Discret::Elements::RedAirway*>(search_discret.g_element(GID1));

    // check if element j is an airway
    if (ele_aw != nullptr)
    {
      double diff_norm = 1e3;
      double min_norm = 1e3;
      int min_index = 0;

      // Get coordinates for airway center point
      double node_coords1[3];
      node_coords1[0] = ele_aw->nodes()[0]->x()[0];
      node_coords1[1] = ele_aw->nodes()[0]->x()[1];
      node_coords1[2] = ele_aw->nodes()[0]->x()[2];

      double node_coords2[3];
      node_coords2[0] = ele_aw->nodes()[1]->x()[0];
      node_coords2[1] = ele_aw->nodes()[1]->x()[1];
      node_coords2[2] = ele_aw->nodes()[1]->x()[2];


      double node_coords_center[3];
      for (int p = 0; p < 3; p++) node_coords_center[p] = (node_coords1[p] + node_coords2[p]) / 2;

      // Loop over all acinus elements (on processor)
      for (int i = 0; i < (search_discret.num_my_col_elements()); i++)
      {
        // global acinus element ID
        int GID2 = search_discret.element_col_map()->gid(i);

        Discret::Elements::RedAcinus* ele_ac =
            dynamic_cast<Discret::Elements::RedAcinus*>(search_discret.g_element(GID2));

        // Check if element is an acinus
        if (ele_ac != nullptr)
        {
          // Get coordinates of acini
          double ac_coords[3];
          ac_coords[0] = ele_ac->nodes()[1]->x()[0];
          ac_coords[1] = ele_ac->nodes()[1]->x()[1];
          ac_coords[2] = ele_ac->nodes()[1]->x()[2];

          double diff_vec[3];  // = ac_coords - airway_node_coords_center;
          for (int k = 0; k < 3; k++) diff_vec[k] = ac_coords[k] - node_coords_center[k];
          double accum = 0;

          for (int m = 0; m < 3; m++) accum += diff_vec[m] * diff_vec[m];
          diff_norm = std::sqrt(accum);

          if (diff_norm < min_norm)
          {
            min_norm = diff_norm;
            min_index = i;
          }
        }
      }

      // global element ID
      int GID3 = search_discret.element_col_map()->gid(min_index);

      // why cast
      Discret::Elements::RedAcinus* ele_acinus =
          dynamic_cast<Discret::Elements::RedAcinus*>(search_discret.g_element(GID3));

      // extend ele and node col map
      if (elecolset != nullptr and nodecolset != nullptr)
      {
        elecolset->insert(GID3);
        const int* nodeids = ele_acinus->node_ids();
        for (int inode = 0; inode < ele_acinus->num_node(); ++inode)
          nodecolset->insert(nodeids[inode]);
      }

      if (airway_acinus_dep != nullptr)
        (*airway_acinus_dep).get_values()[j] = (ele_acinus->nodes()[1])->lid();
    }
  }
}

/*----------------------------------------------------------------------*
 | Time loop for red_airway problems                                    |
 |                                                          ismail 01/10|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::time_loop(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  coupledTo3D_ = CoupledTo3D;

  // Time measurement: time loop
  if (!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR(" + time loop");
  }

  // Do the time-stepping
  while (step_ < stepmax_ and time_ < maxtime_)
  {
    // Calculate a single timestep
    this->time_step(CoupledTo3D, CouplingTo3DParams);

    // Stop-criterion for timeloop
    if (CoupledTo3D)
    {
      break;
    }
  }

}  // RedAirwayImplicitTimeInt::TimeLoop



/*----------------------------------------------------------------------*
 | Contains one timestep: Prepare, Solve, Update, Output                |
 |                                                          ismail 09/12|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::time_step(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  coupledTo3D_ = CoupledTo3D;

  double time3D = time_;
  if (coupledTo3D_)
  {
    time3D = CouplingTo3DParams->get<double>("time");
    time_ = time3D - dta_;
  }

  // Prepare current timestep
  prepare_time_step();

  // Output to screen
  if (myrank_ == 0)
  {
    if (!coupledTo3D_)
    {
      printf(
          "TIME: %11.4E/%11.4E  DT = %11.4E   Solving Reduced Dimensional Airways    STEP = "
          "%4d/%4d \n",
          time_, maxtime_, dta_, step_, stepmax_);
    }
    else
    {
      printf(
          "SUBSCALE_TIME: %11.4E/%11.4E  SUBSCALE_DT = %11.4E   Solving Reduced Dimensional "
          "Airways    SUBSCALE_STEP = %4d/%4d \n",
          time_, maxtime_, dta_, step_, stepmax_);
    }
  }

  // Get the solver type parameter: linear or nonlinear solver and solve current timestep
  if (Teuchos::getIntegralValue<Airway::RedAirwaysSolvertype>(params_, "solver type") ==
      Airway::RedAirwaysSolvertype::Nonlinear)
  {
    // Nonlinear solve of current timestep
    non_lin_solve(CouplingTo3DParams);
    if (!myrank_) std::cout << std::endl;
  }
  else if (Teuchos::getIntegralValue<Airway::RedAirwaysSolvertype>(params_, "solver type") ==
           Airway::RedAirwaysSolvertype::Linear)
  {
    // Linear solve of current timestep
    solve(CouplingTo3DParams);
    if (!myrank_) std::cout << std::endl << std::endl;
  }
  else
  {
    FOUR_C_THROW("Not supported solver type!");
  }

  // Update solution: current solution becomes old solution of next timestep
  time_update();


  // Normal red_airway Output
  if (!CoupledTo3D)
  {
    output(CoupledTo3D, CouplingTo3DParams);
  }

  // Update time step sizes
  dtp_ = dta_;

}  // RedAirwayImplicitTimeInt::TimeStep


/*----------------------------------------------------------------------*
 | Contains the one step time loop for red_airway_tissue problems       |
 |                                                          ismail 09/12|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::integrate_step(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  // Output to screen
  if (myrank_ == 0)
  {
    printf(
        "----------------------------------------- STARTING NEW ITERATION "
        "-----------------------------------------\n");
    printf(
        "TIME: %11.4E/%11.4E  DT = %11.4E   Solving Reduced Dimensional Airways    STEP = %4d/%4d "
        "\n",
        time_, maxtime_, dta_, step_, stepmax_);
  }

  // Get the solver type parameter: linear or nonlinear solver and solve current timestep
  if (Teuchos::getIntegralValue<Airway::RedAirwaysSolvertype>(params_, "solver type") ==
      Airway::RedAirwaysSolvertype::Nonlinear)
  {
    non_lin_solve(CouplingTo3DParams);
    if (!myrank_) std::cout << std::endl;
  }
  else if (Teuchos::getIntegralValue<Airway::RedAirwaysSolvertype>(params_, "solver type") ==
           Airway::RedAirwaysSolvertype::Linear)
  {
    solve(CouplingTo3DParams);
    if (!myrank_) std::cout << std::endl << std::endl;
  }
  else
  {
    FOUR_C_THROW("Not supported solver type!");
  }

}  // RedAirwayImplicitTimeInt::IntegrateStep


/*----------------------------------------------------------------------*
 | Setup the variables to do a new time step                ismail 01/10|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::prepare_time_step()
{
  rhs_->put_scalar(0.0);
  // Set time dependent parameters
  step_ += 1;
  time_ += dta_;
}  // RedAirwayImplicitTimeInt::prepare_time_step


/*----------------------------------------------------------------------*
 | Nonlinear iterative solver for reduced-dimensional airway problem    |
 |                                                         ismail 01/11 |
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::non_lin_solve(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  double error_norm1 = 1.e7;
  double error_norm2 = 1.e7;

  // Evaluate total acinar volume
  double acinar_volume_np = 0.0;
  bool err1 = this->sum_all_col_elem_val(*acini_e_volumenp_, *acini_bc_, acinar_volume_np);
  if (err1)
  {
    FOUR_C_THROW("Error in summing acinar volumes");
  }

  // Evaluate total airway volume
  double airway_volume_np = 0.0;
  bool err2 = this->sum_all_col_elem_val(*elemVolumenp_, *open_, airway_volume_np);
  if (err2)
  {
    FOUR_C_THROW("Error in summing airway volumes");
  }

  // Evaluate total lung volume (in acini and airways)
  double lung_volume_np = acinar_volume_np + airway_volume_np;

  // Print out the different volumes
  if (!myrank_)
  {
    std::cout << "time: " << time_ - dta_ << "\t\tTotalLungVolume: " << lung_volume_np
              << "\tAirwayVolume: " << airway_volume_np << "\tAcinarVolume: " << acinar_volume_np
              << std::endl;
  }

  // Loop over nonlinear iterations
  for (int i = 1; i <= maxiter_; i++)
  {
    // Update the pressures of the previous time step
    p_nonlin_->update(1.0, *pnp_, 0.0);

    // Solve the reduced dimensional model
    this->solve(CouplingTo3DParams);

    // Find the change of pressure between the last two iteration steps
    p_nonlin_->update(1.0, *pnp_, -1.0);

    // Evaluate the L2 norm of the pressure difference
    p_nonlin_->norm_2(&error_norm1);

    // Evaluate the residual (=flow) and compute the L2 norm
    this->eval_residual(CouplingTo3DParams);
    residual_->norm_2(&error_norm2);

    // Print output to screen
    if (!myrank_)
    {
      printf("Nonlinear iteration step %4d/%4d ", i, maxiter_);
      printf(" | ||P{%d}-P{%d}||_L2 = %10.3E\t\t|Qresidual|_2 = %10.3E\n", i - 1, i, error_norm1,
          error_norm2);
    }

    // If L2 norm is smaller than tolerance then proceed
    if (error_norm1 <= non_lin_tol_) break;
  }

  // Compute maximal and minimal pressure pnp_ and flux q_in for screen information
  {
    double maxQ = 0.0;
    double maxP = 0.0;
    double minQ = 0.0;
    double minP = 0.0;
    Core::LinAlg::Vector<double> qabs(*qin_np_);
    Core::LinAlg::Vector<double> pabs(*pnp_);
    qabs.abs(*qin_np_);
    pabs.abs(*pnp_);

    qabs.max_value(&maxQ);
    pabs.max_value(&maxP);
    qabs.min_value(&minQ);
    pabs.min_value(&minP);
    if (!myrank_)
    {
      printf(" |Pressure|_max: %10.3E \t\t\t |Q|_max: %10.3E\n", maxP, maxQ);
      printf(" |Pressure|_min: %10.3E \t\t\t |Q|_min: %10.3E\n", minP, minQ);
    }
  }

  if (!myrank_) printf("\n");
}


/*----------------------------------------------------------------------*
 | Single Newton step for reduced dimensional airways                   |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::solve(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  // Time measurement:  solving reduced dimensional airways
  if (!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("   + solving reduced dimensional airways");
  }

  /***
   * 1. Call elements to calculate system matrix and rhs
   ***/
  {
    // Time measurement: element calls
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + element calls");
    }

    // Set both system matrix and rhs vector to zero
    sysmat_->zero();
    rhs_->put_scalar(0.0);

    // Create the element parameters for the discretization
    Teuchos::ParameterList eleparams;

    // Set action for elements: calc_sys_matrix_rhs
    eleparams.set("action", "calc_sys_matrix_rhs");

    // Set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);
    discret_->set_state("on", *on_);
    discret_->set_state("pnm", *pnm_);
    discret_->set_state("intr_ac_link", *n_intr_ac_ln_);

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.qin_np = qin_np_;
    evaluation_data.qin_n = qin_n_;
    evaluation_data.qin_nm = qin_nm_;

    evaluation_data.x_np = x_np_;
    evaluation_data.x_n = x_n_;
    evaluation_data.open = open_;

    evaluation_data.airway_acinus_dep = airway_acinus_dep_;
    evaluation_data.p_extnp = p_extnp_;
    evaluation_data.p_extn = p_extn_;
    evaluation_data.compute_awacinter = compAwAcInter_;

    evaluation_data.acinar_vn = acini_e_volumen_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;

    evaluation_data.qout_np = qout_np_;
    evaluation_data.qout_n = qout_n_;
    evaluation_data.qout_nm = qout_nm_;

    evaluation_data.elemVolumen = elemVolumen_;
    evaluation_data.elemVolumenp = elemVolumenp_;

    evaluation_data.dt = dta_;
    evaluation_data.time = time_;

    // Evaluate Lung volumes nm, n, np and set to eleparams
    double lung_volume_np = 0.0;
    bool err = this->sum_all_col_elem_val(*acini_e_volumenp_, *acini_bc_, lung_volume_np);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    double lung_volume_n = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumen_, *acini_bc_, lung_volume_n);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    double lung_volume_nm = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumenm_, *acini_bc_, lung_volume_nm);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    evaluation_data.lungVolume_np = lung_volume_np;
    evaluation_data.lungVolume_n = lung_volume_n;
    evaluation_data.lungVolume_nm = lung_volume_nm;


    // Call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();

    // finalize the complete matrix
    sysmat_->complete();
    discret_->clear_state();

  }  // end time measurement for element

  /***
   * 2. Solve the boundary conditions
   ***/
  bcval_->put_scalar(0.0);
  dbctog_->put_scalar(0.0);
  {
    // Create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // Set action for elements: set_bc
    eleparams.set("action", "set_bc");

    // Set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);
    discret_->set_state("on", *on_);
    discret_->set_state("pnm", *pnm_);

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.acinar_vn = acini_e_volumen_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;

    evaluation_data.qin_np = qin_np_;
    evaluation_data.qin_n = qin_n_;

    evaluation_data.qout_np = qout_np_;
    evaluation_data.qout_n = qout_n_;

    evaluation_data.airway_acinus_dep = airway_acinus_dep_;
    evaluation_data.p_extnp = p_extnp_;
    evaluation_data.p_extn = p_extn_;

    evaluation_data.dt = dta_;
    evaluation_data.time = time_;
    evaluation_data.bcval = bcval_;
    evaluation_data.dbctog = dbctog_;

    // Evaluate Lung volumes n, np and set to eleparams
    double lung_volume_np = 0.0;
    bool err = this->sum_all_col_elem_val(*acini_e_volumenp_, *acini_bc_, lung_volume_np);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }
    double lung_volume_n = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumen_, *acini_bc_, lung_volume_n);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }
    evaluation_data.lungVolume_np = lung_volume_np;
    evaluation_data.lungVolume_n = lung_volume_n;

    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params", CouplingTo3DParams);

    // Call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();

  }  // end of solving terminal BCs

  /*std::cout<<"----------------------- My SYSMAT IS
  ("<<myrank_<<"-----------------------"<<std::endl; std::shared_ptr<Core::LinAlg::SparseMatrix>
  A_debug = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_); if (A_debug !=
  nullptr)
  {
     A_debug->print(std::cout);
  }
   //               std::cout<<"Map is:
  ("<<myrank_<<")"<<std::endl<<*(discret_->dof_row_map())<<std::endl;
  std::cout<<"---------------------------------------("<<myrank_<<"------------------------"<<std::endl;

  std::cout << "rhs_ = " << std::endl;
  rhs_->print(std::cout);*/

  // double norm_bc_tog = 0.0;
  // rhs_->Norm1(&norm_bc_tog);

  /***
   * 3. Apply the BCs to the system matrix and rhs
   ***/
  {
    // Time measurement: application of dbc
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + apply DBC");
    }

    Core::LinAlg::apply_dirichlet_to_system(*sysmat_, *pnp_, *rhs_, *bcval_, *dbctog_);
  }

  /***
   * 4. Solve for total new velocities and pressures
   ***/
  // Get cpu time
  const double tcpusolve = Teuchos::Time::wallTime();
  {
    // Time measurement: solver
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + solver calls");
    }
    // Call solver
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(sysmat_, pnp_, rhs_, solver_params);
  }

  // end time measurement for solver
  dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;

  if (myrank_ == 0) printf("ts=%4.7f |", dtsolve_);

  /***
   * 5. Compute the flow rates
   ***/
  {
    // Create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // Set action for elements: calculate flow rates
    eleparams.set("action", "calc_flow_rates");

    // Set solution type
    eleparams.set("solver type", params_.get<Airway::RedAirwaysSolvertype>("solver type"));

    // Set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);
    discret_->set_state("on", *on_);
    discret_->set_state("pnm", *pnm_);
    discret_->set_state("intr_ac_link", *n_intr_ac_ln_);

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.qin_nm = qin_nm_;
    evaluation_data.qout_nm = qout_nm_;
    evaluation_data.qin_n = qin_n_;
    evaluation_data.qout_n = qout_n_;
    evaluation_data.qin_np = qin_np_;
    evaluation_data.qout_np = qout_np_;
    evaluation_data.elemVolumen = elemVolumen_;
    evaluation_data.elemVolumenp = elemVolumenp_;
    evaluation_data.acinar_vnp_strain = acini_e_volume_strain_;
    evaluation_data.acinar_vn = acini_e_volumen_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;

    evaluation_data.x_n = x_n_;
    evaluation_data.x_np = x_np_;
    evaluation_data.open = open_;
    evaluation_data.airway_acinus_dep = airway_acinus_dep_;
    evaluation_data.p_extnp = p_extnp_;
    evaluation_data.p_extn = p_extn_;
    evaluation_data.compute_awacinter = compAwAcInter_;

    evaluation_data.dt = dta_;
    evaluation_data.time = time_;

    // Call standard loop over all elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
    discret_->clear_state();
  }

  /***
   * 6. Compute the element volume
   ***/
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set("action", "calc_elem_volumes");

    // set vector values needed by elements
    discret_->clear_state();

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.acinar_vn = acini_e_volumen_;

    evaluation_data.dt = dta_;
    evaluation_data.qin_n = qin_n_;
    evaluation_data.qout_n = qout_n_;
    evaluation_data.qin_np = qin_np_;
    evaluation_data.qout_np = qout_np_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;
    evaluation_data.elemVolumen = elemVolumen_;
    evaluation_data.elemVolumenp = elemVolumenp_;
    evaluation_data.elemRadiusnp = elemRadiusnp_;

    // call standard loop over all elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
    discret_->clear_state();
  }

  /***
   * 7. In case of coupling to 3D fluid
   ***/
  if (coupledTo3D_)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set("action", "get_coupled_values");

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.qin_np = qin_np_;
    evaluation_data.qin_n = qin_n_;

    evaluation_data.qout_np = qout_np_;
    evaluation_data.qout_n = qout_n_;

    evaluation_data.dt = dta_;
    evaluation_data.time = time_;

    evaluation_data.x_n = x_n_;
    evaluation_data.x_np = x_np_;
    evaluation_data.open = open_;
    evaluation_data.airway_acinus_dep = airway_acinus_dep_;
    evaluation_data.p_extnp = p_extnp_;
    evaluation_data.p_extn = p_extn_;
    evaluation_data.compute_awacinter = compAwAcInter_;
    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params", CouplingTo3DParams);

    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();
  }
}  // RedAirwayImplicitTimeInt::Solve


/*----------------------------------------------------------------------*
 | Call elements to calculate system matrix/rhs and assemble.           |
 | This function is currently not used but will be kept empty until     |
 | further use.                                            ismail 01/10 |
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::assemble_mat_and_rhs()
{
  dtele_ = 0.0;
  dtfilter_ = 0.0;
  // time measurement: element
  if (!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("      + element calls");
  }

}  // RedAirwayImplicitTimeInt::assemble_mat_and_rhs


/*----------------------------------------------------------------------*
 | Current solution becomes most recent solution of next timestep       |
 |                                                                      |
 |  pnm_  =  on_                                                        |
 |  on_   =  pnp_                                                       |
 |                                                          ismail 06/09|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::time_update()
{
  // Volumetric Flow rate and acini volume of this step become most recent
  pnm_->update(1.0, *on_, 0.0);
  on_->update(1.0, *pnp_, 0.0);

  qin_nm_->update(1.0, *qin_n_, 0.0);
  qin_n_->update(1.0, *qin_np_, 0.0);

  // Timeupdate x-vector x_np ->x_n
  x_n_->update(1.0, *x_np_, 0.0);

  qout_nm_->update(1.0, *qout_n_, 0.0);
  qout_n_->update(1.0, *qout_np_, 0.0);

  acini_e_volumenm_->update(1.0, *acini_e_volumen_, 0.0);
  acini_e_volumen_->update(1.0, *acini_e_volumenp_, 0.0);

  elemVolumenm_->update(1.0, *elemVolumen_, 0.0);
  elemVolumen_->update(1.0, *elemVolumenp_, 0.0);

  return;
}  // RedAirwayImplicitTimeInt::TimeUpdate


/*----------------------------------------------------------------------*
 | Initializes state saving vectors                                     |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-0D fields         |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::init_save_state()
{
  // Get discretizations DOF row map
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // Get discretizations element row map
  const Core::LinAlg::Map* elementcolmap = discret_->element_col_map();

  // saving vector for pressure
  saved_pnm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  saved_on_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  saved_pnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // saving vector for inflow rate
  saved_qin_nm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_qin_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_qin_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // saving vector for outflow rate
  saved_qout_nm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_qout_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_qout_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // saving vector for trajectory
  saved_x_n_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_x_np_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // saving vector for acinar volume
  saved_acini_e_volumenm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_acini_e_volumen_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_acini_e_volumenp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

  // saving vector for element volume
  saved_elemVolumenm_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_elemVolumen_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);
  saved_elemVolumenp_ = std::make_shared<Core::LinAlg::Vector<double>>(*elementcolmap, true);

}  // RedAirwayImplicitTimeInt::InitSaveState()


/*----------------------------------------------------------------------*
 | Saves and backs up the current state.                                |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-0D fields          |
 |  example:                                                            |
 |  saved_on_   =  on_                                                  |
 |  saved_qn_   =  qn_                                                  |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::save_state()
{
  // save pressure vectors
  saved_pnm_->update(1.0, *pnm_, 0.0);
  saved_on_->update(1.0, *on_, 0.0);
  saved_pnp_->update(1.0, *pnp_, 0.0);

  // save inflow rate vectors
  saved_qin_nm_->update(1.0, *qin_nm_, 0.0);
  saved_qin_n_->update(1.0, *qin_n_, 0.0);
  saved_qin_np_->update(1.0, *qin_np_, 0.0);

  // save outflow rate vectors
  saved_qout_nm_->update(1.0, *qout_nm_, 0.0);
  saved_qout_n_->update(1.0, *qout_n_, 0.0);
  saved_qout_np_->update(1.0, *qout_np_, 0.0);

  // save trajectory vectors
  saved_x_n_->update(1.0, *x_n_, 0.0);
  saved_x_np_->update(1.0, *x_np_, 0.0);

  // save acinar volume vectors
  saved_acini_e_volumenm_->update(1.0, *acini_e_volumenm_, 0.0);
  saved_acini_e_volumen_->update(1.0, *acini_e_volumen_, 0.0);
  saved_acini_e_volumenp_->update(1.0, *acini_e_volumenp_, 0.0);

  // save element volume vectors
  saved_elemVolumenm_->update(1.0, *elemVolumenm_, 0.0);
  saved_elemVolumen_->update(1.0, *elemVolumen_, 0.0);
  saved_elemVolumenp_->update(1.0, *elemVolumenp_, 0.0);

  return;
}  // RedAirwayImplicitTimeInt::SaveState


/*----------------------------------------------------------------------*
 | Loads backed up states.                                              |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-0D fields          |
 |  example:                                                            |
 |  on_   =  saved_on_                                                  |
 |  qn_   =  saved_qn_                                                  |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::load_state()
{
  // save pressure vectors
  pnm_->update(1.0, *saved_pnm_, 0.0);
  on_->update(1.0, *saved_on_, 0.0);
  pnp_->update(1.0, *saved_pnp_, 0.0);

  // save inflow rate vectors
  qin_nm_->update(1.0, *saved_qin_nm_, 0.0);
  qin_n_->update(1.0, *saved_qin_n_, 0.0);
  qin_np_->update(1.0, *saved_qin_np_, 0.0);

  // save outflow rate vectors
  qout_nm_->update(1.0, *saved_qout_np_, 0.0);
  qout_n_->update(1.0, *saved_qout_n_, 0.0);
  qout_np_->update(1.0, *saved_qout_np_, 0.0);

  // save trajectory vectors
  x_n_->update(1.0, *saved_x_n_, 0.0);
  x_np_->update(1.0, *saved_x_np_, 0.0);

  // save acinar volume vectors
  acini_e_volumenm_->update(1.0, *saved_acini_e_volumenm_, 0.0);
  acini_e_volumen_->update(1.0, *saved_acini_e_volumen_, 0.0);
  acini_e_volumenp_->update(1.0, *saved_acini_e_volumenp_, 0.0);

  // save element volume vectors
  elemVolumenm_->update(1.0, *saved_elemVolumenm_, 0.0);
  elemVolumen_->update(1.0, *saved_elemVolumen_, 0.0);
  elemVolumenp_->update(1.0, *saved_elemVolumenp_, 0.0);

  return;
}  // RedAirwayImplicitTimeInt::LoadState


/*----------------------------------------------------------------------*
 | Output of solution vector to binio                       ismail 07/09|
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::output(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingParams)
{
  int step = 0;
  int upres = 0;
  int uprestart = 0;
  double time_backup = 0.0;

  // If coupled to 3D problem, then get the export information from
  // the 3D problem
  if (CoupledTo3D)
  {
    step_ = CouplingParams->get<int>("step");
    upres_ = CouplingParams->get<int>("upres");
    uprestart_ = CouplingParams->get<int>("uprestart");
    time_ = CouplingParams->get<double>("time");
  }

  if (step_ % upres_ == 0)
  {
    // step number and time
    output_.new_step(step_, time_);

    // "pressure" vectors
    output_.write_vector("pnm", pnm_);
    output_.write_vector("on", on_);
    output_.write_vector("pnp", pnp_);
    output_.write_vector("p_nonlin", p_nonlin_);

    // write the flow values
    Core::LinAlg::export_to(*qin_nm_, *qexp_);
    output_.write_vector("qin_nm", qexp_);
    Core::LinAlg::export_to(*qin_n_, *qexp_);
    output_.write_vector("qin_n", qexp_);
    Core::LinAlg::export_to(*qin_np_, *qexp_);
    output_.write_vector("qin_np", qexp_);

    Core::LinAlg::export_to(*qout_nm_, *qexp_);
    output_.write_vector("qout_nm", qexp_);
    Core::LinAlg::export_to(*qout_n_, *qexp_);
    output_.write_vector("qout_n", qexp_);
    Core::LinAlg::export_to(*qout_np_, *qexp_);
    output_.write_vector("qout_np", qexp_);

    Core::LinAlg::export_to(*x_n_, *qexp_);
    output_.write_vector("x_n", qexp_);
    Core::LinAlg::export_to(*x_np_, *qexp_);
    output_.write_vector("x_np", qexp_);
    Core::LinAlg::export_to(*open_, *qexp_);
    output_.write_vector("open", qexp_);
    Core::LinAlg::export_to(*p_extnp_, *qexp_);
    output_.write_vector("p_extnp", qexp_);
    Core::LinAlg::export_to(*p_extn_, *qexp_);
    output_.write_vector("p_extn", qexp_);
    Core::LinAlg::export_to(*airway_acinus_dep_, *qexp_);
    output_.write_vector("airway_acinus_dep", qexp_);

    Core::LinAlg::export_to(*elemVolumenm_, *qexp_);
    output_.write_vector("elemVolumenm", qexp_);
    Core::LinAlg::export_to(*elemVolumen_, *qexp_);
    output_.write_vector("elemVolumen", qexp_);
    Core::LinAlg::export_to(*elemVolumenp_, *qexp_);
    output_.write_vector("elemVolumenp", qexp_);

    Core::LinAlg::export_to(*elemRadiusnp_, *qexp_);
    output_.write_vector("elemRadius_current", qexp_);

    {
      Core::LinAlg::Export exporter(acini_e_volumenm_->get_map(), qexp_->get_map());
      qexp_->export_to(*acini_e_volumenm_, exporter, Core::LinAlg::CombineMode::zero);
      output_.write_vector("acini_vnm", qexp_);
    }
    {
      Core::LinAlg::Export exporter(acini_e_volumen_->get_map(), qexp_->get_map());
      qexp_->export_to(*acini_e_volumen_, exporter, Core::LinAlg::CombineMode::zero);
      output_.write_vector("acini_vn", qexp_);
    }
    {
      Core::LinAlg::Export exporter(acini_e_volumenp_->get_map(), qexp_->get_map());
      qexp_->export_to(*acini_e_volumenp_, exporter, Core::LinAlg::CombineMode::zero);
      output_.write_vector("acini_vnp", qexp_);
    }
    {
      Core::LinAlg::Export exporter(acini_e_volume_strain_->get_map(), qexp_->get_map());
      qexp_->export_to(*acini_e_volume_strain_, exporter, Core::LinAlg::CombineMode::zero);
      output_.write_vector("acini_volumetric_strain", qexp_);
    }
    {
      Core::LinAlg::Export exporter(acini_e_volume0_->get_map(), qexp_->get_map());
      qexp_->export_to(*acini_e_volume0_, exporter, Core::LinAlg::CombineMode::zero);
      output_.write_vector("acini_v0", qexp_);
    }

    if (step_ == upres_)
    {
      Core::LinAlg::export_to(*elemVolume0_, *qexp_);
      output_.write_vector("elemVolume0", qexp_);
      output_.write_vector("NodeIDs", nodeIds_);
      output_.write_vector("radii", radii_);
      Core::LinAlg::export_to(*generations_, *qexp_);
      output_.write_vector("generations", qexp_);
      Core::LinAlg::export_to(*acini_bc_, *qexp_);
      output_.write_vector("acin_bc", qexp_);
      output_.write_element_data(true);
      Core::LinAlg::export_to(*elemArea0_, *qexp_);
      output_.write_vector("elemArea0", qexp_);
    }

    if (CoupledTo3D)
    {
      output_.write_int("Actual_RedD_step", step);
    }
  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ != 0 && step_ % uprestart_ == 0)
  {
    // step number and time
    output_.new_step(step_, time_);

    // "pressure" vectors
    output_.write_vector("pnm", pnm_);
    output_.write_vector("on", on_);
    output_.write_vector("pnp", pnp_);
    output_.write_vector("p_nonlin", p_nonlin_);

    // write the flow values
    Core::LinAlg::export_to(*qin_nm_, *qexp_);
    output_.write_vector("qin_nm", qexp_);
    Core::LinAlg::export_to(*qin_n_, *qexp_);
    output_.write_vector("qin_n", qexp_);
    Core::LinAlg::export_to(*qin_np_, *qexp_);
    output_.write_vector("qin_np", qexp_);
    //
    Core::LinAlg::export_to(*qout_nm_, *qexp_);
    output_.write_vector("qout_nm", qexp_);
    Core::LinAlg::export_to(*qout_n_, *qexp_);
    output_.write_vector("qout_n", qexp_);
    Core::LinAlg::export_to(*qout_np_, *qexp_);
    output_.write_vector("qout_np", qexp_);

    Core::LinAlg::export_to(*x_n_, *qexp_);
    output_.write_vector("x_n", qexp_);
    Core::LinAlg::export_to(*x_np_, *qexp_);
    output_.write_vector("x_np", qexp_);
    Core::LinAlg::export_to(*open_, *qexp_);
    output_.write_vector("open", qexp_);
    Core::LinAlg::export_to(*p_extnp_, *qexp_);
    output_.write_vector("p_extnp", qexp_);
    Core::LinAlg::export_to(*p_extn_, *qexp_);
    output_.write_vector("p_extn", qexp_);
    Core::LinAlg::export_to(*airway_acinus_dep_, *qexp_);
    output_.write_vector("airway_acinus_dep", qexp_);

    Core::LinAlg::export_to(*elemVolumenm_, *qexp_);
    output_.write_vector("elemVolumenm", qexp_);
    Core::LinAlg::export_to(*elemVolumen_, *qexp_);
    output_.write_vector("elemVolumen", qexp_);
    Core::LinAlg::export_to(*elemVolumenp_, *qexp_);
    output_.write_vector("elemVolumenp", qexp_);

    //
    Core::LinAlg::export_to(*acini_e_volumenm_, *qexp_);
    output_.write_vector("acini_vnm", qexp_);
    Core::LinAlg::export_to(*acini_e_volumen_, *qexp_);
    output_.write_vector("acini_vn", qexp_);
    Core::LinAlg::export_to(*acini_e_volumenp_, *qexp_);
    output_.write_vector("acini_vnp", qexp_);
    Core::LinAlg::export_to(*acini_e_volume_strain_, *qexp_);
    output_.write_vector("acini_volumetric_strain", qexp_);
    Core::LinAlg::export_to(*acini_e_volume0_, *qexp_);
    output_.write_vector("acini_v0", qexp_);

    // write mesh in each restart step --- the elements are required since
    // they contain history variables (the time dependent subscales)
    output_.write_mesh(step_, time_);

    if (CoupledTo3D)
    {
      output_.write_int("Actual_RedD_step", step);
    }
  }

  // If coupled to 3D problem, then retrieve the old information of the
  // the reduced model problem
  if (CoupledTo3D)
  {
    step_ = step;
    upres_ = upres;
    uprestart_ = uprestart;
    time_ = time_backup;
  }
  return;
}  // RedAirwayImplicitTimeInt::Output

/*----------------------------------------------------------------------*
 | read_restart (public)                                     ismail 01/10|
 -----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::read_restart(int step, bool coupledTo3D)
{
  coupledTo3D_ = coupledTo3D;
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step);
  time_ = reader.read_double("time");

  if (coupledTo3D_)
  {
    step_ = reader.read_int("Actual_RedD_step");
  }
  else
  {
    step_ = reader.read_int("step");
  }

  reader.read_vector(pnp_, "pnp");
  reader.read_vector(on_, "on");
  reader.read_vector(pnm_, "pnm");
  reader.read_vector(p_nonlin_, "p_nonlin");

  reader.read_vector(qexp_, "acini_vnm");
  Core::LinAlg::export_to(*qexp_, *acini_e_volumenm_);
  reader.read_vector(qexp_, "acini_vn");
  Core::LinAlg::export_to(*qexp_, *acini_e_volumen_);
  reader.read_vector(qexp_, "acini_vnp");
  Core::LinAlg::export_to(*qexp_, *acini_e_volumenp_);
  reader.read_vector(qexp_, "acini_volumetric_strain");
  Core::LinAlg::export_to(*qexp_, *acini_e_volume_strain_);
  reader.read_vector(qexp_, "acini_v0");
  Core::LinAlg::export_to(*qexp_, *acini_e_volume0_);

  reader.read_vector(qexp_, "qin_nm");
  Core::LinAlg::export_to(*qexp_, *qin_nm_);
  reader.read_vector(qexp_, "qin_n");
  Core::LinAlg::export_to(*qexp_, *qin_n_);
  reader.read_vector(qexp_, "qin_np");
  Core::LinAlg::export_to(*qexp_, *qin_np_);

  reader.read_vector(qexp_, "qout_nm");
  Core::LinAlg::export_to(*qexp_, *qout_nm_);
  reader.read_vector(qexp_, "qout_n");
  Core::LinAlg::export_to(*qexp_, *qout_n_);
  reader.read_vector(qexp_, "qout_np");
  Core::LinAlg::export_to(*qexp_, *qout_np_);

  reader.read_vector(qexp_, "elemVolumenm");
  Core::LinAlg::export_to(*qexp_, *elemVolumenm_);
  reader.read_vector(qexp_, "elemVolumen");
  Core::LinAlg::export_to(*qexp_, *elemVolumen_);
  reader.read_vector(qexp_, "elemVolumenp");
  Core::LinAlg::export_to(*qexp_, *elemVolumenp_);

  reader.read_vector(qexp_, "x_n");
  Core::LinAlg::export_to(*qexp_, *x_n_);
  reader.read_vector(qexp_, "x_np");
  Core::LinAlg::export_to(*qexp_, *x_np_);
  reader.read_vector(qexp_, "open");
  Core::LinAlg::export_to(*qexp_, *open_);
  reader.read_vector(qexp_, "p_extn");
  Core::LinAlg::export_to(*qexp_, *p_extn_);
  reader.read_vector(qexp_, "p_extnp");
  Core::LinAlg::export_to(*qexp_, *p_extnp_);
  reader.read_vector(qexp_, "airway_acinus_dep");
  Core::LinAlg::export_to(*qexp_, *airway_acinus_dep_);

}  // RedAirwayImplicitTimeInt::read_restart


/*----------------------------------------------------------------------*
 | Create the field test for redairway field                 roth 10/13 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Airway::RedAirwayImplicitTimeInt::create_field_test()
{
  return std::make_shared<RedAirwayResultTest>(*this);
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Airway::RedAirwayImplicitTimeInt::eval_residual(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  residual_->put_scalar(0.0);

  // Call elements to calculate system matrix
  {
    // set both system matrix and rhs vector to zero
    sysmat_->zero();
    rhs_->put_scalar(0.0);

    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set("action", "calc_sys_matrix_rhs");

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);
    discret_->set_state("on", *on_);
    discret_->set_state("pnm", *pnm_);
    discret_->set_state("intr_ac_link", *n_intr_ac_ln_);

    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.acinar_vn = acini_e_volumen_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;

    evaluation_data.qin_np = qin_np_;
    evaluation_data.qin_n = qin_n_;
    evaluation_data.qin_nm = qin_nm_;

    evaluation_data.qout_np = qout_np_;
    evaluation_data.qout_n = qout_n_;
    evaluation_data.qout_nm = qout_nm_;

    evaluation_data.x_np = x_np_;
    evaluation_data.x_n = x_n_;
    evaluation_data.open = open_;
    evaluation_data.airway_acinus_dep = airway_acinus_dep_;
    evaluation_data.p_extnp = p_extnp_;
    evaluation_data.p_extn = p_extn_;
    evaluation_data.compute_awacinter = compAwAcInter_;

    evaluation_data.elemVolumen = elemVolumen_;
    evaluation_data.elemVolumenp = elemVolumenp_;

    evaluation_data.dt = dta_;
    evaluation_data.time = time_;

    // get lung volume
    double lung_volume_np = 0.0;
    bool err = this->sum_all_col_elem_val(*acini_e_volumenp_, *acini_bc_, lung_volume_np);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    double lung_volume_n = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumen_, *acini_bc_, lung_volume_n);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    double lung_volume_nm = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumenm_, *acini_bc_, lung_volume_nm);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }

    evaluation_data.lungVolume_np = lung_volume_np;
    evaluation_data.lungVolume_n = lung_volume_n;
    evaluation_data.lungVolume_nm = lung_volume_nm;


    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();

    // finalize the complete matrix
    discret_->clear_state();
  }

  // Solve the boundary conditions
  bcval_->put_scalar(0.0);
  dbctog_->put_scalar(0.0);
  // Solve terminal BCs
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set("action", "set_bc");

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("pnp", *pnp_);
    discret_->set_state("on", *on_);
    discret_->set_state("pnm", *pnm_);
    //    discret_->set_state("qcnp",qcnp_);
    //    discret_->set_state("qcn" ,qcn_ );
    //    discret_->set_state("qcnm",qcnm_);
    // note: We use an RCP because ParameterList wants something printable and comparable
    Discret::ReducedLung::EvaluationData& evaluation_data =
        Discret::ReducedLung::EvaluationData::get();

    evaluation_data.acinar_vn = acini_e_volumen_;
    evaluation_data.acinar_vnp = acini_e_volumenp_;

    evaluation_data.qin_np = qin_np_;
    evaluation_data.qin_n = qin_n_;

    evaluation_data.qout_np = qout_np_;
    evaluation_data.qout_n = qout_n_;

    evaluation_data.bcval = bcval_;
    evaluation_data.dbctog = dbctog_;
    evaluation_data.dt = dta_;
    evaluation_data.time = time_;

    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params", CouplingTo3DParams);

    // get lung volume
    double lung_volume_np = 0.0;
    bool err = this->sum_all_col_elem_val(*acini_e_volumenp_, *acini_bc_, lung_volume_np);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }
    double lung_volume_n = 0.0;
    err = this->sum_all_col_elem_val(*acini_e_volumen_, *acini_bc_, lung_volume_n);
    if (err)
    {
      FOUR_C_THROW("Error by summing all acinar volumes");
    }
    evaluation_data.lungVolume_np = lung_volume_np;
    evaluation_data.lungVolume_n = lung_volume_n;


    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();
  }

  // Apply the BCs to the system matrix and rhs
  {
    Core::LinAlg::apply_dirichlet_to_system(*sysmat_, *pnp_, *rhs_, *bcval_, *dbctog_);
  }

  // Evaluate Residual
  sysmat_->multiply(false, *pnp_, *residual_);
  residual_->update(-1.0, *rhs_, 1.0);
}  // EvalResidual



/*----------------------------------------------------------------------*
 | Sum all ColElement values                                            |
 |                                                          ismail 11/12|
 *----------------------------------------------------------------------*/
bool Airway::RedAirwayImplicitTimeInt::sum_all_col_elem_val(
    Core::LinAlg::Vector<double>& vec, Core::LinAlg::Vector<double>& sumCond, double& sum)
{
  // Check if the vector is a ColElement vector
  const Core::LinAlg::Map* elementcolmap = discret_->element_col_map();
  if (!vec.get_map().same_as(*elementcolmap) && !sumCond.get_map().same_as(*elementcolmap))
  {
    return true;
  }

  // Since the acinar_volume vector is a ColMap, we first need to export
  // it to a RowMap and eliminate the ghosted values
  {
    // define exporter
    Core::LinAlg::Export exporter(vec.get_map(), qexp_->get_map());
    // export from ColMap to RowMap
    qexp_->export_to(vec, exporter, Core::LinAlg::CombineMode::zero);

    Core::LinAlg::Export exporter2(sumCond.get_map(), qexp2_->get_map());
    // export from ColMap to RowMap
    qexp2_->export_to(sumCond, exporter2, Core::LinAlg::CombineMode::zero);
  }

  // Get the mean acinar volume on the current processor
  qexp_->dot(*qexp2_, &sum);

  // return all is fine
  return false;
}

FOUR_C_NAMESPACE_CLOSE
