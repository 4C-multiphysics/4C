// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_utils.hpp"

#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_interface.hpp"
#include "4C_fem_general_l2_projection.hpp"
#include "4C_fem_general_utils_superconvergent_patch_recovery.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"

#include <stdio.h>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | constructor                                         Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
FLD::Utils::StressManager::StressManager(std::shared_ptr<Core::FE::Discretization> discret,
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp, const bool alefluid, const int numdim)
    : discret_(discret),
      dispnp_(dispnp),
      alefluid_(alefluid),
      numdim_(numdim),
      sep_enr_(nullptr),
      wss_type_(Teuchos::getIntegralValue<Inpar::FLUID::WSSType>(
          Global::Problem::instance()->fluid_dynamic_params(), "WSS_TYPE")),
      sum_stresses_(nullptr),
      sum_wss_(nullptr),
      sum_dt_stresses_(0.0),
      sum_dt_wss_(0.0),
      isinit_(false)
{
  switch (wss_type_)
  {
    case Inpar::FLUID::wss_standard:
      isinit_ = true;  // in this cases nothing has to be initialized
      break;
    case Inpar::FLUID::wss_aggregation:
      isinit_ = false;  // we do this in InitAggr()
      break;
    case Inpar::FLUID::wss_mean:
      sum_stresses_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true),
      sum_wss_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true),
      isinit_ = true;
      break;
    default:
      FOUR_C_THROW(
          "There are only the wss calculation types 'standard', 'aggregation' and 'mean'!!");
      break;
  }
}

/*----------------------------------------------------------------------*
 | constructor                                         Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
void FLD::Utils::StressManager::init_aggr(std::shared_ptr<Core::LinAlg::SparseOperator> sysmat)
{
  if (wss_type_ != Inpar::FLUID::wss_aggregation)
    FOUR_C_THROW("One should end up here just in case of aggregated stresses!");

  calc_sep_enr(sysmat);
  if (sep_enr_ == nullptr) FOUR_C_THROW("SepEnr matrix has not been build correctly. Strange...");

  isinit_ = true;

  return;
}

/*----------------------------------------------------------------------*
 | update and return WSS vector                        Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::get_wall_shear_stresses(
    const Core::LinAlg::Vector<double>& trueresidual, const double dt)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> wss = get_wall_shear_stresses_wo_agg(trueresidual);

  switch (wss_type_)
  {
    case Inpar::FLUID::wss_standard:
      // nothing to do
      break;
    case Inpar::FLUID::wss_aggregation:
      wss = aggreagte_stresses(*wss);
      break;
    case Inpar::FLUID::wss_mean:
      wss = time_average_wss(*wss, dt);
      break;
    default:
      FOUR_C_THROW(
          "There are only the wss calculation types 'standard', 'aggregation' and 'mean'!!");
      break;
  }

  return wss;
}

/*--------------------------------------------------------------------------*
 | return wss vector (without updating the mean stress vector)   Thon 11/14 |
 *--------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
FLD::Utils::StressManager::get_pre_calc_wall_shear_stresses(
    const Core::LinAlg::Vector<double>& trueresidual)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> wss =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

  switch (wss_type_)
  {
    case Inpar::FLUID::wss_standard:
      wss = get_wall_shear_stresses_wo_agg(trueresidual);
      break;
    case Inpar::FLUID::wss_aggregation:
      wss = get_wall_shear_stresses_wo_agg(trueresidual);
      wss = aggreagte_stresses(*wss);
      break;
    case Inpar::FLUID::wss_mean:
      if (sum_dt_wss_ > 0.0)  // iff we have actually calculated some mean wss
        wss->update(1.0 / sum_dt_wss_, *sum_wss_, 0.0);  // weighted sum of all prior stresses
      break;
    default:
      FOUR_C_THROW(
          "There are only the wss calculation types 'standard', 'aggregation' and 'mean'!!");
      break;
  }

  return wss;
}

/*----------------------------------------------------------------------*
 | return WSS vector always without aggregation        Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
FLD::Utils::StressManager::get_wall_shear_stresses_wo_agg(
    const Core::LinAlg::Vector<double>& trueresidual)
{
  if (not isinit_) FOUR_C_THROW("StressManager not initialized");

  std::shared_ptr<Core::LinAlg::Vector<double>> stresses = calc_stresses(trueresidual);
  // calculate wss from stresses
  std::shared_ptr<Core::LinAlg::Vector<double>> wss = calc_wall_shear_stresses(stresses);

  return wss;
}

/*-----------------------------------------------------------------------------*
 |  calculate traction vector at (Dirichlet) boundary (public) Thon/Krank 07/07|
 *-----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::get_stresses_wo_agg(
    const Core::LinAlg::Vector<double>& trueresidual)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> stresses = calc_stresses(trueresidual);

  return stresses;
}  // FLD::Utils::StressManager::GetStressesWOAgg()

/*-----------------------------------------------------------------------------*
 |  update and return stress vector                            Thon/Krank 07/07|
 *-----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::get_stresses(
    const Core::LinAlg::Vector<double>& trueresidual, const double dt)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> stresses = get_stresses_wo_agg(trueresidual);

  switch (wss_type_)
  {
    case Inpar::FLUID::wss_standard:
      // nothing to do
      break;
    case Inpar::FLUID::wss_aggregation:
      stresses = aggreagte_stresses(*stresses);
      break;
    case Inpar::FLUID::wss_mean:
      stresses = time_average_stresses(*stresses, dt);
      break;
    default:
      FOUR_C_THROW(
          "There are only the wss calculation types 'standard', 'aggregation' and 'mean'!!");
      break;
  }

  return stresses;
}  // FLD::Utils::StressManager::GetStresses()

/*-----------------------------------------------------------------------------*
 | return stress vector (without updating the mean stress vector)   Thon 03/15 |
 *-----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::get_pre_calc_stresses(
    const Core::LinAlg::Vector<double>& trueresidual)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> stresses =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

  switch (wss_type_)
  {
    case Inpar::FLUID::wss_standard:
      stresses = get_stresses_wo_agg(trueresidual);
      break;
    case Inpar::FLUID::wss_aggregation:
      stresses = get_stresses_wo_agg(trueresidual);
      stresses = aggreagte_stresses(*stresses);
      break;
    case Inpar::FLUID::wss_mean:
      if (sum_dt_stresses_ > 0.0)  // iff we have actually calculated some mean stresses
        stresses->update(
            1.0 / sum_dt_stresses_, *sum_stresses_, 0.0);  // weighted sum of all prior stresses
      break;
    default:
      FOUR_C_THROW(
          "There are only the wss calculation types 'standard', 'aggregation' and 'mean'!!");
      break;
  }
  return stresses;
}

/*-----------------------------------------------------------------------------*
 |  calculate traction vector at (Dirichlet) boundary (public) Thon/Krank 07/07|
 *-----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::calc_stresses(
    const Core::LinAlg::Vector<double>& trueresidual)
{
  if (not isinit_) FOUR_C_THROW("StressManager not initialized");
  std::string condstring("FluidStressCalc");
  std::shared_ptr<Core::LinAlg::Vector<double>> integratedshapefunc =
      integrate_interface_shape(condstring);

  // compute traction values at specified nodes; otherwise do not touch the zero values
  for (int i = 0; i < integratedshapefunc->local_length(); i++)
  {
    if (integratedshapefunc->local_values_as_span()[i] != 0.0)
    {
      // overwrite integratedshapefunc values with the calculated traction coefficients,
      // which are reconstructed out of the nodal forces (trueresidual_) using the
      // same shape functions on the boundary as for velocity and pressure.
      (*integratedshapefunc).get_values()[i] =
          trueresidual.local_values_as_span()[i] / integratedshapefunc->local_values_as_span()[i];
    }
  }

  return integratedshapefunc;
}  // FLD::Utils::StressManager::calc_stresses()

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::integrate_interface_shape(
    std::string condname)
{
  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<FLD::BoundaryAction>("action", FLD::integrate_Shapefunction);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> integratedshapefunc =
      Core::LinAlg::create_vector(*dofrowmap, true);

  // call loop over elements
  discret_->clear_state();
  if (alefluid_)
  {
    discret_->set_state("dispnp", *dispnp_);
  }
  discret_->evaluate_condition(eleparams, integratedshapefunc, condname);
  discret_->clear_state();

  return integratedshapefunc;
}

/*----------------------------------------------------------------------*
 |  calculate wall sheer stress from stresses at dirichlet boundary     |
 |                                                     Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::calc_wall_shear_stresses(
    std::shared_ptr<Core::LinAlg::Vector<double>> stresses)
{
  // -------------------------------------------------------------------
  // first evaluate the normals at the nodes
  // -------------------------------------------------------------------

  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<FLD::BoundaryAction>("action", FLD::boundary_calc_node_normal);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // vector ndnorm0 with pressure-entries is needed for evaluate_condition
  std::shared_ptr<Core::LinAlg::Vector<double>> ndnorm0 =
      Core::LinAlg::create_vector(*dofrowmap, true);

  // call loop over elements, note: normal vectors do not yet have length = 1.0
  discret_->clear_state();  // TODO: (Thon) Do we really have to to this in here?
  if (alefluid_)
  {
    discret_->set_state("dispnp", *dispnp_);
  }
  // evaluate the normals of the surface
  discret_->evaluate_condition(eleparams, ndnorm0, "FluidStressCalc");
  discret_->clear_state();

  // -------------------------------------------------------------------
  // normalize the normal vectors
  // -------------------------------------------------------------------
  for (int i = 0; i < ndnorm0->local_length(); i += numdim_ + 1)
  {
    // calculate the length of the normal
    double L = 0.0;
    for (int j = 0; j < numdim_; j++)
    {
      L += (ndnorm0->local_values_as_span()[i + j]) * (ndnorm0->local_values_as_span()[i + j]);
    }
    L = sqrt(L);

    // normalise the normal vector (if present for the current node)
    if (L > 1e-15)
    {
      for (int j = 0; j < numdim_; j++)
      {
        (*ndnorm0).get_values()[i + j] /= L;
      }
    }
  }

  // -------------------------------------------------------------------
  // evaluate the wall shear stress from the traction by removing
  // the normal stresses
  // -------------------------------------------------------------------

  // get traction
  std::shared_ptr<Core::LinAlg::Vector<double>> wss = stresses;

  // loop over all entities within the traction vector
  for (int i = 0; i < ndnorm0->local_length(); i += numdim_ + 1)
  {
    // evaluate the normal stress = < traction . normal >
    double normal_stress = 0.0;
    for (int j = 0; j < numdim_; j++)
    {
      normal_stress += wss->local_values_as_span()[i + j] * ndnorm0->local_values_as_span()[i + j];
    }

    // subtract the normal stresses from traction
    for (int j = 0; j < numdim_; j++)
    {
      (*wss).get_values()[i + j] -= normal_stress * ndnorm0->local_values_as_span()[i + j];
    }
  }

  // -------------------------------------------------------------------
  // return the wall_shear_stress vector
  // -------------------------------------------------------------------
  return wss;
}


/*----------------------------------------------------------------------*
 | smooth stresses/wss via ML-aggregation              Thon/Krank 11/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::aggreagte_stresses(
    Core::LinAlg::Vector<double>& wss)
{
  if (sep_enr_ == nullptr) FOUR_C_THROW("no scale separation matrix");

  std::shared_ptr<Core::LinAlg::Vector<double>> mean_wss =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

  // Do the actual aggregation
  sep_enr_->multiply(false, wss, *mean_wss);

  return mean_wss;
}

/*----------------------------------------------------------------------*
 | time average stresses                                     Thon 03/15 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::time_average_stresses(
    const Core::LinAlg::Vector<double>& stresses, const double dt)
{
  sum_stresses_->update(dt, stresses, 1.0);  // weighted sum of all prior stresses
  sum_dt_stresses_ += dt;

  std::shared_ptr<Core::LinAlg::Vector<double>> mean_stresses =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  mean_stresses->update(1.0 / sum_dt_stresses_, *sum_stresses_, 0.0);

  return mean_stresses;
}

/*----------------------------------------------------------------------*
 | time average wss                                          Thon 03/15 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Utils::StressManager::time_average_wss(
    const Core::LinAlg::Vector<double>& wss, const double dt)
{
  sum_wss_->update(dt, wss, 1.0);  // weighted sum of all prior stresses
  sum_dt_wss_ += dt;

  std::shared_ptr<Core::LinAlg::Vector<double>> mean_wss =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  mean_wss->update(1.0 / sum_dt_wss_, *sum_wss_, 0.0);

  return mean_wss;
}

/*----------------------------------------------------------------------*
 | Calculate Aggregation Matrix and set is as member variable SepEnr_   |
 |                                                     Thon/Krank 11/14 |
 *------------------------------------------------- --------------------*/
void FLD::Utils::StressManager::calc_sep_enr(std::shared_ptr<Core::LinAlg::SparseOperator> sysmat)
{
  // if we have not specified a multigrid-solver one does not want to smooth the wss
  if (wss_type_ == Inpar::FLUID::wss_aggregation)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat2;

    // Try this:
    sysmat2 = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat);
    if (sysmat2 == nullptr)
      sysmat2 = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat)->merge();
    if (sysmat2 == nullptr)
      FOUR_C_THROW("One of these two dynamic casts should have worked... Sorry!");

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      std::cout << "Calculating mean WSS via multilevel-aggregation:" << std::endl;

    Teuchos::ParameterList params;
    Core::LinearSolver::Parameters::compute_solver_parameters(*discret_, params);

    // get nullspace parameters
    auto nullspace = params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");

    // get plain aggregation Ptent
    Core::LinAlg::SparseMatrix Ptent =
        Core::LinAlg::create_interpolation_matrix(*sysmat2, *nullspace, params);

    // compute scale-separation matrix: S = Ptent*Ptent^T
    sep_enr_ = Core::LinAlg::matrix_multiply(Ptent, false, Ptent, true);
    sep_enr_->complete();
  }

  return;
}


//----------------------------------------------------------------------*/
//----------------------------------------------------------------------*/
void FLD::Utils::setup_fluid_fluid_vel_pres_split(const Core::FE::Discretization& fluiddis,
    int ndim, const Core::FE::Discretization& alefluiddis, Core::LinAlg::MapExtractor& extractor,
    std::shared_ptr<Core::LinAlg::Map> fullmap)
{
  std::set<int> veldofset;
  std::set<int> presdofset;

  // for fluid elements
  int numfluidrownodes = fluiddis.num_my_row_nodes();
  for (int i = 0; i < numfluidrownodes; ++i)
  {
    Core::Nodes::Node* fluidnode = fluiddis.l_row_node(i);

    std::vector<int> fluiddof = fluiddis.dof(0, fluidnode);
    for (unsigned j = 0; j < fluiddof.size(); ++j)
    {
      // test for dof position
      if (j < static_cast<unsigned>(ndim))
      {
        veldofset.insert(fluiddof[j]);
      }
      else
      {
        presdofset.insert(fluiddof[j]);
      }
    }
  }

  // for ale_fluid elements
  int numalefluidrownodes = alefluiddis.num_my_row_nodes();
  for (int i = 0; i < numalefluidrownodes; ++i)
  {
    Core::Nodes::Node* alefluidnode = alefluiddis.l_row_node(i);

    std::vector<int> alefluiddof = alefluiddis.dof(alefluidnode);
    for (unsigned j = 0; j < alefluiddof.size(); ++j)
    {
      // test for dof position
      if (j < static_cast<unsigned>(ndim))
      {
        veldofset.insert(alefluiddof[j]);
      }
      else
      {
        presdofset.insert(alefluiddof[j]);
      }
    }
  }

  std::vector<int> veldofmapvec;
  veldofmapvec.reserve(veldofset.size());
  veldofmapvec.assign(veldofset.begin(), veldofset.end());
  veldofset.clear();
  std::shared_ptr<Core::LinAlg::Map> velrowmap = std::make_shared<Core::LinAlg::Map>(
      -1, veldofmapvec.size(), veldofmapvec.data(), 0, fluiddis.get_comm());
  veldofmapvec.clear();

  std::vector<int> presdofmapvec;
  presdofmapvec.reserve(presdofset.size());
  presdofmapvec.assign(presdofset.begin(), presdofset.end());
  presdofset.clear();
  std::shared_ptr<Core::LinAlg::Map> presrowmap = std::make_shared<Core::LinAlg::Map>(
      -1, presdofmapvec.size(), presdofmapvec.data(), 0, alefluiddis.get_comm());
  extractor.setup(*fullmap, presrowmap, velrowmap);
}



// -------------------------------------------------------------------
// compute forces and moments                          rasthofer 08/13
// -------------------------------------------------------------------
void FLD::Utils::lift_drag(const std::shared_ptr<const Core::FE::Discretization> dis,
    const Core::LinAlg::Vector<double>& trueresidual,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp, const int ndim,
    std::shared_ptr<std::map<int, std::vector<double>>>& liftdragvals, bool alefluid)
{
  int myrank = Core::Communication::my_mpi_rank(dis->get_comm());

  std::map<const int, std::set<Core::Nodes::Node*>> ldnodemap;
  std::map<const int, const std::vector<double>*> ldcoordmap;
  std::map<const int, const std::vector<double>*> ldaxismap;
  bool axis_for_moment = false;

  // allocate and initialise lift_drag conditions
  std::vector<const Core::Conditions::Condition*> ldconds;
  dis->get_condition("LIFTDRAG", ldconds);

  // there is an L&D condition if it has a size
  if (ldconds.size())
  {
    // vector with lift&drag forces after communication
    liftdragvals = std::make_shared<std::map<int, std::vector<double>>>();

    for (unsigned i = 0; i < ldconds.size(); ++i)
    {
      // get label of present lift_drag condition
      const int label = ldconds[i]->parameters().get<int>("label");

      ((*liftdragvals))
          .insert(std::pair<int, std::vector<double>>(label, std::vector<double>(6, 0.0)));
    }

    // prepare output
    if (myrank == 0)
    {
      std::cout << "Lift and drag calculation:"
                << "\n";
      if (ndim == 2)
      {
        std::cout << "lift'n'drag Id      F_x             F_y             M_z :"
                  << "\n";
      }
      if (ndim == 3)
      {
        std::cout << "lift'n'drag Id      F_x             F_y             F_z           ";
        std::cout << "M_x             M_y             M_z :"
                  << "\n";
      }
    }

    // sort data
    for (unsigned i = 0; i < ldconds.size(); ++i)
    {
      // get label of present lift_drag condition
      const int label = ldconds[i]->parameters().get<int>("label");

      /* get new nodeset for new label OR:
         return pointer to nodeset for known label ... */
      std::set<Core::Nodes::Node*>& nodes = ldnodemap[label];

      // center coordinates to present label
      ldcoordmap[label] = &ldconds[i]->parameters().get<std::vector<double>>("CENTER");

      // axis of rotation for present label (only needed for 3D)
      if (ldconds[i]->type() == Core::Conditions::SurfLIFTDRAG)
      {
        ldaxismap[label] = &ldconds[i]->parameters().get<std::vector<double>>("AXIS");
        // get pointer to axis vector (if available)
        const std::vector<double>* axisvecptr = ldaxismap[label];
        if (axisvecptr->size() != 3) FOUR_C_THROW("axis vector has not length 3");
        Core::LinAlg::Matrix<3, 1> axisvec(axisvecptr->data(), false);
        if (axisvec.norm2() > 1.0e-9) axis_for_moment = true;  // axis has been set
      }

      // get pointer to its nodal Ids
      const std::vector<int>* ids = ldconds[i]->get_nodes();

      /* put all nodes belonging to the L&D line or surface into 'nodes' which are
         associated with the present label */
      for (unsigned j = 0; j < ids->size(); ++j)
      {
        // give me present node Id
        const int node_id = (*ids)[j];
        // put it into nodeset of actual label if node is new and mine
        if (dis->have_global_node(node_id) && dis->g_node(node_id)->owner() == myrank)
          nodes.insert(dis->g_node(node_id));
      }
    }  // end loop over conditions


    // now step the label map
    for (std::map<const int, std::set<Core::Nodes::Node*>>::const_iterator labelit =
             ldnodemap.begin();
        labelit != ldnodemap.end(); ++labelit)
    {
      const std::set<Core::Nodes::Node*>& nodes =
          labelit->second;                    // pointer to nodeset of present label
      const int label = labelit->first;       // the present label
      std::vector<double> myforces(3, 0.0);   // vector with lift&drag forces
      std::vector<double> mymoments(3, 0.0);  // vector with lift&drag moments

      // get also pointer to center coordinates
      const std::vector<double>* centerCoordvec = ldcoordmap[label];
      if (centerCoordvec->size() != 3) FOUR_C_THROW("axis vector has not length 3");
      Core::LinAlg::Matrix<3, 1> centerCoord(centerCoordvec->data(), false);

      // loop all nodes within my set
      for (std::set<Core::Nodes::Node*>::const_iterator actnode = nodes.begin();
          actnode != nodes.end(); ++actnode)
      {
        const Core::LinAlg::Matrix<3, 1> x(
            (*actnode)->x().data(), false);  // pointer to nodal coordinates
        const Core::LinAlg::Map& rowdofmap = trueresidual.get_map();
        const std::vector<int> dof = dis->dof(*actnode);

        // get nodal forces
        Core::LinAlg::Matrix<3, 1> actforces(Core::LinAlg::Initialization::zero);
        for (int idim = 0; idim < ndim; idim++)
        {
          actforces(idim, 0) = trueresidual.local_values_as_span()[rowdofmap.lid(dof[idim])];
          myforces[idim] += trueresidual.local_values_as_span()[rowdofmap.lid(dof[idim])];
        }
        // z-component remains zero for ndim=2

        // get moment
        Core::LinAlg::Matrix<3, 1> actmoments(Core::LinAlg::Initialization::zero);
        // get vector of point to center point
        Core::LinAlg::Matrix<3, 1> distances;
        distances.update(1.0, x, -1.0, centerCoord);

        // ALE case: take displacements into account
        if (alefluid)
        {
          if (dispnp == nullptr) FOUR_C_THROW("Displacement expected for ale fluid!");
          for (int idim = 0; idim < ndim; idim++)
          {
            distances(idim, 0) += dispnp->local_values_as_span()[rowdofmap.lid(dof[idim])];
          }
        }

        // calculate nodal angular moment with respect to global coordinate system
        Core::LinAlg::Matrix<3, 1> actmoment_gc(Core::LinAlg::Initialization::zero);
        actmoment_gc(0, 0) =
            distances(1) * actforces(2, 0) - distances(2) * actforces(1, 0);  // zero for 2D
        actmoment_gc(1, 0) =
            distances(2) * actforces(0, 0) - distances(0) * actforces(2, 0);  // zero for 2D
        actmoment_gc(2, 0) = distances(0) * actforces(1, 0) - distances(1) * actforces(0, 0);

        if (axis_for_moment)
        {
          const std::vector<double>* axisvecptr = ldaxismap[label];
          Core::LinAlg::Matrix<3, 1> axisvec(axisvecptr->data(), false);
          double norm = 0.0;
          if (axisvec.norm2() != 0.0)
          {
            norm = axisvec.norm2();
            // normed axis vector
            axisvec.scale(1.0 / norm);
          }
          else
            FOUR_C_THROW("norm==0.0!");
          // projection of moment on given axis
          double mdir = actmoment_gc.dot(axisvec);

          actmoments(2, 0) = mdir;
        }
        else
        {
          for (int idim = 0; idim < 3; idim++) actmoments(idim, 0) = actmoment_gc(idim, 0);
        }

        for (int idim = 0; idim < 3; idim++) mymoments[idim] += actmoments(idim, 0);
      }  // end: loop over nodes

      // care for the fact that we are (most likely) parallel
      std::vector<double> global_forces =
          Core::Communication::sum_all(myforces, trueresidual.get_comm());
      std::vector<double> global_moments =
          Core::Communication::sum_all(mymoments, trueresidual.get_comm());
      liftdragvals->at(label)[0] = global_forces[0];
      liftdragvals->at(label)[1] = global_forces[1];
      liftdragvals->at(label)[2] = global_forces[2];
      liftdragvals->at(label)[3] = global_moments[0];
      liftdragvals->at(label)[4] = global_moments[1];
      liftdragvals->at(label)[5] = global_moments[2];

      // do the output
      if (myrank == 0)
      {
        if (ndim == 2)
        {
          std::cout << "     " << label << "         ";
          std::cout << std::scientific << ((*liftdragvals)[label])[0] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[1] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[5];
          std::cout << "\n";
        }
        if (ndim == 3)
        {
          std::cout << "     " << label << "         ";
          std::cout << std::scientific << ((*liftdragvals)[label])[0] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[1] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[2] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[3] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[4] << "    ";
          std::cout << std::scientific << ((*liftdragvals)[label])[5];
          std::cout << "\n";
        }
      }
    }  // end: loop over L&D labels
    if (myrank == 0)
    {
      std::cout << "\n";
    }
  }

  return;
}

// -------------------------------------------------------------------
// write forces and moments to file                    rasthofer 08/13
// -------------------------------------------------------------------
void FLD::Utils::write_lift_drag_to_file(
    const double time, const int step, const std::map<int, std::vector<double>>& liftdragvals)
{
  // print to file
  std::ostringstream header;
  header << std::right << std::setw(16) << "Time" << std::right << std::setw(10) << "Step"
         << std::right << std::setw(10) << "Label" << std::right << std::setw(16) << "F_x"
         << std::right << std::setw(16) << "F_y" << std::right << std::setw(16) << "F_z"
         << std::right << std::setw(16) << "M_x" << std::right << std::setw(16) << "M_y"
         << std::right << std::setw(16) << "M_z";


  for (std::map<int, std::vector<double>>::const_iterator liftdragval = liftdragvals.begin();
      liftdragval != liftdragvals.end(); ++liftdragval)
  {
    std::ostringstream s;
    s << std::right << std::setw(16) << std::scientific << time << std::right << std::setw(10)
      << std::scientific << step << std::right << std::setw(10) << std::scientific
      << liftdragval->first << std::right << std::setw(16) << std::scientific
      << liftdragval->second[0] << std::right << std::setw(16) << std::scientific
      << liftdragval->second[1] << std::right << std::setw(16) << std::scientific
      << liftdragval->second[2] << std::right << std::setw(16) << std::scientific
      << liftdragval->second[3] << std::right << std::setw(16) << std::scientific
      << liftdragval->second[4] << std::right << std::setw(16) << std::scientific
      << liftdragval->second[5];

    std::ostringstream slabel;
    slabel << std::setw(3) << std::setfill('0') << liftdragval->first;
    std::ofstream f;
    const std::string fname = Global::Problem::instance()->output_control_file()->file_name() +
                              ".liftdrag_label_" + slabel.str() + ".txt";

    if (step <= 1)
    {
      f.open(fname.c_str(), std::fstream::trunc);
      f << header.str() << std::endl;
    }
    else
    {
      f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
    }

    f << s.str() << "\n";
    f.close();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, double> FLD::Utils::compute_flow_rates(Core::FE::Discretization& dis,
    Core::LinAlg::Vector<double>& velnp, const std::string& condstring,
    const Inpar::FLUID::PhysicalType physicaltype)
{
  return compute_flow_rates(dis, velnp, nullptr, nullptr, condstring, physicaltype);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, double> FLD::Utils::compute_flow_rates(Core::FE::Discretization& dis,
    Core::LinAlg::Vector<double>& velnp, const std::shared_ptr<Core::LinAlg::Vector<double>>& gridv,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp, const std::string& condstring,
    const Inpar::FLUID::PhysicalType physicaltype)
{
  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_flowrate);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype);

  // note that the flowrate is not yet divided by the area
  std::map<int, double> volumeflowrateperline;

  // get condition
  std::vector<const Core::Conditions::Condition*> conds;
  dis.get_condition(condstring, conds);

  // each condition is on every proc , but might not have condition elements there
  for (std::vector<const Core::Conditions::Condition*>::const_iterator conditer = conds.begin();
      conditer != conds.end(); ++conditer)
  {
    const Core::Conditions::Condition* cond = *conditer;
    const int condID = cond->parameters().get<int>("ConditionID");

    // get a vector layout from the discretization to construct matching
    // vectors and matrices local <-> global dof numbering
    const Core::LinAlg::Map* dofrowmap = dis.dof_row_map();

    // create vector (+ initialization with zeros)
    std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
        Core::LinAlg::create_vector(*dofrowmap, true);

    // call loop over elements
    dis.clear_state();

    dis.set_state("velaf", velnp);
    if (dispnp != nullptr) dis.set_state("dispnp", *dispnp);
    if (gridv != nullptr) dis.set_state("gridv", *gridv);

    dis.evaluate_condition(eleparams, flowrates, condstring, condID);
    dis.clear_state();

    double local_flowrate = 0.0;
    for (int i = 0; i < dofrowmap->num_my_elements(); i++)
    {
      local_flowrate += (flowrates->local_values_as_span()[i]);
    }

    double flowrate = 0.0;
    flowrate = Core::Communication::sum_all(local_flowrate, dofrowmap->get_comm());

    // if(dofrowmap->Comm().MyPID()==0)
    // std::cout << "global flow rate = " << flowrate << "\t condition ID = " << condID <<
    // std::endl;

    // ATTENTION: new definition: outflow is positive and inflow is negative
    volumeflowrateperline[condID] = flowrate;
  }
  return volumeflowrateperline;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, double> FLD::Utils::compute_volume(Core::FE::Discretization& dis,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& velnp,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& gridv,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp,
    const Inpar::FLUID::PhysicalType physicaltype)
{
  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<FLD::Action>("action", FLD::calc_volume);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype);

  std::map<int, double> volumeperline;

  // call loop over elements
  dis.clear_state();
  dis.set_state("velnp", *velnp);
  if (dispnp != nullptr) dis.set_state("dispnp", *dispnp);
  if (gridv != nullptr) dis.set_state("gridv", *gridv);

  std::shared_ptr<Core::LinAlg::SerialDenseVector> volumes =
      std::make_shared<Core::LinAlg::SerialDenseVector>(1);

  // call loop over elements (assemble nothing)
  dis.evaluate_scalars(eleparams, *volumes);
  dis.clear_state();

  volumeperline[0] = (*volumes)(0);

  return volumeperline;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::Utils::write_doubles_to_file(
    const double time, const int step, const std::map<int, double>& data, const std::string& name)
{
  if (data.empty()) FOUR_C_THROW("data vector is empty");

  // print to file
  std::ostringstream header;
  header << std::right << std::setw(16) << "Time" << std::right << std::setw(10) << "Step"
         << std::right << std::setw(10) << "ID" << std::right << std::setw(16) << name;

  for (std::map<int, double>::const_iterator iter = data.begin(); iter != data.end(); ++iter)
  {
    std::ostringstream s;
    s << std::right << std::setw(16) << std::scientific << time << std::right << std::setw(10)
      << std::scientific << step << std::right << std::setw(10) << std::scientific << iter->first
      << std::right << std::setw(29) << std::setprecision(14) << std::scientific << iter->second;

    std::ostringstream slabel;
    slabel << std::setw(3) << std::setfill('0') << iter->first;
    std::ofstream f;
    const std::string fname = Global::Problem::instance()->output_control_file()->file_name() +
                              "." + name + "_ID_" + slabel.str() + ".txt";

    if (step <= 1)
      f.open(fname.c_str(), std::fstream::trunc);  // f << header.str() << std::endl;
    else
      f.open(fname.c_str(), std::fstream::ate | std::fstream::app);

    f << s.str() << "\n";
    f.close();
  }
}


/*----------------------------------------------------------------------*|
 | project vel gradient and store it in given param list        bk 05/15 |
 *----------------------------------------------------------------------*/
void FLD::Utils::project_gradient_and_set_param(Core::FE::Discretization& discret,
    Teuchos::ParameterList& eleparams, const Core::LinAlg::Vector<double>& vel,
    const std::string paraname, bool alefluid)
{
  // project gradient
  std::shared_ptr<Core::LinAlg::MultiVector<double>> projected_velgrad =
      FLD::Utils::project_gradient(discret, vel, alefluid);

  // store multi vector in parameter list after export to col layout
  if (projected_velgrad != nullptr)
  {
    auto tmp = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *discret.node_col_map(), projected_velgrad->num_vectors());
    Core::LinAlg::export_to(*projected_velgrad, *tmp);
    eleparams.set(paraname, tmp);
  }
}


/*----------------------------------------------------------------------*|
 | Project velocity gradient                                    bk 05/15 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> FLD::Utils::project_gradient(
    Core::FE::Discretization& discret, const Core::LinAlg::Vector<double>& vel, bool alefluid)
{
  // reconstruction of second derivatives for fluid residual
  auto recomethod = Teuchos::getIntegralValue<Inpar::FLUID::GradientReconstructionMethod>(
      Global::Problem::instance()->fluid_dynamic_params(), "VELGRAD_PROJ_METHOD");

  const int dim = Global::Problem::instance()->n_dim();
  const int numvec = dim * dim;
  Teuchos::ParameterList params;
  std::shared_ptr<Core::LinAlg::MultiVector<double>> projected_velgrad = nullptr;

  // dependent on the desired projection, just remove this line
  if (not vel.get_map().same_as(*discret.dof_row_map()))
    FOUR_C_THROW("input map is not a dof row map of the fluid");

  switch (recomethod)
  {
    case Inpar::FLUID::gradreco_none:
    {
      // no projection and no parameter in parameter list
    }
    break;
    case Inpar::FLUID::gradreco_spr:
    {
      if (alefluid)
        FOUR_C_THROW(
            "ale fluid is currently not supported everywhere for superconvergent patch recovery, "
            "but it is easy to implement");
      params.set<FLD::Action>("action", FLD::calc_velgrad_ele_center);
      // project velocity gradient of fluid to nodal level via superconvergent patch recovery
      switch (dim)
      {
        case 3:
          projected_velgrad = Core::FE::compute_superconvergent_patch_recovery<3>(
              discret, vel, "vel", numvec, params);
          break;
        case 2:
          projected_velgrad = Core::FE::compute_superconvergent_patch_recovery<2>(
              discret, vel, "vel", numvec, params);
          break;
        case 1:
          projected_velgrad = Core::FE::compute_superconvergent_patch_recovery<1>(
              discret, vel, "vel", numvec, params);
          break;
        default:
          FOUR_C_THROW("only 1/2/3D implementation available for superconvergent patch recovery");
          break;
      }
    }
    break;
    case Inpar::FLUID::gradreco_l2:
    {
      const int solvernumber =
          Global::Problem::instance()->fluid_dynamic_params().get<int>("VELGRAD_PROJ_SOLVER");
      if (solvernumber < 1) FOUR_C_THROW("you have to specify a VELGRAD_PROJ_SOLVER");
      const auto& solverparams = Global::Problem::instance()->solver_params(solvernumber);

      params.set<FLD::Action>("action", FLD::velgradient_projection);

      // set given state for element evaluation
      discret.clear_state();
      discret.set_state("vel", vel);

      // project velocity gradient of fluid to nodal level via L2 projection
      projected_velgrad = Core::FE::compute_nodal_l2_projection(discret, "vel", numvec, params,
          solverparams, Global::Problem::instance()->solver_params_callback());
    }
    break;
    default:
      FOUR_C_THROW("desired projection method not available");
      break;
  }

  return projected_velgrad;
}

FOUR_C_NAMESPACE_CLOSE
