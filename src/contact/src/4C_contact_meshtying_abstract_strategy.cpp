// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_meshtying_abstract_strategy.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_meshtying_noxinterface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                             popp 05/09 |
 *----------------------------------------------------------------------*/
CONTACT::MtAbstractStrategy::MtAbstractStrategy(const Core::LinAlg::Map* dof_row_map,
    const Core::LinAlg::Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<std::shared_ptr<Mortar::Interface>> interface, const int spatialDim,
    const MPI_Comm& comm, const double alphaf, const int maxdof)
    : Mortar::StrategyBase(std::make_shared<Mortar::StrategyDataContainer>(), dof_row_map,
          NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      dualquadslavetrafo_(false)
{
  // call setup method with flag redistributed=FALSE
  setup(false);

  // store interface maps with parallel distribution of underlying
  // problem discretization (i.e. interface maps before parallel
  // redistribution of slave and master sides)
  non_redist_glmdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*glmdofrowmap_);
  non_redist_gsdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*gsdofrowmap_);
  non_redist_gmdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*gmdofrowmap_);
  non_redist_gsmdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*gsmdofrowmap_);

  // build the NOX::Nln::CONSTRAINT::Interface::Required object
  noxinterface_ptr_ = std::make_shared<CONTACT::MtNoxInterface>();

  return;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const CONTACT::MtAbstractStrategy& strategy)
{
  strategy.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 | parallel redistribution                                   popp 09/10 |
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::redistribute_meshtying()
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtAbstractStrategy::redistribute_meshtying");

  // Do we really want to redistribute?
  if (par_redist() && Core::Communication::num_mpi_ranks(get_comm()) > 1)
  {
    // time measurement
    Core::Communication::barrier(get_comm());
    const double t_start = Teuchos::Time::wallTime();

    // do some more stuff with interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      // print parallel distribution
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "\nInterface parallel distribution before rebalancing:" << std::endl;
      interface_[i]->print_parallel_distribution();

      // redistribute optimally among all procs
      interface_[i]->redistribute();

      // call fill complete again
      interface_[i]->fill_complete(Global::Problem::instance()->discretization_map(),
          Global::Problem::instance()->binning_strategy_params(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type(), true, maxdof_);

      // print parallel distribution again
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Interface parallel distribution after rebalancing:" << std::endl;
      interface_[i]->print_parallel_distribution();
    }

    // re-setup strategy with flag redistributed=TRUE
    setup(true);

    // time measurement
    Core::Communication::barrier(get_comm());
    const double t_sum = Teuchos::Time::wallTime() - t_start;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\nTime for parallel redistribution..............." << std::scientific
                << std::setprecision(6) << t_sum << " secs\n";
  }
  else
  {
    // No parallel redistribution to be performed. Just print the current distribution to screen.
    for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->print_parallel_distribution();
  }

  return;
}

/*----------------------------------------------------------------------*
 | setup this strategy object                                popp 08/10 |
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::setup(bool redistributed)
{
  // ------------------------------------------------------------------------
  // setup global accessible Core::LinAlg::Maps
  // ------------------------------------------------------------------------

  // make sure to remove all existing maps first
  // (do NOT remove map of non-interface dofs after redistribution)
  gsdofrowmap_ = nullptr;
  gmdofrowmap_ = nullptr;
  gsmdofrowmap_ = nullptr;
  glmdofrowmap_ = nullptr;
  gdisprowmap_ = nullptr;
  gsnoderowmap_ = nullptr;
  gmnoderowmap_ = nullptr;
  if (!redistributed) gndofrowmap_ = nullptr;

  // element col. map for binning
  initial_elecolmap_.clear();
  initial_elecolmap_.resize(0);

  // make numbering of LM dofs consecutive and unique across N interfaces
  int offset_if = 0;

  // merge interface maps to global maps
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->create_search_tree();

    // build Lagrange multiplier dof map
    interface_[i]->update_lag_mult_sets(offset_if);

    // merge interface Lagrange multiplier dof maps to global LM dof map
    glmdofrowmap_ = Core::LinAlg::merge_map(glmdofrowmap_, interface_[i]->lag_mult_dofs());
    offset_if = glmdofrowmap_->num_global_elements();
    if (offset_if < 0) offset_if = 0;

    // merge interface master, slave maps to global master, slave map
    gsdofrowmap_ = Core::LinAlg::merge_map(gsdofrowmap_, interface_[i]->slave_row_dofs());
    gmdofrowmap_ = Core::LinAlg::merge_map(gmdofrowmap_, interface_[i]->master_row_dofs());
    gsnoderowmap_ = Core::LinAlg::merge_map(gsnoderowmap_, interface_[i]->slave_row_nodes());
    gmnoderowmap_ = Core::LinAlg::merge_map(gmnoderowmap_, interface_[i]->master_row_nodes());

    // store initial element col map for binning strategy
    initial_elecolmap_.push_back(
        std::make_shared<Core::LinAlg::Map>(*interface_[i]->discret().element_col_map()));
  }

  // setup global non-slave-or-master dof map
  // (this is done by splitting from the discretization dof map)
  // (no need to rebuild this map after redistribution)
  if (!redistributed)
  {
    gndofrowmap_ = Core::LinAlg::split_map(*problem_dofs(), *gsdofrowmap_);
    gndofrowmap_ = Core::LinAlg::split_map(*gndofrowmap_, *gmdofrowmap_);
  }

  // setup combined global slave and master dof map
  // setup global displacement dof map
  gsmdofrowmap_ = Core::LinAlg::merge_map(*gsdofrowmap_, *gmdofrowmap_, false);
  gdisprowmap_ = Core::LinAlg::merge_map(*gndofrowmap_, *gsmdofrowmap_, false);

  // ------------------------------------------------------------------------
  // setup global accessible vectors and matrices
  // ------------------------------------------------------------------------

  // setup Lagrange multiplier vectors
  z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  zincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  zold_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  zuzawa_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);

  // setup constraint rhs vector
  constrrhs_ = nullptr;  // only for saddle point problem formulation

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // These matrices need to be applied to the slave displacements
  // in the cases of dual LM interpolation for tet10/hex20 meshes
  // in 3D or for locally linear Lagrange multipliers for line3 meshes
  // in 2D. Here, the displacement basis functions have been modified
  // in order to assure positivity of the D matrix entries and at
  // the same time biorthogonality. Thus, to scale back the modified
  // discrete displacements \hat{d} to the nodal discrete displacements
  // {d}, we have to apply the transformation matrix T and vice versa
  // with the transformation matrix T^(-1).
  //----------------------------------------------------------------------
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (shapefcn == Inpar::Mortar::shape_dual &&
      (n_dim() == 3 || (n_dim() == 2 && lagmultquad == Inpar::Mortar::lagmult_lin)))
    for (int i = 0; i < (int)interface_.size(); ++i)
      dualquadslavetrafo_ += (interface_[i]->quadslave() && !(interface_[i]->is_nurbs()));

  //----------------------------------------------------------------------
  // COMPUTE TRAFO MATRIX AND ITS INVERSE
  //----------------------------------------------------------------------
  if (dualquadslavetrafo())
  {
    // for locally linear Lagrange multipliers, consider both slave and master DOFs,
    // and otherwise, only consider slave DOFs
    if (lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      trafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsmdofrowmap_, 10);
      invtrafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsmdofrowmap_, 10);
    }
    else
    {
      trafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
      invtrafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
    }

    // set of already processed nodes
    // (in order to avoid double-assembly for N interfaces)
    std::set<int> donebefore;

    // for all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
      interface_[i]->assemble_trafo(*trafo_, *invtrafo_, donebefore);

    // fill_complete() transformation matrices
    trafo_->complete();
    invtrafo_->complete();
  }

  return;
}

/*----------------------------------------------------------------------*
 | global evaluation method called from time integrator      popp 06/09 |
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::apply_force_stiff_cmt(
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
    std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
    bool predictor)
{
  // set displacement state
  set_state(Mortar::state_new_displacement, *dis);

  // apply meshtying forces and stiffness
  evaluate(kt, f, dis);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::set_state(
    const enum Mortar::StateType& statetype, const Core::LinAlg::Vector<double>& vec)
{
  switch (statetype)
  {
    case Mortar::state_new_displacement:
    case Mortar::state_old_displacement:
    {
      // set state on interfaces
      for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->set_state(statetype, vec);
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Unsupported state type! (state type = {})", Mortar::state_type_to_string(statetype));
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  do mortar coupling in reference configuration             popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::mortar_coupling(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis)
{
  //********************************************************************
  // initialize and evaluate interfaces
  //********************************************************************
  // for all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // initialize / reset interfaces
    interface_[i]->initialize();

    // evaluate interfaces
    interface_[i]->evaluate();
  }

  //********************************************************************
  // restrict mortar treatment to actual meshtying zone
  //********************************************************************
  restrict_meshtying_zone();

  //********************************************************************
  // initialize and evaluate global mortar stuff
  //********************************************************************
  // (re)setup global Mortar Core::LinAlg::SparseMatrices and Core::LinAlg::Vectors
  dmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
  mmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);
  g_ = Core::LinAlg::create_vector(*gsdofrowmap_, true);

  // assemble D- and M-matrix on all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->assemble_dm(*dmatrix_, *mmatrix_);

  // fill_complete() global Mortar matrices
  dmatrix_->complete();
  mmatrix_->complete(*gmdofrowmap_, *gsdofrowmap_);

  // compute g-vector at global level
  std::shared_ptr<Core::LinAlg::Vector<double>> xs =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> xm =
      Core::LinAlg::create_vector(*gmdofrowmap_, true);
  assemble_coords("slave", true, *xs);
  assemble_coords("master", true, *xm);
  Core::LinAlg::Vector<double> Dxs(*gsdofrowmap_);
  dmatrix_->multiply(false, *xs, Dxs);
  Core::LinAlg::Vector<double> Mxm(*gsdofrowmap_);
  mmatrix_->multiply(false, *xm, Mxm);
  g_->update(1.0, Dxs, 1.0);
  g_->update(-1.0, Mxm, 1.0);

  return;
}

/*----------------------------------------------------------------------*
 |  restrict slave boundary to actual meshtying zone          popp 08/10|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::restrict_meshtying_zone()
{
  // Step 1: detect tied slave nodes on all interfaces
  int localfounduntied = 0;
  int globalfounduntied = 0;
  for (int i = 0; i < (int)interface_.size(); ++i)
    interface_[i]->detect_tied_slave_nodes(localfounduntied);
  globalfounduntied = Core::Communication::sum_all(localfounduntied, get_comm());

  // get out of here if the whole slave surface is tied
  if (globalfounduntied == 0) return;

  // print message
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "*restrict_meshtying_zone*...............";
    fflush(stdout);
  }

  //**********************************************************************
  // check for infeasible discretization types and LM types     popp 05/11
  // (currently RMZ only for 1st order slave elements and standard LM)
  //**********************************************************************
  // Currently, we need strictly positive LM shape functions for RMZ
  // to work properly. This is only satisfied for 1st order interpolation
  // with standard Lagrange multipliers. As soon as we have implemented
  // the modified biorthogonality (only defined on projecting slave part),
  // the 1st order interpolation case with dual Lagrange multiplier will
  // work, too. All 2nd order interpolation cases are difficult, since
  // we would require strictly positive displacement shape functions, which
  // is only possible via a proper basis transformation.
  //**********************************************************************
  bool quadratic = false;
  for (int i = 0; i < (int)interface_.size(); ++i) quadratic += interface_[i]->quadslave();
  if (quadratic) FOUR_C_THROW("restrict_meshtying_zone only implemented for first-order elements");

  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  if ((shapefcn == Inpar::Mortar::shape_dual || shapefcn == Inpar::Mortar::shape_petrovgalerkin) &&
      Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
          params(), "LM_DUAL_CONSISTENT") == Inpar::Mortar::consistent_none)
    FOUR_C_THROW(
        "ERROR: restrict_meshtying_zone for dual shape functions "
        "only implemented in combination with consistent boundary modification");

  // Step 2: restrict slave node/dof sets of all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->restrict_slave_sets();

  // Step 3: re-setup global maps and vectors with flag redistributed=FALSE
  // (this flag must be FALSE here, because the slave set has been reduced and
  // thus the non-interface set N needs to be updated / re-setup as well)
  setup(false);

  // Step 4: re-setup slave dof row map with parallel distribution of
  // underlying problem discretization (i.e. slave dof row maps before
  // parallel redistribution) -> introduce restriction!
  if (par_redist())
  {
    // allreduce restricted slave dof row map in new distribution
    std::shared_ptr<Core::LinAlg::Map> fullsdofs = Core::LinAlg::allreduce_e_map(*gsdofrowmap_);

    // map data to be filled
    std::vector<int> data;

    // loop over all entries of allreduced map
    const int numMyFullSlaveDofs = fullsdofs->num_my_elements();
    for (int k = 0; k < numMyFullSlaveDofs; ++k)
    {
      // get global ID of current dof
      int gid = fullsdofs->gid(k);

      /* Check if this GID is stored on this processor in the slave dof row map based on the old
       * distribution and add to data vector if so.
       */
      if (non_redist_gsdofrowmap_->my_gid(gid)) data.push_back(gid);
    }

    // re-setup old slave dof row map (with restriction now)
    non_redist_gsdofrowmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)data.size(), data.data(), 0, get_comm());
  }

  // Step 5: re-setup internal dof row map (non-interface dofs)
  // (this one has been re-computed in setup() above, but is possibly
  // incorrect due to parallel redistribution of the interfaces)
  // --> recompute based on splitting with slave and master dof row maps
  // before parallel redistribution!
  if (par_redist())
  {
    gndofrowmap_ = Core::LinAlg::split_map(*problem_dofs(), *non_redist_gsdofrowmap_);
    gndofrowmap_ = Core::LinAlg::split_map(*gndofrowmap_, *non_redist_gmdofrowmap_);
  }

  // Step 6: re-setup displacement dof row map with current parallel
  // distribution (possibly wrong, same argument as above)
  if (par_redist())
  {
    gdisprowmap_ = Core::LinAlg::merge_map(*gndofrowmap_, *gsmdofrowmap_, false);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  mesh initialization for rotational invariance              popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::mesh_initialization(
    std::shared_ptr<Core::LinAlg::Vector<double>> Xslavemod)
{
  //**********************************************************************
  // (1) perform mesh initialization node by node
  //**********************************************************************
  // IMPORTANT NOTE:
  // We have to be very careful on which nodes on which processor to
  // relocate! Basically, every processor needs to know about relocation
  // of all its column nodes in the standard column map with overlap=1,
  // because all these nodes participate in the processor's element
  // evaluation! Thus, the modified slave positions are first exported
  // to the column map of the respective interface and the modification
  // loop is then also done with respect to this node column map!
  // A second concern is that we are dealing with a special interface
  // discretization (including special meshtying nodes, too) here, This
  // interface discretization has been set up for dealing with meshtying
  // ONLY, and there is still the underlying problem discretization
  // dealing with the classical finite element evaluation. Thus, it is
  // very important that we apply the nodal relocation to BOTH the
  // Mortar::Node(s) in the meshtying interface discretization AND to the
  // Core::Nodes::Node(s) in the underlying problem discretization. However, the
  // second aspect needs to be done by the respective control routine,
  // i.e. in the case of 4C in strtimint.cpp and NOT here.
  //**********************************************************************

  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // export Xslavemod to column map for current interface
    Core::LinAlg::Vector<double> Xslavemodcol(*(interface_[i]->slave_col_dofs()), false);
    Core::LinAlg::export_to(*Xslavemod, Xslavemodcol);

    // loop over all slave column nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_col_nodes()->num_my_elements(); ++j)
    {
      // get global ID of current node
      int gid = interface_[i]->slave_col_nodes()->gid(j);

      // get the mortar node
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Mortar::Node* mtnode = dynamic_cast<Mortar::Node*>(node);

      // new nodal position and problem dimension
      std::vector<double> Xnew(n_dim(), 0.0);

      //******************************************************************
      // compute new nodal position
      //******************************************************************
      // get corresponding entries from Xslavemodcol
      int numdof = mtnode->num_dof();
      if (n_dim() != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // find DOFs of current node in Xslavemodcol and extract this node's position
      std::vector<int> locindex(numdof);

      for (int dof = 0; dof < numdof; ++dof)
      {
        locindex[dof] = (Xslavemodcol.get_map()).lid(mtnode->dofs()[dof]);
        if (locindex[dof] < 0) FOUR_C_THROW("Did not find dof in map");
        Xnew[dof] = Xslavemodcol.local_values_as_span()[locindex[dof]];
      }

      // check is mesh distortion is still OK
      // (throw a FOUR_C_THROW if length of relocation is larger than 80%
      // of an adjacent element edge -> see Puso, IJNME, 2004)
      const double limit = 0.8;
      double relocation = 0.0;
      if (n_dim() == 2)
      {
        relocation = sqrt((Xnew[0] - mtnode->x()[0]) * (Xnew[0] - mtnode->x()[0]) +
                          (Xnew[1] - mtnode->x()[1]) * (Xnew[1] - mtnode->x()[1]));
      }
      else if (n_dim() == 3)
      {
        relocation = sqrt((Xnew[0] - mtnode->x()[0]) * (Xnew[0] - mtnode->x()[0]) +
                          (Xnew[1] - mtnode->x()[1]) * (Xnew[1] - mtnode->x()[1]) +
                          (Xnew[2] - mtnode->x()[2]) * (Xnew[2] - mtnode->x()[2]));
      }
      else
      {
        FOUR_C_THROW("Problem dimension must be either 2 or 3!");
      }

      // check is only done once per node (by owing processor)
      if (Core::Communication::my_mpi_rank(get_comm()) == mtnode->owner())
      {
        bool isok = mtnode->check_mesh_distortion(relocation, limit);
        if (!isok) FOUR_C_THROW("Mesh distortion generated by relocation is too large!");
      }

      // modification of xspatial (spatial coordinates)
      for (int k = 0; k < n_dim(); ++k) mtnode->xspatial()[k] = Xnew[k];

      // modification of xref (reference coordinates)
      mtnode->set_pos(Xnew);
    }
  }

  //**********************************************************************
  // (2) re-evaluate constraints in reference configuration
  //**********************************************************************
  // initialize
  g_ = Core::LinAlg::create_vector(*gsdofrowmap_, true);

  // compute g-vector at global level
  std::shared_ptr<Core::LinAlg::Vector<double>> xs =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> xm =
      Core::LinAlg::create_vector(*gmdofrowmap_, true);
  assemble_coords("slave", true, *xs);
  assemble_coords("master", true, *xm);
  Core::LinAlg::Vector<double> Dxs(*gsdofrowmap_);
  dmatrix_->multiply(false, *xs, Dxs);
  Core::LinAlg::Vector<double> Mxm(*gsdofrowmap_);
  mmatrix_->multiply(false, *xm, Mxm);
  g_->update(1.0, Dxs, 1.0);
  g_->update(-1.0, Mxm, 1.0);
}

/*----------------------------------------------------------------------*
 | call appropriate evaluate for contact evaluation           popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::evaluate(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis)
{
  // trivial (no choice as for contact)
  evaluate_meshtying(kteff, feff, dis);
  return;
}

/*----------------------------------------------------------------------*
 |  Store Lagrange multipliers into Mortar::Node                popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::store_nodal_quantities(Mortar::StrategyBase::QuantityType type)
{
  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // get global quantity to be stored in nodes
    std::shared_ptr<const Core::LinAlg::Vector<double>> vectorglobal = nullptr;
    switch (type)
    {
      case Mortar::StrategyBase::lmcurrent:
      {
        vectorglobal = lagrange_multiplier();
        break;
      }
      case Mortar::StrategyBase::lmold:
      {
        vectorglobal = lagrange_multiplier_old();
        break;
      }
      case Mortar::StrategyBase::lmupdate:
      {
        vectorglobal = lagrange_multiplier();
        break;
      }
      case Mortar::StrategyBase::lmuzawa:
      {
        vectorglobal = lagr_mult_uzawa();
        break;
      }
      default:
      {
        FOUR_C_THROW("store_nodal_quantities: Unknown state std::string variable!");
        break;
      }
    }  // switch

    // export global quantity to current interface slave dof row map
    std::shared_ptr<const Core::LinAlg::Map> sdofrowmap = interface_[i]->slave_row_dofs();
    Core::LinAlg::Vector<double> vectorinterface(*sdofrowmap);

    if (vectorglobal != nullptr)
      Core::LinAlg::export_to(*vectorglobal, vectorinterface);
    else
      FOUR_C_THROW("store_nodal_quantities: Null vector handed in!");

    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->slave_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Mortar::Node* mtnode = dynamic_cast<Mortar::Node*>(node);

      // be aware of problem dimension
      int numdof = mtnode->num_dof();
      if (n_dim() != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // find indices for DOFs of current node in Core::LinAlg::Vector<double>
      // and extract this node's quantity from vectorinterface
      std::vector<int> locindex(n_dim());

      for (int dof = 0; dof < n_dim(); ++dof)
      {
        locindex[dof] = (vectorinterface.get_map()).lid(mtnode->dofs()[dof]);
        if (locindex[dof] < 0) FOUR_C_THROW("StoreNodalQuantities: Did not find dof in map");

        switch (type)
        {
          case Mortar::StrategyBase::lmcurrent:
          {
            mtnode->mo_data().lm()[dof] = (vectorinterface).local_values_as_span()[locindex[dof]];
            break;
          }
          case Mortar::StrategyBase::lmold:
          {
            mtnode->mo_data().lmold()[dof] =
                (vectorinterface).local_values_as_span()[locindex[dof]];
            break;
          }
          case Mortar::StrategyBase::lmuzawa:
          {
            mtnode->mo_data().lmuzawa()[dof] =
                (vectorinterface).local_values_as_span()[locindex[dof]];
            break;
          }
          case Mortar::StrategyBase::lmupdate:
          {
            // throw a FOUR_C_THROW if node is Active and DBC
            if (mtnode->is_dbc())
              FOUR_C_THROW(
                  "Slave Node {} is active and at the same time carries D.B.C.s!", mtnode->id());

            // store updated LM into node
            mtnode->mo_data().lm()[dof] = (vectorinterface).local_values_as_span()[locindex[dof]];
            break;
          }
          default:
          {
            FOUR_C_THROW("store_nodal_quantities: Unknown state std::string variable!");
            break;
          }
        }  // switch
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Store dirichlet B.C. status into Mortar::Node               popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::store_dirichlet_status(
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmaps)
{
  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->slave_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Mortar::Node* mtnode = dynamic_cast<Mortar::Node*>(node);

      // check if this node's dofs are in dbcmap
      for (int k = 0; k < mtnode->num_dof(); ++k)
      {
        int currdof = mtnode->dofs()[k];
        int lid = (dbcmaps->cond_map())->lid(currdof);

        // store dbc status if found
        if (lid >= 0 && mtnode->dbc_dofs()[k] == false)
        {
          mtnode->set_dbc() = true;

          // check compatibility of contact symmetry condition and displacement dirichlet conditions
          if (lid < 0 && mtnode->dbc_dofs()[k] == true)
            FOUR_C_THROW(
                "Inconsistency in structure Dirichlet conditions and Mortar symmetry conditions");
        }
      }
    }
  }

  // create old style dirichtoggle vector (supposed to go away)
  non_redist_gsdirichtoggle_ = Core::LinAlg::create_vector(*gsdofrowmap_, true);
  Core::LinAlg::Vector<double> temp(*(dbcmaps->cond_map()));
  temp.put_scalar(1.0);
  Core::LinAlg::export_to(temp, *non_redist_gsdirichtoggle_);

  return;
}

/*----------------------------------------------------------------------*
 | Update meshtying at end of time step                       popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // store Lagrange multipliers
  // (we need this for interpolation of the next generalized mid-point)
  zold_->update(1.0, *z_, 0.0);
  store_nodal_quantities(Mortar::StrategyBase::lmold);

  // old displacements in nodes
  // (this is needed for calculating the auxiliary positions in
  // binarytree contact search)
  set_state(Mortar::state_old_displacement, *dis);
}

/*----------------------------------------------------------------------*
 |  read restart information for meshtying                    popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::do_read_restart(
    Core::IO::DiscretizationReader& reader, std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // check whether this is a restart with meshtying of a previously
  // non-meshtying simulation run
  const bool restartwithmeshtying = params().get<bool>("RESTART_WITH_MESHTYING");

  // set displacement state
  set_state(Mortar::state_new_displacement, *dis);

  // read restart information on Lagrange multipliers
  z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  zincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  if (!restartwithmeshtying) reader.read_vector(z_, "mt_lagrmultold");
  store_nodal_quantities(Mortar::StrategyBase::lmcurrent);
  zold_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  if (!restartwithmeshtying) reader.read_vector(zold_, "mt_lagrmultold");
  store_nodal_quantities(Mortar::StrategyBase::lmold);

  // only for Uzawa strategy
  auto st = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params(), "STRATEGY");
  if (st == CONTACT::SolvingStrategy::uzawa)
  {
    zuzawa_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    if (!restartwithmeshtying) reader.read_vector(zuzawa_, "mt_lagrmultold");
    store_nodal_quantities(Mortar::StrategyBase::lmuzawa);
  }
}

/*----------------------------------------------------------------------*
 |  print interfaces (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::print(std::ostream& os) const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    os << "--------------------------------- CONTACT::MtAbstractStrategy\n"
       << "Meshtying interfaces: " << (int)interface_.size() << std::endl
       << "-------------------------------------------------------------\n";
  }
  Core::Communication::barrier(get_comm());
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    std::cout << *(interface_[i]);
  }
  Core::Communication::barrier(get_comm());

  return;
}

/*----------------------------------------------------------------------*
 | print active set information                               popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::print_active_set() const { return; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::assemble_coords(
    const std::string& sidename, bool ref, Core::LinAlg::Vector<double>& vec)
{
  // NOTE:
  // An alternative way of doing this would be to loop over
  // all interfaces and to assemble the coordinates there.
  // In that case, one would have to be very careful with
  // edge nodes / crosspoints, which must not be assembled
  // twice. The solution would be to overwrite the corresponding
  // entries in the Core::LinAlg::Vector<double> instead of using Assemble().

  // decide which side (slave or master)
  std::shared_ptr<Core::LinAlg::Map> sidemap = nullptr;
  if (sidename == "slave")
    sidemap = gsnoderowmap_;
  else if (sidename == "master")
    sidemap = gmnoderowmap_;
  else
    FOUR_C_THROW("Unknown sidename");

  // loop over all row nodes of this side (at the global level)
  for (int j = 0; j < sidemap->num_my_elements(); ++j)
  {
    int gid = sidemap->gid(j);

    // find this node in interface discretizations
    bool found = false;
    Core::Nodes::Node* node = nullptr;
    for (int k = 0; k < (int)interface_.size(); ++k)
    {
      found = interface_[k]->discret().have_global_node(gid);
      if (found)
      {
        node = interface_[k]->discret().g_node(gid);
        break;
      }
    }
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    Mortar::Node* mtnode = dynamic_cast<Mortar::Node*>(node);

    // prepare assembly
    Core::LinAlg::SerialDenseVector val(n_dim());
    std::vector<int> lm(n_dim());
    std::vector<int> lmowner(n_dim());

    for (int k = 0; k < n_dim(); ++k)
    {
      // reference (true) or current (false) configuration
      if (ref)
        val[k] = mtnode->x()[k];
      else
        val[k] = mtnode->xspatial()[k];
      lm[k] = mtnode->dofs()[k];
      lmowner[k] = mtnode->owner();
    }

    // do assembly
    Core::LinAlg::assemble(vec, val, lm, lmowner);
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::collect_maps_for_preconditioner(
    std::shared_ptr<Core::LinAlg::Map>& MasterDofMap,
    std::shared_ptr<Core::LinAlg::Map>& SlaveDofMap,
    std::shared_ptr<Core::LinAlg::Map>& InnerDofMap,
    std::shared_ptr<Core::LinAlg::Map>& ActiveDofMap) const
{
  InnerDofMap = gndofrowmap_;  // global internal dof row map

  // global active slave dof row map (all slave dofs are active  in meshtying)
  if (non_redist_gsdofrowmap_ != nullptr)
    ActiveDofMap = non_redist_gsdofrowmap_;
  else
    ActiveDofMap = gsdofrowmap_;

  // check if parallel redistribution is used
  // if parallel redistribution is activated, then use (original) maps before redistribution
  // otherwise we use just the standard master/slave maps
  if (non_redist_gsdofrowmap_ != nullptr)
    SlaveDofMap = non_redist_gsdofrowmap_;
  else
    SlaveDofMap = gsdofrowmap_;
  if (non_redist_gmdofrowmap_ != nullptr)
    MasterDofMap = non_redist_gmdofrowmap_;
  else
    MasterDofMap = gmdofrowmap_;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool CONTACT::MtAbstractStrategy::is_saddle_point_system() const
{
  if (system_type() == CONTACT::SystemType::saddlepoint) return true;

  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool CONTACT::MtAbstractStrategy::is_condensed_system() const
{
  if (system_type() != CONTACT::SystemType::saddlepoint) return true;

  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::fill_maps_for_preconditioner(
    std::vector<Teuchos::RCP<Core::LinAlg::Map>>& maps) const
{
  /* FixMe This function replaces the deprecated collect_maps_for_preconditioner(),
   * the old version can be deleted, as soon as the contact uses the new
   * structure framework. */

  if (maps.size() != 4) FOUR_C_THROW("The vector size has to be 4!");
  /* check if parallel redistribution is used
   * if parallel redistribution is activated, then use (original) maps
   * before redistribution otherwise we use just the standard master/slave
   * maps */
  // (0) masterDofMap
  if (non_redist_gmdofrowmap_ != nullptr)
    maps[0] = Teuchos::rcp(non_redist_gmdofrowmap_);
  else
    maps[0] = Teuchos::rcp(gmdofrowmap_);
  // (1) slaveDofMap
  // (3) activeDofMap (all slave nodes are active!)
  if (non_redist_gsdofrowmap_ != nullptr)
  {
    maps[1] = Teuchos::rcp(non_redist_gsdofrowmap_);
    maps[3] = Teuchos::rcp(non_redist_gsdofrowmap_);
  }
  else
  {
    maps[1] = Teuchos::rcp(gsdofrowmap_);
    maps[3] = Teuchos::rcp(gsdofrowmap_);
  }
  // (2) innerDofMap
  maps[2] = Teuchos::rcp(gndofrowmap_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtAbstractStrategy::postprocess_quantities_per_interface(
    std::shared_ptr<Teuchos::ParameterList> outputParams) const
{
  // Evaluate slave and master forces
  auto fcslave = std::make_shared<Core::LinAlg::Vector<double>>(d_matrix()->row_map());
  auto fcmaster = std::make_shared<Core::LinAlg::Vector<double>>(m_matrix()->domain_map());
  d_matrix()->multiply(true, *zold_, *fcslave);
  m_matrix()->multiply(true, *zold_, *fcmaster);

  // Append data to parameter list
  outputParams->set(
      "interface traction", std::const_pointer_cast<const Core::LinAlg::Vector<double>>(zold_));
  outputParams->set(
      "slave forces", std::const_pointer_cast<const Core::LinAlg::Vector<double>>(fcslave));
  outputParams->set(
      "master forces", std::const_pointer_cast<const Core::LinAlg::Vector<double>>(fcmaster));

  for (std::vector<std::shared_ptr<Mortar::Interface>>::const_iterator it = interface_.begin();
      it < interface_.end(); ++it)
    (*it)->postprocess_quantities(*outputParams);
}

FOUR_C_NAMESPACE_CLOSE
