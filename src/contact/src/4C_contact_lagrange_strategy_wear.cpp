// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_lagrange_strategy_wear.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_lagrange_strategy.hpp"
#include "4C_contact_wear_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*
 | ctor (public)                                             farah 09/13|
 *----------------------------------------------------------------------*/
Wear::LagrangeStrategyWear::LagrangeStrategyWear(
    const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interfaces,
    int dim, MPI_Comm comm, double alphaf, int maxdof)
    : LagrangeStrategy(
          data_ptr, dof_row_map, NodeRowMap, params, interfaces, dim, comm, alphaf, maxdof),
      weightedwear_(false),
      wbothpv_(false),
      wearimpl_(false),
      wearprimvar_(false),
      wearbothpv_(false),
      weartimescales_(false),
      sswear_(LagrangeStrategyWear::params().get<bool>("SSWEAR"))
{
  // cast to  wearinterfaces
  for (int z = 0; z < (int)interfaces.size(); ++z)
  {
    interface_.push_back(std::dynamic_pointer_cast<Wear::WearInterface>(interfaces[z]));
    if (interface_[z] == nullptr) FOUR_C_THROW("LagrangeStrategyWear: Interface-cast failed!");
  }

  // set wear contact status
  auto wtype =
      Teuchos::getIntegralValue<Inpar::Wear::WearType>(LagrangeStrategyWear::params(), "WEARTYPE");
  auto wside =
      Teuchos::getIntegralValue<Inpar::Wear::WearSide>(LagrangeStrategyWear::params(), "WEAR_SIDE");
  auto wtime = Teuchos::getIntegralValue<Inpar::Wear::WearTimeScale>(
      LagrangeStrategyWear::params(), "WEAR_TIMESCALE");
  auto wtimint = Teuchos::getIntegralValue<Inpar::Wear::WearTimInt>(
      LagrangeStrategyWear::params(), "WEARTIMINT");
  auto wlaw =
      Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(LagrangeStrategyWear::params(), "WEARLAW");

  // set wear contact status
  if (wlaw != Inpar::Wear::wear_none and wtype == Inpar::Wear::wear_intstate) weightedwear_ = true;

  // discrete both-sided wear for active set output
  if (wside == Inpar::Wear::wear_both and wtype == Inpar::Wear::wear_primvar) wbothpv_ = true;
  if (wtimint == Inpar::Wear::wear_impl) wearimpl_ = true;

  // set wear contact discretization
  if (wtype == Inpar::Wear::wear_primvar) wearprimvar_ = true;

  // both sided wear for discrete wear
  if (wside == Inpar::Wear::wear_both and wtype == Inpar::Wear::wear_primvar) wearbothpv_ = true;

  // different wear timescales?
  if (wtime == Inpar::Wear::wear_time_different) weartimescales_ = true;

  return;
}


/*----------------------------------------------------------------------*
 | setup this strategy object                               seitz 11/16 |
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::setup(bool redistributed, bool init)
{
  // base class setup
  AbstractStrategy::setup(redistributed, init);

  // wear specific setup
  setup_wear(redistributed, init);
}


/*----------------------------------------------------------------------*
 | setup this strategy object                               farah 09/13 |
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::setup_wear(bool redistributed, bool init)
{
  // max dof number -- disp dofs and lm dofs considered
  maxdofwear_ = maxdof_ + glmdofrowmap_->num_global_elements();

  // ------------------------------------------------------------------------
  // setup global accessible Core::LinAlg::Maps
  // ------------------------------------------------------------------------

  // make sure to remove all existing maps first
  // (do NOT remove map of non-interface dofs after redistribution)
  gminvolvednodes_ = nullptr;  // all involved master nodes
  gminvolveddofs_ = nullptr;   // all involved master dofs
  gwdofrowmap_ = nullptr;
  gwmdofrowmap_ = nullptr;
  gslipn_ = nullptr;        // vector dummy for wear - slave slip dofs
  gsdofnrowmap_ = nullptr;  // vector dummy for wear - slave all dofs
  gwinact_ = nullptr;       // vector dummy for wear - slave inactive dofss

  gmdofnrowmap_ = nullptr;  // vector dummy for wear - master all dofs
  gmslipn_ = nullptr;
  gwminact_ = nullptr;

  galldofnrowmap_ = nullptr;
  gwalldofrowmap_ = nullptr;

  // make numbering of LM dofs consecutive and unique across N interfaces
  int offset_if = 0;

  // merge interface maps to global maps
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // ****************************************************
    // for wear as own variable
    // ****************************************************
    if (wearprimvar_)
    {
      // build wear dof map
      interface_[i]->update_w_sets(offset_if, maxdofwear_, wearbothpv_);

      // merge interface slave wear dof maps to global slave wear dof map
      gwdofrowmap_ = Core::LinAlg::merge_map(gwdofrowmap_, interface_[i]->w_dofs());
      offset_if = gwdofrowmap_->num_global_elements();
      if (offset_if < 0) offset_if = 0;

      // merge interface master wear dof maps to global slave wear dof map
      if (wearbothpv_)
      {
        gwmdofrowmap_ = Core::LinAlg::merge_map(gwmdofrowmap_, interface_[i]->wm_dofs());
        offset_if = gwmdofrowmap_->num_global_elements();
        if (offset_if < 0) offset_if = 0;
      }

      // slavenode normal part (first entry)
      interface_[i]->split_slave_dofs();
      gsdofnrowmap_ = Core::LinAlg::merge_map(gsdofnrowmap_, interface_[i]->sn_dofs());

      // masternode normal part (first entry)
      if (wearbothpv_)
      {
        interface_[i]->split_master_dofs();
        gmdofnrowmap_ = Core::LinAlg::merge_map(gmdofnrowmap_, interface_[i]->mn_dofs());

        interface_[i]->build_active_set_master();
        gmslipn_ = Core::LinAlg::merge_map(gmslipn_, interface_[i]->slip_master_n_dofs(), false);
      }

      // initialize nodal wcurr for integrator (mod. gap)
      for (int j = 0; j < (int)interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
      {
        int gid = interface_[i]->slave_row_nodes()->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

        cnode->wear_data().wcurr()[0] = 0.0;
      }

      if (wearbothpv_)
      {
        for (int j = 0; j < (int)interface_[i]->master_col_nodes()->num_my_elements(); ++j)
        {
          int gid = interface_[i]->master_col_nodes()->gid(j);
          Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
          if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

          cnode->wear_data().wcurr()[0] = 0.0;
        }
      }
    }
    // ****************************************************
    // both-sided wear specific
    // ****************************************************
    if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(params(), "WEAR_SIDE") ==
            Inpar::Wear::wear_both and
        wearprimvar_ == false)
    {
      gminvolvednodes_ =
          Core::LinAlg::merge_map(gminvolvednodes_, interface_[i]->involved_nodes(), false);
      gminvolveddofs_ =
          Core::LinAlg::merge_map(gminvolveddofs_, interface_[i]->involved_dofs(), false);
    }
  }

  // get the normal dof of slip nodes for wear definition
  if (wearprimvar_)
  {
    gslipn_ = Core::LinAlg::split_map(*gslipdofs_, *gslipt_);

    if (sswear_)
      gwinact_ = Core::LinAlg::split_map(*gsdofnrowmap_, *gactiven_);
    else
      gwinact_ = Core::LinAlg::split_map(*gsdofnrowmap_, *gslipn_);

    if (wearbothpv_)
    {
      gwminact_ = Core::LinAlg::split_map(*gmdofnrowmap_, *gmslipn_);

      // complete wear dofs (s+m)
      galldofnrowmap_ = Core::LinAlg::merge_map(*gsdofnrowmap_, *gmdofnrowmap_, false);
      gwalldofrowmap_ = Core::LinAlg::merge_map(*gwdofrowmap_, *gwmdofrowmap_, false);
    }
  }

  // ------------------------------------------------------------------------
  // setup global accessible vectors and matrices
  // ------------------------------------------------------------------------
  if (wearprimvar_)
  {
    // initialize vectors and matrices
    if (!redistributed)
    {
      // setup Lagrange multiplier vectors
      w_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
      wincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
    }

    // In the redistribution case, first check if the vectors and
    // matrices have already been defined, If yes, transform them
    // to the new redistributed maps. If not, initialize them.
    // Moreover, store redistributed quantities into nodes!!!
    else
    {
      // setup Lagrange multiplier vectors
      if (w_ == nullptr)
      {
        w_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
      }
      else
      {
        std::shared_ptr<Core::LinAlg::Vector<double>> neww =
            std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
        Core::LinAlg::export_to(*w_, *neww);
        w_ = neww;
      }

      if (wincr_ == nullptr)
      {
        wincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
      }
      else
      {
        std::shared_ptr<Core::LinAlg::Vector<double>> newwincr =
            std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
        Core::LinAlg::export_to(*wincr_, *newwincr);
        wincr_ = newwincr;
      }
    }

    // both sided wear
    if (wearbothpv_)
    {
      // initialize vectors and matrices
      if (!redistributed)
      {
        // setup Lagrange multiplier vectors
        wm_ = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
        wmincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
      }

      // In the redistribution case, first check if the vectors and
      // matrices have already been defined, If yes, transform them
      // to the new redistributed maps. If not, initialize them.
      // Moreover, store redistributed quantities into nodes!!!
      else
      {
        // setup Lagrange multiplier vectors
        if (wm_ == nullptr)
        {
          wm_ = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
        }
        else
        {
          std::shared_ptr<Core::LinAlg::Vector<double>> neww =
              std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
          Core::LinAlg::export_to(*wm_, *neww);
          wm_ = neww;
        }

        if (wmincr_ == nullptr)
        {
          wmincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
        }
        else
        {
          std::shared_ptr<Core::LinAlg::Vector<double>> newwincr =
              std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
          Core::LinAlg::export_to(*wmincr_, *newwincr);
          wmincr_ = newwincr;
        }
      }
    }
  }

  // output wear ... this is for the unweighted wear*n vector
  wearoutput_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  wearoutput2_ = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);

  return;
}

/*----------------------------------------------------------------------*
 | initialize wear stuff for next Newton step                farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::initialize_mortar()
{
  // for self contact, slave and master sets may have changed,
  // thus we have to update them before initializing D,M etc.
  update_global_self_contact_state();

  /**********************************************************************/
  /* initialize Dold and Mold if not done already                       */
  /**********************************************************************/
  if (dold_ == nullptr)
  {
    dold_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
    dold_->zero();
    dold_->complete();
  }
  if (mold_ == nullptr)
  {
    mold_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);
    mold_->zero();
    mold_->complete(*gmdofrowmap_, *gsdofrowmap_);
  }

  /**********************************************************************/
  /* (re)setup global Mortar Core::LinAlg::SparseMatrices and Core::LinAlg::Vectors  */
  /**********************************************************************/
  dmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
  d2matrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  mmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);

  // global gap
  wgap_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_, true);

  /**********************************************************************/
  /* (re)setup global wear Core::LinAlg::Vector<double> (for all wear problems)        */
  /**********************************************************************/
  if (!wearprimvar_)
    wearvector_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_, true);

  /**********************************************************************/
  /* in the case of dual quad 3D, the modified D matrices are setup     */
  /**********************************************************************/
  if (friction_ && is_dual_quad_slave_trafo())
  {
    // initialize Dold and Mold if not done already
    if (doldmod_ == nullptr)
    {
      doldmod_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
      doldmod_->zero();
      doldmod_->complete();
    }
    // setup of dmatrixmod_
    dmatrixmod_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Assemble wear stuff for next Newton step                  farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::assemble_mortar()
{
  // call base routine
  CONTACT::AbstractStrategy::assemble_mortar();

  // for all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    //************************************************************
    // only assemble D2 for both-sided wear --> unweights the
    // weighted wear increment in master side
    // --> based on weak dirichlet bc!
    if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(params(), "WEAR_SIDE") ==
            Inpar::Wear::wear_both and
        !wearprimvar_)
      interface_[i]->assemble_d2(*d2matrix_);

    //************************************************************
    // assemble wear vector
    if (!wearprimvar_) interface_[i]->assemble_wear(*wearvector_);
  }  // end interface loop

  // *********************************************************************************
  // modify gap vector towards wear, only if no structure with ale is applied
  // This additional gap is also required for an implicit ALE-wear algorithm !!!
  // THIS IS THE EXPLICIT WEAR ALGORITHM
  // wearvector_ only updated at the end of a time step --> this newton-step-wise
  // update is not elegant!
  // *********************************************************************************
  if (!wearimpl_ and !wearprimvar_ and
      params().get<CONTACT::Problemtype>("PROBTYPE") != CONTACT::Problemtype::structalewear)
  {
    wgap_->update(1.0, *wearvector_, 1.0);
  }
  else
  {
    // internal state variable approach:
    if (wearimpl_ and !wearprimvar_)
    {
      // update the gap function with the current wear increment
      // this is for the implicit wear algorithm
      // we have to update the wear-increment in every newton-step and not just
      // after a time step!
      store_nodal_quantities(Mortar::StrategyBase::weightedwear);
      interface_[0]->assemble_wear(*wearvector_);
      wgap_->update(1.0, *wearvector_, 1.0);

      // update all gap function entries for slave nodes!
      for (int i = 0; i < (int)interface_.size(); ++i)
      {
        for (int j = 0; j < (int)interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
        {
          int gid = interface_[i]->slave_row_nodes()->gid(j);
          Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
          if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

          if (cnode->fri_data().slip() == true)
            cnode->data().getg() += cnode->wear_data().weighted_wear();
        }
      }
    }
  }

  //********************************************
  // fill_complete() matrix for both-sided wear *
  //********************************************
  d2matrix_->complete(*gmdofrowmap_, *gmdofrowmap_);

  return;
}

/*----------------------------------------------------------------------*
 | initialize global contact variables for next Newton step  farah 09/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::initialize()
{
  CONTACT::LagrangeStrategy::initialize();

  // (re)setup global tangent matrix
  tmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivet_, 3);

  if (wearimpl_ and !wearprimvar_)
  {
    // create matrices for implicit wear: these matrices are due to
    // the gap-change in the compl. fnc.
    // Here are only the lin. w.r.t. the lagr. mult.
    wlinmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 3);
    wlinmatrixsl_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipt_, 3);

#ifdef CONSISTENTSTICK
    wlinmatrixst_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gstickt, 3);
#endif
  }

  if (wearprimvar_)
  {
    // steady state scenario
    if (sswear_)
    {
      // basic matrices
      twmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 100);  // gsdofnrowmap_
      ematrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 100);   // gsdofnrowmap_

      // linearizations w.r.t d and z
      lintdis_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gactiven_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      lintlm_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gactiven_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      linedis_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gactiven_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
    }

    // general scenario
    else
    {
      // basic matrices
      twmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipn_, 100);  // gsdofnrowmap_
      ematrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipn_, 100);   // gsdofnrowmap_

      // linearizations w.r.t d and z
      lintdis_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      lintlm_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      linedis_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
    }

    // linearizations w.r.t w
    smatrixW_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 3);  // gactiven_
    linslip_w_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipt_, 3);

    // w - rhs
    inactive_wear_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*gwinact_, true);

    if (sswear_)
      wear_cond_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*gactiven_, true);
    else
      wear_cond_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*gslipn_, true);

    // both-sided discr wear
    if (wearbothpv_)
    {
      // basic matrices
      twmatrix_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);  // gsdofnrowmap_
      ematrix_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);  // gsdofnrowmap_

      // linearizations w.r.t d and z
      lintdis_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      lintlm_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      linedis_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmslipn_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

      // w - rhs
      inactive_wear_rhs_m_ = std::make_shared<Core::LinAlg::FEVector<double>>(*gwminact_);
      wear_cond_rhs_m_ = std::make_shared<Core::LinAlg::FEVector<double>>(*gmslipn_);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | condense wear and lm. for impl/expl wear algorithm        farah 10/13|
 | Internal state variable approach!                                    |
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::condense_wear_impl_expl(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff, Core::LinAlg::Vector<double>& gact)
{
  auto shapefcn = Teuchos::getIntegralValue<Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  // double-check if this is a dual LM system
  if (shapefcn != Mortar::shape_dual && shapefcn != Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Condensation only for dual LM");

  // get stick map
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);

  /********************************************************************/
  /* (1) Multiply Mortar matrices: m^ = inv(d) * m                    */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
      std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  invd->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
    if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  err = invd->replace_diagonal_values(*diag);
  if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

  // do the multiplication mhat = inv(D) * M
  mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd, false, *mmatrix_, false, false, false, true);

  /********************************************************************/
  /* (2) Add contact stiffness terms to kteff                         */
  /********************************************************************/

  // transform if necessary
  if (parallel_redistribution_status())
  {
    lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
    linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
  }

  kteff->un_complete();
  kteff->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->complete();

  /********************************************************************/
  /* (3) Split kteff into 3x3 matrix blocks                           */
  /********************************************************************/

  // we want to split k into 3 groups s,m,n = 9 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss, ksm, ksn, kms, kmm, kmn, kns, knm, knn;

  // temporarily we need the blocks ksmsm, ksmn, knsm
  // (FIXME: because a direct SplitMatrix3x3 is still missing!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksmsm, ksmn, knsm;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;

  // split into slave/master part + structure part
  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);
  if (parallel_redistribution_status())
  {
    // split and transform to redistributed maps
    Core::LinAlg::split_matrix2x2(kteffmatrix, non_redist_gsmdofrowmap_, gndofrowmap_,
        non_redist_gsmdofrowmap_, gndofrowmap_, ksmsm, ksmn, knsm, knn);
    ksmsm = Core::LinAlg::matrix_row_col_transform(*ksmsm, *gsmdofrowmap_, *gsmdofrowmap_);
    ksmn = Core::LinAlg::matrix_row_transform(*ksmn, *gsmdofrowmap_);
    knsm = Core::LinAlg::matrix_col_transform(*knsm, *gsmdofrowmap_);
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_matrix2x2(kteffmatrix, gsmdofrowmap_, gndofrowmap_, gsmdofrowmap_,
        gndofrowmap_, ksmsm, ksmn, knsm, knn);
  }

  // further splits into slave part + master part
  Core::LinAlg::split_matrix2x2(
      ksmsm, gsdofrowmap_, gmdofrowmap_, gsdofrowmap_, gmdofrowmap_, kss, ksm, kms, kmm);
  Core::LinAlg::split_matrix2x2(
      ksmn, gsdofrowmap_, gmdofrowmap_, gndofrowmap_, tempmap, ksn, tempmtx1, kmn, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      knsm, gndofrowmap_, tempmap, gsdofrowmap_, gmdofrowmap_, kns, knm, tempmtx1, tempmtx2);

  /********************************************************************/
  /* (4) Split feff into 3 subvectors                                 */
  /********************************************************************/

  // we want to split f into 3 groups s.m,n
  std::shared_ptr<Core::LinAlg::Vector<double>> fs, fm, fn;

  // temporarily we need the group sm
  std::shared_ptr<Core::LinAlg::Vector<double>> fsm;

  // do the vector splitting smn -> sm+n
  if (parallel_redistribution_status())
  {
    // split and transform to redistributed maps
    Core::LinAlg::split_vector(
        *problem_dofs(), *feff, non_redist_gsmdofrowmap_, fsm, gndofrowmap_, fn);
    std::shared_ptr<Core::LinAlg::Vector<double>> fsmtemp =
        std::make_shared<Core::LinAlg::Vector<double>>(*gsmdofrowmap_);
    Core::LinAlg::export_to(*fsm, *fsmtemp);
    fsm = fsmtemp;
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_vector(*problem_dofs(), *feff, gsmdofrowmap_, fsm, gndofrowmap_, fn);
  }

  // abbreviations for slave and master set
  int sset = gsdofrowmap_->num_global_elements();
  int mset = gmdofrowmap_->num_global_elements();

  // we want to split fsm into 2 groups s,m
  fs = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  fm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);

  // do the vector splitting sm -> s+m
  Core::LinAlg::split_vector(*gsmdofrowmap_, *fsm, gsdofrowmap_, fs, gmdofrowmap_, fm);

  // store some stuff for static condensation of LM
  fs_ = fs;
  invd_ = invd;
  ksn_ = ksn;
  ksm_ = ksm;
  kss_ = kss;

  //--------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //--------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  // D^(-1)    ---->   T * D^(-1)
  // \hat{M}   ---->   T * \hat{M}
  //--------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify dmatrix_, invd_ and mhatmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp3 =
        Core::LinAlg::matrix_multiply(*trafo_, false, *invd_, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp4 =
        Core::LinAlg::matrix_multiply(*trafo_, false, *mhatmatrix_, false, false, false, true);
    dmatrix_ = temp2;
    invd_ = temp3;
    mhatmatrix_ = temp4;
  }

  /********************************************************************/
  /* (5) Split slave quantities into active / inactive, stick / slip  */
  /********************************************************************/
  // we want to split kssmod into 2 groups a,i = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kaa, kai, kia, kii;

  // we want to split ksn / ksm / kms into 2 groups a,i = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kan, kin, kam, kim, kma, kmi;

  // we will get the i rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> gidofs;

  // do the splitting
  Core::LinAlg::split_matrix2x2(
      kss, gactivedofs_, gidofs, gactivedofs_, gidofs, kaa, kai, kia, kii);
  Core::LinAlg::split_matrix2x2(
      ksn, gactivedofs_, gidofs, gndofrowmap_, tempmap, kan, tempmtx1, kin, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      ksm, gactivedofs_, gidofs, gmdofrowmap_, tempmap, kam, tempmtx1, kim, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      kms, gmdofrowmap_, tempmap, gactivedofs_, gidofs, kma, kmi, tempmtx1, tempmtx2);

  // we want to split kaa into 2 groups sl,st = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslsl, kslst, kstsl, kstst, kast, kasl;

  // we want to split kan / kam / kai into 2 groups sl,st = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksln, kstn, kslm, kstm, ksli, ksti;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> temp1map;
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp1mtx4, temp1mtx5;

  // we will get the stick rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> gstdofs;

  Core::LinAlg::split_matrix2x2(
      kaa, gactivedofs_, gidofs, gstdofs, gslipdofs_, kast, kasl, temp1mtx4, temp1mtx5);

  // abbreviations for active and inactive set, stick and slip set
  const int aset = gactivedofs_->num_global_elements();
  const int iset = gidofs->num_global_elements();
  const int stickset = gstdofs->num_global_elements();
  const int slipset = gslipdofs_->num_global_elements();

  // we want to split fs into 2 groups a,i
  std::shared_ptr<Core::LinAlg::Vector<double>> fa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fi =
      std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

  // do the vector splitting s -> a+i
  Core::LinAlg::split_vector(*gsdofrowmap_, *fs, gactivedofs_, fa, gidofs, fi);

  // we want to split fa into 2 groups sl,st
  std::shared_ptr<Core::LinAlg::Vector<double>> fsl =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fst =
      std::make_shared<Core::LinAlg::Vector<double>>(*gstdofs);

  // do the vector splitting a -> sl+st
  if (aset) Core::LinAlg::split_vector(*gactivedofs_, *fa, gslipdofs_, fsl, gstdofs, fst);

  /********************************************************************/
  /* (6) Isolate necessary parts from invd and mhatmatrix             */
  /********************************************************************/

  // active, stick and slip part of invd
  std::shared_ptr<Core::LinAlg::SparseMatrix> invda, invdsl, invdst;
  Core::LinAlg::split_matrix2x2(
      invd_, gactivedofs_, gidofs, gactivedofs_, gidofs, invda, tempmtx1, tempmtx2, tempmtx3);
  Core::LinAlg::split_matrix2x2(
      invda, gactivedofs_, gidofs, gslipdofs_, gstdofs, invdsl, tempmtx1, tempmtx2, tempmtx3);
  Core::LinAlg::split_matrix2x2(
      invda, gactivedofs_, gidofs, gstdofs, gslipdofs_, invdst, tempmtx1, tempmtx2, tempmtx3);

  // for Implicit Wear
  std::shared_ptr<Core::LinAlg::SparseMatrix> wa, wi, wsl, wst;  // split lin.matrix for Cgap
  std::shared_ptr<Core::LinAlg::SparseMatrix> wsla, wsli, wslsl,
      wslst;  // split lin.matrix for Cslip
  Core::LinAlg::Vector<double> fw(*gactiven_);
  Core::LinAlg::Vector<double> fwsl(*gslipt_);

  // implicit wear
  if (wearimpl_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> za =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
    std::shared_ptr<Core::LinAlg::Vector<double>> zi =
        std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

    Core::LinAlg::split_vector(*gsdofrowmap_, *z_, gactivedofs_, za, gidofs, zi);

    // split wlinmatrix into wa for active dofs
    Core::LinAlg::split_matrix2x2(
        wlinmatrix_, gactiven_, gactiven_, gactivedofs_, gidofs, wa, tempmtx1, tempmtx2, tempmtx3);

    // split wlinmatrixsl into wa for active dofs
    Core::LinAlg::split_matrix2x2(
        wlinmatrixsl_, gslipt_, gslipt_, gactivedofs_, gidofs, wsla, tempmtx1, tempmtx2, tempmtx3);

    // WEAR LIN. W.R.T. Z *****************************************************
    // for normal contact
    if (aset) wa->multiply(false, *za, fw);

    // WEAR LIN. W.R.T. Z *****************************************************
    // for slip contact
    if (slipset) wsla->multiply(false, *za, fwsl);
  }

  // coupling part of dmatrix (only nonzero for 3D quadratic case!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> dai;
  Core::LinAlg::split_matrix2x2(
      dmatrix_, gactivedofs_, gidofs, gactivedofs_, gidofs, tempmtx1, dai, tempmtx2, tempmtx3);

  // do the multiplication dhat = invda * dai
  std::shared_ptr<Core::LinAlg::SparseMatrix> dhat =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 10);
  if (aset && iset)
    dhat = Core::LinAlg::matrix_multiply(*invda, false, *dai, false, false, false, true);
  dhat->complete(*gidofs, *gactivedofs_);

  // active part of mmatrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrixa;
  Core::LinAlg::split_matrix2x2(mmatrix_, gactivedofs_, gidofs, gmdofrowmap_, tempmap, mmatrixa,
      tempmtx1, tempmtx2, tempmtx3);

  // do the multiplication mhataam = invda * mmatrixa
  // (this is only different from mhata for 3D quadratic case!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> mhataam =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 10);
  if (aset)
    mhataam = Core::LinAlg::matrix_multiply(*invda, false, *mmatrixa, false, false, false, true);
  mhataam->complete(*gmdofrowmap_, *gactivedofs_);

  // for the case without full linearization, we still need the
  // "classical" active part of mhat, which is isolated here
  std::shared_ptr<Core::LinAlg::SparseMatrix> mhata;
  Core::LinAlg::split_matrix2x2(mhatmatrix_, gactivedofs_, gidofs, gmdofrowmap_, tempmap, mhata,
      tempmtx1, tempmtx2, tempmtx3);

  // scaling of invd and dai
  invda->scale(1 / (1 - alphaf_));
  invdsl->scale(1 / (1 - alphaf_));
  invdst->scale(1 / (1 - alphaf_));
  dai->scale(1 - alphaf_);

  /********************************************************************/
  /* (7) Build the final K blocks                                     */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // knn: nothing to do

  // knm: nothing to do

  // kns: nothing to do

  //-------------------------------------------------------- SECOND LINE
  // kmn: add T(mhataam)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(*kmn, false, 1.0, *kmnmod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
  Core::LinAlg::matrix_add(*kmnadd, false, 1.0, *kmnmod, 1.0);
  kmnmod->complete(kmn->domain_map(), kmn->row_map());

  // kmm: add T(mhataam)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(*kmm, false, 1.0, *kmmmod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
  Core::LinAlg::matrix_add(*kmmadd, false, 1.0, *kmmmod, 1.0);
  kmmmod->complete(kmm->domain_map(), kmm->row_map());

  // kmi: add T(mhataam)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
  if (iset)
  {
    kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*kmi, false, 1.0, *kmimod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
    Core::LinAlg::matrix_add(*kmiadd, false, 1.0, *kmimod, 1.0);
    kmimod->complete(kmi->domain_map(), kmi->row_map());
  }

  // kma: add T(mhataam)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
  if (aset)
  {
    kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*kma, false, 1.0, *kmamod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
    Core::LinAlg::matrix_add(*kmaadd, false, 1.0, *kmamod, 1.0);
    kmamod->complete(kma->domain_map(), kma->row_map());
  }

  //--------------------------------------------------------- THIRD LINE
  // kin: subtract T(dhat)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  Core::LinAlg::matrix_add(*kin, false, 1.0, *kinmod, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
    Core::LinAlg::matrix_add(*kinadd, false, -1.0, *kinmod, 1.0);
  }
  kinmod->complete(kin->domain_map(), kin->row_map());

  // kim: subtract T(dhat)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  Core::LinAlg::matrix_add(*kim, false, 1.0, *kimmod, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
    Core::LinAlg::matrix_add(*kimadd, false, -1.0, *kimmod, 1.0);
  }
  kimmod->complete(kim->domain_map(), kim->row_map());

  // kii: subtract T(dhat)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
  if (iset)
  {
    kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    Core::LinAlg::matrix_add(*kii, false, 1.0, *kiimod, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
      Core::LinAlg::matrix_add(*kiiadd, false, -1.0, *kiimod, 1.0);
    }
    kiimod->complete(kii->domain_map(), kii->row_map());
  }

  // kia: subtract T(dhat)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
  if (iset && aset)
  {
    kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    Core::LinAlg::matrix_add(*kia, false, 1.0, *kiamod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
    Core::LinAlg::matrix_add(*kiaadd, false, -1.0, *kiamod, 1.0);
    kiamod->complete(kia->domain_map(), kia->row_map());
  }

  //-------------------------------------------------------- FOURTH LINE

  // create matrices for implicit wear
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgnmod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgmmod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgimod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgaamod;

  // implicit wear
  if (wearimpl_)
  {
    if (aset)
    {
      kgnmod = Core::LinAlg::matrix_multiply(*wa, false, *invda, true, false, false, true);
      kgnmod = Core::LinAlg::matrix_multiply(*kgnmod, false, *kan, false, false, false, true);
    }

    if (aset)
    {
      kgmmod = Core::LinAlg::matrix_multiply(*wa, false, *invda, true, false, false, true);
      kgmmod = Core::LinAlg::matrix_multiply(*kgmmod, false, *kam, false, false, false, true);
    }

    if (aset and iset)
    {
      kgimod = Core::LinAlg::matrix_multiply(*wa, false, *invda, true, false, false, true);
      kgimod = Core::LinAlg::matrix_multiply(*kgimod, false, *kai, false, false, false, true);
    }

    if (aset)
    {
      kgaamod = Core::LinAlg::matrix_multiply(*wa, false, *invda, true, false, false, true);
      kgaamod = Core::LinAlg::matrix_multiply(*kgaamod, false, *kaa, false, false, false, true);
    }
  }
  //--------------------------------------------------------- FIFTH LINE
  // blocks for complementary conditions (stick nodes)

  // kstn: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstnmod;
  if (stickset)
  {
    kstnmod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstnmod = Core::LinAlg::matrix_multiply(*kstnmod, false, *kan, false, false, false, true);
  }

  // kstm: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstmmod;
  if (stickset)
  {
    kstmmod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstmmod = Core::LinAlg::matrix_multiply(*kstmmod, false, *kam, false, false, false, true);
  }

  // ksti: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstimod;
  if (stickset && iset)
  {
    kstimod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstimod = Core::LinAlg::matrix_multiply(*kstimod, false, *kai, false, false, false, true);
  }

  // kstsl: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstslmod;
  if (stickset && slipset)
  {
    kstslmod =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstslmod = Core::LinAlg::matrix_multiply(*kstslmod, false, *kasl, false, false, false, true);
  }

  // kststmod: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kststmod;
  if (stickset)
  {
    kststmod =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kststmod = Core::LinAlg::matrix_multiply(*kststmod, false, *kast, false, false, false, true);
  }

  //--------------------------------------------------------- SIXTH LINE
  // blocks for complementary conditions (slip nodes)

  // ksln: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslnmod;
  if (slipset)
  {
    kslnmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslnmod = Core::LinAlg::matrix_multiply(*kslnmod, false, *kan, false, false, false, true);
  }

  // kslm: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslmmod;
  if (slipset)
  {
    kslmmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslmmod = Core::LinAlg::matrix_multiply(*kslmmod, false, *kam, false, false, false, true);
  }

  // ksli: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslimod;
  if (slipset && iset)
  {
    kslimod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslimod = Core::LinAlg::matrix_multiply(*kslimod, false, *kai, false, false, false, true);
  }

  // kslsl: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslslmod;
  if (slipset)
  {
    kslslmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslslmod = Core::LinAlg::matrix_multiply(*kslslmod, false, *kasl, false, false, false, true);
  }

  // slstmod: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslstmod;
  if (slipset && stickset)
  {
    kslstmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslstmod = Core::LinAlg::matrix_multiply(*kslstmod, false, *kast, false, false, false, true);
  }

  // create matrices for implicit wear
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslwnmod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslwmmod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslwimod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslwaamod;

  // implicit wear
  if (wearimpl_)
  {
    if (slipset)
    {
      kslwnmod = Core::LinAlg::matrix_multiply(*wsla, false, *invda, true, false, false, true);
      kslwnmod = Core::LinAlg::matrix_multiply(*kslwnmod, false, *kan, false, false, false, true);
    }

    if (slipset)
    {
      kslwmmod = Core::LinAlg::matrix_multiply(*wsla, false, *invda, true, false, false, true);
      kslwmmod = Core::LinAlg::matrix_multiply(*kslwmmod, false, *kam, false, false, false, true);
    }

    if (slipset and iset)
    {
      kslwimod = Core::LinAlg::matrix_multiply(*wsla, false, *invda, true, false, false, true);
      kslwimod = Core::LinAlg::matrix_multiply(*kslwimod, false, *kai, false, false, false, true);
    }

    if (slipset)
    {
      kslwaamod = Core::LinAlg::matrix_multiply(*wsla, false, *invda, true, false, false, true);
      kslwaamod = Core::LinAlg::matrix_multiply(*kslwaamod, false, *kaa, false, false, false, true);
    }
  }

  /********************************************************************/
  /* (8) Build the final f blocks                                     */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // fn: nothing to do

  //---------------------------------------------------------- SECOND LINE
  // fm: add alphaf * old contact forces (t_n)
  // for self contact, slave and master sets may have changed,
  // thus we have to export the product Mold^T * zold to fit
  if (is_self_contact())
  {
    Core::LinAlg::Vector<double> tempvecm(*gmdofrowmap_);
    Core::LinAlg::Vector<double> tempvecm2(mold_->domain_map());
    Core::LinAlg::Vector<double> zoldexp(mold_->row_map());
    if (mold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
    mold_->multiply(true, zoldexp, tempvecm2);
    if (mset) Core::LinAlg::export_to(tempvecm2, tempvecm);
    fm->update(alphaf_, tempvecm, 1.0);
  }
  // if there is no self contact everything is ok
  else
  {
    Core::LinAlg::Vector<double> tempvecm(*gmdofrowmap_);
    mold_->multiply(true, *zold_, tempvecm);
    fm->update(alphaf_, tempvecm, 1.0);
  }

  // fs: prepare alphaf * old contact forces (t_n)
  Core::LinAlg::Vector<double> fsadd(*gsdofrowmap_);

  // for self contact, slave and master sets may have changed,
  // thus we have to export the product Dold^T * zold to fit
  if (is_self_contact())
  {
    Core::LinAlg::Vector<double> tempvec(dold_->domain_map());
    Core::LinAlg::Vector<double> zoldexp(dold_->row_map());
    if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
    dold_->multiply(true, zoldexp, tempvec);
    if (sset) Core::LinAlg::export_to(tempvec, fsadd);
  }
  // if there is no self contact everything is ok
  else
  {
    dold_->multiply(true, *zold_, fsadd);
  }

  // fa: subtract alphaf * old contact forces (t_n)
  if (aset)
  {
    Core::LinAlg::Vector<double> faadd(*gactivedofs_);
    Core::LinAlg::export_to(fsadd, faadd);
    fa->update(-alphaf_, faadd, 1.0);
  }

  // fm: add T(mhat)*fa
  Core::LinAlg::Vector<double> fmmod(*gmdofrowmap_);
  if (aset) mhataam->multiply(true, *fa, fmmod);
  fmmod.update(1.0, *fm, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // fi: subtract alphaf * old contact forces (t_n)
  if (iset)
  {
    Core::LinAlg::Vector<double> fiadd(*gidofs);
    Core::LinAlg::export_to(fsadd, fiadd);
    fi->update(-alphaf_, fiadd, 1.0);
  }

  // fi: add T(dhat)*fa
  Core::LinAlg::Vector<double> fimod(*gidofs);
  if (aset && iset) dhat->multiply(true, *fa, fimod);
  fimod.update(1.0, *fi, -1.0);

  //-------------------------------------------------------- FOURTH LINE
  std::shared_ptr<Core::LinAlg::Vector<double>> fgmod;
  // implicit wear
  if (wearimpl_)
  {
    if (aset)
    {
      fgmod = std::make_shared<Core::LinAlg::Vector<double>>(*gactiven_);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*wa, false, *invda, true, false, false, true);
      temp1->multiply(false, *fa, *fgmod);
    }
  }
  //--------------------------------------------------------- FIFTH LINE
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);  // get global stick dofs

  // split the lagrange multiplier vector in stick and slip part
  std::shared_ptr<Core::LinAlg::Vector<double>> za =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> zi =
      std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);
  std::shared_ptr<Core::LinAlg::Vector<double>> zst =
      std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
  std::shared_ptr<Core::LinAlg::Vector<double>> zsl =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);

  Core::LinAlg::split_vector(*gsdofrowmap_, *z_, gactivedofs_, za, gidofs, zi);
  Core::LinAlg::split_vector(*gactivedofs_, *za, gstickdofs, zst, gslipdofs_, zsl);
  std::shared_ptr<Core::LinAlg::Vector<double>> tempvec1;

  // fst: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::Vector<double>> fstmod;
  if (stickset)
  {
    fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    temp1->multiply(false, *fa, *fstmod);

    tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);

    linstickLM_->multiply(false, *zst, *tempvec1);
    fstmod->update(-1.0, *tempvec1, 1.0);
  }

  //--------------------------------------------------------- SIXTH LINE
  // fsl: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::Vector<double>> fslmod;
  std::shared_ptr<Core::LinAlg::Vector<double>> fslwmod;

  if (slipset)
  {
    fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
        Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    temp->multiply(false, *fa, *fslmod);

    tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);

    linslipLM_->multiply(false, *zsl, *tempvec1);

    fslmod->update(-1.0, *tempvec1, 1.0);

    // implicit wear
    if (wearimpl_)
    {
      fslwmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
          Core::LinAlg::matrix_multiply(*wsla, false, *invda, true, false, false, true);
      temp2->multiply(false, *fa, *fslwmod);
    }
  }

  /********************************************************************/
  /* (9) Transform the final K blocks                                 */
  /********************************************************************/
  // The row maps of all individual matrix blocks are transformed to
  // the parallel layout of the underlying problem discretization.
  // Of course, this is only necessary in the parallel redistribution
  // case, where the contact interfaces have been redistributed
  // independently of the underlying problem discretization.

  if (parallel_redistribution_status())
  {
    //----------------------------------------------------------- FIRST LINE
    // nothing to do (ndof-map independent of redistribution)

    //---------------------------------------------------------- SECOND LINE
    kmnmod = Core::LinAlg::matrix_row_transform(*kmnmod, *non_redist_gmdofrowmap_);
    kmmmod = Core::LinAlg::matrix_row_transform(*kmmmod, *non_redist_gmdofrowmap_);
    if (iset) kmimod = Core::LinAlg::matrix_row_transform(*kmimod, *non_redist_gmdofrowmap_);
    if (aset) kmamod = Core::LinAlg::matrix_row_transform(*kmamod, *non_redist_gmdofrowmap_);

    //----------------------------------------------------------- THIRD LINE
    if (iset)
    {
      kinmod = Core::LinAlg::matrix_row_transform(*kinmod, *non_redist_gsdofrowmap_);
      kimmod = Core::LinAlg::matrix_row_transform(*kimmod, *non_redist_gsdofrowmap_);
      kiimod = Core::LinAlg::matrix_row_transform(*kiimod, *non_redist_gsdofrowmap_);
      if (aset) kiamod = Core::LinAlg::matrix_row_transform(*kiamod, *non_redist_gsdofrowmap_);
    }

    //---------------------------------------------------------- FOURTH LINE
    if (aset)
    {
      smatrix_ = Core::LinAlg::matrix_row_transform(*smatrix_, *non_redist_gsdofrowmap_);

      // implicit wear
      if (wearimpl_)
      {
        kgnmod = Core::LinAlg::matrix_row_transform(*kgnmod, *non_redist_gsdofrowmap_);
        kgmmod = Core::LinAlg::matrix_row_transform(*kgmmod, *non_redist_gsdofrowmap_);
        if (iset) kgimod = Core::LinAlg::matrix_row_transform(*kgimod, *non_redist_gsdofrowmap_);
        kgaamod = Core::LinAlg::matrix_row_transform(*kgaamod, *non_redist_gsdofrowmap_);
      }
    }

    //----------------------------------------------------------- FIFTH LINE
    if (stickset)
    {
      kstnmod = Core::LinAlg::matrix_row_transform(*kstnmod, *non_redist_gsdofrowmap_);
      kstmmod = Core::LinAlg::matrix_row_transform(*kstmmod, *non_redist_gsdofrowmap_);
      if (iset) kstimod = Core::LinAlg::matrix_row_transform(*kstimod, *non_redist_gsdofrowmap_);
      if (slipset)
        kstslmod = Core::LinAlg::matrix_row_transform(*kstslmod, *non_redist_gsdofrowmap_);
      kststmod = Core::LinAlg::matrix_row_transform(*kststmod, *non_redist_gsdofrowmap_);
      linstickDIS_ = Core::LinAlg::matrix_row_transform(*linstickDIS_, *non_redist_gsdofrowmap_);
    }

    //----------------------------------------------------------- SIXTH LINE
    if (slipset)
    {
      kslnmod = Core::LinAlg::matrix_row_transform(*kslnmod, *non_redist_gsdofrowmap_);
      kslmmod = Core::LinAlg::matrix_row_transform(*kslmmod, *non_redist_gsdofrowmap_);
      if (iset) kslimod = Core::LinAlg::matrix_row_transform(*kslimod, *non_redist_gsdofrowmap_);
      if (stickset)
        kslstmod = Core::LinAlg::matrix_row_transform(*kslstmod, *non_redist_gsdofrowmap_);
      kslslmod = Core::LinAlg::matrix_row_transform(*kslslmod, *non_redist_gsdofrowmap_);
      linslipDIS_ = Core::LinAlg::matrix_row_transform(*linslipDIS_, *non_redist_gsdofrowmap_);

      // implicit wear
      if (wearimpl_)
      {
        kslwnmod = Core::LinAlg::matrix_row_transform(*kslwnmod, *non_redist_gsdofrowmap_);
        kslwmmod = Core::LinAlg::matrix_row_transform(*kslwmmod, *non_redist_gsdofrowmap_);
        if (iset)
          kslwimod = Core::LinAlg::matrix_row_transform(*kslwimod, *non_redist_gsdofrowmap_);
        kslwaamod = Core::LinAlg::matrix_row_transform(*kslwaamod, *non_redist_gsdofrowmap_);
      }
    }
  }

  /********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                */
  /********************************************************************/

  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
  std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());

  //--------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  Core::LinAlg::matrix_add(*knn, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*knm, false, 1.0, *kteffnew, 1.0);
  if (sset) Core::LinAlg::matrix_add(*kns, false, 1.0, *kteffnew, 1.0);

  //-------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  Core::LinAlg::matrix_add(*kmnmod, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*kmmmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kmimod, false, 1.0, *kteffnew, 1.0);
  if (aset) Core::LinAlg::matrix_add(*kmamod, false, 1.0, *kteffnew, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) Core::LinAlg::matrix_add(*kinmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kimmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kiimod, false, 1.0, *kteffnew, 1.0);
  if (iset && aset) Core::LinAlg::matrix_add(*kiamod, false, 1.0, *kteffnew, 1.0);

  //-------------------------------------------------------- FOURTH LINE

  // add a submatrices to kteffnew
  if (aset) Core::LinAlg::matrix_add(*smatrix_, false, 1.0, *kteffnew, 1.0);

  // implicit wear
  if (wearimpl_)
  {
    // if (aset) kteffnew->Add(*wlinmatrix_,false,1.0,1.0);
    if (aset) Core::LinAlg::matrix_add(*kgnmod, false, -1.0, *kteffnew, 1.0);
    if (aset) Core::LinAlg::matrix_add(*kgmmod, false, -1.0, *kteffnew, 1.0);
    if (aset and iset) Core::LinAlg::matrix_add(*kgimod, false, -1.0, *kteffnew, 1.0);
    if (aset) Core::LinAlg::matrix_add(*kgaamod, false, -1.0, *kteffnew, 1.0);
    // if (aset and stickset) kteffnew->Add(*kgstmod,false,1.0,1.0);
  }

  //--------------------------------------------------------- FIFTH LINE
  // add st submatrices to kteffnew
  if (stickset) Core::LinAlg::matrix_add(*kstnmod, false, 1.0, *kteffnew, 1.0);
  if (stickset) Core::LinAlg::matrix_add(*kstmmod, false, 1.0, *kteffnew, 1.0);
  if (stickset && iset) Core::LinAlg::matrix_add(*kstimod, false, 1.0, *kteffnew, 1.0);
  if (stickset && slipset) Core::LinAlg::matrix_add(*kstslmod, false, 1.0, *kteffnew, 1.0);
  if (stickset) Core::LinAlg::matrix_add(*kststmod, false, 1.0, *kteffnew, 1.0);

  // add terms of linearization of sick condition to kteffnew
  if (stickset) Core::LinAlg::matrix_add(*linstickDIS_, false, -1.0, *kteffnew, 1.0);

  //--------------------------------------------------------- SIXTH LINE
  // add sl submatrices to kteffnew
  if (slipset) Core::LinAlg::matrix_add(*kslnmod, false, 1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslmmod, false, 1.0, *kteffnew, 1.0);
  if (slipset && iset) Core::LinAlg::matrix_add(*kslimod, false, 1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslslmod, false, 1.0, *kteffnew, 1.0);
  if (slipset && stickset) Core::LinAlg::matrix_add(*kslstmod, false, 1.0, *kteffnew, 1.0);

  // add terms of linearization of slip condition to kteffnew and feffnew
  if (slipset) Core::LinAlg::matrix_add(*linslipDIS_, false, -1.0, *kteffnew, +1.0);

  // implicit wear
  if (wearimpl_)
  {
    // if (slipset) kteffnew->Add(*wlinmatrixsl_,false,1.0,1.0);
    if (slipset) Core::LinAlg::matrix_add(*kslwnmod, false, 1.0, *kteffnew, 1.0);
    if (slipset) Core::LinAlg::matrix_add(*kslwmmod, false, 1.0, *kteffnew, 1.0);
    if (slipset and iset) Core::LinAlg::matrix_add(*kslwimod, false, 1.0, *kteffnew, 1.0);
    if (slipset) Core::LinAlg::matrix_add(*kslwaamod, false, 1.0, *kteffnew, 1.0);
    // if (slipset and stickset) kteffnew->Add(*kslwstmod,false,1.0,1.0);
  }

  // fill_complete kteffnew (square)
  kteffnew->complete();

  /********************************************************************/
  /* (11) Global setup of feffnew (including contact)                 */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // add n subvector to feffnew
  Core::LinAlg::Vector<double> fnexp(*problem_dofs());
  Core::LinAlg::export_to(*fn, fnexp);
  feffnew->update(1.0, fnexp, 1.0);

  //-------------------------------------------------------- SECOND LINE
  // add m subvector to feffnew
  Core::LinAlg::Vector<double> fmmodexp(*problem_dofs());
  Core::LinAlg::export_to(fmmod, fmmodexp);
  feffnew->update(1.0, fmmodexp, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // add i subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fimodexp;
  if (iset)
  {
    fimodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fimod, *fimodexp);
    feffnew->update(1.0, *fimodexp, 1.0);
  }

  //-------------------------------------------------------- FOURTH LINE
  // add weighted gap vector to feffnew, if existing
  std::shared_ptr<Core::LinAlg::Vector<double>> gexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fgmodexp;

  if (aset)
  {
    gexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(gact, *gexp);
    feffnew->update(-1.0, *gexp, 1.0);

    // implicit wear
    if (wearimpl_)
    {
      // commented due to incremental solution algorithm.
      fwexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(fw, *fwexp);
      feffnew->update(+1.0, *fwexp, 1.0);

      fgmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(*fgmod, *fgmodexp);
      feffnew->update(-1.0, *fgmodexp, +1.0);
    }
  }

  //--------------------------------------------------------- FIFTH LINE
  // add st subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fstmodexp;
  if (stickset)
  {
    fstmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*fstmod, *fstmodexp);
    feffnew->update(1.0, *fstmodexp, +1.0);
  }

  // add terms of linearization feffnew
  if (stickset)
  {
    Core::LinAlg::Vector<double> linstickRHSexp(*problem_dofs());
    Core::LinAlg::export_to(*linstickRHS_, linstickRHSexp);
    feffnew->update(-1.0, linstickRHSexp, 1.0);
  }

  //--------------------------------------------------------- SIXTH LINE

  // add a subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fslmodexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwslexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fslwmodexp;


  if (slipset)
  {
    fslmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*fslmod, *fslmodexp);
    feffnew->update(1.0, *fslmodexp, 1.0);
  }

  // implicit wear
  if (wearimpl_)
  {
    if (slipset)
    {
      fslwmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(*fslwmod, *fslwmodexp);
      feffnew->update(+1.0, *fslwmodexp, 1.0);
    }
    // commented due to incremental solution algorithm
    if (slipset)
    {
      fwslexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(fwsl, *fwslexp);
      feffnew->update(-1.0, *fwslexp, 1.0);
    }
  }

  if (slipset)
  {
    Core::LinAlg::Vector<double> linslipRHSexp(*problem_dofs());
    Core::LinAlg::export_to(*linslipRHS_, linslipRHSexp);
    feffnew->update(-1.0, linslipRHSexp, 1.0);
  }

  // finally do the replacement
  kteff = kteffnew;
  feff = feffnew;

  return;
}

/*----------------------------------------------------------------------*
 | condense wear and lm. for discr. wear algorithm           farah 10/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::condense_wear_discr(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff, Core::LinAlg::Vector<double>& gact)
{
  auto shapefcn = Teuchos::getIntegralValue<Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto wearshapefcn = Teuchos::getIntegralValue<Inpar::Wear::WearShape>(params(), "WEAR_SHAPEFCN");

  // double-check if this is a dual LM system
  if (shapefcn != Mortar::shape_dual && shapefcn != Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Condensation only for dual LM");
  if (wearshapefcn != Inpar::Wear::wear_shape_dual) FOUR_C_THROW("Condensation only for dual wear");

  // get stick map
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);

  /********************************************************************/
  /* (1a) Multiply Mortar matrices: m^ = inv(d) * m                    */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
      std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  invd->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
    if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  err = invd->replace_diagonal_values(*diag);
  if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

  // do the multiplication mhat = inv(D) * M
  mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd, false, *mmatrix_, false, false, false, true);

  /********************************************************************/
  /* (2) Add contact stiffness terms to kteff                         */
  /********************************************************************/
  // transform if necessary
  if (parallel_redistribution_status())
  {
    lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
    linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
  }

  kteff->un_complete();
  kteff->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff->complete();

  /********************************************************************/
  /* (3a) Split kteff into 3x3 matrix blocks                          */
  /********************************************************************/

  // we want to split k into 3 groups s,m,n = 9 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss, ksm, ksn, kms, kmm, kmn, kns, knm, knn;

  // temporarily we need the blocks ksmsm, ksmn, knsm
  // (FIXME: because a direct SplitMatrix3x3 is still missing!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksmsm, ksmn, knsm;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;

  // split into slave/master part + structure part
  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);
  if (parallel_redistribution_status())
  {
    // split and transform to redistributed maps
    Core::LinAlg::split_matrix2x2(kteffmatrix, non_redist_gsmdofrowmap_, gndofrowmap_,
        non_redist_gsmdofrowmap_, gndofrowmap_, ksmsm, ksmn, knsm, knn);
    ksmsm = Core::LinAlg::matrix_row_col_transform(*ksmsm, *gsmdofrowmap_, *gsmdofrowmap_);
    ksmn = Core::LinAlg::matrix_row_transform(*ksmn, *gsmdofrowmap_);
    knsm = Core::LinAlg::matrix_col_transform(*knsm, *gsmdofrowmap_);
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_matrix2x2(kteffmatrix, gsmdofrowmap_, gndofrowmap_, gsmdofrowmap_,
        gndofrowmap_, ksmsm, ksmn, knsm, knn);
  }

  // further splits into slave part + master part
  Core::LinAlg::split_matrix2x2(
      ksmsm, gsdofrowmap_, gmdofrowmap_, gsdofrowmap_, gmdofrowmap_, kss, ksm, kms, kmm);
  Core::LinAlg::split_matrix2x2(
      ksmn, gsdofrowmap_, gmdofrowmap_, gndofrowmap_, tempmap, ksn, tempmtx1, kmn, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      knsm, gndofrowmap_, tempmap, gsdofrowmap_, gmdofrowmap_, kns, knm, tempmtx1, tempmtx2);

  /********************************************************************/
  /* (4) Split feff into 3 subvectors                                 */
  /********************************************************************/

  // we want to split f into 3 groups s.m,n
  std::shared_ptr<Core::LinAlg::Vector<double>> fs, fm, fn;

  // temporarily we need the group sm
  std::shared_ptr<Core::LinAlg::Vector<double>> fsm;

  // do the vector splitting smn -> sm+n
  if (parallel_redistribution_status())
  {
    // split and transform to redistributed maps
    Core::LinAlg::split_vector(
        *problem_dofs(), *feff, non_redist_gsmdofrowmap_, fsm, gndofrowmap_, fn);
    std::shared_ptr<Core::LinAlg::Vector<double>> fsmtemp =
        std::make_shared<Core::LinAlg::Vector<double>>(*gsmdofrowmap_);
    Core::LinAlg::export_to(*fsm, *fsmtemp);
    fsm = fsmtemp;
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_vector(*problem_dofs(), *feff, gsmdofrowmap_, fsm, gndofrowmap_, fn);
  }

  // abbreviations for slave and master set
  const int sset = gsdofrowmap_->num_global_elements();
  const int mset = gmdofrowmap_->num_global_elements();

  // we want to split fsm into 2 groups s,m
  fs = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  fm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);

  // do the vector splitting sm -> s+m
  Core::LinAlg::split_vector(*gsmdofrowmap_, *fsm, gsdofrowmap_, fs, gmdofrowmap_, fm);

  // store some stuff for static condensation of LM
  fs_ = fs;
  invd_ = invd;
  ksn_ = ksn;
  ksm_ = ksm;
  kss_ = kss;

  //--------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //--------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  // D^(-1)    ---->   T * D^(-1)
  // \hat{M}   ---->   T * \hat{M}
  //--------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify dmatrix_, invd_ and mhatmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp3 =
        Core::LinAlg::matrix_multiply(*trafo_, false, *invd_, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp4 =
        Core::LinAlg::matrix_multiply(*trafo_, false, *mhatmatrix_, false, false, false, true);
    dmatrix_ = temp2;
    invd_ = temp3;
    mhatmatrix_ = temp4;
  }

  /********************************************************************/
  /* (5a) Split slave quantities into active / inactive, stick / slip  */
  /********************************************************************/
  // we want to split kssmod into 2 groups a,i = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kaa, kai, kia, kii;

  // we want to split ksn / ksm / kms into 2 groups a,i = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kan, kin, kam, kim, kma, kmi;

  // we will get the i rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> gidofs;

  // do the splitting
  Core::LinAlg::split_matrix2x2(
      kss, gactivedofs_, gidofs, gactivedofs_, gidofs, kaa, kai, kia, kii);
  Core::LinAlg::split_matrix2x2(
      ksn, gactivedofs_, gidofs, gndofrowmap_, tempmap, kan, tempmtx1, kin, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      ksm, gactivedofs_, gidofs, gmdofrowmap_, tempmap, kam, tempmtx1, kim, tempmtx2);
  Core::LinAlg::split_matrix2x2(
      kms, gmdofrowmap_, tempmap, gactivedofs_, gidofs, kma, kmi, tempmtx1, tempmtx2);

  // we want to split kaa into 2 groups sl,st = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslsl, kslst, kstsl, kstst, kast, kasl;

  // we want to split kan / kam / kai into 2 groups sl,st = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksln, kstn, kslm, kstm, ksli, ksti;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> temp1map;
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp1mtx4, temp1mtx5;

  // we will get the stick rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> gstdofs;

  Core::LinAlg::split_matrix2x2(
      kaa, gactivedofs_, gidofs, gstdofs, gslipdofs_, kast, kasl, temp1mtx4, temp1mtx5);

  // abbreviations for active and inactive set, stick and slip set
  const int aset = gactivedofs_->num_global_elements();
  const int iset = gidofs->num_global_elements();
  const int stickset = gstdofs->num_global_elements();
  const int slipset = gslipdofs_->num_global_elements();
  gidofs_ = gidofs;

  // we want to split fs into 2 groups a,i
  std::shared_ptr<Core::LinAlg::Vector<double>> fa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fi =
      std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

  // do the vector splitting s -> a+i
  Core::LinAlg::split_vector(*gsdofrowmap_, *fs, gactivedofs_, fa, gidofs, fi);

  // we want to split fa into 2 groups sl,st
  std::shared_ptr<Core::LinAlg::Vector<double>> fsl =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fst =
      std::make_shared<Core::LinAlg::Vector<double>>(*gstdofs);

  // do the vector splitting a -> sl+st
  if (aset) Core::LinAlg::split_vector(*gactivedofs_, *fa, gslipdofs_, fsl, gstdofs, fst);

  /********************************************************************/
  /* (6) Isolate necessary parts from invd and mhatmatrix             */
  /********************************************************************/

  // active, stick and slip part of invd
  std::shared_ptr<Core::LinAlg::SparseMatrix> invda, invdsl, invdst;
  Core::LinAlg::split_matrix2x2(
      invd_, gactivedofs_, gidofs, gactivedofs_, gidofs, invda, tempmtx1, tempmtx2, tempmtx3);
  Core::LinAlg::split_matrix2x2(
      invda, gactivedofs_, gidofs, gslipdofs_, gstdofs, invdsl, tempmtx1, tempmtx2, tempmtx3);
  Core::LinAlg::split_matrix2x2(
      invda, gactivedofs_, gidofs, gstdofs, gslipdofs_, invdst, tempmtx1, tempmtx2, tempmtx3);

  // for Implicit Wear
  std::shared_ptr<Core::LinAlg::SparseMatrix> wa, wi, wsl, wst;  // split lin.matrix for Cgap
  std::shared_ptr<Core::LinAlg::SparseMatrix> wsla, wsli, wslsl,
      wslst;  // split lin.matrix for Cslip

  // coupling part of dmatrix (only nonzero for 3D quadratic case!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> dai;
  Core::LinAlg::split_matrix2x2(
      dmatrix_, gactivedofs_, gidofs, gactivedofs_, gidofs, tempmtx1, dai, tempmtx2, tempmtx3);

  // do the multiplication dhat = invda * dai
  std::shared_ptr<Core::LinAlg::SparseMatrix> dhat =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 10);
  if (aset && iset)
    dhat = Core::LinAlg::matrix_multiply(*invda, false, *dai, false, false, false, true);
  dhat->complete(*gidofs, *gactivedofs_);

  // active part of mmatrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrixa;
  Core::LinAlg::split_matrix2x2(mmatrix_, gactivedofs_, gidofs, gmdofrowmap_, tempmap, mmatrixa,
      tempmtx1, tempmtx2, tempmtx3);

  // do the multiplication mhataam = invda * mmatrixa
  // (this is only different from mhata for 3D quadratic case!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> mhataam =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 10);
  if (aset)
    mhataam = Core::LinAlg::matrix_multiply(*invda, false, *mmatrixa, false, false, false, true);
  mhataam->complete(*gmdofrowmap_, *gactivedofs_);

  // for the case without full linearization, we still need the
  // "classical" active part of mhat, which is isolated here
  std::shared_ptr<Core::LinAlg::SparseMatrix> mhata;
  Core::LinAlg::split_matrix2x2(mhatmatrix_, gactivedofs_, gidofs, gmdofrowmap_, tempmap, mhata,
      tempmtx1, tempmtx2, tempmtx3);

  // scaling of invd and dai
  invda->scale(1 / (1 - alphaf_));
  invdsl->scale(1 / (1 - alphaf_));
  invdst->scale(1 / (1 - alphaf_));
  dai->scale(1 - alphaf_);

  /********************************************************************/
  //                       WEARBLOCKS - BASIC
  /********************************************************************/
  // wcoeff: all blocks based on T-matrix have to be scaled with this
  // coefficient...
  double wcoeff = params().get<double>("WEARCOEFF");

  /********************************************************************/
  /* (a) create inv E                                                */
  /********************************************************************/
  Core::LinAlg::SparseMatrix inve(*ematrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diage =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipn_, true);

  // extract diagonal of inve into diage
  inve.extract_diagonal_copy(*diage);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diage->local_length(); ++i)
    if (diage->local_values_as_span()[i] == 0.0) (*diage).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diage->reciprocal(*diage);

  // re-insert inverted diagonal into invd
  inve.replace_diagonal_values(*diage);

  /********************************************************************/
  /* (b) build linedis + lintdis                                     */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> linetdis =
      std::make_shared<Core::LinAlg::SparseMatrix>(*lintdis_);

  // scale T-part with wear coeff.
  linetdis->scale(-wcoeff);
  linetdis->un_complete();

  // add E-part
  Core::LinAlg::matrix_add(*linedis_, false, 1.0, *linetdis, 1.0);

  // complete
  linetdis->complete(*gsmdofrowmap_, *gslipn_);

  /********************************************************************/
  /* (c) Split linetdis into 4 matrix blocks                         */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> linte_m, linte_s, dummy1, dummy2;
  std::shared_ptr<Core::LinAlg::SparseMatrix> linte_st, linte_sl, linte_i, linte_a;

  // split slave master...
  Core::LinAlg::split_matrix2x2(
      linetdis, gslipn_, gslipn_, gsdofrowmap_, gmdofrowmap_, linte_s, linte_m, dummy1, dummy2);

  // split active inactive
  Core::LinAlg::split_matrix2x2(
      linte_s, gslipn_, gslipn_, gactivedofs_, gidofs, linte_a, linte_i, dummy1, dummy2);

  // split stick slip
  Core::LinAlg::split_matrix2x2(
      linte_a, gslipn_, gslipn_, gslipdofs_, gstdofs, linte_sl, linte_st, dummy1, dummy2);

  /********************************************************************/
  /* (d) Split lintlm_ into 4 matrix blocks                           */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> lintlm_st, lintlm_sl, lintlm_i, lintlm_a;

  // split active inactive
  lintlm_->scale(wcoeff);
  Core::LinAlg::split_matrix2x2(
      lintlm_, gslipn_, gslipn_, gactivedofs_, gidofs, lintlm_a, lintlm_i, dummy1, dummy2);

  // split stick slip
  Core::LinAlg::split_matrix2x2(
      lintlm_a, gslipn_, gslipn_, gslipdofs_, gstdofs, lintlm_sl, lintlm_st, dummy1, dummy2);

  /********************************************************************/
  /* (e) create Tn*D^-T                                               */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> tndt_a;
  if (aset)
    tndt_a = Core::LinAlg::matrix_multiply(*lintlm_a, false, *invda, true, false, false, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> wear_dn, wear_dm, wear_di, wear_da;

  /********************************************************************/
  /* (f) delta_w = wear_dn*delta_dn + wear_dm*delta_dm + wear_da*delta_da + wear_di*delta_di + fw_z
   */
  /********************************************************************/
  // these are the matrices for the wear --> wear at slipnodes
  if (slipset)
  {
    // dn
    if (aset)
    {
      wear_dn = Core::LinAlg::matrix_multiply(inve, false, *tndt_a, false, false, false, true);
      wear_dn = Core::LinAlg::matrix_multiply(*wear_dn, false, *kan, false, false, false, true);

      wear_dn->scale(-1.0);
    }
    // dm
    if (aset)
    {
      wear_dm = Core::LinAlg::matrix_multiply(inve, false, *tndt_a, false, false, false, true);
      wear_dm = Core::LinAlg::matrix_multiply(*wear_dm, false, *kam, false, false, false, true);

      wear_dm->scale(-1.0);
    }
    // da
    if (aset)
    {
      wear_da = Core::LinAlg::matrix_multiply(inve, false, *tndt_a, false, false, false, true);
      wear_da = Core::LinAlg::matrix_multiply(*wear_da, false, *kaa, false, false, false, false);

      std::shared_ptr<Core::LinAlg::SparseMatrix> inve_te;
      inve_te = Core::LinAlg::matrix_multiply(inve, false, *linte_a, false, false, false, true);

      Core::LinAlg::matrix_add(*inve_te, false, 1.0, *wear_da, 1.0);
      wear_da->complete(*gactivedofs_, *gslipn_);
      wear_da->scale(-1.0);
    }
    // di
    if (aset && iset)
    {
      wear_di = Core::LinAlg::matrix_multiply(inve, false, *tndt_a, false, false, false, true);
      wear_di = Core::LinAlg::matrix_multiply(*wear_di, false, *kai, false, false, false, false);

      std::shared_ptr<Core::LinAlg::SparseMatrix> inve_te;
      inve_te = Core::LinAlg::matrix_multiply(inve, false, *linte_i, false, false, false, true);

      Core::LinAlg::matrix_add(*inve_te, false, 1.0, *wear_di, 1.0);
      wear_di->complete(*gidofs, *gslipn_);
      wear_di->scale(-1.0);
    }
  }

  // store to header_variables --> reused for recovering
  dnblock_ = wear_dn;
  dmblock_ = wear_dm;
  diblock_ = wear_di;
  dablock_ = wear_da;

  //*************************************************************
  // (g) prepare f_w
  //*************************************************************
  Core::LinAlg::Vector<double> dvec(*gslipn_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fw_z =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipn_);
  Core::LinAlg::Vector<double> fw_wrhs(*gslipn_);

  if (slipset)
  {
    tndt_a->multiply(false, *fa, dvec);
    inve.multiply(false, dvec, *fw_z);

    // fa = -ra
    fw_z->scale(-1.0);

    // this wear condrhs excludes the lm-part of the rhs
    wear_cond_rhs_->scale(-1.0);
    inve.multiply(false, *wear_cond_rhs_, fw_wrhs);
    fw_z->update(1.0, fw_wrhs, 1.0);

    fw_z->scale(-1.0);

    fw_ = std::make_shared<Core::LinAlg::Vector<double>>(*gslipn_);
    fw_ = fw_z;
  }

  // ************************************************************
  // (h) prepare smatrixW and linslipW_
  // ************************************************************
  std::shared_ptr<Core::LinAlg::SparseMatrix> smatrixW_sl, smatrixW_i, d2, d3;
  Core::LinAlg::split_matrix2x2(
      smatrixW_, gactiven_, gactiven_, gslipn_, gwinact_, smatrixW_sl, smatrixW_i, d2, d3);

  std::shared_ptr<Core::LinAlg::SparseMatrix> linslipW_sl, linslipW_i, d5, d6;
  Core::LinAlg::split_matrix2x2(
      linslip_w_, gslipt_, gslipt_, gslipn_, gwinact_, linslipW_sl, linslipW_i, d5, d6);

  /********************************************************************/
  /* (7) Build the final K blocks                                     */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // knn: nothing to do

  // knm: nothing to do

  // kns: nothing to do

  //-------------------------------------------------------- SECOND LINE
  // kmn: add T(mhataam)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(*kmn, false, 1.0, *kmnmod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
  Core::LinAlg::matrix_add(*kmnadd, false, 1.0, *kmnmod, 1.0);
  kmnmod->complete(kmn->domain_map(), kmn->row_map());

  // kmm: add T(mhataam)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(*kmm, false, 1.0, *kmmmod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
  Core::LinAlg::matrix_add(*kmmadd, false, 1.0, *kmmmod, 1.0);
  kmmmod->complete(kmm->domain_map(), kmm->row_map());

  // kmi: add T(mhataam)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
  if (iset)
  {
    kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*kmi, false, 1.0, *kmimod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
    Core::LinAlg::matrix_add(*kmiadd, false, 1.0, *kmimod, 1.0);
    kmimod->complete(kmi->domain_map(), kmi->row_map());
  }

  // kma: add T(mhataam)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
  if (aset)
  {
    kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*kma, false, 1.0, *kmamod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
    Core::LinAlg::matrix_add(*kmaadd, false, 1.0, *kmamod, 1.0);
    kmamod->complete(kma->domain_map(), kma->row_map());
  }

  //--------------------------------------------------------- THIRD LINE
  //------------------- FOR 3D QUADRATIC CASE ----------------------------
  // kin: subtract T(dhat)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  Core::LinAlg::matrix_add(*kin, false, 1.0, *kinmod, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
    Core::LinAlg::matrix_add(*kinadd, false, -1.0, *kinmod, 1.0);
  }
  kinmod->complete(Core::LinAlg::Map(kin->domain_map()), kin->row_map());

  // kim: subtract T(dhat)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  Core::LinAlg::matrix_add(*kim, false, 1.0, *kimmod, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
    Core::LinAlg::matrix_add(*kimadd, false, -1.0, *kimmod, 1.0);
  }
  kimmod->complete(Core::LinAlg::Map(kim->domain_map()), kim->row_map());

  // kii: subtract T(dhat)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
  if (iset)
  {
    kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    Core::LinAlg::matrix_add(*kii, false, 1.0, *kiimod, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
      Core::LinAlg::matrix_add(*kiiadd, false, -1.0, *kiimod, 1.0);
    }
    kiimod->complete(Core::LinAlg::Map(kii->domain_map()), kii->row_map());
  }

  // kia: subtract T(dhat)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
  if (iset && aset)
  {
    kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    Core::LinAlg::matrix_add(*kia, false, 1.0, *kiamod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
    Core::LinAlg::matrix_add(*kiaadd, false, -1.0, *kiamod, 1.0);
    kiamod->complete(Core::LinAlg::Map(kia->domain_map()), kia->row_map());
  }

  //--------------------------------------------------------- FOURTH LINE
  // WEAR STUFF
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgnw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgmw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgiw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kgaw;

  if (aset && slipset)
    kgnw = Core::LinAlg::matrix_multiply(*smatrixW_sl, false, *wear_dn, false, false, false, true);
  if (aset && slipset)
    kgmw = Core::LinAlg::matrix_multiply(*smatrixW_sl, false, *wear_dm, false, false, false, true);
  if (aset && slipset && iset)
    kgiw = Core::LinAlg::matrix_multiply(*smatrixW_sl, false, *wear_di, false, false, false, true);
  if (aset && slipset)
    kgaw = Core::LinAlg::matrix_multiply(*smatrixW_sl, false, *wear_da, false, false, false, true);

  //--------------------------------------------------------- FIFTH  LINE
  // blocks for complementary conditions (stick nodes)

  // kstn: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstnmod;
  if (stickset)
  {
    kstnmod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstnmod = Core::LinAlg::matrix_multiply(*kstnmod, false, *kan, false, false, false, true);
  }

  // kstm: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstmmod;
  if (stickset)
  {
    kstmmod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstmmod = Core::LinAlg::matrix_multiply(*kstmmod, false, *kam, false, false, false, true);
  }

  // ksti: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstimod;
  if (stickset && iset)
  {
    kstimod = Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstimod = Core::LinAlg::matrix_multiply(*kstimod, false, *kai, false, false, false, true);
  }

  // kstsl: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kstslmod;
  if (stickset && slipset)
  {
    kstslmod =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kstslmod = Core::LinAlg::matrix_multiply(*kstslmod, false, *kasl, false, false, false, true);
  }

  // kststmod: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kststmod;
  if (stickset)
  {
    kststmod =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    kststmod = Core::LinAlg::matrix_multiply(*kststmod, false, *kast, false, false, false, true);
  }

  //--------------------------------------------------------- SIXTH LINE
  // blocks for complementary conditions (slip nodes)

  // ksln: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslnmod;
  if (slipset)
  {
    kslnmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslnmod = Core::LinAlg::matrix_multiply(*kslnmod, false, *kan, false, false, false, true);
  }

  // kslm: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslmmod;
  if (slipset)
  {
    kslmmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslmmod = Core::LinAlg::matrix_multiply(*kslmmod, false, *kam, false, false, false, true);
  }

  // ksli: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslimod;
  if (slipset && iset)
  {
    kslimod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslimod = Core::LinAlg::matrix_multiply(*kslimod, false, *kai, false, false, false, true);
  }

  // kslsl: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslslmod;
  if (slipset)
  {
    kslslmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslslmod = Core::LinAlg::matrix_multiply(*kslslmod, false, *kasl, false, false, false, true);
  }

  // slstmod: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslstmod;
  if (slipset && stickset)
  {
    kslstmod = Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    kslstmod = Core::LinAlg::matrix_multiply(*kslstmod, false, *kast, false, false, false, true);
  }

  // WEARSTUFF FOR THIS LINE
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslnw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslmw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksliw;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kslaw;

  if (slipset)
    kslnw = Core::LinAlg::matrix_multiply(*linslipW_sl, false, *wear_dn, false, false, false, true);
  if (slipset)
    kslmw = Core::LinAlg::matrix_multiply(*linslipW_sl, false, *wear_dm, false, false, false, true);
  if (slipset && iset)
    ksliw = Core::LinAlg::matrix_multiply(*linslipW_sl, false, *wear_di, false, false, false, true);
  if (slipset)
    kslaw = Core::LinAlg::matrix_multiply(*linslipW_sl, false, *wear_da, false, false, false, true);

  /********************************************************************/
  /* (8) Build the final f blocks                                     */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // fn: nothing to do

  //---------------------------------------------------------- SECOND LINE
  // fm: add alphaf * old contact forces (t_n)
  // for self contact, slave and master sets may have changed,
  // thus we have to export the product Mold^T * zold to fit
  if (is_self_contact())
  {
    Core::LinAlg::Vector<double> tempvecm(*gmdofrowmap_);
    Core::LinAlg::Vector<double> tempvecm2(mold_->domain_map());
    Core::LinAlg::Vector<double> zoldexp(mold_->row_map());
    if (mold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
    mold_->multiply(true, zoldexp, tempvecm2);
    if (mset) Core::LinAlg::export_to(tempvecm2, tempvecm);
    fm->update(alphaf_, tempvecm, 1.0);
  }
  // if there is no self contact everything is ok
  else
  {
    Core::LinAlg::Vector<double> tempvecm(*gmdofrowmap_);
    mold_->multiply(true, *zold_, tempvecm);
    fm->update(alphaf_, tempvecm, 1.0);
  }

  // fs: prepare alphaf * old contact forces (t_n)
  Core::LinAlg::Vector<double> fsadd(*gsdofrowmap_);

  // for self contact, slave and master sets may have changed,
  // thus we have to export the product Dold^T * zold to fit
  if (is_self_contact())
  {
    Core::LinAlg::Vector<double> tempvec(dold_->domain_map());
    Core::LinAlg::Vector<double> zoldexp(dold_->row_map());
    if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
    dold_->multiply(true, zoldexp, tempvec);
    if (sset) Core::LinAlg::export_to(tempvec, fsadd);
  }
  // if there is no self contact everything is ok
  else
  {
    dold_->multiply(true, *zold_, fsadd);
  }

  // fa: subtract alphaf * old contact forces (t_n)
  if (aset)
  {
    Core::LinAlg::Vector<double> faadd(*gactivedofs_);
    Core::LinAlg::export_to(fsadd, faadd);
    fa->update(-alphaf_, faadd, 1.0);
  }

  // fm: add T(mhat)*fa
  Core::LinAlg::Vector<double> fmmod(*gmdofrowmap_);
  if (aset) mhataam->multiply(true, *fa, fmmod);
  fmmod.update(1.0, *fm, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // fi: subtract alphaf * old contact forces (t_n)
  if (iset)
  {
    Core::LinAlg::Vector<double> fiadd(*gidofs);
    Core::LinAlg::export_to(fsadd, fiadd);
    fi->update(-alphaf_, fiadd, 1.0);
  }

  // fi: add T(dhat)*fa
  Core::LinAlg::Vector<double> fimod(*gidofs);
  if (aset && iset) dhat->multiply(true, *fa, fimod);
  fimod.update(1.0, *fi, -1.0);

  //--------------------------------------------------------- FOURTH LINE
  //
  Core::LinAlg::Vector<double> fw_g(*gactiven_);
  Core::LinAlg::Vector<double> fwi_g(*gactiven_);

  if (aset && slipset) smatrixW_sl->multiply(false, *fw_z, fw_g);

  if (aset && (iset || stickset)) smatrixW_i->multiply(false, *inactive_wear_rhs_, fwi_g);

  //--------------------------------------------------------- FIFTH LINE
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);  // get global stick dofs

  // split the lagrange multiplier vector in stick and slip part
  std::shared_ptr<Core::LinAlg::Vector<double>> za =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> zi =
      std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);
  std::shared_ptr<Core::LinAlg::Vector<double>> zst =
      std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
  std::shared_ptr<Core::LinAlg::Vector<double>> zsl =
      std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);

  Core::LinAlg::split_vector(*gsdofrowmap_, *z_, gactivedofs_, za, gidofs, zi);
  Core::LinAlg::split_vector(*gactivedofs_, *za, gstickdofs, zst, gslipdofs_, zsl);
  std::shared_ptr<Core::LinAlg::Vector<double>> tempvec1;

  // fst: multiply with linstickLM
  std::shared_ptr<Core::LinAlg::Vector<double>> fstmod;
  if (stickset)
  {
    fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    temp1->multiply(false, *fa, *fstmod);

    tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);

    linstickLM_->multiply(false, *zst, *tempvec1);
    fstmod->update(-1.0, *tempvec1, 1.0);
  }

  //--------------------------------------------------------- SIXTH LINE
  // fsl: multiply with linslipLM
  std::shared_ptr<Core::LinAlg::Vector<double>> fslmod;

  if (slipset)
  {
    fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
        Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    temp->multiply(false, *fa, *fslmod);

    tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);

    linslipLM_->multiply(false, *zsl, *tempvec1);

    fslmod->update(-1.0, *tempvec1, 1.0);
  }

  // WEAR
  Core::LinAlg::Vector<double> fw_sl(*gslipt_);
  if (slipset) linslipW_sl->multiply(false, *fw_z, fw_sl);

  Core::LinAlg::Vector<double> fwi_sl(*gslipt_);
  if (slipset && (iset || stickset)) linslipW_i->multiply(false, *inactive_wear_rhs_, fwi_sl);

  /********************************************************************/
  /* (9) Transform the final K blocks                                 */
  /********************************************************************/
  // The row maps of all individual matrix blocks are transformed to
  // the parallel layout of the underlying problem discretization.
  // Of course, this is only necessary in the parallel redistribution
  // case, where the contact interfaces have been redistributed
  // independently of the underlying problem discretization.

  if (parallel_redistribution_status())
  {
    //----------------------------------------------------------- FIRST LINE
    // nothing to do (ndof-map independent of redistribution)

    //---------------------------------------------------------- SECOND LINE
    kmnmod = Core::LinAlg::matrix_row_transform(*kmnmod, *non_redist_gmdofrowmap_);
    kmmmod = Core::LinAlg::matrix_row_transform(*kmmmod, *non_redist_gmdofrowmap_);
    if (iset) kmimod = Core::LinAlg::matrix_row_transform(*kmimod, *non_redist_gmdofrowmap_);
    if (aset) kmamod = Core::LinAlg::matrix_row_transform(*kmamod, *non_redist_gmdofrowmap_);

    //----------------------------------------------------------- THIRD LINE
    if (iset)
    {
      kinmod = Core::LinAlg::matrix_row_transform(*kinmod, *non_redist_gsdofrowmap_);
      kimmod = Core::LinAlg::matrix_row_transform(*kimmod, *non_redist_gsdofrowmap_);
      kiimod = Core::LinAlg::matrix_row_transform(*kiimod, *non_redist_gsdofrowmap_);
      if (aset) kiamod = Core::LinAlg::matrix_row_transform(*kiamod, *non_redist_gsdofrowmap_);
    }

    //---------------------------------------------------------- FOURTH LINE
    if (aset)
    {
      smatrix_ = Core::LinAlg::matrix_row_transform(*smatrix_, *non_redist_gsdofrowmap_);

      // WEAR
      if (slipset) kgnw = Core::LinAlg::matrix_row_transform(*kgnw, *non_redist_gsdofrowmap_);
      if (slipset) kgmw = Core::LinAlg::matrix_row_transform(*kgmw, *non_redist_gsdofrowmap_);
      if (slipset && iset)
        kgiw = Core::LinAlg::matrix_row_transform(*kgiw, *non_redist_gsdofrowmap_);
      if (slipset) kgaw = Core::LinAlg::matrix_row_transform(*kgaw, *non_redist_gsdofrowmap_);
    }

    //----------------------------------------------------------- FIFTH LINE
    if (stickset)
    {
      kstnmod = Core::LinAlg::matrix_row_transform(*kstnmod, *non_redist_gsdofrowmap_);
      kstmmod = Core::LinAlg::matrix_row_transform(*kstmmod, *non_redist_gsdofrowmap_);
      if (iset) kstimod = Core::LinAlg::matrix_row_transform(*kstimod, *non_redist_gsdofrowmap_);
      if (slipset)
        kstslmod = Core::LinAlg::matrix_row_transform(*kstslmod, *non_redist_gsdofrowmap_);
      kststmod = Core::LinAlg::matrix_row_transform(*kststmod, *non_redist_gsdofrowmap_);
      linstickDIS_ = Core::LinAlg::matrix_row_transform(*linstickDIS_, *non_redist_gsdofrowmap_);
    }

    //----------------------------------------------------------- SIXTH LINE
    if (slipset)
    {
      kslnmod = Core::LinAlg::matrix_row_transform(*kslnmod, *non_redist_gsdofrowmap_);
      kslmmod = Core::LinAlg::matrix_row_transform(*kslmmod, *non_redist_gsdofrowmap_);
      if (iset) kslimod = Core::LinAlg::matrix_row_transform(*kslimod, *non_redist_gsdofrowmap_);
      if (stickset)
        kslstmod = Core::LinAlg::matrix_row_transform(*kslstmod, *non_redist_gsdofrowmap_);
      kslslmod = Core::LinAlg::matrix_row_transform(*kslslmod, *non_redist_gsdofrowmap_);
      linslipDIS_ = Core::LinAlg::matrix_row_transform(*linslipDIS_, *non_redist_gsdofrowmap_);

      // WEAR
      kslnw = Core::LinAlg::matrix_row_transform(*kslnw, *non_redist_gsdofrowmap_);
      kslmw = Core::LinAlg::matrix_row_transform(*kslmw, *non_redist_gsdofrowmap_);
      if (iset) ksliw = Core::LinAlg::matrix_row_transform(*ksliw, *non_redist_gsdofrowmap_);
      kslaw = Core::LinAlg::matrix_row_transform(*kslaw, *non_redist_gsdofrowmap_);
    }
  }

  /********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                */
  /********************************************************************/

  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
  std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());

  //--------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  Core::LinAlg::matrix_add(*knn, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*knm, false, 1.0, *kteffnew, 1.0);
  if (sset) Core::LinAlg::matrix_add(*kns, false, 1.0, *kteffnew, 1.0);

  //-------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  Core::LinAlg::matrix_add(*kmnmod, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*kmmmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kmimod, false, 1.0, *kteffnew, 1.0);
  if (aset) Core::LinAlg::matrix_add(*kmamod, false, 1.0, *kteffnew, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) Core::LinAlg::matrix_add(*kinmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kimmod, false, 1.0, *kteffnew, 1.0);
  if (iset) Core::LinAlg::matrix_add(*kiimod, false, 1.0, *kteffnew, 1.0);
  if (iset && aset) Core::LinAlg::matrix_add(*kiamod, false, 1.0, *kteffnew, 1.0);

  //-------------------------------------------------------- FOURTH LINE
  // WEAR - STUFF
  // add a submatrices to kteffnew
  if (aset) Core::LinAlg::matrix_add(*smatrix_, false, 1.0, *kteffnew, 1.0);
  if (aset && slipset) Core::LinAlg::matrix_add(*kgnw, false, 1.0, *kteffnew, 1.0);
  if (aset && slipset) Core::LinAlg::matrix_add(*kgmw, false, 1.0, *kteffnew, 1.0);
  if (aset && slipset && iset) Core::LinAlg::matrix_add(*kgiw, false, 1.0, *kteffnew, 1.0);
  if (aset && slipset) Core::LinAlg::matrix_add(*kgaw, false, 1.0, *kteffnew, 1.0);

  //--------------------------------------------------------- FIFTH LINE
  // add st submatrices to kteffnew
  if (stickset) Core::LinAlg::matrix_add(*kstnmod, false, 1.0, *kteffnew, 1.0);
  if (stickset) Core::LinAlg::matrix_add(*kstmmod, false, 1.0, *kteffnew, 1.0);
  if (stickset && iset) Core::LinAlg::matrix_add(*kstimod, false, 1.0, *kteffnew, 1.0);
  if (stickset && slipset) Core::LinAlg::matrix_add(*kstslmod, false, 1.0, *kteffnew, 1.0);
  if (stickset) Core::LinAlg::matrix_add(*kststmod, false, 1.0, *kteffnew, 1.0);

  // add terms of linearization of sick condition to kteffnew
  if (stickset) Core::LinAlg::matrix_add(*linstickDIS_, false, -1.0, *kteffnew, 1.0);

  //--------------------------------------------------------- SIXTH LINE
  // add sl submatrices to kteffnew
  if (slipset) Core::LinAlg::matrix_add(*kslnmod, false, 1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslmmod, false, 1.0, *kteffnew, 1.0);
  if (slipset && iset) Core::LinAlg::matrix_add(*kslimod, false, 1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslslmod, false, 1.0, *kteffnew, 1.0);
  if (slipset && stickset) Core::LinAlg::matrix_add(*kslstmod, false, 1.0, *kteffnew, 1.0);

  // add terms of linearization of slip condition to kteffnew and feffnew
  if (slipset) Core::LinAlg::matrix_add(*linslipDIS_, false, -1.0, *kteffnew, +1.0);

  // WEAR - STUFF
  if (slipset) Core::LinAlg::matrix_add(*kslnw, false, -1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslmw, false, -1.0, *kteffnew, 1.0);
  if (slipset && (iset)) Core::LinAlg::matrix_add(*ksliw, false, -1.0, *kteffnew, 1.0);
  if (slipset) Core::LinAlg::matrix_add(*kslaw, false, -1.0, *kteffnew, 1.0);

  // fill_complete kteffnew (square)
  kteffnew->complete();

  /********************************************************************/
  /* (11) Global setup of feffnew (including contact)                 */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // add n subvector to feffnew
  Core::LinAlg::Vector<double> fnexp(*problem_dofs());
  Core::LinAlg::export_to(*fn, fnexp);
  feffnew->update(1.0, fnexp, 1.0);

  //-------------------------------------------------------- SECOND LINE
  // add m subvector to feffnew
  Core::LinAlg::Vector<double> fmmodexp(*problem_dofs());
  Core::LinAlg::export_to(fmmod, fmmodexp);
  feffnew->update(1.0, fmmodexp, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // add i subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fimodexp;
  if (iset)
  {
    fimodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fimod, *fimodexp);
    feffnew->update(1.0, *fimodexp, 1.0);
  }

  //-------------------------------------------------------- FOURTH LINE
  // add weighted gap vector to feffnew, if existing
  std::shared_ptr<Core::LinAlg::Vector<double>> gexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwiexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fgmodexp;

  if (aset)
  {
    gexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(gact, *gexp);
    feffnew->update(-1.0, *gexp, 1.0);

    // DUE TO WEAR
    fwexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fw_g, *fwexp);
    feffnew->update(-1.0, *fwexp, 1.0);

    // DUE TO WEAR
    if (iset || stickset)
    {
      fwiexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(fwi_g, *fwiexp);
      feffnew->update(+1.0, *fwiexp, 1.0);
    }
  }

  //--------------------------------------------------------- FIFTH LINE
  // add st subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fstmodexp;
  if (stickset)
  {
    fstmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*fstmod, *fstmodexp);
    feffnew->update(1.0, *fstmodexp, +1.0);
  }

  // add terms of linearization feffnew
  if (stickset)
  {
    Core::LinAlg::Vector<double> linstickRHSexp(*problem_dofs());
    Core::LinAlg::export_to(*linstickRHS_, linstickRHSexp);
    feffnew->update(-1.0, linstickRHSexp, 1.0);
  }

  //--------------------------------------------------------- SIXTH LINE

  // add a subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fslmodexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwslexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fwsliexp;
  std::shared_ptr<Core::LinAlg::Vector<double>> fslwmodexp;

  if (slipset)
  {
    fslmodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*fslmod, *fslmodexp);
    feffnew->update(1.0, *fslmodexp, 1.0);
  }

  if (slipset)
  {
    Core::LinAlg::Vector<double> linslipRHSexp(*problem_dofs());
    Core::LinAlg::export_to(*linslipRHS_, linslipRHSexp);
    feffnew->update(-1.0, linslipRHSexp, 1.0);
  }

  // DUE TO WEAR
  if (slipset)
  {
    fwslexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fw_sl, *fwslexp);
    feffnew->update(+1.0, *fwslexp, 1.0);
  }
  if (slipset && (iset || stickset))
  {
    fwsliexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fwi_sl, *fwsliexp);
    feffnew->update(-1.0, *fwsliexp, 1.0);
  }

  // finally do the replacement
  kteff = kteffnew;
  feff = feffnew;

  return;
}


/*----------------------------------------------------------------------*
 | evaluate frictional wear contact (public)                 farah 10/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::evaluate_friction(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  // systemtype
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  // get wear shapefunction type
  auto wearshapefcn = Teuchos::getIntegralValue<Inpar::Wear::WearShape>(params(), "WEAR_SHAPEFCN");

  /**********************************************************************/
  /* export weighted gap vector to gactiveN-map                         */
  /* Here, the additional wear is already included !!!                  */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::Vector<double>> gact =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivenodes_, true);
  if (gact->global_length())
  {
    Core::LinAlg::export_to(*wgap_, *gact);
    gact->replace_map(*gactiven_);
  }

  /**********************************************************************/
  /* build global matrix t with tangent vectors of active nodes         */
  /* and global matrix s with normal derivatives of active nodes        */
  /* and global matrix linstick with derivatives of stick nodes         */
  /* and global matrix linslip with derivatives of slip nodes           */
  /* and inactive right-hand side with old lagrange multipliers (incr)  */
  /**********************************************************************/
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->assemble_tn(tmatrix_, nullptr);
    interface_[i]->assemble_s(*smatrix_);
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);
    interface_[i]->assemble_lin_stick(*linstickLM_, *linstickDIS_, *linstickRHS_);
    interface_[i]->assemble_lin_slip(*linslipLM_, *linslipDIS_, *linslipRHS_);
    if (systype != CONTACT::SystemType::condensed)
      interface_[i]->assemble_inactiverhs(*inactiverhs_);

    //***************************************************
    // Assemble lin. for implicit internal state wear algorithm
    if (wearimpl_ and !wearprimvar_)
    {  // assemble wear-specific matrices
      interface_[i]->assemble_lin_w_lm(*wlinmatrix_);
      interface_[i]->assemble_lin_w_lm_sl(*wlinmatrixsl_);

#ifdef CONSISTENTSTICK
      interface_[i]->AssembleLinWLmSt(*wlinmatrixst_);
#endif
    }

    //***************************************************
    // Assemble all Entries for implicit primary variable wear algorithm
    if (wearimpl_ and wearprimvar_)
    {
      // blocks for w-lines
      interface_[i]->assemble_te(*twmatrix_, *ematrix_);
      interface_[i]->assemble_lin_t_d(*lintdis_);
      interface_[i]->assemble_lin_t_lm(*lintlm_);
      interface_[i]->assemble_lin_e_d(*linedis_);

      // blocks for z-lines
      interface_[i]->assemble_lin_g_w(*smatrixW_);
      interface_[i]->assemble_lin_slip_w(*linslip_w_);

      // w-line rhs
      interface_[i]->assemble_inactive_wear_rhs(*inactive_wear_rhs_);
      interface_[i]->assemble_wear_cond_rhs(*wear_cond_rhs_);

      // for both-sided discrete wear
      if (wearbothpv_)
      {
        // blocks for w-lines
        interface_[i]->assemble_te_master(*twmatrix_m_, *ematrix_m_);
        interface_[i]->assemble_lin_t_d_master(*lintdis_m_);
        interface_[i]->assemble_lin_t_lm_master(*lintlm_m_);
        interface_[i]->assemble_lin_e_d_master(*linedis_m_);

        // w-line rhs
        interface_[i]->assemble_inactive_wear_rhs_master(*inactive_wear_rhs_m_);
        interface_[i]->assemble_wear_cond_rhs_master(*wear_cond_rhs_m_);
      }
    }
  }  // end interface loop

  /**********************************************************************/
  /* Complete matrices                                                  */
  /**********************************************************************/
  // fill_complete() global matrix T
  tmatrix_->complete(*gactivedofs_, *gactivet_);

  // fill_complete() global matrix S
  smatrix_->complete(*gsmdofrowmap_, *gactiven_);

  // fill_complete() global matrices LinD, LinM
  // (again for linD gsdofrowmap_ is sufficient as domain map,
  // but in the edge node modification case, master entries occur!)
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);

  // fill_complete global Matrix linstickLM_, linstickDIS_
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);
  linstickLM_->complete(*gstickdofs, *gstickt);
  linstickDIS_->complete(*gsmdofrowmap_, *gstickt);

  // fill_complete global Matrix linslipLM_ and linslipDIS_
  linslipLM_->complete(*gslipdofs_, *gslipt_);
  linslipDIS_->complete(*gsmdofrowmap_, *gslipt_);

  /**********************************************************************/
  /* Complete impl internal state wear-specific matrices                */
  /**********************************************************************/
  if (wearimpl_ and !wearprimvar_)
  {
    // complete wear-specific matrices
    wlinmatrix_->complete(*gsdofrowmap_, *gactiven_);
    wlinmatrixsl_->complete(*gsdofrowmap_, *gslipt_);

#ifdef CONSISTENTSTICK
    wlinmatrixst_->Complete(*gsdofrowmap_, *gstickt);
#endif
  }

  /**********************************************************************/
  /* Complete implicit primary variable wear-specific matrices          */
  /**********************************************************************/
  if (wearimpl_ and wearprimvar_)
  {
    // steady state scenario
    if (sswear_)
    {
      if (wearshapefcn == Inpar::Wear::wear_shape_dual)
        ematrix_->complete(*gactiven_, *gactiven_);  // quadr. matrix --> for dual shapes --> diag.
      else if (wearshapefcn == Inpar::Wear::wear_shape_standard)
        ematrix_->complete(
            *gsdofnrowmap_, *gactiven_);  // quadr. matrix --> for dual shapes --> diag.
      else
        FOUR_C_THROW("chosen shape fnc for wear not supported!");

      twmatrix_->complete(*gsdofnrowmap_, *gactiven_);

      lintdis_->complete(*gsmdofrowmap_, *gactiven_);
      lintlm_->complete(*gsdofrowmap_, *gactiven_);
      linedis_->complete(*gsmdofrowmap_, *gactiven_);
    }
    // general scenario
    else
    {
      if (wearshapefcn == Inpar::Wear::wear_shape_dual)
        ematrix_->complete(*gslipn_, *gslipn_);  // quadr. matrix --> for dual shapes --> diag.
      else if (wearshapefcn == Inpar::Wear::wear_shape_standard)
        ematrix_->complete(
            *gsdofnrowmap_, *gslipn_);  // quadr. matrix --> for dual shapes --> diag.
      else
        FOUR_C_THROW("chosen shape fnc for wear not supported!");

      twmatrix_->complete(*gsdofnrowmap_, *gslipn_);

      lintdis_->complete(*gsmdofrowmap_, *gslipn_);
      lintlm_->complete(*gsdofrowmap_, *gslipn_);
      linedis_->complete(*gsmdofrowmap_, *gslipn_);
    }

    // if both-sided complete with all wear dofs!
    if (wearbothpv_)
    {
      smatrixW_->complete(*galldofnrowmap_, *gactiven_);
      linslip_w_->complete(*galldofnrowmap_, *gslipt_);
    }
    else
    {
      smatrixW_->complete(*gsdofnrowmap_, *gactiven_);
      linslip_w_->complete(*gsdofnrowmap_, *gslipt_);
    }

    if (wearbothpv_)
    {
      ematrix_m_->complete(*gmdofnrowmap_, *gmslipn_);
      twmatrix_m_->complete(*gsdofnrowmap_, *gmslipn_);

      lintdis_m_->complete(*gsmdofrowmap_, *gmslipn_);
      lintlm_m_->complete(*gsdofrowmap_, *gmslipn_);
      linedis_m_->complete(*gsmdofrowmap_, *gmslipn_);
    }
  }

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify lindmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::matrix_multiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
    lindmatrix_ = temp1;
  }

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL) ------ wearprimvar_
  // HERE THE LM ARE SOLVED ABSOLUTELY !!!
  //**********************************************************************
  //**********************************************************************
  if ((systype == CONTACT::SystemType::condensed) && wearprimvar_)
  {
    condense_wear_discr(kteff, feff, *gact);
  }
  //**********************************************************************
  //**********************************************************************
  // CASE B: CONDENSED SYSTEM (DUAL) ------ WEARIMPL
  // HERE THE LM ARE SOLVED ABSOLUTELY !!!
  //**********************************************************************
  //**********************************************************************
  else if ((systype == CONTACT::SystemType::condensed) && !wearprimvar_)
  {
    condense_wear_impl_expl(kteff, feff, *gact);
  }
  //**********************************************************************
  //**********************************************************************
  // CASE C: SADDLE POINT SYSTEM
  //**********************************************************************
  //**********************************************************************
  else
  {
    prepare_saddle_point_system(*kteff, *feff);
  }
  // FD checks...
#ifdef WEARIMPLICITFDLM
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (!wearimpl_) FOUR_C_THROW("Explicit wear algorithm: no FD check necessary!");

    interface_[i]->FDCheckWearDerivLm();
  }
#endif

#ifdef WEARIMPLICITFD
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (!wearimpl_) FOUR_C_THROW("Explicit wear algorithm: no FD check necessary!");

    interface_[i]->FDCheckWearDeriv();
  }
#endif

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckGapDeriv();
    interface_[i]->FDCheckGapDeriv_W();
  }
#endif

#ifdef CONTACTFDSLIPINCR
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->fd_check_slip_incr_deriv_txi();
    if (Dim() == 3) interface_[i]->fd_check_slip_incr_deriv_teta();
  }
#endif  // #ifdef CONTACTFDGAP

#ifdef CONTACTFDSTICK

  if (gstickt->NumGlobalElements())
  {
    // FD check of stick condition
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckStickDeriv(*linstickLM_, *linstickDIS_);
    }
  }
#endif  // #ifdef CONTACTFDSTICK

#ifdef CONTACTFDT_D

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of stick condition
  for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->FDCheckDerivT_D(*lintdis_);

#endif

#ifdef CONTACTFDT_D_MASTER

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  if (!wearbothpv_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of stick condition
  for (int i = 0; i < (int)interface_.size(); ++i)
    interface_[i]->fd_check_deriv_t_d_master(*lintdisM_);

#endif

#ifdef CONTACTFDE_D

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of stick condition
  for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->FDCheckDerivE_D(*linedis_);

#endif

#ifdef CONTACTFDE_D_MASTER

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  if (!wearbothpv_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of stick condition
  for (int i = 0; i < (int)interface_.size(); ++i)
    interface_[i]->fd_check_deriv_e_d_master(*linedisM_);

#endif

#ifdef CONTACTFDSLIP

  if (gslipnodes_->NumGlobalElements())
  {
    // FD check of slip condition
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckSlipDeriv(*linslipLM_, *linslipDIS_, *linslipW_);
    }
  }
#endif  // #ifdef CONTACTFDSLIP

#ifdef CONTACTFDMORTART

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of Mortar matrix D derivatives
  std::cout << " -- CONTACTFDMORTART -----------------------------------" << std::endl;
  twmatrix_->Complete();
  if (twmatrix_->NormOne())
    for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->FDCheckMortarTDeriv();
  // twmatrix_->UnComplete();
  std::cout << " -- CONTACTFDMORTART -----------------------------------" << std::endl;
#endif  // #ifdef CONTACTFDMORTARD

#ifdef CONTACTFDMORTARE

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of Mortar matrix D derivatives
  std::cout << " -- CONTACTFDMORTARE -----------------------------------" << std::endl;
  ematrix_->Complete();
  if (ematrix_->NormOne())
    for (int i = 0; i < (int)interface_.size(); ++i) interface_[i]->FDCheckMortarEDeriv();
  // ematrix_->UnComplete();
  std::cout << " -- CONTACTFDMORTARE -----------------------------------" << std::endl;
#endif  // #ifdef CONTACTFDMORTARD

#ifdef CONTACTFDMORTARE_MASTER

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  if (!wearbothpv_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of Mortar matrix D derivatives
  std::cout << " -- CONTACTFDMORTARE_MASTER -----------------------------------" << std::endl;
  ematrixM_->Complete();
  if (ematrixM_->NormOne())
    for (int i = 0; i < (int)interface_.size(); ++i)
      interface_[i]->fd_check_mortar_e_master_deriv();
  // ematrix_->UnComplete();
  std::cout << " -- CONTACTFDMORTARE_MASTER -----------------------------------" << std::endl;
#endif  // #ifdef CONTACTFDMORTARD

#ifdef CONTACTFDMORTART_MASTER

  if (!wearprimvar_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  if (!wearbothpv_) FOUR_C_THROW("FD CHECK ONLY FOR DISCRETE WEAR!");

  // FD check of Mortar matrix D derivatives
  std::cout << " -- CONTACTFDMORTART_MASTER -----------------------------------" << std::endl;
  twmatrixM_->Complete();
  if (twmatrixM_->NormOne())
    for (int i = 0; i < (int)interface_.size(); ++i)
      interface_[i]->fd_check_mortar_t_master_deriv();
  // ematrix_->UnComplete();
  std::cout << " -- CONTACTFDMORTART_MASTER -----------------------------------" << std::endl;
#endif  // #ifdef CONTACTFDMORTARD

  return;
}

/*----------------------------------------------------------------------*
 | preparation for self-contact and assemble lind/m          farah 10/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::prepare_saddle_point_system(
    Core::LinAlg::SparseOperator& kteff, Core::LinAlg::Vector<double>& feff)
{
  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify dmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    dmatrix_ = temp2;
  }

  // transform if necessary
  if (parallel_redistribution_status())
  {
    lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
    linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
  }

  // add contact stiffness
  kteff.un_complete();
  kteff.add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff.add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
  kteff.complete();

  // for self contact, slave and master sets may have changed,
  // thus we have to export the products Dold^T * zold / D^T * z to fit
  // thus we have to export the products Mold^T * zold / M^T * z to fit
  if (is_self_contact())
  {
    // add contact force terms
    Core::LinAlg::Vector<double> fsexp(*problem_dofs());
    Core::LinAlg::Vector<double> tempvecd(dmatrix_->domain_map());
    Core::LinAlg::Vector<double> zexp(dmatrix_->row_map());
    if (dmatrix_->row_map().num_global_elements()) Core::LinAlg::export_to(*z_, zexp);
    dmatrix_->multiply(true, zexp, tempvecd);
    Core::LinAlg::export_to(tempvecd, fsexp);
    feff.update(-(1.0 - alphaf_), fsexp, 1.0);

    Core::LinAlg::Vector<double> fmexp(*problem_dofs());
    Core::LinAlg::Vector<double> tempvecm(mmatrix_->domain_map());
    mmatrix_->multiply(true, zexp, tempvecm);
    Core::LinAlg::export_to(tempvecm, fmexp);
    feff.update(1.0 - alphaf_, fmexp, 1.0);

    // add old contact forces (t_n)
    Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
    Core::LinAlg::Vector<double> tempvecdold(dold_->domain_map());
    Core::LinAlg::Vector<double> zoldexp(dold_->row_map());
    if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
    dold_->multiply(true, zoldexp, tempvecdold);
    Core::LinAlg::export_to(tempvecdold, fsoldexp);
    feff.update(-alphaf_, fsoldexp, 1.0);

    Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
    Core::LinAlg::Vector<double> tempvecmold(mold_->domain_map());
    mold_->multiply(true, zoldexp, tempvecmold);
    Core::LinAlg::export_to(tempvecmold, fmoldexp);
    feff.update(alphaf_, fmoldexp, 1.0);
  }
  // if there is no self contact everything is ok
  else
  {
    // add contact force terms
    Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
    dmatrix_->multiply(true, *z_, fs);
    Core::LinAlg::Vector<double> fsexp(*problem_dofs());
    Core::LinAlg::export_to(fs, fsexp);
    feff.update(-(1.0 - alphaf_), fsexp, 1.0);

    Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
    mmatrix_->multiply(true, *z_, fm);
    Core::LinAlg::Vector<double> fmexp(*problem_dofs());
    Core::LinAlg::export_to(fm, fmexp);
    feff.update(1.0 - alphaf_, fmexp, 1.0);

    ///////////////////////////////////////////////////////////////////
    //// FOR STATIC PROBLEMS --> alphaf_=0 !!! --> this is not needed!
    // add old contact forces (t_n)
    Core::LinAlg::Vector<double> fsold(*gsdofrowmap_);
    dold_->multiply(true, *zold_, fsold);
    Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
    Core::LinAlg::export_to(fsold, fsoldexp);
    feff.update(-alphaf_, fsoldexp, 1.0);

    Core::LinAlg::Vector<double> fmold(*gmdofrowmap_);
    mold_->multiply(true, *zold_, fmold);
    Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
    Core::LinAlg::export_to(fmold, fmoldexp);
    feff.update(alphaf_, fmoldexp, 1.0);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Setup 2x2 saddle point system for contact problems      wiesner 11/14|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::build_saddle_point_system(
    std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
    std::shared_ptr<Core::LinAlg::Vector<double>> fd,
    std::shared_ptr<Core::LinAlg::Vector<double>> sold,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
    std::shared_ptr<Core::LinAlg::SparseOperator>& blockMat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
    std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs)
{
  // create old style dirichtoggle vector (supposed to go away)
  // the use of a toggle vector is more flexible here. It allows to apply dirichlet
  // conditions on different matrix blocks separately.
  Core::LinAlg::Vector<double> dirichtoggle(*(dbcmaps->full_map()));
  Core::LinAlg::Vector<double> temp(*(dbcmaps->cond_map()));
  temp.put_scalar(1.0);
  Core::LinAlg::export_to(temp, dirichtoggle);

  // get system type
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  //**********************************************************************
  // prepare saddle point system
  //**********************************************************************
  // the standard stiffness matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiffmt =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kdd);

  // initialize merged system (matrix, rhs, sol);
  std::shared_ptr<Core::LinAlg::Map> mergedmap = nullptr;
  if (!wearprimvar_)
    mergedmap = Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
  else if (wearprimvar_ and !wearbothpv_)
  {
    std::shared_ptr<Core::LinAlg::Map> map_dummy =
        Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(map_dummy, gwdofrowmap_, false);
  }
  else
  {
    std::shared_ptr<Core::LinAlg::Map> map_dummy =
        Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(map_dummy, gwdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(mergedmap, gwmdofrowmap_, false);  // slave + master wear
  }

  std::shared_ptr<Core::LinAlg::SparseMatrix> mergedmt = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedrhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap);
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedsol =
      std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap);
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedzeros =
      std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap);

  // initialize constraint r.h.s. (still with wrong map)
  std::shared_ptr<Core::LinAlg::Vector<double>> constrrhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);

  // initialize transformed constraint matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkdz, trkzd, trkzz;

  // Wear stuff for own discretization:
  // initialize wear r.h.s. (still with wrong map)
  std::shared_ptr<Core::LinAlg::Vector<double>> wearrhs = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> wearrhsM = nullptr;  // for master

  if (wearprimvar_) wearrhs = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofnrowmap_);
  if (wearprimvar_ and wearbothpv_)
    wearrhsM = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofnrowmap_);
  // initialize transformed constraint matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkwd, trkwz, trkww, trkzw, trkdw;        // slave
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkwmd, trkwmz, trkwmwm, trkzwm, trkdwm;  // master
  // currently slave + master coeff
  double wcoeff = params().get<double>("WEARCOEFF");
  double wcoeffM = params().get<double>("WEARCOEFF_MASTER");

  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // *** CASE 1: FRICTIONLESS CONTACT ************************************
  if (!friction_)
  {
    FOUR_C_THROW("LagrangeStrategyWear::SaddlePointSolve: Wear called without friction!");
  }

  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // *** CASE 2: Wear as pp quantity *************************************
  else if (!wearprimvar_)
  {
    // global stick dof map
    std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);

    // build constraint matrix kdz
    Core::LinAlg::SparseMatrix kdz(*gdisprowmap_, 100, false, true);
    Core::LinAlg::matrix_add(*dmatrix_, true, 1.0 - alphaf_, kdz, 1.0);
    Core::LinAlg::matrix_add(*mmatrix_, true, -(1.0 - alphaf_), kdz, 1.0);
    kdz.complete(*gsdofrowmap_, *gdisprowmap_);

    // transform constraint matrix kzd to lmdofmap (matrix_col_transform)
    trkdz = Core::LinAlg::matrix_col_transform_gids(kdz, *glmdofrowmap_);

    // transform parallel row distribution of constraint matrix kdz
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkdz = Core::LinAlg::matrix_row_transform(*trkdz, *problem_dofs());

    // build constraint matrix kzd
    Core::LinAlg::SparseMatrix kzd(*gsdofrowmap_, 100, false, true);
    if (gactiven_->num_global_elements()) Core::LinAlg::matrix_add(*smatrix_, false, 1.0, kzd, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickDIS_, false, 1.0, kzd, 1.0);
    if (gslipt_->num_global_elements())
      Core::LinAlg::matrix_add(*linslipDIS_, false, 1.0, kzd, 1.0);
    kzd.complete(*gdisprowmap_, *gsdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzd = Core::LinAlg::matrix_row_transform_gids(kzd, *glmdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkzd = Core::LinAlg::matrix_col_transform(*trkzd, *problem_dofs());

    // build unity matrix for inactive dofs
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    Core::LinAlg::Vector<double> ones(*gidofs);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    // build constraint matrix kzz
    Core::LinAlg::SparseMatrix kzz(*gsdofrowmap_, 100, false, true);
    if (gidofs->num_global_elements()) Core::LinAlg::matrix_add(onesdiag, false, 1.0, kzz, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickLM_, false, 1.0, kzz, 1.0);
    if (gslipt_->num_global_elements()) Core::LinAlg::matrix_add(*linslipLM_, false, 1.0, kzz, 1.0);

    if (wearimpl_)
    {
      // add C-fnc. derivatives w.r.t. lm-values to kzz
      if (gactiven_->num_global_elements())
        Core::LinAlg::matrix_add(*wlinmatrix_, false, 1.0, kzz, 1.0);
      if (gslipt_->num_global_elements())
        Core::LinAlg::matrix_add(*wlinmatrixsl_, false, 1.0, kzz, 1.0);

#ifdef CONSISTENTSTICK
      if (gstickt->NumGlobalElements()) kzz->Add(*wlinmatrixst_, false, 1.0, 1.0);
#endif
    }
    kzz.complete(*gsdofrowmap_, *gsdofrowmap_);

    // transform constraint matrix kzz to lmdofmap (matrix_row_col_transform)
    trkzz = Core::LinAlg::matrix_row_col_transform_gids(kzz, *glmdofrowmap_, *glmdofrowmap_);

    /****************************************************************************************
     ***                RIGHT-HAND SIDE                   ***
     ****************************************************************************************/


    // We solve for the incremental Lagrange multiplier dz_. Hence,
    // we can keep the contact force terms on the right-hand side!
    //
    // r = r_effdyn,co = r_effdyn + a_f * B_co(d_(n)) * z_(n) + (1-a_f) * B_co(d^(i)_(n+1)) *
    // z^(i)_(n+1)

    // export weighted gap vector
    std::shared_ptr<Core::LinAlg::Vector<double>> gact =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivenodes_, true);
    if (gactiven_->num_global_elements())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
    Core::LinAlg::Vector<double> gactexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*gact, gactexp);

    // export stick and slip r.h.s.
    Core::LinAlg::Vector<double> stickexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linstickRHS_, stickexp);
    Core::LinAlg::Vector<double> slipexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linslipRHS_, slipexp);

    // export inactive rhs
    Core::LinAlg::Vector<double> inactiverhsexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*inactiverhs_, inactiverhsexp);

    // build constraint rhs (1)
    constrrhs->update(1.0, inactiverhsexp, 1.0);

    // build constraint rhs
    constrrhs->update(-1.0, gactexp, 1.0);
    constrrhs->update(1.0, stickexp, 1.0);
    constrrhs->update(1.0, slipexp, 1.0);
    constrrhs->replace_map(*glmdofrowmap_);

    constrrhs_ = constrrhs;  // set constraint rhs vector
  }
  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // *** CASE 3: Wear as discrete variable *******************************
  else if (wearprimvar_ and !wearbothpv_)
  {
    // global stick dof map
    std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);

    // build constraint matrix kdz
    Core::LinAlg::SparseMatrix kdz(*gdisprowmap_, 100, false, true);
    Core::LinAlg::matrix_add(*dmatrix_, true, 1.0 - alphaf_, kdz, 1.0);
    Core::LinAlg::matrix_add(*mmatrix_, true, -(1.0 - alphaf_), kdz, 1.0);
    kdz.complete(*gsdofrowmap_, *gdisprowmap_);

    // transform constraint matrix kzd to lmdofmap (matrix_col_transform)
    trkdz = Core::LinAlg::matrix_col_transform_gids(kdz, *glmdofrowmap_);

    // transform parallel row distribution of constraint matrix kdz
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkdz = Core::LinAlg::matrix_row_transform(*trkdz, *problem_dofs());

    // build constraint matrix kzd
    Core::LinAlg::SparseMatrix kzd(*gsdofrowmap_, 100, false, true);
    if (gactiven_->num_global_elements()) Core::LinAlg::matrix_add(*smatrix_, false, 1.0, kzd, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickDIS_, false, 1.0, kzd, 1.0);
    if (gslipt_->num_global_elements())
      Core::LinAlg::matrix_add(*linslipDIS_, false, 1.0, kzd, 1.0);
    kzd.complete(*gdisprowmap_, *gsdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzd = Core::LinAlg::matrix_row_transform_gids(kzd, *glmdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkzd = Core::LinAlg::matrix_col_transform(*trkzd, *problem_dofs());

    // build unity matrix for inactive dofs
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    Core::LinAlg::Vector<double> ones(*gidofs);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    // build constraint matrix kzz
    Core::LinAlg::SparseMatrix kzz(*gsdofrowmap_, 100, false, true);
    if (gidofs->num_global_elements()) Core::LinAlg::matrix_add(onesdiag, false, 1.0, kzz, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickLM_, false, 1.0, kzz, 1.0);
    if (gslipt_->num_global_elements()) Core::LinAlg::matrix_add(*linslipLM_, false, 1.0, kzz, 1.0);

    kzz.complete(*gsdofrowmap_, *gsdofrowmap_);

    // transform constraint matrix kzz to lmdofmap (matrix_row_col_transform)
    trkzz = Core::LinAlg::matrix_row_col_transform_gids(kzz, *glmdofrowmap_, *glmdofrowmap_);


    // ***************************************************************************************************
    // additional wear
    // ***************************************************************************************************
    // build wear matrix kwd
    Core::LinAlg::SparseMatrix kwd(*gsdofnrowmap_, 100, false, true);
    if (sswear_)
    {
      if (gactiven_->num_global_elements())
        Core::LinAlg::matrix_add(*lintdis_, false, -wcoeff, kwd, 1.0);
      if (gactiven_->num_global_elements())
        Core::LinAlg::matrix_add(*linedis_, false, 1.0, kwd, 1.0);
      if (gactiven_->num_global_elements()) kwd.complete(*gdisprowmap_, *gsdofnrowmap_);
    }
    else
    {
      if (gslipn_->num_global_elements())
        Core::LinAlg::matrix_add(*lintdis_, false, -wcoeff, kwd, 1.0);
      if (gslipn_->num_global_elements()) Core::LinAlg::matrix_add(*linedis_, false, 1.0, kwd, 1.0);
      if (gslipn_->num_global_elements()) kwd.complete(*gdisprowmap_, *gsdofnrowmap_);
    }

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwd = Core::LinAlg::matrix_row_transform_gids(kwd, *gwdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkwd = Core::LinAlg::matrix_col_transform(*trkwd, *problem_dofs());

    // *********************************
    // build wear matrix kwz
    Core::LinAlg::SparseMatrix kwz(*gsdofnrowmap_, 100, false, true);
    if (sswear_)
    {
      if (gactiven_->num_global_elements())
        Core::LinAlg::matrix_add(*lintlm_, false, -wcoeff, kwz, 1.0);
    }
    else
    {
      if (gslipn_->num_global_elements())
        Core::LinAlg::matrix_add(*lintlm_, false, -wcoeff, kwz, 1.0);
    }

    kwz.complete(*gsdofrowmap_, *gsdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwz = Core::LinAlg::matrix_row_col_transform_gids(kwz, *gwdofrowmap_, *glmdofrowmap_);

    // *********************************
    // build wear matrix kww
    Core::LinAlg::SparseMatrix kww(*gsdofnrowmap_, 100, false, true);
    if (sswear_)
    {
      if (gactiven_->num_global_elements())
        Core::LinAlg::matrix_add(*ematrix_, false, 1.0, kww, 1.0);
    }
    else
    {
      if (gslipn_->num_global_elements()) Core::LinAlg::matrix_add(*ematrix_, false, 1.0, kww, 1.0);
    }

    // build unity matrix for inactive dofs
    Core::LinAlg::Vector<double> onesw(*gwinact_);
    onesw.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiagw(onesw);
    onesdiagw.complete();
    // build constraint matrix kzz
    if (gwinact_->num_global_elements()) Core::LinAlg::matrix_add(onesdiagw, false, 1.0, kww, 1.0);

    kww.complete(*gsdofnrowmap_, *gsdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkww = Core::LinAlg::matrix_row_col_transform_gids(kww, *gwdofrowmap_, *gwdofrowmap_);

    // *********************************
    // build wear matrix kzw
    Core::LinAlg::SparseMatrix kzw(*gsdofrowmap_, 100, false, true);
    if (gactiven_->num_global_elements())
      Core::LinAlg::matrix_add(*smatrixW_, false, 1.0, kzw, 1.0);
    if (gslipt_->num_global_elements()) Core::LinAlg::matrix_add(*linslip_w_, false, 1.0, kzw, 1.0);
    kzw.complete(*gsdofnrowmap_, *gsdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzw = Core::LinAlg::matrix_row_col_transform_gids(kzw, *glmdofrowmap_, *gwdofrowmap_);

    /****************************************************************************************
     ***                RIGHT-HAND SIDE                   ***
     ****************************************************************************************/


    // We solve for the incremental Lagrange multiplier dz_. Hence,
    // we can keep the contact force terms on the right-hand side!
    //
    // r = r_effdyn,co = r_effdyn + a_f * B_co(d_(n)) * z_(n) + (1-a_f) * B_co(d^(i)_(n+1)) *
    // z^(i)_(n+1)

    // export weighted gap vector
    std::shared_ptr<Core::LinAlg::Vector<double>> gact =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivenodes_, true);
    if (gactiven_->num_global_elements())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
    Core::LinAlg::Vector<double> gactexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*gact, gactexp);

    // export stick and slip r.h.s.
    Core::LinAlg::Vector<double> stickexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linstickRHS_, stickexp);
    Core::LinAlg::Vector<double> slipexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linslipRHS_, slipexp);

    // export inactive rhs
    Core::LinAlg::Vector<double> inactiverhsexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*inactiverhs_, inactiverhsexp);

    // build constraint rhs (1)
    constrrhs->update(1.0, inactiverhsexp, 1.0);

    // build constraint rhs
    constrrhs->update(-1.0, gactexp, 1.0);
    constrrhs->update(1.0, stickexp, 1.0);
    constrrhs->update(1.0, slipexp, 1.0);
    constrrhs->replace_map(*glmdofrowmap_);

    constrrhs_ = constrrhs;  // set constraint rhs vector

    // ***************************************************************************************************
    // additional wear-rhs
    // ***************************************************************************************************

    // export inactive wear rhs
    Core::LinAlg::Vector<double> WearCondRhsexp(*gsdofnrowmap_);
    Core::LinAlg::export_to(*wear_cond_rhs_, WearCondRhsexp);

    // export inactive wear rhs
    Core::LinAlg::Vector<double> inactiveWearRhsexp(*gsdofnrowmap_);
    Core::LinAlg::export_to(*inactive_wear_rhs_, inactiveWearRhsexp);

    wearrhs->update(1.0, WearCondRhsexp, 1.0);
    wearrhs->update(1.0, inactiveWearRhsexp, 1.0);
    wearrhs->replace_map(*gwdofrowmap_);

    wearrhs_ = wearrhs;
  }
  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // *** CASE 4: Wear as discrete variable on both sides******************
  else if (wearprimvar_ and wearbothpv_)
  {
    // global stick dof map
    std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);

    // build constraint matrix kdz
    Core::LinAlg::SparseMatrix kdz(*gdisprowmap_, 100, false, true);
    Core::LinAlg::matrix_add(*dmatrix_, true, 1.0 - alphaf_, kdz, 1.0);
    Core::LinAlg::matrix_add(*mmatrix_, true, -(1.0 - alphaf_), kdz, 1.0);
    kdz.complete(*gsdofrowmap_, *gdisprowmap_);

    // transform constraint matrix kzd to lmdofmap (matrix_col_transform)
    trkdz = Core::LinAlg::matrix_col_transform_gids(kdz, *glmdofrowmap_);

    // transform parallel row distribution of constraint matrix kdz
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkdz = Core::LinAlg::matrix_row_transform(*trkdz, *problem_dofs());

    // build constraint matrix kzd
    Core::LinAlg::SparseMatrix kzd(*gsdofrowmap_, 100, false, true);
    if (gactiven_->num_global_elements()) Core::LinAlg::matrix_add(*smatrix_, false, 1.0, kzd, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickDIS_, false, 1.0, kzd, 1.0);
    if (gslipt_->num_global_elements())
      Core::LinAlg::matrix_add(*linslipDIS_, false, 1.0, kzd, 1.0);
    kzd.complete(*gdisprowmap_, *gsdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzd = Core::LinAlg::matrix_row_transform_gids(kzd, *glmdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkzd = Core::LinAlg::matrix_col_transform(*trkzd, *problem_dofs());

    // build unity matrix for inactive dofs
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    Core::LinAlg::Vector<double> ones(*gidofs);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    // build constraint matrix kzz
    Core::LinAlg::SparseMatrix kzz(*gsdofrowmap_, 100, false, true);
    if (gidofs->num_global_elements()) Core::LinAlg::matrix_add(onesdiag, false, 1.0, kzz, 1.0);
    if (gstickt->num_global_elements())
      Core::LinAlg::matrix_add(*linstickLM_, false, 1.0, kzz, 1.0);
    if (gslipt_->num_global_elements()) Core::LinAlg::matrix_add(*linslipLM_, false, 1.0, kzz, 1.0);

    kzz.complete(*gsdofrowmap_, *gsdofrowmap_);

    // transform constraint matrix kzz to lmdofmap (matrix_row_col_transform)
    trkzz = Core::LinAlg::matrix_row_col_transform_gids(kzz, *glmdofrowmap_, *glmdofrowmap_);


    // ***************************************************************************************************
    // additional wear SLAVE
    // ***************************************************************************************************
    // build wear matrix kwd
    Core::LinAlg::SparseMatrix kwd(*gsdofnrowmap_, 100, false, true);
    if (gslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*lintdis_, false, -wcoeff, kwd, 1.0);
    if (gslipn_->num_global_elements()) Core::LinAlg::matrix_add(*linedis_, false, 1.0, kwd, 1.0);
    if (gslipn_->num_global_elements()) kwd.complete(*gdisprowmap_, *gsdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwd = Core::LinAlg::matrix_row_transform_gids(kwd, *gwdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkwd = Core::LinAlg::matrix_col_transform(*trkwd, *problem_dofs());

    // *********************************
    // build wear matrix kwz
    Core::LinAlg::SparseMatrix kwz(*gsdofnrowmap_, 100, false, true);
    if (gslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*lintlm_, false, -wcoeff, kwz, 1.0);
    kwz.complete(*gsdofrowmap_, *gsdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwz = Core::LinAlg::matrix_row_col_transform_gids(kwz, *gwdofrowmap_, *glmdofrowmap_);

    // *********************************
    // build wear matrix kww
    Core::LinAlg::SparseMatrix kww(*gsdofnrowmap_, 100, false, true);
    if (gslipn_->num_global_elements()) Core::LinAlg::matrix_add(*ematrix_, false, 1.0, kww, 1.0);

    // build unity matrix for inactive dofs
    Core::LinAlg::Vector<double> onesw(*gwinact_);
    onesw.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiagw(onesw);
    onesdiagw.complete();
    // build constraint matrix kzz
    if (gwinact_->num_global_elements()) Core::LinAlg::matrix_add(onesdiagw, false, 1.0, kww, 1.0);

    kww.complete(*gsdofnrowmap_, *gsdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkww = Core::LinAlg::matrix_row_col_transform_gids(kww, *gwdofrowmap_, *gwdofrowmap_);

    // FOR SLAVE AND MASTER
    // ********************************* S+M
    // build wear matrix kzw
    Core::LinAlg::SparseMatrix kzw(*gsdofrowmap_, 100, false, true);
    if (gactiven_->num_global_elements())
      Core::LinAlg::matrix_add(*smatrixW_, false, 1.0, kzw, 1.0);
    if (gslipt_->num_global_elements()) Core::LinAlg::matrix_add(*linslip_w_, false, 1.0, kzw, 1.0);
    kzw.complete(*galldofnrowmap_, *gsdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzw = Core::LinAlg::matrix_row_col_transform_gids(kzw, *glmdofrowmap_, *gwalldofrowmap_);

    // ***************************************************************************************************
    // additional wear MASTER
    // ***************************************************************************************************
    // build wear matrix kwmd
    Core::LinAlg::SparseMatrix kwmd(*gmdofnrowmap_, 100, false, true);
    if (gmslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*lintdis_m_, false, -wcoeffM, kwmd, 1.0);
    if (gmslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*linedis_m_, false, 1.0, kwmd, 1.0);
    if (gmslipn_->num_global_elements()) kwmd.complete(*gdisprowmap_, *gmdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwmd = Core::LinAlg::matrix_row_transform_gids(kwmd, *gwmdofrowmap_);

    // transform parallel column distribution of constraint matrix kzd
    // (only necessary in the parallel redistribution case)
    if (parallel_redistribution_status())
      trkwmd = Core::LinAlg::matrix_col_transform(*trkwmd, *problem_dofs());

    // *********************************
    // build wear matrix kwmz
    Core::LinAlg::SparseMatrix kwmz(*gmdofnrowmap_, 100, false, true);
    if (gmslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*lintlm_m_, false, -wcoeffM, kwmz, 1.0);
    kwmz.complete(*gsdofrowmap_, *gmdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwmz = Core::LinAlg::matrix_row_col_transform_gids(kwmz, *gwmdofrowmap_, *glmdofrowmap_);

    // *********************************
    // build wear matrix kwmwm
    Core::LinAlg::SparseMatrix kwmwm(*gmdofnrowmap_, 100, false, true);
    if (gmslipn_->num_global_elements())
      Core::LinAlg::matrix_add(*ematrix_m_, false, 1.0, kwmwm, 1.0);

    // build unity matrix for inactive dofs
    Core::LinAlg::Vector<double> oneswm(*gwminact_);
    oneswm.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiagwm(oneswm);
    onesdiagwm.complete();
    // build constraint matrix kzz
    if (gwminact_->num_global_elements())
      Core::LinAlg::matrix_add(onesdiagwm, false, 1.0, kwmwm, 1.0);

    kwmwm.complete(*gmdofnrowmap_, *gmdofnrowmap_);

    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkwmwm = Core::LinAlg::matrix_row_col_transform_gids(kwmwm, *gwmdofrowmap_, *gwmdofrowmap_);

    /****************************************************************************************
    ***                                   RIGHT-HAND SIDE                                 ***
    ****************************************************************************************/


    // We solve for the incremental Lagrange multiplier dz_. Hence,
    // we can keep the contact force terms on the right-hand side!
    //
    // r = r_effdyn,co = r_effdyn + a_f * B_co(d_(n)) * z_(n) + (1-a_f) * B_co(d^(i)_(n+1)) *
    // z^(i)_(n+1)

    // export weighted gap vector
    std::shared_ptr<Core::LinAlg::Vector<double>> gact =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivenodes_, true);
    if (gactiven_->num_global_elements())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
    Core::LinAlg::Vector<double> gactexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*gact, gactexp);

    // export stick and slip r.h.s.
    Core::LinAlg::Vector<double> stickexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linstickRHS_, stickexp);
    Core::LinAlg::Vector<double> slipexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linslipRHS_, slipexp);

    // export inactive rhs
    Core::LinAlg::Vector<double> inactiverhsexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*inactiverhs_, inactiverhsexp);

    // build constraint rhs (1)
    constrrhs->update(1.0, inactiverhsexp, 1.0);

    // build constraint rhs
    constrrhs->update(-1.0, gactexp, 1.0);
    constrrhs->update(1.0, stickexp, 1.0);
    constrrhs->update(1.0, slipexp, 1.0);
    constrrhs->replace_map(*glmdofrowmap_);

    constrrhs_ = constrrhs;  // set constraint rhs vector

    // ***************************************************************************************************
    // additional wear-rhs SLAVE
    // ***************************************************************************************************

    // export inactive wear rhs
    Core::LinAlg::Vector<double> WearCondRhsexp(*gsdofnrowmap_);
    Core::LinAlg::export_to(*wear_cond_rhs_, WearCondRhsexp);

    // export inactive wear rhs
    Core::LinAlg::Vector<double> inactiveWearRhsexp(*gsdofnrowmap_);
    Core::LinAlg::export_to(*inactive_wear_rhs_, inactiveWearRhsexp);

    wearrhs->update(1.0, WearCondRhsexp, 1.0);
    wearrhs->update(1.0, inactiveWearRhsexp, 1.0);
    wearrhs->replace_map(*gwdofrowmap_);

    wearrhs_ = wearrhs;
    // ***************************************************************************************************
    // additional wear-rhs Master
    // ***************************************************************************************************
    // export inactive wear rhs
    Core::LinAlg::Vector<double> WearCondRhsexpM(*gmdofnrowmap_);
    Core::LinAlg::export_to(wear_cond_rhs_m_->as_multi_vector(), WearCondRhsexpM.as_multi_vector());

    // export inactive wear rhs
    Core::LinAlg::Vector<double> inactiveWearRhsexpM(*gmdofnrowmap_);
    Core::LinAlg::export_to(
        inactive_wear_rhs_m_->as_multi_vector(), inactiveWearRhsexpM.as_multi_vector());

    wearrhsM->update(1.0, WearCondRhsexpM, 1.0);
    wearrhsM->update(1.0, inactiveWearRhsexpM, 1.0);
    wearrhsM->replace_map(*gwmdofrowmap_);

    wearmrhs_ = wearrhsM;
  }
  else
    FOUR_C_THROW("unknown wear algorithm!");

  //**********************************************************************
  // build and solve saddle point system
  //**********************************************************************
  if (systype == CONTACT::SystemType::saddlepoint)
  {
    // apply Dirichlet conditions to (0,0) and (0,1) blocks
    Core::LinAlg::Vector<double> zeros(*problem_dofs(), true);
    Core::LinAlg::Vector<double> rhscopy(*fd);
    Core::LinAlg::apply_dirichlet_to_system(*stiffmt, *sold, rhscopy, zeros, dirichtoggle);
    trkdz->apply_dirichlet(dirichtoggle, false);

    // row map (equals domain map) extractor
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> mapvec;
    mapvec.push_back(problem_dofs());
    mapvec.push_back(glmdofrowmap_);
    if (wearprimvar_)
    {
      mapvec.push_back(gwdofrowmap_);
      if (wearbothpv_) mapvec.push_back(gwmdofrowmap_);
    }

    // MODIFICATION OF SYSTEM:
    // =======================
    // build 2x2 soe by inserting wear blocks into lm blocks
    // this results to a well-suited block system for the iterative solvers

    if (wearprimvar_)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> trkgd = nullptr;
      std::shared_ptr<Core::LinAlg::SparseMatrix> trkgg = nullptr;

      if (!wearbothpv_)
      {
        // merged map ws + wm + z
        std::shared_ptr<Core::LinAlg::Map> gmap = nullptr;
        gmap = Core::LinAlg::merge_map(gwdofrowmap_, glmdofrowmap_, false);

        // row map (equals domain map) extractor
        Core::LinAlg::MapExtractor rowmapext(*mergedmap, gmap, problem_dofs());
        Core::LinAlg::MapExtractor dommapext(*mergedmap, gmap, problem_dofs());

        blockMat = std::make_shared<
            Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

            dommapext, rowmapext, 81, false, false);
        std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
            mat = std::dynamic_pointer_cast<
                Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                blockMat);

        trkgd = std::make_shared<Core::LinAlg::SparseMatrix>(*gmap, 100, false, true);
        trkgg = std::make_shared<Core::LinAlg::SparseMatrix>(*gmap, 100, false, true);

        Core::LinAlg::matrix_add(*trkzd, false, 1.0, *trkgd, 1.0);
        Core::LinAlg::matrix_add(*trkwd, false, 1.0, *trkgd, 1.0);

        Core::LinAlg::matrix_add(*trkzz, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkwz, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkww, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkzw, false, 1.0, *trkgg, 1.0);

        trkgd->complete(*problem_dofs(), *gmap);
        trkgg->complete(*gmap, *gmap);

        trkdz->un_complete();
        trkdz->complete(*gmap, *problem_dofs());

        mat->assign(0, 0, Core::LinAlg::DataAccess::Share, *stiffmt);
        mat->assign(0, 1, Core::LinAlg::DataAccess::Share, *trkdz);
        mat->assign(1, 0, Core::LinAlg::DataAccess::Share, *trkgd);
        mat->assign(1, 1, Core::LinAlg::DataAccess::Share, *trkgg);
        mat->complete();
      }
      // BOTH_SIDED DISCRETE WEAR
      else
      {
        // merged map ws + wm + z
        std::shared_ptr<Core::LinAlg::Map> gmap = nullptr;
        std::shared_ptr<Core::LinAlg::Map> map_dummyg =
            Core::LinAlg::merge_map(gwdofrowmap_, gwmdofrowmap_, false);
        gmap = Core::LinAlg::merge_map(map_dummyg, glmdofrowmap_, false);

        // row map (equals domain map) extractor
        Core::LinAlg::MapExtractor rowmapext(*mergedmap, gmap, problem_dofs());
        Core::LinAlg::MapExtractor dommapext(*mergedmap, gmap, problem_dofs());

        blockMat = std::make_shared<
            Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

            dommapext, rowmapext, 81, false, false);
        std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
            mat = std::dynamic_pointer_cast<
                Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                blockMat);

        trkgd = std::make_shared<Core::LinAlg::SparseMatrix>(*gmap, 100, false, true);
        trkgg = std::make_shared<Core::LinAlg::SparseMatrix>(*gmap, 100, false, true);

        Core::LinAlg::matrix_add(*trkzd, false, 1.0, *trkgd, 1.0);
        Core::LinAlg::matrix_add(*trkwd, false, 1.0, *trkgd, 1.0);
        Core::LinAlg::matrix_add(*trkwmd, false, 1.0, *trkgd, 1.0);

        Core::LinAlg::matrix_add(*trkzz, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkwz, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkww, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkzw, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkwmz, false, 1.0, *trkgg, 1.0);
        Core::LinAlg::matrix_add(*trkwmwm, false, 1.0, *trkgg, 1.0);

        trkgd->complete(*problem_dofs(), *gmap);
        trkgg->complete(*gmap, *gmap);

        trkdz->un_complete();
        trkdz->complete(*gmap, *problem_dofs());

        mat->assign(0, 0, Core::LinAlg::DataAccess::Share, *stiffmt);
        mat->assign(0, 1, Core::LinAlg::DataAccess::Share, *trkdz);
        mat->assign(1, 0, Core::LinAlg::DataAccess::Share, *trkgd);
        mat->assign(1, 1, Core::LinAlg::DataAccess::Share, *trkgg);
        mat->complete();
      }
    }
    // without wear unknowns...
    else
    {
      Core::LinAlg::MultiMapExtractor rowmapext(*mergedmap, mapvec);
      Core::LinAlg::MultiMapExtractor dommapext(*mergedmap, mapvec);

      // build block matrix for SIMPLER
      blockMat = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          dommapext, rowmapext, 81, false, false);
      std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
          mat = std::dynamic_pointer_cast<
              Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(blockMat);
      mat->assign(0, 0, Core::LinAlg::DataAccess::Share, *stiffmt);
      mat->assign(0, 1, Core::LinAlg::DataAccess::Share, *trkdz);
      mat->assign(1, 0, Core::LinAlg::DataAccess::Share, *trkzd);
      mat->assign(1, 1, Core::LinAlg::DataAccess::Share, *trkzz);
      mat->complete();
    }

    // we also need merged rhs here
    Core::LinAlg::Vector<double> fresmexp(*mergedmap);
    Core::LinAlg::export_to(*fd, fresmexp);
    mergedrhs->update(1.0, fresmexp, 1.0);
    Core::LinAlg::Vector<double> constrexp(*mergedmap);
    Core::LinAlg::export_to(*constrrhs, constrexp);
    mergedrhs->update(1.0, constrexp, 1.0);

    // add wear rhs
    if (wearprimvar_)
    {
      Core::LinAlg::Vector<double> wearexp(*mergedmap);
      Core::LinAlg::export_to(*wearrhs, wearexp);
      mergedrhs->update(1.0, wearexp, 1.0);

      if (wearbothpv_)
      {
        Core::LinAlg::Vector<double> wearexpM(*mergedmap);
        Core::LinAlg::export_to(*wearrhsM, wearexpM);
        mergedrhs->update(1.0, wearexpM, 1.0);
      }
    }

    // apply Dirichlet B.C. to mergedrhs and mergedsol
    Core::LinAlg::Vector<double> dirichtoggleexp(*mergedmap);
    Core::LinAlg::export_to(dirichtoggle, dirichtoggleexp);
    Core::LinAlg::apply_dirichlet_to_system(*mergedsol, *mergedrhs, *mergedzeros, dirichtoggleexp);

    // return references to solution and rhs vector
    blocksol = mergedsol;
    blockrhs = mergedrhs;
    return;
  }

  //**********************************************************************
  // invalid system types
  //**********************************************************************
  else
    FOUR_C_THROW("Invalid system type in BuildSaddlePointProblem");

  return;
}

/*------------------------------------------------------------------------*
 | Update internal member variables after saddle point solve wiesner 11/14|
 *------------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::update_displacements_and_l_mincrements(
    std::shared_ptr<Core::LinAlg::Vector<double>> sold,
    std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol)
{
  //**********************************************************************
  // extract results for displacement and LM increments
  //**********************************************************************
  Core::LinAlg::Vector<double> sollm(*glmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Vector<double>> solw = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> solwm = nullptr;

  if (wearprimvar_) solw = std::make_shared<Core::LinAlg::Vector<double>>(*gwdofrowmap_);
  if (wearbothpv_) solwm = std::make_shared<Core::LinAlg::Vector<double>>(*gwmdofrowmap_);

  // initialize merged system (matrix, rhs, sol);
  std::shared_ptr<Core::LinAlg::Map> mergedmap = nullptr;
  if (!wearprimvar_)
  {
    mergedmap = Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
  }
  else if (wearprimvar_ and !wearbothpv_)
  {
    std::shared_ptr<Core::LinAlg::Map> map_dummy =
        Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(map_dummy, gwdofrowmap_, false);
  }
  else
  {
    std::shared_ptr<Core::LinAlg::Map> map_dummy =
        Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(map_dummy, gwdofrowmap_, false);
    mergedmap = Core::LinAlg::merge_map(mergedmap, gwmdofrowmap_, false);  // slave + master wear
  }

  Core::LinAlg::MapExtractor mapextd(*mergedmap, problem_dofs(), glmdofrowmap_);
  Core::LinAlg::MapExtractor mapextlm(*mergedmap, glmdofrowmap_, glmdofrowmap_);
  mapextd.extract_cond_vector(*blocksol, *sold);
  mapextlm.extract_cond_vector(*blocksol, sollm);
  sollm.replace_map(*gsdofrowmap_);

  if (wearprimvar_)
  {
    Core::LinAlg::MapExtractor mapextw(*mergedmap, gwdofrowmap_, glmdofrowmap_);
    mapextw.extract_cond_vector(*blocksol, *solw);
    solw->replace_map(*gsdofnrowmap_);
  }
  if (wearbothpv_)
  {
    Core::LinAlg::MapExtractor mapextwm(*mergedmap, gwmdofrowmap_, glmdofrowmap_);
    mapextwm.extract_cond_vector(*blocksol, *solwm);
    solwm->replace_map(*gmdofnrowmap_);
  }

  if (is_self_contact())
  // for self contact, slave and master sets may have changed,
  // thus we have to reinitialize the LM vector map
  {
    zincr_ = std::make_shared<Core::LinAlg::Vector<double>>(sollm);
    Core::LinAlg::export_to(*z_, *zincr_);  // change the map of z_
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*zincr_);
    zincr_->update(1.0, sollm, 0.0);  // save sollm in zincr_
    z_->update(1.0, *zincr_, 1.0);    // update z_
  }
  else
  {
    zincr_->update(1.0, sollm, 0.0);
    z_->update(1.0, *zincr_, 1.0);

    if (wearprimvar_)
    {
      wincr_->update(1.0, *solw, 0.0);
      w_->update(1.0, *wincr_, 1.0);

      if (wearbothpv_)
      {
        wmincr_->update(1.0, *solwm, 0.0);
        wm_->update(1.0, *wmincr_, 1.0);
      }
    }
  }
  return;
}

/*-----------------------------------------------------------------------*
|  Reset the wear vector                                                 |
*-----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::reset_wear() { wearoutput_->put_scalar(0.0); }

/*-----------------------------------------------------------------------*
|  Output de-weighted wear vector                             farah 09/14|
*-----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::output_wear()
{
  //***********************************************
  //                 primvar wear
  //***********************************************
  if (wearprimvar_)
  {
    // Wear post processing only for internal state variable approach necessary
    return;
  }
  //***********************************************
  //                 weighted wear (is)
  //***********************************************
  else
  {
    // only for dual/pg Lagrange multiplier so far
    // diagonality of mortar matrix D is assumed
    auto shapefcn = Teuchos::getIntegralValue<Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
    if (shapefcn == Mortar::shape_standard)
      FOUR_C_THROW("Evaluation of wear only for dual shape functions so far.");

    // vectors
    Core::LinAlg::Vector<double> wear_vector(*gsdofrowmap_);
    Core::LinAlg::Vector<double> real_weara(*gactivedofs_, true);
    Core::LinAlg::Vector<double> real_wear(*gsdofrowmap_, true);

    // multiply the wear with its normal direction and store in wear_vector
    // loop over all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      // FIRST: get the wear values and the normal directions for the interface
      // loop over all slave row nodes on the current interface
      for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
      {
        int gid = interface_[i]->slave_row_nodes()->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

        // be aware of problem dimension
        int dim = n_dim();
        int numdof = frinode->num_dof();
        if (dim != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

        // nodal normal vector and wear
        double nn[3];
        double wear = 0.0;

        for (int j = 0; j < 3; ++j) nn[j] = frinode->mo_data().n()[j];
        wear = frinode->wear_data().weighted_wear();

        // find indices for DOFs of current node in Core::LinAlg::Vector<double>
        // and put node values (normal and tangential stress components) at these DOFs
        std::vector<int> locindex(dim);

        for (int dof = 0; dof < dim; ++dof)
        {
          locindex[dof] = (wear_vector.get_map()).lid(frinode->dofs()[dof]);
          (wear_vector).get_values()[locindex[dof]] = wear * nn[dof];
        }
      }
    }

    // extract active parts of D matrix
    // matrices, maps
    std::shared_ptr<Core::LinAlg::SparseMatrix> data, dai, dia, dii;
    std::shared_ptr<Core::LinAlg::Map> gidofs;

    // ****************************************************************
    // split the matrix
    // why is here an empty gidofs map instead of a full map used???
    // ****************************************************************
    Core::LinAlg::split_matrix2x2(
        dmatrix_, gactivedofs_, gidofs, gactivedofs_, gidofs, data, dai, dia, dii);

    // extract active parts of wear vector
    std::shared_ptr<Core::LinAlg::Vector<double>> wear_vectora =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_, true);
    std::shared_ptr<Core::LinAlg::Vector<double>> wear_vectori =
        std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

    // split the vector
    Core::LinAlg::split_vector(
        *gsdofrowmap_, wear_vector, gactivedofs_, wear_vectora, gidofs, wear_vectori);

    /* approx. undo the weighting of the wear by solving D * w = w~
     * dmatrix_ * real_wear = wear_
     *
     * Note: Due to dual shape functions, data is diagonal. So we don't need an actual solver.
     *       We rather divide by the diagonal element of data.
     */
    if (gactivedofs_->num_global_elements())
    {
      // number of active DOFs on this proc
      const int lNumActiveDOFs = wear_vectora->local_length();

      // extract diagonal of data
      Core::LinAlg::Vector<double> diagD(*gactivedofs_);
      data->extract_diagonal_copy(diagD);

      // solve by dividing through diagonal elements of data. Do not divide by 0.
      for (int i = 0; i < lNumActiveDOFs; ++i)
        if ((diagD).local_values_as_span()[i] != 0.0)
          (real_weara).get_values()[i] =
              wear_vectora->local_values_as_span()[i] / (diagD).local_values_as_span()[i];
    }

    Core::LinAlg::Vector<double> real_wearexp(*gsdofrowmap_);
    Core::LinAlg::export_to(real_weara, real_wearexp);
    real_wear.update(1.0, real_wearexp, 0.0);

    // different wear coefficients on both sides...
    const double wearcoeff_s = params().get<double>("WEARCOEFF", 0.0);
    const double wearcoeff_m = params().get<double>("WEARCOEFF_MASTER", 0.0);
    if (wearcoeff_s < 1e-12) FOUR_C_THROW("wcoeff negative!!!");

    const double fac = wearcoeff_s / (wearcoeff_s + wearcoeff_m);

    // copy the local part of real_wear into wearoutput_
    for (int i = 0; i < (int)gsdofrowmap_->num_my_elements(); ++i)
    {
      const int gid = gsdofrowmap_->my_global_elements()[i];
      const double tmp = (real_wear).local_values_as_span()[real_wear.get_map().lid(gid)];
      (*wearoutput_).get_values()[wearoutput_->get_map().lid(gid)] = tmp * fac;
    }

    /**********************************************************************
     * Here the wearoutput_ - vector is the unweighted ("real") wearvector.
     * To calculate the wearvector for the master surface we transform
     * the slavewear vector via w_2~ = M^T * D^-1 * w~. In addition, we
     * unweight the resulting vector by D_2^-1*w_2~ and get the final
     * unweighted wear vector.
     **********************************************************************/
    if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(params(), "WEAR_SIDE") ==
        Inpar::Wear::wear_both)
    {
      // different wear coefficients on both sides...
      double wearcoeff_s = params().get<double>("WEARCOEFF", 0.0);
      double wearcoeff_m = params().get<double>("WEARCOEFF_MASTER", 0.0);
      if (wearcoeff_s < 1e-12) FOUR_C_THROW("wcoeff negative!!!");

      double fac = wearcoeff_m / (wearcoeff_s + wearcoeff_m);

      // extract involved parts of d2 matrix
      // matrices, maps - i: involved ; n: non-involved
      std::shared_ptr<Core::LinAlg::SparseMatrix> d2ii, d2in, d2ni, d2nn;
      std::shared_ptr<Core::LinAlg::Map> gndofs;  // non-involved dofs

      Core::LinAlg::split_matrix2x2(
          d2matrix_, gminvolveddofs_, gndofs, gminvolveddofs_, gndofs, d2ii, d2in, d2ni, d2nn);

      Core::LinAlg::Vector<double> wear_master(*gmdofrowmap_);
      Core::LinAlg::Vector<double> real_wear2(*gmdofrowmap_);

      // now we calc the weighted wear on the master surface
      mmatrix_->multiply(true, real_wear, wear_master);

      // extract involved parts of wear vector
      Core::LinAlg::Vector<double> wear2_real(*gminvolveddofs_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wear2_vectori =
          std::make_shared<Core::LinAlg::Vector<double>>(*gminvolveddofs_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wear2_vectorn =
          std::make_shared<Core::LinAlg::Vector<double>>(*gndofs);

      // split the vector
      Core::LinAlg::split_vector(
          *gmdofrowmap_, wear_master, gminvolveddofs_, wear2_vectori, gndofs, wear2_vectorn);

      /* Note: Due to dual shape functions, d2ii is diagonal. So we don't need an actual solver.
       *       We rather divide by the diagonal element of d2ii.
       */
      if (gminvolveddofs_->num_global_elements())
      {
        // number of active DOFs on this proc
        const int lNumActiveDOFs = wear2_vectori->local_length();

        // extract diagonal of d2ii
        Core::LinAlg::Vector<double> diagD(*wear2_vectori);
        d2ii->extract_diagonal_copy(diagD);

        // solve by dividing through diagonal elements of data. Do not divide by 0.
        for (int i = 0; i < lNumActiveDOFs; ++i)
          if ((diagD).local_values_as_span()[i] != 0.0)
            (wear2_real).get_values()[i] =
                wear2_vectori->local_values_as_span()[i] / (diagD).local_values_as_span()[i];
      }

      Core::LinAlg::Vector<double> real_wear2exp(*gmdofrowmap_);
      Core::LinAlg::export_to(wear2_real, real_wear2exp);
      real_wear2.update(1.0, real_wear2exp, 0.0);

      // copy the local part of real_wear into wearoutput_
      for (int i = 0; i < (int)gmdofrowmap_->num_my_elements(); ++i)
      {
        int gid = gmdofrowmap_->my_global_elements()[i];
        double tmp = (real_wear2).local_values_as_span()[real_wear2.get_map().lid(gid)];
        (*wearoutput2_).get_values()[wearoutput2_->get_map().lid(gid)] =
            -(tmp * fac);  // negative sign because on other interface side
        //--> this Wear-vector (defined on master side) is along slave-side normal field!
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  write restart information for contact                     popp 03/08|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::do_write_restart(
    std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
    bool forcedrestart) const
{
  // TODO: extend this function to forcedrestart -- write output for
  // last converged wear... see contact_lagrange_strategy.cpp!

  // initialize
  std::shared_ptr<Core::LinAlg::Vector<double>> activetoggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);
  std::shared_ptr<Core::LinAlg::Vector<double>> sliptoggle, weightedwear, realwear;

  // write toggle
  restart_vectors["activetoggle"] = activetoggle;
  if (friction_)
  {
    sliptoggle = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);
    restart_vectors["sliptoggle"] = sliptoggle;
  }

  if (weightedwear_)
  {
    weightedwear = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);  // weighted
    restart_vectors["weightedwear"] = weightedwear;
    realwear = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);  // unweighted
    restart_vectors["realwear"] = realwear;
  }

  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // loop over all slave nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->slave_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
      int dof = (activetoggle->get_map()).lid(gid);

      // set value active / inactive in toggle vector
      if (cnode->active()) (*activetoggle).get_values()[dof] = 1;

      // set value slip / stick in the toggle vector
      if (friction_)
      {
        CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(cnode);
        if (frinode->fri_data().slip()) (*sliptoggle).get_values()[dof] = 1;
        if (weightedwear_)
        {
          (*weightedwear).get_values()[dof] = frinode->wear_data().weighted_wear();
        }
      }
    }
  }

  if (friction_ and weightedwear_) realwear = wearoutput_;

  return;
}

/*----------------------------------------------------------------------*
 | Recovery method                                           farah 10/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // shape function and system types
  auto shapefcn = Teuchos::getIntegralValue<Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL) + WEAR DISCR (DUAL)
  //**********************************************************************
  //**********************************************************************
  if ((systype == CONTACT::SystemType::condensed) && wearprimvar_)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Mortar::shape_dual && shapefcn != Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    // **********************************************
    // LAGR MULT RECOVERING
    // extract slave displacements from disi
    Core::LinAlg::Vector<double> disis(*gsdofrowmap_);
    if (gsdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disis);

    // extract master displacements from disi
    Core::LinAlg::Vector<double> disim(*gmdofrowmap_);
    if (gmdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disim);

    // extract other displacements from disi
    Core::LinAlg::Vector<double> disin(*gndofrowmap_);
    if (gndofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disin);

    // condensation has been performed for active LM only,
    // thus we construct a modified invd matrix here which
    // only contains the active diagonal block
    // (this automatically renders the inactive LM to be zero)
    std::shared_ptr<Core::LinAlg::SparseMatrix> invda;
    std::shared_ptr<Core::LinAlg::Map> tempmap;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2, tempmtx3;
    Core::LinAlg::split_matrix2x2(
        invd_, gactivedofs_, tempmap, gactivedofs_, tempmap, invda, tempmtx1, tempmtx2, tempmtx3);
    Core::LinAlg::SparseMatrix invdmod(*gsdofrowmap_, 10);
    Core::LinAlg::matrix_add(*invda, false, 1.0, invdmod, 1.0);
    invdmod.complete();

    // approximate update
    // invdmod->Multiply(false,*fs_,*z_);

    // full update
    // zincr_->Update(1.0,*z_,0.0); // z_i

    z_->update(1.0, *fs_, 0.0);
    Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
    kss_->multiply(false, disis, mod);
    z_->update(-1.0, mod, 1.0);
    ksm_->multiply(false, disim, mod);
    z_->update(-1.0, mod, 1.0);
    ksn_->multiply(false, disin, mod);
    z_->update(-1.0, mod, 1.0);
    dold_->multiply(true, *zold_, mod);
    z_->update(-alphaf_, mod, 1.0);
    Core::LinAlg::Vector<double> zcopy(*z_);
    invdmod.multiply(true, zcopy, *z_);
    z_->scale(1 / (1 - alphaf_));

    // zincr_->Update(-1.0,*z_,1.0); // zi-zi+1

    // **********************************************
    // WEAR RECOVERING
    // wincr_ up to w_
    // extract active slave displacements from disi
    Core::LinAlg::Vector<double> disia(*gactivedofs_);
    if (gactivedofs_->num_global_elements()) Core::LinAlg::export_to(*disi, disia);

    // extract inactive slave displacements from disi
    Core::LinAlg::Vector<double> disii(*gidofs_);
    if (gidofs_->num_global_elements()) Core::LinAlg::export_to(*disi, disii);

    // recovering just for slipnodes!
    if (gslipnodes_->num_global_elements() > 0)
    {
      // rhs
      Core::LinAlg::Vector<double> fwexp(*gsdofnrowmap_);
      Core::LinAlg::export_to(*fw_, fwexp);
      wincr_->update(1.0, fwexp, 0.0);

      Core::LinAlg::Vector<double> modw(*gsdofnrowmap_);

      // neutral part
      dnblock_ = Core::LinAlg::matrix_row_transform_gids(*dnblock_, *gsdofnrowmap_);
      dnblock_->multiply(false, disin, modw);
      wincr_->update(1.0, modw, 1.0);

      // master part
      dmblock_ = Core::LinAlg::matrix_row_transform_gids(*dmblock_, *gsdofnrowmap_);
      dmblock_->multiply(false, disim, modw);
      wincr_->update(1.0, modw, 1.0);

      // active part (stick and slip)
      dablock_ = Core::LinAlg::matrix_row_transform_gids(*dablock_, *gsdofnrowmap_);
      dablock_->multiply(false, disia, modw);
      wincr_->update(1.0, modw, 1.0);

      // inactive part
      if (gidofs_->num_global_elements() > 0)
      {
        diblock_ = Core::LinAlg::matrix_row_transform_gids(*diblock_, *gsdofnrowmap_);
        diblock_->multiply(false, disii, modw);
        wincr_->update(1.0, modw, 1.0);
      }
    }
    else
    {
      wincr_->put_scalar(0.0);
    }
    // wear rhs  for inactive/stick nodes
    Core::LinAlg::Vector<double> wrhsexp(*gsdofnrowmap_);
    Core::LinAlg::export_to(*inactive_wear_rhs_, wrhsexp);
    wincr_->update(1.0, wrhsexp, 1.0);
    w_->update(1.0, *wincr_, 1.0);
  }
  //**********************************************************************
  //**********************************************************************
  // CASE B: CONDENSED SYSTEM (DUAL) + WEAR IMPLICIT/EXPLICIT
  //**********************************************************************
  //**********************************************************************
  else if ((systype == CONTACT::SystemType::condensed) && !wearprimvar_)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Mortar::shape_dual && shapefcn != Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    // extract slave displacements from disi
    Core::LinAlg::Vector<double> disis(*gsdofrowmap_);
    if (gsdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disis);

    // extract master displacements from disi
    Core::LinAlg::Vector<double> disim(*gmdofrowmap_);
    if (gmdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disim);

    // extract other displacements from disi
    Core::LinAlg::Vector<double> disin(*gndofrowmap_);
    if (gndofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disin);

    // condensation has been performed for active LM only,
    // thus we construct a modified invd matrix here which
    // only contains the active diagonal block
    // (this automatically renders the inactive LM to be zero)
    std::shared_ptr<Core::LinAlg::SparseMatrix> invda;
    std::shared_ptr<Core::LinAlg::Map> tempmap;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2, tempmtx3;
    Core::LinAlg::split_matrix2x2(
        invd_, gactivedofs_, tempmap, gactivedofs_, tempmap, invda, tempmtx1, tempmtx2, tempmtx3);
    Core::LinAlg::SparseMatrix invdmod(*gsdofrowmap_, 10);
    Core::LinAlg::matrix_add(*invda, false, 1.0, invdmod, 1.0);
    invdmod.complete();

    /**********************************************************************/
    /* Update Lagrange multipliers z_n+1                                  */
    /**********************************************************************/

    // for self contact, slave and master sets may have changed,
    // thus we have to export the products Dold * zold and Mold^T * zold to fit
    if (is_self_contact())
    {
      // approximate update
      // z_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*gsdofrowmap_));
      // invdmod->Multiply(false,*fs_,*z_);

      // full update
      z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
      z_->update(1.0, *fs_, 0.0);
      Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
      kss_->multiply(false, disis, mod);
      z_->update(-1.0, mod, 1.0);
      ksm_->multiply(false, disim, mod);
      z_->update(-1.0, mod, 1.0);
      ksn_->multiply(false, disin, mod);
      z_->update(-1.0, mod, 1.0);
      Core::LinAlg::Vector<double> mod2((dold_->row_map()));
      if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, mod2);
      Core::LinAlg::Vector<double> mod3((dold_->row_map()));
      dold_->multiply(true, mod2, mod3);
      Core::LinAlg::Vector<double> mod4(*gsdofrowmap_);
      if (gsdofrowmap_->num_global_elements()) Core::LinAlg::export_to(mod3, mod4);
      z_->update(-alphaf_, mod4, 1.0);
      Core::LinAlg::Vector<double> zcopy(*z_);
      invdmod.multiply(true, zcopy, *z_);
      z_->scale(1 / (1 - alphaf_));
    }
    else
    {
      // approximate update
      // invdmod->Multiply(false,*fs_,*z_);

      // full update
      z_->update(1.0, *fs_, 0.0);
      Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
      kss_->multiply(false, disis, mod);
      z_->update(-1.0, mod, 1.0);
      ksm_->multiply(false, disim, mod);
      z_->update(-1.0, mod, 1.0);
      ksn_->multiply(false, disin, mod);
      z_->update(-1.0, mod, 1.0);
      dold_->multiply(true, *zold_, mod);
      z_->update(-alphaf_, mod, 1.0);
      Core::LinAlg::Vector<double> zcopy(*z_);
      invdmod.multiply(true, zcopy, *z_);
      z_->scale(1 / (1 - alphaf_));
    }
  }
  //**********************************************************************
  //**********************************************************************
  // CASE C: SADDLE POINT SYSTEM
  //**********************************************************************
  //**********************************************************************
  else
  {
    // do nothing (z_ was part of solution already)
  }

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate);

  if (wearprimvar_)
  {
    store_nodal_quantities(Mortar::StrategyBase::wupdate);
  }

  if (wearbothpv_)
  {
    store_nodal_quantities(Mortar::StrategyBase::wmupdate);
  }

  return;
}

/*----------------------------------------------------------------------*
 | parallel redistribution                                   popp 09/10 |
 *----------------------------------------------------------------------*/
bool Wear::LagrangeStrategyWear::redistribute_contact(
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  // get out of here if parallel redistribution is switched off
  // or if this is a single processor (serial) job
  if (!parallel_redistribution_status() || Core::Communication::num_mpi_ranks(get_comm()) == 1)
    return false;

  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->is_redistributed() = false;
  }

  // decide whether redistribution should be applied or not
  double taverage = 0.0;
  double eaverage = 0;
  bool doredist = false;
  const double max_balance =
      params().sublist("PARALLEL REDISTRIBUTION").get<double>("MAX_BALANCE_EVAL_TIME");
  const double max_ele_unbalance =
      params().sublist("PARALLEL REDISTRIBUTION").get<double>("MAX_BALANCE_SLAVE_ELES");

  //**********************************************************************
  // (1) static redistribution: ONLY at time t=0 or after restart
  // (both cases can be identified via empty unbalance vectors)
  //**********************************************************************
  if (which_parallel_redistribution() == Mortar::ParallelRedist::redist_static)
  {
    // this is the first time step (t=0) or restart
    if ((int)unbalanceEvaluationTime_.size() == 0 && (int)unbalanceNumSlaveElements_.size() == 0)
    {
      // do redistribution
      doredist = true;
    }

    // this is a regular time step (neither t=0 nor restart)
    else
    {
      // compute average balance factors of last time step
      for (int k = 0; k < (int)unbalanceEvaluationTime_.size(); ++k)
        taverage += unbalanceEvaluationTime_[k];
      taverage /= (int)unbalanceEvaluationTime_.size();
      for (int k = 0; k < (int)unbalanceNumSlaveElements_.size(); ++k)
        eaverage += unbalanceNumSlaveElements_[k];
      eaverage /= (int)unbalanceNumSlaveElements_.size();

      // delete balance factors of last time step
      unbalanceEvaluationTime_.resize(0);
      unbalanceNumSlaveElements_.resize(0);

      // no redistribution
      doredist = false;
    }
  }

  //**********************************************************************
  // (2) dynamic redistribution: whenever system is out of balance
  //**********************************************************************
  else if (which_parallel_redistribution() == Mortar::ParallelRedist::redist_dynamic)
  {
    // this is the first time step (t=0) or restart
    if ((int)unbalanceEvaluationTime_.size() == 0 && (int)unbalanceNumSlaveElements_.size() == 0)
    {
      // do redistribution
      doredist = true;
    }

    // this is a regular time step (neither t=0 nor restart)
    else
    {
      // compute average balance factors of last time step
      for (int k = 0; k < (int)unbalanceEvaluationTime_.size(); ++k)
        taverage += unbalanceEvaluationTime_[k];
      taverage /= (int)unbalanceEvaluationTime_.size();
      for (int k = 0; k < (int)unbalanceNumSlaveElements_.size(); ++k)
        eaverage += unbalanceNumSlaveElements_[k];
      eaverage /= (int)unbalanceNumSlaveElements_.size();

      // delete balance factors of last time step
      unbalanceEvaluationTime_.resize(0);
      unbalanceNumSlaveElements_.resize(0);

      /* Decide on redistribution
       *
       * We allow a maximum value of the balance measure in the system as defined in the input
       * parameter MAX_BALANCE_EVAL_TIME, i.e. the maximum local processor workload and the
       * minimum local processor workload for mortar evaluation of all interfaces may not differ
       * by more than (MAX_BALANCE_EVAL_TIME - 1.0)*100%)
       *
       * Moreover, we redistribute if in the majority of iteration steps of the last time step
       * there has been an unbalance in element distribution.
       */
      if (taverage >= max_balance || eaverage >= max_ele_unbalance) doredist = true;
    }
  }

  // print balance information to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "**********************************************************" << std::endl;
    if (taverage > 0)
    {
      printf("Parallel balance (time): %e (limit %e) \n", taverage, max_balance);
      printf("Parallel balance (eles): %e (limit %e) \n", eaverage, 0.5);
    }
    else
      printf("Parallel balance: t=0/restart \n");
    std::cout << "**********************************************************" << std::endl;
  }

  // get out of here if simulation is still in balance
  if (!doredist) return false;

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  // set old and current displacement state
  // (needed for search within redistribution)
  set_state(Mortar::state_new_displacement, *dis);
  set_state(Mortar::state_old_displacement, *dis);

  // parallel redistribution of all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // redistribute optimally among procs
    interface_[i]->redistribute();

    // call fill complete again
    interface_[i]->fill_complete(Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true, maxdof_);

    // print new parallel distribution
    interface_[i]->print_parallel_distribution();

    // re-create binary search tree
    interface_[i]->create_search_tree();

    // set bool for redistribution
    interface_[i]->is_redistributed() = true;
  }

  // re-setup strategy with redistributed=TRUE, init=FALSE
  setup(true, false);
  setup_wear(true, false);

  // time measurement
  Core::Communication::barrier(get_comm());
  double t_end = Teuchos::Time::wallTime() - t_start;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\nTime for parallel redistribution..............." << std::scientific
              << std::setprecision(6) << t_end << " secs\n"
              << std::endl;

  return doredist;
}

/*----------------------------------------------------------------------*
 |  read restart information for contact                      popp 03/08|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::do_read_restart(
    Core::IO::DiscretizationReader& reader, std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // check whether this is a restart with contact of a previously
  // non-contact simulation run (if yes, we have to be careful not
  // to try to read certain, in this case non-existing, vectors
  // such as the activetoggle or sliptoggle vectors, but rather
  // initialize the restart active and slip sets as being empty)
  const bool restartwithcontact = params().get<bool>("RESTART_WITH_CONTACT");

  // set restart displacement state
  set_state(Mortar::state_new_displacement, *dis);
  set_state(Mortar::state_old_displacement, *dis);

  // evaluate interface and restart mortar quantities
  // in the case of SELF CONTACT, also re-setup master/slave maps
  initialize_mortar();
  initialize_and_evaluate_interface();
  assemble_mortar();

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    // modify dmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    dmatrix_ = temp;
  }

  // read restart information on active set and slip set (leave sets empty
  // if this is a restart with contact of a non-contact simulation run)
  std::shared_ptr<Core::LinAlg::Vector<double>> activetoggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);
  if (!restartwithcontact) reader.read_vector(activetoggle, "activetoggle");

  // friction
  std::shared_ptr<Core::LinAlg::Vector<double>> sliptoggle;
  std::shared_ptr<Core::LinAlg::Vector<double>> weightedwear;
  std::shared_ptr<Core::LinAlg::Vector<double>> realwear;

  if (friction_)
  {
    sliptoggle = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);
    if (!restartwithcontact) reader.read_vector(sliptoggle, "sliptoggle");
  }

  // wear
  if (weightedwear_)
  {
    weightedwear = std::make_shared<Core::LinAlg::Vector<double>>(*gsnoderowmap_);
    reader.read_vector(weightedwear, "weightedwear");

    realwear = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    reader.read_vector(realwear, "realwear");
  }

  // store restart information on active set and slip set
  // into nodes, therefore first loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // loop over all slave nodes on the current interface
    for (int j = 0; j < (interface_[i]->slave_row_nodes())->num_my_elements(); ++j)
    {
      int gid = (interface_[i]->slave_row_nodes())->gid(j);
      int dof = (activetoggle->get_map()).lid(gid);

      if (activetoggle->local_values_as_span()[dof] == 1)
      {
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

        // set value active / inactive in cnode
        cnode->active() = true;

        if (friction_)
        {
          // set value stick / slip in cnode
          // set wear value
          if (sliptoggle->local_values_as_span()[dof] == 1)
            dynamic_cast<CONTACT::FriNode*>(cnode)->fri_data().slip() = true;
          if (weightedwear_)
            dynamic_cast<CONTACT::FriNode*>(cnode)->wear_data().weighted_wear() =
                weightedwear->local_values_as_span()[dof];
        }
      }
    }
  }

  if (friction_ and weightedwear_) wearoutput_ = realwear;


  // read restart information on Lagrange multipliers
  z_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  zold_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  if (!restartwithcontact) reader.read_vector(z_, "lagrmultold");
  if (!restartwithcontact) reader.read_vector(data().old_lm_ptr(), "lagrmultold");

  // Lagrange multiplier increment is always zero (no restart value to be read)
  zincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);

  // store restart information on Lagrange multipliers into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmcurrent);
  store_nodal_quantities(Mortar::StrategyBase::lmold);

  // TODO: same procedure for discrete wear...

  // only for Uzawa Augmented strategy
  // TODO: this should be moved to contact_penalty_strategy
  auto st = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params(), "STRATEGY");
  if (st == CONTACT::SolvingStrategy::uzawa)
  {
    zuzawa_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    if (!restartwithcontact) reader.read_vector(data().lm_uzawa_ptr(), "lagrmultold");
    store_nodal_quantities(Mortar::StrategyBase::lmuzawa);
  }

  // store restart Mortar quantities
  store_dm("old");

  if (friction_)
  {
    store_nodal_quantities(Mortar::StrategyBase::activeold);
    store_to_old(Mortar::StrategyBase::dm);
  }

  // (re)setup active global Core::LinAlg::Maps
  gactivenodes_ = nullptr;
  gactivedofs_ = nullptr;
  gactiven_ = nullptr;
  gactivet_ = nullptr;
  gslipnodes_ = nullptr;
  gslipdofs_ = nullptr;
  gslipt_ = nullptr;

  // update active sets of all interfaces
  // (these maps are NOT allowed to be overlapping !!!)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->build_active_set();
    gactivenodes_ = Core::LinAlg::merge_map(gactivenodes_, interface_[i]->active_nodes(), false);
    gactivedofs_ = Core::LinAlg::merge_map(gactivedofs_, interface_[i]->active_dofs(), false);
    gactiven_ = Core::LinAlg::merge_map(gactiven_, interface_[i]->active_n_dofs(), false);
    gactivet_ = Core::LinAlg::merge_map(gactivet_, interface_[i]->active_t_dofs(), false);
    if (friction_)
    {
      gslipnodes_ = Core::LinAlg::merge_map(gslipnodes_, interface_[i]->slip_nodes(), false);
      gslipdofs_ = Core::LinAlg::merge_map(gslipdofs_, interface_[i]->slip_dofs(), false);
      gslipt_ = Core::LinAlg::merge_map(gslipt_, interface_[i]->slip_t_dofs(), false);
    }
  }

  // update flags for global contact status
  if (gactivenodes_->num_global_elements())
  {
    isincontact_ = true;
    wasincontact_ = true;
    wasincontactlts_ = true;
  }

  // evaluate relative movement (jump)
  // needed because it is not called in the predictor of the
  // lagrange multiplier strategy
  evaluate_relative_movement();

  // reset unbalance factors for redistribution
  // (during restart the interface has been evaluated once)
  unbalanceEvaluationTime_.resize(0);
  unbalanceNumSlaveElements_.resize(0);

  return;
}

/*----------------------------------------------------------------------*
 |  Update active set and check for convergence (public)     farah 02/16|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::update_active_set_semi_smooth(const bool firstStepPredictor)
{
  // call base routine
  CONTACT::LagrangeStrategy::update_active_set_semi_smooth(firstStepPredictor);

  // for both-sided wear
  gminvolvednodes_ = nullptr;
  gminvolveddofs_ = nullptr;

  // for wear with own discretization
  gslipn_ = nullptr;
  gwinact_ = nullptr;
  gmslipn_ = nullptr;
  gwminact_ = nullptr;
  gmslipnodes_ = nullptr;
  gmactivenodes_ = nullptr;

  // update active sets of all interfaces
  // (these maps are NOT allowed to be overlapping !!!)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // for both-sided wear
    if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(scontact_, "WEAR_SIDE") ==
            Inpar::Wear::wear_both and
        wearprimvar_ == false)
    {
      gminvolvednodes_ =
          Core::LinAlg::merge_map(gminvolvednodes_, interface_[i]->involved_nodes(), false);
      gminvolveddofs_ =
          Core::LinAlg::merge_map(gminvolveddofs_, interface_[i]->involved_dofs(), false);
    }

    if (wearprimvar_ and wearbothpv_)
    {
      interface_[i]->build_active_set_master();
      gmslipn_ = Core::LinAlg::merge_map(gmslipn_, interface_[i]->slip_master_n_dofs(), false);
      gmslipnodes_ =
          Core::LinAlg::merge_map(gmslipnodes_, interface_[i]->slip_master_nodes(), false);
      gmactivenodes_ =
          Core::LinAlg::merge_map(gmactivenodes_, interface_[i]->active_master_nodes(), false);
    }
  }  // end interface loop

  if (wearprimvar_)
  {
    gslipn_ = Core::LinAlg::split_map(*gslipdofs_, *gslipt_);
    if (sswear_)
      gwinact_ = Core::LinAlg::split_map(*gsdofnrowmap_, *gactiven_);
    else
      gwinact_ = Core::LinAlg::split_map(*gsdofnrowmap_, *gslipn_);

    if (wearbothpv_) gwminact_ = Core::LinAlg::split_map(*gmdofnrowmap_, *gmslipn_);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Update Wear rhs for seq. staggered partitioned sol.      farah 11/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::update_wear_discret_iterate(bool store)
{
  if (store)
  {
    store_nodal_quantities(Mortar::StrategyBase::wold);
    if (wearbothpv_) store_nodal_quantities(Mortar::StrategyBase::wmold);
  }
  else
  {
    // loop over all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      for (int j = 0; j < (int)interface_[i]->slave_col_nodes()->num_my_elements(); ++j)
      {
        int gid = interface_[i]->slave_col_nodes()->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

        // reset
        cnode->wear_data().wcurr()[0] = 0.0;
        cnode->wear_data().wold()[0] = 0.0;
        cnode->wear_data().waccu()[0] = 0.0;
      }
      if (wearbothpv_)
      {
        const std::shared_ptr<Core::LinAlg::Map> masternodes =
            Core::LinAlg::allreduce_e_map(*(interface_[i]->master_row_nodes()));

        for (int j = 0; j < (int)masternodes->num_my_elements(); ++j)
        {
          int gid = masternodes->gid(j);
          Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
          if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
          CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

          // reset
          cnode->wear_data().wcurr()[0] = 0.0;
          cnode->wear_data().wold()[0] = 0.0;
          cnode->wear_data().waccu()[0] = 0.0;
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Update Wear for different time scales                    farah 12/13|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::update_wear_discret_accumulation()
{
  if (weartimescales_) store_nodal_quantities(Mortar::StrategyBase::wupdateT);

  return;
}

/*----------------------------------------------------------------------*
 |  Update and output contact at end of time step            farah 02/16|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  // call base routine
  CONTACT::AbstractStrategy::update(dis);

  // wear: store history values
  if (weightedwear_) store_nodal_quantities(Mortar::StrategyBase::weightedwear);

  return;
}


/*----------------------------------------------------------------------*
 |  Store wear data                                          farah 02/16|
 *----------------------------------------------------------------------*/
void Wear::LagrangeStrategyWear::store_nodal_quantities(Mortar::StrategyBase::QuantityType type)
{
  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // get global quantity to be stored in nodes
    std::shared_ptr<const Core::LinAlg::Vector<double>> vectorglobal = nullptr;

    // start type switch
    switch (type)
    {
      case Mortar::StrategyBase::wmupdate:
      case Mortar::StrategyBase::wmold:
      {
        vectorglobal = wear_var_m();
        break;
      }
      case Mortar::StrategyBase::wupdate:
      case Mortar::StrategyBase::wold:
      case Mortar::StrategyBase::wupdateT:
      {
        vectorglobal = wear_var();
        break;
      }
      case Mortar::StrategyBase::weightedwear:
      {
        break;
      }
      default:
        CONTACT::AbstractStrategy::store_nodal_quantities(type);
        break;
    }  // switch

    // slave dof and node map of the interface
    // columnmap for current or updated LM
    // rowmap for remaining cases
    std::shared_ptr<const Core::LinAlg::Map> sdofmap, snodemap;
    if (type == Mortar::StrategyBase::wupdate or type == Mortar::StrategyBase::wold or
        type == Mortar::StrategyBase::wupdateT)
    {
      sdofmap = interface_[i]->slave_col_dofs();
      snodemap = interface_[i]->slave_col_nodes();
    }
    else
    {
      sdofmap = interface_[i]->slave_row_dofs();
      snodemap = interface_[i]->slave_row_nodes();
    }

    // master side wear
    std::shared_ptr<Core::LinAlg::Vector<double>> vectorinterface = nullptr;
    if (type == Mortar::StrategyBase::wmupdate or type == Mortar::StrategyBase::wmold)
    {
      // export global quantity to current interface slave dof map (column or row)
      const std::shared_ptr<Core::LinAlg::Map> masterdofs =
          Core::LinAlg::allreduce_e_map(*(interface_[i]->master_row_dofs()));
      vectorinterface = std::make_shared<Core::LinAlg::Vector<double>>(*masterdofs);

      if (vectorglobal != nullptr)  // necessary for case "activeold" and wear
        Core::LinAlg::export_to(*vectorglobal, *vectorinterface);
    }
    else
    {
      // export global quantity to current interface slave dof map (column or row)
      vectorinterface = std::make_shared<Core::LinAlg::Vector<double>>(*sdofmap);

      if (vectorglobal != nullptr)  // necessary for case "activeold" and wear
        Core::LinAlg::export_to(*vectorglobal, *vectorinterface);
    }

    // master specific
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(interface_[i]->master_row_nodes()));
    if (type == Mortar::StrategyBase::wmupdate)
    {
      for (int j = 0; j < masternodes->num_my_elements(); ++j)
      {
        int gid = masternodes->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

        // store updated wcurr into node
        fnode->wear_data().wcurr()[0] =
            (*vectorinterface)
                .local_values_as_span()[vectorinterface->get_map().lid(fnode->dofs()[0])];
      }
    }
    else if (type == Mortar::StrategyBase::wmold)
    {
      for (int j = 0; j < masternodes->num_my_elements(); ++j)
      {
        int gid = masternodes->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(node);

        // store updated wcurr into node
        fnode->wear_data().wold()[0] +=
            (*vectorinterface)
                .local_values_as_span()[vectorinterface->get_map().lid(fnode->dofs()[0])];
      }
    }
    else
    {
      // loop over all slave nodes (column or row) on the current interface
      for (int j = 0; j < snodemap->num_my_elements(); ++j)
      {
        int gid = snodemap->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);

        // be aware of problem dimension
        const int dim = n_dim();
        const int numdof = cnode->num_dof();
        if (dim != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

        // find indices for DOFs of current node in Core::LinAlg::Vector<double>
        // and extract this node's quantity from vectorinterface
        std::vector<int> locindex(dim);

        for (int dof = 0; dof < dim; ++dof)
        {
          locindex[dof] = (vectorinterface->get_map()).lid(cnode->dofs()[dof]);
          if (locindex[dof] < 0) FOUR_C_THROW("StoreNodalQuantities: Did not find dof in map");

          switch (type)
          {
            case Mortar::StrategyBase::wupdate:
            {
              // throw a FOUR_C_THROW if node is Active and DBC
              if (cnode->is_dbc() && cnode->active())
                FOUR_C_THROW("Slave node {} is active AND carries D.B.C.s!", cnode->id());

              // explicitly set global Lag. Mult. to zero for D.B.C nodes
              if (cnode->is_dbc()) (*vectorinterface).get_values()[locindex[dof]] = 0.0;

              // store updated wcurr into node
              CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(cnode);
              fnode->wear_data().wcurr()[0] =
                  vectorinterface->local_values_as_span()[locindex[(int)(dof / n_dim())]];
              dof = dof + n_dim() - 1;
              break;
            }
            case Mortar::StrategyBase::wupdateT:
            {
              // throw a FOUR_C_THROW if node is Active and DBC
              if (cnode->is_dbc() && cnode->active())
                FOUR_C_THROW("Slave node {} is active AND carries D.B.C.s!", cnode->id());

              // explicitly set global Lag. Mult. to zero for D.B.C nodes
              if (cnode->is_dbc()) (*vectorinterface).get_values()[locindex[dof]] = 0.0;

              // store updated wcurr into node
              CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(cnode);
              fnode->wear_data().waccu()[0] +=
                  vectorinterface->local_values_as_span()[locindex[(int)(dof / n_dim())]];
              dof = dof + n_dim() - 1;
              break;
            }
            case Mortar::StrategyBase::wold:
            {
              // throw a FOUR_C_THROW if node is Active and DBC
              if (cnode->is_dbc() && cnode->active())
                FOUR_C_THROW("Slave node {} is active AND carries D.B.C.s!", cnode->id());

              // explicitly set global Lag. Mult. to zero for D.B.C nodes
              if (cnode->is_dbc()) (*vectorinterface).get_values()[locindex[dof]] = 0.0;

              // store updated wcurr into node
              CONTACT::FriNode* fnode = dynamic_cast<CONTACT::FriNode*>(cnode);
              fnode->wear_data().wold()[0] +=
                  (*vectorinterface)
                      .local_values_as_span()[vectorinterface->get_map().lid(fnode->dofs()[0])];
              dof = dof + n_dim() - 1;

              break;
            }
            // weighted wear
            case Mortar::StrategyBase::weightedwear:
            {
              if (!friction_)
                FOUR_C_THROW("This should not be called for contact without friction");

              // update wear only once per node
              if (dof == 0)
              {
                CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(cnode);
                const double wearcoeffs = params().get<double>("WEARCOEFF", 0.0);
                const double wearcoeffm = params().get<double>("WEARCOEFF_MASTER", 0.0);
                const double wearcoeff = wearcoeffs + wearcoeffm;

                // amount of wear
                if (params().get<CONTACT::Problemtype>("PROBTYPE") !=
                    CONTACT::Problemtype::structalewear)
                  frinode->wear_data().weighted_wear() +=
                      wearcoeff * frinode->wear_data().delta_weighted_wear();

                // wear for each ale step
                else
                  frinode->wear_data().weighted_wear() =
                      wearcoeff * frinode->wear_data().delta_weighted_wear();
              }
              break;
            }
            default:
              break;
          }  // switch
        }
      }  // end slave loop
    }
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
