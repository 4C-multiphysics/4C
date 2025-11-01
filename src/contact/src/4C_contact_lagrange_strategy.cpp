// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_lagrange_strategy.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_contact_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_utils_epetra_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::LagrangeStrategy::LagrangeStrategy(
    const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
    const int spatialDim, MPI_Comm comm, const double alphaf, const int maxdof)
    : AbstractStrategy(data_ptr, dof_row_map, NodeRowMap, params, spatialDim, comm, alphaf, maxdof),
      interface_(interface),
      evalForceCalled_(false),
      activesetssconv_(false),
      activesetconv_(false),
      activesetsteps_(1),
      fLTLOld_(nullptr),
      fLTL_(nullptr),
      fLTLn_(nullptr),
      fLTLt_(nullptr),
      fconservation_(nullptr)
{
  // empty constructor body
  return;
}

/*----------------------------------------------------------------------*
 | initialize global contact variables for next Newton step   popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::initialize()
{
  // (re)setup global matrices containing fc derivatives
  // must use FE_MATRIX type here, as we will do non-local assembly!
  lindmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  linmmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    // (re)setup global tangent matrix
    if (!friction_) tmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 3);

    // (re)setup global matrix containing gap derivatives
    smatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 3);

    // inactive rhs for the saddle point problem
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    inactiverhs_ = Core::LinAlg::create_vector(*gidofs, true);

    // further terms depend on friction case
    // (re)setup global matrix containing "no-friction"-derivatives
    if (!friction_)
    {
      // tangential rhs
      tangrhs_ = Core::LinAlg::create_vector(*gactivedofs_, true);
      tderivmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivedofs_, 3);
    }
    // (re)setup of global friction
    else
    {
      // here the calculation of gstickt is necessary
      std::shared_ptr<Core::LinAlg::Map> gstickdofs =
          Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);
      linstickLM_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gstickdofs, 3);
      linstickDIS_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gstickdofs, 3);
      linstickRHS_ = Core::LinAlg::create_vector(*gstickdofs, true);

      linslipLM_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipdofs_, 3);
      linslipDIS_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipdofs_, 3);
      linslipRHS_ = Core::LinAlg::create_vector(*gslipdofs_, true);
    }
  }

  else
  {
    // (re)setup global tangent matrix
    if (!friction_) tmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivet_, 3);

    // (re)setup global matrix containing gap derivatives
    smatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 3);

    // inactive rhs for the saddle point problem
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    inactiverhs_ = Core::LinAlg::create_vector(*gidofs, true);

    // further terms depend on friction case
    // (re)setup global matrix containing "no-friction"-derivatives
    if (!friction_)
    {
      // tangential rhs
      tangrhs_ = Core::LinAlg::create_vector(*gactivet_, true);
      tderivmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivet_, 3);
    }
    // (re)setup of global friction
    else
    {
      // here the calculation of gstickt is necessary
      std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
      linstickLM_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gstickt, 3);
      linstickDIS_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gstickt, 3);
      linstickRHS_ = Core::LinAlg::create_vector(*gstickt, true);

      linslipLM_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipt_, 3);
      linslipDIS_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gslipt_, 3);
      linslipRHS_ = Core::LinAlg::create_vector(*gslipt_, true);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 | evaluate frictional contact (public)                   gitterle 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::evaluate_friction(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  // In case of nonsmooth contact the scenario of contacting edges (non parallel)
  // requires a penalty regularization. Here, the penalty contributions for this
  // special case are applied:
  if (nonSmoothContact_)
  {
    // add_line_to_lin_contributions(kteff,feff);
    add_line_to_lin_contributions_friction(*kteff, feff);

    // FD check of weighted gap g derivatives + jump for LTL case
#ifdef CONTACTFDJUMPLTL
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckJumpDerivLTL();
    }
#endif
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  /**********************************************************************/
  /* export weighted gap vector to gactiveN-map                         */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*gactivedofs_, true);
    if (gact->global_length()) Core::LinAlg::export_to(*wgap_, *gact);
  }
  else
  {
    gact = Core::LinAlg::create_vector(*gactivenodes_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
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
    interface_[i]->assemble_s(*smatrix_);
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);
    interface_[i]->assemble_lin_stick(*linstickLM_, *linstickDIS_, *linstickRHS_);
    interface_[i]->assemble_lin_slip(*linslipLM_, *linslipDIS_, *linslipRHS_);
    if (system_type() != CONTACT::SystemType::condensed)
      interface_[i]->assemble_inactiverhs(*inactiverhs_);
  }
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    smatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
  }
  else
  {
    // fill_complete() global matrix S
    smatrix_->complete(*gsmdofrowmap_, *gactiven_);
  }

  // fill_complete() global matrices LinD, LinM
  // (again for linD gsdofrowmap_ is sufficient as domain map,
  // but in the edge node modification case, master entries occur!)
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);

  // fill_complete global Matrix linstickLM_, linstickDIS_
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);

  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    linstickLM_->complete(*gstickdofs, *gstickdofs);
    linstickDIS_->complete(*gsmdofrowmap_, *gstickdofs);
    linslipLM_->complete(*gslipdofs_, *gslipdofs_);
    linslipDIS_->complete(*gsmdofrowmap_, *gslipdofs_);
  }
  else
  {
    linstickLM_->complete(*gstickdofs, *gstickt);
    linstickDIS_->complete(*gsmdofrowmap_, *gstickt);

    // fill_complete global Matrix linslipLM_ and linslipDIS_
    linslipLM_->complete(*gslipdofs_, *gslipt_);
    linslipDIS_->complete(*gsmdofrowmap_, *gslipt_);
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

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype_ == CONTACT::SystemType::condensed)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    /********************************************************************/
    /* (1) Multiply Mortar matrices: m^ = inv(d) * m                    */
    /********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);

    // for nonsmooth contact inverting D is more complex:
    // Note: this inversion if only applicable when vertex, edge and surface nodes
    // are involved. For a falling coin (only surface and edge nodes), a special but
    // more easy implementation is needed.
    if (nonSmoothContact_)
    {
      // 1. split d matrix in vertex edge and surf part
      std::shared_ptr<Core::LinAlg::SparseMatrix> dss, dsev, devs, devev;
      std::shared_ptr<Core::LinAlg::Map> gEVdofs;  // merged edge and vertex dofs

      // get dss
      Core::LinAlg::split_matrix2x2(
          dmatrix_, gsdofSurf_, gEVdofs, gsdofSurf_, gEVdofs, dss, dsev, devs, devev);

      // get dse and dsv
      std::shared_ptr<Core::LinAlg::Map> temp;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dse, dsv;

      Core::LinAlg::split_matrix2x2(
          dsev, gsdofSurf_, temp, gsdofEdge_, gsdofVertex_, dse, dsv, tempmtx1, tempmtx2);

      // get dee dev dve dvv
      std::shared_ptr<Core::LinAlg::SparseMatrix> dee, dev, dve, dvv;
      Core::LinAlg::split_matrix2x2(
          devev, gsdofEdge_, gsdofVertex_, gsdofEdge_, gsdofVertex_, dee, dev, dve, dvv);

      // 2. invert diagonal matrices dss dee dvv
      std::shared_ptr<Core::LinAlg::Vector<double>> diagV =
          Core::LinAlg::create_vector(*gsdofVertex_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagE =
          Core::LinAlg::create_vector(*gsdofEdge_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagS =
          Core::LinAlg::create_vector(*gsdofSurf_, true);
      Core::LinAlg::SparseMatrix invdV(*dvv);
      Core::LinAlg::SparseMatrix invdE(*dee);
      Core::LinAlg::SparseMatrix invdS(*dss);

      int err = 0;

      // extract diagonal of invd into diag
      invdV.extract_diagonal_copy(*diagV);
      invdE.extract_diagonal_copy(*diagE);
      invdS.extract_diagonal_copy(*diagS);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diagV->local_length(); ++i)
        if (abs((*diagV).get_values()[i]) < 1e-12) (*diagV).get_values()[i] = 1.0;
      for (int i = 0; i < diagE->local_length(); ++i)
        if (abs((*diagE).get_values()[i]) < 1e-12) (*diagE).get_values()[i] = 1.0;
      for (int i = 0; i < diagS->local_length(); ++i)
        if (abs((*diagS).get_values()[i]) < 1e-12) (*diagS).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diagV->reciprocal(*diagV);
      diagE->reciprocal(*diagE);
      diagS->reciprocal(*diagS);

      // re-insert inverted diagonal into invd
      err = invdV.replace_diagonal_values(*diagV);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdE.replace_diagonal_values(*diagE);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdS.replace_diagonal_values(*diagS);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

      // 3. multiply all sub matrices
      invd = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100, true, true);

      dse->scale(-1.0);
      dsv->scale(-1.0);
      dev->scale(-1.0);

      // inv_dse
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det1;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dse;
      inv_det1 = Core::LinAlg::matrix_multiply(*dse, false, invdE, false, false, false, true);
      dinv_dse = Core::LinAlg::matrix_multiply(invdS, false, *inv_det1, false, false, false, true);

      // inv_dev
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dev;
      inv_det2 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      dinv_dev = Core::LinAlg::matrix_multiply(invdE, false, *inv_det2, false, false, false, true);

      // inv_dsv part1
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det3;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det4;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det5;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv1;
      inv_det3 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      inv_det4 = Core::LinAlg::matrix_multiply(invdE, false, *inv_det3, false, false, false, true);
      inv_det5 = Core::LinAlg::matrix_multiply(*dse, false, *inv_det4, false, false, false, true);
      dinv_dsv1 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det5, false, false, false, true);

      // inv_dsv part2
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det6;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv2;
      inv_det6 = Core::LinAlg::matrix_multiply(*dsv, false, invdV, false, false, false, true);
      dinv_dsv2 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det6, false, false, false, true);

      // diagonal entries
      invd->add(invdS, false, 1.0, 1.0);
      invd->add(invdE, false, 1.0, 1.0);
      invd->add(invdV, false, 1.0, 1.0);

      invd->add(*dinv_dev, false, 1.0, 1.0);
      invd->add(*dinv_dse, false, 1.0, 1.0);
      invd->add(*dinv_dsv1, false, 1.0, 1.0);
      invd->add(*dinv_dsv2, false, 1.0, 1.0);

      // complete
      invd->complete();
    }
    // standard inverse diagonal matrix:
    else
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      int err = 0;

      // extract diagonal of invd into diag
      invd->extract_diagonal_copy(*diag);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diag->local_length(); ++i)
        if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diag->reciprocal(*diag);

      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      Core::LinAlg::export_to(*non_redist_gsdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);

      // re-insert inverted diagonal into invd
      err = invd->replace_diagonal_values(*diag);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
    }

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
    kmnmod->add(*kmn, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
    kmnmod->add(*kmnadd, false, 1.0, 1.0);
    kmnmod->complete(kmn->domain_map(), kmn->row_map());

    // kmm: add T(mhataam)*kam
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmmmod->add(*kmm, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
    kmmmod->add(*kmmadd, false, 1.0, 1.0);
    kmmmod->complete(kmm->domain_map(), kmm->row_map());

    // kmi: add T(mhataam)*kai
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
    if (iset)
    {
      kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
      kmimod->add(*kmi, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
          Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
      kmimod->add(*kmiadd, false, 1.0, 1.0);
      kmimod->complete(kmi->domain_map(), kmi->row_map());
    }

    // kma: add T(mhataam)*kaa
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
    if (aset)
    {
      kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
      kmamod->add(*kma, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
          Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
      kmamod->add(*kmaadd, false, 1.0, 1.0);
      kmamod->complete(kma->domain_map(), kma->row_map());
    }

    //--------------------------------------------------------- THIRD LINE
    // kin: subtract T(dhat)*kan
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kinmod->add(*kin, false, 1.0, 1.0);
    if (aset && iset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
      kinmod->add(*kinadd, false, -1.0, 1.0);
    }
    kinmod->complete(kin->domain_map(), kin->row_map());

    // kim: subtract T(dhat)*kam
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kimmod->add(*kim, false, 1.0, 1.0);
    if (aset && iset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
      kimmod->add(*kimadd, false, -1.0, 1.0);
    }
    kimmod->complete(kim->domain_map(), kim->row_map());

    // kii: subtract T(dhat)*kai
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
    if (iset)
    {
      kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
      kiimod->add(*kii, false, 1.0, 1.0);
      if (aset)
      {
        std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
            Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
        kiimod->add(*kiiadd, false, -1.0, 1.0);
      }
      kiimod->complete(kii->domain_map(), kii->row_map());
    }

    // kia: subtract T(dhat)*kaa
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
    if (iset && aset)
    {
      kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
      kiamod->add(*kia, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
      kiamod->add(*kiaadd, false, -1.0, 1.0);
      kiamod->complete(kia->domain_map(), kia->row_map());
    }

    //-------------------------------------------------------- FOURTH LINE

    //--------------------------------------------------------- FIFTH LINE
    // blocks for complementary conditions (stick nodes)

    // kstn: multiply with linstickLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kstnmod;
    if (stickset)
    {
      kstnmod =
          Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
      kstnmod = Core::LinAlg::matrix_multiply(*kstnmod, false, *kan, false, false, false, true);
    }

    // kstm: multiply with linstickLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kstmmod;
    if (stickset)
    {
      kstmmod =
          Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
      kstmmod = Core::LinAlg::matrix_multiply(*kstmmod, false, *kam, false, false, false, true);
    }

    // ksti: multiply with linstickLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kstimod;
    if (stickset && iset)
    {
      kstimod =
          Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
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
      kslnmod =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      kslnmod = Core::LinAlg::matrix_multiply(*kslnmod, false, *kan, false, false, false, true);
    }

    // kslm: multiply with linslipLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kslmmod;
    if (slipset)
    {
      kslmmod =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      kslmmod = Core::LinAlg::matrix_multiply(*kslmmod, false, *kam, false, false, false, true);
    }

    // ksli: multiply with linslipLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kslimod;
    if (slipset && iset)
    {
      kslimod =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      kslimod = Core::LinAlg::matrix_multiply(*kslimod, false, *kai, false, false, false, true);
    }

    // kslsl: multiply with linslipLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kslslmod;
    if (slipset)
    {
      kslslmod =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      kslslmod = Core::LinAlg::matrix_multiply(*kslslmod, false, *kasl, false, false, false, true);
    }

    // slstmod: multiply with linslipLM
    std::shared_ptr<Core::LinAlg::SparseMatrix> kslstmod;
    if (slipset && stickset)
    {
      kslstmod =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      kslstmod = Core::LinAlg::matrix_multiply(*kslstmod, false, *kast, false, false, false, true);
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
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
      else
        fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
      temp1->multiply(false, *fa, *fstmod);

      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
      else
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
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
      else
        fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
          Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
      temp->multiply(false, *fa, *fslmod);

      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
      else
        tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);

      linslipLM_->multiply(false, *zsl, *tempvec1);

      fslmod->update(-1.0, *tempvec1, 1.0);
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
      }
    }

    /********************************************************************/
    /* (10) Global setup of kteffnew (including contact)                */
    /********************************************************************/

    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
    std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
        Core::LinAlg::create_vector(*problem_dofs());

    //--------------------------------------------------------- FIRST LINE
    // add n submatrices to kteffnew
    kteffnew->add(*knn, false, 1.0, 1.0);
    kteffnew->add(*knm, false, 1.0, 1.0);
    if (sset) kteffnew->add(*kns, false, 1.0, 1.0);

    //-------------------------------------------------------- SECOND LINE
    // add m submatrices to kteffnew
    kteffnew->add(*kmnmod, false, 1.0, 1.0);
    kteffnew->add(*kmmmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kmimod, false, 1.0, 1.0);
    if (aset) kteffnew->add(*kmamod, false, 1.0, 1.0);

    //--------------------------------------------------------- THIRD LINE
    // add i submatrices to kteffnew
    if (iset) kteffnew->add(*kinmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kimmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kiimod, false, 1.0, 1.0);
    if (iset && aset) kteffnew->add(*kiamod, false, 1.0, 1.0);

    //-------------------------------------------------------- FOURTH LINE

    // add a submatrices to kteffnew
    if (aset) kteffnew->add(*smatrix_, false, 1.0, 1.0);

    //--------------------------------------------------------- FIFTH LINE
    // add st submatrices to kteffnew
    if (stickset) kteffnew->add(*kstnmod, false, 1.0, 1.0);
    if (stickset) kteffnew->add(*kstmmod, false, 1.0, 1.0);
    if (stickset && iset) kteffnew->add(*kstimod, false, 1.0, 1.0);
    if (stickset && slipset) kteffnew->add(*kstslmod, false, 1.0, 1.0);
    if (stickset) kteffnew->add(*kststmod, false, 1.0, 1.0);

    // add terms of linearization of sick condition to kteffnew
    if (stickset) kteffnew->add(*linstickDIS_, false, -1.0, 1.0);

    //--------------------------------------------------------- SIXTH LINE
    // add sl submatrices to kteffnew
    if (slipset) kteffnew->add(*kslnmod, false, 1.0, 1.0);
    if (slipset) kteffnew->add(*kslmmod, false, 1.0, 1.0);
    if (slipset && iset) kteffnew->add(*kslimod, false, 1.0, 1.0);
    if (slipset) kteffnew->add(*kslslmod, false, 1.0, 1.0);
    if (slipset && stickset) kteffnew->add(*kslstmod, false, 1.0, 1.0);

    // add terms of linearization of slip condition to kteffnew and feffnew
    if (slipset) kteffnew->add(*linslipDIS_, false, -1.0, +1.0);

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
      Core::LinAlg::export_to(*gact, *gexp);
      feffnew->update(-1.0, *gexp, 1.0);
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

    if (slipset)
    {
      Core::LinAlg::Vector<double> linslipRHSexp(*problem_dofs());
      Core::LinAlg::export_to(*linslipRHS_, linslipRHSexp);
      feffnew->update(-1.0, linslipRHSexp, 1.0);
    }

    // finally do the replacement
    kteff = kteffnew;
    feff = feffnew;
  }

  //**********************************************************************
  //**********************************************************************
  // CASE B: SADDLE POINT SYSTEM
  //**********************************************************************
  //**********************************************************************
  else
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
    kteff->un_complete();
    kteff->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
    kteff->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
    kteff->complete();

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
      feff->update(-(1.0 - alphaf_), fsexp, 1.0);

      Core::LinAlg::Vector<double> fmexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecm(mmatrix_->domain_map());
      mmatrix_->multiply(true, zexp, tempvecm);
      Core::LinAlg::export_to(tempvecm, fmexp);
      feff->update(1.0 - alphaf_, fmexp, 1.0);

      // add old contact forces (t_n)
      Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecdold(dold_->domain_map());
      Core::LinAlg::Vector<double> zoldexp(dold_->row_map());
      if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
      dold_->multiply(true, zoldexp, tempvecdold);
      Core::LinAlg::export_to(tempvecdold, fsoldexp);
      feff->update(-alphaf_, fsoldexp, 1.0);

      Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecmold(mold_->domain_map());
      mold_->multiply(true, zoldexp, tempvecmold);
      Core::LinAlg::export_to(tempvecmold, fmoldexp);
      feff->update(alphaf_, fmoldexp, 1.0);
    }
    // if there is no self contact everything is ok
    else
    {
      // add contact force terms
      Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
      dmatrix_->multiply(true, *z_, fs);
      Core::LinAlg::Vector<double> fsexp(*problem_dofs());
      Core::LinAlg::export_to(fs, fsexp);
      feff->update(-(1.0 - alphaf_), fsexp, 1.0);

      Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
      mmatrix_->multiply(true, *z_, fm);
      Core::LinAlg::Vector<double> fmexp(*problem_dofs());
      Core::LinAlg::export_to(fm, fmexp);
      feff->update(1.0 - alphaf_, fmexp, 1.0);

      // add old contact forces (t_n)
      Core::LinAlg::Vector<double> fsold(*gsdofrowmap_);
      dold_->multiply(true, *zold_, fsold);
      Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
      Core::LinAlg::export_to(fsold, fsoldexp);
      feff->update(-alphaf_, fsoldexp, 1.0);

      Core::LinAlg::Vector<double> fmold(*gmdofrowmap_);
      mold_->multiply(true, *zold_, fmold);
      Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
      Core::LinAlg::export_to(fmold, fmoldexp);
      feff->update(alphaf_, fmoldexp, 1.0);
    }
  }

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckGapDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

#ifdef CONTACTFDALPHA
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckAlphaDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

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
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv1 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv2 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //
      //      deriv1->Add(*linstickLM_,false,1.0,1.0);
      //      deriv1->Complete(*gsmdofrowmap_,*gactivet_);
      //
      //      deriv2->Add(*linstickDIS_,false,1.0,1.0);
      //      deriv2->Complete(*gsmdofrowmap_,*gactivet_);
      //
      //      std::cout << "DERIV 1 *********** "<< *deriv1 << std::endl;
      //      std::cout << "DERIV 2 *********** "<< *deriv2 << std::endl;

      interface_[i]->FDCheckStickDeriv(*linstickLM_, *linstickDIS_);
    }
  }
#endif  // #ifdef CONTACTFDSTICK

#ifdef CONTACTFDSLIP

  if (gslipnodes_->NumGlobalElements())
  {
    // FD check of slip condition
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv1 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv2 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //
      //      deriv1->Add(*linslipLM_,false,1.0,1.0);
      //      deriv1->Complete(*gsmdofrowmap_,*gslipt_);
      //
      //      deriv2->Add(*linslipDIS_,false,1.0,1.0);
      //      deriv2->Complete(*gsmdofrowmap_,*gslipt_);
      //
      //      std::cout << *deriv1 << std::endl;
      //      std::cout << *deriv2 << std::endl;

      interface_[i]->FDCheckSlipDeriv(*linslipLM_, *linslipDIS_);
    }
  }
#endif  // #ifdef CONTACTFDSLIP

  return;
}

/*----------------------------------------------------------------------*
 |  pp stresses                                              farah 11/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::compute_contact_stresses()
{
  static int step = 0;
  // call abstract function
  CONTACT::AbstractStrategy::compute_contact_stresses();

  // further scaling for nonsmooth contact
  if (nonSmoothContact_)
  {
    forcenormal_ = std::make_shared<Core::LinAlg::Vector<double>>(slave_dof_row_map(true));
    d_matrix()->multiply(true, *stressnormal_, *forcenormal_);
    forcetangential_ = std::make_shared<Core::LinAlg::Vector<double>>(slave_dof_row_map(true));
    d_matrix()->multiply(true, *stresstangential_, *forcetangential_);

    Core::LinAlg::Vector<double> forcenormal(slave_dof_row_map(true));
    d_matrix()->multiply(true, *stressnormal_, forcenormal);

    Core::LinAlg::Vector<double> forcetangential(slave_dof_row_map(true));
    d_matrix()->multiply(true, *stresstangential_, forcetangential);

    // add penalty force normal
    if (fLTLn_ != nullptr)
    {
      Core::LinAlg::Vector<double> dummy(slave_dof_row_map(true));
      Core::LinAlg::export_to(*fLTLn_, dummy);
      forcenormal_->update(1.0, dummy, 1.0);
      forcenormal.update(1.0, dummy, 1.0);
    }

    // add penalty force tangential
    if (fLTLt_ != nullptr)
    {
      Core::LinAlg::Vector<double> dummy(slave_dof_row_map(true));
      Core::LinAlg::export_to(*fLTLt_, dummy);
      forcetangential_->update(1.0, dummy, 1.0);
      forcetangential.update(1.0, dummy, 1.0);
    }

    // loop over all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      // loop over all slave row nodes on the current interface
      for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
      {
        int gid = interface_[i]->slave_row_nodes()->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* cnode = dynamic_cast<Node*>(node);

        std::vector<int> locindex(n_dim());

        for (int dof = 0; dof < n_dim(); ++dof)
        {
          locindex[dof] = (forcenormal.get_map()).lid(cnode->dofs()[dof]);

          if (cnode->mo_data().get_dscale() < 1e-8 and cnode->active())
          {
            std::cout << "WARNING: dscale not valid!" << std::endl;
            continue;
          }
          else if (cnode->mo_data().get_dscale() < 1e-8)
          {
            continue;
          }

          (forcenormal).get_values()[locindex[dof]] /= cnode->mo_data().get_dscale();
          (forcetangential).get_values()[locindex[dof]] /= cnode->mo_data().get_dscale();
        }
      }
    }
    stresstangential_->update(1.0, forcetangential, 0.0);
    stressnormal_->update(1.0, forcenormal, 0.0);

    // temporary output:
    double tangforce = 0.0;
    forcetangential_->norm_2(&tangforce);
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "tangential force = " << tangforce << std::endl;

    double normalforce = 0.0;
    forcenormal_->norm_2(&normalforce);
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "normal force = " << normalforce << std::endl;

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      FILE* MyFile = nullptr;
      std::ostringstream filename;
      const std::string filebase = "xxx";
      filename << filebase << ".fric";
      MyFile = fopen(filename.str().c_str(), "at+");

      // store data
      if (MyFile)
      {
        fprintf(MyFile, "%d\t", step);
        fprintf(MyFile, "%g\t", tangforce);
        fprintf(MyFile, "%g\n", normalforce);
        fclose(MyFile);
      }
      else
        FOUR_C_THROW("File could not be opened.");
    }
  }

  step++;

  return;
}

/*----------------------------------------------------------------------*
 |  add penalty terms for ltl contact                        farah 11/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::save_reference_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (!nonSmoothContact_) return;

  // initialize the displacement field
  set_state(Mortar::state_new_displacement, *dis);

  // guarantee uniqueness
  std::set<std::pair<int, int>> donebefore;

  // kappa will be the shape function integral on the slave sides
  // (1) build the nodal information
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // interface needs to be complete
    if (!interface_[i]->filled() && Core::Communication::my_mpi_rank(get_comm()) == 0)
      FOUR_C_THROW("fill_complete() not called on interface %", i);

    // reset kappa
    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->master_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->master_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);
      cnode->data().kappa() = 0.0;
    }

    // loop over proc's slave elements of the interface for integration
    // use standard column map to include processor's ghosted elements
    for (int j = 0; j < interface_[i]->master_col_elements()->num_my_elements(); ++j)
    {
      int gid1 = interface_[i]->master_col_elements()->gid(j);
      Core::Elements::Element* ele1 = interface_[i]->discret().g_element(gid1);
      if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
      Element* selement = dynamic_cast<Element*>(ele1);

      // loop over slave edges -> match node number for tri3/quad4
      for (int k = 0; k < selement->num_node(); ++k)
      {
        int nodeIds[2] = {0, 0};
        int nodeLIds[2] = {0, 0};

        if (selement->shape() == Core::FE::CellType::quad4)
        {
          if (k == 0)
          {
            nodeIds[0] = selement->node_ids()[0];
            nodeIds[1] = selement->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (k == 1)
          {
            nodeIds[0] = selement->node_ids()[1];
            nodeIds[1] = selement->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (k == 2)
          {
            nodeIds[0] = selement->node_ids()[2];
            nodeIds[1] = selement->node_ids()[3];

            nodeLIds[0] = 2;
            nodeLIds[1] = 3;
          }
          else if (k == 3)
          {
            nodeIds[0] = selement->node_ids()[3];
            nodeIds[1] = selement->node_ids()[0];

            nodeLIds[0] = 3;
            nodeLIds[1] = 0;
          }
          else
            FOUR_C_THROW("loop counter and edge number do not match!");
        }
        else if (selement->shape() == Core::FE::CellType::tri3)
        {
          if (k == 0)
          {
            nodeIds[0] = selement->node_ids()[0];
            nodeIds[1] = selement->node_ids()[1];

            nodeLIds[0] = 0;
            nodeLIds[1] = 1;
          }
          else if (k == 1)
          {
            nodeIds[0] = selement->node_ids()[1];
            nodeIds[1] = selement->node_ids()[2];

            nodeLIds[0] = 1;
            nodeLIds[1] = 2;
          }
          else if (k == 2)
          {
            nodeIds[0] = selement->node_ids()[2];
            nodeIds[1] = selement->node_ids()[0];

            nodeLIds[0] = 2;
            nodeLIds[1] = 0;
          }
          else
            FOUR_C_THROW("loop counter and edge number do not match!");
        }

        // check if both nodes on edge geometry
        bool node0Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[0]])->is_on_edge();
        bool node1Edge = dynamic_cast<Mortar::Node*>(selement->nodes()[nodeLIds[1]])->is_on_edge();

        if (!node0Edge or !node1Edge) continue;

        // create pair
        std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
        std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

        // check if processed before
        std::set<std::pair<int, int>>::iterator iter = donebefore.find(actIDs);
        std::set<std::pair<int, int>>::iterator itertw = donebefore.find(actIDstw);

        // if not then create ele
        if (iter == donebefore.end() and itertw == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(actIDs);
          donebefore.insert(actIDstw);

          // create line ele:
          Mortar::Element lineEle(
              j, selement->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

          // get nodes
          std::array<Core::Nodes::Node*, 2> nodes = {
              selement->nodes()[nodeLIds[0]], selement->nodes()[nodeLIds[1]]};
          lineEle.build_nodal_pointers(nodes.data());

          // init data container for dual shapes
          lineEle.initialize_data_container();

          // create integrator
          CONTACT::Integrator integrator(params(), lineEle.shape(), get_comm());

          // integrate kappe penalty
          integrator.integrate_kappa_penalty_lts(lineEle);
        }
      }  // end edge loop
    }

    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interface_[i]->master_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->master_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      // only for edge nodes!
      if (cnode->is_on_edge() and !cnode->is_on_corner())
      {
        // get nodal weighted gap
        // (this is where we stored the shape function integrals)
        double kappainv = cnode->data().kappa();

        // safety
        if (abs(kappainv) < 1e-12)
        {
          kappainv = 1.0;
          //          FOUR_C_THROW("gap is zero!");
        }
        else
        {
          // store kappa
          cnode->data().kappa() = 1.0 / kappainv;
        }
      }
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  add penalty terms for ltl contact                        farah 11/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::add_master_contributions(Core::LinAlg::SparseOperator& kteff,
    Core::LinAlg::Vector<double>& feff, bool add_time_integration)
{
  // create new contact force vector for LTL contact
  auto fc = std::make_shared<Core::LinAlg::FEVector<double>>(feff.get_map());

  // create new contact stiffness matric for LTL contact
  Core::LinAlg::SparseMatrix kc(dynamic_cast<Core::LinAlg::SparseMatrix*>(&kteff)->row_map(), 100,
      true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // loop over interface and assemble force and stiffness
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // line to segment
    interface_[i]->add_lts_forces_master(*fc);
    interface_[i]->add_lts_stiffness_master(kc);
    // node to segment
    interface_[i]->add_nts_forces_master(*fc);
    interface_[i]->add_nts_stiffness_master(kc);
  }

  // force
  fc->complete();

  // store fLTL values for time integration
  fLTL_ = std::make_shared<Core::LinAlg::Vector<double>>(fc->get_map());
  fLTL_->update(1.0, *fc, 0.0);

  if (add_time_integration)
    if (fLTLOld_ != nullptr) feff.update(alphaf_, *fLTLOld_, 1.);

  double fac = 0.;
  if (add_time_integration)
    fac = 1. - alphaf_;
  else
    fac = 1.;
  feff.update(fac, *fc, 1.);

  // stiffness
  kc.complete();
  kteff.un_complete();
  kteff.add(kc, false, fac, 1.);
  kteff.complete();

  // bye bye
  return;
}


/*----------------------------------------------------------------------*
 |  add penalty terms for ltl contact                        farah 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::add_line_to_lin_contributions(Core::LinAlg::SparseOperator& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff, bool add_time_integration)
{
  // create new contact force vector for LTL contact
  auto fc = std::make_shared<Core::LinAlg::FEVector<double>>(feff->get_map());

  fconservation_ = std::make_shared<Core::LinAlg::Vector<double>>(feff->get_map());

  // create new contact stiffness matric for LTL contact
  Core::LinAlg::SparseMatrix kc(dynamic_cast<Core::LinAlg::SparseMatrix*>(&kteff)->row_map(), 100,
      true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // loop over interface and assemble force and stiffness
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->add_ltl_forces(*fc);
    interface_[i]->add_ltl_stiffness(kc);
  }

  // get info for conservation check
  fconservation_->update(1.0, *fc, 0.0);

  // force
  fc->complete();

  // store fLTL values for time integration
  fLTL_ = std::make_shared<Core::LinAlg::Vector<double>>(fc->get_map());
  fLTL_->update(1.0, *fc, 0.0);

  if (add_time_integration)
    if (fLTLOld_ != nullptr) feff->update(alphaf_, *fLTLOld_, 1.);

  double fac = 0.;
  if (add_time_integration)
    fac = 1. - alphaf_;
  else
    fac = 1.;
  feff->update(fac, *fc, 1.);

  // stiffness
  kc.complete();
  kteff.un_complete();
  kteff.add(kc, false, fac, 1.);
  kteff.complete();

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 |  add frictional penalty terms for ltl contact             farah 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::add_line_to_lin_contributions_friction(
    Core::LinAlg::SparseOperator& kteff, std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
    bool add_time_integration)
{
  // create new contact force vector for LTL contact
  auto fc = std::make_shared<Core::LinAlg::FEVector<double>>(feff->get_map());

  fconservation_ = std::make_shared<Core::LinAlg::Vector<double>>(feff->get_map());

  // create new contact stiffness matric for LTL contact
  Core::LinAlg::SparseMatrix kc(dynamic_cast<Core::LinAlg::SparseMatrix*>(&kteff)->row_map(), 100,
      true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // loop over interface and assemble force and stiffness
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->add_ltl_forces(*fc);
    interface_[i]->add_ltl_stiffness(kc);
  }

  // store normal forces
  fLTLn_ = std::make_shared<Core::LinAlg::Vector<double>>(fc->get_map());
  fLTLn_->update(1.0, *fc, 0.0);

  // loop over interface and assemble force and stiffness
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->add_ltl_forces_friction(*fc);
    interface_[i]->add_ltl_stiffness_friction(kc);
  }

  // get info for conservation check
  fconservation_->update(1.0, *fc, 0.0);

  // store tangential forces
  fLTLt_ = std::make_shared<Core::LinAlg::Vector<double>>(fc->get_map());
  fLTLt_->update(1.0, *fc, 0.0);
  fLTLt_->update(-1.0, *fLTLn_, 1.0);

  // force
  fc->complete();

  // store fLTL values for time integration
  fLTL_ = std::make_shared<Core::LinAlg::Vector<double>>(fc->get_map());
  fLTL_->update(1.0, *fc, 0.0);

  if (add_time_integration)
    if (fLTLOld_ != nullptr) feff->update(alphaf_, *fLTLOld_, 1.);

  double fac = 0.;
  if (add_time_integration)
    fac = 1. - alphaf_;
  else
    fac = 1.;
  feff->update(fac, *fc, 1.);

  // stiffness
  kc.complete();
  kteff.un_complete();
  kteff.add(kc, false, fac, 1.);
  kteff.complete();

  // bye bye
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate contact (public)                                 popp 04/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::evaluate_contact(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  // shape function type and type of LM interpolation for quadratic elements
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  // In case of nonsmooth contact the scenario of contacting edges (non parallel)
  // requires a penalty regularization. Here, the penalty contriutions for this
  // special case are applied:
  if (nonSmoothContact_)
  {
    // LTL contributions:
    add_line_to_lin_contributions(*kteff, feff);

    // penalty support for master side quantities:
    add_master_contributions(*kteff, *feff);

#ifdef CONTACTFDGAPLTL
    // FD check of weighted gap g derivatives (non-penetr. condition)
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckGapDerivLTL();
    }
#endif  // #ifdef CONTACTFDGAPLTL
  }


  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  /**********************************************************************/
  /* export weighted gap vector to gactiveN-map                         */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*gactivedofs_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
    }
  }
  else
  {
    gact = Core::LinAlg::create_vector(*gactivenodes_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
  }
  /**********************************************************************/
  /* calculate                                                          */
  /**********************************************************************/
  /* build global matrix tmatrix_ with tangent vectors of active nodes  */
  /* and global matrix nmatrix_ with normal vectors of active nodes     */
  /* and global matrix s with normal+D+M derivatives of active nodes    */
  /* and global matrix tderivmatrix_ with tangent derivatives           */
  /*     of active nodes                                                */
  /* and global matrix nderivmatrix_ with normal derivatives            */
  /*     of active nodes                                                */
  /* and inactive right-hand side with old lagrange multipliers (incr)  */
  /* and tangential right-hand side (incr)                              */
  /**********************************************************************/
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->assemble_tn(tmatrix_, nmatrix_);
    interface_[i]->assemble_s(*smatrix_);
    interface_[i]->assemble_t_nderiv(tderivmatrix_, nderivmatrix_);
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);

    if (system_type() != CONTACT::SystemType::condensed)
    {
      interface_[i]->assemble_inactiverhs(*inactiverhs_);
      interface_[i]->assemble_tangrhs(*tangrhs_);
    }
  }
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    // fill_complete() global matrix T
    tmatrix_->complete(*gactivedofs_, *gactivedofs_);

    // fill_complete() global matrix N
    if (nmatrix_ != nullptr) nmatrix_->complete(*gactivedofs_, *gactivedofs_);
    smatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
    tderivmatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
    if (nderivmatrix_ != nullptr) nderivmatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
  }
  else
  {
    // fill_complete() global matrix T
    tmatrix_->complete(*gactivedofs_, *gactivet_);

    // fill_complete() global matrix N
    if (nmatrix_ != nullptr) nmatrix_->complete(*gactivedofs_, *gactiven_);

    // fill_complete() global matrix S
    smatrix_->complete(*gsmdofrowmap_, *gactiven_);

    // fill_complete() global matrix Tderiv
    // (actually gsdofrowmap_ is in general sufficient as domain map,
    // but in the edge node modification case, master entries occur!)
    tderivmatrix_->complete(*gsmdofrowmap_, *gactivet_);

    // fill_complete() global matrix Nderiv
    if (nderivmatrix_ != nullptr) nderivmatrix_->complete(*gsmdofrowmap_, *gactiven_);
  }

  // fill_complete() global matrices LinD, LinM
  // (again for linD gsdofrowmap_ is sufficient as domain map,
  // but in the edge node modification case, master entries occur!)
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // LinD      ---->   T^(-T) * LinD
  //----------------------------------------------------------------------
  if (is_dual_quad_slave_trafo())
  {
    if (lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      if (parallel_redistribution_status())
        trafo_ = Core::LinAlg::matrix_row_transform(*trafo_, *gsmdofrowmap_);
      lindmatrix_ =
          Core::LinAlg::matrix_multiply(*lindmatrix_, false, *trafo_, false, false, false, true);
      linmmatrix_ =
          Core::LinAlg::matrix_multiply(*linmmatrix_, false, *trafo_, false, false, false, true);
      smatrix_ =
          Core::LinAlg::matrix_multiply(*smatrix_, false, *trafo_, false, false, false, true);
      tderivmatrix_ =
          Core::LinAlg::matrix_multiply(*tderivmatrix_, false, *trafo_, false, false, false, true);
    }
    else
    {
      // modify lindmatrix_
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
      lindmatrix_ = temp1;
    }
  }

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype_ == CONTACT::SystemType::condensed)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin &&
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD") !=
            Inpar::Mortar::lagmult_const)
      FOUR_C_THROW("Condensation only for dual LM");

    /**********************************************************************/
    /* (1) Multiply Mortar matrices: m^ = inv(d) * m                      */
    /**********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);

    // for nonsmooth contact inverting D is more complex:
    // Note: this inversion if only applicable when vertex, edge and surface nodes
    // are involved. For a falling coin (only surface and edge nodes), a special but
    // more easy implementation is needed.
    if (nonSmoothContact_)
    {
      // 1. split d matrix in vertex edge and surf part
      std::shared_ptr<Core::LinAlg::SparseMatrix> dss, dsev, devs, devev;
      std::shared_ptr<Core::LinAlg::Map> gEVdofs;  // merged edge and vertex dofs

      // get dss
      Core::LinAlg::split_matrix2x2(
          dmatrix_, gsdofSurf_, gEVdofs, gsdofSurf_, gEVdofs, dss, dsev, devs, devev);

      // get dse and dsv
      std::shared_ptr<Core::LinAlg::Map> temp;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dse, dsv;

      Core::LinAlg::split_matrix2x2(
          dsev, gsdofSurf_, temp, gsdofEdge_, gsdofVertex_, dse, dsv, tempmtx1, tempmtx2);

      // get dee dev dve dvv
      std::shared_ptr<Core::LinAlg::SparseMatrix> dee, dev, dve, dvv;
      Core::LinAlg::split_matrix2x2(
          devev, gsdofEdge_, gsdofVertex_, gsdofEdge_, gsdofVertex_, dee, dev, dve, dvv);

      // 2. invert diagonal matrices dss dee dvv
      std::shared_ptr<Core::LinAlg::Vector<double>> diagV =
          Core::LinAlg::create_vector(*gsdofVertex_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagE =
          Core::LinAlg::create_vector(*gsdofEdge_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagS =
          Core::LinAlg::create_vector(*gsdofSurf_, true);
      Core::LinAlg::SparseMatrix invdV(*dvv);
      Core::LinAlg::SparseMatrix invdE(*dee);
      Core::LinAlg::SparseMatrix invdS(*dss);

      int err = 0;

      // extract diagonal of invd into diag
      invdV.extract_diagonal_copy(*diagV);
      invdE.extract_diagonal_copy(*diagE);
      invdS.extract_diagonal_copy(*diagS);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diagV->local_length(); ++i)
        if (abs((*diagV).get_values()[i]) < 1e-12) (*diagV).get_values()[i] = 1.0;
      for (int i = 0; i < diagE->local_length(); ++i)
        if (abs((*diagE).get_values()[i]) < 1e-12) (*diagE).get_values()[i] = 1.0;
      for (int i = 0; i < diagS->local_length(); ++i)
        if (abs((*diagS).get_values()[i]) < 1e-12) (*diagS).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diagV->reciprocal(*diagV);
      diagE->reciprocal(*diagE);
      diagS->reciprocal(*diagS);

      // re-insert inverted diagonal into invd
      err = invdV.replace_diagonal_values(*diagV);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdE.replace_diagonal_values(*diagE);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdS.replace_diagonal_values(*diagS);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

      // 3. multiply all sub matrices
      invd = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100, true, true);

      dse->scale(-1.0);
      dsv->scale(-1.0);
      dev->scale(-1.0);

      // inv_dse
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det1;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dse;
      inv_det1 = Core::LinAlg::matrix_multiply(*dse, false, invdE, false, false, false, true);
      dinv_dse = Core::LinAlg::matrix_multiply(invdS, false, *inv_det1, false, false, false, true);

      // inv_dev
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dev;
      inv_det2 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      dinv_dev = Core::LinAlg::matrix_multiply(invdE, false, *inv_det2, false, false, false, true);

      // inv_dsv part1
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det3;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det4;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det5;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv1;
      inv_det3 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      inv_det4 = Core::LinAlg::matrix_multiply(invdE, false, *inv_det3, false, false, false, true);
      inv_det5 = Core::LinAlg::matrix_multiply(*dse, false, *inv_det4, false, false, false, true);
      dinv_dsv1 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det5, false, false, false, true);

      // inv_dsv part2
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det6;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv2;
      inv_det6 = Core::LinAlg::matrix_multiply(*dsv, false, invdV, false, false, false, true);
      dinv_dsv2 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det6, false, false, false, true);

      // diagonal entries
      invd->add(invdS, false, 1.0, 1.0);
      invd->add(invdE, false, 1.0, 1.0);
      invd->add(invdV, false, 1.0, 1.0);

      invd->add(*dinv_dev, false, 1.0, 1.0);
      invd->add(*dinv_dse, false, 1.0, 1.0);
      invd->add(*dinv_dsv1, false, 1.0, 1.0);
      invd->add(*dinv_dsv2, false, 1.0, 1.0);

      invd->complete();
    }
    // standard inverse diagonal matrix:
    else
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      int err = 0;

      // extract diagonal of invd into diag
      invd->extract_diagonal_copy(*diag);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diag->local_length(); ++i)
        if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diag->reciprocal(*diag);

      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      Core::LinAlg::export_to(*non_redist_gsdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);

      // re-insert inverted diagonal into invd
      err = invd->replace_diagonal_values(*diag);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
    }

    // do the multiplication mhat = inv(D) * M
    mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd, false, *mmatrix_, false, false, false, true);

    /**********************************************************************/
    /* (2) Add contact stiffness terms to kteff                           */
    /**********************************************************************/
    // declare sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);

    if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // basis transformation
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (parallel_redistribution_status())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();

      // apply basis transformation to K and f
      kteffmatrix =
          Core::LinAlg::matrix_multiply(*kteffmatrix, false, systrafo, false, false, false, true);
      kteffmatrix =
          Core::LinAlg::matrix_multiply(systrafo, true, *kteffmatrix, false, false, false, true);
      Core::LinAlg::Vector<double> feffnew(*feff);
      systrafo.multiply(true, feffnew, *feff);
    }

    // transform if necessary
    if (parallel_redistribution_status())
    {
      lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
      linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
    }

    kteffmatrix->un_complete();
    kteffmatrix->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
    kteffmatrix->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
    kteffmatrix->complete();

    /**********************************************************************/
    /* (3) Split kteff into 3x3 matrix blocks                             */
    /**********************************************************************/
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

    /**********************************************************************/
    /* (4) Split feff into 3 subvectors                                   */
    /**********************************************************************/
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

    // abbreviations for slave  and master set
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

    //----------------------------------------------------------------------
    // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
    //----------------------------------------------------------------------
    // Concretely, we apply the following transformations:
    // D         ---->   D * T^(-1)
    // D^(-1)    ---->   T * D^(-1)
    // \hat{M}   ---->   T * \hat{M}
    //----------------------------------------------------------------------
    if (is_dual_quad_slave_trafo() && lagmultquad != Inpar::Mortar::lagmult_lin)
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

    /**********************************************************************/
    /* (5) Split slave quantities into active / inactive                  */
    /**********************************************************************/
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

    // abbreviations for active and inactive set
    const int aset = gactivedofs_->num_global_elements();
    const int iset = gidofs->num_global_elements();

    // we want to split fsmod into 2 groups a,i
    std::shared_ptr<Core::LinAlg::Vector<double>> fa =
        std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
    std::shared_ptr<Core::LinAlg::Vector<double>> fi =
        std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

    // do the vector splitting s -> a+i
    Core::LinAlg::split_vector(*gsdofrowmap_, *fs, gactivedofs_, fa, gidofs, fi);

    /**********************************************************************/
    /* (6) Isolate necessary parts from invd and mhatmatrix               */
    /**********************************************************************/
    // active part of invd
    std::shared_ptr<Core::LinAlg::SparseMatrix> invda;
    Core::LinAlg::split_matrix2x2(
        invd_, gactivedofs_, gidofs, gactivedofs_, gidofs, invda, tempmtx1, tempmtx2, tempmtx3);

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
    dai->scale(1 - alphaf_);

    save_coupling_matrices(dhat, mhataam, invda);
    /**********************************************************************/
    /* (7) Build the final K blocks                                       */
    /**********************************************************************/

    //----------------------------------------------------------- FIRST LINE
    // knn: nothing to do

    // knm: nothing to do

    // kns: nothing to do

    //---------------------------------------------------------- SECOND LINE
    // kmn: add T(mhataam)*kan
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmnmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmnmod->add(*kmn, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
    kmnmod->add(*kmnadd, false, 1.0, 1.0);
    kmnmod->complete(kmn->domain_map(), kmn->row_map());

    // kmm: add T(mhataam)*kam
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmmmod->add(*kmm, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
    kmmmod->add(*kmmadd, false, 1.0, 1.0);
    kmmmod->complete(kmm->domain_map(), kmm->row_map());

    // kmi: add T(mhataam)*kai
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
    if (iset)
    {
      kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
      kmimod->add(*kmi, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
          Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
      kmimod->add(*kmiadd, false, 1.0, 1.0);
      kmimod->complete(kmi->domain_map(), kmi->row_map());
    }

    // kma: add T(mhataam)*kaa
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
    if (aset)
    {
      kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
      kmamod->add(*kma, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
          Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
      kmamod->add(*kmaadd, false, 1.0, 1.0);
      kmamod->complete(kma->domain_map(), kma->row_map());
    }

    //----------------------------------------------------------- THIRD LINE
    //------------------- FOR 3D QUADRATIC CASE ----------------------------
    // kin: subtract T(dhat)*kan --
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kinmod->add(*kin, false, 1.0, 1.0);
    if (aset && iset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
      kinmod->add(*kinadd, false, -1.0, 1.0);
    }
    kinmod->complete(kin->domain_map(), kin->row_map());

    // kim: subtract T(dhat)*kam
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kimmod->add(*kim, false, 1.0, 1.0);
    if (aset && iset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
      kimmod->add(*kimadd, false, -1.0, 1.0);
    }
    kimmod->complete(kim->domain_map(), kim->row_map());

    // kii: subtract T(dhat)*kai
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
    if (iset)
    {
      kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
      kiimod->add(*kii, false, 1.0, 1.0);
      if (aset)
      {
        std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
            Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
        kiimod->add(*kiiadd, false, -1.0, 1.0);
      }
      kiimod->complete(kii->domain_map(), kii->row_map());
    }

    // kia: subtract T(dhat)*kaa
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
    if (iset && aset)
    {
      kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
      kiamod->add(*kia, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
      kiamod->add(*kiaadd, false, -1.0, 1.0);
      kiamod->complete(kia->domain_map(), kia->row_map());
    }

    //---------------------------------------------------------- FOURTH LINE
    // nothing to do

    //----------------------------------------------------------- FIFTH LINE
    // kan: multiply tmatrix with invda and kan
    std::shared_ptr<Core::LinAlg::SparseMatrix> kanmod;
    if (aset)
    {
      kanmod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
      kanmod = Core::LinAlg::matrix_multiply(*kanmod, false, *kan, false, false, false, true);
    }

    // kam: multiply tmatrix with invda and kam
    std::shared_ptr<Core::LinAlg::SparseMatrix> kammod;
    if (aset)
    {
      kammod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
      kammod = Core::LinAlg::matrix_multiply(*kammod, false, *kam, false, false, false, true);
    }

    // kai: multiply tmatrix with invda and kai
    std::shared_ptr<Core::LinAlg::SparseMatrix> kaimod;
    if (aset && iset)
    {
      kaimod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
      kaimod = Core::LinAlg::matrix_multiply(*kaimod, false, *kai, false, false, false, true);
    }

    // kaa: multiply tmatrix with invda and kaa
    std::shared_ptr<Core::LinAlg::SparseMatrix> kaamod;
    if (aset)
    {
      kaamod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
      kaamod = Core::LinAlg::matrix_multiply(*kaamod, false, *kaa, false, false, false, true);
    }

    /**********************************************************************/
    /* (8) Build the final f blocks                                       */
    /**********************************************************************/

    //----------------------------------------------------------- FIRST LINE
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

    //----------------------------------------------------------- THIRD LINE
    // fi: subtract alphaf * old contact forces (t_n)
    if (iset)
    {
      Core::LinAlg::Vector<double> fiadd(*gidofs);
      Core::LinAlg::export_to(fsadd, fiadd);
      fi->update(-alphaf_, fiadd, 1.0);
    }

    // fi: add T(dhat)*fa
    Core::LinAlg::Vector<double> fimod(*gidofs);
    if (iset && aset) dhat->multiply(true, *fa, fimod);
    fimod.update(1.0, *fi, -1.0);

    //---------------------------------------------------------- FOURTH LINE
    // gactive: nothing to do

    //----------------------------------------------------------- FIFTH LINE
    // fa: multiply tmatrix with invda and fa
    std::shared_ptr<Core::LinAlg::Vector<double>> famod;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tinvda;
    if (aset)
    {
      if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
        famod = std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
      else
        famod = std::make_shared<Core::LinAlg::Vector<double>>(*gactivet_);

      tinvda = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
      tinvda->multiply(false, *fa, *famod);
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
      if (aset) smatrix_ = Core::LinAlg::matrix_row_transform(*smatrix_, *non_redist_gsdofrowmap_);

      //----------------------------------------------------------- FIFTH LINE
      if (aset)
      {
        kanmod = Core::LinAlg::matrix_row_transform(*kanmod, *non_redist_gsdofrowmap_);
        kammod = Core::LinAlg::matrix_row_transform(*kammod, *non_redist_gsdofrowmap_);
        kaamod = Core::LinAlg::matrix_row_transform(*kaamod, *non_redist_gsdofrowmap_);
        if (iset) kaimod = Core::LinAlg::matrix_row_transform(*kaimod, *non_redist_gsdofrowmap_);
        tderivmatrix_ =
            Core::LinAlg::matrix_row_transform(*tderivmatrix_, *non_redist_gsdofrowmap_);
      }
    }

    /**********************************************************************/
    /* (10) Global setup of kteffnew (including contact)                  */
    /**********************************************************************/

    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
    std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
        Core::LinAlg::create_vector(*problem_dofs());

    //----------------------------------------------------------- FIRST LINE
    // add n submatrices to kteffnew
    kteffnew->add(*knn, false, 1.0, 1.0);
    kteffnew->add(*knm, false, 1.0, 1.0);
    if (sset) kteffnew->add(*kns, false, 1.0, 1.0);

    //---------------------------------------------------------- SECOND LINE
    // add m submatrices to kteffnew
    kteffnew->add(*kmnmod, false, 1.0, 1.0);
    kteffnew->add(*kmmmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kmimod, false, 1.0, 1.0);
    if (aset) kteffnew->add(*kmamod, false, 1.0, 1.0);

    //----------------------------------------------------------- THIRD LINE
    // add i submatrices to kteffnew
    if (iset) kteffnew->add(*kinmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kimmod, false, 1.0, 1.0);
    if (iset) kteffnew->add(*kiimod, false, 1.0, 1.0);
    if (iset && aset) kteffnew->add(*kiamod, false, 1.0, 1.0);

    //---------------------------------------------------------- FOURTH LINE
    // add a submatrices to kteffnew
    if (aset) kteffnew->add(*smatrix_, false, 1.0, 1.0);

    //----------------------------------------------------------- FIFTH LINE
    // add a submatrices to kteffnew
    if (aset) kteffnew->add(*kanmod, false, 1.0, 1.0);
    if (aset) kteffnew->add(*kammod, false, 1.0, 1.0);
    if (aset && iset) kteffnew->add(*kaimod, false, 1.0, 1.0);
    if (aset) kteffnew->add(*kaamod, false, 1.0, 1.0);
    if (aset) kteffnew->add(*tderivmatrix_, false, -1.0, 1.0);

    // fill_complete kteffnew (square)
    kteffnew->complete();

    /**********************************************************************/
    /* (11) Global setup of feffnew (including contact)                   */
    /**********************************************************************/

    //----------------------------------------------------------- FIRST LINE
    // add n subvector to feffnew
    Core::LinAlg::Vector<double> fnexp(*problem_dofs());
    Core::LinAlg::export_to(*fn, fnexp);
    feffnew->update(1.0, fnexp, 1.0);

    //---------------------------------------------------------- SECOND LINE
    // add m subvector to feffnew
    Core::LinAlg::Vector<double> fmmodexp(*problem_dofs());
    Core::LinAlg::export_to(fmmod, fmmodexp);
    feffnew->update(1.0, fmmodexp, 1.0);

    //----------------------------------------------------------- THIRD LINE
    // add i subvector to feffnew
    std::shared_ptr<Core::LinAlg::Vector<double>> fimodexp;
    if (iset)
    {
      fimodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(fimod, *fimodexp);
      feffnew->update(1.0, *fimodexp, 1.0);
    }

    //---------------------------------------------------------- FOURTH LINE
    // add weighted gap vector to feffnew, if existing
    std::shared_ptr<Core::LinAlg::Vector<double>> gexp;
    if (aset)
    {
      gexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(*gact, *gexp);
      feffnew->update(-1.0, *gexp, 1.0);
    }

    //----------------------------------------------------------- FIFTH LINE
    // add a subvector to feffnew
    std::shared_ptr<Core::LinAlg::Vector<double>> famodexp;
    if (aset)
    {
      famodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      Core::LinAlg::export_to(*famod, *famodexp);
      feffnew->update(1.0, *famodexp, 1.0);
    }

    // finally do the replacement
    kteff = kteffnew;
    feff = feffnew;
  }

  //************************************************************************
  //************************************************************************
  // CASE B: SADDLE POINT SYSTEM
  //************************************************************************
  //************************************************************************
  else
  {
    //----------------------------------------------------------------------
    // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
    //----------------------------------------------------------------------
    // Concretely, we apply the following transformations:
    // D         ---->   D * T^(-1)
    //----------------------------------------------------------------------
    if (is_dual_quad_slave_trafo())
    {
      if (lagmultquad == Inpar::Mortar::lagmult_lin)
      {
        // basis transformation
        Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
        std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
            Core::LinAlg::create_identity_matrix(*gndofrowmap_);
        systrafo.add(*eye, false, 1.0, 1.0);
        if (parallel_redistribution_status())
          trafo_ = Core::LinAlg::matrix_row_col_transform(
              *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
        systrafo.add(*trafo_, false, 1.0, 1.0);
        systrafo.complete();

        // apply basis transformation to K and f
        std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
            std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);
        std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
            std::make_shared<Core::LinAlg::SparseMatrix>(
                *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
        kteffnew =
            Core::LinAlg::matrix_multiply(*kteffmatrix, false, systrafo, false, false, false, true);
        kteffnew =
            Core::LinAlg::matrix_multiply(systrafo, true, *kteffnew, false, false, false, true);
        kteff = kteffnew;
        Core::LinAlg::Vector<double> feffnew(*feff);
        systrafo.multiply(true, feffnew, *feff);
      }
    }

    // transform if necessary
    if (parallel_redistribution_status())
    {
      lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
      linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
    }

    // add contact stiffness
    kteff->un_complete();
    kteff->add(*lindmatrix_, false, 1.0 - alphaf_, 1.0);
    kteff->add(*linmmatrix_, false, 1.0 - alphaf_, 1.0);
    kteff->complete();

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
      feff->update(-(1.0 - alphaf_), fsexp, 1.0);

      Core::LinAlg::Vector<double> fmexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecm(mmatrix_->domain_map());
      mmatrix_->multiply(true, zexp, tempvecm);
      Core::LinAlg::export_to(tempvecm, fmexp);
      feff->update(1.0 - alphaf_, fmexp, 1.0);

      // add old contact forces (t_n)
      Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecdold(dold_->domain_map());
      Core::LinAlg::Vector<double> zoldexp(dold_->row_map());
      if (dold_->row_map().num_global_elements()) Core::LinAlg::export_to(*zold_, zoldexp);
      dold_->multiply(true, zoldexp, tempvecdold);
      Core::LinAlg::export_to(tempvecdold, fsoldexp);
      feff->update(-alphaf_, fsoldexp, 1.0);

      Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
      Core::LinAlg::Vector<double> tempvecmold(mold_->domain_map());
      mold_->multiply(true, zoldexp, tempvecmold);
      Core::LinAlg::export_to(tempvecmold, fmoldexp);
      feff->update(alphaf_, fmoldexp, 1.0);
    }
    // if there is no self contact everything is ok
    else
    {
      // add contact force terms
      Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
      dmatrix_->multiply(true, *z_, fs);
      Core::LinAlg::Vector<double> fsexp(*problem_dofs());
      Core::LinAlg::export_to(fs, fsexp);
      feff->update(-(1.0 - alphaf_), fsexp, 1.0);

      Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
      mmatrix_->multiply(true, *z_, fm);
      Core::LinAlg::Vector<double> fmexp(*problem_dofs());
      Core::LinAlg::export_to(fm, fmexp);
      feff->update(1.0 - alphaf_, fmexp, 1.0);

      // add old contact forces (t_n)
      Core::LinAlg::Vector<double> fsold(*gsdofrowmap_);
      dold_->multiply(true, *zold_, fsold);
      Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
      Core::LinAlg::export_to(fsold, fsoldexp);
      feff->update(-alphaf_, fsoldexp, 1.0);

      Core::LinAlg::Vector<double> fmold(*gmdofrowmap_);
      mold_->multiply(true, *zold_, fmold);
      Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
      Core::LinAlg::export_to(fmold, fmoldexp);
      feff->update(alphaf_, fmoldexp, 1.0);
    }
  }

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckGapDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

#ifdef CONTACTFDALPHA
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckAlphaDeriv();
  }
#endif  // #ifdef CONTACTFDGAP
#ifdef CONTACTFDTANGLM
  // FD check of tangential LM derivatives (frictionless condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    std::cout << *tderivmatrix_ << std::endl;
    interface_[i]->FDCheckTangLMDeriv();
  }
#endif  // #ifdef CONTACTFDTANGLM

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::build_saddle_point_system(
    std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
    std::shared_ptr<Core::LinAlg::Vector<double>> fd,
    std::shared_ptr<Core::LinAlg::Vector<double>> sold,
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
    std::shared_ptr<Core::LinAlg::SparseOperator>& blockMat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
    std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs)
{
  // Check for saddle-point formulation
  if (system_type() != CONTACT::SystemType::saddlepoint)
    FOUR_C_THROW("Invalid system type! Cannot build a saddle-point system for this system type.");

  // create old style dirichtoggle vector (supposed to go away)
  // the use of a toggle vector is more flexible here. It allows to apply dirichlet
  // conditions on different matrix blocks separately.
  Core::LinAlg::Vector<double> dirichtoggle(*(dbcmaps->full_map()));
  Core::LinAlg::Vector<double> temp(*(dbcmaps->cond_map()));
  temp.put_scalar(1.0);
  Core::LinAlg::export_to(temp, dirichtoggle);

  // Initialize constraint matrices
  Core::LinAlg::SparseMatrix kzz(*gsdofrowmap_, 100, true, true);
  Core::LinAlg::SparseMatrix kzd(*gsdofrowmap_, 100, false, true);
  Core::LinAlg::SparseMatrix kdz(*gdisprowmap_, 100, false, true);

  // Declare transformed constraint matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkdz = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkzd = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> trkzz = nullptr;

  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // build constraint matrix kdz
  kdz.add(*dmatrix_, true, 1.0 - alphaf_, 1.0);
  kdz.add(*mmatrix_, true, -(1.0 - alphaf_), 1.0);
  kdz.complete(*gsdofrowmap_, *gdisprowmap_);

  // *** CASE 1: FRICTIONLESS CONTACT ************************************
  if (!friction_)
  {
    // build constraint matrix kzd
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    {
      if (gactivedofs_->num_global_elements())
      {
        kzd.add(*smatrix_, false, 1.0, 1.0);
        kzd.add(*tderivmatrix_, false, 1.0, 1.0);
      }
    }
    else
    {
      if (gactiven_->num_global_elements()) kzd.add(*smatrix_, false, 1.0, 1.0);
      if (gactivet_->num_global_elements()) kzd.add(*tderivmatrix_, false, 1.0, 1.0);
    }
    kzd.complete(*gdisprowmap_, *gsdofrowmap_);

    // build unity matrix for inactive dofs
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    Core::LinAlg::Vector<double> ones(*gidofs);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    // build constraint matrix kzz
    if (gidofs->num_global_elements()) kzz.add(onesdiag, false, 1.0, 1.0);
    if (gactivet_->num_global_elements()) kzz.add(*tmatrix_, false, 1.0, 1.0);
    kzz.complete(*gsdofrowmap_, *gsdofrowmap_);
  }

  //**********************************************************************
  // build matrix and vector blocks
  //**********************************************************************
  // *** CASE 2: FRICTIONAL CONTACT **************************************
  else
  {
    // global stick dof map
    std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
    std::shared_ptr<Core::LinAlg::Map> gstickdofs =
        Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);

    // build constraint matrix kzd
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    {
      if (gactivedofs_->num_global_elements()) kzd.add(*smatrix_, false, 1.0, 1.0);
      if (gstickdofs->num_global_elements()) kzd.add(*linstickDIS_, false, 1.0, 1.0);
      if (gslipdofs_->num_global_elements()) kzd.add(*linslipDIS_, false, 1.0, 1.0);
    }
    else
    {
      if (gactiven_->num_global_elements()) kzd.add(*smatrix_, false, 1.0, 1.0);
      if (gstickt->num_global_elements()) kzd.add(*linstickDIS_, false, 1.0, 1.0);
      if (gslipt_->num_global_elements()) kzd.add(*linslipDIS_, false, 1.0, 1.0);
    }
    kzd.complete(*gdisprowmap_, *gsdofrowmap_);

    // build unity matrix for inactive dofs
    std::shared_ptr<Core::LinAlg::Map> gidofs =
        Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
    Core::LinAlg::Vector<double> ones(*gidofs);
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    // build constraint matrix kzz
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
    {
      if (gidofs->num_global_elements()) kzz.add(onesdiag, false, 1.0, 1.0);
      if (gstickdofs->num_global_elements()) kzz.add(*linstickLM_, false, 1.0, 1.0);
      if (gslipdofs_->num_global_elements()) kzz.add(*linslipLM_, false, 1.0, 1.0);
    }
    else
    {
      if (gidofs->num_global_elements()) kzz.add(onesdiag, false, 1.0, 1.0);
      if (gstickt->num_global_elements()) kzz.add(*linstickLM_, false, 1.0, 1.0);
      if (gslipt_->num_global_elements()) kzz.add(*linslipLM_, false, 1.0, 1.0);
    }
    kzz.complete(*gsdofrowmap_, *gsdofrowmap_);
  }

  /* Step 1: Transform matrix blocks to current Lagrange multiplier dof_row_map
   *
   * This does not change the parallel layout, but only replace the slave displacement GIDs with the
   * Lagrange multiplier DOF GIDs. There is no communication involved.
   */
  {
    // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
    trkzd = Core::LinAlg::matrix_row_transform_gids(kzd, *glmdofrowmap_);

    // transform constraint matrix kzz to lmdofmap (matrix_row_col_transform)
    trkzz = Core::LinAlg::matrix_row_col_transform_gids(kzz, *glmdofrowmap_, *glmdofrowmap_);

    // transform constraint matrix kzd to lmdofmap (matrix_col_transform)
    trkdz = Core::LinAlg::matrix_col_transform_gids(kdz, *glmdofrowmap_);
  }

  /* Step 2: Transform matrix blocks back to original dof_row_map as it was prior to the
   * redistribution of the interface
   *
   * Now, we keep the GID numbering, but change the parallel layout. This actually moves data
   * between MPI ranks.
   */
  if (parallel_redistribution_status())
  {
    trkzd =
        Core::LinAlg::matrix_row_col_transform(*trkzd, *non_redist_glmdofrowmap_, *problem_dofs());
    trkzz = Core::LinAlg::matrix_row_col_transform(
        *trkzz, *non_redist_glmdofrowmap_, *non_redist_glmdofrowmap_);
    trkdz =
        Core::LinAlg::matrix_row_col_transform(*trkdz, *problem_dofs(), *non_redist_glmdofrowmap_);
  }

  // Assemble the saddle point system
  {
    // Get the standard stiffness matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiffmt =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kdd);

    // Initialize merged system (matrix, rhs, sol)
    std::shared_ptr<Core::LinAlg::Map> mergedmap = nullptr;
    if (parallel_redistribution_status())
      mergedmap = Core::LinAlg::merge_map(problem_dofs(), non_redist_glmdofrowmap_, false);
    else
      mergedmap = Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);

    std::shared_ptr<Core::LinAlg::Vector<double>> mergedrhs =
        Core::LinAlg::create_vector(*mergedmap);
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedsol =
        Core::LinAlg::create_vector(*mergedmap);
    std::shared_ptr<Core::LinAlg::Vector<double>> mergedzeros =
        Core::LinAlg::create_vector(*mergedmap);

    /* ToDo (mayr.mt) Is this due to symmetry BCs? Basically, slave DOFs should not carry any
     * Dirichlet BCs.
     */
    Core::LinAlg::Vector<double> dirichtoggleexp(*mergedmap);
    {
      Core::LinAlg::export_to(dirichtoggle, dirichtoggleexp);
      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC = nullptr;
      if (parallel_redistribution_status())
        lmDBC = std::make_shared<Core::LinAlg::Vector<double>>(*non_redist_gsdofrowmap_, true);
      else
        lmDBC = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
      Core::LinAlg::export_to(dirichtoggle, *lmDBC);

      if (parallel_redistribution_status())
        lmDBC->replace_map(*non_redist_glmdofrowmap_);
      else
        lmDBC->replace_map(*glmdofrowmap_);

      Core::LinAlg::Vector<double> lmDBCexp(*mergedmap);
      Core::LinAlg::export_to(*lmDBC, lmDBCexp);
      dirichtoggleexp.update(1., lmDBCexp, 1.);
      trkzd->apply_dirichlet(*lmDBC, false);

      trkzz->complete();
      trkzz->apply_dirichlet(*lmDBC, true);

      // apply Dirichlet conditions to (0,0) and (0,1) blocks
      Core::LinAlg::Vector<double> zeros(*problem_dofs(), true);
      Core::LinAlg::Vector<double> rhscopy(*fd);
      Core::LinAlg::apply_dirichlet_to_system(*stiffmt, *sold, rhscopy, zeros, dirichtoggle);
      trkdz->apply_dirichlet(dirichtoggle, false);
    }

    // row map (equals domain map) extractor
    std::shared_ptr<Core::LinAlg::MapExtractor> rowmapext = nullptr;
    std::shared_ptr<Core::LinAlg::MapExtractor> dommapext = nullptr;
    if (parallel_redistribution_status())
    {
      rowmapext = std::make_shared<Core::LinAlg::MapExtractor>(
          *mergedmap, non_redist_glmdofrowmap_, problem_dofs());
      dommapext = std::make_shared<Core::LinAlg::MapExtractor>(
          *mergedmap, non_redist_glmdofrowmap_, problem_dofs());
    }
    else
    {
      rowmapext =
          std::make_shared<Core::LinAlg::MapExtractor>(*mergedmap, glmdofrowmap_, problem_dofs());
      dommapext =
          std::make_shared<Core::LinAlg::MapExtractor>(*mergedmap, glmdofrowmap_, problem_dofs());
    }

    // build block matrix for SIMPLER
    blockMat =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *dommapext, *rowmapext, 81, false, false);
    // blockMat is declared as an Epetra_Operator, so we need to cast it to an actual block matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>> mat =
        std::dynamic_pointer_cast<
            Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(blockMat);
    mat->assign(0, 0, Core::LinAlg::DataAccess::Share, *stiffmt);
    mat->assign(0, 1, Core::LinAlg::DataAccess::Share, *trkdz);
    mat->assign(1, 0, Core::LinAlg::DataAccess::Share, *trkzd);
    mat->assign(1, 1, Core::LinAlg::DataAccess::Share, *trkzz);
    mat->complete();

    // we also need merged rhs here
    Core::LinAlg::Vector<double> fresmexp(*mergedmap);
    Core::LinAlg::export_to(*fd, fresmexp);
    mergedrhs->update(1.0, fresmexp, 1.0);
    Core::LinAlg::Vector<double> constrexp(*mergedmap);
    Core::LinAlg::export_to(*constrrhs_, constrexp);
    mergedrhs->update(1.0, constrexp, 1.0);

    // apply Dirichlet B.C. to mergedrhs and mergedsol
    Core::LinAlg::apply_dirichlet_to_system(*mergedsol, *mergedrhs, *mergedzeros, dirichtoggleexp);

    blocksol = mergedsol;
    blockrhs = mergedrhs;
  }

  return;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::update_displacements_and_l_mincrements(
    std::shared_ptr<Core::LinAlg::Vector<double>> sold,
    std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol)
{
  // Extract results for displacement and LM increments
  std::shared_ptr<Core::LinAlg::Vector<double>> sollm = nullptr;
  if (parallel_redistribution_status())
  {
    Core::LinAlg::Vector<double> sollmOrig(*non_redist_glmdofrowmap_);
    std::shared_ptr<Core::LinAlg::Map> mergedmapOrig =
        Core::LinAlg::merge_map(problem_dofs(), non_redist_glmdofrowmap_, false);
    Core::LinAlg::MapExtractor mapext(*mergedmapOrig, problem_dofs(), non_redist_glmdofrowmap_);
    mapext.extract_cond_vector(*blocksol, *sold);
    mapext.extract_other_vector(*blocksol, sollmOrig);

    sollm = std::make_shared<Core::LinAlg::Vector<double>>(*glmdofrowmap_);
    Core::LinAlg::export_to(sollmOrig, *sollm);
    sollm->replace_map(*gsdofrowmap_);
  }
  else
  {
    sollm = std::make_shared<Core::LinAlg::Vector<double>>(*glmdofrowmap_);
    std::shared_ptr<Core::LinAlg::Map> mergedmap =
        Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
    Core::LinAlg::MapExtractor mapext(*mergedmap, problem_dofs(), glmdofrowmap_);
    mapext.extract_cond_vector(*blocksol, *sold);
    mapext.extract_other_vector(*blocksol, *sollm);
    sollm->replace_map(*gsdofrowmap_);
  }

  /* For self contact, slave and master sets may have changed, thus we have to reinitialize the LM
   * vector map
   */
  if (is_self_contact())
  {
    zincr_ = std::make_shared<Core::LinAlg::Vector<double>>(*sollm);
    Core::LinAlg::export_to(*z_, *zincr_);  // change the map of z_
    z_ = std::make_shared<Core::LinAlg::Vector<double>>(*zincr_);
    zincr_->update(1.0, *sollm, 0.0);  // save sollm in zincr_
    z_->update(1.0, *zincr_, 1.0);     // update z_
  }
  else
  {
    zincr_->update(1.0, *sollm, 0.0);
    z_->update(1.0, *zincr_, 1.0);
  }

  return;
}

/*----------------------------------------------------------------------*
 | calculate constraint RHS entries                      hiermeier 08/13|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::evaluate_constr_rhs()
{
  if (system_type() == CONTACT::SystemType::condensed) return;

  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step())
  {
    // (re)setup the vector
    constrrhs_ = nullptr;
    return;
  }

  // initialize constraint r.h.s. (still with wrong map)
  std::shared_ptr<Core::LinAlg::Vector<double>> constrrhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);

  // We solve for the incremental Lagrange multiplier dz_. Hence,
  // we can keep the contact force terms on the right-hand side!
  //
  // r = r_effdyn,co = r_effdyn + a_f * B_co(d_(n)) * z_(n) + (1-a_f) * B_co(d^(i)_(n+1)) *
  // z^(i)_(n+1)

  // export weighted gap vector
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*gactivedofs_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
    }
  }
  else
  {
    gact = Core::LinAlg::create_vector(*gactivenodes_, true);
    if (gactiven_->num_global_elements())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
  }

  Core::LinAlg::Vector<double> gact_exp(*gsdofrowmap_);
  Core::LinAlg::export_to(*gact, gact_exp);

  constrrhs->update(-1.0, gact_exp, 1.0);

  // export inactive rhs
  Core::LinAlg::Vector<double> inactiverhsexp(*gsdofrowmap_);
  Core::LinAlg::export_to(*inactiverhs_, inactiverhsexp);

  // build constraint rhs (1)
  constrrhs->update(1.0, inactiverhsexp, 1.0);

  // *** CASE 1: FRICTIONLESS CONTACT *******************************************************
  if (!friction_)
  {
    // export tangential rhs
    Core::LinAlg::Vector<double> tangrhs_exp(*gsdofrowmap_);
    Core::LinAlg::export_to(*tangrhs_, tangrhs_exp);

    // build constraint rhs (2)
    constrrhs->update(1.0, tangrhs_exp, 1.0);
  }
  // *** CASE 2: FRICTIONAL CONTACT *******************************************************
  else
  {
    // export stick and slip r.h.s.
    Core::LinAlg::Vector<double> stickexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linstickRHS_, stickexp);
    Core::LinAlg::Vector<double> slipexp(*gsdofrowmap_);
    Core::LinAlg::export_to(*linslipRHS_, slipexp);

    // build constraint rhs
    constrrhs->update(1.0, stickexp, 1.0);
    constrrhs->update(1.0, slipexp, 1.0);
  }

  constrrhs->replace_map(*glmdofrowmap_);

  // export and set constraint rhs vector
  if (parallel_redistribution_status())
  {
    constrrhs_ = std::make_shared<Core::LinAlg::Vector<double>>(lm_dof_row_map(false));
    Core::LinAlg::export_to(*constrrhs, *constrrhs_);
  }
  else
    constrrhs_ = constrrhs;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::evaluate_force(CONTACT::ParamsInterface& cparams)
{
  //---------------------------------------------------------------
  // For selfcontact the master/slave sets are updated within the -
  // contact search, see SelfBinaryTree.                          -
  // Therefore, we have to initialize the mortar matrices after   -
  // interface evaluations.                                       -
  //---------------------------------------------------------------
  if (is_self_contact())
  {
    initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
    initialize_mortar();                  // initialize mortar matrices and vectors
    assemble_mortar();                    // assemble mortar terms into global matrices
  }
  else
  {
    initialize_mortar();                  // initialize mortar matrices and vectors
    initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
    assemble_mortar();                    // assemble mortar terms into global matrices
  }

  // evaluate relative movement for friction
  if (cparams.is_predictor())
    predict_relative_movement();
  else
    evaluate_relative_movement();

  // update active set
  const bool firstTimeStepAndPredictor =
      (cparams.is_predictor() and (cparams.get_step_np() == cparams.get_restart_step() + 1));
  update_active_set_semi_smooth(firstTimeStepAndPredictor);

  // apply contact forces and stiffness
  initialize();  // init lin-matrices
  assemble_all_contact_terms();

  if (system_type() != CONTACT::SystemType::condensed)
  {
    eval_str_contact_rhs();  // evaluate the structure/displacement rhs
    evaluate_constr_rhs();   // evaluate the constraint rhs (saddle-point system only)

    if (constrrhs_ != nullptr)
      constrrhs_->scale(-1.0);  // scale with -1.0 --> when old structure is deleted change this!!!
  }
  else
    eval_str_contact_rhs();

  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
  {
    systrafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
        Core::LinAlg::create_identity_matrix(*gndofrowmap_);
    systrafo_->add(*eye, false, 1.0, 1.0);
    if (parallel_redistribution_status())
      trafo_ = Core::LinAlg::matrix_row_col_transform(
          *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
    systrafo_->add(*trafo_, false, 1.0, 1.0);
    systrafo_->complete();

    invsystrafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);
    invsystrafo_->add(*eye, false, 1.0, 1.0);
    if (parallel_redistribution_status())
      invtrafo_ = Core::LinAlg::matrix_row_col_transform(
          *invtrafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
    invsystrafo_->add(*invtrafo_, false, 1.0, 1.0);
    invsystrafo_->complete();
  }

  // bye bye
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::assemble_all_contact_terms()
{
  if (nonSmoothContact_)
  {
    nonsmooth_Penalty_force_ = std::make_shared<Core::LinAlg::Vector<double>>(*probdofs_);
    nonsmooth_Penalty_stiff_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*probdofs_, 100, true, true);
    std::shared_ptr<Core::LinAlg::SparseOperator> k =
        std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(nonsmooth_Penalty_stiff_);
    if (!friction_)
    {
      // LTL contributions:
      add_line_to_lin_contributions(*k, nonsmooth_Penalty_force_, false);

      // penalty support for master side quantities:
      add_master_contributions(*k, *nonsmooth_Penalty_force_, false);
    }
    else
    {
      // add_line_to_lin_contributions(kteff,feff);
      add_line_to_lin_contributions_friction(*k, nonsmooth_Penalty_force_, false);
    }

    nonsmooth_Penalty_stiff_->complete();
    nonsmooth_Penalty_force_->scale(-1.);
  }

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  if (friction_)
    assemble_all_contact_terms_friction();
  else
    assemble_all_contact_terms_frictionless();
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::assemble_all_contact_terms_friction()
{
  /**********************************************************************/
  /* build global matrix t with tangent vectors of active nodes         */
  /* and global matrix s with normal derivatives of active nodes        */
  /* and global matrix linstick with derivatives of stick nodes         */
  /* and global matrix linslip with derivatives of slip nodes           */
  /* and inactive right-hand side with old lagrange multipliers (incr)  */
  /**********************************************************************/
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->assemble_s(*smatrix_);
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);
    interface_[i]->assemble_lin_stick(*linstickLM_, *linstickDIS_, *linstickRHS_);
    interface_[i]->assemble_lin_slip(*linslipLM_, *linslipDIS_, *linslipRHS_);
    if (system_type() != CONTACT::SystemType::condensed)
      interface_[i]->assemble_inactiverhs(*inactiverhs_);
  }
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    smatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
  }
  else
  {
    // fill_complete() global matrix S
    smatrix_->complete(*gsmdofrowmap_, *gactiven_);
  }

  // fill_complete() global matrices LinD, LinM
  // (again for linD gsdofrowmap_ is sufficient as domain map,
  // but in the edge node modification case, master entries occur!)
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);

  // fill_complete global Matrix linstickLM_, linstickDIS_
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);

  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    linstickLM_->complete(*gstickdofs, *gstickdofs);
    linstickDIS_->complete(*gsmdofrowmap_, *gstickdofs);
    linslipLM_->complete(*gslipdofs_, *gslipdofs_);
    linslipDIS_->complete(*gsmdofrowmap_, *gslipdofs_);
  }
  else
  {
    linstickLM_->complete(*gstickdofs, *gstickt);
    linstickDIS_->complete(*gsmdofrowmap_, *gstickt);

    // fill_complete global Matrix linslipLM_ and linslipDIS_
    linslipLM_->complete(*gslipdofs_, *gslipt_);
    linslipDIS_->complete(*gsmdofrowmap_, *gslipt_);
  }

  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (is_dual_quad_slave_trafo())
  {
    if (lagmultquad == Inpar::Mortar::lagmult_lin)
      FOUR_C_THROW("no linear LM interpolation for frictional contact");
    else
    {
      // modify lindmatrix_
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
      lindmatrix_ = temp1;
    }
  }

  if (system_type() == CONTACT::SystemType::condensed)
  {
    /**********************************************************************/
    /* (1) Multiply Mortar matrices: m^ = inv(d) * m                      */
    /**********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);

    // for nonsmooth contact inverting D is more complex:
    // Note: this inversion if only applicable when vertex, edge and surface nodes
    // are involved. For a falling coin (only surface and edge nodes), a special but
    // more easy implementation is needed.
    if (nonSmoothContact_)
    {
      // 1. split d matrix in vertex edge and surf part
      std::shared_ptr<Core::LinAlg::SparseMatrix> dss, dsev, devs, devev;
      std::shared_ptr<Core::LinAlg::Map> gEVdofs;  // merged edge and vertex dofs

      // get dss
      Core::LinAlg::split_matrix2x2(
          dmatrix_, gsdofSurf_, gEVdofs, gsdofSurf_, gEVdofs, dss, dsev, devs, devev);

      // get dse and dsv
      std::shared_ptr<Core::LinAlg::Map> temp;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dse, dsv;

      Core::LinAlg::split_matrix2x2(
          dsev, gsdofSurf_, temp, gsdofEdge_, gsdofVertex_, dse, dsv, tempmtx1, tempmtx2);

      // get dee dev dve dvv
      std::shared_ptr<Core::LinAlg::SparseMatrix> dee, dev, dve, dvv;
      Core::LinAlg::split_matrix2x2(
          devev, gsdofEdge_, gsdofVertex_, gsdofEdge_, gsdofVertex_, dee, dev, dve, dvv);

      // 2. invert diagonal matrices dss dee dvv
      std::shared_ptr<Core::LinAlg::Vector<double>> diagV =
          Core::LinAlg::create_vector(*gsdofVertex_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagE =
          Core::LinAlg::create_vector(*gsdofEdge_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagS =
          Core::LinAlg::create_vector(*gsdofSurf_, true);
      Core::LinAlg::SparseMatrix invdV(*dvv);
      Core::LinAlg::SparseMatrix invdE(*dee);
      Core::LinAlg::SparseMatrix invdS(*dss);

      int err = 0;

      // extract diagonal of invd into diag
      invdV.extract_diagonal_copy(*diagV);
      invdE.extract_diagonal_copy(*diagE);
      invdS.extract_diagonal_copy(*diagS);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diagV->local_length(); ++i)
        if (abs((*diagV).get_values()[i]) < 1e-12) (*diagV).get_values()[i] = 1.0;
      for (int i = 0; i < diagE->local_length(); ++i)
        if (abs((*diagE).get_values()[i]) < 1e-12) (*diagE).get_values()[i] = 1.0;
      for (int i = 0; i < diagS->local_length(); ++i)
        if (abs((*diagS).get_values()[i]) < 1e-12) (*diagS).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diagV->reciprocal(*diagV);
      diagE->reciprocal(*diagE);
      diagS->reciprocal(*diagS);

      // re-insert inverted diagonal into invd
      err = invdV.replace_diagonal_values(*diagV);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdE.replace_diagonal_values(*diagE);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdS.replace_diagonal_values(*diagS);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

      // 3. multiply all sub matrices
      invd = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100, true, true);

      dse->scale(-1.0);
      dsv->scale(-1.0);
      dev->scale(-1.0);

      // inv_dse
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det1;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dse;
      inv_det1 = Core::LinAlg::matrix_multiply(*dse, false, invdE, false, false, false, true);
      dinv_dse = Core::LinAlg::matrix_multiply(invdS, false, *inv_det1, false, false, false, true);

      // inv_dev
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dev;
      inv_det2 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      dinv_dev = Core::LinAlg::matrix_multiply(invdE, false, *inv_det2, false, false, false, true);

      // inv_dsv part1
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det3;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det4;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det5;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv1;
      inv_det3 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      inv_det4 = Core::LinAlg::matrix_multiply(invdE, false, *inv_det3, false, false, false, true);
      inv_det5 = Core::LinAlg::matrix_multiply(*dse, false, *inv_det4, false, false, false, true);
      dinv_dsv1 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det5, false, false, false, true);

      // inv_dsv part2
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det6;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv2;
      inv_det6 = Core::LinAlg::matrix_multiply(*dsv, false, invdV, false, false, false, true);
      dinv_dsv2 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det6, false, false, false, true);

      // diagonal entries
      invd->add(invdS, false, 1.0, 1.0);
      invd->add(invdE, false, 1.0, 1.0);
      invd->add(invdV, false, 1.0, 1.0);

      invd->add(*dinv_dev, false, 1.0, 1.0);
      invd->add(*dinv_dse, false, 1.0, 1.0);
      invd->add(*dinv_dsv1, false, 1.0, 1.0);
      invd->add(*dinv_dsv2, false, 1.0, 1.0);

      // complete
      invd->complete();
    }
    // standard inverse diagonal matrix:
    else
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      int err = 0;

      // extract diagonal of invd into diag
      invd->extract_diagonal_copy(*diag);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diag->local_length(); ++i)
        if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diag->reciprocal(*diag);

      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      Core::LinAlg::export_to(*non_redist_gsdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);

      // re-insert inverted diagonal into invd
      err = invd->replace_diagonal_values(*diag);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
    }

    invd_ = invd;

    // do the multiplication mhat = inv(D) * M
    mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd, false, *mmatrix_, false, false, false, true);
  }

  if (is_dual_quad_slave_trafo() && lagmultquad != Inpar::Mortar::lagmult_lin)
  {
    // modify dmatrix_, invd_ and mhatmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    dmatrix_ = temp2;
    if (system_type() == CONTACT::SystemType::condensed)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp3 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *invd_, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp4 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *mhatmatrix_, false, false, false, true);
      invd_ = temp3;
      mhatmatrix_ = temp4;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::assemble_all_contact_terms_frictionless()
{
  /**********************************************************************/
  /* calculate                                                          */
  /**********************************************************************/
  /* build global matrix tmatrix_ with tangent vectors of active nodes  */
  /* and global matrix nmatrix_ with normal vectors of active nodes     */
  /* and global matrix s with normal+D+M derivatives of active nodes    */
  /* and global matrix tderivmatrix_ with tangent derivatives           */
  /*     of active nodes                                                */
  /* and global matrix nderivmatrix_ with normal derivatives            */
  /*     of active nodes                                                */
  /* and inactive right-hand side with old lagrange multipliers (incr)  */
  /* and tangential right-hand side (incr)                              */
  /**********************************************************************/
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->assemble_tn(tmatrix_, nmatrix_);
    interface_[i]->assemble_s(*smatrix_);
    interface_[i]->assemble_t_nderiv(tderivmatrix_, nderivmatrix_);
    interface_[i]->assemble_lin_dm(*lindmatrix_, *linmmatrix_);

    if (system_type() != CONTACT::SystemType::condensed)
    {
      interface_[i]->assemble_inactiverhs(*inactiverhs_);
      interface_[i]->assemble_tangrhs(*tangrhs_);
    }
  }
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    // fill_complete() global matrix T
    tmatrix_->complete(*gactivedofs_, *gactivedofs_);

    // fill_complete() global matrix N
    if (nmatrix_ != nullptr) nmatrix_->complete(*gactivedofs_, *gactivedofs_);
    smatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
    tderivmatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
    if (nderivmatrix_ != nullptr) nderivmatrix_->complete(*gsmdofrowmap_, *gactivedofs_);
  }
  else
  {
    // fill_complete() global matrix T
    tmatrix_->complete(*gactivedofs_, *gactivet_);

    // fill_complete() global matrix N
    if (nmatrix_ != nullptr) nmatrix_->complete(*gactivedofs_, *gactiven_);

    // fill_complete() global matrix S
    smatrix_->complete(*gsmdofrowmap_, *gactiven_);

    // fill_complete() global matrix Tderiv
    // (actually gsdofrowmap_ is in general sufficient as domain map,
    // but in the edge node modification case, master entries occur!)
    tderivmatrix_->complete(*gsmdofrowmap_, *gactivet_);

    // fill_complete() global matrix Nderiv
    if (nderivmatrix_ != nullptr) nderivmatrix_->complete(*gsmdofrowmap_, *gactiven_);
  }

  // fill_complete() global matrices LinD, LinM
  // (again for linD gsdofrowmap_ is sufficient as domain map,
  // but in the edge node modification case, master entries occur!)
  lindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
  linmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);

  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (is_dual_quad_slave_trafo())
  {
    if (lagmultquad == Inpar::Mortar::lagmult_lin)
    {
    }
    else
    {
      // modify lindmatrix_
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*invtrafo_, true, *lindmatrix_, false, false, false, true);
      lindmatrix_ = temp1;
    }
  }

  if (system_type() == CONTACT::SystemType::condensed)
  {
    /**********************************************************************/
    /* (1) Multiply Mortar matrices: m^ = inv(d) * m                      */
    /**********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);

    // for nonsmooth contact inverting D is more complex:
    // Note: this inversion if only applicable when vertex, edge and surface nodes
    // are involved. For a falling coin (only surface and edge nodes), a special but
    // more easy implementation is needed.
    if (nonSmoothContact_)
    {
      // 1. split d matrix in vertex edge and surf part
      std::shared_ptr<Core::LinAlg::SparseMatrix> dss, dsev, devs, devev;
      std::shared_ptr<Core::LinAlg::Map> gEVdofs;  // merged edge and vertex dofs

      // get dss
      Core::LinAlg::split_matrix2x2(
          dmatrix_, gsdofSurf_, gEVdofs, gsdofSurf_, gEVdofs, dss, dsev, devs, devev);

      // get dse and dsv
      std::shared_ptr<Core::LinAlg::Map> temp;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dse, dsv;

      Core::LinAlg::split_matrix2x2(
          dsev, gsdofSurf_, temp, gsdofEdge_, gsdofVertex_, dse, dsv, tempmtx1, tempmtx2);

      // get dee dev dve dvv
      std::shared_ptr<Core::LinAlg::SparseMatrix> dee, dev, dve, dvv;
      Core::LinAlg::split_matrix2x2(
          devev, gsdofEdge_, gsdofVertex_, gsdofEdge_, gsdofVertex_, dee, dev, dve, dvv);

      // 2. invert diagonal matrices dss dee dvv
      std::shared_ptr<Core::LinAlg::Vector<double>> diagV =
          Core::LinAlg::create_vector(*gsdofVertex_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagE =
          Core::LinAlg::create_vector(*gsdofEdge_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagS =
          Core::LinAlg::create_vector(*gsdofSurf_, true);
      Core::LinAlg::SparseMatrix invdV(*dvv);
      Core::LinAlg::SparseMatrix invdE(*dee);
      Core::LinAlg::SparseMatrix invdS(*dss);

      int err = 0;

      // extract diagonal of invd into diag
      invdV.extract_diagonal_copy(*diagV);
      invdE.extract_diagonal_copy(*diagE);
      invdS.extract_diagonal_copy(*diagS);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diagV->local_length(); ++i)
        if (abs((*diagV).get_values()[i]) < 1e-12) (*diagV).get_values()[i] = 1.0;
      for (int i = 0; i < diagE->local_length(); ++i)
        if (abs((*diagE).get_values()[i]) < 1e-12) (*diagE).get_values()[i] = 1.0;
      for (int i = 0; i < diagS->local_length(); ++i)
        if (abs((*diagS).get_values()[i]) < 1e-12) (*diagS).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diagV->reciprocal(*diagV);
      diagE->reciprocal(*diagE);
      diagS->reciprocal(*diagS);

      // re-insert inverted diagonal into invd
      err = invdV.replace_diagonal_values(*diagV);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdE.replace_diagonal_values(*diagE);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
      err = invdS.replace_diagonal_values(*diagS);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

      // 3. multiply all sub matrices
      invd = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100, true, true);

      dse->scale(-1.0);
      dsv->scale(-1.0);
      dev->scale(-1.0);

      // inv_dse
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det1;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dse;
      inv_det1 = Core::LinAlg::matrix_multiply(*dse, false, invdE, false, false, false, true);
      dinv_dse = Core::LinAlg::matrix_multiply(invdS, false, *inv_det1, false, false, false, true);

      // inv_dev
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dev;
      inv_det2 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      dinv_dev = Core::LinAlg::matrix_multiply(invdE, false, *inv_det2, false, false, false, true);

      // inv_dsv part1
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det3;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det4;
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det5;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv1;
      inv_det3 = Core::LinAlg::matrix_multiply(*dev, false, invdV, false, false, false, true);
      inv_det4 = Core::LinAlg::matrix_multiply(invdE, false, *inv_det3, false, false, false, true);
      inv_det5 = Core::LinAlg::matrix_multiply(*dse, false, *inv_det4, false, false, false, true);
      dinv_dsv1 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det5, false, false, false, true);

      // inv_dsv part2
      std::shared_ptr<Core::LinAlg::SparseMatrix> inv_det6;
      std::shared_ptr<Core::LinAlg::SparseMatrix> dinv_dsv2;
      inv_det6 = Core::LinAlg::matrix_multiply(*dsv, false, invdV, false, false, false, true);
      dinv_dsv2 = Core::LinAlg::matrix_multiply(invdS, false, *inv_det6, false, false, false, true);

      // diagonal entries
      invd->add(invdS, false, 1.0, 1.0);
      invd->add(invdE, false, 1.0, 1.0);
      invd->add(invdV, false, 1.0, 1.0);

      invd->add(*dinv_dev, false, 1.0, 1.0);
      invd->add(*dinv_dse, false, 1.0, 1.0);
      invd->add(*dinv_dsv1, false, 1.0, 1.0);
      invd->add(*dinv_dsv2, false, 1.0, 1.0);

      invd->complete();
    }
    // standard inverse diagonal matrix:
    else
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      int err = 0;

      // extract diagonal of invd into diag
      invd->extract_diagonal_copy(*diag);

      // set zero diagonal values to dummy 1.0
      for (int i = 0; i < diag->local_length(); ++i)
        if (diag->local_values_as_span()[i] == 0.0) (*diag).get_values()[i] = 1.0;

      // scalar inversion of diagonal values
      diag->reciprocal(*diag);

      std::shared_ptr<Core::LinAlg::Vector<double>> lmDBC =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      Core::LinAlg::export_to(*non_redist_gsdirichtoggle_, *lmDBC);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      tmp->multiply(1., *diag, *lmDBC, 0.);
      diag->update(-1., *tmp, 1.);

      // re-insert inverted diagonal into invd
      err = invd->replace_diagonal_values(*diag);
      if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);
    }

    invd_ = invd;

    // do the multiplication mhat = inv(D) * M
    mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd, false, *mmatrix_, false, false, false, true);
  }

  if (is_dual_quad_slave_trafo() && lagmultquad != Inpar::Mortar::lagmult_lin)
  {
    // modify dmatrix_, invd_ and mhatmatrix_
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
        Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
    dmatrix_ = temp2;
    if (system_type() == CONTACT::SystemType::condensed)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp3 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *invd_, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp4 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *mhatmatrix_, false, false, false, true);
      invd_ = temp3;
      mhatmatrix_ = temp4;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::assemble_contact_rhs()
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (system_type() != CONTACT::SystemType::condensed)
    {
      interface_[i]->assemble_inactiverhs(*inactiverhs_);
      if (!is_friction()) interface_[i]->assemble_tangrhs(*tangrhs_);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::eval_str_contact_rhs()
{
  if (!is_in_contact() and !was_in_contact() and !was_in_contact_last_time_step())
  {
    strcontactrhs_ = nullptr;
    return;
  }

  strcontactrhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs(), true);

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
    strcontactrhs_->update(1., fsexp, 1.0);

    Core::LinAlg::Vector<double> fmexp(*problem_dofs());
    Core::LinAlg::Vector<double> tempvecm(mmatrix_->domain_map());
    mmatrix_->multiply(true, zexp, tempvecm);
    Core::LinAlg::export_to(tempvecm, fmexp);
    strcontactrhs_->update(-1., fmexp, 1.0);
  }
  // if there is no self contact everything is ok
  else
  {
    // add contact force terms
    Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
    dmatrix_->multiply(true, *z_, fs);
    Core::LinAlg::Vector<double> fsexp(*problem_dofs());
    Core::LinAlg::export_to(fs, fsexp);
    strcontactrhs_->update(+1., fsexp, 1.0);

    Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
    mmatrix_->multiply(true, *z_, fm);
    Core::LinAlg::Vector<double> fmexp(*problem_dofs());
    Core::LinAlg::export_to(fm, fmexp);
    strcontactrhs_->update(-1., fmexp, 1.0);
  }
}

/*----------------------------------------------------------------------*
 | set force evaluation flag before evaluation step          farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::pre_evaluate(CONTACT::ParamsInterface& cparams)
{
  const enum Mortar::ActionType& act = cparams.get_action_type();

  switch (act)
  {
      // -------------------------------------------------------------------
      // reset force evaluation flag for predictor step
      // -------------------------------------------------------------------
    case Mortar::eval_force_stiff:
    {
      if (cparams.is_predictor()) evalForceCalled_ = false;
      break;
    }
    // -------------------------------------------------------------------
    // default
    // -------------------------------------------------------------------
    default:
    {
      // do nothing
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 | set force evaluation flag after evaluation                farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::post_evaluate(CONTACT::ParamsInterface& cparams)
{
  const enum Mortar::ActionType& act = cparams.get_action_type();

  switch (act)
  {
    // -------------------------------------------------------------------
    // set flag to false after force stiff evaluation
    // -------------------------------------------------------------------
    case Mortar::eval_force_stiff:
    {
      evalForceCalled_ = false;
      break;
    }
    // -------------------------------------------------------------------
    // set flag for force evaluation to true
    // -------------------------------------------------------------------
    case Mortar::eval_force:
    {
      evalForceCalled_ = true;
      break;
    }
    // -------------------------------------------------------------------
    // default
    // -------------------------------------------------------------------
    default:
    {
      // do nothing
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::evaluate_force_stiff(CONTACT::ParamsInterface& cparams)
{
  // call the evaluate force routine if not done before
  if (!evalForceCalled_) evaluate_force(cparams);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::LagrangeStrategy::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt, const ParamsInterface* cparams) const
{
  // if there are no active LM contact contributions
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step())
  {
    if (nonSmoothContact_ && bt == CONTACT::MatBlockType::displ_displ)
      return nonsmooth_Penalty_stiff_;
    else
      return nullptr;
  }
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
    {
      mat_ptr = std::make_shared<Core::LinAlg::SparseMatrix>(
          slave_master_dof_row_map(true), 100, false, true);

      // build matrix kdd
      mat_ptr->add(*lindmatrix_, false, 1.0, 1.0);
      mat_ptr->add(*linmmatrix_, false, 1.0, 1.0);
      if (nonSmoothContact_ && nonsmooth_Penalty_stiff_)
        mat_ptr->add(*nonsmooth_Penalty_stiff_, false, 1.0, 1.0);
      mat_ptr->complete();

      // transform parallel row/column distribution of matrix kdd
      // (only necessary in the parallel redistribution case)
      if (parallel_redistribution_status())
        mat_ptr = Core::LinAlg::matrix_row_col_transform(
            *mat_ptr, *slave_master_dof_row_map_ptr(false), *slave_master_dof_row_map_ptr(false));

      std::shared_ptr<Core::LinAlg::SparseMatrix> full_mat_ptr =
          std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);
      full_mat_ptr->add(*mat_ptr, false, 1., 1.);
      full_mat_ptr->complete();
      if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
        full_mat_ptr = Core::LinAlg::matrix_multiply(
            *invsystrafo_, true, *full_mat_ptr, false, false, false, true);

      mat_ptr = full_mat_ptr;

      break;
    }
    case CONTACT::MatBlockType::displ_lm:
    {
      // build constraint matrix kdz
      Core::LinAlg::SparseMatrix kdz_ptr(*gdisprowmap_, 100, false, true);

      kdz_ptr.add(*dmatrix_, true, 1.0, 1.0);
      kdz_ptr.add(*mmatrix_, true, -1.0, 1.0);
      kdz_ptr.complete(*gsdofrowmap_, *gdisprowmap_);

      // transform constraint matrix kzd to lmdofmap (matrix_col_transform)
      mat_ptr = Core::LinAlg::matrix_col_transform_gids(kdz_ptr, *lm_dof_row_map_ptr(true));

      // transform parallel row/column distribution of matrix kdz
      // (only necessary in the parallel redistribution case)
      if (parallel_redistribution_status() or is_self_contact())
        mat_ptr = Core::LinAlg::matrix_row_col_transform(
            *mat_ptr, *problem_dofs(), *lin_system_lm_dof_row_map_ptr());

      break;
    }
    case CONTACT::MatBlockType::lm_displ:
    {
      // build constraint matrix kzd
      Core::LinAlg::SparseMatrix kzd_ptr(slave_dof_row_map(true), 100, false, true);

      // build constraint matrix kzd
      if (gactiven_->num_global_elements())
      {
        kzd_ptr.add(*smatrix_, false, 1.0, 1.0);

        // frictionless contact
        if (!is_friction()) kzd_ptr.add(*tderivmatrix_, false, 1.0, 1.0);

        // frictional contact
        else
        {
          kzd_ptr.add(*linslipDIS_, false, 1.0, 1.0);
          kzd_ptr.add(*linstickDIS_, false, 1.0, 1.0);
        }
      }
      kzd_ptr.complete(*gdisprowmap_, *gsdofrowmap_);

      // transform constraint matrix kzd to lmdofmap (MatrixRowTransform)
      mat_ptr = Core::LinAlg::matrix_row_transform_gids(kzd_ptr, *lm_dof_row_map_ptr(true));

      // transform parallel row/column distribution of matrix kzd
      // (only necessary in the parallel redistribution case)
      if (parallel_redistribution_status() or is_self_contact())
        mat_ptr = Core::LinAlg::matrix_row_col_transform(
            *mat_ptr, *lin_system_lm_dof_row_map_ptr(), *problem_dofs());
      break;
    }
    case CONTACT::MatBlockType::lm_lm:
    {
      // build constraint matrix kzz
      std::shared_ptr<Core::LinAlg::SparseMatrix> kzz_ptr = nullptr;
      if (is_self_contact())
      {
        kzz_ptr = std::make_shared<Core::LinAlg::SparseMatrix>(
            global_self_contact_ref_map(), 100, false, true);

        std::shared_ptr<Core::LinAlg::Map> unused_lmdofs =
            Core::LinAlg::split_map(global_self_contact_ref_map(), *gsdofrowmap_);
        Core::LinAlg::Vector<double> ones = Core::LinAlg::Vector<double>(*unused_lmdofs, false);
        ones.put_scalar(1.0);
        if (Core::LinAlg::insert_my_row_diagonal_into_unfilled_matrix(*kzz_ptr, ones))
          FOUR_C_THROW("Unexpected error!");
      }
      else
      {
        kzz_ptr =
            std::make_shared<Core::LinAlg::SparseMatrix>(slave_dof_row_map(true), 100, false, true);
      }

      // build unity matrix for inactive dofs
      std::shared_ptr<Core::LinAlg::Map> gidofs =
          Core::LinAlg::split_map(*gsdofrowmap_, *gactivedofs_);
      Core::LinAlg::Vector<double> ones(*gidofs);
      ones.put_scalar(1.0);
      Core::LinAlg::SparseMatrix onesdiag(ones);
      onesdiag.complete();

      // build constraint matrix kzz
      if (gidofs->num_global_elements()) kzz_ptr->add(onesdiag, false, 1.0, 1.0);

      if (!is_friction())
      {
        if (gactivet_->num_global_elements()) kzz_ptr->add(*tmatrix_, false, 1.0, 1.0);
      }
      else
      {
        kzz_ptr->add(*linslipLM_, false, 1., 1.);
        kzz_ptr->add(*linstickLM_, false, 1., 1.);
      }

      // transform constraint matrix kzz to lmdofmap
      if (is_self_contact())
      {
        kzz_ptr->complete(*gsmdofrowmap_, *gsmdofrowmap_);
        mat_ptr = Core::LinAlg::matrix_row_col_transform_gids(
            *kzz_ptr, *lin_system_lm_dof_row_map_ptr(), *lin_system_lm_dof_row_map_ptr());
      }
      else
      {
        kzz_ptr->complete(*gsdofrowmap_, *gsdofrowmap_);
        mat_ptr = Core::LinAlg::matrix_row_col_transform_gids(
            *kzz_ptr, *lm_dof_row_map_ptr(true), *lm_dof_row_map_ptr(true));
      }

      // transform parallel row/column distribution of matrix kzz
      // (only necessary in the parallel redistribution case)
      if (parallel_redistribution_status())
        mat_ptr = Core::LinAlg::matrix_row_col_transform(
            *mat_ptr, *lin_system_lm_dof_row_map_ptr(), *lin_system_lm_dof_row_map_ptr());

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Solid::MatBlockType!");
      break;
    }
  }

  return mat_ptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::run_post_compute_x(const CONTACT::ParamsInterface& cparams,
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
  if (system_type() != CONTACT::SystemType::condensed)
  {
    if (lm_dof_row_map(true).num_global_elements() > 0)
    {
      Core::LinAlg::Vector<double> zdir_ptr(lm_dof_row_map(true), true);
      Core::LinAlg::export_to(dir, zdir_ptr);
      // get the current step length
      const double stepLength = cparams.get_step_length();
      // ---------------------------------------------------------------------
      // store the SCALED Lagrange multiplier increment in the contact
      // strategy
      // ---------------------------------------------------------------------
      zdir_ptr.replace_map(zincr_->get_map());
      zincr_->scale(stepLength, zdir_ptr);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> CONTACT::LagrangeStrategy::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  // if there are no active LM contact contributions
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step())
  {
    if (nonSmoothContact_ && bt == CONTACT::VecBlockType::displ)
    {
      return nonsmooth_Penalty_force_;
    }
    else
    {
      return nullptr;
    }
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> vec_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
    {
      if (nonSmoothContact_ && nonsmooth_Penalty_force_ != nullptr)
      {
        vec_ptr = std::make_shared<Core::LinAlg::Vector<double>>(*nonsmooth_Penalty_force_);
      }

      if (vec_ptr == nullptr)
        vec_ptr = strcontactrhs_;
      else if (strcontactrhs_ != nullptr)
        vec_ptr->update(1., *strcontactrhs_, 1.);

      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
      if (is_dual_quad_slave_trafo() && Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                                            params(), "LM_QUAD") == Inpar::Mortar::lagmult_lin)
      {
        invsystrafo_->multiply(true, *vec_ptr, *tmp);
        vec_ptr = tmp;
      }

      break;
    }
    case CONTACT::VecBlockType::constraint:
    {
      vec_ptr = constrrhs_;
      if (is_self_contact() && !is_condensed_system())
      {
        static std::shared_ptr<Core::LinAlg::Vector<double>> tmp_ptr =
            std::make_shared<Core::LinAlg::Vector<double>>(lin_system_lm_dof_row_map(), false);
        tmp_ptr->put_scalar(0.0);
        Core::LinAlg::export_to(*vec_ptr, *tmp_ptr);
        vec_ptr = tmp_ptr;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Solid::VecBlockType!");
      break;
    }
  }

  return vec_ptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::reset_lagrange_multipliers(
    const CONTACT::ParamsInterface& cparams, const Core::LinAlg::Vector<double>& xnew)
{
  if (system_type() != CONTACT::SystemType::condensed)
  {
    if (lm_dof_row_map(true).num_global_elements() == 0) return;

    Core::LinAlg::Vector<double> znew_ptr(lm_dof_row_map(true), true);
    Core::LinAlg::export_to(xnew, znew_ptr);
    // ---------------------------------------------------------------------
    // Update the current lagrange multiplier
    // ---------------------------------------------------------------------
    znew_ptr.replace_map(z_->get_map());

    z_->scale(1.0, znew_ptr);

    // ---------------------------------------------------------------------
    // store the new Lagrange multiplier in the nodes
    // ---------------------------------------------------------------------
    store_nodal_quantities(Mortar::StrategyBase::lmupdate);
  }
}


/*----------------------------------------------------------------------*
 | Recovery method                                            popp 04/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi)
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::LagrangeStrategy::recover");

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // shape function type and type of LM interpolation for quadratic elements
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (system_type() == CONTACT::SystemType::condensed)
  {
    // double-check if this is a dual LM system
    if ((shapefcn != Inpar::Mortar::shape_dual &&
            shapefcn != Inpar::Mortar::shape_petrovgalerkin) &&
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD") !=
            Inpar::Mortar::lagmult_const)
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
    invdmod.add(*invda, false, 1.0, 1.0);
    invdmod.complete();

    /**********************************************************************/
    /* Undo basis transformation to solution                              */
    /* (currently only needed for quadratic FE with linear dual LM)       */
    /**********************************************************************/
    if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // undo basis transformation to solution
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (parallel_redistribution_status())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();
      Core::LinAlg::Vector<double> disinew(*disi);
      systrafo.multiply(false, disinew, *disi);
    }

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
  // CASE B: SADDLE POINT SYSTEM
  //**********************************************************************
  //**********************************************************************
  else
  {
    // do nothing (z_ was part of solution already)

    /**********************************************************************/
    /* Undo basis transformation to solution                              */
    /* (currently only needed for quadratic FE with linear dual LM)       */
    /**********************************************************************/
    if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // undo basis transformation to solution
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (parallel_redistribution_status())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();
      Core::LinAlg::Vector<double> disinew(*disi);
      systrafo.multiply(false, disinew, *disi);
    }
  }

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate);

  return;
}

/*----------------------------------------------------------------------*
 |  Update active set and check for convergence               popp 02/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::update_active_set()
{
  // get input parameter ftype
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(params(), "FRICTION");

  // assume that active set has converged and check for opposite
  activesetconv_ = true;

  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    // loop over all slave nodes on the current interface
    for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
    {
      int gid = interface_[i]->slave_row_nodes()->gid(j);
      Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      Node* cnode = dynamic_cast<Node*>(node);

      // compute weighted gap
      double wgap = wgap_->local_values_as_span()[wgap_->get_map().lid(gid)];

      // compute normal part of Lagrange multiplier
      double nz = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        nz += cnode->mo_data().n()[k] * cnode->mo_data().lm()[k];
      }

      // friction
      double tz = 0.0;
      double tjump = 0.0;

      if (friction_)
      {
        FriNode* frinode = dynamic_cast<FriNode*>(cnode);

        // compute tangential part of Lagrange multiplier
        tz = frinode->data().txi()[0] * frinode->mo_data().lm()[0] +
             frinode->data().txi()[1] * frinode->mo_data().lm()[1];

        // compute tangential part of jump FIXME -- also the teta component should be considered
        if (params().get<bool>("GP_SLIP_INCR"))
          tjump = frinode->fri_data().jump_var()[0];
        else
          tjump = frinode->data().txi()[0] * frinode->fri_data().jump()[0] +
                  frinode->data().txi()[1] * frinode->fri_data().jump()[1];
      }

      // check nodes of inactive set *************************************
      // (by definition they fulfill the condition z_j = 0)
      // (thus we only have to check ncr.disp. jump and weighted gap)
      if (cnode->active() == false)
      {
        // check for penetration
        if (wgap < 0)
        {
          cnode->active() = true;
          activesetconv_ = false;
        }
      }

      // check nodes of active set ***************************************
      // (by definition they fulfill the non-penetration condition)
      // (thus we only have to check for positive Lagrange multipliers)
      else
      {
        // check for tensile contact forces
        if (nz <= 0)  // no averaging of Lagrange multipliers
        // if (0.5*nz+0.5*nzold <= 0) // averaging of Lagrange multipliers
        {
          cnode->active() = false;
          activesetconv_ = false;

          // friction
          if (friction_) dynamic_cast<FriNode*>(cnode)->fri_data().slip() = false;
        }

        // only do something for friction
        else
        {
          // friction tresca
          if (ftype == CONTACT::FrictionType::tresca)
          {
            FriNode* frinode = dynamic_cast<FriNode*>(cnode);
            const Core::LinAlg::Vector<double>& ct_ref = interface_[i]->ct_ref();
            double ct = ct_ref.local_values_as_span()[ct_ref.get_map().lid(frinode->id())];

            // CAREFUL: friction bound is now interface-local (popp 08/2012)
            double frbound = interface_[i]->interface_params().get<double>("FRBOUND");

            if (frinode->fri_data().slip() == false)
            {
              // check (tz+ct*tjump)-frbound <= 0
              if (abs(tz + ct * tjump) - frbound <= 0)
              {
              }
              // do nothing (stick was correct)
              else
              {
                frinode->fri_data().slip() = true;
                activesetconv_ = false;
              }
            }
            else
            {
              // check (tz+ct*tjump)-frbound > 0
              if (abs(tz + ct * tjump) - frbound > 0)
              {
              }
              // do nothing (slip was correct)
              else
              {
                frinode->fri_data().slip() = false;
                activesetconv_ = false;
              }
            }
          }  // if (ftype == CONTACT::FrictionType::tresca)

          // friction coulomb
          if (ftype == CONTACT::FrictionType::coulomb)
          {
            FriNode* frinode = dynamic_cast<FriNode*>(cnode);
            const Core::LinAlg::Vector<double>& ct_ref = interface_[i]->ct_ref();
            double ct = ct_ref.local_values_as_span()[ct_ref.get_map().lid(frinode->id())];

            // CAREFUL: friction coefficient is now interface-local (popp 08/2012)
            double frcoeff = interface_[i]->interface_params().get<double>("FRCOEFF");

            if (frinode->fri_data().slip() == false)
            {
              // check (tz+ct*tjump)-frbound <= 0
              if (abs(tz + ct * tjump) - frcoeff * nz <= 0)
              {
              }
              // do nothing (stick was correct)
              else
              {
                frinode->fri_data().slip() = true;
                activesetconv_ = false;
              }
            }
            else
            {
              // check (tz+ct*tjump)-frbound > 0
              if (abs(tz + ct * tjump) - frcoeff * nz > 0)
              {
              }
              // do nothing (slip was correct)
              else
              {
                frinode->fri_data().slip() = false;
                activesetconv_ = false;
              }
            }
          }  // if (ftype == CONTACT::FrictionType::coulomb)
        }  // if (nz <= 0)
      }  // if (cnode->Active()==false)
    }  // loop over all slave nodes
  }  // loop over all interfaces

  // broadcast convergence status among processors
  int convcheck = 0;
  int localcheck = activesetconv_;
  convcheck = Core::Communication::sum_all(localcheck, get_comm());

  // active set is only converged, if converged on all procs
  // if not, increase no. of active set steps too
  if (convcheck != Core::Communication::num_mpi_ranks(get_comm()))
  {
    activesetconv_ = false;
    activesetsteps_ += 1;
  }

  // update zig-zagging history (shift by one)
  if (zigzagtwo_ != nullptr) zigzagthree_ = std::make_shared<Core::LinAlg::Map>(*zigzagtwo_);
  if (zigzagone_ != nullptr) zigzagtwo_ = std::make_shared<Core::LinAlg::Map>(*zigzagone_);
  if (gactivenodes_ != nullptr) zigzagone_ = std::make_shared<Core::LinAlg::Map>(*gactivenodes_);

  // (re)setup active global Core::LinAlg::Maps
  gactivenodes_ = nullptr;
  gactivedofs_ = nullptr;
  ginactivenodes_ = nullptr;
  ginactivedofs_ = nullptr;
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

    ginactivenodes_ =
        Core::LinAlg::merge_map(ginactivenodes_, interface_[i]->inactive_nodes(), false);
    ginactivedofs_ = Core::LinAlg::merge_map(ginactivedofs_, interface_[i]->inactive_dofs(), false);

    gactiven_ = Core::LinAlg::merge_map(gactiven_, interface_[i]->active_n_dofs(), false);
    gactivet_ = Core::LinAlg::merge_map(gactivet_, interface_[i]->active_t_dofs(), false);

    if (friction_)
    {
      gslipnodes_ = Core::LinAlg::merge_map(gslipnodes_, interface_[i]->slip_nodes(), false);
      gslipdofs_ = Core::LinAlg::merge_map(gslipdofs_, interface_[i]->slip_dofs(), false);
      gslipt_ = Core::LinAlg::merge_map(gslipt_, interface_[i]->slip_t_dofs(), false);
    }
  }

  // CHECK FOR ZIG-ZAGGING / JAMMING OF THE ACTIVE SET
  // *********************************************************************
  // A problem of the active set strategy which sometimes arises is known
  // from optimization literature as jamming or zig-zagging. This means
  // that within a load/time-step the algorithm can have more than one
  // solution due to the fact that the active set is not unique. Hence the
  // algorithm jumps between the solutions of the active set. The non-
  // uniquenesss results either from highly curved contact surfaces or
  // from the FE discretization, Thus the uniqueness of the closest-point-
  // projection cannot be guaranteed.
  // *********************************************************************
  // To overcome this problem we monitor the development of the active
  // set scheme in our contact algorithms. We can identify zig-zagging by
  // comparing the current active set with the active set of the second-
  // and third-last iteration. If an identity occurs, we consider the
  // active set strategy as converged instantly, accepting the current
  // version of the active set and proceeding with the next time/load step.
  // This very simple approach helps stabilizing the contact algorithm!
  // *********************************************************************
  bool zigzagging = false;
  // FIXGIT: For tresca friction zig-zagging is not eliminated
  if (ftype != CONTACT::FrictionType::tresca && ftype != CONTACT::FrictionType::coulomb)
  {
    // frictionless contact
    if (active_set_steps() > 2)
    {
      if (zigzagtwo_ != nullptr)
      {
        if (zigzagtwo_->same_as(*gactivenodes_))
        {
          // set active set converged
          activesetconv_ = true;
          zigzagging = true;

          // output to screen
          if (Core::Communication::my_mpi_rank(get_comm()) == 0)
            std::cout << "DETECTED 1-2 ZIG-ZAGGING OF ACTIVE SET................." << std::endl;
        }
      }

      if (zigzagthree_ != nullptr)
      {
        if (zigzagthree_->same_as(*gactivenodes_))
        {
          // set active set converged
          activesetconv_ = true;
          zigzagging = true;

          // output to screen
          if (Core::Communication::my_mpi_rank(get_comm()) == 0)
            std::cout << "DETECTED 1-2-3 ZIG-ZAGGING OF ACTIVE SET................" << std::endl;
        }
      }
    }
  }  // if (ftype != CONTACT::FrictionType::tresca && ftype != CONTACT::FrictionType::coulomb)


  // reset zig-zagging history
  if (activesetconv_ == true)
  {
    zigzagone_ = nullptr;
    zigzagtwo_ = nullptr;
    zigzagthree_ = nullptr;
  }

  // output of active set status to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 && activesetconv_ == false)
    std::cout << "ACTIVE SET ITERATION " << active_set_steps() - 1
              << " NOT CONVERGED - REPEAT TIME STEP................." << std::endl;
  else if (Core::Communication::my_mpi_rank(get_comm()) == 0 && activesetconv_ == true)
    std::cout << "ACTIVE SET CONVERGED IN " << active_set_steps() - zigzagging
              << " STEP(S)................." << std::endl;

  // update flag for global contact status
  if (gactivenodes_->num_global_elements())
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  return;
}

/*----------------------------------------------------------------------*
 |  Update active set and check for convergence (public)      popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::update_active_set_semi_smooth(const bool firstStepPredictor)
{
  // FIXME: Here we do not consider zig-zagging yet!
  //  print_active_set();

  // get out gof here if not in the semi-smooth Newton case
  // (but before doing this, check if there are invalid active nodes)
  const bool semismooth = params().get<bool>("SEMI_SMOOTH_NEWTON");
  if (!semismooth)
  {
    // loop over all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      // loop over all slave nodes on the current interface
      for (int j = 0; j < interface_[i]->slave_row_nodes()->num_my_elements(); ++j)
      {
        int gid = interface_[i]->slave_row_nodes()->gid(j);
        Core::Nodes::Node* node = interface_[i]->discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
        Node* cnode = dynamic_cast<Node*>(node);

        // The nested active set strategy cannot deal with the case of
        // active nodes that have no integration segments/cells attached,
        // as this leads to zero rows in D and M and thus to singular systems.
        // However, this case might possibly happen when slave nodes slide
        // over the edge of a master body within one fixed active set step.
        // (Remark: Semi-smooth Newton has no problems in this case, as it
        // updates the active set after EACH Newton step, see below, and thus
        // would always set the corresponding nodes to INACTIVE.)
        if (cnode->active() && !cnode->has_segment() && !cnode->is_on_boundor_ce())
          FOUR_C_THROW("Active node {} without any segment/cell attached", cnode->id());
      }
    }
    return;
  }

  // get input parameter ftype
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(params(), "FRICTION");

  // assume that active set has converged and check for opposite
  activesetconv_ = true;

  // loop over all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    bool my_check = interface_[i]->update_active_set_semi_smooth();
    activesetconv_ = activesetconv_ and my_check;
  }  // loop over all interfaces

  // Overwrite active set with input file information in predictor step of first time step
  if (firstStepPredictor)
  {
    // loop over all interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->update_active_set_initial_status();
    }  // loop over all interfaces
  }

  // broadcast convergence status among processors
  int convcheck = 0;
  int localcheck = activesetconv_;
  convcheck = Core::Communication::sum_all(localcheck, get_comm());

  // active set is only converged, if converged on all procs
  // if not, increase no. of active set steps too
  if (convcheck != Core::Communication::num_mpi_ranks(get_comm()))
  {
    activesetconv_ = false;
    activesetsteps_ += 1;
  }

  // only if it's a full Newton step...
  // store the previous active set
  if (gactivenodes_ != nullptr)
  {
    gOldActiveSlaveNodes_ = std::make_shared<Core::LinAlg::Map>(*gactivenodes_);
    if (friction_) gOldslipnodes_ = std::make_shared<Core::LinAlg::Map>(*gslipnodes_);
  }
  else
  {
    gOldActiveSlaveNodes_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    if (friction_) gOldslipnodes_ = std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
  }

  // also update special flag for semi-smooth Newton convergence
  activesetssconv_ = activesetconv_;

  // update zig-zagging history (shift by one)
  if (zigzagtwo_ != nullptr) zigzagthree_ = std::make_shared<Core::LinAlg::Map>(*zigzagtwo_);
  if (zigzagone_ != nullptr) zigzagtwo_ = std::make_shared<Core::LinAlg::Map>(*zigzagone_);
  if (gactivenodes_ != nullptr) zigzagone_ = std::make_shared<Core::LinAlg::Map>(*gactivenodes_);

  // (re)setup active global Core::LinAlg::Maps
  gactivenodes_ = nullptr;
  gactivedofs_ = nullptr;
  ginactivenodes_ = nullptr;
  ginactivedofs_ = nullptr;
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
    ginactivenodes_ =
        Core::LinAlg::merge_map(ginactivenodes_, interface_[i]->inactive_nodes(), false);
    ginactivedofs_ = Core::LinAlg::merge_map(ginactivedofs_, interface_[i]->inactive_dofs(), false);
    gactiven_ = Core::LinAlg::merge_map(gactiven_, interface_[i]->active_n_dofs(), false);
    gactivet_ = Core::LinAlg::merge_map(gactivet_, interface_[i]->active_t_dofs(), false);

    if (friction_)
    {
      gslipnodes_ = Core::LinAlg::merge_map(gslipnodes_, interface_[i]->slip_nodes(), false);
      gslipdofs_ = Core::LinAlg::merge_map(gslipdofs_, interface_[i]->slip_dofs(), false);
      gslipt_ = Core::LinAlg::merge_map(gslipt_, interface_[i]->slip_t_dofs(), false);
    }
  }

  // CHECK FOR ZIG-ZAGGING / JAMMING OF THE ACTIVE SET
  // *********************************************************************
  // A problem of the active set strategy which sometimes arises is known
  // from optimization literature as jamming or zig-zagging. This means
  // that within a load/time-step the semi-smooth Newton algorithm can get
  // stuck between more than one intermediate solution due to the fact that
  // the active set decision is a discrete decision. Hence the semi-smooth
  // Newton algorithm fails to converge. The non-uniquenesss results either
  // from highly curved contact surfaces or from the FE discretization.
  // *********************************************************************
  // To overcome this problem we monitor the development of the active
  // set scheme in our contact algorithms. We can identify zig-zagging by
  // comparing the current active set with the active set of the second-
  // and third-last iteration. If an identity occurs, we interfere and
  // let the semi-smooth Newton algorithm restart from another active set
  // (e.g. intermediate set between the two problematic candidates), thus
  // leading to some kind of damped / modified semi-smooth Newton method.
  // This very simple approach helps stabilizing the contact algorithm!
  // *********************************************************************
  int zigzagging = 0;
  // FIXGIT: For friction zig-zagging is not eliminated
  if (ftype != CONTACT::FrictionType::tresca && ftype != CONTACT::FrictionType::coulomb)
  {
    // frictionless contact
    if (active_set_steps() > 2)
    {
      if (zigzagtwo_ != nullptr)
      {
        if (zigzagtwo_->same_as(*gactivenodes_))
        {
          // detect zig-zagging
          zigzagging = 1;
        }
      }

      if (zigzagthree_ != nullptr)
      {
        if (zigzagthree_->same_as(*gactivenodes_))
        {
          // detect zig-zagging
          zigzagging = 2;
        }
      }
    }
  }  // if (ftype != CONTACT::FrictionType::tresca && ftype != CONTACT::FrictionType::coulomb)

  // output to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (zigzagging == 1)
    {
      std::cout << "DETECTED 1-2 ZIG-ZAGGING OF ACTIVE SET................." << std::endl;
    }
    else if (zigzagging == 2)
    {
      std::cout << "DETECTED 1-2-3 ZIG-ZAGGING OF ACTIVE SET................" << std::endl;
    }
    else
    {
      // do nothing, no zig-zagging
    }
  }

  // reset zig-zagging history
  if (activesetconv_ == true)
  {
    zigzagone_ = nullptr;
    zigzagtwo_ = nullptr;
    zigzagthree_ = nullptr;
  }

  // output of active set status to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 && activesetconv_ == false)
    std::cout << "ACTIVE CONTACT SET HAS CHANGED... CHANGE No. " << active_set_steps() - 1
              << std::endl;

  // update flag for global contact status
  if (gactivenodes_->num_global_elements())
  {
    isincontact_ = true;
    wasincontact_ = true;
  }
  else
    isincontact_ = false;

  return;
}

/*----------------------------------------------------------------------*
 |  update routine for ltl forces                            farah 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategy::update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (fLTL_ != nullptr)
  {
    // store fLTL values for time integration
    fLTLOld_ = std::make_shared<Core::LinAlg::Vector<double>>(fLTL_->get_map());
    fLTLOld_->update(1.0, *fLTL_, 0.0);
  }

  // abstract routine
  CONTACT::AbstractStrategy::update(dis);

  if (fconservation_ == nullptr) return;

  // *****************************************************************
  // This is output functionality for conservation properties of LTL
  // penalty contact
  // *****************************************************************

  //  std::shared_ptr<Core::LinAlg::Vector<double>> fconservationS =
  //      Teuchos::rcp(new Core::LinAlg::Vector<double>(slave_dof_row_map(true)),true);
  //  std::shared_ptr<Core::LinAlg::Vector<double>> fconservationM =
  //      Teuchos::rcp(new Core::LinAlg::Vector<double>(master_dof_row_map(true)),true);
  //
  //  Core::LinAlg::export_to(*fconservation_,*fconservationS);
  //  Core::LinAlg::export_to(*fconservation_,*fconservationM);
  ////  fconservationM->Scale(-1.0);
  //  // check conservation properties:
  //  interface_[0]->EvalResultantMoment(*fconservationS, *fconservationM);
  //
  //
  //  double lssum = 0.0;   // local slave sum
  //  double gssum = 0.0;   // global slave sum
  //  double lmsum = 0.0;   // local master sum
  //  double gmsum = 0.0;   // global master sum
  //  double gcsum = 0.0;   // global complete sum
  //  // slave
  ////  for (int i=0;i<fconservationS->MyLength();++i)
  ////  {
  ////    lssum+=(*fconservationS)[i];
  ////  }
  ////  Core::Communication::sum_all(&lssum,&gssum,1, Comm());
  ////  // master
  ////  for (int i=0;i<fconservationM->MyLength();++i)
  ////  {
  ////    lmsum+=(*fconservationM)[i];
  ////  }
  ////  Core::Communication::sum_all(&lmsum,&gmsum,1, Comm());
  //
  //
  //
  //  // complete balance check
  //  gcsum = gssum+gmsum;
  //  if (abs(gcsum)>1.0e-11)
  //    FOUR_C_THROW("Conservation of linear momentum is not fulfilled!");
  //  if (Core::Communication::my_mpi_rank(Comm())==0)
  //  {
  //    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<" << std::endl;
  //    std::cout << ">>      Linear Momentum Conservation      <<" << std::endl;
  //    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<" << std::endl;
  //    std::cout << ">>      Standard terms (lm)               <<" << std::endl;
  //    std::cout << "SLAVE:   " << std::setw(14) << gssum<< std::endl;
  //    std::cout << "MASTER:  " << std::setw(14) << gmsum << std::endl;
  //    std::cout << "Balance: " << std::setw(14) << gcsum << std::endl;
  //    std::cout << "--------------------------------------------" << std::endl;
  //  }
  //
  //
  //  FILE* MyFile = nullptr;
  //  std::ostringstream filename;
  //  const std::string filebase = "xxx";
  //  filename << filebase <<".lmom";
  //  MyFile = fopen(filename.str().c_str(), "at+");
  //
  //  // store data
  //  if (MyFile)
  //  {
  //    fprintf(MyFile, "%d\t", step);
  //    fprintf(MyFile, "%g\t", gssum);
  //    fprintf(MyFile, "%g\t", gmsum);
  //    fprintf(MyFile, "%g\n", gcsum);
  //    fclose(MyFile);
  //  }
  //  else
  //    FOUR_C_THROW("File could not be opened.");
  //
  //  ++step;

  return;
}

void CONTACT::LagrangeStrategy::condense_friction(
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> feff =
      Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(rhs);

  feff->update(-1. + alphaf_, *strcontactrhs_, 1.);
  feff->scale(-1.);

  std::shared_ptr<Core::LinAlg::SparseOperator> kteff_op =
      std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(kteff);

  // In case of nonsmooth contact the scenario of contacting edges (non parallel)
  // requires a penalty regularization. Here, the penalty contriutions for this
  // special case are applied:
  if (nonSmoothContact_)
  {
    // add_line_to_lin_contributions(kteff,feff);
    add_line_to_lin_contributions_friction(*kteff_op, feff);

    // FD check of weighted gap g derivatives + jump for LTL case
#ifdef CONTACTFDJUMPLTL
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckJumpDerivLTL();
    }
#endif  // #ifdef CONTACTFDJUMPLTL
  }

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  /**********************************************************************/
  /* export weighted gap vector to gactiveN-map                         */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*gactivedofs_, true);
    if (gact->global_length()) Core::LinAlg::export_to(*wgap_, *gact);
  }
  else
  {
    gact = Core::LinAlg::create_vector(*gactivenodes_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
  }

  // fill_complete global Matrix linstickLM_, linstickDIS_
  std::shared_ptr<Core::LinAlg::Map> gstickt = Core::LinAlg::split_map(*gactivet_, *gslipt_);
  std::shared_ptr<Core::LinAlg::Map> gstickdofs =
      Core::LinAlg::split_map(*gactivedofs_, *gslipdofs_);

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  // double-check if this is a dual LM system
  if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Condensation only for dual LM");

  /********************************************************************/
  /* (1) Multiply Mortar matrices: m^ = inv(d) * m                    */
  /********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> invd = invd_;

  /********************************************************************/
  /* (2) Add contact stiffness terms to kteff                         */
  /********************************************************************/

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

  // abbreviations for slave set
  const int sset = gsdofrowmap_->num_global_elements();

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
  kmnmod->add(*kmn, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
  kmnmod->add(*kmnadd, false, 1.0, 1.0);
  kmnmod->complete(kmn->domain_map(), kmn->row_map());

  // kmm: add T(mhataam)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  kmmmod->add(*kmm, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
  kmmmod->add(*kmmadd, false, 1.0, 1.0);
  kmmmod->complete(kmm->domain_map(), kmm->row_map());

  // kmi: add T(mhataam)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
  if (iset)
  {
    kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmimod->add(*kmi, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
    kmimod->add(*kmiadd, false, 1.0, 1.0);
    kmimod->complete(kmi->domain_map(), kmi->row_map());
  }

  // kma: add T(mhataam)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
  if (aset)
  {
    kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmamod->add(*kma, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
    kmamod->add(*kmaadd, false, 1.0, 1.0);
    kmamod->complete(kma->domain_map(), kma->row_map());
  }

  //--------------------------------------------------------- THIRD LINE
  // kin: subtract T(dhat)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  kinmod->add(*kin, false, 1.0, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
    kinmod->add(*kinadd, false, -1.0, 1.0);
  }
  kinmod->complete(kin->domain_map(), kin->row_map());

  // kim: subtract T(dhat)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  kimmod->add(*kim, false, 1.0, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
    kimmod->add(*kimadd, false, -1.0, 1.0);
  }
  kimmod->complete(kim->domain_map(), kim->row_map());

  // kii: subtract T(dhat)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
  if (iset)
  {
    kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kiimod->add(*kii, false, 1.0, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
      kiimod->add(*kiiadd, false, -1.0, 1.0);
    }
    kiimod->complete(kii->domain_map(), kii->row_map());
  }

  // kia: subtract T(dhat)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
  if (iset && aset)
  {
    kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kiamod->add(*kia, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
    kiamod->add(*kiaadd, false, -1.0, 1.0);
    kiamod->complete(kia->domain_map(), kia->row_map());
  }

  //-------------------------------------------------------- FOURTH LINE

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

  /********************************************************************/
  /* (8) Build the final f blocks                                     */
  /********************************************************************/

  //--------------------------------------------------------- FIRST LINE
  // fn: nothing to do

  //---------------------------------------------------------- SECOND LINE

  // fm: add T(mhat)*fa
  Core::LinAlg::Vector<double> fmmod(*gmdofrowmap_);
  if (aset) mhataam->multiply(true, *fa, fmmod);
  fmmod.update(1.0, *fm, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // fi: subtract alphaf * old contact forces (t_n)

  // fi: add T(dhat)*fa
  Core::LinAlg::Vector<double> fimod(*gidofs);
  if (aset && iset) dhat->multiply(true, *fa, fimod);
  fimod.update(1.0, *fi, -1.0);

  //-------------------------------------------------------- FOURTH LINE

  //--------------------------------------------------------- FIFTH LINE
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
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
    else
      fstmod = std::make_shared<Core::LinAlg::Vector<double>>(*gstickt);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
        Core::LinAlg::matrix_multiply(*linstickLM_, false, *invdst, true, false, false, true);
    temp1->multiply(false, *fa, *fstmod);

    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gstickdofs);
    else
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
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
    else
      fslmod = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
        Core::LinAlg::matrix_multiply(*linslipLM_, false, *invdsl, true, false, false, true);
    temp->multiply(false, *fa, *fslmod);

    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipdofs_);
    else
      tempvec1 = std::make_shared<Core::LinAlg::Vector<double>>(*gslipt_);

    linslipLM_->multiply(false, *zsl, *tempvec1);

    fslmod->update(-1.0, *tempvec1, 1.0);
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
    }
  }

  /********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                */
  /********************************************************************/

  Core::LinAlg::SparseMatrix kteffnew(
      *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
  std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
      Core::LinAlg::create_vector(*problem_dofs());

  //--------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  kteffnew.add(*knn, false, 1.0, 1.0);
  kteffnew.add(*knm, false, 1.0, 1.0);
  if (sset) kteffnew.add(*kns, false, 1.0, 1.0);

  //-------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  kteffnew.add(*kmnmod, false, 1.0, 1.0);
  kteffnew.add(*kmmmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kmimod, false, 1.0, 1.0);
  if (aset) kteffnew.add(*kmamod, false, 1.0, 1.0);

  //--------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) kteffnew.add(*kinmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kimmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kiimod, false, 1.0, 1.0);
  if (iset && aset) kteffnew.add(*kiamod, false, 1.0, 1.0);

  //-------------------------------------------------------- FOURTH LINE

  // add a submatrices to kteffnew
  if (aset) kteffnew.add(*smatrix_, false, 1.0, 1.0);

  //--------------------------------------------------------- FIFTH LINE
  // add st submatrices to kteffnew
  if (stickset) kteffnew.add(*kstnmod, false, 1.0, 1.0);
  if (stickset) kteffnew.add(*kstmmod, false, 1.0, 1.0);
  if (stickset && iset) kteffnew.add(*kstimod, false, 1.0, 1.0);
  if (stickset && slipset) kteffnew.add(*kstslmod, false, 1.0, 1.0);
  if (stickset) kteffnew.add(*kststmod, false, 1.0, 1.0);

  // add terms of linearization of sick condition to kteffnew
  if (stickset) kteffnew.add(*linstickDIS_, false, -1.0, 1.0);

  //--------------------------------------------------------- SIXTH LINE
  // add sl submatrices to kteffnew
  if (slipset) kteffnew.add(*kslnmod, false, 1.0, 1.0);
  if (slipset) kteffnew.add(*kslmmod, false, 1.0, 1.0);
  if (slipset && iset) kteffnew.add(*kslimod, false, 1.0, 1.0);
  if (slipset) kteffnew.add(*kslslmod, false, 1.0, 1.0);
  if (slipset && stickset) kteffnew.add(*kslstmod, false, 1.0, 1.0);

  // add terms of linearization of slip condition to kteffnew and feffnew
  if (slipset) kteffnew.add(*linslipDIS_, false, -1.0, +1.0);

  // add diagonal entries to sparsity pattern for dbc
  for (int i = 0; i < kteffnew.row_map().num_my_elements(); ++i)
  {
    int gid = kteffnew.row_map().gid(i);
    kteffnew.assemble(0., gid, gid);
  }

  // fill_complete kteffnew (square)
  kteffnew.complete();

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
    Core::LinAlg::export_to(*gact, *gexp);
    feffnew->update(-1.0, *gexp, 1.0);
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

  if (slipset)
  {
    Core::LinAlg::Vector<double> linslipRHSexp(*problem_dofs());
    Core::LinAlg::export_to(*linslipRHS_, linslipRHSexp);
    feffnew->update(-1.0, linslipRHSexp, 1.0);
  }

  // finally do the replacement
  *kteff = kteffnew;
  *feff = *feffnew;

#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckGapDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

#ifdef CONTACTFDALPHA
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckAlphaDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

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
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv1 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv2 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //
      //      deriv1->Add(*linstickLM_,false,1.0,1.0);
      //      deriv1->Complete(*gsmdofrowmap_,*gactivet_);
      //
      //      deriv2->Add(*linstickDIS_,false,1.0,1.0);
      //      deriv2->Complete(*gsmdofrowmap_,*gactivet_);
      //
      //      std::cout << "DERIV 1 *********** "<< *deriv1 << std::endl;
      //      std::cout << "DERIV 2 *********** "<< *deriv2 << std::endl;

      interface_[i]->FDCheckStickDeriv(*linstickLM_, *linstickDIS_);
    }
  }
#endif  // #ifdef CONTACTFDSTICK

#ifdef CONTACTFDSLIP

  if (gslipnodes_->NumGlobalElements())
  {
    // FD check of slip condition
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv1 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //      std::shared_ptr<Core::LinAlg::SparseMatrix> deriv2 = Teuchos::rcp(new
      //      Core::LinAlg::SparseMatrix(*gactivet_,81));
      //
      //      deriv1->Add(*linslipLM_,false,1.0,1.0);
      //      deriv1->Complete(*gsmdofrowmap_,*gslipt_);
      //
      //      deriv2->Add(*linslipDIS_,false,1.0,1.0);
      //      deriv2->Complete(*gsmdofrowmap_,*gslipt_);
      //
      //      std::cout << *deriv1 << std::endl;
      //      std::cout << *deriv2 << std::endl;

      interface_[i]->FDCheckSlipDeriv(*linslipLM_, *linslipDIS_);
    }
  }
#endif  // #ifdef CONTACTFDSLIP

  feff->scale(-1.);
  return;
}

void CONTACT::LagrangeStrategy::condense_frictionless(
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> feff =
      Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(rhs);

  feff->update(-1. + alphaf_, *strcontactrhs_, 1.);
  feff->scale(-1.);
  std::shared_ptr<Core::LinAlg::SparseOperator> kteff_op =
      std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(kteff);

  // shape function type and type of LM interpolation for quadratic elements
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  // In case of nonsmooth contact the scenario of contacting edges (non parallel)
  // requires a penalty regularization. Here, the penalty contriutions for this
  // special case are applied:
  if (nonSmoothContact_)
  {
    // LTL contributions:
    add_line_to_lin_contributions(*kteff_op, feff);

    // penalty support for master side quantities:
    add_master_contributions(*kteff_op, *feff);

#ifdef CONTACTFDGAPLTL
    // FD check of weighted gap g derivatives (non-penetr. condition)
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      interface_[i]->FDCheckGapDerivLTL();
    }
#endif  // #ifdef CONTACTFDGAPLTL
  }

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  /**********************************************************************/
  /* export weighted gap vector to gactiveN-map                         */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::Vector<double>> gact;
  if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
  {
    gact = Core::LinAlg::create_vector(*gactivedofs_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
    }
  }
  else
  {
    gact = Core::LinAlg::create_vector(*gactivenodes_, true);
    if (gact->global_length())
    {
      Core::LinAlg::export_to(*wgap_, *gact);
      gact->replace_map(*gactiven_);
    }
  }

  // double-check if this is a dual LM system
  if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin &&
      Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD") !=
          Inpar::Mortar::lagmult_const)
    FOUR_C_THROW("Condensation only for dual LM");

  /**********************************************************************/
  /* (1) Multiply Mortar matrices: m^ = inv(d) * m                      */
  /**********************************************************************/
  std::shared_ptr<Core::LinAlg::SparseMatrix> invd = invd_;

  /**********************************************************************/
  /* (2) Add contact stiffness terms to kteff                           */
  /**********************************************************************/
  // declare sparse matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);

  /**********************************************************************/
  /* (3) Split kteff into 3x3 matrix blocks                             */
  /**********************************************************************/
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

  /**********************************************************************/
  /* (4) Split feff into 3 subvectors                                   */
  /**********************************************************************/
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

  // abbreviations for slave set
  const int sset = gsdofrowmap_->num_global_elements();

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

  /**********************************************************************/
  /* (5) Split slave quantities into active / inactive                  */
  /**********************************************************************/
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

  // abbreviations for active and inactive set
  const int aset = gactivedofs_->num_global_elements();
  const int iset = gidofs->num_global_elements();

  // we want to split fsmod into 2 groups a,i
  std::shared_ptr<Core::LinAlg::Vector<double>> fa =
      std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fi =
      std::make_shared<Core::LinAlg::Vector<double>>(*gidofs);

  // do the vector splitting s -> a+i
  Core::LinAlg::split_vector(*gsdofrowmap_, *fs, gactivedofs_, fa, gidofs, fi);

  /**********************************************************************/
  /* (6) Isolate necessary parts from invd and mhatmatrix               */
  /**********************************************************************/
  // active part of invd
  std::shared_ptr<Core::LinAlg::SparseMatrix> invda;
  Core::LinAlg::split_matrix2x2(
      invd_, gactivedofs_, gidofs, gactivedofs_, gidofs, invda, tempmtx1, tempmtx2, tempmtx3);

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
  dai->scale(1 - alphaf_);

  save_coupling_matrices(dhat, mhataam, invda);
  /**********************************************************************/
  /* (7) Build the final K blocks                                       */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // knn: nothing to do

  // knm: nothing to do

  // kns: nothing to do

  //---------------------------------------------------------- SECOND LINE
  // kmn: add T(mhataam)*kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  kmnmod->add(*kmn, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kan, false, false, false, true);
  kmnmod->add(*kmnadd, false, 1.0, 1.0);
  kmnmod->complete(kmn->domain_map(), kmn->row_map());

  // kmm: add T(mhataam)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
  kmmmod->add(*kmm, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
      Core::LinAlg::matrix_multiply(*mhataam, true, *kam, false, false, false, true);
  kmmmod->add(*kmmadd, false, 1.0, 1.0);
  kmmmod->complete(kmm->domain_map(), kmm->row_map());

  // kmi: add T(mhataam)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmimod;
  if (iset)
  {
    kmimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmimod->add(*kmi, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmiadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kai, false, false, false, true);
    kmimod->add(*kmiadd, false, 1.0, 1.0);
    kmimod->complete(kmi->domain_map(), kmi->row_map());
  }

  // kma: add T(mhataam)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmamod;
  if (aset)
  {
    kmamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmamod->add(*kma, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmaadd =
        Core::LinAlg::matrix_multiply(*mhataam, true, *kaa, false, false, false, true);
    kmamod->add(*kmaadd, false, 1.0, 1.0);
    kmamod->complete(kma->domain_map(), kma->row_map());
  }

  //----------------------------------------------------------- THIRD LINE
  //------------------- FOR 3D QUADRATIC CASE ----------------------------
  // kin: subtract T(dhat)*kan --
  std::shared_ptr<Core::LinAlg::SparseMatrix> kinmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  kinmod->add(*kin, false, 1.0, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kinadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kan, false, false, false, true);
    kinmod->add(*kinadd, false, -1.0, 1.0);
  }
  kinmod->complete(kin->domain_map(), kin->row_map());

  // kim: subtract T(dhat)*kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kimmod =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
  kimmod->add(*kim, false, 1.0, 1.0);
  if (aset && iset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> kimadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kam, false, false, false, true);
    kimmod->add(*kimadd, false, -1.0, 1.0);
  }
  kimmod->complete(kim->domain_map(), kim->row_map());

  // kii: subtract T(dhat)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiimod;
  if (iset)
  {
    kiimod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kiimod->add(*kii, false, 1.0, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiiadd =
          Core::LinAlg::matrix_multiply(*dhat, true, *kai, false, false, false, true);
      kiimod->add(*kiiadd, false, -1.0, 1.0);
    }
    kiimod->complete(kii->domain_map(), kii->row_map());
  }

  // kia: subtract T(dhat)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kiamod;
  if (iset && aset)
  {
    kiamod = std::make_shared<Core::LinAlg::SparseMatrix>(*gidofs, 100);
    kiamod->add(*kia, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kiaadd =
        Core::LinAlg::matrix_multiply(*dhat, true, *kaa, false, false, false, true);
    kiamod->add(*kiaadd, false, -1.0, 1.0);
    kiamod->complete(kia->domain_map(), kia->row_map());
  }

  //---------------------------------------------------------- FOURTH LINE
  // nothing to do

  //----------------------------------------------------------- FIFTH LINE
  // kan: multiply tmatrix with invda and kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> kanmod;
  if (aset)
  {
    kanmod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
    kanmod = Core::LinAlg::matrix_multiply(*kanmod, false, *kan, false, false, false, true);
  }

  // kam: multiply tmatrix with invda and kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> kammod;
  if (aset)
  {
    kammod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
    kammod = Core::LinAlg::matrix_multiply(*kammod, false, *kam, false, false, false, true);
  }

  // kai: multiply tmatrix with invda and kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> kaimod;
  if (aset && iset)
  {
    kaimod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
    kaimod = Core::LinAlg::matrix_multiply(*kaimod, false, *kai, false, false, false, true);
  }

  // kaa: multiply tmatrix with invda and kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> kaamod;
  if (aset)
  {
    kaamod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
    kaamod = Core::LinAlg::matrix_multiply(*kaamod, false, *kaa, false, false, false, true);
  }

  /**********************************************************************/
  /* (8) Build the final f blocks                                       */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // fn: nothing to do

  //---------------------------------------------------------- SECOND LINE

  // fm: add T(mhat)*fa
  Core::LinAlg::Vector<double> fmmod(*gmdofrowmap_);
  if (aset) mhataam->multiply(true, *fa, fmmod);
  fmmod.update(1.0, *fm, 1.0);

  //----------------------------------------------------------- THIRD LINE
  // fi: add T(dhat)*fa
  Core::LinAlg::Vector<double> fimod(*gidofs);
  if (iset && aset) dhat->multiply(true, *fa, fimod);
  fimod.update(1.0, *fi, -1.0);

  //---------------------------------------------------------- FOURTH LINE
  // gactive: nothing to do

  //----------------------------------------------------------- FIFTH LINE
  // fa: multiply tmatrix with invda and fa
  std::shared_ptr<Core::LinAlg::Vector<double>> famod;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tinvda;
  if (aset)
  {
    if (constr_direction_ == CONTACT::ConstraintDirection::xyz)
      famod = std::make_shared<Core::LinAlg::Vector<double>>(*gactivedofs_);
    else
      famod = std::make_shared<Core::LinAlg::Vector<double>>(*gactivet_);

    tinvda = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda, true, false, false, true);
    tinvda->multiply(false, *fa, *famod);
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
    if (aset) smatrix_ = Core::LinAlg::matrix_row_transform(*smatrix_, *non_redist_gsdofrowmap_);

    //----------------------------------------------------------- FIFTH LINE
    if (aset)
    {
      kanmod = Core::LinAlg::matrix_row_transform(*kanmod, *non_redist_gsdofrowmap_);
      kammod = Core::LinAlg::matrix_row_transform(*kammod, *non_redist_gsdofrowmap_);
      kaamod = Core::LinAlg::matrix_row_transform(*kaamod, *non_redist_gsdofrowmap_);
      if (iset) kaimod = Core::LinAlg::matrix_row_transform(*kaimod, *non_redist_gsdofrowmap_);
      tderivmatrix_ = Core::LinAlg::matrix_row_transform(*tderivmatrix_, *non_redist_gsdofrowmap_);
    }
  }

  /**********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                  */
  /**********************************************************************/

  Core::LinAlg::SparseMatrix kteffnew(
      *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
  std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
      Core::LinAlg::create_vector(*problem_dofs());

  //----------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  kteffnew.add(*knn, false, 1.0, 1.0);
  kteffnew.add(*knm, false, 1.0, 1.0);
  if (sset) kteffnew.add(*kns, false, 1.0, 1.0);

  //---------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  kteffnew.add(*kmnmod, false, 1.0, 1.0);
  kteffnew.add(*kmmmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kmimod, false, 1.0, 1.0);
  if (aset) kteffnew.add(*kmamod, false, 1.0, 1.0);

  //----------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) kteffnew.add(*kinmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kimmod, false, 1.0, 1.0);
  if (iset) kteffnew.add(*kiimod, false, 1.0, 1.0);
  if (iset && aset) kteffnew.add(*kiamod, false, 1.0, 1.0);

  //---------------------------------------------------------- FOURTH LINE
  // add a submatrices to kteffnew
  if (aset) kteffnew.add(*smatrix_, false, 1.0, 1.0);

  //----------------------------------------------------------- FIFTH LINE
  // add a submatrices to kteffnew
  if (aset) kteffnew.add(*kanmod, false, 1.0, 1.0);
  if (aset) kteffnew.add(*kammod, false, 1.0, 1.0);
  if (aset && iset) kteffnew.add(*kaimod, false, 1.0, 1.0);
  if (aset) kteffnew.add(*kaamod, false, 1.0, 1.0);
  if (aset) kteffnew.add(*tderivmatrix_, false, -1.0, 1.0);

  // fill_complete kteffnew (square)
  kteffnew.complete();

  /**********************************************************************/
  /* (11) Global setup of feffnew (including contact)                   */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // add n subvector to feffnew
  Core::LinAlg::Vector<double> fnexp(*problem_dofs());
  Core::LinAlg::export_to(*fn, fnexp);
  feffnew->update(1.0, fnexp, 1.0);

  //---------------------------------------------------------- SECOND LINE
  // add m subvector to feffnew
  Core::LinAlg::Vector<double> fmmodexp(*problem_dofs());
  Core::LinAlg::export_to(fmmod, fmmodexp);
  feffnew->update(1.0, fmmodexp, 1.0);

  //----------------------------------------------------------- THIRD LINE
  // add i subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fimodexp;
  if (iset)
  {
    fimodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(fimod, *fimodexp);
    feffnew->update(1.0, *fimodexp, 1.0);
  }

  //---------------------------------------------------------- FOURTH LINE
  // add weighted gap vector to feffnew, if existing
  std::shared_ptr<Core::LinAlg::Vector<double>> gexp;
  if (aset)
  {
    gexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*gact, *gexp);
    feffnew->update(-1.0, *gexp, 1.0);
  }

  //----------------------------------------------------------- FIFTH LINE
  // add a subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> famodexp;
  if (aset)
  {
    famodexp = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
    Core::LinAlg::export_to(*famod, *famodexp);
    feffnew->update(1.0, *famodexp, 1.0);
  }

  // finally do the replacement
  *kteff = kteffnew;
  *feff = *feffnew;


#ifdef CONTACTFDGAP
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckGapDeriv();
  }
#endif  // #ifdef CONTACTFDGAP

#ifdef CONTACTFDALPHA
  // FD check of weighted gap g derivatives (non-penetr. condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->FDCheckAlphaDeriv();
  }
#endif  // #ifdef CONTACTFDGAP
#ifdef CONTACTFDTANGLM
  // FD check of tangential LM derivatives (frictionless condition)
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    std::cout << *tderivmatrix_ << std::endl;
    interface_[i]->FDCheckTangLMDeriv();
  }
#endif  // #ifdef CONTACTFDTANGLM


  feff->scale(-1.);
}

void CONTACT::LagrangeStrategy::run_pre_apply_jacobian_inverse(
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (is_dual_quad_slave_trafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
  {
    if (not(systrafo_->domain_map().same_as(kteff->domain_map()))) FOUR_C_THROW("stop");

    *kteff = *Core::LinAlg::matrix_multiply(*systrafo_, true, *kteff, false, false, false, true);
    Core::LinAlg::Vector<double> rhs_str(*problem_dofs());
    Core::LinAlg::Vector<double> rhs_str2(*problem_dofs());
    Core::LinAlg::export_to(rhs, rhs_str);
    systrafo_->multiply(true, rhs_str, rhs_str2);
    for (int i = 0; i < rhs_str2.get_map().num_my_elements(); ++i)
      rhs.get_values()[rhs.get_map().lid(rhs_str2.get_map().gid(i))] =
          rhs_str2.local_values_as_span()[i];
  }


  if (systype_ != CONTACT::SystemType::condensed) return;

  if (friction_)
    condense_friction(kteff, rhs);
  else
    condense_frictionless(kteff, rhs);

  return;
}

void CONTACT::LagrangeStrategy::run_post_apply_jacobian_inverse(
    const CONTACT::ParamsInterface& cparams, const Core::LinAlg::Vector<double>& rhs,
    Core::LinAlg::Vector<double>& result, const Core::LinAlg::Vector<double>& xold,
    const NOX::Nln::Group& grp)
{
  if (system_type() != CONTACT::SystemType::condensed) return;

  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  std::shared_ptr<Core::LinAlg::Vector<double>> disi =
      Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(result);
  disi->scale(-1.);
  {
    // shape function type and type of LM interpolation for quadratic elements
    auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

    // double-check if this is a dual LM system
    if ((shapefcn != Inpar::Mortar::shape_dual &&
            shapefcn != Inpar::Mortar::shape_petrovgalerkin) &&
        Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD") !=
            Inpar::Mortar::lagmult_const)
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
    invdmod.add(*invda, false, 1.0, 1.0);
    invdmod.complete();

    /**********************************************************************/
    /* Update Lagrange multipliers z_n+1                                  */
    /**********************************************************************/
    // for self contact, slave and master sets may have changed,
    // thus we have to export the products Dold * zold and Mold^T * zold to fit
    if (is_self_contact())
    {
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
      Core::LinAlg::Vector<double> zcopy(*z_);
      invdmod.multiply(true, zcopy, *z_);
      z_->scale(1 / (1 - alphaf_));
    }
    else
    {
      // full update
      z_->update(1.0, *fs_, 0.0);
      Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
      kss_->multiply(false, disis, mod);
      z_->update(-1.0, mod, 1.0);
      ksm_->multiply(false, disim, mod);
      z_->update(-1.0, mod, 1.0);
      ksn_->multiply(false, disin, mod);
      z_->update(-1.0, mod, 1.0);
      Core::LinAlg::Vector<double> zcopy(*z_);
      invdmod.multiply(true, zcopy, *z_);
      z_->scale(1 / (1 - alphaf_));
    }

    // store updated LM into nodes
    store_nodal_quantities(Mortar::StrategyBase::lmupdate);
  }
  disi->scale(-1.);
}

FOUR_C_NAMESPACE_CLOSE
