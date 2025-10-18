// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_lagrange_strategy_poro.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_utils.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                              ager 08/14|
 *----------------------------------------------------------------------*/
CONTACT::LagrangeStrategyPoro::LagrangeStrategyPoro(
    const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
    int dim, MPI_Comm comm, double alphaf, int maxdof, bool poroslave, bool poromaster)
    : MonoCoupledLagrangeStrategy(
          data_ptr, dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof),
      no_penetration_(params.get<bool>("CONTACT_NO_PENETRATION")),
      nopenalpha_(0.0),
      poroslave_(poroslave),
      poromaster_(poromaster)
{
  if (!poroslave_ and !poromaster_)
    FOUR_C_THROW(
        "you called a poroelastic meshtying method without participating poroelastic domains on "
        "your interface");
  return;
}

/*----------------------------------------------------------------------*
 |  read restart information for contact                      ager 12/16|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::do_read_restart(Core::IO::DiscretizationReader& reader,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr)
{
  std::shared_ptr<Core::FE::Discretization> discret =
      Global::Problem::instance()->get_dis("structure");
  if (discret == nullptr) FOUR_C_THROW("didn't get my discretization");

  std::shared_ptr<Core::LinAlg::Vector<double>> global =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret->dof_col_map(), true);
  // it's clear that we get some zeros here ... but poroelast monolithic fixes this a little bit
  // later by doing the same thing with correct displacements again :-)
  Core::LinAlg::export_to(*dis, *global);
  set_parent_state(Mortar::StateType::state_new_displacement, *global, *discret);

  // Call (nearly absolute)Base Class
  CONTACT::AbstractStrategy::do_read_restart(reader, dis, cparams_ptr);
}

/*----------------------------------------------------------------------*
 | setup this strategy object                                ager 12/16 |
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::setup(bool redistributed, bool init)
{
  // Call Base Class
  CONTACT::AbstractStrategy::setup(redistributed, init);

  if (no_penetration_) setup_no_penetration_condition();
}

/*----------------------------------------------------------------------*
 | Activate No-Penetration for the contact surface (public)   ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::setup_no_penetration_condition()
{
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);
  lambdaold_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);
  if (!poroslave_ and poromaster_)
    FOUR_C_THROW("poroelastic meshtying/contact method needs the slave side to be poroelastic");
  /*
   *  The first error may also occur when there is no single element on the interface (it is empty)
   * but POROELASTICITY DYNAMIC coupling algorithms (COUPALGO) is chosen as
   * poro_monolithicmeshtying that means that the method creates an interface but has nothing to
   * fill in
   *
   *  the second error shows that there are no elements with PoroCoupling on the slave side of the
   * interface and the master side is fully poroelastic
   *
   *  having the condition with structure slave and poro master the problem is that there is no
   * diagonal D Matrix, from the porofluid lagrange multiplier equality condition, that can be
   * inverted easily to condense out the respective lagrange multipliers
   *
   *  there are two alternatives:
   *  1. do not condense out the lagrange multiplier for the porofluid meshtying condition in
   *  this constellation (master/slave) and solve the saddlepoint system
   *
   *  2. invert the M Matrix to condense out the Lagrange multiplier
   *  - which is only applicable with adequate costs for small problems as the M Matrix is not at
   * all diagonal, except for matching meshes - for matching meshes it doesn't hurt to chose the
   * poro side as the slave side. Being applicable only for small problems this alternative is too
   * restricted in its application overall to be implemented.
   */
}

/*----------------------------------------------------------------------*
 | initialize global poro contact variables                             |
 |                            for next Newton step (public)   ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::poro_initialize(
    Coupling::Adapter::Coupling& coupfs, const Core::LinAlg::Map& fluiddofs, bool fullinit)
{
  if (fullinit)  // fullinit is true by default, but needed when this method is called for
                 // meshtying, as the maps and matrix mapping stay the same for meshtying. would
                 // work without this but would do things repeatedly unnecessarily
  {
    if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
    {
      //  (1)                                                          //
      //      Get required fluid maps from structural maps             //
      //                                                               //
      fgsdofrowmap_ = coupfs.master_to_slave_map(*gsdofrowmap_);
      fgmdofrowmap_ = coupfs.master_to_slave_map(*gmdofrowmap_);
      fgsmdofrowmap_ = coupfs.master_to_slave_map(*gsmdofrowmap_);
      fgndofrowmap_ = Core::LinAlg::split_map(fluiddofs,
          *fgsmdofrowmap_);  // Not equal to transforming gndofrowmap_ (pressure dofs missing!)
      fgactivedofs_ = coupfs.master_to_slave_map(*gactivedofs_);
      falldofrowmap_ = std::make_shared<Core::LinAlg::Map>(fluiddofs);
      fgactiven_ = coupfs.master_to_slave_map(*gactiven_);
      fgactivet_ = coupfs.master_to_slave_map(*gactivet_);
    }
  }
  //  (2)                                                          //
  //      Initialize Matrices                                      //
  //                                                               //
  if (fullinit)
  {
    if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
    {
      // (re)setup global nCoup Vector
      NCoup_ = std::make_shared<Core::LinAlg::Vector<double>>(*gactiven_);

      // (re)setup global linearisation matrices of nCoup
      NCoup_lindisp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 10);
      NCoup_linvel_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactiven_, 10);

      // (re)setup global tangential and lin(tangential)*lambda matrices
      Tangential_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivet_, 10);
      linTangentiallambda_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gactivet_, 10);

      // (re)setup global lin of D & M * lambda - Matrix
      porolindmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
      porolinmmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
    }
  }
  else
  {
    // In the case of meshtying resetting the matrices is sufficient as they retain their size
    NCoup_->put_scalar(0.0);

    // (re)setup global linearisation matrices of nCoup
    NCoup_lindisp_->zero();
    NCoup_linvel_->zero();

    // (re)setup global tangential and lin(tangential)*lambda matrices
    Tangential_->zero();
    linTangentiallambda_->zero();

    // (re)setup global lin of D & M * lambda - Matrix
    porolindmatrix_->zero();
    porolinmmatrix_->zero();
  }
  //  (3)                                                          //
  //      Assemble Matrices                                        //
  //                                                               //
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
    {
      interface_[i]->assemble_normal_coupling(*NCoup_);

      interface_[i]->assemble_normal_coupling_linearisation(*NCoup_lindisp_, coupfs);
      interface_[i]->assemble_normal_coupling_linearisation(*NCoup_linvel_, coupfs, true);

      interface_[i]->assemble_tn(Tangential_, nullptr);
      interface_[i]->assemble_t_nderiv(linTangentiallambda_, nullptr,
          true);  // use lambda(n +1) for tangential condition!!!

      interface_[i]->assemble_lin_dm(*porolindmatrix_, *porolinmmatrix_, true);
    }
  }

  //  (4)                                                          //
  //      Complete Matrices                                        //
  //                                                               //
  if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
  {
    NCoup_lindisp_->complete(*problem_dofs(), *gactiven_);
    NCoup_linvel_->complete(fluiddofs, *gactiven_);

    Tangential_->complete(*gactivedofs_, *gactivet_);
    linTangentiallambda_->complete(*gsmdofrowmap_, *gactivet_);

    porolindmatrix_->complete(*gsmdofrowmap_, *gsdofrowmap_);
    porolinmmatrix_->complete(*gsmdofrowmap_, *gmdofrowmap_);
  }

  //  (5)                                                          //
  //      Reset Matrix Transform Objects                           //
  //                                                               //
  if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
  {
    linncoupveltransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    linncoupdisptransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    tanginvtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
    lintangentlambdatransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    porolindmatrixtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    porolinmmatrixtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    mhataamtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();  // h.Willmann
    dhattransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    doldtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    moldtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
    invDatransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  }

  //  (6)                                                          //
  //      Transform Matrices from structural dofs to fluid dofs    //
  //                                                               //
  if (no_penetration_ && (is_in_contact() || was_in_contact() || was_in_contact_last_time_step()))
  // eventually a coupling object just on the mortar interface would make sense!!! ChrAg
  {
    // transform matrices coming from contact to fluid maps, as they are all in structure maps!
    //
    // A general problem here is that we would need to update coupling objects in everey newton
    // step if the active set changes. To avoid this, a 'bigger' coupling object is used - but
    // therefore now the Row-Maps of the created Sparse Matrixes are to big! --- Leads to problems
    // for Matrix - Multiplications where the Row - Map is used!
    //
    // At the moment this is solved by using split_matrix2x2 to cut out just the relevant part of
    // the matrix, but this shouldn't be the final solution!
    //
    //************************************************************************************************
    //
    std::shared_ptr<Core::LinAlg::Vector<double>> tmpfullncoup =
        std::make_shared<Core::LinAlg::Vector<double>>(*coupfs.master_dof_map());
    Core::LinAlg::export_to(*NCoup_, *tmpfullncoup);
    tmpfullncoup = coupfs.master_to_slave(*tmpfullncoup);
    fNCoup_ = std::make_shared<Core::LinAlg::Vector<double>>(*fgactiven_);
    Core::LinAlg::export_to(*tmpfullncoup, *fNCoup_);
    //
    //************************************************************************************************
    //
    fdoldtransp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
    (*doldtransform_)(*Core::LinAlg::matrix_transpose(*dold_), 1.0,
        Coupling::Adapter::CouplingMasterConverter(coupfs), *fdoldtransp_, false);
    fdoldtransp_->complete(dold_->row_map(), *fgsdofrowmap_);
    //
    //************************************************************************************************
    //
    if (poromaster_)
    {
      fmoldtransp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
      (*moldtransform_)(*Core::LinAlg::matrix_transpose(*mold_), 1.0,
          Coupling::Adapter::CouplingMasterConverter(coupfs), *fmoldtransp_, false);
      fmoldtransp_->complete(mold_->row_map(), *fgmdofrowmap_);
    }
    //
    //************************************************************************************************
    //
    fporolindmatrix_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
    (*porolindmatrixtransform_)(*porolindmatrix_, 1.0,
        Coupling::Adapter::CouplingMasterConverter(coupfs), *fporolindmatrix_, false);
    fporolindmatrix_->complete(porolindmatrix_->domain_map(), *fgsdofrowmap_);
    //
    //************************************************************************************************
    //
    if (poromaster_)
    {
      fporolinmmatrix_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
      (*porolinmmatrixtransform_)(*porolinmmatrix_, 1.0,
          Coupling::Adapter::CouplingMasterConverter(coupfs), *fporolinmmatrix_, false);
      fporolinmmatrix_->complete(porolinmmatrix_->domain_map(), *fgmdofrowmap_);
    }
    // porolinmmatrixtransform_ is no longer missing as twosided poro meshtying is considered!
    //
    //************************************************************************************************
    if (poromaster_)
    //
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> tmpfmhataam =
          std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);

      fmhataam_ = std::make_shared<Core::LinAlg::SparseMatrix>(*fgmdofrowmap_, 1, true, false);
      (*mhataamtransform_)(*mhataam_, 1.0, Coupling::Adapter::CouplingMasterConverter(coupfs),
          Coupling::Adapter::CouplingMasterConverter(coupfs), *tmpfmhataam, false, false);

      tmpfmhataam->complete(*fgmdofrowmap_, *falldofrowmap_);

      // better solution to get maps as wanted? -- for this matrix map as important as there will be
      // a matrix-matrix multiplication

      std::shared_ptr<Core::LinAlg::Map> restfgmdofrowmap, restfgactivedofs;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tmpm1, tmpm2, tmpm3;

      // This should just be a temporary solution to change the row map of the matrix ...
      Core::LinAlg::split_matrix2x2(tmpfmhataam, fgactivedofs_, restfgactivedofs, fgmdofrowmap_,
          restfgmdofrowmap, fmhataam_, tmpm1, tmpm2, tmpm3);
      fmhataam_->complete(*fgmdofrowmap_, *fgactivedofs_);
    }
    //
    //************************************************************************************************
    //
    fdhat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
    (*dhattransform_)(
        *dhat_, 1.0, Coupling::Adapter::CouplingMasterConverter(coupfs), *fdhat_, false);
    fdhat_->complete(dhat_->domain_map(), *fgactivedofs_);
    // fdhat is expected to be zero in this method but still used for condensation
    //
    //************************************************************************************************
    //
    if (gactivedofs_->num_global_elements())
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> tanginvD =
          Core::LinAlg::matrix_multiply(*Tangential_, false, *invda_, true, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> tmpftanginvD =
          std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
      (*tanginvtransform_)(*tanginvD, 1.0, Coupling::Adapter::CouplingMasterConverter(coupfs),
          Coupling::Adapter::CouplingMasterConverter(coupfs), *tmpftanginvD, false, false);
      tmpftanginvD->complete(*fgactivedofs_, *falldofrowmap_);
      // better solution to get maps as wanted? -- for this matrix map as important as there will be
      // a matrix-matrix multiplication

      std::shared_ptr<Core::LinAlg::Map> restfgactivet, restfgactivedofs;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tmpm1, tmpm2, tmpm3;

      // This should just be a temporary solution to change the row map of the matrix ...
      Core::LinAlg::split_matrix2x2(tmpftanginvD, fgactivet_, restfgactivet, fgactivedofs_,
          restfgactivedofs, ftanginvD_, tmpm1, tmpm2, tmpm3);

      //
      //************************************************************************************************
      //
      fNCoup_linvel_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 108, false);

      (*linncoupveltransform_)(*NCoup_linvel_, 1.0,
          Coupling::Adapter::CouplingMasterConverter(coupfs), *fNCoup_linvel_, false);
      fNCoup_linvel_->complete(*falldofrowmap_, *fgactiven_);
      //
      //************************************************************************************************
      //
      fNCoup_lindisp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 81, false);
      (*linncoupdisptransform_)(*NCoup_lindisp_, 1.0,
          Coupling::Adapter::CouplingMasterConverter(coupfs), *fNCoup_lindisp_, false);
      fNCoup_lindisp_->complete(*problem_dofs(), *fgactiven_);
      //
      //************************************************************************************************
      //
      flinTangentiallambda_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 81, false);
      (*lintangentlambdatransform_)(*linTangentiallambda_, 1.0,
          Coupling::Adapter::CouplingMasterConverter(coupfs), *flinTangentiallambda_, false);
      flinTangentiallambda_->complete(*gsdofrowmap_, *fgactivet_);

      //
      //************************************************************************************************
      //
      finvda_ = std::make_shared<Core::LinAlg::SparseMatrix>(*falldofrowmap_, 1, true, false);
      (*invDatransform_)(
          *invda_, 1.0, Coupling::Adapter::CouplingMasterConverter(coupfs), *finvda_, false);
      finvda_->complete(invda_->domain_map(), *fgactivedofs_);
    }
  }
}  // CONTACT::LagrangeStrategyPoro::PoroInitialize

/*-------------------------------------------------------------------------*
 |  evaluate poro coupling contact matrices for no penetration             |
 |  condition on contact surface (pure porous problem)(public)   ager 07/15|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::evaluate_poro_no_pen_contact(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& k_fseff,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& Feff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  evaluate_mat_poro_no_pen(k_fseff, feff);
  evaluate_other_mat_poro_no_pen(Feff, 0);
}

/*-------------------------------------------------------------------------*
 |  evaluate poro coupling contact matrices for no penetration             |
 |  condition on contact surface(public)                         ager 07/15|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::evaluate_poro_no_pen_contact(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& k_fseff,
    std::map<int, std::shared_ptr<Core::LinAlg::SparseMatrix>*>& Feff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  evaluate_mat_poro_no_pen(k_fseff, feff);

  // Take care of the alternative condensation of the off-diagonal blocks!!!
  std::map<int, std::shared_ptr<Core::LinAlg::SparseMatrix>*>::iterator matiter;
  for (matiter = Feff.begin(); matiter != Feff.end(); ++matiter)
  {
    evaluate_other_mat_poro_no_pen(*(matiter->second), matiter->first);
  }
}

/*----------------------------------------------------------------------*
 |  evaluate poro coupling contact matrices for no penetration          |
 |  condition on contact surface (public)                     ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::evaluate_mat_poro_no_pen(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& k_fseff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!no_penetration_ ||
      (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()))
    return;
  // h.Willmann this method should be renamed as it handles twosided poro meshtying now aswell and
  // is able to handle fluid coupling for twosided contact

  nopenalpha_ = alphaf_;  // to use different alpha for nopen condition (not used at the moment)

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************

  // double-check if this is a dual LM system
  if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Condensation only for dual LM");

  /**********************************************************************/
  /* (2) Add contact stiffness terms to kteff                           */
  /**********************************************************************/

  // transform if necessary
  if (parallel_redistribution_status())
  {
    FOUR_C_THROW("CHECK ME!!!");
    lindmatrix_ = Core::LinAlg::matrix_row_transform(*lindmatrix_, *non_redist_gsdofrowmap_);
    linmmatrix_ = Core::LinAlg::matrix_row_transform(*linmmatrix_, *non_redist_gmdofrowmap_);
  }

  k_fseff->un_complete();
  k_fseff->add(*fporolindmatrix_, false, (1.0 - nopenalpha_) * 1.0, 1.0);

  if (poromaster_)
    k_fseff->add(*fporolinmmatrix_, false, (1.0 - nopenalpha_) * 1.0,
        1.0);  // is needed only for twosided poro contact or meshtying

  k_fseff->complete(*problem_dofs(), *falldofrowmap_);  // gets bigger because of linearisation
                                                        // w.r.t. to pure structural displacements!
  /**********************************************************************/
  /* (3) Split k_fseff and Feff into 3x3 matrix blocks                             */
  /**********************************************************************/

  // we want to split k into 3 groups s,m,n = 9 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_ss, k_fs_sm, k_fs_sn, k_fs_ms, k_fs_mm, k_fs_mn,
      k_fs_ns, k_fs_nm, k_fs_nn;

  // temporarily we need the blocks ksmsm, ksmn, knsm
  // (FIXME: because a direct SplitMatrix3x3 is still missing!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_smsm, k_fs_smn, k_fs_nsm;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap;
  std::shared_ptr<Core::LinAlg::Map> ftempmap1, ftempmap2, ftempmap3;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;

  // split into slave/master part + structure part
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fseffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_fseff);
  if (parallel_redistribution_status())
  {
    FOUR_C_THROW("CHECK ME!");
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_matrix2x2(k_fseffmatrix, fgsmdofrowmap_, fgndofrowmap_, gsmdofrowmap_,
        gndofrowmap_, k_fs_smsm, k_fs_smn, k_fs_nsm, k_fs_nn);
  }

  // further splits into slave part + master part
  Core::LinAlg::split_matrix2x2(k_fs_smsm, fgsdofrowmap_, fgmdofrowmap_, gsdofrowmap_, gmdofrowmap_,
      k_fs_ss, k_fs_sm, k_fs_ms, k_fs_mm);
  Core::LinAlg::split_matrix2x2(k_fs_smn, fgsdofrowmap_, fgmdofrowmap_, gndofrowmap_, tempmap,
      k_fs_sn, tempmtx1, k_fs_mn, tempmtx2);
  Core::LinAlg::split_matrix2x2(k_fs_nsm, fgndofrowmap_, ftempmap1, gsdofrowmap_, gmdofrowmap_,
      k_fs_ns, k_fs_nm, tempmtx1, tempmtx2);

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
    FOUR_C_THROW("CHECK ME!");
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
    Core::LinAlg::split_vector(*falldofrowmap_, *feff, fgsmdofrowmap_, fsm, fgndofrowmap_, fn);
  }

  // abbreviations for slave  and master set
  int sset = fgsdofrowmap_->num_global_elements();
  int mset = fgmdofrowmap_->num_global_elements();

  // we want to split fsm into 2 groups s,m
  fs = std::make_shared<Core::LinAlg::Vector<double>>(*fgsdofrowmap_);
  fm = std::make_shared<Core::LinAlg::Vector<double>>(*fgmdofrowmap_);

  // do the vector splitting sm -> s+m
  Core::LinAlg::split_vector(*fgsmdofrowmap_, *fsm, fgsdofrowmap_, fs, fgmdofrowmap_, fm);

  // store some stuff for static condensation of poro no pen. LM

  ffs_ = fs;

  cfssn_ = k_fs_sn;
  cfssm_ = k_fs_sm;
  cfsss_ = k_fs_ss;

  /**********************************************************************/
  /* (5) Split slave quantities into active / inactive                  */
  /**********************************************************************/

  // we want to split kssmod into 2 groups a,i = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_aa, k_fs_ai, k_fs_ia, k_fs_ii;

  // we want to split ksn / ksm / kms into 2 groups a,i = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_an, k_fs_in, k_fs_am, k_fs_im, k_fs_ma, k_fs_mi;

  // we will get the i rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> gidofs;
  std::shared_ptr<Core::LinAlg::Map> fgidofs;

  // some more temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap1, tempmap2;
  std::shared_ptr<Core::LinAlg::Map> ftempmap4, ftempmap5, ftempmap6, ftempmap7;

  // do the splitting
  Core::LinAlg::split_matrix2x2(
      k_fs_ss, fgactivedofs_, fgidofs, gactivedofs_, gidofs, k_fs_aa, k_fs_ai, k_fs_ia, k_fs_ii);
  Core::LinAlg::split_matrix2x2(k_fs_sn, fgactivedofs_, fgidofs, gndofrowmap_, tempmap1, k_fs_an,
      tempmtx1, k_fs_in, tempmtx2);
  Core::LinAlg::split_matrix2x2(k_fs_sm, fgactivedofs_, fgidofs, gmdofrowmap_, tempmap2, k_fs_am,
      tempmtx1, k_fs_im, tempmtx2);
  Core::LinAlg::split_matrix2x2(k_fs_ms, fgmdofrowmap_, ftempmap4, gactivedofs_, gidofs, k_fs_ma,
      k_fs_mi, tempmtx1, tempmtx2);

  // abbreviations for active and inactive set
  int aset = fgactivedofs_->num_global_elements();
  int iset = fgidofs->num_global_elements();

  // we want to split fsmod into 2 groups a,i
  std::shared_ptr<Core::LinAlg::Vector<double>> fa =
      std::make_shared<Core::LinAlg::Vector<double>>(*fgactivedofs_);
  std::shared_ptr<Core::LinAlg::Vector<double>> fi =
      std::make_shared<Core::LinAlg::Vector<double>>(*fgidofs);

  // do the vector splitting s -> a+i
  Core::LinAlg::split_vector(*fgsdofrowmap_, *fs, fgactivedofs_, fa, fgidofs, fi);

  /**********************************************************************/
  /* (7) Build the final K blocks                                       */
  /* where K stands for k_fs and F!!!                                   */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // knn: nothing to do

  // knm: nothing to do

  // kns: nothing to do
  //---------------------------------------------------------- SECOND LINE --- Will just exist when
  // starting with two sided poro contact!!!
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mnmod;

  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mmmod;

  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mimod;

  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mamod;
  if (mset)
  {
    // kmn: add T(mhataam)*kan
    k_fs_mnmod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgmdofrowmap_, 100);
    k_fs_mnmod->add(*k_fs_mn, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mnadd =
        Core::LinAlg::matrix_multiply(*fmhataam_, true, *k_fs_an, false, false, false, true);
    k_fs_mnmod->add(*k_fs_mnadd, false, 1.0, 1.0);
    k_fs_mnmod->complete(k_fs_mn->domain_map(), k_fs_mn->row_map());

    // kmm: add T(mhataam)*kam
    k_fs_mmmod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgmdofrowmap_, 100);
    k_fs_mmmod->add(*k_fs_mm, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_mmadd =
        Core::LinAlg::matrix_multiply(*fmhataam_, true, *k_fs_am, false, false, false, true);
    k_fs_mmmod->add(*k_fs_mmadd, false, 1.0, 1.0);
    k_fs_mmmod->complete(k_fs_mm->domain_map(), k_fs_mm->row_map());

    // kmi: add T(mhataam)*kai
    if (iset)
    {
      k_fs_mimod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgmdofrowmap_, 100);
      k_fs_mimod->add(*k_fs_mi, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_miadd =
          Core::LinAlg::matrix_multiply(*fmhataam_, true, *k_fs_ai, false, false, false, true);
      k_fs_mimod->add(*k_fs_miadd, false, 1.0, 1.0);
      k_fs_mimod->complete(k_fs_mi->domain_map(), k_fs_mi->row_map());
    }

    // kma: add T(mhataam)*kaa
    if (aset)
    {
      k_fs_mamod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgmdofrowmap_, 100);
      k_fs_mamod->add(*k_fs_ma, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_maadd =
          Core::LinAlg::matrix_multiply(*fmhataam_, true, *k_fs_aa, false, false, false, true);
      k_fs_mamod->add(*k_fs_maadd, false, 1.0, 1.0);
      k_fs_mamod->complete(k_fs_ma->domain_map(), k_fs_ma->row_map());
    }
  }

  //----------------------------------------------------------- THIRD LINE
  //------------------- FOR 3D QUADRATIC CASE ----------------------------
  // fdhat is expected to be zero here but still used for condensation
  // kin: subtract T(dhat)*kan --
  Core::LinAlg::SparseMatrix k_fs_inmod(*fgidofs, 100);
  k_fs_inmod.add(*k_fs_in, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_inadd =
      Core::LinAlg::matrix_multiply(*fdhat_, true, *k_fs_an, false, false, false, true);
  k_fs_inmod.add(*k_fs_inadd, false, -1.0, 1.0);
  k_fs_inmod.complete(k_fs_in->domain_map(), k_fs_in->row_map());

  // kim: subtract T(dhat)*kam
  Core::LinAlg::SparseMatrix k_fs_immod(*fgidofs, 100);
  k_fs_immod.add(*k_fs_im, false, 1.0, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_imadd =
      Core::LinAlg::matrix_multiply(*fdhat_, true, *k_fs_am, false, false, false, true);
  k_fs_immod.add(*k_fs_imadd, false, -1.0, 1.0);
  k_fs_immod.complete(k_fs_im->domain_map(), k_fs_im->row_map());

  // kii: subtract T(dhat)*kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_iimod;
  if (iset)
  {
    k_fs_iimod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgidofs, 100);
    k_fs_iimod->add(*k_fs_ii, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_iiadd =
        Core::LinAlg::matrix_multiply(*fdhat_, true, *k_fs_ai, false, false, false, true);
    k_fs_iimod->add(*k_fs_iiadd, false, -1.0, 1.0);
    k_fs_iimod->complete(k_fs_ii->domain_map(), k_fs_ii->row_map());
  }

  // kia: subtract T(dhat)*kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_iamod;
  if (iset && aset)
  {
    k_fs_iamod = std::make_shared<Core::LinAlg::SparseMatrix>(*fgidofs, 100);
    k_fs_iamod->add(*k_fs_ia, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_iaadd =
        Core::LinAlg::matrix_multiply(*fdhat_, true, *k_fs_aa, false, false, false, true);
    k_fs_iamod->add(*k_fs_iaadd, false, -1.0, 1.0);
    k_fs_iamod->complete(k_fs_ia->domain_map(), k_fs_ia->row_map());
  }

  //---------------------------------------------------------- FOURTH LINE
  // nothing to do
  //----------------------------------------------------------- FIFTH LINE
  // kan: multiply tmatrix with invda and kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_anmod;
  if (aset)
    k_fs_anmod =
        Core::LinAlg::matrix_multiply(*ftanginvD_, false, *k_fs_an, false, false, false, true);

  // kam: multiply tmatrix with invda and kam
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_ammod;
  if (aset)
    k_fs_ammod =
        Core::LinAlg::matrix_multiply(*ftanginvD_, false, *k_fs_am, false, false, false, true);

  // kai: multiply tmatrix with invda and kai
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_aimod;
  if (aset && iset)
    k_fs_aimod =
        Core::LinAlg::matrix_multiply(*ftanginvD_, false, *k_fs_ai, false, false, false, true);

  // kaa: multiply tmatrix with invda and kaa
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_aamod;
  if (aset)
    k_fs_aamod =
        Core::LinAlg::matrix_multiply(*ftanginvD_, false, *k_fs_aa, false, false, false, true);

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
    FOUR_C_THROW("CHECK ME!");
    //        std::shared_ptr<Core::LinAlg::Vector<double>> tempvecm = Teuchos::rcp(new
    //        Core::LinAlg::Vector<double>(*gmdofrowmap_));
    //        std::shared_ptr<Core::LinAlg::Vector<double>> tempvecm2  = Teuchos::rcp(new
    //        Core::LinAlg::Vector<double>(mold_->DomainMap()));
    //        std::shared_ptr<Core::LinAlg::Vector<double>> zoldexp  = Teuchos::rcp(new
    //        Core::LinAlg::Vector<double>(mold_->RowMap())); if
    //        (mold_->RowMap().NumGlobalElements()) Core::LinAlg::export_to(*zold_,*zoldexp);
    //        mold_->Multiply(true,*zoldexp,*tempvecm2); if (mset)
    //        Core::LinAlg::export_to(*tempvecm2,*tempvecm); fm->Update(alphaf_,*tempvecm,1.0);
  }
  // if there is no self contact everything is ok
  else if (poromaster_)
  {
    Core::LinAlg::Vector<double> tempvecm(*fgmdofrowmap_);
    fmoldtransp_->multiply(false, *lambdaold_, tempvecm);
    fm->update(nopenalpha_, tempvecm, 1.0);
  }

  // fs: prepare alphaf * old contact forces (t_n)
  Core::LinAlg::Vector<double> fsadd(*fgsdofrowmap_);

  // for self contact, slave and master sets may have changed,
  // thus we have to export the product Dold^T * zold to fit
  if (is_self_contact())
  {
    FOUR_C_THROW("CHECK ME!");
    //        std::shared_ptr<Core::LinAlg::Vector<double>> tempvec  = Teuchos::rcp(new
    //        Core::LinAlg::Vector<double>(dold_->DomainMap()));
    //        std::shared_ptr<Core::LinAlg::Vector<double>> zoldexp = Teuchos::rcp(new
    //        Core::LinAlg::Vector<double>(dold_->RowMap())); if
    //        (dold_->RowMap().NumGlobalElements()) Core::LinAlg::export_to(*zold_,*zoldexp);
    //        dold_->Multiply(true,*zoldexp,*tempvec);
    //        if (sset) Core::LinAlg::export_to(*tempvec,*fsadd);
  }
  // if there is no self contact everything is ok
  else
  {
    fdoldtransp_->multiply(false, *lambdaold_, fsadd);
  }

  // fa: subtract alphaf * old contact forces (t_n)
  if (aset)
  {
    Core::LinAlg::Vector<double> faadd(*fgactivedofs_);
    Core::LinAlg::export_to(fsadd, faadd);

    fa->update(-nopenalpha_, faadd, 1.0);
  }

  // fm: add T(mhat)*fa
  std::shared_ptr<Core::LinAlg::Vector<double>> fmmod;
  if (mset)
  {
    fmmod = std::make_shared<Core::LinAlg::Vector<double>>(*fgmdofrowmap_);
    if (aset) fmhataam_->multiply(true, *fa, *fmmod);
    fmmod->update(1.0, *fm, 1.0);
  }

  //----------------------------------------------------------- THIRD LINE
  // fi: subtract alphaf * old contact forces (t_n)
  if (iset)
  {
    Core::LinAlg::Vector<double> fiadd(*fgidofs);
    Core::LinAlg::export_to(fsadd, fiadd);
    fi->update(-nopenalpha_, fiadd, 1.0);
  }

  // fi: add T(dhat)*fa
  Core::LinAlg::Vector<double> fimod(*fgidofs);
  if (aset) fdhat_->multiply(true, *fa, fimod);
  fimod.update(1.0, *fi, -1.0);

  //---------------------------------------------------------- FOURTH LINE
  // gactive: nothing to do
  //----------------------------------------------------------- FIFTH LINE
  // fa: multiply tmatrix with invda and fa
  std::shared_ptr<Core::LinAlg::Vector<double>> famod;
  if (aset)
  {
    famod = std::make_shared<Core::LinAlg::Vector<double>>(*fgactivet_);
    ftanginvD_->multiply(false, *fa, *famod);
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
    FOUR_C_THROW("CHECK ME!");
  }
  /**********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                  */
  /**********************************************************************/

  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs_effnew =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *falldofrowmap_, 81, true, false, k_fseff->get_matrixtype());
  std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
      Core::LinAlg::create_vector(*falldofrowmap_);

  //----------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  k_fs_effnew->add(*k_fs_nn, false, 1.0, 1.0);
  k_fs_effnew->add(*k_fs_nm, false, 1.0, 1.0);
  if (sset) k_fs_effnew->add(*k_fs_ns, false, 1.0, 1.0);

  //---------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  if (mset)
  {
    k_fs_effnew->add(*k_fs_mnmod, false, 1.0, 1.0);
    k_fs_effnew->add(*k_fs_mmmod, false, 1.0, 1.0);
    if (iset) k_fs_effnew->add(*k_fs_mimod, false, 1.0, 1.0);
    if (aset) k_fs_effnew->add(*k_fs_mamod, false, 1.0, 1.0);
  }

  //----------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) k_fs_effnew->add(k_fs_inmod, false, 1.0, 1.0);
  if (iset) k_fs_effnew->add(k_fs_immod, false, 1.0, 1.0);
  if (iset) k_fs_effnew->add(*k_fs_iimod, false, 1.0, 1.0);
  if (iset && aset) k_fs_effnew->add(*k_fs_iamod, false, 1.0, 1.0);

  //---------------------------------------------------------- FOURTH LINE
  // add a submatrices to kteffnew
  if (aset) k_fs_effnew->add(*fNCoup_lindisp_, false, 1.0, 1.0);

  //----------------------------------------------------------- FIFTH LINE
  // add a submatrices to kteffnew
  if (aset) k_fs_effnew->add(*k_fs_anmod, false, -1.0, 1.0);
  if (aset) k_fs_effnew->add(*k_fs_ammod, false, -1.0, 1.0);
  if (aset && iset) k_fs_effnew->add(*k_fs_aimod, false, -1.0, 1.0);
  if (aset) k_fs_effnew->add(*k_fs_aamod, false, -1.0, 1.0);
  if (aset) k_fs_effnew->add(*flinTangentiallambda_, false, 1.0, 1.0);

  // fill_complete kteffnew (square)
  k_fs_effnew->complete(*problem_dofs(), *falldofrowmap_);
  /**********************************************************************/
  /* (11) Global setup of feffnew (including contact)                   */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // add n subvector to feffnew
  Core::LinAlg::Vector<double> fnexp(*falldofrowmap_);

  Core::LinAlg::export_to(*fn, fnexp);

  feffnew->update(1.0, fnexp, 1.0);
  //---------------------------------------------------------- SECOND LINE
  // add m subvector to feffnew
  if (mset)
  {
    Core::LinAlg::Vector<double> fmmodexp(*falldofrowmap_);
    Core::LinAlg::export_to(*fmmod, fmmodexp);
    feffnew->update(1.0, fmmodexp, 1.0);
  }
  //----------------------------------------------------------- THIRD LINE
  // add i subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> fimodexp;
  if (iset)
  {
    fimodexp = std::make_shared<Core::LinAlg::Vector<double>>(*falldofrowmap_);
    Core::LinAlg::export_to(fimod, *fimodexp);
    feffnew->update(1.0, *fimodexp, 1.0);
  }

  //---------------------------------------------------------- FOURTH LINE
  // add weighted nCoup vector to feffnew, if existing
  std::shared_ptr<Core::LinAlg::Vector<double>> nCoupexp;
  if (aset)
  {
    nCoupexp = std::make_shared<Core::LinAlg::Vector<double>>(*falldofrowmap_);
    Core::LinAlg::export_to(*fNCoup_, *nCoupexp);
    feffnew->update(-1.0, *nCoupexp, 1.0);
  }

  //----------------------------------------------------------- FIFTH LINE
  // add a subvector to feffnew
  std::shared_ptr<Core::LinAlg::Vector<double>> famodexp;
  if (aset)
  {
    famodexp = std::make_shared<Core::LinAlg::Vector<double>>(*falldofrowmap_);
    Core::LinAlg::export_to(*famod, *famodexp);
    feffnew->update(-1.0, *famodexp, 1.0);
  }

  // finally do the replacement
  k_fseff = k_fs_effnew;
  feff = feffnew;
}  // CONTACT::LagrangeStrategyPoro::EvaluatePoroNoPen

/*----------------------------------------------------------------------*
 |  evaluate poro coupling contact matrices for no penetration          |
 |  condition on contact surface (public)                     ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::evaluate_other_mat_poro_no_pen(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& Feff, int Column_Block_Id)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!no_penetration_ ||
      (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()))
    return;
  // this method should be renamed as it handles twosided poro meshtying now aswell and is
  // able to handle fluid coupling for twosided contact

  nopenalpha_ = alphaf_;  // to use different alpha for nopen condition (not used at the moment)

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  Feff->complete();

  std::shared_ptr<Core::LinAlg::Map> domainmap =
      std::make_shared<Core::LinAlg::Map>(Feff->domain_map());

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************

  // double-check if this is a dual LM system
  if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Condensation only for dual LM");

  /**********************************************************************/
  /* (3) Split k_fseff and Feff into 3x3 matrix blocks                             */
  /**********************************************************************/

  // we want to split k into 3 groups s,m,n = 9 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> F_s, F_m, F_n;

  // temporarily we need the blocks ksmsm, ksmn, knsm
  // (FIXME: because a direct SplitMatrix3x3 is still missing!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> F_sm, F_sm0, F_n0, F_m0, F_s0;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap0;
  std::shared_ptr<Core::LinAlg::Map> tempmap1;
  std::shared_ptr<Core::LinAlg::Map> ftempmap;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;

  // split into slave/master part + structure part
  std::shared_ptr<Core::LinAlg::SparseMatrix> Feffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(Feff);
  if (parallel_redistribution_status())
  {
    FOUR_C_THROW("CHECK ME!");
  }
  else
  {
    // only split, no need to transform
    Core::LinAlg::split_matrix2x2(
        Feffmatrix, fgsmdofrowmap_, fgndofrowmap_, domainmap, tempmap0, F_sm, F_sm0, F_n, F_n0);
  }

  // further splits into slave part + master part
  Core::LinAlg::split_matrix2x2(
      F_sm, fgsdofrowmap_, fgmdofrowmap_, domainmap, tempmap0, F_s, F_s0, F_m, F_m0);

  // store some stuff for static condensation of LM
  cfx_s_.insert(std::pair<int, std::shared_ptr<Core::LinAlg::SparseMatrix>>(Column_Block_Id, F_s));

  /**********************************************************************/
  /* (5) Split slave quantities into active / inactive                  */
  /**********************************************************************/

  // we want to split kssmod into 2 groups a,i = 4 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> F_a, F_a0, F_i, F_i0;

  // we want to split ksn / ksm / kms into 2 groups a,i = 2 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> F_an, F_in, F_am, F_im, F_ma, F_mi;

  // we will get the i rowmap as a by-product
  std::shared_ptr<Core::LinAlg::Map> fgidofs;

  Core::LinAlg::split_matrix2x2(
      F_s, fgactivedofs_, fgidofs, domainmap, tempmap1, F_a, F_a0, F_i, F_i0);

  // abbreviations for active and inactive set
  int aset = fgactivedofs_->num_global_elements();
  int iset = fgidofs->num_global_elements();

  // abbreviations for slave  and master set
  // int sset = fgsdofrowmap_->NumGlobalElements(); // usually slave should anyway exist!
  int mset = fgmdofrowmap_->num_global_elements();

  /**********************************************************************/
  /* (7) Build the final K blocks                                       */
  /* where K stands for k_fs and F!!!                                   */
  /**********************************************************************/

  //----------------------------------------------------------- FIRST LINE
  // kn: nothing to do
  //---------------------------------------------------------- SECOND LINE --- Will just exist when
  // starting with two-sided poro contact!!!
  // km: add T(mhataam)*kan
  Core::LinAlg::SparseMatrix F_mmod(*fgmdofrowmap_, 100);
  F_mmod.add(*F_m, false, 1.0, 1.0);
  if (aset && mset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> F_madd =
        Core::LinAlg::matrix_multiply(*fmhataam_, true, *F_a, false, false, false, true);
    F_mmod.add(*F_madd, false, 1.0, 1.0);
  }
  F_mmod.complete(F_m->domain_map(), F_m->row_map());

  //----------------------------------------------------------- THIRD LINE
  //------------------- FOR 3D QUADRATIC CASE ----------------------------

  //--- For using non-diagonal D-Matrix, it should be checked if this assumption isn't anywhere
  // else!!!

  // kin: subtract T(dhat)*kan --
  Core::LinAlg::SparseMatrix F_imod(*fgidofs, 100);
  F_imod.add(*F_i, false, 1.0, 1.0);
  if (aset)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> F_iadd =
        Core::LinAlg::matrix_multiply(*fdhat_, true, *F_a, false, false, false, true);
    F_imod.add(*F_iadd, false, -1.0, 1.0);
  }
  F_imod.complete(F_i->domain_map(), F_i->row_map());

  //---------------------------------------------------------- FOURTH LINE
  // nothing to do
  //----------------------------------------------------------- FIFTH LINE
  // kan: multiply tmatrix with invda and kan
  std::shared_ptr<Core::LinAlg::SparseMatrix> F_amod;
  if (aset)
  {
    F_amod = Core::LinAlg::matrix_multiply(*ftanginvD_, false, *F_a, false, false, false, true);
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
    FOUR_C_THROW("CHECK ME!");
  }
  /**********************************************************************/
  /* (10) Global setup of kteffnew (including contact)                  */
  /**********************************************************************/

  std::shared_ptr<Core::LinAlg::SparseMatrix> F_effnew =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *falldofrowmap_, 108, true, false, Feff->get_matrixtype());

  //----------------------------------------------------------- FIRST LINE
  // add n submatrices to kteffnew
  F_effnew->add(*F_n, false, 1.0, 1.0);

  //---------------------------------------------------------- SECOND LINE
  // add m submatrices to kteffnew
  if (mset) F_effnew->add(F_mmod, false, 1.0, 1.0);
  //----------------------------------------------------------- THIRD LINE
  // add i submatrices to kteffnew
  if (iset) F_effnew->add(F_imod, false, 1.0, 1.0);
  //---------------------------------------------------------- FOURTH LINE
  // add a submatrices to kteffnew //assume that Column_Block_Id==0 is the porofluid coupling
  // block!!!
  if (Column_Block_Id == 0 && aset) F_effnew->add(*fNCoup_linvel_, false, 1.0, 1.0);
  //----------------------------------------------------------- FIFTH LINE
  // add a submatrices to kteffnew
  if (aset) F_effnew->add(*F_amod, false, -1.0, 1.0);

  // fill_complete kteffnew (square)
  F_effnew->complete(*domainmap, *falldofrowmap_);

  // finally do the replacement
  Feff = F_effnew;
}  // CONTACT::LagrangeStrategyPoro::evaluate_other_mat_poro_no_pen

/*----------------------------------------------------------------------------*
 | Poro Recovery method for no penetration LM (pure porous problem) ager 08/14|
 *---------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::recover_poro_no_pen(
    std::shared_ptr<Core::LinAlg::Vector<double>> disi,
    std::shared_ptr<Core::LinAlg::Vector<double>> inc)
{
  std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>> incm;
  incm.insert(std::pair<int, std::shared_ptr<Core::LinAlg::Vector<double>>>(0, inc));

  recover_poro_no_pen(*disi, incm);
}

/*----------------------------------------------------------------------*
 | Poro Recovery method for no penetration LM                 ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::recover_poro_no_pen(Core::LinAlg::Vector<double>& disi,
    std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>> inc)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!no_penetration_ ||
      (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()))
    return;

  // shape function and system types
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    // extract slave displacements from disi
    Core::LinAlg::Vector<double> disis(*gsdofrowmap_);
    if (gsdofrowmap_->num_global_elements()) Core::LinAlg::export_to(disi, disis);

    // extract master displacements from disi
    Core::LinAlg::Vector<double> disim(*gmdofrowmap_);
    if (gmdofrowmap_->num_global_elements()) Core::LinAlg::export_to(disi, disim);

    // extract other displacements from disi
    Core::LinAlg::Vector<double> disin(*gndofrowmap_);
    if (gndofrowmap_->num_global_elements()) Core::LinAlg::export_to(disi, disin);

    // condensation has been performed for active LM only,
    // thus we construct a modified invd matrix here which
    // only contains the active diagonal block
    // (this automatically renders the inactive LM to be zero)
    std::shared_ptr<Core::LinAlg::SparseMatrix> finvda;
    std::shared_ptr<Core::LinAlg::Map> tempmap1, tempmap2;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1, tempmtx2, tempmtx3;
    Core::LinAlg::split_matrix2x2(finvda_, fgactivedofs_, tempmap1, gactivedofs_, tempmap2, finvda,
        tempmtx1, tempmtx2, tempmtx3);
    Core::LinAlg::SparseMatrix finvdmod(*fgsdofrowmap_, 10);
    finvdmod.add(*finvda, false, 1.0, 1.0);
    finvdmod.complete(*gsdofrowmap_, *fgsdofrowmap_);

    /**********************************************************************/
    /* Update Lagrange multipliers lambda_n+1                                  */
    /**********************************************************************/
    {
      Core::LinAlg::Vector<double> flambda(*fgsdofrowmap_, true);

      Core::LinAlg::Vector<double> mod(*fgsdofrowmap_);

      cfssn_->multiply(false, disin, mod);
      flambda.update(-1.0, mod, 0.0);
      cfssm_->multiply(false, disim, mod);
      flambda.update(-1.0, mod, 1.0);
      cfsss_->multiply(false, disis, mod);
      flambda.update(-1.0, mod, 1.0);

      // loop over all offdiag blocks!!!
      std::map<int, std::shared_ptr<Core::LinAlg::SparseOperator>>::iterator matiter;
      std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>>::iterator inciter;
      for (matiter = cfx_s_.begin(); matiter != cfx_s_.end(); ++matiter)
      {
        inciter = inc.find(matiter->first);
        if (inciter == inc.end())
          FOUR_C_THROW(
              "CONTACT::LagrangeStrategyPoro::RecoverPoroNoPen: Couldn't find increment block {} "
              "for recovery of the lagrange multiplier!",
              matiter->first);

        matiter->second->multiply(false, *inciter->second, mod);
        flambda.update(-1.0, mod, 1.0);
      }

      flambda.update(1.0, *ffs_, 1.0);

      fdoldtransp_->multiply(false, *lambdaold_, mod);

      flambda.update(-nopenalpha_, mod, 1.0);

      Core::LinAlg::Vector<double> lambdacopy(flambda);

      finvdmod.multiply(true, lambdacopy, *lambda_);  // should be lambda_ at the end!!!

      lambda_->scale(
          (1 - alphaf_) / (1 - nopenalpha_));  //-- is already scaled by this factor by scaling
                                               // invda_!!! --- scale it back to with nopenalpha_...
    }
  }
  // store updated LM into nodes
  set_state(Mortar::state_lagrange_multiplier, *lambda_);
}

/*------------------------------------------------------------------------------*
 |  Additional update and output poro contact at end of time step     ager 08/14|
 *-----------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::update_poro_contact()
{
  if (no_penetration_)
  {
    // std::cout << "print lambda: " << *lambda_ << std::endl;
    lambdaold_->update(1.0, *lambda_, 0.0);
  }
}

/*------------------------------------------------------------------------*
 | Assign general poro contact state!                          ager 08/14|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::set_state(
    const enum Mortar::StateType& statetype, const Core::LinAlg::Vector<double>& vec)
{
  switch (statetype)
  {
    case Mortar::state_fvelocity:
    case Mortar::state_svelocity:
    case Mortar::state_lagrange_multiplier:
    case Mortar::state_fpressure:
    {
      // set state on interfaces
      for (int i = 0; i < (int)interface_.size(); ++i)
      {
        // interface_[i]->set_state(statename, vec);
        Core::FE::Discretization& idiscret_ = interface_[i]->discret();

        switch (statetype)
        {
          case Mortar::state_fvelocity:
          case Mortar::state_svelocity:
          {
            // alternative method to get vec to full overlap
            Core::LinAlg::Vector<double> global(*idiscret_.dof_col_map(), true);

            Core::LinAlg::export_to(vec, global);

            // loop over all nodes to set current velocity
            // (use fully overlapping column map)
            for (int i = 0; i < idiscret_.num_my_col_nodes(); ++i)
            {
              CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscret_.l_col_node(i));
              const int numdof = node->num_dof();
              std::vector<int> lm(numdof);

              for (int j = 0; j < numdof; ++j) lm[j] = node->dofs()[j];
              std::vector<double> myvel = Core::FE::extract_values(global, lm);

              // add myvel[2]=0 for 2D problems
              if (myvel.size() < 3) myvel.resize(3);
              // set current configuration
              for (int j = 0; j < 3; ++j)
              {
                if (statetype == Mortar::state_fvelocity)
                  node->poro_data().fvel()[j] = myvel[j];
                else
                  node->poro_data().svel()[j] = myvel[j];
              }
            }
            break;
          }
          case Mortar::state_lagrange_multiplier:
          {
            // alternative method to get vec to full overlap
            Core::LinAlg::Vector<double> global(*idiscret_.dof_col_map(), true);
            Core::LinAlg::export_to(vec, global);

            // loop over all nodes to set current velocity
            // (use fully overlapping column map)
            for (int i = 0; i < idiscret_.num_my_col_nodes(); ++i)
            {
              CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscret_.l_col_node(i));

              const int numdof = node->num_dof();
              std::vector<int> lm(numdof);

              for (int j = 0; j < numdof; ++j) lm[j] = node->dofs()[j];

              std::vector<double> mylm = Core::FE::extract_values(global, lm);

              // add myvel[2]=0 for 2D problems
              if (mylm.size() < 3) mylm.resize(3);
              // set current configuration
              if (node->is_slave())
              {
                for (int j = 0; j < 3; ++j)
                {
                  node->poro_data().poro_lm()[j] = mylm[j];
                }
              }
            }
            break;
          }
          case Mortar::state_fpressure:
          {
            // alternative method to get vec to full overlap
            Core::LinAlg::Vector<double> global(*idiscret_.dof_col_map(), true);
            Core::LinAlg::export_to(vec, global);

            // loop over all nodes to set current pressure
            // (use fully overlapping column map)
            for (int i = 0; i < idiscret_.num_my_col_nodes(); ++i)
            {
              CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscret_.l_col_node(i));

              double myfpres;
              int fpres;

              fpres = node->dofs()[0];  // here get ids of first component of node

              myfpres = global.get_values()[global.get_map().lid(fpres)];

              *node->poro_data().fpres() = myfpres;
            }
            break;
          }
          default:
          {
            FOUR_C_THROW("Shouldn't happen!");
            break;
          }
        }  // end inner switch statement
      }  // end loop over all interfaces
      break;
    }
    default:
    {
      CONTACT::AbstractStrategy::set_state(statetype, vec);
      break;
    }
  }  // end outer switch statement
}


// this should add the displacement of the parent element to the contact element into the
// datacontainer...
//...and it should additionally add the dof IDs to its Data container
/*------------------------------------------------------------------------*
 | Assign general poro contact state!                          ager 10/14|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::set_parent_state(const enum Mortar::StateType& statetype,
    const Core::LinAlg::Vector<double>& vec, const Core::FE::Discretization& dis)
{
  if (statetype == Mortar::StateType::state_new_displacement)
  {
    Core::LinAlg::Vector<double> global(*dis.dof_col_map(), true);
    Core::LinAlg::export_to(vec, global);

    // set state on interfaces
    for (int i = 0; i < (int)interface_.size(); ++i)
    {
      Core::FE::Discretization& idiscret_ = interface_[i]->discret();

      if (poroslave_)
      {
        for (int j = 0; j < interface_[i]->slave_col_elements()->num_my_elements();
            ++j)  // will just work for onesided poro contact as the porosity is just on slave
                  // side!!!
        {
          int gid = interface_[i]->slave_col_elements()->gid(j);

          Mortar::Element* ele = dynamic_cast<Mortar::Element*>(idiscret_.g_element(gid));

          std::vector<int> lm;
          std::vector<int> lmowner;
          std::vector<int> lmstride;

          // this gets values in local order
          ele->parent_element()->location_vector(dis, lm, lmowner, lmstride);

          std::vector<double> myval = Core::FE::extract_values(global, lm);

          ele->mo_data().parent_disp() = myval;
          ele->mo_data().parent_dof() = lm;
        }
      }
      if (poromaster_)  // add master parent element displacements
      {
        for (int j = 0; j < interface_[i]->master_col_elements()->num_my_elements(); ++j)
        {
          int gid = interface_[i]->master_col_elements()->gid(j);

          Mortar::Element* mele = dynamic_cast<Mortar::Element*>(idiscret_.g_element(gid));

          std::vector<int> lm;
          std::vector<int> lmowner;
          std::vector<int> lmstride;

          // this gets values in local order
          mele->parent_element()->location_vector(dis, lm, lmowner, lmstride);

          std::vector<double> myval = Core::FE::extract_values(global, lm);

          mele->mo_data().parent_disp() = myval;
          mele->mo_data().parent_dof() = lm;
        }
      }
    }
  }
}

/*------------------------------------------------------------------------*
 | Initialize poro meshtying matrices with corresponding maps  h.Willmann |
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::poro_mt_initialize()
{
  // (re)setup global lin of D & M * lambda - Matrix
  porolindmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  porolinmmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // (re)setup global Mortar Core::LinAlg::SparseMatrices and Core::LinAlg::Vectors
  dmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 10);
  mmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);

  // this is needed because on the mesh tying way to build this strategy it is not done elsewhere
  setup_no_penetration_condition();

  isincontact_ = true;      // simply set true for meshtying
  wasincontact_ = true;     // as meshtying interfaces stay the same and are fully in contact
  wasincontactlts_ = true;  // this is necessary for other methods in this strategy
}  // CONTACT::LagrangeStrategyPoro::PoroLinkDM

/*------------------------------------------------------------------------*
 | assemble porofluid meshtying matrices                       h.Willmann |
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::poro_mt_prepare_fluid_coupling()
{
  // reset D and M matrices for new Newton iteration

  dmatrix_->zero();
  mmatrix_->zero();

  // Assemble D and M matrices
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->assemble_dm(*dmatrix_, *mmatrix_);
  }

  // complete D and M matrices
  dmatrix_->complete();
  mmatrix_->complete(*gmdofrowmap_, *gsdofrowmap_);

  // as mhataam-, dhat_ and invda_ are not computed in poro - meshtying before this point it is
  // necessary here
  poro_mt_set_coupling_matrices();
}  // CONTACT::LagrangeStrategyPoro::PoroAssembleFluidCoupling()

/*------------------------------------------------------------------------*
 | assemble porofluid meshtying matrices                       h.Willmann |
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::poro_mt_set_coupling_matrices()
{
  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;

  std::shared_ptr<Core::LinAlg::Map> gidofs;

  int aset = gactivedofs_->num_global_elements();

  // invert dmatrix to invd
  invd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);

  // extract diagonal of invd into diag
  invd_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  //  for (int i = 0; i < diag->MyLength(); ++i)
  //    if ((*diag)[i] == 0.0)
  //      (*diag)[i] = 1.0;

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  invd_->replace_diagonal_values(*diag);
  // inversion end

  // active part of invd
  std::shared_ptr<Core::LinAlg::SparseMatrix> invda;
  Core::LinAlg::split_matrix2x2(
      invd_, gactivedofs_, gidofs, gactivedofs_, gidofs, invda, tempmtx1, tempmtx2, tempmtx3);

  // coupling part of dmatrix (only nonzero for 3D quadratic case!)(not considered for poro
  // mt/contact yet-> ==0)
  std::shared_ptr<Core::LinAlg::SparseMatrix> dai;
  Core::LinAlg::split_matrix2x2(
      dmatrix_, gactivedofs_, gidofs, gactivedofs_, gidofs, tempmtx1, dai, tempmtx2, tempmtx3);

  int iset = gidofs->num_global_elements();
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

  // scaling of invd and dai
  invda->scale(1 / (1 - alphaf_));

  save_coupling_matrices(dhat, mhataam, invda);
}  // CONTACT::LagrangeStrategyPoro::poro_mt_prepare_fluid_coupling

/*------------------------------------------------------------------------*
 | update old meshtying matrices and LMP                       h.Willmann |
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyPoro::poro_mt_update()
{
  // set dold_ mold_ and lambdaold_ after every time step
  dold_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  mold_ = std::make_shared<Core::LinAlg::SparseMatrix>(*mmatrix_);
  dold_->complete(dmatrix_->domain_map(), dmatrix_->range_map());
  mold_->complete(mmatrix_->domain_map(), mmatrix_->range_map());
  update_poro_contact();
}  // CONTACT::LagrangeStrategyPoro::PoroMtUpdate()

FOUR_C_NAMESPACE_CLOSE
