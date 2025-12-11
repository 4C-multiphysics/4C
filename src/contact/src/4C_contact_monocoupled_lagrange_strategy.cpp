// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_monocoupled_lagrange_strategy.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                              ager 02/15|
 *----------------------------------------------------------------------*/
CONTACT::MonoCoupledLagrangeStrategy::MonoCoupledLagrangeStrategy(
    const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
    const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
    Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
    int dim, MPI_Comm comm, double alphaf, int maxdof)
    : LagrangeStrategy(
          data_ptr, dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof),
      has_to_evaluate_(false),
      has_to_recover_(false)
{
  // do some security checks ...
  return;
}

/*----------------------------------------------------------------------*
 | structural contact global evaluation method called from time         |
 | integrator + condensation of offdiagonal Matrixes (public) ager 02/15|
 *----------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::apply_force_stiff_cmt_coupled(
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::SparseOperator>& k_ss,
    std::map<int, std::shared_ptr<Core::LinAlg::SparseOperator>*> k_sx,
    std::shared_ptr<Core::LinAlg::Vector<double>>& rhs_s, const int step, const int iter,
    bool predictor)
{
  // call the main routine for contact!!!
  CONTACT::AbstractStrategy::apply_force_stiff_cmt(dis, k_ss, rhs_s, step, iter, predictor);

  // Take care of the alternative condensation of the off-diagonal blocks!!!
  std::map<int, std::shared_ptr<Core::LinAlg::SparseOperator>*>::iterator matiter;
  for (matiter = k_sx.begin(); matiter != k_sx.end(); ++matiter)
  {
    evaluate_off_diag_contact(*(matiter->second), matiter->first);
  }
  has_to_evaluate_ = false;
  return;
}

/*-----------------------------------------------------------------------------*
 | structural contact global evaluation method called from time                |
 | integrator + condensation of one!!! offdiagonal Matrixes (public) ager 02/15|
 *----------------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::apply_force_stiff_cmt_coupled(
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::SparseOperator>& k_ss,
    std::shared_ptr<Core::LinAlg::SparseOperator>& k_sx,
    std::shared_ptr<Core::LinAlg::Vector<double>>& rhs_s, const int step, const int iter,
    bool predictor)
{
  // call the main routine for contact!!!
  CONTACT::AbstractStrategy::apply_force_stiff_cmt(dis, k_ss, rhs_s, step, iter, predictor);

  // Take care of the alternative condensation of the off-diagonal blocks!!!
  evaluate_off_diag_contact(k_sx, 0);

  has_to_evaluate_ = false;
  return;
}

/*------------------------------------------------------------------------*
 |  condense off-diagonal blocks                      (public)  ager 02/15|
 *-----------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::evaluate_off_diag_contact(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff, int Column_Block_Id)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  // complete stiffness matrix
  // (this is a prerequisite for the Split2x2 methods to be called later)
  kteff->complete();

  std::shared_ptr<Core::LinAlg::Map> domainmap =
      std::make_shared<Core::LinAlg::Map>(kteff->domain_map());

  // system type
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == CONTACT::SystemType::condensed)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    /**********************************************************************/
    /* (3) Split kteff into 3x3 matrix blocks                             */  // just split the rows
                                                                              // !!!
    /**********************************************************************/

    // we want to split k into 3 groups s,m,n = 9 blocks
    std::shared_ptr<Core::LinAlg::SparseMatrix> ks, km, kn;

    // temporarily we need the blocks ksmsm, ksmn, knsm
    // (FIXME: because a direct SplitMatrix3x3 is still missing!)
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksm, ksm0, kn0, km0, ks0;

    // some temporary std::shared_ptrs
    std::shared_ptr<Core::LinAlg::Map> tempmap0;
    std::shared_ptr<Core::LinAlg::Map> tempmap1;
    std::shared_ptr<Core::LinAlg::Map> ftempmap;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;

    // split into slave/master part + structure part
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);

    if (parallel_redistribution_status())  // TODO Check if how to modify
    {
      FOUR_C_THROW("parallel_redistribution_status(): CHECK ME!");
    }
    else
    {
      // only split, no need to transform
      Core::LinAlg::split_matrix2x2(
          kteffmatrix, gsmdofrowmap_, gndofrowmap_, domainmap, tempmap0, ksm, ksm0, kn, kn0);
    }

    // further splits into slave part + master part
    Core::LinAlg::split_matrix2x2(
        ksm, gsdofrowmap_, gmdofrowmap_, domainmap, tempmap0, ks, ks0, km, km0);

    // store some stuff for static condensation of LM
    csx_s_.insert(std::pair<int, std::shared_ptr<Core::LinAlg::SparseMatrix>>(Column_Block_Id, ks));


    /**********************************************************************/
    /* (5) Split slave quantities into active / inactive                  */
    /**********************************************************************/

    // we want to split kssmod into 2 groups a,i = 4 blocks
    std::shared_ptr<Core::LinAlg::SparseMatrix> ka, ka0, ki, ki0;

    // we want to split ksn / ksm / kms into 2 groups a,i = 2 blocks
    std::shared_ptr<Core::LinAlg::SparseMatrix> kan, kin, kam, kim, kma, kmi;

    // we will get the i rowmap as a by-product
    std::shared_ptr<Core::LinAlg::Map> gidofs;
    std::shared_ptr<Core::LinAlg::Map> fgidofs;

    // do the splitting
    Core::LinAlg::split_matrix2x2(ks, gactivedofs_, gidofs, domainmap, tempmap1, ka, ka0, ki, ki0);

    // abbreviations for master, active and inactive set
    int aset = gactivedofs_->num_global_elements();
    int iset = gidofs->num_global_elements();

    /**********************************************************************/
    /* (7) Build the final K blocks                                       */
    /**********************************************************************/

    //----------------------------------------------------------- FIRST LINE
    // kn: nothing to do

    //---------------------------------------------------------- SECOND LINE
    // km: add T(mhataam)*kan
    Core::LinAlg::SparseMatrix kmmod(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*km, false, 1.0, kmmod, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmadd =
          Core::LinAlg::matrix_multiply(*mhataam_, true, *ka, false, false, false, true);
      Core::LinAlg::matrix_add(*kmadd, false, 1.0, kmmod, 1.0);
    }
    kmmod.complete(Core::LinAlg::Map(kteff->domain_map()), km->row_map());

    //----------------------------------------------------------- THIRD LINE
    //------------------- FOR 3D QUADRATIC CASE ----------------------------

    //--- For using non diagonal D-Matrix, it should be checked if this assumption isn't anywhere
    // else!!!

    // kin: subtract T(dhat)*kan --
    Core::LinAlg::SparseMatrix kimod(*gidofs, 100);
    Core::LinAlg::matrix_add(*ki, false, 1.0, kimod, 1.0);
    if (aset)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> kiadd =
          Core::LinAlg::matrix_multiply(*dhat_, true, *ka, false, false, false, true);
      Core::LinAlg::matrix_add(*kiadd, false, -1.0, kimod, 1.0);
    }
    kimod.complete(Core::LinAlg::Map(kteff->domain_map()), ki->row_map());

    //---------------------------------------------------------- FOURTH LINE
    // nothing to do

    //----------------------------------------------------------- FIFTH LINE
    // ka: multiply tmatrix with invda and ka
    std::shared_ptr<Core::LinAlg::SparseMatrix> kamod;
    if (aset)
    {
      kamod = Core::LinAlg::matrix_multiply(*tmatrix_, false, *invda_, true, false, false, true);
      kamod = Core::LinAlg::matrix_multiply(*kamod, false, *ka, false, false, false, true);
    }

    /********************************************************************/
    /* (9) Transform the final K blocks                                 */
    /********************************************************************/
    // The row maps of all individual matrix blocks are transformed to
    // the parallel layout of the underlying problem discretization.
    // Of course, this is only necessary in the parallel redistribution
    // case, where the contact interfaces have been redistributed
    // independently of the underlying problem discretization.

    if (parallel_redistribution_status())  // check what to do
    {
      FOUR_C_THROW("not checked so far!!!");
    }

    /**********************************************************************/
    /* (10) Global setup of kteffnew (including contact)                  */
    /**********************************************************************/

    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *gdisprowmap_, 81, true, false, kteffmatrix->get_matrixtype());

    //----------------------------------------------------------- FIRST LINE
    // add n submatrices to kteffnew
    Core::LinAlg::matrix_add(*kn, false, 1.0, *kteffnew, 1.0);
    //---------------------------------------------------------- SECOND LINE
    // add m submatrices to kteffnew
    Core::LinAlg::matrix_add(kmmod, false, 1.0, *kteffnew, 1.0);
    //----------------------------------------------------------- THIRD LINE
    // add i submatrices to kteffnew
    if (iset) Core::LinAlg::matrix_add(kimod, false, 1.0, *kteffnew, 1.0);

    //---------------------------------------------------------- FOURTH LINE
    // for off diag blocks this line is empty (weighted normal = f(disp))

    //----------------------------------------------------------- FIFTH LINE
    // add a submatrices to kteffnew
    if (aset) Core::LinAlg::matrix_add(*kamod, false, 1.0, *kteffnew, 1.0);

    // fill_complete kteffnew (square)
    kteffnew->complete(*domainmap, *gdisprowmap_);

    // finally do the replacement
    kteff = kteffnew;
  }
  else
  {
    FOUR_C_THROW("Trying to use not condensed form --- Feel Free to implement!");
  }
  return;
}

/*------------------------------------------------------------------------*
 | Coupled Recovery method for contact LM                       ager 02/15|
 *-----------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::recover_coupled(
    std::shared_ptr<Core::LinAlg::Vector<double>> disi,
    std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>> inc)
{
  // check if contact contributions are present,
  // if not we can skip this routine to speed things up
  if (!is_in_contact() && !was_in_contact() && !was_in_contact_last_time_step()) return;

  LagrangeStrategy::recover(
      disi);  // Update Structural Part! --> Here just Part from Coupling Matrix will be added!

  if (inc.size() == 0 && csx_s_.size() == 0)
    return;  // done already here if there are no off-diag blocks

  // shape function and system types
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == CONTACT::SystemType::condensed)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    if (inc.size() != csx_s_.size())
      FOUR_C_THROW(
          "CONTACT::MonoCoupledLagrangeStrategy::RecoverCoupled: For Recovery the same number of "
          "off-diagonal increment blocks is required! {} != {} !",
          inc.size(), csx_s_.size());

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

    std::map<int, std::shared_ptr<Core::LinAlg::SparseOperator>>::iterator matiter;
    std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>>::iterator inciter;

    // loop over all offdiag blocks!!!
    for (matiter = csx_s_.begin(); matiter != csx_s_.end(); ++matiter)
    {
      inciter = inc.find(matiter->first);
      if (inciter == inc.end())
        FOUR_C_THROW(
            "CONTACT::MonoCoupledLagrangeStrategy::RecoverCoupled: Couldn't find increment block "
            "{} for recovery of the lagrange multiplier!",
            matiter->first);

      /**********************************************************************/
      /* Update Lagrange multipliers z_n+1                                  */
      /**********************************************************************/
      // for self contact, slave and master sets may have changed,
      // thus we have to export the products Dold * zold and Mold^T * zold to fit
      if (is_self_contact())  // is not considered yet!
      {
        FOUR_C_THROW(
            "Trying to make coupled selfcontact condensation... Check if this makes any sense!!!");
        // approximate update
        // z_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*gsdofrowmap_));
        // invdmod->Multiply(false,*fs_,*z_);

        // full update
        //      z_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*gsdofrowmap_));
        //      z_->Update(1.0,*fs_,0.0);
        //      std::shared_ptr<Core::LinAlg::Vector<double>> mod = Teuchos::rcp(new
        //      Core::LinAlg::Vector<double>(*gsdofrowmap_)); kss_->Multiply(false,*disis,*mod);
        //      z_->Update(-1.0,*mod,1.0);
        //      ksm_->Multiply(false,*disim,*mod);
        //      z_->Update(-1.0,*mod,1.0);
        //      ksn_->Multiply(false,*disin,*mod);
        //      z_->Update(-1.0,*mod,1.0);
        //      std::shared_ptr<Core::LinAlg::Vector<double>> mod2 = Teuchos::rcp(new
        //      Core::LinAlg::Vector<double>((dold_->RowMap()))); if
        //      (dold_->RowMap().NumGlobalElements()) Core::LinAlg::export_to(*zold_,*mod2);
        //      std::shared_ptr<Core::LinAlg::Vector<double>> mod3 = Teuchos::rcp(new
        //      Core::LinAlg::Vector<double>((dold_->RowMap()))); dold_->Multiply(true,*mod2,*mod3);
        //      std::shared_ptr<Core::LinAlg::Vector<double>> mod4 = Teuchos::rcp(new
        //      Core::LinAlg::Vector<double>(*gsdofrowmap_)); if (gsdofrowmap_->NumGlobalElements())
        //      Core::LinAlg::export_to(*mod3,*mod4); z_->Update(-alphaf_,*mod4,1.0);
        //      std::shared_ptr<Core::LinAlg::Vector<double>> zcopy = Teuchos::rcp(new
        //      Core::LinAlg::Vector<double>(*z_)); invdmod->Multiply(true,*zcopy,*z_);
        //      z_->Scale(1/(1-alphaf_));
      }
      else
      {
        Core::LinAlg::Vector<double> zfluid(z_->get_map(), true);

        Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
        matiter->second->multiply(false, *inciter->second, mod);
        zfluid.update(-1.0, mod, 0.0);
        Core::LinAlg::Vector<double> zcopy(zfluid);
        invdmod.multiply(true, zcopy, zfluid);
        zfluid.scale(1 / (1 - alphaf_));

        z_->update(1.0, zfluid, 1.0);  // Add Offdiag  -  Coupling Contribution to LM!!!
      }
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
  }

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate);  // Here done twice: already in structural
                                                           // contact --> not wanted

  has_to_recover_ = false;
  return;
}


/*-------------------------------------------------------------------------*
 | Coupled Recovery method for contact LM with one offdiag block ager 02/15|
 *------------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::recover_coupled(
    std::shared_ptr<Core::LinAlg::Vector<double>> disi,
    std::shared_ptr<Core::LinAlg::Vector<double>> inc)
{
  std::map<int, std::shared_ptr<Core::LinAlg::Vector<double>>> incm;
  incm.insert(std::pair<int, std::shared_ptr<Core::LinAlg::Vector<double>>>(0, inc));

  recover_coupled(disi, incm);
  return;
}

/*---------------------------------------------------------------------------*
 | Save mortar coupling matrices for evaluation of off diag terms! ager 08/14|
 *--------------------------------------------------------------------------*/
void CONTACT::MonoCoupledLagrangeStrategy::save_coupling_matrices(
    std::shared_ptr<Core::LinAlg::SparseMatrix> dhat,
    std::shared_ptr<Core::LinAlg::SparseMatrix> mhataam,
    std::shared_ptr<Core::LinAlg::SparseMatrix> invda)
{
  dhat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dhat);
  mhataam_ = std::make_shared<Core::LinAlg::SparseMatrix>(*mhataam);
  invda_ = std::make_shared<Core::LinAlg::SparseMatrix>(*invda);
}

FOUR_C_NAMESPACE_CLOSE
