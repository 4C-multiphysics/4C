/*-----------------------------------------------------------------------*/
/*! \file
\brief Meshtying of porous media using Lagrange multipliers

// Masterthesis of h.Willmann under supervision of Anh-Tu Vuong and Matthias Mayr
// Originates from contact_poro_lagrange_strategy


\level 3
*/
/*-----------------------------------------------------------------------*/

#include "4C_contact_meshtying_poro_lagrange_strategy.hpp"

#include "4C_inpar_contact.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Epetra_SerialComm.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                      h.Willmann    2015|
 *----------------------------------------------------------------------*/
CONTACT::PoroMtLagrangeStrategy::PoroMtLagrangeStrategy(const Epetra_Map* dof_row_map,
    const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<Mortar::Interface>> interface, int dim, Teuchos::RCP<Epetra_Comm> comm,
    double alphaf, int maxdof)
    : MtLagrangeStrategy(dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof)
{
  return;
}


/*----------------------------------------------------------------------*
 | Poro Meshtying initialization calculations         h.Willmann    2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::initialize_poro_mt(
    Teuchos::RCP<Core::LinAlg::SparseMatrix>& kteffoffdiag)

{
  Teuchos::RCP<Core::LinAlg::SparseMatrix> kteffmatrix =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(kteffoffdiag);

  fvelrow_ = Teuchos::rcp(new Epetra_Map(kteffmatrix->OperatorDomainMap()));

  return;
}


/*----------------------------------------------------------------------*
 | Poro Meshtying method regarding coupling terms      h.Willmann   2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::evaluate_meshtying_poro_off_diag(
    Teuchos::RCP<Core::LinAlg::SparseMatrix>& kteffoffdiag)
{
  // system type
  auto systype = Teuchos::getIntegralValue<Inpar::CONTACT::SystemType>(params(), "SYSTEM");

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == Inpar::CONTACT::system_condensed ||
      systype == Inpar::CONTACT::system_condensed_lagmult)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Condensation only for dual LM");

    // h.Willmann actual method

    // complete stiffness matrix
    // (this is a prerequisite for the Split2x2 methods to be called later)
    kteffoffdiag->complete();

    /**********************************************************************/
    /* Split kteffoffdiag into 3 block matrix rows                        */
    /**********************************************************************/
    // we want to split k into 3 rows n, m and s
    Teuchos::RCP<Core::LinAlg::SparseMatrix> cn, cm, cs;

    // temporarily we need the block row csm
    // (FIXME: because a direct SplitMatrix3x1 is missing here!)
    Teuchos::RCP<Core::LinAlg::SparseMatrix> csm;


    // some temporary Teuchos::RCPs
    Teuchos::RCP<Epetra_Map> tempmap1;
    Teuchos::RCP<Epetra_Map> tempmap2;
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx1;
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx2;
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx3;
    Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx4;

    Teuchos::RCP<Core::LinAlg::SparseMatrix> kteffmatrix =
        Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(kteffoffdiag);

    //    std::cout<< " kteffmatrix " << std::endl;
    //    kteffmatrix->DomainMap().print(std::cout);

    if (par_redist())  // asdf
    {
      FOUR_C_THROW(
          "no parallel redistribution of poro meshtying implemented - feel free to implement");
    }

    // first split into slave/master block row + remaining part
    Core::LinAlg::split_matrix2x2(
        kteffmatrix, gsmdofrowmap_, gndofrowmap_, fvelrow_, tempmap1, csm, tempmtx1, cn, tempmtx2);

    //    std::cout<< " tempmap1 " << std::endl;
    //    tempmap1->print(std::cout);

    // second split slave/master block row
    Core::LinAlg::split_matrix2x2(
        csm, gsdofrowmap_, gmdofrowmap_, fvelrow_, tempmap2, cs, tempmtx3, cm, tempmtx4);

    // store some stuff for the recovery of the lagrange multiplier
    cs_ = cs;


    /**********************************************************************/
    /* Build the final matrix block row                                   */
    /**********************************************************************/
    // cn: nothing to do

    // cm: add T(mbar)*cs
    Teuchos::RCP<Core::LinAlg::SparseMatrix> cmmod =
        Teuchos::rcp(new Core::LinAlg::SparseMatrix(*gmdofrowmap_, 100));
    cmmod->add(*cm, false, 1.0, 1.0);
    Teuchos::RCP<Core::LinAlg::SparseMatrix> cmadd =
        Core::LinAlg::matrix_multiply(*get_m_hat(), true, *cs, false, false, false, true);
    cmmod->add(*cmadd, false, 1.0, 1.0);
    cmmod->complete(cm->domain_map(), cm->row_map());

    // cs: nothing to do as it remains zero

    /**********************************************************************/
    /* Global setup of kteffoffdiagnew,  (including meshtying)            */
    /**********************************************************************/
    Teuchos::RCP<Core::LinAlg::SparseMatrix> kteffoffdiagnew =
        Teuchos::rcp(new Core::LinAlg::SparseMatrix(
            *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype()));

    // add n matrix row
    kteffoffdiagnew->add(*cn, false, 1.0, 1.0);

    // add m matrix row
    kteffoffdiagnew->add(*cmmod, false, 1.0, 1.0);

    // s matrix row remains zero (thats what it was all about)

    kteffoffdiagnew->complete(kteffmatrix->domain_map(), kteffmatrix->range_map());

    kteffoffdiag = kteffoffdiagnew;
  }
  else
  {
    FOUR_C_THROW("Trying to use not condensed PoroMeshtying --- Feel Free to implement!");
  }
  return;
}


/*----------------------------------------------------------------------*
 | Poro Recovery method for structural displacement LM  h.Willmann  2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::recover_coupling_matrix_partof_lmp(
    Teuchos::RCP<Epetra_Vector> veli)
{
  Teuchos::RCP<Epetra_Vector> zfluid = Teuchos::rcp(new Epetra_Vector(z_->Map(), true));

  Teuchos::RCP<Epetra_Vector> mod = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));

  cs_->multiply(false, *veli, *mod);
  zfluid->Update(-1.0, *mod, 1.0);
  Teuchos::RCP<Epetra_Vector> zcopy = Teuchos::rcp(new Epetra_Vector(*zfluid));
  get_d_inverse()->multiply(true, *zcopy, *zfluid);
  zfluid->Scale(1 / (1 - alphaf_));

  z_->Update(1.0, *zfluid, 1.0);  // Add FluidCoupling Contribution to LM!
  return;
}

FOUR_C_NAMESPACE_CLOSE
