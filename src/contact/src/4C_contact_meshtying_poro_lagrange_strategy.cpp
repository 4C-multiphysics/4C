// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_meshtying_poro_lagrange_strategy.hpp"

#include "4C_contact_input.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                      h.Willmann    2015|
 *----------------------------------------------------------------------*/
CONTACT::PoroMtLagrangeStrategy::PoroMtLagrangeStrategy(const Core::LinAlg::Map* dof_row_map,
    const Core::LinAlg::Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<std::shared_ptr<Mortar::Interface>> interface, int dim, MPI_Comm comm,
    double alphaf, int maxdof)
    : MtLagrangeStrategy(dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof)
{
}


/*----------------------------------------------------------------------*
 | Poro Meshtying initialization calculations         h.Willmann    2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::initialize_poro_mt(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& kteffoffdiag)

{
  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteffoffdiag);

  fvelrow_ = std::make_shared<Core::LinAlg::Map>(kteffmatrix->OperatorDomainMap());
}


/*----------------------------------------------------------------------*
 | Poro Meshtying method regarding coupling terms      h.Willmann   2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::evaluate_meshtying_poro_off_diag(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& kteffoffdiag)
{
  // system type
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  // shape function
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == CONTACT::SystemType::condensed ||
      systype == CONTACT::SystemType::condensed_lagmult)
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
    std::shared_ptr<Core::LinAlg::SparseMatrix> cn, cm, cs;

    // temporarily we need the block row csm
    // (FIXME: because a direct SplitMatrix3x1 is missing here!)
    std::shared_ptr<Core::LinAlg::SparseMatrix> csm;


    // some temporary std::shared_ptrs
    std::shared_ptr<Core::LinAlg::Map> tempmap1;
    std::shared_ptr<Core::LinAlg::Map> tempmap2;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;
    std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx4;

    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteffoffdiag);

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
    Core::LinAlg::SparseMatrix cmmod(*gmdofrowmap_, 100);
    Core::LinAlg::matrix_add(*cm, false, 1.0, cmmod, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> cmadd =
        Core::LinAlg::matrix_multiply(*get_m_hat(), true, *cs, false, false, false, true);
    Core::LinAlg::matrix_add(*cmadd, false, 1.0, cmmod, 1.0);
    cmmod.complete(cm->domain_map(), cm->row_map());

    // cs: nothing to do as it remains zero

    /**********************************************************************/
    /* Global setup of kteffoffdiagnew,  (including meshtying)            */
    /**********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffoffdiagnew =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());

    // add n matrix row
    Core::LinAlg::matrix_add(*cn, false, 1.0, *kteffoffdiagnew, 1.0);

    // add m matrix row
    Core::LinAlg::matrix_add(cmmod, false, 1.0, *kteffoffdiagnew, 1.0);

    // s matrix row remains zero (thats what it was all about)

    kteffoffdiagnew->complete(kteffmatrix->domain_map(), kteffmatrix->range_map());

    kteffoffdiag = kteffoffdiagnew;
  }
  else
  {
    FOUR_C_THROW("Trying to use not condensed PoroMeshtying --- Feel Free to implement!");
  }
}


/*----------------------------------------------------------------------*
 | Poro Recovery method for structural displacement LM  h.Willmann  2015|
 *----------------------------------------------------------------------*/
void CONTACT::PoroMtLagrangeStrategy::recover_coupling_matrix_partof_lmp(
    Core::LinAlg::Vector<double>& veli)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> zfluid =
      std::make_shared<Core::LinAlg::Vector<double>>(z_->get_map(), true);

  Core::LinAlg::Vector<double> mod(*gsdofrowmap_);

  cs_->multiply(false, veli, mod);
  zfluid->update(-1.0, mod, 1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> zcopy =
      std::make_shared<Core::LinAlg::Vector<double>>(*zfluid);
  get_d_inverse()->multiply(true, *zcopy, *zfluid);
  zfluid->scale(1 / (1 - alphaf_));

  z_->update(1.0, *zfluid, 1.0);  // Add FluidCoupling Contribution to LM!
}

FOUR_C_NAMESPACE_CLOSE
