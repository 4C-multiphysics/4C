// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_meshtying_lagrange_strategy.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_input.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                              popp 05/09|
 *----------------------------------------------------------------------*/
CONTACT::MtLagrangeStrategy::MtLagrangeStrategy(const Core::LinAlg::Map* dof_row_map,
    const Core::LinAlg::Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<std::shared_ptr<Mortar::Interface>> interface, const int spatialDim,
    const MPI_Comm& comm, const double alphaf, const int maxdof)
    : MtAbstractStrategy(
          dof_row_map, NodeRowMap, params, interface, spatialDim, comm, alphaf, maxdof)
{
  // empty constructor body
}

/*----------------------------------------------------------------------*
 |  do mortar coupling in reference configuration             popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::mortar_coupling(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis)
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtLagrangeStrategy::mortar_coupling");

  // print message
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Performing mortar coupling...............";
    fflush(stdout);
  }

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  // refer call to parent class
  MtAbstractStrategy::mortar_coupling(dis);

  //----------------------------------------------------------------------
  // Multiply Mortar matrices: m^ = inv(d) * m
  //----------------------------------------------------------------------
  invd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  invd_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
    if ((*diag)[i] == 0.0) (*diag).get_values()[i] = 1.0;

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  err = invd_->replace_diagonal_values(*diag);
  if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

  // do the multiplication M^ = inv(D) * M
  mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd_, false, *mmatrix_, false, false, false, true);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  // D^(-1)    ---->   T * D^(-1)
  // \hat{M}   ---->   T * \hat{M}
  // These modifications are applied once right here, thus the
  // following code (evaluate_meshtying, recover) remains unchanged.
  //----------------------------------------------------------------------
  if (dualquadslavetrafo())
  {
    // type of LM interpolation for quadratic elements
    auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

    if (lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // do nothing
    }
    else
    {
      // modify dmatrix_, invd_ and mhatmatrix_
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *invd_, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp3 =
          Core::LinAlg::matrix_multiply(*trafo_, false, *mhatmatrix_, false, false, false, true);
      dmatrix_ = temp1;
      invd_ = temp2;
      mhatmatrix_ = temp3;
    }
  }

  //----------------------------------------------------------------------
  // Build constraint matrix (containing D and M)
  //----------------------------------------------------------------------
  // case 1: saddle point system
  //    -> constraint matrix with rowmap=Problemmap, colmap=LMmap
  // case 2: condensed system
  //    -> no explicit constraint matrix needed
  //----------------------------------------------------------------------
  bool setup = true;
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");
  if (systype == CONTACT::SystemType::condensed ||
      systype == CONTACT::SystemType::condensed_lagmult)
    setup = false;

  // build constraint matrix only if necessary
  if (setup)
  {
    // first setup
    std::shared_ptr<Core::LinAlg::SparseMatrix> constrmt =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gdisprowmap_, 100, false, true);
    constrmt->add(*dmatrix_, true, 1.0, 1.0);
    constrmt->add(*mmatrix_, true, -1.0, 1.0);
    constrmt->complete(*gsdofrowmap_, *gdisprowmap_);

    // transform parallel row distribution
    // (only necessary in the parallel redistribution case)
    std::shared_ptr<Core::LinAlg::SparseMatrix> temp;
    if (par_redist())
      temp = Core::LinAlg::matrix_row_transform(*constrmt, *problem_dofs());
    else
      temp = constrmt;

    // always transform column GIDs of constraint matrix
    conmatrix_ = Core::LinAlg::matrix_col_transform_gids(*temp, *glmdofrowmap_);
    conmatrix_->scale(1. - alphaf_);
  }

  dm_matrix_ = nullptr;
  dm_matrix_t_ = nullptr;
  lm_diag_matrix_ = nullptr;

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_end = Teuchos::Time::wallTime() - t_start;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "in...." << std::scientific << std::setprecision(6) << t_end << " secs"
              << std::endl;
}

/*----------------------------------------------------------------------*
 |  mesh initialization for rotational invariance             popp 12/09|
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
CONTACT::MtLagrangeStrategy::mesh_initialization()
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtLagrangeStrategy::mesh_initialization");

  // get out of here if NTS algorithm is activated
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(params(), "ALGORITHM") ==
      Inpar::Mortar::algorithm_nts)
    return nullptr;

  // print message
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Performing mesh initialization...........";
    fflush(stdout);
  }

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  //**********************************************************************
  // (1) get master positions on global level
  //**********************************************************************
  // fill Xmaster first
  std::shared_ptr<Core::LinAlg::Vector<double>> Xmaster =
      Core::LinAlg::create_vector(*gmdofrowmap_, true);
  assemble_coords("master", true, *Xmaster);

  //**********************************************************************
  // (2) solve for modified slave positions on global level
  //**********************************************************************
  // initialize modified slave positions
  std::shared_ptr<Core::LinAlg::Vector<double>> Xslavemod =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);

  // shape function type and type of LM interpolation for quadratic elements
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  // quadratic FE with dual LM
  if (dualquadslavetrafo())
  {
    if (lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // split T^-1
      std::shared_ptr<Core::LinAlg::SparseMatrix> it_ss, it_sm, it_ms, it_mm;
      Core::LinAlg::split_matrix2x2(invtrafo_, gsdofrowmap_, gmdofrowmap_, gsdofrowmap_,
          gmdofrowmap_, it_ss, it_sm, it_ms, it_mm);

      // build lhs
      Core::LinAlg::SparseMatrix lhs(*gsdofrowmap_, 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> direct =
          Core::LinAlg::matrix_multiply(*dmatrix_, false, *it_ss, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> mixed =
          Core::LinAlg::matrix_multiply(*mmatrix_, false, *it_ms, false, false, false, true);
      lhs.add(*direct, false, 1.0, 1.0);
      lhs.add(*mixed, false, -1.0, 1.0);
      lhs.complete();

      // build rhs
      std::shared_ptr<Core::LinAlg::Vector<double>> xm =
          Core::LinAlg::create_vector(*gmdofrowmap_, true);
      assemble_coords("master", true, *xm);
      std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
          std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
      mmatrix_->multiply(false, *xm, *rhs);

      // solve with default solver

      Teuchos::ParameterList solvparams;
      Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
          "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
      Core::LinAlg::Solver solver(
          solvparams, get_comm(), nullptr, Core::IO::Verbositylevel::standard);

      Core::LinAlg::SolverParams solver_params;
      solver_params.refactor = true;
      solver.solve(Core::Utils::shared_ptr_from_ref(lhs), Xslavemod, rhs, solver_params);
    }
    else
    {
      // this is trivial for dual Lagrange multipliers
      mhatmatrix_->multiply(false, *Xmaster, *Xslavemod);
    }
  }

  // other cases (quadratic FE with std LM, linear FE)
  else
  {
    // CASE A: DUAL LM SHAPE FUNCTIONS
    if (shapefcn == Inpar::Mortar::shape_dual)
    {
      // this is trivial for dual Lagrange multipliers
      mhatmatrix_->multiply(false, *Xmaster, *Xslavemod);
    }

    // CASE B: STANDARD LM SHAPE FUNCTIONS
    else if (shapefcn == Inpar::Mortar::shape_standard)
    {
      // create linear problem
      std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
          Core::LinAlg::create_vector(*gsdofrowmap_, true);
      mmatrix_->multiply(false, *Xmaster, *rhs);

      // solve with default solver

      Teuchos::ParameterList solvparams;
      Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
          "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
      Core::LinAlg::Solver solver(
          solvparams, get_comm(), nullptr, Core::IO::Verbositylevel::standard);

      Core::LinAlg::SolverParams solver_params;
      solver_params.refactor = true;
      solver.solve(dmatrix_, Xslavemod, rhs, solver_params);
    }
  }

  //**********************************************************************
  // (3) perform mesh initialization node by node
  //**********************************************************************
  // this can be done in the AbstractStrategy now
  MtAbstractStrategy::mesh_initialization(Xslavemod);

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_end = Teuchos::Time::wallTime() - t_start;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "in...." << std::scientific << std::setprecision(6) << t_end << " secs"
              << std::endl;
  }

  //**********************************************************************
  // blank symmetry rows in dinv
  //**********************************************************************
  invd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dmatrix_);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  int err = 0;

  // extract diagonal of invd into diag
  invd_->extract_diagonal_copy(*diag);

  // set zero diagonal values to dummy 1.0
  for (int i = 0; i < diag->local_length(); ++i)
    if ((*diag)[i] == 0.0) (*diag).get_values()[i] = 1.0;

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
  err = invd_->replace_diagonal_values(*diag);
  if (err < 0) FOUR_C_THROW("replace_diagonal_values() failed with error code {}.", err);

  // do the multiplication M^ = inv(D) * M
  mhatmatrix_ = Core::LinAlg::matrix_multiply(*invd_, false, *mmatrix_, false, false, false, true);

  // return xslavemod for global problem
  return Xslavemod;
}

/*----------------------------------------------------------------------*
 |  evaluate meshtying (public)                               popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::evaluate_meshtying(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis)
{
  // system type, shape function type and type of LM interpolation for quadratic elements
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == CONTACT::SystemType::condensed ||
      systype == CONTACT::SystemType::condensed_lagmult)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual) FOUR_C_THROW("Condensation only for dual LM");

    // complete stiffness matrix
    // (this is a prerequisite for the Split2x2 methods to be called later)
    kteff->complete();

    /**********************************************************************/
    /* Split kteff into 3x3 block matrix                                  */
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
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffmatrix =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kteff);

    /**********************************************************************/
    /* Apply basis transformation to K and f                              */
    /* (currently only needed for quadratic FE with linear dual LM)       */
    /**********************************************************************/
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // basis transformation
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (par_redist())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();

      // apply basis transformation to K and f
      kteffmatrix =
          Core::LinAlg::matrix_multiply(*kteffmatrix, false, systrafo, false, false, false, true);
      kteffmatrix =
          Core::LinAlg::matrix_multiply(systrafo, true, *kteffmatrix, false, false, false, true);
      systrafo.multiply(true, *feff, *feff);
    }

    if (par_redist())
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
    /* Split feff into 3 subvectors                                       */
    /**********************************************************************/
    // we want to split f into 3 groups s.m,n
    std::shared_ptr<Core::LinAlg::Vector<double>> fs, fm, fn;

    // temporarily we need the group sm
    std::shared_ptr<Core::LinAlg::Vector<double>> fsm;

    // do the vector splitting smn -> sm+n
    Core::LinAlg::split_vector(*problem_dofs(), *feff, gsmdofrowmap_, fsm, gndofrowmap_, fn);

    // we want to split fsm into 2 groups s,m
    fs = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
    fm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);

    // do the vector splitting sm -> s+m
    Core::LinAlg::split_vector(*gsmdofrowmap_, *fsm, gsdofrowmap_, fs, gmdofrowmap_, fm);

    // store some stuff for static condensation of LM
    fs_ = fs;
    ksn_ = ksn;
    ksm_ = ksm;
    kss_ = kss;

    //***************************************************************************
    // build constraint vector
    //***************************************************************************
    // Since we enforce the meshtying constraint for the displacements u,
    // and not for the configurations x (which would also be possible in theory),
    // we avoid artificial initial stresses (+), but we might not guarantee
    // exact rotational invariance (-). However, since we always apply the
    // so-called mesh initialization procedure, we can then also guarantee
    // exact rotational invariance (+).
    //***************************************************************************

    // (nothing needs not be done here, as the constraints for meshtying are
    // LINEAR w.r.t. the displacements and in the first step, dis is zero.
    // Thus, the right hand side of the constraint rows is ALWAYS zero!)

    /**********************************************************************/
    /* Build the final K and f blocks                                     */
    /**********************************************************************/
    // knn: nothing to do

    // knm:
    std::shared_ptr<Core::LinAlg::SparseMatrix> knmmod;
    if (systype == CONTACT::SystemType::condensed)
    {
      // knm: add kns*mbar
      knmmod = std::make_shared<Core::LinAlg::SparseMatrix>(*gndofrowmap_, 100);
      knmmod->add(*knm, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> knmadd =
          Core::LinAlg::matrix_multiply(*kns, false, *mhatmatrix_, false, false, false, true);
      knmmod->add(*knmadd, false, 1.0, 1.0);
      knmmod->complete(knm->domain_map(), knm->row_map());
    }

    // kmn: add T(mbar)*ksn
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmnmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmnmod->add(*kmn, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmnadd =
        Core::LinAlg::matrix_multiply(*mhatmatrix_, true, *ksn, false, false, false, true);
    kmnmod->add(*kmnadd, false, 1.0, 1.0);
    kmnmod->complete(kmn->domain_map(), kmn->row_map());

    // kmm: add T(mbar)*ksm
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmmod =
        std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
    kmmmod->add(*kmm, false, 1.0, 1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd =
        Core::LinAlg::matrix_multiply(*mhatmatrix_, true, *ksm, false, false, false, true);
    kmmmod->add(*kmmadd, false, 1.0, 1.0);
    if (systype == CONTACT::SystemType::condensed)
    {
      // kmm: add kms*mbar + T(mbar)*kss*mbar - additionally
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd2 =
          Core::LinAlg::matrix_multiply(*kms, false, *mhatmatrix_, false, false, false, true);
      kmmmod->add(*kmmadd2, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmmtemp =
          Core::LinAlg::matrix_multiply(*kss, false, *mhatmatrix_, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmmadd3 =
          Core::LinAlg::matrix_multiply(*mhatmatrix_, true, *kmmtemp, false, false, false, true);
      kmmmod->add(*kmmadd3, false, 1.0, 1.0);
    }
    kmmmod->complete(kmm->domain_map(), kmm->row_map());

    // some modifications for kns, kms, (,ksn) ksm, kss if slave displacement increment is not
    // condensed

    // kns: nothing to do

    // kms: add T(mbar)*kss
    std::shared_ptr<Core::LinAlg::SparseMatrix> kmsmod;
    if (systype == CONTACT::SystemType::condensed_lagmult)
    {
      kmsmod = std::make_shared<Core::LinAlg::SparseMatrix>(*gmdofrowmap_, 100);
      kmsmod->add(*kms, false, 1.0, 1.0);
      std::shared_ptr<Core::LinAlg::SparseMatrix> kmsadd =
          Core::LinAlg::matrix_multiply(*mhatmatrix_, true, *kss, false, false, false, true);
      kmsmod->add(*kmsadd, false, 1.0, 1.0);
      kmsmod->complete(kms->domain_map(), kms->row_map());
    }

    // (ksn: do nothing as block is supposed to remain zero)

    // ksm: subtract mmatrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> ksmmod;
    if (systype == CONTACT::SystemType::condensed_lagmult)
    {
      ksmmod = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);
      ksmmod->add(*mmatrix_, false, -1.0, 1.0);  //<---- causes problems in parallel
      ksmmod->complete(ksm->domain_map(), ksm->row_map());
    }

    // kss: add dmatrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> kssmod;
    if (systype == CONTACT::SystemType::condensed_lagmult)
    {
      kssmod = std::make_shared<Core::LinAlg::SparseMatrix>(*gsdofrowmap_, 100);
      kssmod->add(*dmatrix_, false, 1.0, 1.0);  //<---- causes problems in parallel
      kssmod->complete(kss->domain_map(), kss->row_map());
    }

    // fn: subtract kns*inv(D)*g
    // (nothing needs to be done, since the right hand side g is ALWAYS zero)

    // fs: subtract alphaf * old interface forces (t_n)
    Core::LinAlg::Vector<double> tempvecs(*gsdofrowmap_);
    dmatrix_->multiply(true, *zold_, tempvecs);
    tempvecs.update(1.0, *fs, -alphaf_);

    // fm: add alphaf * old interface forces (t_n)
    Core::LinAlg::Vector<double> tempvecm(*gmdofrowmap_);
    mmatrix_->multiply(true, *zold_, tempvecm);
    fm->update(alphaf_, tempvecm, 1.0);

    // fm: add T(mbar)*fs
    Core::LinAlg::Vector<double> fmmod(*gmdofrowmap_);
    mhatmatrix_->multiply(true, tempvecs, fmmod);
    fmmod.update(1.0, *fm, 1.0);

    // fm: subtract kmsmod*inv(D)*g
    // (nothing needs to be done, since the right hand side g is ALWAYS zero)

    // RHS can remain unchanged, if slave displacement increments are not condensed
    // build identity matrix for slave dofs
    Core::LinAlg::Vector<double> ones(*gsdofrowmap_);
    ones.put_scalar(1.0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> onesdiag =
        std::make_shared<Core::LinAlg::SparseMatrix>(ones);
    onesdiag->complete();

    /********************************************************************/
    /* Transform the final K blocks                                     */
    /********************************************************************/
    // The row maps of all individual matrix blocks are transformed to
    // the parallel layout of the underlying problem discretization.
    // Of course, this is only necessary in the parallel redistribution
    // case, where the meshtying interfaces have been redistributed
    // independently of the underlying problem discretization.
    if (par_redist())
    {
      kmnmod = Core::LinAlg::matrix_row_transform(*kmnmod, *non_redist_gmdofrowmap_);
      kmmmod = Core::LinAlg::matrix_row_transform(*kmmmod, *non_redist_gmdofrowmap_);
      onesdiag = Core::LinAlg::matrix_row_transform(*onesdiag, *non_redist_gsdofrowmap_);
    }

    /**********************************************************************/
    /* Global setup of kteffnew, feffnew (including meshtying)            */
    /**********************************************************************/
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *problem_dofs(), 81, true, false, kteffmatrix->get_matrixtype());
    std::shared_ptr<Core::LinAlg::Vector<double>> feffnew =
        Core::LinAlg::create_vector(*problem_dofs());

    // add n submatrices to kteffnew
    kteffnew->add(*knn, false, 1.0, 1.0);
    if (systype == CONTACT::SystemType::condensed)
    {
      kteffnew->add(*knmmod, false, 1.0, 1.0);
    }
    else if (systype == CONTACT::SystemType::condensed_lagmult)
    {
      kteffnew->add(*knm, false, 1.0, 1.0);
      kteffnew->add(*kns, false, 1.0, 1.0);
    }

    // add m submatrices to kteffnew
    kteffnew->add(*kmnmod, false, 1.0, 1.0);
    kteffnew->add(*kmmmod, false, 1.0, 1.0);
    if (systype == CONTACT::SystemType::condensed_lagmult) kteffnew->add(*kmsmod, false, 1.0, 1.0);

    // add s submatrices to kteffnew
    if (systype == CONTACT::SystemType::condensed)
    {
      // add identity for slave increments
      kteffnew->add(*onesdiag, false, 1.0, 1.0);
    }
    else
    {
      kteffnew->add(*ksmmod, false, 1.0, 1.0);
      kteffnew->add(*kssmod, false, 1.0, 1.0);
    }

    // fill_complete kteffnew (square)
    kteffnew->complete();

    // add n subvector to feffnew
    Core::LinAlg::Vector<double> fnexp(*problem_dofs());
    Core::LinAlg::export_to(*fn, fnexp);
    feffnew->update(1.0, fnexp, 1.0);

    // add m subvector to feffnew
    Core::LinAlg::Vector<double> fmmodexp(*problem_dofs());
    Core::LinAlg::export_to(fmmod, fmmodexp);
    feffnew->update(1.0, fmmodexp, 1.0);

    /**********************************************************************/
    /* Replace kteff and feff by kteffnew and feffnew                     */
    /**********************************************************************/
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
    /**********************************************************************/
    /* Apply basis transformation to K and f                              */
    /* (currently only needed for quadratic FE with linear dual LM)       */
    /**********************************************************************/
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // basis transformation
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (par_redist())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();

      // apply basis transformation to K and f
      kteff->complete();
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
      systrafo.multiply(true, *feff, *feff);
    }

    // add meshtying force terms
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
    dmatrix_->multiply(true, *zold_, fsold);
    Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
    Core::LinAlg::export_to(fsold, fsoldexp);
    feff->update(-alphaf_, fsoldexp, 1.0);

    Core::LinAlg::Vector<double> fmold(*gmdofrowmap_);
    mmatrix_->multiply(true, *zold_, fmold);
    Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
    Core::LinAlg::export_to(fmold, fmoldexp);
    feff->update(alphaf_, fmoldexp, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::build_saddle_point_system(
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

  //**********************************************************************
  // prepare saddle point system
  //**********************************************************************
  // get system type
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  // the standard stiffness matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiffmt =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(kdd);

  // initialize merged system (matrix, rhs, sol)
  std::shared_ptr<Core::LinAlg::Map> mergedmap =
      Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedrhs = Core::LinAlg::create_vector(*mergedmap);
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedsol = Core::LinAlg::create_vector(*mergedmap);
  std::shared_ptr<Core::LinAlg::Vector<double>> mergedzeros =
      Core::LinAlg::create_vector(*mergedmap);

  //**********************************************************************
  // finalize matrix and vector blocks
  //**********************************************************************
  // get constraint matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> constrmt = conmatrix_;

  // build constraint rhs (=empty)
  std::shared_ptr<Core::LinAlg::Vector<double>> constrrhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*glmdofrowmap_);
  constrrhs_ = constrrhs;  // set constraint rhs vector

  //**********************************************************************
  // build and solve saddle point system
  //**********************************************************************
  if (systype == CONTACT::SystemType::saddlepoint)
  {
    // build transposed constraint matrix
    Core::LinAlg::SparseMatrix trconstrmt(*glmdofrowmap_, 100, false, true);
    trconstrmt.add(*constrmt, true, 1.0, 0.0);
    trconstrmt.complete(*problem_dofs(), *glmdofrowmap_);

    // apply Dirichlet conditions to (0,1) block
    Core::LinAlg::Vector<double> zeros(*problem_dofs(), true);
    Core::LinAlg::Vector<double> rhscopy(*fd);
    Core::LinAlg::apply_dirichlet_to_system(*stiffmt, *sold, rhscopy, zeros, dirichtoggle);
    constrmt->apply_dirichlet(dirichtoggle, false);

    // row map (equals domain map) extractor
    Core::LinAlg::MapExtractor mapext(*mergedmap, glmdofrowmap_, problem_dofs());

    // build block matrix for SIMPLER
    blockMat =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            mapext, mapext, 81, false, false);
    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>> mat =
        std::dynamic_pointer_cast<
            Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(blockMat);

    mat->assign(0, 0, Core::LinAlg::DataAccess::Share, *stiffmt);
    mat->assign(0, 1, Core::LinAlg::DataAccess::Share, *constrmt);
    mat->assign(1, 0, Core::LinAlg::DataAccess::Share, trconstrmt);
    mat->complete();

    // we also need merged rhs here
    Core::LinAlg::Vector<double> fresmexp(*mergedmap);
    Core::LinAlg::export_to(*fd, fresmexp);
    mergedrhs->update(1.0, fresmexp, 1.0);
    Core::LinAlg::Vector<double> constrexp(*mergedmap);
    Core::LinAlg::export_to(*constrrhs, constrexp);
    mergedrhs->update(1.0, constrexp, 1.0);

    // apply Dirichlet B.C. to mergedrhs and mergedsol
    Core::LinAlg::Vector<double> dirichtoggleexp(*mergedmap);
    Core::LinAlg::export_to(dirichtoggle, dirichtoggleexp);
    Core::LinAlg::apply_dirichlet_to_system(*mergedsol, *mergedrhs, *mergedzeros, dirichtoggleexp);

    // make solver SIMPLER-ready
    // solver.Params().set<bool>("MESHTYING", true); // flag makes sure that SIMPLER sets correct
    // nullptr space for constraint equations

    blocksol = mergedsol;
    blockrhs = mergedrhs;
  }

  //**********************************************************************
  // invalid system types
  //**********************************************************************
  else
  {
    FOUR_C_THROW("Invalid system type in SaddlePointSolve");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::update_displacements_and_l_mincrements(
    std::shared_ptr<Core::LinAlg::Vector<double>> sold,
    std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol)
{
  //**********************************************************************
  // extract results for displacement and LM increments
  //**********************************************************************
  Core::LinAlg::Vector<double> sollm(*glmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> mergedmap =
      Core::LinAlg::merge_map(problem_dofs(), glmdofrowmap_, false);
  Core::LinAlg::MapExtractor mapext(*mergedmap, problem_dofs(), glmdofrowmap_);
  mapext.extract_cond_vector(*blocksol, *sold);
  mapext.extract_other_vector(*blocksol, sollm);
  sollm.replace_map(*gsdofrowmap_);

  zincr_->update(1.0, sollm, 0.0);
  z_->update(1.0, *zincr_, 1.0);
}


/*----------------------------------------------------------------------*
 | Recovery method                                            popp 04/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi)
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtLagrangeStrategy::recover");

  // system type, shape function type and type of LM interpolation for quadratic elements
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(params(), "LM_SHAPEFCN");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

  //**********************************************************************
  //**********************************************************************
  // CASE A: CONDENSED SYSTEM (DUAL)
  //**********************************************************************
  //**********************************************************************
  if (systype == CONTACT::SystemType::condensed ||
      systype == CONTACT::SystemType::condensed_lagmult)
  {
    // double-check if this is a dual LM system
    if (shapefcn != Inpar::Mortar::shape_dual) FOUR_C_THROW("Condensation only for dual LM");

    // extract slave displacements from disi
    Core::LinAlg::Vector<double> disis(*gsdofrowmap_);
    if (gsdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disis);

    // extract master displacements from disi
    Core::LinAlg::Vector<double> disim(*gmdofrowmap_);
    if (gmdofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disim);

    // extract other displacements from disi
    Core::LinAlg::Vector<double> disin(*gndofrowmap_);
    if (gndofrowmap_->num_global_elements()) Core::LinAlg::export_to(*disi, disin);

    /**********************************************************************/
    /* Update slave increment \Delta d_s                                  */
    /**********************************************************************/

    if (systype == CONTACT::SystemType::condensed)
    {
      mhatmatrix_->multiply(false, disim, disis);
      Core::LinAlg::Vector<double> disisexp(*problem_dofs());
      Core::LinAlg::export_to(disis, disisexp);
      disi->update(1.0, disisexp, 1.0);
    }

    /**********************************************************************/
    /* Undo basis transformation to solution                              */
    /* (currently only needed for quadratic FE with linear dual LM)       */
    /**********************************************************************/
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // undo basis transformation to solution
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (par_redist())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();
      systrafo.multiply(false, *disi, *disi);
    }

    /**********************************************************************/
    /* Update Lagrange multipliers z_n+1                                  */
    /**********************************************************************/

    // approximate update
    // invd_->Multiply(false,*fs_,*z_);
    // full update
    z_->update(1.0, *fs_, 0.0);
    Core::LinAlg::Vector<double> mod(*gsdofrowmap_);
    kss_->multiply(false, disis, mod);
    z_->update(-1.0, mod, 1.0);
    ksm_->multiply(false, disim, mod);
    z_->update(-1.0, mod, 1.0);
    ksn_->multiply(false, disin, mod);
    z_->update(-1.0, mod, 1.0);
    dmatrix_->multiply(true, *zold_, mod);
    z_->update(-alphaf_, mod, 1.0);
    Core::LinAlg::Vector<double> zcopy(*z_);
    invd_->multiply(true, zcopy, *z_);
    z_->scale(1 / (1 - alphaf_));
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
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // undo basis transformation to solution
      Core::LinAlg::SparseMatrix systrafo(*problem_dofs(), 100, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
          Core::LinAlg::create_identity_matrix(*gndofrowmap_);
      systrafo.add(*eye, false, 1.0, 1.0);
      if (par_redist())
        trafo_ = Core::LinAlg::matrix_row_col_transform(
            *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
      systrafo.add(*trafo_, false, 1.0, 1.0);
      systrafo.complete();
      systrafo.multiply(false, *disi, *disi);
    }
  }

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool CONTACT::MtLagrangeStrategy::evaluate_force(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (!f_) f_ = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
  f_->put_scalar(0.);

  if (system_type() != CONTACT::SystemType::condensed)
  {
    // add meshtying force terms
    Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
    if (dmatrix_->multiply(true, *z_, fs)) FOUR_C_THROW("multiply failed");
    Core::LinAlg::Vector<double> fsexp(*problem_dofs());
    Core::LinAlg::export_to(fs, fsexp);
    f_->update(1.0, fsexp, 1.0);

    Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
    mmatrix_->multiply(true, *z_, fm);
    Core::LinAlg::Vector<double> fmexp(*problem_dofs());
    Core::LinAlg::export_to(fm, fmexp);
    f_->update(-1.0, fmexp, 1.0);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool CONTACT::MtLagrangeStrategy::evaluate_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (dm_matrix_ && dm_matrix_t_ && lm_diag_matrix_) return true;

  std::shared_ptr<Core::LinAlg::SparseMatrix> constrmt =
      std::make_shared<Core::LinAlg::SparseMatrix>(*gdisprowmap_, 100, false, true);
  constrmt->add(*dmatrix_, true, 1.0, 1.0);
  constrmt->add(*mmatrix_, true, -1.0, 1.0);
  constrmt->complete(*gsdofrowmap_, *gdisprowmap_);

  // transform parallel row distribution
  // (only necessary in the parallel redistribution case)
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp;
  if (par_redist())
    temp = Core::LinAlg::matrix_row_transform(*constrmt, *problem_dofs());
  else
    temp = constrmt;

  // always transform column GIDs of constraint matrix
  dm_matrix_ = Core::LinAlg::matrix_col_transform_gids(*temp, *lm_dof_row_map_ptr());
  dm_matrix_t_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*lm_dof_row_map_ptr(), 100, false, true);
  dm_matrix_t_->add(*dm_matrix_, true, 1., 0.);
  dm_matrix_t_->complete(*problem_dofs(), *lm_dof_row_map_ptr());

  dm_matrix_->scale(1. - alphaf_);
  dm_matrix_t_->scale(1. - alphaf_);

  lm_diag_matrix_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*lm_dof_row_map_ptr(), 1, false, true);
  lm_diag_matrix_->complete();

  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
  {
    systrafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
        Core::LinAlg::create_identity_matrix(*gndofrowmap_);
    systrafo_->add(*eye, false, 1.0, 1.0);
    if (par_redist())
      trafo_ = Core::LinAlg::matrix_row_col_transform(
          *trafo_, *non_redist_gsmdofrowmap_, *non_redist_gsmdofrowmap_);
    systrafo_->add(*trafo_, false, 1.0, 1.0);
    systrafo_->complete();
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool CONTACT::MtLagrangeStrategy::evaluate_force_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  bool successForce = evaluate_force(dis);
  bool successStiff = evaluate_stiff(dis);

  return (successForce && successStiff);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> CONTACT::MtLagrangeStrategy::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vec_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
    {
      vec_ptr = f_;
      break;
    }
    case CONTACT::VecBlockType::constraint:
    {
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Solid::VecBlockType!");
    }
  }

  return vec_ptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::MtLagrangeStrategy::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
      mat_ptr = nullptr;
      break;
    case CONTACT::MatBlockType::displ_lm:
      if (!dm_matrix_) FOUR_C_THROW("matrix not available");
      mat_ptr = dm_matrix_;
      break;
    case CONTACT::MatBlockType::lm_displ:
      if (!dm_matrix_t_) FOUR_C_THROW("matrix not available");
      mat_ptr = dm_matrix_t_;
      break;
    case CONTACT::MatBlockType::lm_lm:
      if (!lm_diag_matrix_) FOUR_C_THROW("matrix not available");
      mat_ptr = lm_diag_matrix_;
      break;
    default:
      FOUR_C_THROW("Unknown Solid::MatBlockType!");
  }
  return mat_ptr;
}

std::shared_ptr<const Core::LinAlg::SparseMatrix>
CONTACT::MtLagrangeStrategy::get_non_redist_m_hat()
{
  return Core::LinAlg::matrix_row_col_transform(
      *mhatmatrix_, *non_redist_slave_row_dofs(), *non_redist_master_row_dofs());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::run_pre_apply_jacobian_inverse(
    std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs)
{
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");

  if (systype == CONTACT::SystemType::condensed)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> k =
        std::make_shared<Core::LinAlg::SparseMatrix>(*kteff);
    std::shared_ptr<Core::LinAlg::Vector<double>> r =
        Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(rhs);

    auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
    {
      // apply basis transformation to K and f
      k = Core::LinAlg::matrix_multiply(*k, false, *systrafo_, false, false, false, true);
      k = Core::LinAlg::matrix_multiply(*systrafo_, true, *k, false, false, false, true);
      systrafo_->multiply(true, *r, *r);
    }

    std::shared_ptr<const Core::LinAlg::SparseMatrix> non_redist_mhatmatrix =
        get_non_redist_m_hat();
    Mortar::Utils::mortar_matrix_condensation(k, non_redist_mhatmatrix, non_redist_mhatmatrix);
    *kteff = *k;

    Mortar::Utils::mortar_rhs_condensation(*r, *mhatmatrix_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::run_post_apply_jacobian_inverse(
    Core::LinAlg::Vector<double>& result)
{
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (systype == CONTACT::SystemType::condensed)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> inc =
        Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(result);
    Mortar::Utils::mortar_recover(*inc, *mhatmatrix_);

    // undo basis transformation to solution
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
      systrafo_->multiply(false, *inc, *inc);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
    const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew)
{
  if (system_type() != CONTACT::SystemType::condensed)
  {
    Core::LinAlg::Vector<double> zdir_ptr(*glmdofrowmap_, true);
    Core::LinAlg::export_to(dir, zdir_ptr);
    zdir_ptr.replace_map(*gsdofrowmap_);
    z_->update(1., zdir_ptr, 1.);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::MtLagrangeStrategy::remove_condensed_contributions_from_rhs(
    Core::LinAlg::Vector<double>& rhs) const
{
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(params(), "SYSTEM");
  auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");
  if (systype == CONTACT::SystemType::condensed)
  {
    // undo basis transformation to solution
    if (dualquadslavetrafo() && lagmultquad == Inpar::Mortar::lagmult_lin)
      systrafo_->multiply(true, rhs, rhs);

    std::shared_ptr<Core::LinAlg::Vector<double>> r =
        Core::Utils::shared_ptr_from_ref<Core::LinAlg::Vector<double>>(rhs);
    Mortar::Utils::mortar_rhs_condensation(*r, *mhatmatrix_);
  }
}

FOUR_C_NAMESPACE_CLOSE
