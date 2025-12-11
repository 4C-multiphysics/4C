// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_meshsliding.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

ALE::Meshsliding::Meshsliding(std::shared_ptr<Core::FE::Discretization> dis,
    Core::LinAlg::Solver& solver, int msht, int nsd, const Utils::MapExtractor* surfacesplitter)
    : Meshtying(dis, solver, msht, nsd, surfacesplitter)
{
}


/*-------------------------------------------------------*/
/*  Call the constructor and the setup of the mortar     */
/*  coupling adapter                         wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshsliding::adapter_mortar(std::vector<int> coupleddof)
{
  adaptermeshsliding_ = std::make_shared<Adapter::CouplingNonLinMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  // Setup and Output of Nonlinear meshtying adapter
  adaptermeshsliding_->setup(discret_, discret_, coupleddof, "Mortar");
}

/*-------------------------------------------------------*/
/*  Setup mesh sliding problem               wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator> ALE::Meshsliding::setup(
    std::vector<int> coupleddof, std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  std::shared_ptr<Core::LinAlg::SparseOperator> mat = Meshtying::setup(coupleddof, dispnp);

  lm_ = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_, true);

  return mat;
}

/*-------------------------------------------------------*/
/*  Compare the size of the slave and master dof row map */
/*                                           wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshsliding::compare_num_dof()
{
  int numdofmaster = (adaptermeshsliding_->master_dof_map())->num_global_elements();
  int numdofslave = (adaptermeshsliding_->slave_dof_map())->num_global_elements();

  std::cout << std::endl << "number of master dof's:   " << numdofmaster << std::endl;
  std::cout << "number of slave dof's:   " << numdofslave << std::endl << std::endl;

  if (numdofmaster > numdofslave)
    std::cout << "The master side is discretized by more elements than the slave side" << std::endl;
  else
    std::cout << "The slave side is discretized by more elements than the master side" << std::endl;
}

/*-------------------------------------------------------*/
/*  Get function for the slave and master dof row map    */
/*                                           wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshsliding::dof_row_maps()
{
  // slave dof rowmap
  gsdofrowmap_ = adaptermeshsliding_->slave_dof_map();

  // master dof rowmap
  gmdofrowmap_ = adaptermeshsliding_->master_dof_map();
}

/*-------------------------------------------------------*/
/*  Get function for the P matrix            wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> ALE::Meshsliding::get_mortar_matrix_p()
{
  return adaptermeshsliding_->get_mortar_matrix_p();
}

/*-------------------------------------------------------*/
/*  Condensation operation for a block matrix            */
/*                                          wirtz 02/16  */
/*-------------------------------------------------------*/
void ALE::Meshsliding::condensation_operation_block_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator>&
        sysmat,  ///> sysmat established by the element routine
    std::shared_ptr<Core::LinAlg::Vector<double>>&
        residual,  ///> residual established by the element routine
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)  ///> current displacement vector
{
  // condensation operation for a block matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);

  /**********************************************************************/
  /* Split residual into 3 subvectors                                   */
  /**********************************************************************/

  // container for split residual vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitres(3);
  split_vector(*residual, splitres);

  /**********************************************************************/
  /* Evaluate mortar matrices                                           */
  /**********************************************************************/

  // evaluate all mortar matrices and vectors necessary for mesh sliding
  adaptermeshsliding_->evaluate_sliding("displacement", dispnp, lm_);

  // get all mortar matrices necessary for mesh sliding
  std::shared_ptr<Core::LinAlg::SparseMatrix> Aco_mm;
  std::shared_ptr<Core::LinAlg::SparseMatrix> Aco_ms;
  std::shared_ptr<Core::LinAlg::SparseMatrix> Aco_sm;
  std::shared_ptr<Core::LinAlg::SparseMatrix> Aco_ss;
  std::shared_ptr<Core::LinAlg::SparseMatrix> N_m;
  std::shared_ptr<Core::LinAlg::SparseMatrix> N_s;
  get_mortar_matrices(Aco_mm, Aco_ms, Aco_sm, Aco_ss, N_m, N_s);
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = get_mortar_matrix_p();
  std::shared_ptr<Core::LinAlg::SparseMatrix> T = adaptermeshsliding_->t_matrix();
  std::shared_ptr<Core::LinAlg::SparseMatrix> H = adaptermeshsliding_->h_matrix();
  std::shared_ptr<Core::LinAlg::Vector<double>> gap = adaptermeshsliding_->gap();

  /**********************************************************************/
  /* Step 1: Add sliding stiffnesses to the system matrix               */
  /**********************************************************************/

  sysmatnew->matrix(1, 1).un_complete();
  Core::LinAlg::matrix_add(*Aco_mm, false, 1.0, sysmatnew->matrix(1, 1), 1.0);

  sysmatnew->matrix(1, 2).un_complete();
  Core::LinAlg::matrix_add(*Aco_ms, false, 1.0, sysmatnew->matrix(1, 2), 1.0);

  sysmatnew->matrix(2, 1).un_complete();
  Core::LinAlg::matrix_add(*Aco_sm, false, 1.0, sysmatnew->matrix(2, 1), 1.0);

  sysmatnew->matrix(2, 2).un_complete();
  Core::LinAlg::matrix_add(*Aco_ss, false, 1.0, sysmatnew->matrix(2, 2), 1.0);

  sysmatnew->complete();

  /**********************************************************************/
  /* Store some quantities for static condensation of LM                */
  /**********************************************************************/

  a_ss_ = std::make_shared<Core::LinAlg::SparseMatrix>(sysmatnew->matrix(2, 2));
  a_sm_ = std::make_shared<Core::LinAlg::SparseMatrix>(sysmatnew->matrix(2, 1));
  a_sn_ = std::make_shared<Core::LinAlg::SparseMatrix>(sysmatnew->matrix(2, 0));
  d_inv_ = adaptermeshsliding_->get_mortar_matrix_dinv();
  rs_ = std::make_shared<Core::LinAlg::Vector<double>>(*splitres[2]);

  /**********************************************************************/
  /* Step 2: condensate the system matrix                               */
  /**********************************************************************/

  // ----------------------------------------------------------------------
  // | A_nn | A_nm | A_ns |  0   |        | A_nn | A_nm | A_ns |
  // | A_mn | A_mm | A_ms | -M^T |   ->   | A_mn'| A_mm'| A_ms'|
  // | A_sn | A_sm | A_ss |  D^T |        | A_sn'| A_sm'| A_ss'|
  // |   0  |  N_m |  N_s |  0   |
  // |   0  |  0   |  H_s |  T   |

  //----------------------------------------------------------- FIRST LINE

  // do nothing

  //---------------------------------------------------------- SECOND LINE

  // compute modification for block mn       (+ P^T * A_sn)
  std::shared_ptr<Core::LinAlg::SparseMatrix> Amn_mod =
      matrix_multiply(*P, true, sysmatnew->matrix(2, 0), false, false, false, true);

  // Add modification block to mn
  sysmatnew->matrix(1, 0).un_complete();  // sonst kann ich auf den Block nichts neues draufaddieren
  Core::LinAlg::matrix_add(*Amn_mod, false, 1.0, sysmatnew->matrix(1, 0), 1.0);

  // compute modification for block mm       (+ P^T * A_sm)
  std::shared_ptr<Core::LinAlg::SparseMatrix> Amm_mod =
      matrix_multiply(*P, true, sysmatnew->matrix(2, 1), false, false, false, true);

  // Add modification block to mm
  sysmatnew->matrix(1, 1).un_complete();
  Core::LinAlg::matrix_add(*Amm_mod, false, 1.0, sysmatnew->matrix(1, 1), 1.0);

  // compute modification for block ms       (+ P^T * A_ss)
  std::shared_ptr<Core::LinAlg::SparseMatrix> Ams_mod =
      matrix_multiply(*P, true, sysmatnew->matrix(2, 2), false, false, false, true);

  // Add modification block to ms
  sysmatnew->matrix(1, 2).un_complete();
  Core::LinAlg::matrix_add(*Ams_mod, false, 1.0, sysmatnew->matrix(1, 2), 1.0);

  //----------------------------------------------------------- THIRD LINE

  // compute replacement for block sn      - (T * D^(-1) * A_sn)
  std::shared_ptr<Core::LinAlg::SparseMatrix> Asn_mod_intermediate =
      matrix_multiply(*d_inv_, false, sysmatnew->matrix(2, 0), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> Asn_mod =
      matrix_multiply(*T, false, *Asn_mod_intermediate, false, false, false, true);

  // Replace sn block with (negative) modification block
  sysmatnew->matrix(2, 0).un_complete();
  Core::LinAlg::matrix_add(*Asn_mod, false, -1.0, sysmatnew->matrix(2, 0), 0.0);


  // compute replacement for block sm      - (T * D^(-1) * A_sm)   +  N_m
  std::shared_ptr<Core::LinAlg::SparseMatrix> Asm_mod_intermediate =
      matrix_multiply(*d_inv_, false, sysmatnew->matrix(2, 1), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> Asm_mod =
      matrix_multiply(*T, false, *Asm_mod_intermediate, false, false, false, true);

  // Replace sm block with (negative) modification block
  sysmatnew->matrix(2, 1).un_complete();
  Core::LinAlg::matrix_add(*Asm_mod, false, -1.0, sysmatnew->matrix(2, 1), 0.0);
  Core::LinAlg::matrix_add(*N_m, false, 1.0, sysmatnew->matrix(2, 1), 1.0);

  // compute replacement for block ss      (- T * D^(-1) *A_ss)   +  H  +  N_s
  std::shared_ptr<Core::LinAlg::SparseMatrix> Ass_mod_intermediate =
      matrix_multiply(*d_inv_, false, sysmatnew->matrix(2, 2), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> Ass_mod =
      matrix_multiply(*T, false, *Ass_mod_intermediate, false, false, false, true);

  // Replace ss block with (negative) modification block
  sysmatnew->matrix(2, 2).un_complete();
  Core::LinAlg::matrix_add(*Ass_mod, false, -1.0, sysmatnew->matrix(2, 2), 0.0);
  Core::LinAlg::matrix_add(*N_s, false, 1.0, sysmatnew->matrix(2, 2), 1.0);
  Core::LinAlg::matrix_add(*H, false, 1.0, sysmatnew->matrix(2, 2), 1.0);

  // complete matrix
  sysmatnew->complete();

  /**********************************************************************/
  /* Step 3: condensate the residual                                    */
  /***********************************************************************/

  //----------------------------------------------------------- FIRST LINE

  // do nothing

  //---------------------------------------------------------- SECOND LINE

  // r_m: add P^T*r_s
  Core::LinAlg::Vector<double> rm_mod(*gmdofrowmap_, true);
  P->multiply(true, *(splitres[2]), rm_mod);

  // export and add r_m subvector to residual
  Core::LinAlg::Vector<double> rm_modexp(*dofrowmap_);
  Core::LinAlg::export_to(rm_mod, rm_modexp);
  residual->update(1.0, rm_modexp, 1.0);

  //----------------------------------------------------------- THIRD LINE

  // r_s: * 0
  Core::LinAlg::Vector<double> rs_delete(*dofrowmap_);
  Core::LinAlg::export_to(*(splitres[2]), rs_delete);
  residual->update(-1.0, rs_delete, 1.0);

  // r_s: add - T*D^(-1)*r_s
  Core::LinAlg::Vector<double> rs_mod_intermediate(*gsdofrowmap_, true);
  d_inv_->multiply(false, *rs_, rs_mod_intermediate);
  Core::LinAlg::Vector<double> rs_mod(*gsdofrowmap_, true);
  T->multiply(false, rs_mod_intermediate, rs_mod);

  // export and subtract rs_mod from residual
  Core::LinAlg::Vector<double> rs_modexp(*dofrowmap_);
  Core::LinAlg::export_to(rs_mod, rs_modexp);
  residual->update(-1.0, rs_modexp, 1.0);

  // r_s: add gap
  Core::LinAlg::Vector<double> g_exp(*dofrowmap_);
  Core::LinAlg::export_to(*gap, g_exp);
  residual->update(1.0, g_exp, 1.0);

  return;
}

/*-------------------------------------------------------*/
/*  Get functions for the mortar matrices    wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
void ALE::Meshsliding::get_mortar_matrices(std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_mm,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_ms,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_sm,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_ss,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& N_m,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& N_s)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> MLin = adaptermeshsliding_->m_lin_matrix();
  std::shared_ptr<Core::LinAlg::SparseMatrix> DLin = adaptermeshsliding_->d_lin_matrix();
  std::shared_ptr<Core::LinAlg::SparseMatrix> N = adaptermeshsliding_->n_matrix();

  split_mortar_matrix(MLin, Aco_mm, Aco_ms, gmdofrowmap_);
  split_mortar_matrix(DLin, Aco_sm, Aco_ss, gsdofrowmap_);
  split_mortar_matrix(N, N_m, N_s, gsdofrowmap_);
}

/*-------------------------------------------------------*/
/*  Split the mortar matrix into its slave and its       */
/*  master part                              wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshsliding::split_mortar_matrix(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& MortarMatrix,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& MasterMatrix,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& SlaveMatrix,
    std::shared_ptr<const Core::LinAlg::Map>& dofrowmapconst)
{
  // dummy Matrices for second row and dummy map
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp21;
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp22;
  std::shared_ptr<Core::LinAlg::Map> dummy;

  // const casts
  std::shared_ptr<Core::LinAlg::Map> gmdofrowmap =
      std::const_pointer_cast<Core::LinAlg::Map>(gmdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> gsdofrowmap =
      std::const_pointer_cast<Core::LinAlg::Map>(gsdofrowmap_);
  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      std::const_pointer_cast<Core::LinAlg::Map>(dofrowmapconst);

  // split matrix operation
  bool suc = Core::LinAlg::split_matrix2x2(MortarMatrix, dofrowmap, dummy, gmdofrowmap, gsdofrowmap,
      MasterMatrix, SlaveMatrix, temp21, temp22);

  if (!suc) FOUR_C_THROW("\nCould not split Mortar Matriz!\n");

  // Complete Matrices
  MasterMatrix->complete();
  SlaveMatrix->complete();
}

/*-------------------------------------------------------*/
/*  Recover method for Lagrange multipliers  wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
void ALE::Meshsliding::recover(std::shared_ptr<Core::LinAlg::Vector<double>>& inc)
{
  // recover lm from the condensation:
  // lm = - D^(-1) * (r_s + A_ss*d_s + A_sm*d_m + A_sn*d_n)
  // Note: A_sm is non zero because of the linearization of the D-Matrix
  // with respect to master dofs is already included

  // split displacement increment
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitinc(3);
  split_vector(*inc, splitinc);

  Core::LinAlg::Vector<double> lm_temp(*gsdofrowmap_, true);
  Core::LinAlg::Vector<double> mod(*gsdofrowmap_, true);

  // r_s
  lm_temp.update(1.0, *rs_, 1.0);

  // + A_ss*d_s
  a_ss_->multiply(false, *(splitinc[2]), mod);
  lm_temp.update(1.0, mod, 1.0);

  // + A_sm*d_m
  a_sm_->multiply(false, *(splitinc[1]), mod);
  lm_temp.update(1.0, mod, 1.0);

  // + A_sn*d_n
  a_sn_->multiply(false, *(splitinc[0]), mod);
  lm_temp.update(1.0, mod, 1.0);

  // - D^(-1) *
  d_inv_->multiply(false, lm_temp, *lm_);
  lm_->scale(-1.0);
}

/*-------------------------------------------------------*/
/*  Solve ALE mesh sliding problem           wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
int ALE::Meshsliding::solve_meshtying(Core::LinAlg::Solver& solver,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& disi,
    std::shared_ptr<Core::LinAlg::Vector<double>> residual,
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("Meshsliding:  3)   Solve ALE mesh sliding problem");

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);

  std::shared_ptr<Core::LinAlg::SparseMatrix> mergedmatrix = nullptr;

  mergedmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(*mergedmap_, 108, false, true);

  int errorcode = 0;

  {
    TEUCHOS_FUNC_TIME_MONITOR("Meshsliding:  3.1)   - Preparation");

    mergedmatrix = sysmatnew->merge();
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("Meshsliding:  3.2)   - Solve");

    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    errorcode = solver_.solve(mergedmatrix, disi, residual, solver_params);

    recover(disi);
  }
  return errorcode;
}

FOUR_C_NAMESPACE_CLOSE
