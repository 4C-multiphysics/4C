// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_fluidfluidmonolithic_structuresplit_nonox.hpp"

#include "4C_adapter_ale_xffsi.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_input.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidFluidMonolithicStructureSplitNoNOX::FluidFluidMonolithicStructureSplitNoNOX(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : MonolithicNoNOX(comm, timeparams)
{
  // Throw an error if there are DBCs on structural interface DOFs.
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
  intersectionmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(structure_field()->interface()->fsi_cond_map());
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  if (intersectionmap->num_global_elements() != 0)
  {
    std::stringstream errormsg;
    errormsg << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  |                DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE   "
                "              |"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  | NOTE: The slave side of the interface is not allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | This is a structure split scheme. Hence, master and slave field are chosen as "
                "follows:      |"
             << std::endl
             << "  |     MASTER  = FLUID                                                           "
                "              |"
             << std::endl
             << "  |     SLAVE   = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | Dirichlet boundary conditions were detected on slave interface degrees of "
                "freedom. Please   |"
             << std::endl
             << "  | remove Dirichlet boundary conditions from the slave side of the FSI "
                "interface.              |"
             << std::endl
             << "  | Only the master side of the FSI interface is allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl;

    FOUR_C_THROW("{}", errormsg.str());
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check if removing Dirichlet conditions was successful
  intersectionmaps.resize(0);
  intersectionmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(structure_field()->interface()->fsi_cond_map());
  intersectionmap = Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);
  if (intersectionmap->num_global_elements() != 0)
    FOUR_C_THROW(
        "Could not remove structural interface Dirichlet conditions from structure DBC map.");
#endif

  sggtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  sgitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  sigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  aigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmiitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fsaigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fsmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  // Recovering of Lagrange multiplier happens on structure field
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map());
  ddiinc_ = nullptr;
  solipre_ = nullptr;
  ddginc_ = nullptr;
  solgpre_ = nullptr;
  ddgpred_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);

  return;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_system()
{
  // setup coupling
  FSI::MonolithicNoNOX::setup_system();

  // create combined map
  create_combined_dof_row_map();

  // Use normal matrix for fluid equations but build (split) mesh movement
  // linearization (if requested in the input file)
  fluid_field()->use_block_matrix(false);

  // Use split structure matrix
  structure_field()->use_block_matrix();

  // build ale system matrix in split system
  ale_field()->create_system_matrix(ale_field()->interface());

  /*----------------------------------------------------------------------*/
  // initialize systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 81, false, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_rhs(
    Core::LinAlg::Vector<double>& f, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_rhs");

  setup_vector(f, *structure_field()->rhs(), *fluid_field()->rhs(), *ale_field()->rhs(),
      fluid_field()->residual_scaling());

  if (firstcall)
  {
    // get structure matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocks =
        structure_field()->block_system_matrix();
    if (blocks == nullptr) FOUR_C_THROW("expect structure block matrix");

    // get fluid shape derivatives matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();

    // get ale matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocka =
        ale_field()->block_system_matrix();
    if (blocka == nullptr) FOUR_C_THROW("expect ale block matrix");

    // extract structure and ale submatrices
    Core::LinAlg::SparseMatrix& sig = blocks->matrix(0, 1);  // S_{I\Gamma}
    Core::LinAlg::SparseMatrix& sgg = blocks->matrix(1, 1);  // S_{\Gamma\Gamma}
    Core::LinAlg::SparseMatrix& aig = blocka->matrix(0, 1);  // A_{I\Gamma}

    // get time integration parameters of structure an fluid time integrators
    // to enable consistent time integration among the fields
    const double stiparam = structure_field()->tim_int_param();
    const double ftiparam = fluid_field()->tim_int_param();

    // some scaling factors for fluid
    const double scale = fluid_field()->residual_scaling();
    const double dt = fluid_field()->dt();

    // some often re-used vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs = nullptr;

    // ---------- inner structural DOFs
    /* The following terms are added to the inner structural DOFs of right hand side:
     *
     * rhs_firstnewtonstep =
     *
     * (1)  S_{I\Gamma} * \Delta d_{\Gamma,p}
     *
     * (2)  - dt * S_{I\Gamma} * u^{n}_{\Gamma}
     *
     *  Remarks on all terms:
     * +  tau: time scaling factor for interface time integration (tau =
     * 1/fluid_field()->TimeScaling())
     *
     */

    // ----------addressing term 1
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.row_map(), true);
    sig.Apply(*ddgpred_, *rhs);

    extractor().add_vector(*rhs, 0, f);

    // ----------addressing term 2
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.row_map(), true);
    std::shared_ptr<Core::LinAlg::Vector<double>> fveln = fluid_field()->extract_interface_veln();
    sig.Apply(*fluid_to_struct(fveln), *rhs);
    rhs->scale(-dt);

    extractor().add_vector(*rhs, 0, f);

    // ---------- end of inner structural DOFs

    // we need a temporary vector with the whole fluid dofs where we
    // can insert the embedded dofrowmap into it
    // kruse 30.04. --> we don't, see adapter
    // std::shared_ptr<Core::LinAlg::Vector<double>> fluidfluidtmp =
    // std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(),true);

    // ---------- inner fluid DOFs
    /* The following terms are added to the inner fluid DOFs of right hand side:
     *
     * rhs_firstnewtonstep =
     *
     * (1)  - dt * F^{G}_{I\Gamma} * u^{n}_{\Gamma}
     *
     */
    // ----------addressing term 1
    if (mmm != nullptr)
    {
      Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.row_map(), true);
      fmig.Apply(*fveln, *rhs);
      rhs->scale(-dt);

      rhs = fluid_field()->interface()->insert_other_vector(*rhs);
      extractor().add_vector(*rhs, 1, f);
    }

    // ---------- end of inner fluid DOFs

    // ---------- interface fluid DOFs
    /* The following terms are added to the interface fluid DOFs of right hand side:
     *
     * rhs_firstnewtonstep =
     *
     * (1)  - dt * F^{G}_{\Gamma\Gamma} * u^{n}_{\Gamma}
     *
     * (2)  - dt * (1-ftiparam)/(1-stiparam) * S_{\Gamma\Gamma} * u^{n}_{\Gamma}
     *
     * (3)  + (1-ftiparam)/(1-stiparam) * S_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
     *
     * Remarks on all terms:
     * +  tau: time scaling factor for interface time integration (tau =
     * 1/fluid_field()->TimeScaling())
     *
     */

    // ----------addressing term 1
    if (mmm != nullptr)
    {
      Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);
      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.row_map(), true);
      fmgg.Apply(*fveln, *rhs);
      rhs->scale(-dt);
      rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);
      extractor().add_vector(*rhs, 1, f);
    }

    // ----------addressing term 2
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(sgg.row_map(), true);
    sgg.Apply(*fluid_to_struct(fveln), *rhs);
    rhs->scale(-dt * (1. - ftiparam) / ((1 - stiparam) * scale));

    rhs = struct_to_fluid(rhs);
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);
    extractor().add_vector(*rhs, 1, f);

    // ----------addressing term 3
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(sgg.row_map(), true);
    sgg.Apply(*ddgpred_, *rhs);
    rhs->scale((1. - ftiparam) / ((1 - stiparam) * scale));

    rhs = struct_to_fluid(rhs);
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);
    extractor().add_vector(*rhs, 1, f);

    // ----------addressing term 1
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.row_map(), true);

    aig.Apply(*fluid_to_ale_interface(fveln), *rhs);
    rhs->scale(-dt);
    extractor().add_vector(*rhs, 2, f);
    // ---------- end of inner ale DOFs
  }

  // -----------------------------------------------------
  // Reset quantities for previous iteration step since they still store values from the last time
  // step
  ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->other_map(), true);
  solipre_ = nullptr;
  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  solgpre_ = nullptr;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_system_matrix()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_system_matrix");

  // extract Jacobian matrices and put them into composite system
  // matrix W

  const Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();
  const Coupling::Adapter::Coupling& icoupfa = interface_fluid_ale_coupling();

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> s = structure_field()->block_system_matrix();
  if (s == nullptr) FOUR_C_THROW("expect structure block matrix");
  std::shared_ptr<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();
  if (f == nullptr) FOUR_C_THROW("expect fluid matrix");
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();
  if (a == nullptr) FOUR_C_THROW("expect ale block matrix");

  Core::LinAlg::SparseMatrix& aii = a->matrix(0, 0);
  Core::LinAlg::SparseMatrix& aig = a->matrix(0, 1);

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure an fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // store parts of structural matrix to know them in the next iteration as previous iteration
  // matricesu
  sgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 0));
  sggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 1));

  /*----------------------------------------------------------------------*/
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // Incomplete fluid matrix to be able to deal with slightly defective
  // interface meshes.
  f->un_complete();

  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, s->matrix(0, 0));

  (*sigtransform_)(s->full_row_map(), s->full_col_map(), s->matrix(0, 1), 1. / timescale,
      Coupling::Adapter::CouplingMasterConverter(coupsf), systemmatrix_->matrix(0, 1));
  (*sggtransform_)(s->matrix(1, 1),
      ((1.0 - ftiparam) / (1.0 - stiparam)) * (1. / (scale * timescale)),
      Coupling::Adapter::CouplingMasterConverter(coupsf),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *f, true, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> lsgi =
      std::make_shared<Core::LinAlg::SparseMatrix>(f->row_map(), 81, false);
  (*sgitransform_)(s->matrix(1, 0), ((1.0 - ftiparam) / (1.0 - stiparam)) * (1. / scale),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *lsgi);

  lsgi->complete(s->matrix(1, 0).domain_map(), f->range_map());

  // systemmatrix_->Assign(1,1,View,*f);
  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, *lsgi);

  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1. / timescale,
      Coupling::Adapter::CouplingSlaveConverter(icoupfa), systemmatrix_->matrix(2, 1));

  systemmatrix_->assign(2, 2, Core::LinAlg::DataAccess::Share, aii);

  /*----------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != nullptr)
  {
    Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);
    Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    Core::LinAlg::matrix_add(fmgg, false, 1. / timescale, systemmatrix_->matrix(1, 1), 1.0);
    Core::LinAlg::matrix_add(fmig, false, 1. / timescale, systemmatrix_->matrix(1, 1), 1.0);

    Core::LinAlg::SparseMatrix lfmgi(f->row_map(), 81, false);
    (*fmgitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmgi, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa),
        // systemmatrix_->Matrix(1,2),
        lfmgi, false, false);

    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), lfmgi, false, true);

    lfmgi.complete(aii.domain_map(), f->range_map());

    systemmatrix_->assign(1, 2, Core::LinAlg::DataAccess::Share, lfmgi);
  }

  f->complete();

  // finally assign fluid block
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *f);

  // done. make sure all blocks are filled.
  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::initial_guess(
    std::shared_ptr<Core::LinAlg::Vector<double>> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::initial_guess");

  setup_vector(*ig, *structure_field()->initial_guess(), *fluid_field()->initial_guess(),
      *ale_field()->initial_guess(), 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  // should we scale the system?
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    // TODO: The matrices are modified here. Do we have to change them back later on?

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    A_00.inv_row_sums(*srowsum_);
    A_00.inv_col_sums(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(0, 2).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_22 = mat.matrix(2, 2);
    arowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_22.row_map(), false);
    acolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_22.row_map(), false);
    A_22.inv_row_sums(*arowsum_);
    A_22.inv_col_sums(*acolsum_);
    A_22.left_scale(*arowsum_);
    A_22.right_scale(*acolsum_);
    mat.matrix(2, 0).left_scale(*arowsum_);
    mat.matrix(2, 1).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ax = extractor().extract_vector(b, 2);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);
    ax->multiply(1.0, *arowsum_, *ax, 0.0);

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);
  }
}

/*----------------------------------------------------------------------*
 |  map containing the dofs with Dirichlet BC
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> FSI::FluidFluidMonolithicStructureSplitNoNOX::combined_dbc_map()
{
  // Create a combined map vector with the 3 field DBC maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> alldbcmaps;

  // structure DBC
  alldbcmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  // fluid DBC
  alldbcmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // ALE-DBC
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->interface()->other_map());
  std::shared_ptr<Core::LinAlg::Map> aleintersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(aleintersectionmaps);
  alldbcmaps.push_back(aleintersectionmap);

  // Merge the maps
  std::shared_ptr<Core::LinAlg::Map> alldbcmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(alldbcmaps);

  return alldbcmap;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::unscale_solution(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> sy = extractor().extract_vector(x, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ay = extractor().extract_vector(x, 2);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);
    ay->multiply(1.0, *acolsum_, *ay, 0.0);

    extractor().insert_vector(*sy, 0, x);
    extractor().insert_vector(*ay, 2, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ax = extractor().extract_vector(b, 2);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);
    ax->reciprocal_multiply(1.0, *arowsum_, *ax, 0.0);

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(0, 2).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_22 = mat.matrix(2, 2);
    arowsum_->reciprocal(*arowsum_);
    acolsum_->reciprocal(*acolsum_);
    A_22.left_scale(*arowsum_);
    A_22.right_scale(*acolsum_);
    mat.matrix(2, 0).left_scale(*arowsum_);
    mat.matrix(2, 1).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_vector(Core::LinAlg::Vector<double>& f,
    const Core::LinAlg::Vector<double>& sv, const Core::LinAlg::Vector<double>& fv,
    const Core::LinAlg::Vector<double>& av, double fluidscale)
{
  // get time integration parameters of structure an fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // structure inner
  std::shared_ptr<Core::LinAlg::Vector<double>> sov =
      structure_field()->interface()->extract_other_vector(sv);

  // ale inner
  std::shared_ptr<Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);

  // add fluid interface values to structure vector
  // scv: structure fsi dofs
  std::shared_ptr<Core::LinAlg::Vector<double>> scv =
      structure_field()->interface()->extract_fsi_cond_vector(sv);

  if (fabs(fluidscale) > 1e-15)
  {
    // modfv: whole embedded fluid map but entries at fsi dofs
    std::shared_ptr<Core::LinAlg::Vector<double>> modfv =
        fluid_field()->interface()->insert_fsi_cond_vector(*struct_to_fluid(scv));

    // modfv = modfv * 1/fluidscale * d/b
    modfv->scale((1. / fluidscale) * (1.0 - ftiparam) / (1.0 - stiparam));

    // add contribution of Lagrange multiplier from previous time step
    if (lambda_ != nullptr)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> lambdaglobal =
          fluid_field()->interface()->insert_fsi_cond_vector(*struct_to_fluid(lambda_));
      modfv->update((-ftiparam + stiparam * (1.0 - ftiparam) / (1.0 - stiparam)) / fluidscale,
          *lambdaglobal, 1.0);
    }

    modfv->update(1.0, fv, 1.0);
    extractor().insert_vector(*modfv, 1, f);
  }
  else
  {
    extractor().insert_vector(fv, 1, f);
  }

  extractor().insert_vector(*sov, 0, f);
  extractor().insert_vector(*aov, 2, f);
}

/*----------------------------------------------------------------------
* - Called from evaluate() method in Newton-loop with x=x_sum_
*   (increment sum)
* - Field contributions sx,fx,ax are recovered from x
----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == nullptr)
  {
    FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
  }
#endif

  // ----------------------
  // process fluid unknowns
  fx = extractor().extract_vector(*x, 1);
  std::shared_ptr<Core::LinAlg::Vector<double>> fcx =
      fluid_field()->interface()->extract_fsi_cond_vector(*fx);

  // ----------------------
  // process ale unknowns
  std::shared_ptr<Core::LinAlg::Vector<double>> fcxforale =
      std::make_shared<Core::LinAlg::Vector<double>>(*fcx);
  fluid_field()->velocity_to_displacement(fcxforale);
  std::shared_ptr<Core::LinAlg::Vector<double>> acx = fluid_to_struct(fcxforale);
  acx = struct_to_ale(acx);

  std::shared_ptr<const Core::LinAlg::Vector<double>> aox = extractor().extract_vector(*x, 2);
  // std::shared_ptr<Core::LinAlg::Vector<double>> acx = StructToAle(scx);

  std::shared_ptr<Core::LinAlg::Vector<double>> a =
      ale_field()->interface()->insert_other_vector(*aox);
  ale_field()->interface()->insert_fsi_cond_vector(*acx, *a);

  ax = a;

  // ----------------------
  // process structure unknowns
  std::shared_ptr<const Core::LinAlg::Vector<double>> sox = extractor().extract_vector(*x, 0);

  // std::shared_ptr<Core::LinAlg::Vector<double>> scx = fluid_to_struct(fcx);
  // convert ALE interface displacements to structure interface displacements
  std::shared_ptr<Core::LinAlg::Vector<double>> scx = ale_to_struct(acx);
  scx->update(-1.0, *ddgpred_, 1.0);

  std::shared_ptr<Core::LinAlg::Vector<double>> s =
      structure_field()->interface()->insert_other_vector(*sox);
  structure_field()->interface()->insert_fsi_cond_vector(*scx, *s);
  sx = s;

  // IMPORTANT:
  // you can get these increments in a similar way like in fluidsplit, without saving the
  // previous variables. This is important if you are considering the whole term of
  // lagrange-multiplier.
  // ----------------------
  // Store field vectors to know them later on as previous quantities
  if (solipre_ != nullptr)
    ddiinc_->update(1.0, *sox, -1.0, *solipre_, 0.0);  // compute current iteration increment
  else
    ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*sox);  // first iteration increment

  solipre_ = sox;  // store current step increment

  if (solgpre_ != nullptr)
    ddginc_->update(1.0, *scx, -1.0, *solgpre_, 0.0);  // compute current iteration increment
  else
    ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*scx);  // first iteration increment

  solgpre_ = scx;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::output()
{
  structure_field()->output();

  // output Lagrange multiplier
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        structure_field()->interface()->insert_fsi_cond_vector(*lambda_);

    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const int uprestart = fsidyn.get<int>("RESTARTEVERY");
    const int upres = fsidyn.get<int>("RESULTSEVERY");
    if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
        fluid_field()->step() % upres == 0)
      structure_field()->disc_writer()->write_vector("fsilambda", lambdafull);
  }

  fluid_field()->output();
  ale_field()->output();

  if (structure_field()->get_constraint_manager()->have_monitor())
  {
    structure_field()->get_constraint_manager()->compute_monitor_values(
        *structure_field()->dispnp());
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      structure_field()->get_constraint_manager()->print_monitor_values();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::read_restart(int step)
{
  // read Lagrange multiplier
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
    Core::IO::DiscretizationReader reader =
        Core::IO::DiscretizationReader(*structure_field()->discretization(),
            Global::Problem::instance()->input_control_file(), step);
    reader.read_vector(lambdafull, "fsilambda");
    lambda_ = structure_field()->interface()->extract_fsi_cond_vector(*lambdafull);
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::create_combined_dof_row_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::SetupNewSystem()");

  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;
  vecSpaces.push_back(structure_field()->interface()->other_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->interface()->other_map());

  if (vecSpaces[0]->num_global_elements() == 0)
    FOUR_C_THROW("No inner structural equations. Splitting not possible. Panic.");

  set_dof_row_maps(vecSpaces);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::build_convergence_norms()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "FSI::FluidFluidMonolithicStructureSplitNoNOX::build_convergence_norms()");
  //----------------------------
  // build residual norms
  rhs_->norm_2(&normrhs_);

  // structural Dofs
  structure_field()
      ->interface()
      ->extract_other_vector(*structure_field()->rhs())
      ->norm_2(&normstrrhsL2_);
  structure_field()
      ->interface()
      ->extract_other_vector(*structure_field()->rhs())
      ->norm_inf(&normstrrhsInf_);

  // extract fluid Dofs
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs = extractor().extract_vector(*rhs_, 1);

  fluid_field()->interface()->extract_fsi_cond_vector(*rhs)->norm_2(&norminterfacerhsL2_);
  fluid_field()->interface()->extract_fsi_cond_vector(*rhs)->norm_inf(&norminterfacerhsInf_);

  // inner fluid velocity Dofs: inner velocity dofs without db-condition
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> innerfluidvel;
  innerfluidvel.push_back(fluid_field()->inner_velocity_row_map());
  innerfluidvel.push_back(nullptr);
  Core::LinAlg::MultiMapExtractor fluidvelextract(*(fluid_field()->dof_row_map()), innerfluidvel);
  fluidvelextract.extract_vector(*fluid_field()->rhs(), 0)->norm_2(&normflvelrhsL2_);
  fluidvelextract.extract_vector(*fluid_field()->rhs(), 0)->norm_inf(&normflvelrhsInf_);

  // fluid pressure Dofs: pressure dofs (at interface and inner) without db-condition
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidpres;
  fluidpres.push_back(fluid_field()->pressure_row_map());
  fluidpres.push_back(nullptr);
  Core::LinAlg::MultiMapExtractor fluidpresextract(*(fluid_field()->dof_row_map()), fluidpres);
  fluidpresextract.extract_vector(*fluid_field()->rhs(), 0)->norm_2(&normflpresrhsL2_);
  fluidpresextract.extract_vector(*fluid_field()->rhs(), 0)->norm_inf(&normflpresrhsInf_);

  // ale
  ale_field()->rhs()->norm_2(&normalerhsL2_);
  //-------------------------------
  // build solution increment norms

  // build increment norm
  iterinc_->norm_2(&norminc_);

  // structural Dofs
  extractor().extract_vector(*iterinc_, 0)->norm_2(&normstrincL2_);
  extractor().extract_vector(*iterinc_, 0)->norm_inf(&normstrincInf_);

  // interface
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc =
      extractor().extract_vector(*iterinc_, 1);
  fluid_field()->interface()->extract_fsi_cond_vector(*inc)->norm_2(&norminterfaceincL2_);
  fluid_field()->interface()->extract_fsi_cond_vector(*inc)->norm_inf(&norminterfaceincInf_);

  // inner fluid velocity Dofs
  fluidvelextract.extract_vector(*extractor().extract_vector(*iterinc_, 1), 0)
      ->norm_2(&normflvelincL2_);
  fluidvelextract.extract_vector(*extractor().extract_vector(*iterinc_, 1), 0)
      ->norm_inf(&normflvelincInf_);

  // fluid pressure Dofs
  fluidpresextract.extract_vector(*extractor().extract_vector(*iterinc_, 1), 0)
      ->norm_2(&normflpresincL2_);
  fluidpresextract.extract_vector(*extractor().extract_vector(*iterinc_, 1), 0)
      ->norm_inf(&normflpresincInf_);

  // ale
  extractor().extract_vector(*iterinc_, 2)->norm_2(&normaleincL2_);

  // get length of the structural, fluid and ale vector
  ns_ = (*(structure_field()->rhs())).global_length();  // structure
  ni_ =
      (*(fluid_field()->interface()->extract_fsi_cond_vector(*rhs))).global_length();  // fluid fsi
  nf_ = (*(fluid_field()->rhs())).global_length();  // fluid inner
  nfv_ = (*(fluidvelextract.extract_vector(*fluid_field()->rhs(), 0)))
             .global_length();  // fluid velocity
  nfp_ = (*(fluidpresextract.extract_vector(*fluid_field()->rhs(), 0)))
             .global_length();                    // fluid pressure
  na_ = (*(ale_field()->rhs())).global_length();  // ale
  nall_ = (*rhs_).global_length();                // all
}

/*----------------------------------------------------------------------*/
/* Recover the Lagrange multiplier at the interface                     */
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::recover_lagrange_multiplier()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "FSI::FluidFluidMonolithicStructureSplitNoNOX::recover_lagrange_multiplier");

  // get time integration parameters of structural time integrator
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
      nullptr;  // stores intermediate result of terms (3)-(8)
  std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;     // just for convenience
  std::shared_ptr<Core::LinAlg::Vector<double>> auxauxvec = nullptr;  // just for convenience

  /* Recovery of Lagrange multiplier lambda^{n+1} is done by the following
   * condensation expression:
   *
   * lambda^{n+1} =
   *
   * (1)  - stiparam / (1.-stiparam) * lambda^{n}
   *
   * (2)  + 1. / (1.-stiparam) * tmpvec
   *
   * with tmpvec =
   *
   * (3)    r_{\Gamma}^{S,n+1}
   *
   * (4)  + S_{\Gamma I} * \Delta d_{I}^{S,n+1}
   *
   * (5)  + tau * S_{\Gamma\Gamma} * \Delta u_{\Gamma}^{F,n+1}
   *
   * (6)  + dt * S_{\Gamma\Gamma} * u_{\Gamma}^n]
   *
   * Remark on term (6):
   * Term (6) has to be considered only in the first Newton iteration.
   * Hence, it will usually not be computed since in general we need more
   * than one nonlinear iteration until convergence.
   *
   * Remarks on all terms:
   * +  Division by (1.0 - stiparam) will be done in the end
   *    since this is common to all terms
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   * +  neglecting terms (4)-(5) should not alter the results significantly
   *    since at the end of the time step the solution increments tend to zero.
   *
   *                                                 Matthias Mayr (10/2012)
   */

  // ---------Addressing term (1)
  lambda_->update(-stiparam, *lambda_, 0.0);
  // ---------End of term (1)

  // ---------Addressing term (3)
  std::shared_ptr<Core::LinAlg::Vector<double>> structureresidual =
      structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->rhs());
  structureresidual->scale(-1.0);  // invert sign to obtain residual, not rhs
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(*structureresidual);
  // ---------End of term (3)

  /* You might want to comment out terms (4) to (6) since they tend to
   * introduce oscillations in the Lagrange multiplier field for certain
   * material properties of the structure.
   *                                                    Matthias Mayr 11/2012
  // ---------Addressing term (4)
  auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sgicur_->RangeMap(),true));
  sgicur_->Apply(*ddiinc_,*auxvec);
  tmpvec->Update(1.0,*auxvec,1.0);
  // ---------End of term (4)

  // ---------Addressing term (5)
  auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggcur_->RangeMap(),true));
  sggcur_->Apply(*ddginc_,*auxvec);
  tmpvec->Update(1.0/timescale,*auxvec,1.0);
  // ---------End of term (5)

  //---------Addressing term (6)
  if (firstcall_)
  {
    auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggprev_->RangeMap(),true));
    sggprev_->Apply(*fluid_to_struct(fluid_field()->extract_interface_veln()),*auxvec);
    tmpvec->Update(Dt(),*auxvec,1.0);
  }
  // ---------End of term (6)
  *
  */

  // ---------Addressing term (2)
  lambda_->update(1.0, *tmpvec, 1.0);
  // ---------End of term (2)

  // finally, divide by -(1.-stiparam) which is common to all terms
  lambda_->scale(1. / (1.0 - stiparam));

  // Finally, the Lagrange multiplier 'lambda_' is recovered here.
  // It represents nodal forces acting onto the structure.

  return;
}

void FSI::FluidFluidMonolithicStructureSplitNoNOX::handle_fluid_dof_map_change_in_newton()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    Core::IO::cout << " New Map!! " << Core::IO::endl;

  // save the old x_sum
  std::shared_ptr<Core::LinAlg::Vector<double>> x_sum_n =
      std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  *x_sum_n = *x_sum_;
  std::shared_ptr<const Core::LinAlg::Vector<double>> sx_n;
  std::shared_ptr<const Core::LinAlg::Vector<double>> ax_n;
  sx_n = extractor().extract_vector(*x_sum_n, 0);
  ax_n = extractor().extract_vector(*x_sum_n, 2);

  // rebuild combined dof-map
  create_combined_dof_row_map();
  // re-initialize systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 81, false, true);

  rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  iterinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  x_sum_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // build the new iter_sum
  extractor().insert_vector(*sx_n, 0, *x_sum_);
  extractor().insert_vector(*fluid_field()->stepinc(), 1, *x_sum_);
  extractor().insert_vector(*ax_n, 2, *x_sum_);
  nf_ = (*(fluid_field()->rhs())).global_length();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::FluidFluidMonolithicStructureSplitNoNOX::has_fluid_dof_map_changed(
    const Core::LinAlg::Map& fluidincrementmap)
{
  return !fluid_field()->dof_row_map()->same_as(fluidincrementmap);
}

FOUR_C_NAMESPACE_CLOSE
