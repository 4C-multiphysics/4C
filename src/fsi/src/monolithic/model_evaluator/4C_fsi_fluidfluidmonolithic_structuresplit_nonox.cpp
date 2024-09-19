/*----------------------------------------------------------------------*/
/*! \file
\brief Control routine for monolithic fluid-fluid-fsi (structuresplit)
using XFEM

\level 3


*----------------------------------------------------------------------*/
#include "4C_fsi_fluidfluidmonolithic_structuresplit_nonox.hpp"

#include "4C_adapter_ale_xffsi.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_debugwriter.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ale.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidFluidMonolithicStructureSplitNoNOX::FluidFluidMonolithicStructureSplitNoNOX(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : MonolithicNoNOX(comm, timeparams)
{
  // Throw an error if there are DBCs on structural interface DOFs.
  std::vector<Teuchos::RCP<const Epetra_Map>> intersectionmaps;
  intersectionmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(structure_field()->interface()->fsi_cond_map());
  Teuchos::RCP<Epetra_Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  if (intersectionmap->NumGlobalElements() != 0)
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

    FOUR_C_THROW(errormsg.str());
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check if removing Dirichlet conditions was successful
  intersectionmaps.resize(0);
  intersectionmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(structure_field()->interface()->fsi_cond_map());
  intersectionmap = Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);
  if (intersectionmap->NumGlobalElements() != 0)
    FOUR_C_THROW(
        "Could not remove structural interface Dirichlet conditions from structure DBC map.");
#endif

  sggtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowColTransform);
  sgitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowTransform);
  sigtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  aigtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  fmiitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  fmgitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  fsaigtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  fsmgitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);

  // Recovering of Lagrange multiplier happens on structure field
  lambda_ = Teuchos::rcp(new Epetra_Vector(*structure_field()->interface()->fsi_cond_map()));
  ddiinc_ = Teuchos::null;
  solipre_ = Teuchos::null;
  ddginc_ = Teuchos::null;
  solgpre_ = Teuchos::null;
  ddgpred_ = Teuchos::rcp(new Epetra_Vector(*structure_field()->interface()->fsi_cond_map(), true));

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

  // Use normal matrix for fluid equations but build (splitted) mesh movement
  // linearization (if requested in the input file)
  fluid_field()->use_block_matrix(false);

  // Use splitted structure matrix
  structure_field()->use_block_matrix();

  // build ale system matrix in splitted system
  ale_field()->create_system_matrix(ale_field()->interface());

  /*----------------------------------------------------------------------*/
  // initialize systemmatrix_
  systemmatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          extractor(), extractor(), 81, false, true));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_rhs(Epetra_Vector& f, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_rhs");

  setup_vector(f, structure_field()->rhs(), fluid_field()->rhs(), ale_field()->rhs(),
      fluid_field()->residual_scaling());

  if (firstcall)
  {
    // get structure matrix
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> blocks =
        structure_field()->block_system_matrix();
    if (blocks == Teuchos::null) FOUR_C_THROW("expect structure block matrix");

    // get fluid shape derivatives matrix
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();

    // get ale matrix
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> blocka = ale_field()->block_system_matrix();
    if (blocka == Teuchos::null) FOUR_C_THROW("expect ale block matrix");

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
    Teuchos::RCP<Epetra_Vector> rhs = Teuchos::null;

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
    rhs = Teuchos::rcp(new Epetra_Vector(sig.row_map(), true));
    sig.Apply(*ddgpred_, *rhs);

    extractor().add_vector(*rhs, 0, f);

    // ----------addressing term 2
    rhs = Teuchos::rcp(new Epetra_Vector(sig.row_map(), true));
    Teuchos::RCP<Epetra_Vector> fveln = fluid_field()->extract_interface_veln();
    sig.Apply(*fluid_to_struct(fveln), *rhs);
    rhs->Scale(-dt);

    extractor().add_vector(*rhs, 0, f);

    // ---------- end of inner structural DOFs

    // we need a temporary vector with the whole fluid dofs where we
    // can insert the embedded dofrowmap into it
    // kruse 30.04. --> we don't, see adapter
    // Teuchos::RCP<Epetra_Vector> fluidfluidtmp =
    // Core::LinAlg::create_vector(*fluid_field()->dof_row_map(),true);

    // ---------- inner fluid DOFs
    /* The following terms are added to the inner fluid DOFs of right hand side:
     *
     * rhs_firstnewtonstep =
     *
     * (1)  - dt * F^{G}_{I\Gamma} * u^{n}_{\Gamma}
     *
     */
    // ----------addressing term 1
    if (mmm != Teuchos::null)
    {
      Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
      rhs = Teuchos::rcp(new Epetra_Vector(fmig.row_map(), true));
      fmig.Apply(*fveln, *rhs);
      rhs->Scale(-dt);

      rhs = fluid_field()->interface()->insert_other_vector(rhs);
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
    if (mmm != Teuchos::null)
    {
      Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);
      rhs = Teuchos::rcp(new Epetra_Vector(fmgg.row_map(), true));
      fmgg.Apply(*fveln, *rhs);
      rhs->Scale(-dt);
      rhs = fluid_field()->interface()->insert_fsi_cond_vector(rhs);
      extractor().add_vector(*rhs, 1, f);
    }

    // ----------addressing term 2
    rhs = Teuchos::rcp(new Epetra_Vector(sgg.row_map(), true));
    sgg.Apply(*fluid_to_struct(fveln), *rhs);
    rhs->Scale(-dt * (1. - ftiparam) / ((1 - stiparam) * scale));

    rhs = struct_to_fluid(rhs);
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(rhs);
    extractor().add_vector(*rhs, 1, f);

    // ----------addressing term 3
    rhs = Teuchos::rcp(new Epetra_Vector(sgg.row_map(), true));
    sgg.Apply(*ddgpred_, *rhs);
    rhs->Scale((1. - ftiparam) / ((1 - stiparam) * scale));

    rhs = struct_to_fluid(rhs);
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(rhs);
    extractor().add_vector(*rhs, 1, f);

    // ----------addressing term 1
    rhs = Teuchos::rcp(new Epetra_Vector(aig.row_map(), true));

    aig.Apply(*fluid_to_ale_interface(fveln), *rhs);
    rhs->Scale(-dt);
    extractor().add_vector(*rhs, 2, f);
    // ---------- end of inner ale DOFs
  }

  // -----------------------------------------------------
  // Reset quantities for previous iteration step since they still store values from the last time
  // step
  ddiinc_ = Core::LinAlg::create_vector(*structure_field()->interface()->other_map(), true);
  solipre_ = Teuchos::null;
  ddginc_ = Core::LinAlg::create_vector(*structure_field()->interface()->fsi_cond_map(), true);
  solgpre_ = Teuchos::null;
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

  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> s = structure_field()->block_system_matrix();
  if (s == Teuchos::null) FOUR_C_THROW("expect structure block matrix");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();
  if (f == Teuchos::null) FOUR_C_THROW("expect fluid matrix");
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();
  if (a == Teuchos::null) FOUR_C_THROW("expect ale block matrix");

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
  sgicur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->matrix(1, 0)));
  sggcur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->matrix(1, 1)));

  /*----------------------------------------------------------------------*/
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // Uncomplete fluid matrix to be able to deal with slightly defective
  // interface meshes.
  f->un_complete();

  systemmatrix_->assign(0, 0, Core::LinAlg::View, s->matrix(0, 0));

  (*sigtransform_)(s->full_row_map(), s->full_col_map(), s->matrix(0, 1), 1. / timescale,
      Coupling::Adapter::CouplingMasterConverter(coupsf), systemmatrix_->matrix(0, 1));
  (*sggtransform_)(s->matrix(1, 1),
      ((1.0 - ftiparam) / (1.0 - stiparam)) * (1. / (scale * timescale)),
      Coupling::Adapter::CouplingMasterConverter(coupsf),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *f, true, true);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> lsgi =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(f->row_map(), 81, false));
  (*sgitransform_)(s->matrix(1, 0), ((1.0 - ftiparam) / (1.0 - stiparam)) * (1. / scale),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *lsgi);

  lsgi->complete(s->matrix(1, 0).domain_map(), f->range_map());

  // systemmatrix_->Assign(1,1,View,*f);
  systemmatrix_->assign(1, 0, Core::LinAlg::View, *lsgi);

  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1. / timescale,
      Coupling::Adapter::CouplingSlaveConverter(icoupfa), systemmatrix_->matrix(2, 1));

  systemmatrix_->assign(2, 2, Core::LinAlg::View, aii);

  /*----------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != Teuchos::null)
  {
    Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);
    Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    systemmatrix_->matrix(1, 1).add(fmgg, false, 1. / timescale, 1.0);
    systemmatrix_->matrix(1, 1).add(fmig, false, 1. / timescale, 1.0);

    Teuchos::RCP<Core::LinAlg::SparseMatrix> lfmgi =
        Teuchos::rcp(new Core::LinAlg::SparseMatrix(f->row_map(), 81, false));
    (*fmgitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmgi, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa),
        // systemmatrix_->Matrix(1,2),
        *lfmgi, false, false);

    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), *lfmgi, false, true);

    lfmgi->complete(aii.domain_map(), f->range_map());

    systemmatrix_->assign(1, 2, Core::LinAlg::View, *lfmgi);
  }

  f->complete();

  // finally assign fluid block
  systemmatrix_->assign(1, 1, Core::LinAlg::View, *f);

  // done. make sure all blocks are filled.
  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::initial_guess(Teuchos::RCP<Epetra_Vector> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::FluidFluidMonolithicStructureSplitNoNOX::initial_guess");

  setup_vector(*ig, structure_field()->initial_guess(), fluid_field()->initial_guess(),
      ale_field()->initial_guess(), 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Epetra_Vector& b)
{
  // should we scale the system?
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    // The matrices are modified here. Do we have to change them back later on?

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.matrix(0, 0).epetra_matrix();
    srowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    scolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*srowsum_);
    A->InvColSums(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.matrix(0, 1).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 2).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(1, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(2, 0).epetra_matrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.matrix(2, 2).epetra_matrix();
    arowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    acolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*arowsum_);
    A->InvColSums(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.matrix(2, 0).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 1).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(0, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(1, 2).epetra_matrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");

    Teuchos::RCP<Epetra_Vector> sx = extractor().extract_vector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = extractor().extract_vector(b, 2);

    if (sx->Multiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->Multiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);
  }
}

/*----------------------------------------------------------------------*
 |  map containing the dofs with Dirichlet BC
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> FSI::FluidFluidMonolithicStructureSplitNoNOX::combined_dbc_map()
{
  // Create a combined map vector with the 3 field DBC maps
  std::vector<Teuchos::RCP<const Epetra_Map>> alldbcmaps;

  // structure DBC
  alldbcmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  // fluid DBC
  alldbcmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // ALE-DBC
  std::vector<Teuchos::RCP<const Epetra_Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->interface()->other_map());
  Teuchos::RCP<Epetra_Map> aleintersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(aleintersectionmaps);
  alldbcmaps.push_back(aleintersectionmap);

  // Merge the maps
  Teuchos::RCP<Epetra_Map> alldbcmap = Core::LinAlg::MultiMapExtractor::merge_maps(alldbcmaps);

  return alldbcmap;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::unscale_solution(
    Core::LinAlg::BlockSparseMatrixBase& mat, Epetra_Vector& x, Epetra_Vector& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    Teuchos::RCP<Epetra_Vector> sy = extractor().extract_vector(x, 0);
    Teuchos::RCP<Epetra_Vector> ay = extractor().extract_vector(x, 2);

    if (sy->Multiply(1.0, *scolsum_, *sy, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ay->Multiply(1.0, *acolsum_, *ay, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sy, 0, x);
    extractor().insert_vector(*ay, 2, x);

    Teuchos::RCP<Epetra_Vector> sx = extractor().extract_vector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = extractor().extract_vector(b, 2);

    if (sx->ReciprocalMultiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->ReciprocalMultiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.matrix(0, 0).epetra_matrix();
    srowsum_->Reciprocal(*srowsum_);
    scolsum_->Reciprocal(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.matrix(0, 1).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 2).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(1, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(2, 0).epetra_matrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.matrix(2, 2).epetra_matrix();
    arowsum_->Reciprocal(*arowsum_);
    acolsum_->Reciprocal(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.matrix(2, 0).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 1).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(0, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(1, 2).epetra_matrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::setup_vector(Epetra_Vector& f,
    Teuchos::RCP<const Epetra_Vector> sv, Teuchos::RCP<const Epetra_Vector> fv,
    Teuchos::RCP<const Epetra_Vector> av, double fluidscale)
{
  // get time integration parameters of structure an fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // structure inner
  Teuchos::RCP<Epetra_Vector> sov = structure_field()->interface()->extract_other_vector(sv);

  // ale inner
  Teuchos::RCP<Epetra_Vector> aov = ale_field()->interface()->extract_other_vector(av);

  // add fluid interface values to structure vector
  // scv: structure fsi dofs
  Teuchos::RCP<Epetra_Vector> scv = structure_field()->interface()->extract_fsi_cond_vector(sv);

  if (fabs(fluidscale) > 1e-15)
  {
    // modfv: whole embedded fluid map but entries at fsi dofs
    Teuchos::RCP<Epetra_Vector> modfv =
        fluid_field()->interface()->insert_fsi_cond_vector(struct_to_fluid(scv));

    // modfv = modfv * 1/fluidscale * d/b
    modfv->Scale((1. / fluidscale) * (1.0 - ftiparam) / (1.0 - stiparam));

    // add contribution of Lagrange multiplier from previous time step
    if (lambda_ != Teuchos::null)
    {
      Teuchos::RCP<Epetra_Vector> lambdaglobal =
          fluid_field()->interface()->insert_fsi_cond_vector(struct_to_fluid(lambda_));
      modfv->Update((-ftiparam + stiparam * (1.0 - ftiparam) / (1.0 - stiparam)) / fluidscale,
          *lambdaglobal, 1.0);
    }

    modfv->Update(1.0, *fv, 1.0);
    extractor().insert_vector(*modfv, 1, f);
  }
  else
  {
    extractor().insert_vector(*fv, 1, f);
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
    Teuchos::RCP<const Epetra_Vector> x, Teuchos::RCP<const Epetra_Vector>& sx,
    Teuchos::RCP<const Epetra_Vector>& fx, Teuchos::RCP<const Epetra_Vector>& ax)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == Teuchos::null)
  {
    FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
  }
#endif

  // ----------------------
  // process fluid unknowns
  fx = extractor().extract_vector(x, 1);
  Teuchos::RCP<Epetra_Vector> fcx = fluid_field()->interface()->extract_fsi_cond_vector(fx);

  // ----------------------
  // process ale unknowns
  Teuchos::RCP<Epetra_Vector> fcxforale = Teuchos::rcp(new Epetra_Vector(*fcx));
  fluid_field()->velocity_to_displacement(fcxforale);
  Teuchos::RCP<Epetra_Vector> acx = fluid_to_struct(fcxforale);
  acx = struct_to_ale(acx);

  Teuchos::RCP<const Epetra_Vector> aox = extractor().extract_vector(x, 2);
  // Teuchos::RCP<Epetra_Vector> acx = StructToAle(scx);

  Teuchos::RCP<Epetra_Vector> a = ale_field()->interface()->insert_other_vector(aox);
  ale_field()->interface()->insert_fsi_cond_vector(acx, a);

  ax = a;

  // ----------------------
  // process structure unknowns
  Teuchos::RCP<const Epetra_Vector> sox = extractor().extract_vector(x, 0);

  // Teuchos::RCP<Epetra_Vector> scx = fluid_to_struct(fcx);
  // convert ALE interface displacements to structure interface displacements
  Teuchos::RCP<Epetra_Vector> scx = ale_to_struct(acx);
  scx->Update(-1.0, *ddgpred_, 1.0);

  Teuchos::RCP<Epetra_Vector> s = structure_field()->interface()->insert_other_vector(sox);
  structure_field()->interface()->insert_fsi_cond_vector(scx, s);
  sx = s;

  // IMPORTANT:
  // you can get these increments in a similar way like in fluidsplit, without saving the
  // previous variables. This is important if you are considering the whole term of
  // lagrange-multiplier.
  // ----------------------
  // Store field vectors to know them later on as previous quantities
  if (solipre_ != Teuchos::null)
    ddiinc_->Update(1.0, *sox, -1.0, *solipre_, 0.0);  // compute current iteration increment
  else
    ddiinc_ = Teuchos::rcp(new Epetra_Vector(*sox));  // first iteration increment

  solipre_ = sox;  // store current step increment

  if (solgpre_ != Teuchos::null)
    ddginc_->Update(1.0, *scx, -1.0, *solgpre_, 0.0);  // compute current iteration increment
  else
    ddginc_ = Teuchos::rcp(new Epetra_Vector(*scx));  // first iteration increment

  solgpre_ = scx;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::output()
{
  structure_field()->output();

  // output Lagrange multiplier
  {
    Teuchos::RCP<Epetra_Vector> lambdafull =
        structure_field()->interface()->insert_fsi_cond_vector(lambda_);

    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const int uprestart = fsidyn.get<int>("RESTARTEVRY");
    const int upres = fsidyn.get<int>("RESULTSEVRY");
    if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
        fluid_field()->step() % upres == 0)
      structure_field()->disc_writer()->write_vector("fsilambda", lambdafull);
  }

  fluid_field()->output();
  ale_field()->output();

  if (structure_field()->get_constraint_manager()->have_monitor())
  {
    structure_field()->get_constraint_manager()->compute_monitor_values(
        structure_field()->dispnp());
    if (get_comm().MyPID() == 0)
      structure_field()->get_constraint_manager()->print_monitor_values();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::read_restart(int step)
{
  // read Lagrange multiplier
  {
    Teuchos::RCP<Epetra_Vector> lambdafull =
        Teuchos::rcp(new Epetra_Vector(*structure_field()->dof_row_map(), true));
    Core::IO::DiscretizationReader reader =
        Core::IO::DiscretizationReader(structure_field()->discretization(),
            Global::Problem::instance()->input_control_file(), step);
    reader.read_vector(lambdafull, "fsilambda");
    lambda_ = structure_field()->interface()->extract_fsi_cond_vector(lambdafull);
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
  std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;
  vecSpaces.push_back(structure_field()->interface()->other_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->interface()->other_map());

  if (vecSpaces[0]->NumGlobalElements() == 0)
    FOUR_C_THROW("No inner structural equations. Splitting not possible. Panic.");

  set_dof_row_maps(vecSpaces);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicStructureSplitNoNOX::build_convergence_norms()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "FSI::FluidFluidMonolithicStructureSplitNoNOX::build_covergence_norms()");
  //----------------------------
  // build residual norms
  rhs_->Norm2(&normrhs_);

  // structural Dofs
  structure_field()
      ->interface()
      ->extract_other_vector(structure_field()->rhs())
      ->Norm2(&normstrrhsL2_);
  structure_field()
      ->interface()
      ->extract_other_vector(structure_field()->rhs())
      ->NormInf(&normstrrhsInf_);

  // extract fluid Dofs
  Teuchos::RCP<const Epetra_Vector> rhs = extractor().extract_vector(rhs_, 1);

  fluid_field()->interface()->extract_fsi_cond_vector(rhs)->Norm2(&norminterfacerhsL2_);
  fluid_field()->interface()->extract_fsi_cond_vector(rhs)->NormInf(&norminterfacerhsInf_);

  // inner fluid velocity Dofs: inner velocity dofs without db-condition
  std::vector<Teuchos::RCP<const Epetra_Map>> innerfluidvel;
  innerfluidvel.push_back(fluid_field()->inner_velocity_row_map());
  innerfluidvel.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor fluidvelextract(*(fluid_field()->dof_row_map()), innerfluidvel);
  fluidvelextract.extract_vector(fluid_field()->rhs(), 0)->Norm2(&normflvelrhsL2_);
  fluidvelextract.extract_vector(fluid_field()->rhs(), 0)->NormInf(&normflvelrhsInf_);

  // fluid pressure Dofs: pressure dofs (at interface and inner) without db-condition
  std::vector<Teuchos::RCP<const Epetra_Map>> fluidpres;
  fluidpres.push_back(fluid_field()->pressure_row_map());
  fluidpres.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor fluidpresextract(*(fluid_field()->dof_row_map()), fluidpres);
  fluidpresextract.extract_vector(fluid_field()->rhs(), 0)->Norm2(&normflpresrhsL2_);
  fluidpresextract.extract_vector(fluid_field()->rhs(), 0)->NormInf(&normflpresrhsInf_);

  // ale
  ale_field()->rhs()->Norm2(&normalerhsL2_);
  //-------------------------------
  // build solution increment norms

  // build increment norm
  iterinc_->Norm2(&norminc_);

  // structural Dofs
  extractor().extract_vector(iterinc_, 0)->Norm2(&normstrincL2_);
  extractor().extract_vector(iterinc_, 0)->NormInf(&normstrincInf_);

  // interface
  Teuchos::RCP<const Epetra_Vector> inc = extractor().extract_vector(iterinc_, 1);
  fluid_field()->interface()->extract_fsi_cond_vector(inc)->Norm2(&norminterfaceincL2_);
  fluid_field()->interface()->extract_fsi_cond_vector(inc)->NormInf(&norminterfaceincInf_);

  // inner fluid velocity Dofs
  fluidvelextract.extract_vector(extractor().extract_vector(iterinc_, 1), 0)
      ->Norm2(&normflvelincL2_);
  fluidvelextract.extract_vector(extractor().extract_vector(iterinc_, 1), 0)
      ->NormInf(&normflvelincInf_);

  // fluid pressure Dofs
  fluidpresextract.extract_vector(extractor().extract_vector(iterinc_, 1), 0)
      ->Norm2(&normflpresincL2_);
  fluidpresextract.extract_vector(extractor().extract_vector(iterinc_, 1), 0)
      ->NormInf(&normflpresincInf_);

  // ale
  extractor().extract_vector(iterinc_, 2)->Norm2(&normaleincL2_);

  // get length of the structural, fluid and ale vector
  ns_ = (*(structure_field()->rhs())).GlobalLength();                                  // structure
  ni_ = (*(fluid_field()->interface()->extract_fsi_cond_vector(rhs))).GlobalLength();  // fluid fsi
  nf_ = (*(fluid_field()->rhs())).GlobalLength();  // fluid inner
  nfv_ = (*(fluidvelextract.extract_vector(fluid_field()->rhs(), 0)))
             .GlobalLength();  // fluid velocity
  nfp_ = (*(fluidpresextract.extract_vector(fluid_field()->rhs(), 0)))
             .GlobalLength();                    // fluid pressure
  na_ = (*(ale_field()->rhs())).GlobalLength();  // ale
  nall_ = (*rhs_).GlobalLength();                // all
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
  Teuchos::RCP<Epetra_Vector> tmpvec =
      Teuchos::null;  // stores intermediate result of terms (3)-(8)
  Teuchos::RCP<Epetra_Vector> auxvec = Teuchos::null;     // just for convenience
  Teuchos::RCP<Epetra_Vector> auxauxvec = Teuchos::null;  // just for convenience

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
  lambda_->Update(-stiparam, *lambda_, 0.0);
  // ---------End of term (1)

  // ---------Addressing term (3)
  Teuchos::RCP<Epetra_Vector> structureresidual =
      structure_field()->interface()->extract_fsi_cond_vector(structure_field()->rhs());
  structureresidual->Scale(-1.0);  // invert sign to obtain residual, not rhs
  tmpvec = Teuchos::rcp(new Epetra_Vector(*structureresidual));
  // ---------End of term (3)

  /* You might want to comment out terms (4) to (6) since they tend to
   * introduce oscillations in the Lagrange multiplier field for certain
   * material properties of the structure.
   *                                                    Matthias Mayr 11/2012
  // ---------Addressing term (4)
  auxvec = Teuchos::rcp(new Epetra_Vector(sgicur_->RangeMap(),true));
  sgicur_->Apply(*ddiinc_,*auxvec);
  tmpvec->Update(1.0,*auxvec,1.0);
  // ---------End of term (4)

  // ---------Addressing term (5)
  auxvec = Teuchos::rcp(new Epetra_Vector(sggcur_->RangeMap(),true));
  sggcur_->Apply(*ddginc_,*auxvec);
  tmpvec->Update(1.0/timescale,*auxvec,1.0);
  // ---------End of term (5)

  //---------Addressing term (6)
  if (firstcall_)
  {
    auxvec = Teuchos::rcp(new Epetra_Vector(sggprev_->RangeMap(),true));
    sggprev_->Apply(*fluid_to_struct(fluid_field()->extract_interface_veln()),*auxvec);
    tmpvec->Update(Dt(),*auxvec,1.0);
  }
  // ---------End of term (6)
  *
  */

  // ---------Addressing term (2)
  lambda_->Update(1.0, *tmpvec, 1.0);
  // ---------End of term (2)

  // finally, divide by -(1.-stiparam) which is common to all terms
  lambda_->Scale(1. / (1.0 - stiparam));

  // Finally, the Lagrange multiplier 'lambda_' is recovered here.
  // It represents nodal forces acting onto the structure.

  return;
}

void FSI::FluidFluidMonolithicStructureSplitNoNOX::handle_fluid_dof_map_change_in_newton()
{
  if (get_comm().MyPID() == 0) Core::IO::cout << " New Map!! " << Core::IO::endl;

  // save the old x_sum
  Teuchos::RCP<Epetra_Vector> x_sum_n = Core::LinAlg::create_vector(*dof_row_map(), true);
  *x_sum_n = *x_sum_;
  Teuchos::RCP<const Epetra_Vector> sx_n;
  Teuchos::RCP<const Epetra_Vector> ax_n;
  sx_n = extractor().extract_vector(x_sum_n, 0);
  ax_n = extractor().extract_vector(x_sum_n, 2);

  // rebuild combined dof-map
  create_combined_dof_row_map();
  // re-initialize systemmatrix_
  systemmatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          extractor(), extractor(), 81, false, true));

  rhs_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  x_sum_ = Core::LinAlg::create_vector(*dof_row_map(), true);

  // build the new iter_sum
  extractor().insert_vector(sx_n, 0, x_sum_);
  extractor().insert_vector(fluid_field()->stepinc(), 1, x_sum_);
  extractor().insert_vector(ax_n, 2, x_sum_);
  nf_ = (*(fluid_field()->rhs())).GlobalLength();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::FluidFluidMonolithicStructureSplitNoNOX::has_fluid_dof_map_changed(
    const Epetra_BlockMap& fluidincrementmap)
{
  return !fluid_field()->dof_row_map()->SameAs(fluidincrementmap);
}

FOUR_C_NAMESPACE_CLOSE
