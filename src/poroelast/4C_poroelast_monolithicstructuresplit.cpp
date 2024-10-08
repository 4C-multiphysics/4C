/*----------------------------------------------------------------------*/
/*! \file

 \brief  monolithic structure split poroelasticity algorithm

\level 2

 *------------------------------------------------------------------------------------------------*/

#include "4C_poroelast_monolithicstructuresplit.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


PoroElast::MonolithicStructureSplit::MonolithicStructureSplit(const Epetra_Comm& comm,
    const Teuchos::ParameterList& timeparams,
    Teuchos::RCP<Core::LinAlg::MapExtractor> porosity_splitter)
    : MonolithicSplit(comm, timeparams, porosity_splitter)
{
  sggtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowColTransform);
  sgitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowTransform);
  sigtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  csggtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowTransform);
  cfggtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);
  csgitransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixRowTransform);
  cfigtransform_ = Teuchos::rcp(new Coupling::Adapter::MatrixColTransform);

  // Recovering of Lagrange multiplier happens on structure field
  lambda_ = Teuchos::rcp(
      new Core::LinAlg::Vector<double>(*structure_field()->interface()->fsi_cond_map()));
}

void PoroElast::MonolithicStructureSplit::setup_system()
{
  {
    // create combined map
    std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;

    vecSpaces.push_back(structure_field()->interface()->other_map());
    vecSpaces.push_back(fluid_field()->dof_row_map());

    if (vecSpaces[0]->NumGlobalElements() == 0) FOUR_C_THROW("No structure equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) FOUR_C_THROW("No fluid equation. Panic.");

    // full Poroelasticity-map
    fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);
    // full Poroelasticity-blockmap
    blockrowdofmap_->setup(*fullmap_, vecSpaces);
  }

  // Use splitted structure matrix
  structure_field()->use_block_matrix();

  setup_coupling_and_matrices();

  build_combined_dbc_map();

  setup_equilibration();
}

void PoroElast::MonolithicStructureSplit::setup_rhs(bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicStructureSplit::setup_rhs");

  // create full monolithic rhs vector
  rhs_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*dof_row_map(), true));

  setup_vector(
      *rhs_, structure_field()->rhs(), fluid_field()->rhs(), fluid_field()->residual_scaling());

  // store interface force onto the structure to know it in the next time step as previous force
  // in order to recover the Lagrange multiplier
  // fgpre_ = fgcur_;
  fgcur_ = structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->rhs());
}

void PoroElast::MonolithicStructureSplit::setup_system_matrix(
    Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicStructureSplit::setup_system_matrix");

  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> s = structure_field()->block_system_matrix();
  if (s == Teuchos::null) FOUR_C_THROW("expect structure block matrix");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();
  if (f == Teuchos::null) FOUR_C_THROW("expect fluid matrix");

  // just to play it safe ...
  mat.zero();

  /*----------------------------------------------------------------------*/

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  /*----------------------------------------------------------------------*/
  // structural part k_sf (3nxn)
  // build mechanical-fluid block

  // create empty matrix
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> k_sf = struct_fluid_coupling_block_matrix();
  if (k_sf == Teuchos::null) FOUR_C_THROW("expect coupling block matrix");

  // call the element and calculate the matrix block
  apply_str_coupl_matrix(k_sf);

  /*----------------------------------------------------------------------*/
  // fluid part k_fs ( (3n+1)x3n )
  // build fluid-mechanical block

  // create empty matrix
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> k_fs = fluid_struct_coupling_block_matrix();
  if (k_fs == Teuchos::null) FOUR_C_THROW("expect coupling block matrix");

  // call the element and calculate the matrix block
  apply_fluid_coupl_matrix(k_fs);

  /*----------------------------------------------------------------------*/

  k_fs->complete();
  k_sf->complete();

  f->un_complete();

  /*----------------------------------------------------------------------*/
  if (evaluateinterface_)
  {
    double scale = fluid_field()->residual_scaling();
    double timescale = fluid_field()->time_scaling();

    // get time integration parameters of structure an fluid time integrators
    // to enable consistent time integration among the fields
    double stiparam = structure_field()->tim_int_param();
    double ftiparam = fluid_field()->tim_int_param();

    (*sigtransform_)(s->full_row_map(), s->full_col_map(), s->matrix(0, 1), 1. / timescale,
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_),
        k_sf->matrix(0, 1),  // k_sf->Matrix(0,1),mat.Matrix(0,1)
        true, true);

    (*sggtransform_)(s->matrix(1, 1), (1.0 - ftiparam) / ((1.0 - stiparam) * scale * timescale),
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_),
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_), *f, true, true);

    (*sgitransform_)(s->matrix(1, 0), (1.0 - ftiparam) / ((1.0 - stiparam) * scale),
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_),
        k_fs->matrix(1, 0),  // k_fs->Matrix(1,0), mat.Matrix(1,0)
        true);

    (*cfggtransform_)(s->full_row_map(),  // k_fs->FullRowMap(),
        s->full_col_map(),                // k_fs->FullColMap(),
        k_fs->matrix(1, 1), 1. / timescale, Coupling::Adapter::CouplingMasterConverter(*icoupfs_),
        *f, true, true);

    (*cfigtransform_)(s->full_row_map(),  // k_fs->FullRowMap(),
        s->full_col_map(),                // k_fs->FullColMap(),
        k_fs->matrix(0, 1), 1. / timescale, Coupling::Adapter::CouplingMasterConverter(*icoupfs_),
        *f, true, true);

    (*csggtransform_)(k_sf->matrix(1, 1), (1.0 - ftiparam) / ((1.0 - stiparam) * scale),
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_), *f, true);

    (*csgitransform_)(k_sf->matrix(1, 0), (1.0 - ftiparam) / ((1.0 - stiparam) * scale),
        Coupling::Adapter::CouplingMasterConverter(*icoupfs_), *f, true);
  }

  /*----------------------------------------------------------------------*/
  // pure structural part
  mat.matrix(0, 0).add(s->matrix(0, 0), false, 1., 0.0);

  // structure coupling part
  mat.matrix(0, 1).add(k_sf->matrix(0, 0), false, 1.0, 0.0);
  mat.matrix(0, 1).add(k_sf->matrix(0, 1), false, 1.0, 1.0);

  // pure fluid part
  mat.assign(1, 1, Core::LinAlg::View, *f);

  // fluid coupling part
  mat.matrix(1, 0).add(k_fs->matrix(0, 0), false, 1.0, 0.0);
  mat.matrix(1, 0).add(k_fs->matrix(1, 0), false, 1.0, 1.0);
  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.complete();

  sgicur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->matrix(1, 0)));
  sggcur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->matrix(1, 1)));
  cgicur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(k_sf->matrix(1, 0)));
  cggcur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(k_sf->matrix(1, 1)));
}

void PoroElast::MonolithicStructureSplit::setup_vector(Core::LinAlg::Vector<double>& f,
    Teuchos::RCP<const Core::LinAlg::Vector<double>> sv,
    Teuchos::RCP<const Core::LinAlg::Vector<double>> fv, double fluidscale)
{
  // extract the inner and boundary dofs of all three fields

  Teuchos::RCP<Core::LinAlg::Vector<double>> sov =
      structure_field()->interface()->extract_other_vector(*sv);

  if (fluidscale != 0.0)
  {
    // get time integration parameters of structure an fluid time integrators
    // to enable consistent time integration among the fields
    double stiparam = structure_field()->tim_int_param();
    double ftiparam = fluid_field()->tim_int_param();

    // add fluid interface values to structure vector
    Teuchos::RCP<Core::LinAlg::Vector<double>> scv =
        structure_field()->interface()->extract_fsi_cond_vector(*sv);
    Teuchos::RCP<Core::LinAlg::Vector<double>> modfv =
        fluid_field()->interface()->insert_fsi_cond_vector(*structure_to_fluid_at_interface(scv));
    modfv->Update(1.0, *fv, (1.0 - ftiparam) / ((1.0 - stiparam) * fluidscale));

    // add contribution of Lagrange multiplier from previous time step
    if (lambda_ != Teuchos::null)
      modfv->Update(-ftiparam + stiparam * (1.0 - ftiparam) / (1.0 - stiparam),
          *structure_to_fluid_at_interface(lambda_), 1.0);

    extractor()->insert_vector(*modfv, 1, f);
  }
  else
  {
    extractor()->insert_vector(*fv, 1, f);
  }

  extractor()->insert_vector(*sov, 0, f);
}

void PoroElast::MonolithicStructureSplit::extract_field_vectors(
    Teuchos::RCP<const Core::LinAlg::Vector<double>> x,
    Teuchos::RCP<const Core::LinAlg::Vector<double>>& sx,
    Teuchos::RCP<const Core::LinAlg::Vector<double>>& fx, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicStructureSplit::extract_field_vectors");

  // process fluid unknowns of the second field
  fx = extractor()->extract_vector(*x, 1);

  // process structure unknowns
  if (evaluateinterface_)
  {
    Teuchos::RCP<Core::LinAlg::Vector<double>> fcx =
        fluid_field()->interface()->extract_fsi_cond_vector(*fx);

    {
      double timescale = 1. / fluid_field()->time_scaling();
      fcx->Scale(timescale);
    }

    Teuchos::RCP<Core::LinAlg::Vector<double>> scx = fluid_to_structure_at_interface(fcx);
    Teuchos::RCP<const Core::LinAlg::Vector<double>> sox = extractor()->extract_vector(*x, 0);

    Teuchos::RCP<Core::LinAlg::Vector<double>> s =
        structure_field()->interface()->insert_other_vector(*sox);
    structure_field()->interface()->insert_fsi_cond_vector(*scx, *s);

    auto zeros = Teuchos::rcp(new const Core::LinAlg::Vector<double>(s->Map(), true));
    Core::LinAlg::apply_dirichlet_to_system(
        *s, *zeros, *(structure_field()->get_dbc_map_extractor()->cond_map()));

    sx = s;

    Teuchos::RCP<Core::LinAlg::Vector<double>> fox =
        fluid_field()->interface()->extract_other_vector(*fx);

    // Store field vectors to know them later on as previous quantities
    if (solipre_ != Teuchos::null)
      ddiinc_->Update(1.0, *sox, -1.0, *solipre_, 0.0);  // compute current iteration increment
    else
      ddiinc_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*sox));  // first iteration increment

    solipre_ = sox;  // store current step increment

    if (solgpre_ != Teuchos::null)
      ddginc_->Update(1.0, *scx, -1.0, *solgpre_, 0.0);  // compute current iteration increment
    else
      ddginc_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*scx));  // first iteration increment

    solgpre_ = scx;  // store current step increment

    if (solivelpre_ != Teuchos::null)
      duiinc_->Update(1.0, *fox, -1.0, *solivelpre_, 0.0);  // compute current iteration increment
    else
      duiinc_ = Teuchos::rcp(new Core::LinAlg::Vector<double>(*fox));  // first iteration increment

    solivelpre_ = fox;  // store current step increment
  }
  else
    sx = extractor()->extract_vector(*x, 0);
}

void PoroElast::MonolithicStructureSplit::recover_lagrange_multiplier_after_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroElast::MonolithicStructureSplit::recover_lagrange_multiplier_after_time_step");
  if (evaluateinterface_)
  {
    // get time integration parameter of structural time integrator
    // to enable consistent time integration among the fields
    double stiparam = structure_field()->tim_int_param();
    double timescale = fluid_field()->time_scaling();

    // store the product S_{\GammaI} \Delta d_I^{n+1} in here
    Teuchos::RCP<Core::LinAlg::Vector<double>> sgiddi =
        Core::LinAlg::create_vector(*structure_field()->interface()->fsi_cond_map(), true);
    // compute the above mentioned product
    sgicur_->multiply(false, *ddiinc_, *sgiddi);

    // store the product C_{\GammaI} \Delta u_I^{n+1} in here
    Teuchos::RCP<Core::LinAlg::Vector<double>> fgiddi =
        Core::LinAlg::create_vector(*structure_field()->interface()->fsi_cond_map(), true);
    // compute the above mentioned product
    cgicur_->multiply(false, *duiinc_, *fgiddi);

    // store the product S_{\Gamma\Gamma} \Delta d_\Gamma^{n+1} in here
    Teuchos::RCP<Core::LinAlg::Vector<double>> sggddg =
        Core::LinAlg::create_vector(*structure_field()->interface()->fsi_cond_map(), true);
    // compute the above mentioned product
    sggcur_->multiply(false, *ddginc_, *sggddg);

    // store the prodcut C_{\Gamma\Gamma} \Delta u_\Gamma^{n+1} in here
    Teuchos::RCP<Core::LinAlg::Vector<double>> cggddg =
        Core::LinAlg::create_vector(*structure_field()->interface()->fsi_cond_map(), true);
    // compute the above mentioned product
    cggcur_->multiply(false, *ddginc_, *cggddg);
    cggddg->Scale(timescale);

    // Update the Lagrange multiplier:
    /* \lambda^{n+1} =  1/b * [ - a*\lambda^n - f_\Gamma^S
     *                          - S_{\Gamma I} \Delta d_I
     *                          - C_{\Gamma I} \Delta u_I
     *                          - S_{\Gamma\Gamma} \Delta d_\Gamma]
     *                          - C_{\Gamma\Gamma} * 2 / \Delta t * \Delta d_\Gamma]
     */
    // lambda_->Update(1.0, *fgpre_, -stiparam);
    lambda_->Update(1.0, *fgcur_, -stiparam);
    lambda_->Update(-1.0, *sgiddi, -1.0, *sggddg, 1.0);
    lambda_->Update(-1.0, *fgiddi, -1.0, *cggddg, 1.0);
    lambda_->Scale(1 / (1.0 - stiparam));  // entire Lagrange multiplier is divided by (1.-stiparam)
  }
}

FOUR_C_NAMESPACE_CLOSE
