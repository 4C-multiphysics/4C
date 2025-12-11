// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_monolithicsplit_nopenetration.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_contact_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


PoroElast::MonolithicSplitNoPenetration::MonolithicSplitNoPenetration(MPI_Comm comm,
    const Teuchos::ParameterList& timeparams,
    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter)
    : MonolithicSplit(comm, timeparams, porosity_splitter), normrhs_nopenetration_(-1.0)
{
  // Initialize Transformation Objects
  k_d_transform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  k_inv_d_transform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();

  k_d_lin_transform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  // Recovering of Lagrange multiplier happens on fluid field
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map());
  lambdanp_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map());

  k_dn_ = nullptr;

  mortar_adapter_ = std::make_shared<Adapter::CouplingNonLinMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());
}

void PoroElast::MonolithicSplitNoPenetration::setup_system()
{
  {
    const int ndim = Global::Problem::instance()->n_dim();
    std::vector<int> coupleddof(ndim + 1, 1);
    coupleddof[ndim] = 0;

    mortar_adapter_->setup(structure_field()->discretization(), fluid_field()->discretization(),
        coupleddof, "FSICoupling");
  }

  // use full maps of both fields. Only Lagrange multipliers are condensed
  {
    // create combined map
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

    vecSpaces.push_back(structure_field()->dof_row_map());
    vecSpaces.push_back(fluid_field()->dof_row_map());

    if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No structure equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No fluid equation. Panic.");

    // full Poroelasticity-map
    fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);
    // full Poroelasticity-blockmap
    blockrowdofmap_->setup(*fullmap_, vecSpaces);
  }

  // Switch fluid to interface split block matrix
  fluid_field()->use_block_matrix(true);

  // setup coupling objects, system and coupling matrices
  setup_coupling_and_matrices();

  // build map of dofs subjected to a DBC of whole problem
  build_combined_dbc_map();

  setup_equilibration();
}

void PoroElast::MonolithicSplitNoPenetration::setup_rhs(bool firstcall)
{
  // only Lagrange multipliers are condensed -> use unchanged maps from single fields
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplitNoPenetration::setup_rhs");

  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  setup_vector(*rhs_, structure_field()->rhs(), fluid_field()->rhs());
}

void PoroElast::MonolithicSplitNoPenetration::setup_vector(Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fv)
{
  // extract dofs of the two fields
  // and put the structural/fluid field vector into the global vector f
  // noticing the block number

  extractor()->insert_vector(*sv, 0, f);

  std::shared_ptr<Core::LinAlg::Vector<double>> fov =
      fluid_field()->interface()->extract_other_vector(*fv);
  std::shared_ptr<Core::LinAlg::Vector<double>> fcv =
      fluid_field()->interface()->extract_fsi_cond_vector(*fv);

  Core::LinAlg::Vector<double> Dlam(*fluid_field()->interface()->fsi_cond_map(), true);
  Core::LinAlg::Vector<double> couprhs(*fluid_field()->interface()->fsi_cond_map(), true);
  if (k_dn_ != nullptr)
  {
    double stiparam = structure_field()->tim_int_param();

    k_dn_->multiply(false, *lambda_, Dlam);  // D(n)*lambda(n)

    Dlam.scale(stiparam);  //*(1-b)
  }
  Dlam.update(-1.0, *fcv, 1.0);
  k_lambdainv_d_->multiply(false, Dlam, couprhs);

  couprhs.update(1.0, *nopenetration_rhs_, 1.0);

  // std::cout << "nopenetration_rhs_: " << *nopenetration_rhs_ << std::endl;

  std::shared_ptr<Core::LinAlg::Vector<double>> fullcouprhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
  Core::LinAlg::export_to(couprhs, *fullcouprhs);
  extractor()->insert_vector(*fullcouprhs, 1, f);

  std::shared_ptr<Core::LinAlg::Vector<double>> fullfov =
      std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
  Core::LinAlg::export_to(*fov, *fullfov);
  extractor()->add_vector(*fullfov, 1, f, 1.0);

  rhs_fgcur_ = fcv;  // Store interface rhs for recovering of lagrange multiplier
}

void PoroElast::MonolithicSplitNoPenetration::recover_lagrange_multiplier_after_newton_step(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x)
{
  // call base class
  Monolithic::recover_lagrange_multiplier_after_newton_step(x);


  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fx;
  extract_field_vectors(x, sx, fx);

  std::shared_ptr<Core::LinAlg::Vector<double>> sox =
      structure_field()->interface()->extract_other_vector(*sx);
  std::shared_ptr<Core::LinAlg::Vector<double>> scx =
      structure_field()->interface()->extract_fsi_cond_vector(*sx);
  std::shared_ptr<Core::LinAlg::Vector<double>> fox =
      fluid_field()->interface()->extract_other_vector(*fx);
  std::shared_ptr<Core::LinAlg::Vector<double>> fcx =
      fluid_field()->interface()->extract_fsi_cond_vector(*fx);

  ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*sox);  // first iteration increment

  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*scx);  // first iteration increment

  duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fox);  // first iteration increment

  duginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fcx);  // first iteration increment

  double stiparam = structure_field()->tim_int_param();

  // store the product Cfs_{\GammaI} \Delta d_I^{n+1} in here
  std::shared_ptr<Core::LinAlg::Vector<double>> cfsgiddi =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);
  // compute the above mentioned product
  cfsgicur_->multiply(false, *ddiinc_, *cfsgiddi);

  // store the product F_{\GammaI} \Delta u_I^{n+1} in here
  std::shared_ptr<Core::LinAlg::Vector<double>> fgiddi =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);
  // compute the above mentioned product
  fgicur_->multiply(false, *duiinc_, *fgiddi);

  // store the product Cfs_{\Gamma\Gamma} \Delta d_\Gamma^{n+1} in here
  std::shared_ptr<Core::LinAlg::Vector<double>> cfsggddg =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);
  // compute the above mentioned product
  cfsggcur_->multiply(false, *ddginc_, *cfsggddg);

  // store the product F_{\Gamma\Gamma} \Delta u_\Gamma^{n+1} in here
  std::shared_ptr<Core::LinAlg::Vector<double>> fggddg =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);
  // compute the above mentioned product
  fggcur_->multiply(false, *duginc_, *fggddg);

  // Update the Lagrange multiplier:
  /* \lambda^{n+1}_{i} =  -1/b * invD^{n+1} * [
   *                          + CFS_{\Gamma I} \Delta d_I
   *                          + CFS_{\Gamma \Gamma} \Delta d_\Gamma
   *                          + F_{\Gamma I} \Delta u_I
   *                          + F_{\Gamma\Gamma} \Delta u_\Gamma
   *                          - f_{\Gamma}^f]
   *                          - (1-b)/b * invD^{n+1} * D^n * \lambda^n
   */

  Core::LinAlg::Vector<double> tmplambda(*fluid_field()->interface()->fsi_cond_map(), true);

  tmplambda.update(1.0, *cfsgiddi, 0.0);
  tmplambda.update(1.0, *fgiddi, 1.0);
  tmplambda.update(1.0, *cfsggddg, 1.0);
  tmplambda.update(1.0, *fggddg, 1.0);
  tmplambda.update(-1.0, *rhs_fgcur_, 1.0);

  if (k_dn_ != nullptr)  // for first timestep lambda = 0 !
  {
    Core::LinAlg::Vector<double> Dlam(*fluid_field()->interface()->fsi_cond_map(), true);
    k_dn_->Apply(*lambda_, Dlam);  // D(n)*lambda(n)
    Dlam.scale(stiparam);          //*(1-b)
    tmplambda.update(1.0, Dlam, 1.0);
  }

  k_inv_d_->Apply(tmplambda, *lambdanp_);
  lambdanp_->scale(-1 / (1.0 - stiparam));  //*-1/b
}

void PoroElast::MonolithicSplitNoPenetration::setup_system_matrix(
    Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplitNoPenetration::setup_system_matrix");

  std::shared_ptr<Core::LinAlg::SparseMatrix> s = structure_field()->system_matrix();
  if (s == nullptr) FOUR_C_THROW("expect structure matrix");
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> f = fluid_field()->block_system_matrix();
  if (f == nullptr) FOUR_C_THROW("expect fluid block matrix");

  // Get Idx of fluid and structure field map extractor
  const int& fidx_other = FLD::Utils::MapExtractor::cond_other;
  const int& fidx_nopen = FLD::Utils::MapExtractor::cond_fsi;

  const int& sidx_other = Solid::MapExtractor::cond_other;
  const int& sidx_nopen = Solid::MapExtractor::cond_fsi;

  /*----------------------------------------------------------------------*/

  // just to play it safe ...
  mat.reset();

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  /*----------------------------------------------------------------------*/
  // structural part k_sf (3nxn)
  // build mechanical-fluid block

  // create empty matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> k_sf = struct_fluid_coupling_block_matrix();

  // call the element and calculate the matrix block
  apply_str_coupl_matrix(k_sf);

  /*----------------------------------------------------------------------*/
  // fluid part k_fs ( (3n+1)x3n )
  // build fluid-mechanical block

  // create empty matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> k_fs = fluid_struct_coupling_block_matrix();

  // call the element and calculate the matrix block
  apply_fluid_coupl_matrix(k_fs);

  /*----------------------------------------------------------------------*/

  k_fs->complete();
  k_sf->complete();

  /*----------------------------------------------------------------------*/
  // pure structural part
  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, *s);

  // structure coupling part
  Core::LinAlg::matrix_add(k_sf->matrix(sidx_other, fidx_other), false, 1.0, mat.matrix(0, 1), 0.0);
  Core::LinAlg::matrix_add(k_sf->matrix(sidx_other, fidx_nopen), false, 1.0, mat.matrix(0, 1), 1.0);
  Core::LinAlg::matrix_add(k_sf->matrix(sidx_nopen, fidx_other), false, 1.0, mat.matrix(0, 1), 1.0);
  Core::LinAlg::matrix_add(k_sf->matrix(sidx_nopen, fidx_nopen), false, 1.0, mat.matrix(0, 1), 1.0);
  /*----------------------------------------------------------------------*/
  // pure fluid part
  // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  // this just once...
  // f->UnComplete();

  Core::LinAlg::matrix_add(f->matrix(fidx_other, fidx_other), false, 1.0, mat.matrix(1, 1), 0.0);
  Core::LinAlg::matrix_add(f->matrix(fidx_other, fidx_nopen), false, 1.0, mat.matrix(1, 1), 1.0);

  // fluid coupling part
  Core::LinAlg::matrix_add(k_fs->matrix(fidx_other, fidx_other), false, 1.0, mat.matrix(1, 0), 0.0);
  Core::LinAlg::matrix_add(k_fs->matrix(fidx_other, fidx_nopen), false, 1.0, mat.matrix(1, 0), 1.0);

  /*----------------------------------------------------------------------*/
  /*Add lines for poro nopenetration condition*/

  fgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(fidx_nopen, fidx_other));
  fggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(fidx_nopen, fidx_nopen));
  cfsgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(k_fs->matrix(fidx_nopen, sidx_other));
  cfsggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(k_fs->matrix(fidx_nopen, sidx_nopen));

  std::shared_ptr<Core::LinAlg::SparseMatrix> tanginvDkfsgi = Core::LinAlg::matrix_multiply(
      *k_lambdainv_d_, false, *cfsgicur_, false, true);  // T*D^-1*K^FS_gi;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tanginvDfgi =
      Core::LinAlg::matrix_multiply(*k_lambdainv_d_, false, *fgicur_, false, true);  // T*D^-1*Fgi;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tanginvDfgg =
      Core::LinAlg::matrix_multiply(*k_lambdainv_d_, false, *fggcur_, false, true);  // T*D^-1*Fgg;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tanginvDkfsgg = Core::LinAlg::matrix_multiply(
      *k_lambdainv_d_, false, *cfsggcur_, false, true);  // T*D^-1*K^FS_gg;

  Core::LinAlg::matrix_add(*tanginvDkfsgi, false, -1.0, mat.matrix(1, 0), 1.0);
  Core::LinAlg::matrix_add(*tanginvDkfsgg, false, -1.0, mat.matrix(1, 0), 1.0);
  mat.matrix(1, 0).add(*k_struct_, false, 1.0, 1.0);
  Core::LinAlg::matrix_add(k_porodisp_->matrix(1, 0), false, 1.0, mat.matrix(1, 0), 1.0);
  Core::LinAlg::matrix_add(k_porodisp_->matrix(1, 1), false, 1.0, mat.matrix(1, 0), 1.0);
  Core::LinAlg::matrix_add(*tanginvDfgi, false, -1.0, mat.matrix(1, 1), 1.0);
  mat.matrix(1, 1).add(*k_fluid_, false, 1.0, 1.0);
  Core::LinAlg::matrix_add(*tanginvDfgg, false, -1.0, mat.matrix(1, 1), 1.0);
  mat.matrix(1, 1).add(*k_porofluid_, false, 1.0, 1.0);

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.complete();
}

void PoroElast::MonolithicSplitNoPenetration::apply_fluid_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_fs)
{
  // call base class
  Monolithic::apply_fluid_coupl_matrix(k_fs);

  // reset
  k_fluid_->zero();
  k_d_->zero();
  k_inv_d_->zero();
  k_struct_->zero();
  k_lambda_->zero();
  k_porodisp_->zero();
  k_porofluid_->zero();
  nopenetration_rhs_->put_scalar(0.0);

  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp_k_D =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *(fluid_field()->interface()->fsi_cond_map()), 81, false, false);

  // fill diagonal blocks
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_splitnopenetration);
    params.set("total time", time());
    params.set("delta time", dt());
    params.set("timescale", fluid_field()->residual_scaling());
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());
    fluid_field()->discretization()->set_state(0, "scaaf", *fluid_field()->scaaf());

    // fluid_field()->discretization()->set_state(0,"lambda",
    //    fluid_field()->Interface()->insert_fsi_cond_vector(structure_to_fluid_at_interface(lambdanp_)));

    // build specific assemble strategy for the fluid-mechanical system matrix
    // from the point of view of fluid_field:
    // fluiddofset = 0, structdofset = 1

    Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
        0,                                       // fluiddofset for column
        k_fluid_, nullptr, nopenetration_rhs_, nullptr, nullptr);
    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "FSICoupling");

    fluid_field()->discretization()->clear_state();
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> disp_interface =
      fluid_field()->interface()->extract_fsi_cond_vector(*fluid_field()->dispnp());
  mortar_adapter_->integrate_lin_d(
      "displacement", disp_interface, structure_to_fluid_at_interface(*lambdanp_));
  tmp_k_D = mortar_adapter_->get_mortar_matrix_d();

  // fill off diagonal blocks
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_splitnopenetration_OD);
    params.set("total time", time());
    params.set("delta time", dt());
    params.set("timescale", fluid_field()->residual_scaling());
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());
    fluid_field()->discretization()->set_state(0, "scaaf", *fluid_field()->scaaf());

    fluid_field()->discretization()->set_state(0, "lambda",
        *fluid_field()->interface()->insert_fsi_cond_vector(
            *structure_to_fluid_at_interface(*lambdanp_)));

    // build specific assemble strategy for the fluid-mechanical system matrix
    // from the point of view of fluid_field:
    // fluiddofset = 0, structdofset = 1
    Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
        1,                                       // structdofset for column
        k_struct_,                               // fluid-mechanical matrix
        k_lambda_, nullptr, nullptr, nullptr);
    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "FSICoupling");

    fluid_field()->discretization()->clear_state();
  }

  // fill off diagonal blocks
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_splitnopenetration_ODdisp);
    params.set("total time", time());
    params.set("delta time", dt());
    params.set("timescale", fluid_field()->residual_scaling());
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());
    //  fluid_field()->discretization()->set_state(0,"scaaf",fluid_field()->Scaaf());

    // build specific assemble strategy for the fluid-mechanical system matrix
    // from the point of view of fluid_field:
    // fluiddofset = 0, structdofset = 1
    Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
        1,                                       // structdofset for column
        k_porodisp_,                             // fluid-mechanical matrix
        nullptr, nullptr, nullptr, nullptr);
    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "FSICoupling");

    fluid_field()->discretization()->clear_state();
  }

  // fill off diagonal blocks
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_splitnopenetration_ODpres);
    params.set("total time", time());
    params.set("delta time", dt());
    params.set("timescale", fluid_field()->residual_scaling());
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());

    // build specific assemble strategy for the fluid-mechanical system matrix
    // from the point of view of fluid_field:
    // fluiddofset = 0, structdofset = 1
    Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
        0,                                       // fluiddofset for column
        k_porofluid_,                            // fluid-mechanical matrix
        nullptr, nullptr, nullptr, nullptr);
    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "FSICoupling");

    fluid_field()->discretization()->clear_state();
  }

  // Complete Coupling matrices which should be *.Add later!
  k_struct_->complete(
      *structure_field()->interface()->fsi_cond_map(), *fluid_field()->interface()->fsi_cond_map());
  k_fluid_->complete();
  k_porofluid_->complete();
  k_porodisp_->complete();

  //------------------------------invert D Matrix!-----------------------------------------------
  tmp_k_D->complete();
  std::shared_ptr<Core::LinAlg::SparseMatrix> invd =
      std::make_shared<Core::LinAlg::SparseMatrix>(*tmp_k_D, Core::LinAlg::DataAccess::Copy);
  // invd->Complete();

  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);


  // extract diagonal of invd into diag
  invd->extract_diagonal_copy(
      *diag);  // That the Reason, why tmp_k_D has to have Fluid Maps for Rows & Columns!!!

  // set zero diagonal values to dummy 1.0 ??
  for (int i = 0; i < diag->local_length(); ++i)
  {
    if (diag->local_values_as_span()[i] == 0.0)
    {
      (*diag).get_values()[i] = 1.0;
      std::cout << "--- --- --- WARNING: D-Matrix Diagonal Element " << i
                << " is zero!!! --- --- ---" << std::endl;
    }
  }

  // scalar inversion of diagonal values
  diag->reciprocal(*diag);

  // re-insert inverted diagonal into invd
  invd->replace_diagonal_values(*diag);
  invd->complete();
  //------------------------------End of invert D
  // Matrix!-----------------------------------------------

  // Transform also column map of D-Matrix
  (*k_d_transform_)(*fluid_field()->interface()->fsi_cond_map(),
      fluid_field()->block_system_matrix()->matrix(1, 1).col_map(), *tmp_k_D, 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*icoupfs_), *k_d_);

  (*k_inv_d_transform_)(
      *invd, 1.0, Coupling::Adapter::CouplingSlaveConverter(*icoupfs_), *k_inv_d_, false);

  double stiparam = structure_field()->tim_int_param();

  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp_k_DLin = mortar_adapter_->d_lin_matrix();
  tmp_k_DLin->complete();

  // Transform also column map of D-Matrix
  (*k_d_lin_transform_)(*fluid_field()->interface()->fsi_cond_map(),
      fluid_field()->block_system_matrix()->matrix(1, 1).col_map(), *tmp_k_DLin,
      1.0 - stiparam,  // *= b
      Coupling::Adapter::CouplingSlaveConverter(*icoupfs_),
      (std::static_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(k_fs))->matrix(1, 1), true,
      true);

  k_lambda_->complete(
      *structure_field()->interface()->fsi_cond_map(), *fluid_field()->interface()->fsi_cond_map());
  k_inv_d_->complete(
      *fluid_field()->interface()->fsi_cond_map(), *structure_field()->interface()->fsi_cond_map());
  k_d_->complete(
      *structure_field()->interface()->fsi_cond_map(), *fluid_field()->interface()->fsi_cond_map());

  // Calculate 1/b*Tangent*invD
  k_lambdainv_d_ = Core::LinAlg::matrix_multiply(*k_lambda_, false, *k_inv_d_, false, true);
  k_lambdainv_d_->scale(1.0 / (1.0 - stiparam));  // *= 1/b
}

void PoroElast::MonolithicSplitNoPenetration::apply_str_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sf  //!< off-diagonal tangent matrix term
)
{
  // call base class
  Monolithic::apply_str_coupl_matrix(k_sf);
}

void PoroElast::MonolithicSplitNoPenetration::recover_lagrange_multiplier_after_time_step()
{
  // we do not need to recover after a time step, it is done after every newton step
}

void PoroElast::MonolithicSplitNoPenetration::update()
{
  // call base class
  MonolithicSplit::update();

  // update lagrangean multiplier
  lambda_->update(1.0, *lambdanp_, 0.0);

  // copy D matrix from current time step to old D matrix
  k_dn_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *k_d_, Core::LinAlg::DataAccess::Copy);  // store D-Matrix from last timestep
}

void PoroElast::MonolithicSplitNoPenetration::output(bool forced_writerestart)
{
  // call base class
  MonolithicSplit::output(forced_writerestart);

  // for now, we always write the lagrange multiplier
  std::shared_ptr<Core::LinAlg::Vector<double>> fulllambda =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map());
  Core::LinAlg::export_to(*lambdanp_, *fulllambda);
  structure_field()->disc_writer()->write_vector("poronopencond_lambda", fulllambda);
}

void PoroElast::MonolithicSplitNoPenetration::setup_coupling_and_matrices()
{
  const int ndim = Global::Problem::instance()->n_dim();
  icoupfs_->setup_condition_coupling(*structure_field()->discretization(),
      structure_field()->interface()->fsi_cond_map(), *fluid_field()->discretization(),
      fluid_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  evaluateinterface_ = false;

  // initialize Poroelasticity-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  // initialize coupling matrices
  k_fs_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *(structure_field()->interface()), *(fluid_field()->interface()), 81, false, true);

  k_sf_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *(fluid_field()->interface()), *(structure_field()->interface()), 81, false, true);

  // initialize no penetration coupling matrices
  k_struct_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->interface()->fsi_cond_map()), 81, true, true);

  k_fluid_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->interface()->fsi_cond_map()), 81, false, false);

  k_lambda_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->interface()->fsi_cond_map()), 81, true, true);

  k_d_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->interface()->fsi_cond_map()), 81, true, true);

  k_inv_d_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(structure_field()->interface()->fsi_cond_map()), 81, true, true);

  k_porodisp_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *(structure_field()->interface()), *(fluid_field()->interface()), 81, true, true);

  k_porofluid_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(fluid_field()->dof_row_map()), 81, true, true);

  nopenetration_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);
}

void PoroElast::MonolithicSplitNoPenetration::prepare_time_step()
{
  // call base class
  PoroElast::Monolithic::prepare_time_step();
}

void PoroElast::MonolithicSplitNoPenetration::read_restart(const int step)
{
  // call base class
  PoroElast::PoroBase::read_restart(step);

  // get lagrange multiplier and D matrix
  if (step)
  {
    // get the structure reader (this is where the lagrange multiplier was saved)
    Core::IO::DiscretizationReader reader(*structure_field()->discretization(),
        Global::Problem::instance()->input_control_file(), structure_field()->step());
    std::shared_ptr<Core::LinAlg::Vector<double>> fulllambda =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map());

    // this is the lagrange multiplier on the whole structure field
    reader.read_vector(fulllambda, "poronopencond_lambda");

    // extract lambda on fsi interface vector
    lambda_ = structure_field()->interface()->extract_fsi_cond_vector(*fulllambda);
    lambdanp_->update(1.0, *lambda_, 0.0);

    // call an additional evaluate to get the old D matrix
    setup_system();
    // call evaluate to recalculate D matrix
    evaluate(zeros_, false);

    // copy D matrix from current time step to old D matrix
    k_dn_ = std::make_shared<Core::LinAlg::SparseMatrix>(
        *k_d_, Core::LinAlg::DataAccess::Copy);  // store D-Matrix from last timestep
  }
}

void PoroElast::MonolithicSplitNoPenetration::print_newton_iter_header_stream(
    std::ostringstream& oss)
{
  Monolithic::print_newton_iter_header_stream(oss);

  oss << std::setw(20) << "abs-crhs-res";
}

void PoroElast::MonolithicSplitNoPenetration::print_newton_iter_text_stream(std::ostringstream& oss)
{
  Monolithic::print_newton_iter_text_stream(oss);

  oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhs_nopenetration_;
}

void PoroElast::MonolithicSplitNoPenetration::build_convergence_norms()
{
  Monolithic::build_convergence_norms();

  normrhs_nopenetration_ = Utils::calculate_vector_norm(vectornormfres_, *nopenetration_rhs_);
}

FOUR_C_NAMESPACE_CLOSE
