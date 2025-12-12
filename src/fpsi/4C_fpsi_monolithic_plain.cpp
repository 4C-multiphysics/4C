// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fpsi_monolithic_plain.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fpsi_monolithic.hpp"
#include "4C_fpsi_utils.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FPSI::MonolithicPlain::MonolithicPlain(MPI_Comm comm, const Teuchos::ParameterList& fpsidynparams,
    const Teuchos::ParameterList& poroelastdynparams)
    : Monolithic(comm, fpsidynparams, poroelastdynparams)
{
  // create transformation object for the condensation


  fggtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  fggtransform2_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  fmgitransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();

  fgitransform1_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  fgitransform2_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  cfgtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  cfptransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  cfptransform2_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();

  figtransform1_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  figtransform2_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  figtransform3_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  figtransform4_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  aigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  aigtransform2_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  couplingcoltransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  couplingcoltransformfs_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  // Recovery of Lagrange multiplier happens on fluid field
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);
  fmgiprev_ = nullptr;
  fmgicur_ = nullptr;
  fmggprev_ = nullptr;
  fmggcur_ = nullptr;
  fgiprev_ = nullptr;
  fgicur_ = nullptr;
  fggprev_ = nullptr;
  fggcur_ = nullptr;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (fggtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fggtransform_' failed.");
  }
  if (fggtransform2_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fggtransform2_' failed.");
  }
  if (fmgitransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fmgitransform_' failed.");
  }

  if (fgitransform1_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fgitransform1_' failed.");
  }
  if (fgitransform2_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fgitransform2_' failed.");
  }
  if (cfgtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'Cfgtransform_' failed.");
  }

  if (figtransform1_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'figtransform1_' failed.");
  }
  if (figtransform2_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'figtransform2_' failed.");
  }
  if (figtransform3_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'figtransform3_' failed.");
  }
  if (figtransform4_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'figtransform4_' failed.");
  }
  if (aigtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'aigtransform_' failed.");
  }
  if (aigtransform2_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'aigtransform2_' failed.");
  }

  if (couplingcoltransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'couplingcoltransform_' failed.");
  }
  if (couplingcoltransformfs_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'couplingcoltransformfs_' failed.");
  }

  if (lambda_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'lambda_' failed.");
  }
#endif
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_system()
{
  const Teuchos::ParameterList& fpsidynparams = Global::Problem::instance()->fpsi_dynamic_params();

  set_default_parameters(fpsidynparams);

  // call SetupSystem in base classes
  poro_field()->setup_system();
  FPSI::Monolithic::setup_system();

  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  vecSpaces.push_back(poro_field()->extractor()->map(0));

  vecSpaces.push_back(poro_field()->extractor()->map(1));

  vecSpaces.push_back(fluid_field()->dof_row_map());

  vecSpaces.push_back(ale_field()->interface()->other_map());

  // Modify block_numbers manually according to vecSpaces.push_back order!
  structure_block_ = 0;
  porofluid_block_ = 1;
  fluid_block_ = 2;
  ale_i_block_ = 3;

  if (vecSpaces[structure_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No inner poro structure equations. Splitting not possible.");
  if (vecSpaces[porofluid_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No inner poro fluid equations. Splitting not possible.");
  if (vecSpaces[fluid_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No inner fluid equations. Splitting not possible.");
  if (vecSpaces[ale_i_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No inner ale equations. Splitting not possible.");
  // merge maps and create full monolithic FPSI-dof_row_map
  set_dof_row_maps(vecSpaces);

  // switch fluid to interface split block matrix
  fluid_field()->use_block_matrix(true, fpsi_coupl()->fluid_fsi_fpsi_extractor());

  // build ale system matrix in split system
  ale_field()->create_system_matrix(ale_field()->interface());

  // initialize FPSI-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 81, false, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::set_dof_row_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  // full FPSI-blockmap
  blockrowdofmap_.setup(*fullmap_, maps);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FPSI::Monolithic::set_default_parameters(const Teuchos::ParameterList& fpsidynparams)
{
  // to do
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_rhs(bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("FPSI::MonolithicPlain::setup_rhs");
  // create full monolithic rhs vector

  rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  firstcall_ = firstcall;

  poro_field()->setup_rhs(firstcall_);

  setup_vector(*rhs_, *poro_field()->extractor()->extract_vector(*poro_field()->rhs(), 0),
      *poro_field()->extractor()->extract_vector(*poro_field()->rhs(), 1), *fluid_field()->rhs(),
      *ale_field()->rhs(), fluid_field()->residual_scaling());

  if (FSI_Interface_exists_)
  {
    setup_rhs_lambda(*rhs_);
    //    if (firstcall) //it is still neccacary to look deeper into that!
    //    {
    //      setup_rhs_first_iter(*rhs_);
    //    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FPSI::MonolithicPlain::setup_system_matrix");
  mat.un_complete();  // basically makes no sense as all blocks will be assigned later!!!

  // get single field block matrices
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> p = poro_field()->block_system_matrix();
  p->un_complete();

  const std::shared_ptr<Core::LinAlg::SparseMatrix> f = fluid_field()->system_sparse_matrix();
  f->un_complete();

  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fbm =
      fluid_field()->block_system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();

  // Get Idx of fluid and ale field map extractors
  const int& fidx_other = FLD::Utils::MapExtractor::cond_other;
  const int& fidx_fsi = FLD::Utils::MapExtractor::cond_fsi;

  const int& aidx_other = ALE::Utils::MapExtractor::cond_other;
  const int& aidx_fsi = ALE::Utils::MapExtractor::cond_fsi;
  const int& aidx_fpsi = ALE::Utils::MapExtractor::cond_fpsi;

  // FPSI Couplings
  const Coupling::Adapter::Coupling& coupsa_fpsi = fpsi_coupl()->poro_structure_ale_coupling();
  const Coupling::Adapter::Coupling& coupsf_fpsi = fpsi_coupl()->poro_structure_fluid_coupling();

  // General Couplings
  const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

  // FSI Couplings
  const Coupling::Adapter::Coupling& coupsf_fsi = structure_fluid_coupling_fsi();
  const Coupling::Adapter::Coupling& coupsa_fsi = structure_ale_coupling_fsi();

  ///////////ADD THE COUPLING HERE////////////////
  p->add(fpsi_coupl()->c_pp(), false, 1.0, 1.0);

  // Fluid Coupling Matrix is created as BlockMatrix, to enable condensation procedure ...
  Core::LinAlg::matrix_add(
      fpsi_coupl()->c_ff().matrix(fidx_other, fidx_other), false, 1.0, *f, 1.0);
  Core::LinAlg::matrix_add(fpsi_coupl()->c_ff().matrix(fidx_other, fidx_fsi), false, 1.0, *f, 1.0);
  Core::LinAlg::matrix_add(fpsi_coupl()->c_ff().matrix(fidx_fsi, fidx_other), false, 1.0, *f, 1.0);
  Core::LinAlg::matrix_add(fpsi_coupl()->c_ff().matrix(fidx_fsi, fidx_fsi), false, 1.0, *f, 1.0);

  f->complete();

  ////////////////////////////////////////////////

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = poro_field()->structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  /*----------------------------------------------------------------------*/
  // build block matrix
  /*----------------------------------------------------------------------*/
  // insert poro
  mat.assign(structure_block_, structure_block_, Core::LinAlg::DataAccess::Share, p->matrix(0, 0));
  mat.assign(structure_block_, porofluid_block_, Core::LinAlg::DataAccess::Share, p->matrix(0, 1));
  mat.assign(porofluid_block_, porofluid_block_, Core::LinAlg::DataAccess::Share, p->matrix(1, 1));
  mat.assign(porofluid_block_, structure_block_, Core::LinAlg::DataAccess::Share, p->matrix(1, 0));

  // Assign fii + Coupling Parts
  mat.assign(fluid_block_, fluid_block_, Core::LinAlg::DataAccess::Share, *f);

  // Assign C_fp
  mat.assign(fluid_block_, structure_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_fp().matrix(0, 0));
  mat.assign(fluid_block_, porofluid_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_fp().matrix(0, 1));

  // Assign C_pf
  mat.assign(structure_block_, fluid_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_pf().matrix(0, 0));
  mat.assign(porofluid_block_, fluid_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_pf().matrix(1, 0));

  // Assign C_pa
  mat.assign(structure_block_, ale_i_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_pa().matrix(0, 0));
  mat.assign(porofluid_block_, ale_i_block_, Core::LinAlg::DataAccess::Share,
      fpsi_coupl()->c_pa().matrix(1, 0));

  // Assign C_fa
  mat.assign(fluid_block_, ale_i_block_, Core::LinAlg::DataAccess::Share, fpsi_coupl()->c_fa());

  // ALE Condensation
  Core::LinAlg::SparseMatrix& aii = a->matrix(aidx_other, aidx_other);
  Core::LinAlg::SparseMatrix& ai_gfpsi = a->matrix(aidx_other, aidx_fpsi);

  // create transformation object for the ale condensation
  (*aigtransform2_)(a->full_row_map(), a->full_col_map(), ai_gfpsi, 1.,
      Coupling::Adapter::CouplingSlaveConverter(coupsa_fpsi),
      mat.matrix(ale_i_block_, structure_block_), true,
      false);  // Add

  mat.assign(ale_i_block_, ale_i_block_, Core::LinAlg::DataAccess::Share, aii);

  // Insert condensed Fluid Blocks: Fgg and Fgi (+ Fg_gFPSI) --> g is on the FSI-Interface
  if (FSI_Interface_exists_)
  {
    // extract fluid submatrices -- use block matrices just for fsi boundary matrices as they have
    // to be condensed!
    // --> others will be assigned directly by the fluid sparse matrix
    Core::LinAlg::SparseMatrix fgi = fbm->matrix(fidx_fsi, fidx_other);
    // Core::LinAlg::SparseMatrix fg_gfpsi =   fbm->Matrix(fidx_fsi,fidx_fpsi);
    // Core::LinAlg::SparseMatrix& fgg     =   fbm->Matrix(fidx_fsi,fidx_fsi);

    // As the Fluid Block Matrix is used here, the already to the f-SparseMatrix
    // added FPSI-Coupling terms have to be added again!!!
    Core::LinAlg::matrix_add(fpsi_coupl()->c_ff().matrix(fidx_fsi, fidx_other), false, 1.0, fgi,
        1.0);  // is missing in old implementation
    // fg_gfpsi.Add(FPSICoupl()->C_ff().Matrix(fidx_fsi,fidx_fpsi),false,1.0,1.0);  //is missing in
    // old implementation fgg.Add(FPSICoupl()->C_ff().Matrix(fidx_fsi,fidx_fsi),false,1.0,1.0);
    //        (*fggtransform_)( fgg,
    //                         (1.0-stiparam)/(1.0-ftiparam)*scale*timescale*1,
    //                         Adapter::CouplingSlaveConverter(coupsf_fsi),
    //                         Adapter::CouplingSlaveConverter(coupsf_fsi),
    //                         *p,
    //                         true,
    //                         true); //not required anymore --> done at (1)

    (*fgitransform1_)(fgi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
        mat.matrix(structure_block_, fluid_block_),
        true);  // Assign

    // Insert ale: Aii, Aig and Ai_gFPSI--> g is on the FSI-Interface

    Core::LinAlg::SparseMatrix& aig = a->matrix(aidx_other, aidx_fsi);

    (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1.,
        Coupling::Adapter::CouplingSlaveConverter(coupsa_fsi),
        mat.matrix(ale_i_block_, structure_block_),
        true);  // as just fsi part is transferred
  }

  //////////////////////////////////////////////
  //////                                  //////
  //////    Linearization of fluid_field   //////
  //////    with respect to ale mesh      //////
  //////             motion               //////
  //////                                  //////
  //////////////////////////////////////////////
  const Teuchos::ParameterList& fpsidynparams = Global::Problem::instance()->fpsi_dynamic_params();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fluidalematrix =
      fluid_field()->shape_derivatives();
  if (fpsidynparams.get<bool>("USESHAPEDERIVATIVES"))
  {
    if (fluidalematrix != nullptr)
    {
      // There is no fpsi-fsi overlap in the block matrixes, all dofs which are on both interfaces
      // belong to the fsi-block matrix!

      Core::LinAlg::SparseMatrix& fluidalematrix_ii = fluidalematrix->matrix(
          FPSI::Utils::MapExtractor::cond_other, FPSI::Utils::MapExtractor::cond_other);

      // add fluid_ale block ii and gi
      // those two blocks are not condensed since they belong to the columns of the inner ale dofs
      (*couplingcoltransform_)(fluid_field()->block_system_matrix()->full_row_map(),
          fluid_field()->block_system_matrix()->full_col_map(), fluidalematrix_ii, 1.0,
          Coupling::Adapter::CouplingMasterConverter(
              coupfa),  // row converter: important to use slave converter
          mat.matrix(fluid_block_, ale_i_block_),
          false,  // bool exactmatch = true (default)
          true);

      if (FSI_Interface_exists_)
      {
        Core::LinAlg::SparseMatrix& fluidalematrix_gg_fsi = fluidalematrix->matrix(
            FPSI::Utils::MapExtractor::cond_fsi, FPSI::Utils::MapExtractor::cond_fsi);
        Core::LinAlg::SparseMatrix& fluidalematrix_gi_fsi = fluidalematrix->matrix(
            FPSI::Utils::MapExtractor::cond_fsi, FPSI::Utils::MapExtractor::cond_other);
        Core::LinAlg::SparseMatrix& fluidalematrix_ig_fsi = fluidalematrix->matrix(
            FPSI::Utils::MapExtractor::cond_other, FPSI::Utils::MapExtractor::cond_fsi);

        Core::LinAlg::SparseMatrix& fluidalematrix_gfsigfpsi = fluidalematrix->matrix(
            FPSI::Utils::MapExtractor::cond_fsi, FPSI::Utils::MapExtractor::cond_fpsi);
        Core::LinAlg::SparseMatrix& fluidalematrix_gfpsigfsi = fluidalematrix->matrix(
            FPSI::Utils::MapExtractor::cond_fpsi, FPSI::Utils::MapExtractor::cond_fsi);


        (*figtransform1_)(fluid_field()->block_system_matrix()->full_row_map(),
            fluid_field()->block_system_matrix()->full_col_map(), fluidalematrix_ig_fsi, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            mat.matrix(fluid_block_, structure_block_),
            false,  // bool exactmatch = true (default)
            true);

        (*figtransform2_)(fluid_field()->block_system_matrix()->full_row_map(),
            fluid_field()->block_system_matrix()->full_col_map(), fluidalematrix_gfpsigfsi, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            mat.matrix(fluid_block_, structure_block_), false, true);

        (*fggtransform_)(fluidalematrix_gg_fsi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            mat.matrix(structure_block_, structure_block_), false, true);

        (*fggtransform2_)(fluidalematrix_gfsigfpsi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fpsi),
            mat.matrix(structure_block_, structure_block_), false, true);

        (*fmgitransform_)(fluidalematrix_gi_fsi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
            Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
            Coupling::Adapter::CouplingMasterConverter(coupfa),
            mat.matrix(structure_block_, ale_i_block_), false, true);
      }
    }
    else  // if shapederivatives = no in FluidDynamics section in input file
    {
      std::cout << "WARNING: Linearization with respect to mesh motion of fluid subproblem is "
                   "switched off!"
                << std::endl;
    }
  }  // if useshapederivatives

  //////////////////////////////////////////////////////
  ///////                                        ///////
  ///////   FILL FSI/FPSI - INTERFACE OVERLAP    ///////
  ///////                                        ///////
  //////////////////////////////////////////////////////

  if (FSI_Interface_exists_)  // Add blocks as already matrices have been added to these blocks!
  {
    // Compete for the Matrix Transformation Object (Already everything filled into the pf part!)

    // Add (Tau * Fig) to Structural Column (condensation of fluid velocities) ... done here to
    // catch also FPSI-Coupling terms (for overlapping FSI/FPSI Interfaces)
    (*figtransform3_)(fluid_field()->block_system_matrix()->full_row_map(),
        fluid_field()->block_system_matrix()->full_col_map(),
        mat.matrix(fluid_block_, fluid_block_), timescale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
        mat.matrix(fluid_block_, structure_block_),  //--> goes into C_fp()
        false,  // no exactmatch! (just FSI Part should be extracted)
        true);  // Add

    // Complete for the Matrix Transformation Object (Already everything filled into the pf part!)
    fpsi_coupl()->c_fa().complete(
        *ale_field()->interface()->other_map(), *fluid_field()->dof_row_map());

    (*cfgtransform_)(fpsi_coupl()->c_fa(), (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
        mat.matrix(structure_block_, ale_i_block_),
        true);  // Addmatrix



    //-->(1)
    // add C_fp into structural equation from adding lagranean multiplier (just in case of
    // overlapping interfaces)
    fpsi_coupl()->c_fp().complete();

    (*cfptransform_)(fpsi_coupl()->c_fp().matrix(
                         0, 0),  //--> also the coupling terms from c_ff are inside here!!!
        (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
        mat.matrix(structure_block_, structure_block_),
        true);  // Addmatrix

    (*cfptransform2_)(fpsi_coupl()->c_fp().matrix(
                          0, 1),  //--> also the coupling terms from c_ff are inside here!!!
        (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf_fsi),
        mat.matrix(structure_block_, porofluid_block_),
        true);  // Addmatrix

    // FPSICoupl()->C_pf().Complete();
    // TODO: scaled by zero??
    //    (*figtransform4_)(
    //        fluid_field()->block_system_matrix()->FullRowMap(),
    //        fluid_field()->block_system_matrix()->FullColMap(),
    //        FPSICoupl()->C_pf(),
    //        timescale * 0,
    //        Adapter::CouplingSlaveConverter(coupsf_fsi),
    //        mat.Matrix(poro_block_, poro_block_),
    //        false, //no exactmatch! (just FSI Part should be extracted)
    //        true); //Add
  }

  //    //+++ part in the f...Matrix from overlap will be removed by apply_dbc of condensed velocity
  //    DOFS in linear_solve()!

  // done. make sure all blocks are filled.

  mat.complete();

  // if (FSI_Interface_exists_)
  {
    fgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(
        fbm->matrix(FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_other));
    fggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(
        fbm->matrix(FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_fsi));

    // store parts of fluid matrix to know them in the next iteration as previous iteration matrices
    fgiprev_ = fgicur_;
    fggprev_ = fggcur_;

    // store parts of fluid shape derivative matrix to know them in the next iteration as previous
    // iteration matrices
    fmgiprev_ = fmgicur_;
    fmggprev_ = fmggcur_;
    if (fluidalematrix != nullptr)
    {
      fmgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(fluidalematrix->matrix(
          FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_other));
      fmggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(fluidalematrix->matrix(
          FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_fsi));
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_vector(Core::LinAlg::Vector<double>& f,
    const Core::LinAlg::Vector<double>& sv, const Core::LinAlg::Vector<double>& pfv,
    const Core::LinAlg::Vector<double>& fv, const Core::LinAlg::Vector<double>& av,
    double fluidscale)
{
  // Get fluid_field Block Matrix
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fbm =
      fluid_field()->block_system_matrix();
  const Core::LinAlg::SparseMatrix fgg =
      fbm->matrix(FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_fsi);

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = poro_field()->structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  extractor().insert_vector(fv, fluid_block_, f);  // add fluid contributions to 'f'

  if (FSI_Interface_exists_)  // in case FSI interface exists, add term from condensation to RHS
  {
    // add fluid interface values to structure vector
    std::shared_ptr<Core::LinAlg::Vector<double>> fcvgfsi =
        fluid_field()->interface()->extract_fsi_cond_vector(fv);
    fcvgfsi->update(1.0,
        *(fluid_field()->interface()->extract_fsi_cond_vector(*fpsi_coupl()->rhs_f())),
        1.0);  // add rhs contribution of fpsi coupling rhs

    std::shared_ptr<Core::LinAlg::Vector<double>> modsv =
        poro_field()->structure_field()->interface()->insert_fsi_cond_vector(
            *fluid_to_struct_fsi(fcvgfsi));  //(fvg)fg -> (fvg)sg -> (fvg)s

    modsv->update(1.0, sv, (1.0 - stiparam) / (1.0 - ftiparam) * fluidscale);

    extractor().insert_vector(*modsv, structure_block_, f);  // add poroelast contributions to 'f'
    extractor().insert_vector(pfv, porofluid_block_, f);
  }
  else
  {
    extractor().insert_vector(sv, structure_block_, f);  // add poroelast contributions to 'f'
    extractor().insert_vector(pfv, porofluid_block_, f);
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);
  extractor().insert_vector(*aov, ale_i_block_, f);  // add ALE contributions to 'f'

  extractor().add_vector(*fpsi_coupl()->rhs_s(), structure_block_, f, 1.0);
  extractor().add_vector(*fpsi_coupl()->rhs_pf(), porofluid_block_, f, 1.0);
  extractor().add_vector(*fpsi_coupl()->rhs_f(), fluid_block_, f, 1.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_rhs_lambda(Core::LinAlg::Vector<double>& f)
{
  if (lambda_ != nullptr)  // FSI - Interface?
  {
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    const double stiparam = poro_field()->structure_field()->tim_int_param();
    const double ftiparam = fluid_field()->tim_int_param();

    // project Lagrange multiplier field onto the master interface DOFs and consider temporal
    // scaling
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        poro_field()->structure_field()->interface()->insert_fsi_cond_vector(
            *fluid_to_struct_fsi(lambda_));  //(lambda)fg -> (lambda)sg -> (lambda)s
    lambdafull->scale(stiparam - (ftiparam * (1.0 - stiparam)) / (1.0 - ftiparam));

    // add Lagrange multiplier
    extractor().add_vector(*lambdafull, structure_block_, f);
  }
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::setup_rhs_first_iter(Core::LinAlg::Vector<double>& f)
{
  // This method is directly take from FSImonolithic_fluidsplit. As at the moment no predictors are
  // considered (does not improve the convergence a lot), all terms coming from predictors are
  // commented out (for later implementation).

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = poro_field()->structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double timescale = fluid_field()->time_scaling();
  const double scale = fluid_field()->residual_scaling();

  // old interface velocity of fluid field (FSI Cond Vector)
  const std::shared_ptr<const Core::LinAlg::Vector<double>> fveln =
      fluid_field()->extract_interface_veln();

  // get fluid matrix
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockf =
      fluid_field()->block_system_matrix();

  // get fluid shape derivatives matrix
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm =
      fluid_field()->shape_derivatives();

  // get ale matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocka = ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (blockf == nullptr)
  {
    FOUR_C_THROW("Expected Teuchos::rcp to fluid block matrix.");
  }
  if (blocka == nullptr)
  {
    FOUR_C_THROW("Expected Teuchos::rcp to ale block matrix.");
  }
#endif

  // extract fluid and ale submatrices
  const Core::LinAlg::SparseMatrix& fig = blockf->matrix(
      FLD::Utils::MapExtractor::cond_other, FLD::Utils::MapExtractor::cond_fsi);  // F_{I\Gamma}
  const Core::LinAlg::SparseMatrix& fgg = blockf->matrix(
      FLD::Utils::MapExtractor::cond_fsi, FLD::Utils::MapExtractor::cond_fsi);  // F_{\Gamma\Gamma}
  // const Core::LinAlg::SparseMatrix& aig =
  // blocka->Matrix(ALE::Utils::MapExtractor::cond_other,ALE::Utils::MapExtractor::cond_fsi); //
  // A_{I\Gamma}

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs = nullptr;

  // Different contributions/terms to the rhs are separated by the following comment line
  // ---------- structural interface DOFs
  /* The following terms are added to the structural interface DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + (1-stiparam)/(1-ftiparam) * dt / tau * F_{\Gamma\Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - (1-stiparam)/(1-ftiparam) / tau * F_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
   *
   * (3)  - (1-stiparam)/(1-ftiparam) * F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(fgg.range_map(), true);

  fgg.Apply(*fveln, *rhs);

  rhs->scale(scale * (1. - stiparam) / (1. - ftiparam) * dt() * timescale);
  rhs = fluid_to_struct_fsi(rhs);
  rhs = poro_field()->structure_field()->interface()->insert_fsi_cond_vector(*rhs);
  rhs = poro_field()->extractor()->insert_vector(*rhs, 0);  // s->p

  if (poro_field()->structure_field()->get_stc_algo() == Inpar::Solid::stc_currsym)  //??ChrAg
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> stcmat =
        poro_field()->structure_field()->get_stc_mat();
    stcmat->multiply(true, *rhs, *rhs);
  }

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 1

  //   // ----------addressing term 2:
  //   rhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(fgg.RangeMap(),true));
  //
  //   fgg.Apply(*struct_to_fluid(ddgpred_), *rhs);
  //
  //   rhs->Scale(-scale * (1.-stiparam) / (1.-ftiparam) * timescale);
  //   rhs = structure_field()->Interface()->insert_fsi_cond_vector(fluid_to_struct(rhs));
  //
  //   Extractor().add_vector(*rhs,0,f);
  //   // ----------end of term 2
  //
  //   // ----------addressing term 3:
  //   if (mmm != nullptr)
  //   {
  //     // extract F^{G}_{\Gamma\Gamma}
  //     const Core::LinAlg::SparseMatrix& fmgg = mmm->Matrix(1,1);
  //
  //     rhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(fmgg.RangeMap(),true));
  //
  //     fmgg.Apply(*struct_to_fluid(ddgpred_), *rhs);
  //
  //     rhs->Scale(-(1.-stiparam) / (1.-ftiparam));
  //     rhs = structure_field()->Interface()->insert_fsi_cond_vector(fluid_to_struct(rhs));
  //
  //     Extractor().add_vector(*rhs,0,f);
  //   }
  //   // ----------end of term 3
  // ----------end of structural interface DOFs

  // ---------- inner fluid DOFs
  /* The following terms are added to the inner fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + dt / tau * F_{I \Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - 1 / tau F_{I \Gamma} * \Delta d_{\Gamma,p}
   *
   * (3)  - F^{G}_{I \Gamma} * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(fig.range_map(), true);

  fig.Apply(*fveln, *rhs);

  rhs->scale(dt() * timescale);

#ifdef FLUIDSPLITAMG
  rhs = fluid_field()->Interface()->insert_other_vector(*rhs);
#endif

  rhs = fluid_field()->interface()->insert_other_vector(*rhs);
  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 1

  //   // ----------addressing term 2
  //   rhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(fig.RangeMap(),true));
  //
  //   fig.Apply(*struct_to_fluid(ddgpred_),*rhs);
  //
  //   rhs->Scale(-timescale);
  //
  // #ifdef FLUIDSPLITAMG
  //   rhs = fluid_field()->Interface()->insert_other_vector(*rhs);
  // #endif
  //
  //   Extractor().add_vector(*rhs,1,f);
  //   // ----------end of term 2
  //
  //   // ----------addressing term 3
  //   if(mmm != nullptr)
  //   {
  //     // extract F^{G}_{I \Gamma}
  //     const Core::LinAlg::SparseMatrix& fmig = mmm->Matrix(0,1);
  //
  //     rhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(fmig.RangeMap(),true));
  //
  //     fmig.Apply(*struct_to_fluid(ddgpred_),*rhs);
  //
  //     rhs->Scale(-1.);
  //
  // #ifdef FLUIDSPLITAMG
  //   rhs = fluid_field()->Interface()->insert_other_vector(*rhs);
  // #endif
  //
  //     Extractor().add_vector(*rhs,1,f);
  //   }
  //   // ----------end of term 3
  //   // ----------end of inner fluid DOFs

  // ---------- inner ale DOFs
  /* The following terms are added to the inner ale DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - A_{I \Gamma} * \Delta d_{\Gamma,p}
   *
   */
  //   // ----------addressing term 1
  //   rhs = Teuchos::rcp(new Core::LinAlg::Vector<double>(aig.RangeMap(),true));
  //
  //   aig.Apply(*StructToAle(ddgpred_),*rhs);
  //   rhs->Scale(-1.0);
  //
  //   Extractor().add_vector(*rhs,2,f);
  //   // ----------end of term 1
  // ---------- end of inner ale DOFs

  // Reset quantities of previous iteration step since they still store values from the last time
  // step
  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *poro_field()->structure_field()->interface()->fsi_cond_map(), true);
  duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->other_map(), true);
  ddialeinc_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->interface()->other_map(), true);
  soliprev_ = nullptr;
  solgprev_ = nullptr;
  fgicur_ = nullptr;
  fggcur_ = nullptr;
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& pfx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax,  ///< ale displacements
    bool firstiter_)                                          ///< firstiteration?
{
  TEUCHOS_FUNC_TIME_MONITOR("FPSI::MonolithicPlain::extract_field_vectors");

  // porous medium
  sx = extractor().extract_vector(*x, structure_block_);
  pfx = extractor().extract_vector(*x, porofluid_block_);

  // extract inner ALE solution increment
  std::shared_ptr<const Core::LinAlg::Vector<double>> aox =
      extractor().extract_vector(*x, ale_i_block_);

  // put inner --- ALE solution together
  std::shared_ptr<Core::LinAlg::Vector<double>> a =
      ale_field()->interface()->insert_other_vector(*aox);
  // ale_field()->Interface()->insert_fpsi_cond_vector(acx_fpsi, a); //Already done by
  // Ale().apply_interface_displacements() ale_field()->Interface()->insert_fsi_cond_vector(acx_fsi,
  // a);
  // //Already done by Ale().apply_interface_displacements()
  ax = a;  // displacement on the interface is zero!!!

  // ---------------------------------------------------------------------------
  // process fluid unknowns
  // ---------------------------------------------------------------------------
  // extract inner fluid solution increment from NOX increment
  std::shared_ptr<Core::LinAlg::Vector<double>> f = extractor().extract_vector(*x, fluid_block_);
#ifdef FLUIDSPLITAMG
  fox = fluid_field()->Interface()->extract_other_vector(*fox);
#endif

  if (FSI_Interface_exists_)
  {
    // convert structure solution increment to ALE solution increment at the interface

    std::shared_ptr<Core::LinAlg::Vector<double>> scx_fsi =
        poro_field()->structure_field()->interface()->extract_fsi_cond_vector(*sx);
    if (firstiter_)  // to consider also DBC on Structure!!!
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> dispnfsi =
          poro_field()->structure_field()->interface()->extract_fsi_cond_vector(
              *poro_field()->structure_field()->dispn());
      std::shared_ptr<Core::LinAlg::Vector<double>> dispnpfsi =
          poro_field()->structure_field()->interface()->extract_fsi_cond_vector(
              *poro_field()->structure_field()->dispnp());
      scx_fsi->update(1.0, *dispnpfsi, -1.0, *dispnfsi, 1.0);
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> acx_fsi = struct_to_ale_fsi(scx_fsi);

    // convert ALE solution increment to fluid solution increment at the interface
    std::shared_ptr<Core::LinAlg::Vector<double>> fcx_fsi = ale_to_fluid_interface_fsi(acx_fsi);

    if (firstiter_)
      fluid_field()->displacement_to_velocity(
          fcx_fsi);  // Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
    else
      fcx_fsi->scale(fluid_field()->time_scaling());  // Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1)

    fluid_field()->interface()->insert_fsi_cond_vector(*fcx_fsi, *f);
    // ---------------------------------------------------------------------------

    // Store field vectors to know them later on as previous quantities
    // inner ale displacement increment
    // interface structure displacement increment
    if (disgprev_ != nullptr)
      ddginc_->update(1.0, *scx_fsi, -1.0, *disgprev_, 0.0);  // compute current iteration increment
    else
      ddginc_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*scx_fsi);  // first iteration increment

    disgprev_ = scx_fsi;  // store current step increment
    // ------------------------------------
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> fox =
      fpsi_coupl()->fluid_fsi_fpsi_extractor()->extract_vector(*f, 0);

  fx = f;
  // inner ale displacement increment
  if (solialeprev_ != nullptr)
    ddialeinc_->update(1.0, *aox, -1.0, *solialeprev_, 0.0);  // compute current iteration increment
  else
    ddialeinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*aox);  // first iteration increment

  solialeprev_ = aox;  // store current step increment
  // ------------------------------------

  // fluid solution increment
  if (soliprev_ != nullptr)  // compute current iteration increment
    duiinc_->update(1.0, *fox, -1.0, *soliprev_, 0.0);
  else
    // first iteration increment
    duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fox);
  // store current step increment
  soliprev_ = fox;
  // ------------------------------------
}

/*----------------------------------------------------------------------*/
/* Recover the Lagrange multiplier at the interface   ... FPSI adapted version of mayr.mt (03/2012)
 */
/*----------------------------------------------------------------------*/
void FPSI::MonolithicPlain::recover_lagrange_multiplier()
{
  // For overlapping interfaces also the FPSI Coupling Matrixes should be considered in the
  // condensation procedure ... not done yet!!!!

  // This method is directly take from FSImonolithic_fluidsplit. As at the moment no predictors are
  // considered (does not improve the convergence a lot), all terms coming from predictors are
  // commented out (for later implementation).

  if (lambda_ != nullptr)  // FSI - Interface?
  {
    // get time integration parameter of fluid time integrator
    // to enable consistent time integration among the fields
    const double ftiparam = fluid_field()->tim_int_param();

    // some scaling factors for fluid
    const double timescale = fluid_field()->time_scaling();
    const double scale = fluid_field()->residual_scaling();

    // get fluid shape derivative matrix
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm =
        fluid_field()->shape_derivatives();

    // some often re-used vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
        nullptr;  // stores intermediate result of terms (3)-(8)
    std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;     // just for convenience
    std::shared_ptr<Core::LinAlg::Vector<double>> auxauxvec = nullptr;  // just for convenience

    /* Recovery of Lagrange multiplier \lambda_^{n+1} is done by the following
     * condensation expression:
     *
     * lambda_^{n+1} =
     *
     * (1)  - ftiparam / (1.-ftiparam) * lambda^{n}
     *
     * (2)  - 1. / (1.-ftiparam) * tmpvec
     *
     * with tmpvec =
     *
     * (3)    r_{\Gamma}^{F,n+1}
     *
     * (4)  + 1 / tau * F_{\Gamma\Gamma} * \Delta d_{\Gamma}^{S,n+1}
     *
     * (5)  + F_{\Gamma\Gamma}^{G} * \Delta d_{\Gamma}^{S,n+1}
     *
     * (6)  + F_{\Gamma I} * \Delta u_{I}^{F,n+1}
     *
     * (7)  + F_{\Gamma I}^{G} * \Delta d_{I}^{G,n+1}
     *
     * (8)  + dt / tau * F_{\Gamma\Gamma} * u_{\Gamma}^n]
     *
     * Remark on term (8):
     * Term (8) has to be considered only in the first Newton iteration.
     * Hence, it will usually not be computed since in general we need more
     * than one nonlinear iteration until convergence.
     *
     * Remarks on all terms:
     * +  Division by -(1.0 - ftiparam) will be done in the end
     *    since this is common to all terms
     * +  tau: time scaling factor for interface time integration (tau =
     * 1/fluid_field()->TimeScaling())
     * +  neglecting terms (4)-(8) should not alter the results significantly
     *    since at the end of the time step the solution increments tend to zero.
     *
     *                                                 Matthias Mayr (10/2012)
     */

    // ---------Addressing term (1)
    lambda_->update(ftiparam, *lambda_, 0.0);
    // ---------End of term (1)

    // ---------Addressing term (3)
    std::shared_ptr<Core::LinAlg::Vector<double>> fluidresidual =
        fluid_field()->interface()->extract_fsi_cond_vector(*fluid_field()->rhs());
    fluidresidual->scale(-1.0);  // invert sign to obtain residual, not rhs
    tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(*fluidresidual);
    // ---------End of term (3)

    // ---------Addressing term (4)
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fggprev_->range_map(), true);

    fggprev_->Apply(*struct_to_fluid_fsi(ddginc_), *auxvec);
    tmpvec->update(timescale, *auxvec, 1.0);
    // ---------End of term (4)

    // ---------Addressing term (5)
    if (fmggprev_ != nullptr)
    {
      auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmggprev_->range_map(), true);
      fmggprev_->Apply(*struct_to_fluid_fsi(ddginc_), *auxvec);
      tmpvec->update(1.0, *auxvec, 1.0);
    }
    // ---------End of term (5)

    // ---------Addressing term (6)
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fgiprev_->range_map(), true);
    Core::LinAlg::Vector<double> tmp(fgiprev_->domain_map(), true);
    Core::LinAlg::export_to(*duiinc_, tmp);
    fgiprev_->Apply(tmp, *auxvec);
    tmpvec->update(1.0, *auxvec, 1.0);
    // ---------End of term (6)

    // ---------Addressing term (7)
    if (fmgiprev_ != nullptr)
    {
      /* For matrix-vector-product, the DomainMap() of the matrix and the Map() of the vector
       * have to match. DomainMap() contains inner velocity DOFs and all pressure DOFs.
       * The inner ale displacement increment is converted to the fluid map using AleToFluid().
       * This results in a map that contains all velocity but no pressure DOFs.
       *
       * We have to circumvent some trouble with Core::LinAlg::Maps since we cannot split
       * an Core::LinAlg::Map into inner and interface DOFs.
       *
       * We create a map extractor 'velothermap' in order to extract the inner velocity
       * DOFs after calling AleToFluid(). Afterwards, a second map extractor
       * 'velotherpressuremapext' is used to append pressure DOFs filled with zeros.
       *
       * Finally, maps match and matrix-vector-multiplication can be done.
       */

      // extract inner velocity DOFs after calling AleToFluid()
      std::shared_ptr<Core::LinAlg::Map> velothermap = Core::LinAlg::split_map(
          *fluid_field()->velocity_row_map(), *interface_fluid_ale_coupling_fsi().master_dof_map());
      Core::LinAlg::MapExtractor velothermapext =
          Core::LinAlg::MapExtractor(*fluid_field()->velocity_row_map(), velothermap, false);
      auxvec = std::make_shared<Core::LinAlg::Vector<double>>(*velothermap, true);
      velothermapext.extract_other_vector(
          *ale_to_fluid(ale_field()->interface()->insert_other_vector(*ddialeinc_)), *auxvec);

      // add pressure DOFs
      Core::LinAlg::MapExtractor velotherpressuremapext =
          Core::LinAlg::MapExtractor(fmgiprev_->domain_map(), velothermap);
      auxauxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmgiprev_->domain_map(), true);
      velotherpressuremapext.insert_cond_vector(*auxvec, *auxauxvec);

      // prepare vector to store result of matrix-vector-product
      auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmgiprev_->range_map(), true);

      // Now, do the actual matrix-vector-product
      fmgiprev_->Apply(*auxauxvec, *auxvec);
      tmpvec->update(1.0, *auxvec, 1.0);
    }
    // ---------End of term (7)

    // ---------Addressing term (8)
    if (firstcall_)
    {
      auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fggprev_->range_map(), true);
      fggprev_->Apply(*fluid_field()->extract_interface_veln(), *auxvec);
      tmpvec->update(dt() * timescale, *auxvec, 1.0);
    }
    // ---------End of term (8)

    // ---------Addressing term (2)
    lambda_->update(scale, *tmpvec, 1.0);  // scale with residual_scaling() to get [N/m^2]
    // ---------End of term (2)

    // Finally, divide by (1.0-ftiparam) which is common to all terms
    lambda_->scale(-1.0 / (1.0 - ftiparam));

    // Finally, the Lagrange multiplier 'lambda_' is recovered here.
    // It represents nodal forces acting onto the structure.
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE
