// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_fluidfluidmonolithic_fluidsplit_nonox.hpp"

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
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

FSI::FluidFluidMonolithicFluidSplitNoNOX::FluidFluidMonolithicFluidSplitNoNOX(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : MonolithicNoNOX(comm, timeparams)
{
  // Determine fluid (=slave) DOF on the FSI interface, for which a Dirichlet boundary
  // condition (DBC) has been prescribed
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;

  // Fluid DBC-DOFs
  intersectionmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // Fluid interface DOF
  intersectionmaps.push_back(fluid_field()->interface()->fsi_cond_map());

  // intersect
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  // It is not allowed, that slave DOFs at the interface hold a Dirichlet
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
             << "  | This is a fluid split scheme. Hence, master and slave field are chosen as "
                "follows:          |"
             << std::endl
             << "  |     MASTER  = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |     SLAVE   = FLUID                                                           "
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

  // Initialization of row/column transformation objects
  // These are needed for the system matrix setup,
  // as matrices from 3 different fields (S,F,A) are set together.
  fggtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  fmggtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();

  fgitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  figtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmiitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmgitransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  aigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  // Lagrange multiplier
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);

  // Storage for matrices from previous time steps
  fggcur_ = nullptr;
  fgicur_ = nullptr;
  fmggcur_ = nullptr;
  fmgicur_ = nullptr;

  // Structural predictor step, initially filled with zeros
  ddgpred_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
}


/*----------------------------------------------------------------------
 * SetupSystem:
 *    - Setup field coupling
 *    - System matrix initialization
 *    - Build of combined DOF map, including all DOFs
 *      except from the fluid FSI DOFs
 *----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_system()
{
  FSI::MonolithicNoNOX::setup_system();

  // Create a combined map for Structure/Fluid/ALE-DOFs all in one
  create_combined_dof_row_map();

  // Tell fluid field to split the fluid system matrix
  // (indicate the fluid split)
  fluid_field()->use_block_matrix(true);

  // Build the ALE-matrix in split system
  ale_field()->create_system_matrix(ale_field()->interface());

  // Initialize the global system matrix!
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 81, false, true);
}

/*----------------------------------------------------------------------
 * setup_rhs:
 *   - Build the RHS-vector for the Newton-loop!
 *   - The RHS-vector is made up of 2 parts:
 *     the single-field RHS-contributions and special
 *     terms resulting from condensation of fluid-DOFs from the FSI interface
 *     and predictor steps. These terms are added at the first Newton
 *     step only!
 *----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_rhs(
    Core::LinAlg::Vector<double>& f, bool firstcall)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (fluid_field()->rhs() == nullptr) FOUR_C_THROW("empty fluid residual");
#endif

  // Get the contributions from the field residuals
  setup_vector(f, *structure_field()->rhs(), *fluid_field()->rhs(), *ale_field()->rhs(),
      fluid_field()->residual_scaling());

  /*----------------------------------------------------------------------
  The following terms are added only at the first Newton iteration!
  ----------------------------------------------------------------------*/

  if (firstcall)
  {
    // Store fluid interface velocity:
    std::shared_ptr<const Core::LinAlg::Vector<double>> fveln =
        fluid_field()->extract_interface_veln();

    /*----------------------------------------------------------------------*/
    // Time integration parameters
    /*----------------------------------------------------------------------*/
    // Structure:
    // a*x_n+(1-a)*x_n+1
    // Fluid:
    // b*y_n+(1-b)*y_n+1
    // a: stimintparam
    // b: ftimintparam
    /*----------------------------------------------------------------------*/

    const double stimintparam = structure_field()->tim_int_param();
    const double ftimintparam = fluid_field()->tim_int_param();

    // Fluid time scaling parameter
    // fluidtimescale: \tau
    //  \tau = 1/\Delta t  for Backward-Euler;
    //  \tau = 2/\Delta t  for Trapezoidal Rule
    const double fluidtimescale = fluid_field()->time_scaling();
    const double fluidresidualscale = fluid_field()->residual_scaling();

    // Get the fluid block system matrix "blockf"
    // shape derivative matrix (linearization
    // of Navier-Stokes with respect to mesh movement: moving mesh matrix "mmm")
    // ALE block matrix "blocka"
    // F_...

    // Fluid block system matrix
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockf =
        fluid_field()->block_system_matrix();
    if (blockf == nullptr) FOUR_C_THROW("Expected fluid block matrix...");

    // F^G_...
    // Fluid shape derivative matrix
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm =
        fluid_field()->shape_derivatives();

    // A_...
    // ALE block system matrix
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocka =
        ale_field()->block_system_matrix();
    if (blocka == nullptr) FOUR_C_THROW("Expected ALE block matrix...");

    // Extracting submatrices for fluid & ALE from block field matrices:
    // F_{\Gamma\Gamma}, F_{I\Gamma} and A_{I\Gamma}
    const Core::LinAlg::SparseMatrix& fgg = blockf->matrix(1, 1);
    const Core::LinAlg::SparseMatrix& fig = blockf->matrix(0, 1);
    const Core::LinAlg::SparseMatrix& aig = blocka->matrix(0, 1);

    // Vector for storage of the temporary result for the RHS vector
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs = nullptr;

    /*----------------------------------------------------------------------*/
    // Starting the setup!
    /*----------------------------------------------------------------------*/

    // Step 1: Taking care of the  DOFs @ structural side
    /*
     * (1)  + (1-stintparam)/(1-flintparam)* \Delta t * timescale * F_{\Gamma\Gamma} *
     * u^{n}_{\Gamma}
     *
     * (2)  - (1-stintparam)/(1-flintparam) * F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
     *
     * (3)  - (1-stintparam)/(1-flintparam) * timescale * F_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
     *
     *
     */

    // ------------------
    // Build term (1):
    // ------------------

    // Create zero-filled vector copy based on the row map of F_{\Gamma\Gamma}
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fgg.range_map(), true);

    // Compute F_{\Gamma\Gamma}*u^n_\Gamma
    // Write into rhs
    fgg.Apply(*fveln, *rhs);

    // Apply scaling
    rhs->scale(
        fluidresidualscale * (1.0 - stimintparam) / (1.0 - ftimintparam) * dt() * fluidtimescale);

    // Insert into structural side of the interface
    rhs = fluid_to_struct(rhs);
    rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

    // Add to the structure block (index 0) of the final RHS vector f
    extractor().add_vector(*rhs, 0, f);

    if (mmm != nullptr)
    {
      // ------------------
      // Build term (2):
      // ------------------

      // Extract the matrix F^G_\Gamma_\Gamma
      const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

      // Re-initialize rhs
      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.range_map(), true);


      // Compute F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
      // Write into rhs
      fmgg.Apply(*struct_to_fluid(ddgpred_), *rhs);

      // Apply scaling
      rhs->scale(-1.0 * (1.0 - stimintparam) / (1.0 - ftimintparam));

      // Insert into structure side of the interface
      rhs = fluid_to_struct(rhs);
      rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

      extractor().add_vector(*rhs, 0, f);
    }

    // ------------------
    // Build term (3):
    // ------------------

    // Re-initialize rhs
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fgg.range_map(), true);

    // Compute F_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    fgg.Apply(*struct_to_fluid(ddgpred_), *rhs);

    // Apply scaling
    rhs->scale(
        -1.0 * fluidresidualscale * (1.0 - stimintparam) / (1.0 - ftimintparam) * fluidtimescale);

    // Insert into structure side of the interface
    rhs = fluid_to_struct(rhs);
    rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

    extractor().add_vector(*rhs, 0, f);


    // Step 2: Fluid sided inner DOFs
    /*
     * (1) F_{I\Gamma} * \Delta t* timescale * u^{n}_{\Gamma}
     *
     * (2)  - timescale * F_{I\Gamma} * \Delta d_{\Gamma,p}
     *
     * (3) - F^{G}_{I\Gamma} * \Delta d_{\Gamma,p}
     *
     */

    // ------------------
    // Build term (1):
    // ------------------

    // Re-initialize rhs
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fig.range_map(), true);

    // Compute term F_{I\Gamma} *u^{n}_{\Gamma}
    // Write into rhs
    fig.Apply(*fveln, *rhs);

    // Apply scaling
    rhs->scale(fluidtimescale * dt());

    // Add to the final vector f, to the fluid block (index 1)
    extractor().add_vector(*rhs, 1, f);


    // ------------------
    // Build term (2):
    // ------------------

    // Re-initialize rhs
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fig.range_map(), true);

    // Compute term F_{I\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    fig.Apply(*struct_to_fluid(ddgpred_), *rhs);

    // Apply scaling
    rhs->scale(-fluidtimescale);

    extractor().add_vector(*rhs, 1, f);

    if (mmm != nullptr)
    {
      // ------------------
      // Build term (3):
      // ------------------

      // Extract the matrix F^G_I\Gamma
      const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);

      // Re-initialize rhs
      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.range_map(), true);

      // Compute F^{G}_{I\Gamma} * \Delta d_{\Gamma,p}
      // Write into rhs
      fmig.Apply(*struct_to_fluid(ddgpred_), *rhs);

      // Apply scaling
      rhs->scale(-1.0);

      extractor().add_vector(*rhs, 1, f);
    }

    // Step 3: Inner ALE DOFs
    //
    // Adding terms to ALE-Part of RHS-vector
    // -A_{I\Gamma} * \Delta d_{\Gamma,p}

    // Re-initialize rhs
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.range_map(), true);

    // Compute term A_{I\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    aig.Apply(*struct_to_ale(ddgpred_), *rhs);

    // Apply scaling
    rhs->scale(-1.0);

    extractor().add_vector(*rhs, 2, f);
  }
}

/*----------------------------------------------------------------------
 * setup_system_matrix:
 *
 *   - Build the final system block matrix extracting, scaling
 *     & transforming the single field submatrices
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_system_matrix()
{
  const Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  const Coupling::Adapter::Coupling& coupsa = structure_ale_coupling();
  const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

  // get single field block matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> s =
      structure_field()->system_matrix();  // can't be 'const' --> is modified by STC
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> f =
      fluid_field()->block_system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (s == nullptr)
  {
    FOUR_C_THROW("expect structure block matrix");
  }
  if (f == nullptr)
  {
    FOUR_C_THROW("expect fluid block matrix");
  }
  if (a == nullptr)
  {
    FOUR_C_THROW("expect ale block matrix");
  }
#endif

  // extract submatrices
  Core::LinAlg::SparseMatrix& fii = f->matrix(0, 0);
  Core::LinAlg::SparseMatrix& fig = f->matrix(0, 1);
  Core::LinAlg::SparseMatrix& fgi = f->matrix(1, 0);
  Core::LinAlg::SparseMatrix& fgg = f->matrix(1, 1);
  Core::LinAlg::SparseMatrix& aii = a->matrix(0, 0);
  Core::LinAlg::SparseMatrix& aig = a->matrix(0, 1);

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  /*----------------------------------------------------------------------*/

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  // this just once...
  s->un_complete();

  (*fggtransform_)(fgg, (1.0 - stiparam) / (1.0 - ftiparam) * scale * timescale,
      Coupling::Adapter::CouplingSlaveConverter(coupsf),
      Coupling::Adapter::CouplingSlaveConverter(coupsf), *s, true, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> lfgi =
      std::make_shared<Core::LinAlg::SparseMatrix>(s->row_map(), 81, false);
  (*fgitransform_)(fgi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
      Coupling::Adapter::CouplingSlaveConverter(coupsf), *lfgi);

  lfgi->complete(fgi.domain_map(), s->range_map());

  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::View, *lfgi);

  Core::LinAlg::SparseMatrix lfig(fig.row_map(), 81, false);
  (*figtransform_)(f->full_row_map(), f->full_col_map(), fig, timescale,
      Coupling::Adapter::CouplingSlaveConverter(coupsf), systemmatrix_->matrix(1, 0));

  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::View, fii);

  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1.,
      Coupling::Adapter::CouplingSlaveConverter(coupsa), systemmatrix_->matrix(2, 0));

  systemmatrix_->assign(2, 2, Core::LinAlg::DataAccess::View, aii);

  /*----------------------------------------------------------------------*/
  // add optional blocks from fluid linearization with respect to mesh motion

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != nullptr)
  {
    Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);

    Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    // reuse transform objects to add shape derivative matrices to structural blocks
    (*figtransform_)(f->full_row_map(), f->full_col_map(), fmig, 1.,
        Coupling::Adapter::CouplingSlaveConverter(coupsf), systemmatrix_->matrix(1, 0), false,
        true);


    (*fmggtransform_)(fmgg, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Coupling::Adapter::CouplingSlaveConverter(coupsf),
        Coupling::Adapter::CouplingSlaveConverter(coupsf), *s, false, true);

    // We cannot copy the pressure value. It is not used anyway. So no exact
    // match here.
    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), systemmatrix_->matrix(1, 2), false);

    {
      Core::LinAlg::SparseMatrix lfmgi(s->row_map(), 81, false);
      (*fmgitransform_)(fmgi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
          Coupling::Adapter::CouplingSlaveConverter(coupsf),
          Coupling::Adapter::CouplingMasterConverter(coupfa), lfmgi, false, false);

      lfmgi.complete(aii.domain_map(), s->range_map());

      systemmatrix_->assign(0, 2, Core::LinAlg::DataAccess::View, lfmgi);
    }
  }

  // finally assign structure block
  systemmatrix_->matrix(0, 0).assign(Core::LinAlg::DataAccess::View, *s);

  // done. make sure all blocks are filled.
  systemmatrix_->complete();

  // Store some submatrices required for RHS-Setup
  // to know them for LM-recovery!
  // store parts of fluid matrix to know them in the next iteration as previous iteration matrices
  fgiprev_ = fgicur_;
  fggprev_ = fggcur_;
  fgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(1, 0));
  fggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(1, 1));

  // store parts of fluid shape derivative matrix to know them in the next iteration as previous
  // iteration matrices
  fmgiprev_ = fmgicur_;
  fmggprev_ = fmggcur_;
  if (mmm != nullptr)
  {
    fmgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(mmm->matrix(1, 0));
    fmggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(mmm->matrix(1, 1));
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::FluidFluidMonolithicFluidSplitNoNOX::initial_guess(
    std::shared_ptr<Core::LinAlg::Vector<double>> ig)
{
  setup_vector(*ig, *structure_field()->initial_guess(), *fluid_field()->initial_guess(),
      *ale_field()->initial_guess(), 0.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> FSI::FluidFluidMonolithicFluidSplitNoNOX::combined_dbc_map()
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
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_vector(Core::LinAlg::Vector<double>& f,
    const Core::LinAlg::Vector<double>& sv, const Core::LinAlg::Vector<double>& fv,
    const Core::LinAlg::Vector<double>& av, double fluidscale)
{
  // Writes the following entries into f :

  // [
  // f^S_I
  // -----
  // f^S_{\Gamma}+
  // (1-stintparam_)/(1-flintparam)*fluidscale*f^F_{\Gamma}
  // lambda*(stintparam_-flintparam*(1-stintparam_)/(1-flintparam)
  // -----
  // f^ F_I
  // -----
  // 0
  // ]

  /*----------------------------------------------------------------------*/
  // Time integration parameters
  /*----------------------------------------------------------------------*/
  // Structure:
  // a*x_n+(1-a)*x_n+1
  // Fluid:
  // b*y_n+(1-b)*y_n+1
  // a: stimintparam
  // b: ftimintparam
  /*----------------------------------------------------------------------*/

  const double stimintparam = structure_field()->tim_int_param();
  const double ftimintparam = fluid_field()->tim_int_param();


  // Extract inner DOFs for ALE-field
  std::shared_ptr<Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);

  // Get the FSI-interface RHS-vector for the fluid side!
  std::shared_ptr<Core::LinAlg::Vector<double>> fcv =
      fluid_field()->interface()->extract_fsi_cond_vector(fv);

  // Convert previously extracted vector to structure !
  std::shared_ptr<Core::LinAlg::Vector<double>> modsv =
      structure_field()->interface()->insert_fsi_cond_vector(*fluid_to_struct(fcv));

  // Add the converted interface RHS-contributions (scaled) to the global structural RHS!
  int err = modsv->update(1.0, sv, (1.0 - stimintparam) / (1.0 - ftimintparam) * fluidscale);
  if (err) FOUR_C_THROW("Update of structural residual vector failed! Error code {}", err);

  // Add the previous Lagrange Multiplier
  if (lambda_ != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaglob =
        structure_field()->interface()->insert_fsi_cond_vector(*fluid_to_struct(lambda_));
    err = modsv->update(stimintparam - ftimintparam * (1.0 - stimintparam) / (1.0 - ftimintparam),
        *lambdaglob, 1.0);
    if (err) FOUR_C_THROW("Update of structural residual vector failed! Error code {}", err);

    // Insert structural contribution
    extractor().insert_vector(*modsv, 0, f);
  }
  else
  {
    extractor().insert_vector(sv, 0, f);
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> fglobalv =
      fluid_field()->interface()->extract_other_vector(fv);

  // Insert fluid contribution
  extractor().insert_vector(*fglobalv, 1, f);

  // Insert ALE contribution
  extractor().insert_vector(*aov, 2, f);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
  /*----------------------------------------------------------------------*/
  // Process structure unknowns
  /*----------------------------------------------------------------------*/
  // Extract whole structure field vector
  sx = extractor().extract_vector(*x, 0);

  // Structural part of FSI interface
  std::shared_ptr<Core::LinAlg::Vector<double>> scx =
      structure_field()->interface()->extract_fsi_cond_vector(*sx);

  /*----------------------------------------------------------------------*/
  // Process ALE unknowns
  /*----------------------------------------------------------------------*/
  std::shared_ptr<const Core::LinAlg::Vector<double>> aox = extractor().extract_vector(*x, 2);
  // Update interface part of structure vector with predictor increment
  scx->update(1.0, *ddgpred_, 1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> acx = struct_to_ale(scx);

  std::shared_ptr<Core::LinAlg::Vector<double>> a =
      ale_field()->interface()->insert_other_vector(*aox);
  // Insert the FSI-DOF vector into full vector a
  ale_field()->interface()->insert_fsi_cond_vector(*acx, *a);
  // Write a into passed argument ax
  ax = a;

  /*----------------------------------------------------------------------*/
  // Process fluid unknowns
  /*----------------------------------------------------------------------*/
  // Extract vector of fluid unknowns from x
  std::shared_ptr<const Core::LinAlg::Vector<double>> fox = extractor().extract_vector(*x, 1);

  // Conversion ALE displacement to fluid field:
  std::shared_ptr<Core::LinAlg::Vector<double>> fcx = ale_to_fluid_interface(acx);
  fluid_field()->displacement_to_velocity(fcx);


  // The previously computed fluid interface values have to be inserted into the fluid field vector
  std::shared_ptr<Core::LinAlg::Vector<double>> f =
      fluid_field()->interface()->insert_other_vector(*fox);
  fluid_field()->interface()->insert_fsi_cond_vector(*fcx, *f);

  fx = f;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::read_restart(int step)
{
  // Read Lagrange Multiplier (associated with embedded fluid)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaemb =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *(fluid_field()->x_fluid_fluid_map_extractor()->fluid_map()), true);
    Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
        fluid_field()->discretization(), Global::Problem::instance()->input_control_file(), step);
    reader.read_vector(lambdaemb, "fsilambda");
    // Insert into vector containing the whole merged fluid DOF
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->x_fluid_fluid_map_extractor()->insert_fluid_vector(*lambdaemb);
    lambda_ = fluid_field()->interface()->extract_fsi_cond_vector(*lambdafull);
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::output()
{
  structure_field()->output();
  fluid_field()->output();

  // output Lagrange multiplier
  {
    // the Lagrange multiplier lives on the FSI interface
    // for output, we want to insert lambda into a full vector, defined on the embedded fluid field
    // 1. insert into vector containing all fluid DOF
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->interface()->insert_fsi_cond_vector(*lambda_);

    // 2. extract the embedded fluid part
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaemb =
        fluid_field()->x_fluid_fluid_map_extractor()->extract_fluid_vector(*lambdafull);

    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const int uprestart = fsidyn.get<int>("RESTARTEVERY");
    const int upres = fsidyn.get<int>("RESULTSEVERY");
    if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
        fluid_field()->step() % upres == 0)
      fluid_field()->disc_writer()->write_vector("fsilambda", lambdaemb);
  }
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
void FSI::FluidFluidMonolithicFluidSplitNoNOX::create_combined_dof_row_map()
{
  // Create a combined map for Structure/Fluid/ALE-DOFs all in one
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  // Append the structural DOF map
  vecSpaces.push_back(structure_field()->dof_row_map());

  // Append the final fluid DOF map, free of FSI DOF
  vecSpaces.push_back(fluid_field()->interface()->other_map());

  // Append ALE DOF map
  vecSpaces.push_back(ale_field()->interface()->other_map());

  // If the non-FSI fluid maps are empty
  if (vecSpaces[1]->num_global_elements() == 0)
    FOUR_C_THROW("No inner fluid equations. Can't split!");

  // The vector is complete, fill the system's global BlockRowMap
  // with the maps previously set together!
  set_dof_row_maps(vecSpaces);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::newton()
{
  /*----------------------------------------------------------------------
  Extract predictor increments
  ----------------------------------------------------------------------*/
  // Increment of structural interface displacement --> structural predictor!!
  ddgpred_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->extract_interface_dispnp());
  ddgpred_->update(-1.0, *structure_field()->extract_interface_dispn(), 1.0);

  /*----------------------------------------------------------------------*/
  // Initialize the increment vectors, they are updated in evaluate(...)->extract_field_vectors(...)
  // at every Newton iteration!

  // Initialization for 1st Newton call
  // structural interface predictor
  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*ddgpred_);
  ddialeinc_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->interface()->other_map());
  duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->other_map(), true);

  FSI::MonolithicNoNOX::newton();

  /*----------------------------------------------------------------------*/
  // Compute the increments needed for recovery of Lagrange Multiplier!
  // After the last Newton iteration, the increments are not updated.
  // We need the last increment for the recovery of lambda.
  /*----------------------------------------------------------------------*/
  // Fluid
  duiinc_->update(1.0, *extractor().extract_vector(*iterinc_, 1), 0.0);
  // Structure
  std::shared_ptr<Core::LinAlg::Vector<double>> ddinc = extractor().extract_vector(*iterinc_, 0);
  ddginc_->update(1.0, *structure_field()->interface()->extract_fsi_cond_vector(*ddinc), 0.0);
  // ALE
  ddialeinc_->update(1.0, *extractor().extract_vector(*iterinc_, 2), 0.0);
}

/*----------------------------------------------------------------------
 * build_convergence_norms:
 *     - Calculate the residual and incremental norms required for
 *        the convergence test in Newton-loop
 *     - Implemented:
 *         The (Euclidean) L2-Norm
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::build_convergence_norms()
{
  /*----------------------------------------------------------------------
  Residual norms - L2 of:
  - global rhs
  - inner structural rhs
  - ale rhs
  - inner fluid-fluid velocity rhs
  - inner fluid-fluid pressure rhs
  - complete interface residual
  ----------------------------------------------------------------------*/
  // Norm of global RHS vector
  rhs_->norm_2(&normrhs_);

  // Inner structural RHS and interface RHS

  // RHS-vector from fluid_field() without FSI-DOFs
  std::shared_ptr<const Core::LinAlg::Vector<double>> innerfluidfluidrhs =
      extractor().extract_vector(*rhs_, 1);
  // (Inner) ALE RHS
  std::shared_ptr<const Core::LinAlg::Vector<double>> alerhs = extractor().extract_vector(*rhs_, 2);

  // Norm of inner structural residual forces
  std::shared_ptr<const Core::LinAlg::Vector<double>> structrhs =
      extractor().extract_vector(*rhs_, 0);
  structure_field()->interface()->extract_other_vector(*structrhs)->norm_2(&normstrrhsL2_);
  structure_field()->interface()->extract_other_vector(*structrhs)->norm_inf(&normstrrhsInf_);

  // Norm of ALE residual forces
  alerhs->norm_2(&normalerhsL2_);

  // Norm of fluid velocity residual
  // This requires an Core::LinAlg::Map of the inner fluid velocity DOFs first!
  std::shared_ptr<const Core::LinAlg::Map> innerfluidvel = fluid_field()->inner_velocity_row_map();

  // Merged FSI-free fluid maps
  std::shared_ptr<const Core::LinAlg::Map> fluidmaps = fluid_field()->interface()->other_map();
  // Create a MapExtractor to access the velocity DOFs from the FSI-free fluid map
  std::shared_ptr<Core::LinAlg::MapExtractor> fluidvelextract =
      std::make_shared<Core::LinAlg::MapExtractor>(*fluidmaps, innerfluidvel, true);

  // Finally, compute the fluid velocity RHS-norm
  fluidvelextract->extract_cond_vector(*innerfluidfluidrhs)->norm_2(&normflvelrhsL2_);
  fluidvelextract->extract_cond_vector(*innerfluidfluidrhs)->norm_inf(&normflvelrhsInf_);

  // Norm of fluid pressure residual
  // This requires an Core::LinAlg::Map of the fluid pressure DOFs
  if (fluid_field()->pressure_row_map() == nullptr) FOUR_C_THROW("Empty pressure row map!");

  // Finally, compute the fluid pressure RHS-norm
  fluidvelextract->extract_other_vector(*innerfluidfluidrhs)->norm_2(&normflpresrhsL2_);
  fluidvelextract->extract_other_vector(*innerfluidfluidrhs)->norm_inf(&normflpresrhsInf_);

  // The true RHS for the FSI interface equation block consists of
  // more than just the structure residual, namely the scaled fluid interface residual and the
  // previous Lagrange multiplier. The first idea is, to test this whole term, which can be easily
  // extracted from rhs_. For a more strict testing, the L_inf-norm should be employed!
  std::shared_ptr<Core::LinAlg::Vector<double>> interfaceresidual =
      structure_field()->interface()->extract_fsi_cond_vector(*structrhs);
  interfaceresidual->norm_2(&norminterfacerhsL2_);
  interfaceresidual->norm_inf(&norminterfacerhsInf_);

  /*----------------------------------------------------------------------
  Incremental norms - L2
  - global inc
  - inner structural inc
  - complete interface inc
  - inner fluid-fluid velocity rhs
  - inner fluid-fluid pressure rhs
  ----------------------------------------------------------------------*/
  // Norm of global increment vector
  iterinc_->norm_2(&norminc_);

  std::shared_ptr<const Core::LinAlg::Vector<double>> structinc =
      extractor().extract_vector(*iterinc_, 0);
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidinc =
      extractor().extract_vector(*iterinc_, 1);

  // Norm of inner structural increment vector
  structure_field()->interface()->extract_other_vector(*structinc)->norm_2(&normstrincL2_);
  structure_field()->interface()->extract_other_vector(*structinc)->norm_inf(&normstrincInf_);

  // Norm of interface increment vector
  structure_field()->interface()->extract_fsi_cond_vector(*structinc)->norm_2(&norminterfaceincL2_);
  structure_field()
      ->interface()
      ->extract_fsi_cond_vector(*structinc)
      ->norm_inf(&norminterfaceincInf_);

  // Norm of fluid velocity increment
  fluidvelextract->extract_cond_vector(*fluidinc)->norm_2(&normflvelincL2_);
  fluidvelextract->extract_cond_vector(*fluidinc)->norm_inf(&normflvelincInf_);

  // Norm of fluid pressure increment
  fluidvelextract->extract_other_vector(*fluidinc)->norm_2(&normflpresincL2_);
  fluidvelextract->extract_other_vector(*fluidinc)->norm_inf(&normflpresincInf_);

  // Norm of ALE increment vector
  extractor().extract_vector(*iterinc_, 2)->norm_2(&normaleincL2_);

  // get length of the structural, fluid and ale vector
  ni_ = (structure_field()->interface()->extract_fsi_cond_vector(*structrhs))->global_length();
  ns_ = (structure_field()->interface()->extract_other_vector(*structrhs))->global_length();
  nf_ = innerfluidfluidrhs->global_length();
  nfv_ = fluidvelextract->extract_cond_vector(*fluidinc)->global_length();
  nfp_ = fluidvelextract->extract_other_vector(*fluidinc)->global_length();
  na_ = alerhs->global_length();
  nall_ = rhs_->global_length();
}

/*----------------------------------------------------------------------
 * recover_lagrange_multiplier:
 *     - Compute the Lagrange multiplier at the FSI-interface
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::recover_lagrange_multiplier()
{
  /*----------------------------------------------------------------------*/
  // Time integration parameter
  /*----------------------------------------------------------------------*/
  // Fluid
  // b*y_n+(1-b)*y_n+1
  // b:flintparam
  /*----------------------------------------------------------------------*/
  const double ftimintparam = fluid_field()->tim_int_param();

  // Fluid time scaling parameter
  // fluidtimescale: \tau
  // \tau = 1/\Delta t  for Backward-Euler;
  // \tau = 2/\Delta t  for Trapezoidal Rule
  const double fluidtimescale = fluid_field()->time_scaling();

  // Scaling factor for different fluid/structural units
  const double fluidresidualscale = fluid_field()->residual_scaling();

  /*----------------------------------------------------------------------
    Lagrange Multiplier Setup
  ------------------------------------------------------------------------
   The Lagrange Multiplier is updated as follows:

   lambda_^{n+1}=

       1/(1-flintparam)*(

   (1)  -flintparam*lambda_^n

   (2)  -r_{\Gamma}^{F,n+1}

   (3)  - 1/Tau *(F_{\Gamma\Gamma)})* \Delta d_{\Gamma}{S,n+1}

   (4)  -F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma}{S,n+1}

   (5)  -F_{\Gamma I} * \Delta u_{I}^{F,n+1}

   (6)  -F^{G}_{\Gamma I} * \Delta d_{I,n+1}^{G,n+1}

      Only at first Newton iteration:
   (7)  +dt / Tau * F_{\Gamma\Gamma} * u_{\Gamma}^n}
          )
    + tau: time scaling factor for interface time integration (tau = 1/fluid_field()->TimeScaling())
    + the terms 2 to 7 are first saved in a tmpvec which will be added to lambda
   ----------------------------------------------------------------------*/

  // creating & initializing the storage vectors for the last four terms
  std::shared_ptr<Core::LinAlg::Vector<double>> fggddg = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> fmggddg = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> fgidui = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> fmgiddia = nullptr;

  // stores intermediate result of terms (3)-(7)
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec = nullptr;

  // ---------Addressing term (2)
  // store f^F_{\Gamma}! As Recover-LM is called after the Newton loop, the RHS will have changed!
  std::shared_ptr<Core::LinAlg::Vector<double>> fluidresidual =
      fluid_field()->interface()->extract_fsi_cond_vector(*fluid_field()->rhs());

  // ---------Addressing term (1)
  lambda_->update(ftimintparam, *lambda_, 0.0);

  // ---------Addressing term (2)
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(*fluidresidual);
  tmpvec->scale(-1.0);


  // ---------Addressing term (3)
  if (fggprev_ != nullptr)
  {
    fggddg = std::make_shared<Core::LinAlg::Vector<double>>(fggprev_->range_map(), true);
    fggprev_->Apply(*struct_to_fluid(ddginc_), *fggddg);
    tmpvec->update(fluidtimescale, *fggddg, 1.0);
  }

  //(4)
  if (fmggprev_ != nullptr)
  {
    Core::LinAlg::Vector<double> fmggddg(fmggprev_->range_map(), true);
    fmggprev_->Apply(*struct_to_fluid(ddginc_), fmggddg);
    tmpvec->update(1.0, fmggddg, 1.0);
  }

  //(5)
  if (fgiprev_ != nullptr)
  {
    Core::LinAlg::Vector<double> fgidui(fgiprev_->range_map(), true);
    fgiprev_->Apply(*duiinc_, fgidui);
    tmpvec->update(1.0, fgidui, 1.0);
  }

  //(6)
  if (fmgiprev_ != nullptr)
  {
    // The domain map of matrix fmgiprev_ contains inner velocity and pressure DOFs!
    // AleToFluid converts the inner ALE displacement increments to inner Fluid velocity DOFs.
    // The underlying map of this vector has to match the domain map!
    // Hence, the missing pressure DOFs have to be appended.
    Core::LinAlg::Vector<double> fmgiddia(fmgiprev_->range_map(), true);

    std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidpresmaps;
    // Merged fluid pressure DOF map
    fluidpresmaps.push_back(fluid_field()->pressure_row_map());
    // Embedded fluid DOF map
    fluidpresmaps.push_back(fluid_field()->x_fluid_fluid_map_extractor()->fluid_map());

    // Embedded fluid pressure map
    std::shared_ptr<const Core::LinAlg::Map> innerfluidpresmap =
        Core::LinAlg::MultiMapExtractor::intersect_maps(fluidpresmaps);

    // To keep merged fluid velocity and pressure DOF apart
    Core::LinAlg::MapExtractor innerfluidvelextractor(
        *fluid_field()->x_fluid_fluid_map_extractor()->fluid_map(), innerfluidpresmap, false);

    // Get the ALE-displacements, convert to inner fluid DOF. Still mapped to the embedded fluid.
    std::shared_ptr<Core::LinAlg::Vector<double>> aux =
        ale_to_fluid(ale_field()->interface()->insert_other_vector(*ddialeinc_));
    // Add the pressure DOF as zeros
    aux = innerfluidvelextractor.insert_cond_vector(*aux);
    aux = fluid_field()->x_fluid_fluid_map_extractor()->insert_fluid_vector(*aux);
    // Remove FSI DOF
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        fluid_field()->interface()->extract_other_vector(*aux);
    fmgiprev_->Apply(*tmp, fmgiddia);
    tmpvec->update(1.0, fmgiddia, 1.0);
  }

  //(7)
  if (firstcall_)
  {
    if (fggprev_ != nullptr)
    {
      Core::LinAlg::Vector<double> tmp(fggprev_->range_map(), true);
      std::shared_ptr<Core::LinAlg::Vector<double>> fveln = fluid_field()->extract_interface_veln();
      fggprev_->Apply(*fveln, tmp);
      tmpvec->update(dt() * fluidtimescale, tmp, 1.0);
    }
  }

  // ---------Adding tmpvec to lambda_
  lambda_->update(
      fluidresidualscale, *tmpvec, 1.0);  // scale with residual_scaling() to get [N/m^2]

  // Scaling everything with -1/(1-flintparam_)
  lambda_->scale(-1.0 / (1.0 - ftimintparam));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::handle_fluid_dof_map_change_in_newton()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    Core::IO::cout << "New Map!" << Core::IO::endl;

  //  Save old sum of increments
  std::shared_ptr<Core::LinAlg::Vector<double>> x_sum_n =
      Core::LinAlg::create_vector(*dof_row_map(), true);
  *x_sum_n = *x_sum_;
  //  Extract structural increment sum
  std::shared_ptr<const Core::LinAlg::Vector<double>> sx_n;
  sx_n = extractor().extract_vector(*x_sum_n, 0);
  //  Extract ALE increment sum
  std::shared_ptr<const Core::LinAlg::Vector<double>> ax_n;
  ax_n = extractor().extract_vector(*x_sum_n, 2);

  create_combined_dof_row_map();

  // Initialize the global system matrix!
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 81, false, true);

  iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  rhs_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  x_sum_ = Core::LinAlg::create_vector(*dof_row_map(), true);

  //  Set the new increment sum x_sum_ together

  extractor().insert_vector(*sx_n, 0, *x_sum_);

  std::shared_ptr<Core::LinAlg::Vector<double>> ff_stepinc =
      fluid_field()->interface()->extract_other_vector(*fluid_field()->stepinc());
  extractor().insert_vector(*ff_stepinc, 1, *x_sum_);

  extractor().insert_vector(*ax_n, 2, *x_sum_);

  //  The fluid length may have changed
  nf_ = fluid_field()->rhs()->global_length();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::FluidFluidMonolithicFluidSplitNoNOX::has_fluid_dof_map_changed(
    const Core::LinAlg::Map& fluidincrementmap)
{
  bool isoldmap = fluidincrementmap.same_as(*fluid_field()->interface()->other_map());
  return !isoldmap;
}

FOUR_C_NAMESPACE_CLOSE
