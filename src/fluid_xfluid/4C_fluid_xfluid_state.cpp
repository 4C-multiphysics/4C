// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_xfluid_state.hpp"

#include "4C_cut_cutwizard.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_discretization.hpp"
#include "4C_xfem_dofset.hpp"
#include "4C_xfem_xfield_state_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor  Initialize coupling matrices                     schott 01/15 |
 *----------------------------------------------------------------------*/
FLD::XFluidState::CouplingState::CouplingState(
    const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofrowmap,
    const std::shared_ptr<Core::FE::Discretization>& slavediscret_mat,
    const std::shared_ptr<Core::FE::Discretization>& slavediscret_rhs)
    : is_active_(true)
{
  if (slavediscret_mat == nullptr)
    FOUR_C_THROW("invalid slave discretization for coupling application");
  if (slavediscret_rhs == nullptr)
    FOUR_C_THROW("invalid slave discretization for coupling application");


  // savegraph flag set to true, as there is no change in the matrix graph expected for the lifetime
  // of this state container
  // no explicit Dirichlet, otherwise new matrices will be created in ApplyDirichlet
  // NOTE: setting explicit Dirichlet to false can cause problems with ML preconditioner (see remark
  // in Core::LinAlg::Sparsematrix) however, we prefer not to build new matrices in ApplyDirichlet
  C_xs_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *xfluiddofrowmap, 300, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  C_sx_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *slavediscret_mat->dof_row_map(), 300, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  C_ss_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *slavediscret_mat->dof_row_map(), 300, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);

  rhC_s_ = Core::LinAlg::create_vector(*slavediscret_rhs->dof_row_map(), true);
  rhC_s_col_ = Core::LinAlg::create_vector(*slavediscret_rhs->dof_col_map(), true);
}

/*----------------------------------------------------------------------*
 |  zero coupling matrices and rhs vectors                 schott 01/15 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::CouplingState::zero_coupling_matrices_and_rhs()
{
  if (!is_active_) return;

  // zero all coupling matrices and rhs vectors
  XFEM::zero_matrix(*C_xs_);
  XFEM::zero_matrix(*C_sx_);
  XFEM::zero_matrix(*C_ss_);

  rhC_s_->put_scalar(0.0);
  rhC_s_col_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*
 |  complete coupling matrices and rhs vectors             schott 01/15 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::CouplingState::complete_coupling_matrices_and_rhs(
    const Core::LinAlg::Map& xfluiddofrowmap, const Core::LinAlg::Map& slavedofrowmap)
{
  if (!is_active_) return;

  // REMARK: for matrices Complete() calls the GlobalAssemble() routine to gather
  // entries from all processors (domain-map are the columns, range-map are the rows)
  C_xs_->complete(slavedofrowmap, xfluiddofrowmap);
  C_sx_->complete(xfluiddofrowmap, slavedofrowmap);
  C_ss_->complete(slavedofrowmap, slavedofrowmap);

  //  std::cout << "number of nonzeros: C_xs" << C_xs_->EpetraMatrix()->MaxNumEntries() <<
  //  std::endl; std::cout << "number of nonzeros: C_sx" << C_sx_->EpetraMatrix()->MaxNumEntries()
  //  << std::endl; std::cout << "number of nonzeros: C_ss" <<
  //  C_ss_->EpetraMatrix()->MaxNumEntries() << std::endl;
  //-------------------------------------------------------------------------------
  // export the rhs coupling vector to a row vector
  Core::LinAlg::Vector<double> rhC_s_tmp(rhC_s_->get_map(), true);
  Core::LinAlg::Export exporter_rhC_s_col(rhC_s_col_->get_map(), rhC_s_tmp.get_map());
  rhC_s_tmp.export_to(*rhC_s_col_, exporter_rhC_s_col, Add);

  rhC_s_->update(1.0, rhC_s_tmp, 0.0);
}


/*----------------------------------------------------------------------*
 |  destroy the coupling objects and it's content          schott 01/15 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::CouplingState::destroy(bool throw_exception)
{
  if (!is_active_) return;

  XFEM::destroy_matrix(C_xs_, throw_exception);
  XFEM::destroy_matrix(C_sx_, throw_exception);
  XFEM::destroy_matrix(C_ss_, throw_exception);

  XFEM::destroy_rcp_object(rhC_s_, throw_exception);
  XFEM::destroy_rcp_object(rhC_s_col_, throw_exception);

  is_active_ = false;
}


/*----------------------------------------------------------------------*
 |  Constructor for XFluidState                             kruse 08/14 |
 *----------------------------------------------------------------------*/
FLD::XFluidState::XFluidState(const std::shared_ptr<XFEM::ConditionManager>& condition_manager,
    const std::shared_ptr<Cut::CutWizard>& wizard, const std::shared_ptr<XFEM::XFEMDofSet>& dofset,
    const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofrowmap,
    const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofcolmap)
    : xfluiddofrowmap_(xfluiddofrowmap),
      xfluiddofcolmap_(xfluiddofcolmap),
      dofset_(dofset),
      wizard_(wizard),
      condition_manager_(condition_manager)
{
  init_system_matrix();
  init_state_vectors();

  init_coupling_matrices_and_rhs();
}

/*----------------------------------------------------------------------*
 |  Initialize (coupling) matrices & rhs-vectors
 |                                                         kruse 08/14
 *----------------------------------------------------------------------*/
void FLD::XFluidState::init_system_matrix()
{
  // create a matrix that does communication for non-local rows and columns
  // * this enables to do the evaluate loop over just row elements instead of col elements
  // * time consuming assemble for cut elements is done only once on a unique row processor
  // REMARK: call the SparseMatrix: * explicitdirichlet = false (is used in ApplyDirichlet, true
  // would create a new matrix when DBS will be applied)
  //                                    setting flag to false can cause problems with ML
  //                                    preconditioner (see remark in Core::LinAlg::Sparsematrix)
  //                                    however, we prefer not to build new matrices in
  //                                    ApplyDirichlet
  //                                * savegraph = true/false: To save the graph (pattern for
  //                                non-zero entries) leads to a speedup in the assembly of the
  //                                matrix
  //                                    for subsequent assemblies.
  //                                    However, do not use the savegraph-option when the matrix
  //                                    graph can change during the usage of this object of
  //                                    SparseMatrix or use the reset()-function instead of Zero().
  //                                    For XFEM-problems, the matrix graph changes between
  //                                    timesteps, however, then a new state class and sparsematrix
  //                                    is created, otherwise a reset()-function has to be called
  //                                    instead of the Zero()-function. We are using the save-graph
  //                                    option.
  // * the estimate of the number of nonzero entries is adapted to hex8 elements with 8 adjacent
  // elements around a node
  //   + edge-based couplings component-wise v_x->u_x, v_y->u_y, v_z->u_z, q->p
  //   number of non-zeros (for hex8 elements): 108+54 = 162
  sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *xfluiddofrowmap_, 162, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
}


/*----------------------------------------------------------------------*
 |  Initialize state vectors                                kruse 08/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::init_state_vectors()
{
  // Vectors passed to the element
  // -----------------------------
  // velocity/pressure at time n+1, n and n-1
  velnp_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  veln_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  velnm_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // velocity/pressure at time n+alpha_F
  velaf_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);


  // acceleration/(scalar time derivative) at time n+1 and n
  accnp_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  accn_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // acceleration/(scalar time derivative) at time n+alpha_M/(n+alpha_M/n)
  accam_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // scalar at time n+alpha_F/n+1 and n+alpha_M/n
  // (only required for low-Mach-number case)
  // ... this is a dummy to avoid errors
  scaaf_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  scaam_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // history vector
  hist_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // the vector containing body and surface forces
  neumann_loads_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // rhs: standard (stabilized) residual vector (rhs for the incremental form)
  residual_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  residual_col_ = Core::LinAlg::create_vector(*xfluiddofcolmap_, true);
  trueresidual_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // nonlinear iteration increment vector
  incvel_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
}

/*----------------------------------------------------------------------*
 |  Initialize coupling matrices                           schott 01/15 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::init_coupling_matrices_and_rhs()
{
  // loop all coupling objects
  for (int coup_idx = 0; coup_idx < condition_manager_->num_coupling(); coup_idx++)
  {
    std::shared_ptr<XFEM::CouplingBase> coupling =
        condition_manager_->get_coupling_by_idx(coup_idx);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (coupling == nullptr) FOUR_C_THROW("invalid coupling object!");
#endif

    std::shared_ptr<XFluidState::CouplingState> coup_state = nullptr;

    if (!condition_manager_->is_coupling_condition(
            coupling->get_name()))  // coupling or one-sided non-coupling object
    {
      // create coupling state object with coupling matrices initialized with nullptr
      coup_state = std::make_shared<XFluidState::CouplingState>();
    }
    else
    {
      if (condition_manager_->is_level_set_condition(coup_idx))
      {
        // coupling matrices can be assembled into the fluid sysmat
        // coupling rhs terms can be assembled into the fluid residual

        coup_state = std::make_shared<XFluidState::CouplingState>(
            sysmat_, sysmat_, sysmat_, residual_, residual_col_);
      }
      else if (condition_manager_->is_mesh_condition(coup_idx))
      {
        // for matrix use the full condition dis to enable assign in blockmatrix (row map of matrix
        // is not changed by complete!) for rhs we need the additional ghosting of the boundary zone
        // on slave side, therefore the specifically modified coupling dis
        coup_state = std::make_shared<XFluidState::CouplingState>(
            xfluiddofrowmap_, coupling->get_cond_dis(), coupling->get_coupling_dis());
      }
      else
        FOUR_C_THROW(
            "coupling object is neither a level-set coupling object nor a mesh-coupling object");
    }

    coup_state_[coup_idx] = coup_state;

  }  // loop coupling objects
}

/*----------------------------------------------------------------------*
 |  Initialize ALE state vectors                           schott 12/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::init_ale_state_vectors(XFEM::DiscretizationXFEM& xdiscret,
    const Core::LinAlg::Vector<double>& dispnp_initmap,
    const Core::LinAlg::Vector<double>& gridvnp_initmap)
{
  //! @name Ale Displacement at time n+1
  dispnp_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  xdiscret.export_initialto_active_vector(dispnp_initmap, *dispnp_);

  //! @name Grid Velocity at time n+1
  gridvnp_ = Core::LinAlg::create_vector(*xfluiddofrowmap_, true);
  xdiscret.export_initialto_active_vector(gridvnp_initmap, *gridvnp_);
}


/*----------------------------------------------------------------------*
 |  Initialize state vectors                               schott 12/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::zero_coupling_matrices_and_rhs()
{
  // loop all coupling objects
  for (std::map<int, std::shared_ptr<CouplingState>>::iterator cs = coup_state_.begin();
      cs != coup_state_.end(); cs++)
    cs->second->zero_coupling_matrices_and_rhs();
}

/*----------------------------------------------------------------------*
 |  Complete coupling matrices and rhs vectors             schott 12/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::complete_coupling_matrices_and_rhs()
{
  complete_coupling_matrices_and_rhs(*xfluiddofrowmap_);
}


/*----------------------------------------------------------------------*
 |  Complete coupling matrices and rhs vectors             schott 12/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::complete_coupling_matrices_and_rhs(
    const Core::LinAlg::Map& fluiddofrowmap  ///< fluid dof row map used for complete
)
{
  // loop all coupling objects
  for (std::map<int, std::shared_ptr<CouplingState>>::iterator cs = coup_state_.begin();
      cs != coup_state_.end(); cs++)
  {
    int coupl_idx = cs->first;

    // complete only the mesh coupling objects, levelset-couplings are completed by the
    // sysmat->Complete call
    if (condition_manager_->is_mesh_condition(coupl_idx))
    {
      std::shared_ptr<XFEM::CouplingBase> coupling =
          condition_manager_->get_coupling_by_idx(coupl_idx);
      cs->second->complete_coupling_matrices_and_rhs(
          fluiddofrowmap, *coupling->get_cond_dis()->dof_row_map());  // complete w.r.t
    }
  }
}


/*----------------------------------------------------------------------*
 |    /// zero system matrix and related rhs vectors       schott 08/15 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::zero_system_matrix_and_rhs()
{
  XFEM::zero_matrix(*sysmat_);

  // zero residual vectors
  residual_col_->put_scalar(0.0);
  residual_->put_scalar(0.0);
  trueresidual_->put_scalar(0.0);
}


/*----------------------------------------------------------------------*
 |  Set dirichlet- and velocity/pressure-map extractor      kruse 08/14 |
 *----------------------------------------------------------------------*/
void FLD::XFluidState::setup_map_extractors(
    const std::shared_ptr<Core::FE::Discretization>& xfluiddiscret, const double& time)
{
  // create dirichlet map extractor
  Teuchos::ParameterList eleparams;
  // other parameters needed by the elements
  eleparams.set("total time", time);
  eleparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());
  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>();
  xfluiddiscret->evaluate_dirichlet(eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_);

  zeros_->put_scalar(0.0);

  // create vel-pres splitter
  const int numdim = Global::Problem::instance()->n_dim();
  velpressplitter_ = std::make_shared<Core::LinAlg::MapExtractor>();
  Core::LinAlg::create_map_extractor_from_discretization(
      *xfluiddiscret, numdim, 1, *velpressplitter_);
}


bool FLD::XFluidState::destroy()
{
  // destroy system matrix (destroy after coupling matrices, as for twophase problems the coupling
  // matrices are identical to the system matrix)
  XFEM::destroy_matrix(sysmat_);

  // destroy all coupling system matrices and rhs vectors (except for levelset coupling objects
  for (std::map<int, std::shared_ptr<CouplingState>>::iterator i = coup_state_.begin();
      i != coup_state_.end(); ++i)
  {
    std::shared_ptr<CouplingState>& cs = i->second;  // RCP reference!
    if (cs != nullptr)
    {
      cs->destroy(true);  // throw exception when object could not be deleted or more than one
                          // strong RCP points to it
      cs = nullptr;       // invalidate coupling state object
    }
  }

  // destroy dofrowmap and dofcolmap
  XFEM::destroy_rcp_object(xfluiddofrowmap_);
  XFEM::destroy_rcp_object(xfluiddofcolmap_);

  // destroy state vectors
  XFEM::destroy_rcp_object(velnp_);
  XFEM::destroy_rcp_object(veln_);
  XFEM::destroy_rcp_object(velnm_);
  XFEM::destroy_rcp_object(velaf_);

  XFEM::destroy_rcp_object(accnp_);
  XFEM::destroy_rcp_object(accn_);
  XFEM::destroy_rcp_object(accam_);

  XFEM::destroy_rcp_object(scaaf_);
  XFEM::destroy_rcp_object(scaam_);

  XFEM::destroy_rcp_object(hist_);
  XFEM::destroy_rcp_object(neumann_loads_);

  XFEM::destroy_rcp_object(residual_);
  XFEM::destroy_rcp_object(trueresidual_);

  XFEM::destroy_rcp_object(zeros_);
  XFEM::destroy_rcp_object(incvel_);

  XFEM::destroy_rcp_object(dispnp_);
  XFEM::destroy_rcp_object(gridvnp_);


  // destroy velpressplitter_
  XFEM::destroy_rcp_object(velpressplitter_);

  // wizard, dofset and conditionmanager keep RCPs pointing to them and cannot be destroyed as they
  // are further used in xfluid-class decrease at least the strong reference counter
  wizard_ = nullptr;
  dofset_ = nullptr;
  condition_manager_ = nullptr;

  return true;
}

void FLD::XFluidState::update_boundary_cell_coords()
{
  // loop all mesh coupling objects
  for (int mc_idx = 0; mc_idx < condition_manager_->num_mesh_coupling(); mc_idx++)
  {
    std::shared_ptr<XFEM::MeshCoupling> mc_coupl = condition_manager_->get_mesh_coupling(mc_idx);

    if (!mc_coupl->cut_geometry()) continue;  // If don't cut the background mesh.

    wizard_->update_boundary_cell_coords(mc_coupl->get_cutter_dis(),
        mc_coupl->get_cutter_disp_col(), condition_manager_->get_mesh_coupling_start_gid(mc_idx));
  }
}

FOUR_C_NAMESPACE_CLOSE
