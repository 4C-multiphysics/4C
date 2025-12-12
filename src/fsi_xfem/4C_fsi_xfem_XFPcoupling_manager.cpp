// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_xfem_XFPcoupling_manager.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN

XFEM::XfpCouplingManager::XfpCouplingManager(std::shared_ptr<XFEM::ConditionManager> condmanager,
    std::shared_ptr<PoroElast::PoroBase> poro, std::shared_ptr<FLD::XFluid> xfluid,
    std::vector<int> idx)
    : CouplingCommManager(poro->structure_field()->discretization(),
          poro->fluid_field()->discretization(), "XFEMSurfFPIMono", 0, 3),
      poro_(poro),
      xfluid_(xfluid),
      cond_name_ps_ps_("XFEMSurfFPIMono_ps_ps"),
      cond_name_ps_pf_("XFEMSurfFPIMono_ps_pf"),
      cond_name_pf_ps_("XFEMSurfFPIMono_pf_ps"),
      cond_name_pf_pf_("XFEMSurfFPIMono_pf_pf"),
      idx_(idx)
{
  // Coupling_Comm_Manager create all Coupling Objects now with Structure has idx = 0, Fluid has idx
  // = 1!

  mcfpi_ps_ps_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
      condmanager->get_mesh_coupling(cond_name_ps_ps_));
  if (mcfpi_ps_ps_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFPI for Porostructure!");
  mcfpi_ps_ps_->initialize_struct_pres_map(*poro_->fluid_structure_coupling().slave_dof_map(),
      *poro_->fluid_structure_coupling().perm_master_dof_map());

  mcfpi_ps_pf_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
      condmanager->get_mesh_coupling(cond_name_ps_pf_));
  if (mcfpi_ps_pf_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFPI for Porofluid!");
  mcfpi_ps_pf_->initialize_struct_pres_map(*poro_->fluid_structure_coupling().slave_dof_map(),
      *poro_->fluid_structure_coupling().perm_master_dof_map());

  mcfpi_pf_ps_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
      condmanager->get_mesh_coupling(cond_name_pf_ps_));
  if (mcfpi_pf_ps_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFPI for Porofluid!");
  mcfpi_pf_ps_->initialize_struct_pres_map(*poro_->fluid_structure_coupling().slave_dof_map(),
      *poro_->fluid_structure_coupling().perm_master_dof_map());

  mcfpi_pf_pf_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
      condmanager->get_mesh_coupling(cond_name_pf_pf_));
  if (mcfpi_pf_pf_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFPI for Porofluid!");
  mcfpi_pf_pf_->initialize_struct_pres_map(*poro_->fluid_structure_coupling().slave_dof_map(),
      *poro_->fluid_structure_coupling().perm_master_dof_map());

  // safety check
  if (!mcfpi_ps_ps_->i_dispnp()->get_map().same_as(*get_map_extractor(0)->map(1)))
    FOUR_C_THROW("XFPCoupling_Manager: Maps of Condition and Mesh Coupling do not fit (psps)!");
  if (!mcfpi_ps_pf_->i_dispnp()->get_map().same_as(*get_map_extractor(0)->map(1)))
    FOUR_C_THROW("XFPCoupling_Manager: Maps of Condition and Mesh Coupling do not fit (pspf)!");
  if (!mcfpi_pf_ps_->i_dispnp()->get_map().same_as(*get_map_extractor(0)->map(1)))
    FOUR_C_THROW("XFPCoupling_Manager: Maps of Condition and Mesh Coupling do not fit (pfps)!");
  if (!mcfpi_pf_pf_->i_dispnp()->get_map().same_as(*get_map_extractor(0)->map(1)))
    FOUR_C_THROW("XFPCoupling_Manager: Maps of Condition and Mesh Coupling do not fit (pfpf)!");

  // storage of the resulting Robin-type structural forces from the old timestep
  // Recovering of Lagrange multiplier happens on fluid field
  lambda_ps_ = std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->map(1), true);
  lambda_pf_ = std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->map(1), true);
}

void XFEM::XfpCouplingManager::init_coupling_states()
{
  mcfpi_ps_ps_->reconnect_parent_pointers();
  mcfpi_pf_ps_->reconnect_parent_pointers();
  mcfpi_ps_pf_->reconnect_parent_pointers();
  mcfpi_pf_pf_->reconnect_parent_pointers();
}

void XFEM::XfpCouplingManager::set_coupling_states()
{
  // 1 Set Displacement on both mesh couplings ... we get them from the structure field!
  insert_vector(0, poro_->structure_field()->dispnp(), 0, mcfpi_ps_ps_->i_dispnp(),
      CouplingCommManager::full_to_partial);
  insert_vector(0, poro_->structure_field()->dispnp(), 0, mcfpi_ps_pf_->i_dispnp(),
      CouplingCommManager::full_to_partial);
  insert_vector(0, poro_->structure_field()->dispnp(), 0, mcfpi_pf_ps_->i_dispnp(),
      CouplingCommManager::full_to_partial);
  insert_vector(0, poro_->structure_field()->dispnp(), 0, mcfpi_pf_pf_->i_dispnp(),
      CouplingCommManager::full_to_partial);

  // As interfaces embedded into the background mesh are fully ghosted, we don't know which
  std::shared_ptr<Core::LinAlg::Map> sfulldofmap =
      Core::LinAlg::allreduce_e_map(*poro_->structure_field()->discretization()->dof_row_map());
  std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*sfulldofmap, true);
  Core::LinAlg::export_to(*poro_->structure_field()->dispnp(), *dispnp_col);
  std::shared_ptr<Core::LinAlg::Map> ffulldofmap =
      Core::LinAlg::allreduce_e_map(*poro_->fluid_field()->discretization()->dof_row_map());
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*ffulldofmap, true);
  Core::LinAlg::export_to(*poro_->fluid_field()->velnp(), *velnp_col);

  mcfpi_ps_ps_->set_full_state(dispnp_col, velnp_col);
  mcfpi_ps_pf_->set_full_state(dispnp_col, velnp_col);
  mcfpi_pf_ps_->set_full_state(dispnp_col, velnp_col);
  mcfpi_pf_pf_->set_full_state(dispnp_col, velnp_col);


  // 2 Set Structural Velocity onto ps mesh coupling
  insert_vector(0, poro_->structure_field()->velnp(), 0, mcfpi_ps_ps_->i_velnp(),
      CouplingCommManager::full_to_partial);
  insert_vector(0, poro_->structure_field()->velnp(), 0, mcfpi_pf_ps_->i_velnp(),
      CouplingCommManager::full_to_partial);
  //  poro_->structure_field()->Velnp()->print(std::cout);

  //  insert_vector(1,poro_->fluid_field()->GridVel(),0,mcfpi_ps_ps_->IVelnp(),Coupling_Comm_Manager::full_to_partial);
  //  insert_vector(1,poro_->fluid_field()->GridVel(),0,mcfpi_pf_ps_->IVelnp(),Coupling_Comm_Manager::full_to_partial);
  // 3 Set Fluid Velocity onto pf mesh coupling
  insert_vector(1, poro_->fluid_field()->velnp(), 0, mcfpi_ps_pf_->i_velnp(),
      CouplingCommManager::full_to_partial);
  insert_vector(1, poro_->fluid_field()->velnp(), 0, mcfpi_pf_pf_->i_velnp(),
      CouplingCommManager::full_to_partial);
}

void XFEM::XfpCouplingManager::add_coupling_matrix(
    Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling)
{
  const double scaling_disp_vel =
      1 / ((1 - poro_->structure_field()->tim_int_param()) * poro_->structure_field()->dt());
  const double dt = poro_->fluid_field()->dt();
  if (idx_.size() == 2)  // assume that the poro field is not split and we just have a blockmatrix
                         // P/F
  {
    Core::LinAlg::SparseMatrix& C_ss_block = (systemmatrix)(idx_[0], idx_[0]);
    Core::LinAlg::SparseMatrix& C_fs_block = (systemmatrix)(idx_[1], idx_[0]);
    Core::LinAlg::SparseMatrix& C_sf_block = (systemmatrix)(idx_[0], idx_[1]);

    // 1// Add Blocks f-ps(2), ps-f(3), ps-ps(4)
    Core::LinAlg::matrix_add(*xfluid_->c_ss_matrix(cond_name_ps_ps_), false,
        scaling * scaling_disp_vel, C_ss_block, 1.0);
    Core::LinAlg::matrix_add(
        *xfluid_->c_sx_matrix(cond_name_ps_ps_), false, scaling, C_sf_block, 1.0);
    Core::LinAlg::matrix_add(*xfluid_->c_xs_matrix(cond_name_ps_ps_), false,
        scaling * scaling_disp_vel, C_fs_block, 1.0);

    // 2// Add Blocks f-pf(5), ps-pf(6)
    Core::LinAlg::SparseMatrix C_ps_pf(
        xfluid_->c_ss_matrix(cond_name_ps_pf_)->row_map(), 81, false);
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_ps_pf_), 1, C_ps_pf,
        CouplingCommManager::col, 1, true, false);
    C_ps_pf.complete(*get_map_extractor(1)->map(1), *get_map_extractor(0)->map(1));
    Core::LinAlg::SparseMatrix C_f_pf(xfluid_->c_xs_matrix(cond_name_ps_pf_)->row_map(), 81, false);
    insert_matrix(-1, 0, *xfluid_->c_xs_matrix(cond_name_ps_pf_), 1, C_f_pf,
        CouplingCommManager::col, 1, true, false);
    C_f_pf.complete(*get_map_extractor(1)->map(1), C_fs_block.range_map());
    Core::LinAlg::matrix_add(C_f_pf, false, scaling, C_fs_block, 1.0);
    Core::LinAlg::matrix_add(C_ps_pf, false, scaling, C_ss_block, 1.0);

    // 3// Add Blocks pf-f(7), pf-ps(8)
    Core::LinAlg::SparseMatrix C_pf_ps(*get_map_extractor(1)->map(1), 81, false);
    Core::LinAlg::SparseMatrix C_pf_f(*get_map_extractor(1)->map(1), 81, false);
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_pf_ps_), 1, C_pf_ps,
        CouplingCommManager::row, 1, true, false);
    C_pf_ps.complete(*get_map_extractor(0)->map(1), *get_map_extractor(1)->map(1));
    insert_matrix(-1, 0, *xfluid_->c_sx_matrix(cond_name_pf_ps_), 1, C_pf_f,
        CouplingCommManager::row, 1, true, false);
    C_pf_f.complete(*xfluid_->dof_row_map(), *get_map_extractor(1)->map(1));
    Core::LinAlg::matrix_add(C_pf_ps, false, scaling * scaling_disp_vel * dt, C_ss_block, 1.0);
    Core::LinAlg::matrix_add(C_pf_f, false, scaling * dt, C_sf_block, 1.0);

    // 4// Add Block pf-pf(9)
    Core::LinAlg::SparseMatrix C_pf_pf(*get_map_extractor(1)->map(1), 81, false);
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_pf_pf_), 1, C_pf_pf,
        CouplingCommManager::row_and_col);
    C_pf_pf.complete(*get_map_extractor(1)->map(1), *get_map_extractor(1)->map(1));
    Core::LinAlg::matrix_add(C_pf_pf, false, scaling * dt, C_ss_block, 1.0);
  }
  else if (idx_.size() == 3)
  {
    Core::LinAlg::SparseMatrix& C_ss_block = (systemmatrix)(idx_[0], idx_[0]);
    Core::LinAlg::SparseMatrix& C_fs_block = (systemmatrix)(idx_[1], idx_[0]);
    Core::LinAlg::SparseMatrix& C_sf_block = (systemmatrix)(idx_[0], idx_[1]);

    Core::LinAlg::SparseMatrix& C_pfpfblock = (systemmatrix)(idx_[2], idx_[2]);
    Core::LinAlg::SparseMatrix& C_fpf_block = (systemmatrix)(idx_[1], idx_[2]);
    Core::LinAlg::SparseMatrix& C_pff_block = (systemmatrix)(idx_[2], idx_[1]);

    Core::LinAlg::SparseMatrix& C_pfs_block = (systemmatrix)(idx_[2], idx_[0]);
    Core::LinAlg::SparseMatrix& C_spf_block = (systemmatrix)(idx_[0], idx_[2]);

    // 1// Add Blocks f-ps(2), ps-f(3), ps-ps(4)
    Core::LinAlg::matrix_add(*xfluid_->c_ss_matrix(cond_name_ps_ps_), false,
        scaling * scaling_disp_vel, C_ss_block, 1.0);
    Core::LinAlg::matrix_add(
        *xfluid_->c_sx_matrix(cond_name_ps_ps_), false, scaling, C_sf_block, 1.0);
    Core::LinAlg::matrix_add(*xfluid_->c_xs_matrix(cond_name_ps_ps_), false,
        scaling * scaling_disp_vel, C_fs_block, 1.0);

    // 2// Add Blocks f-pf(5), ps-pf(6)
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_ps_pf_), 1, C_spf_block,
        CouplingCommManager::col, scaling, true, true);
    insert_matrix(-1, 0, *xfluid_->c_xs_matrix(cond_name_ps_pf_), 1, C_fpf_block,
        CouplingCommManager::col, scaling, true, true);

    // 3// Add Blocks pf-f(7), pf-ps(8)
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_pf_ps_), 1, C_pfs_block,
        CouplingCommManager::row, scaling * scaling_disp_vel * dt, true, true);
    insert_matrix(-1, 0, *xfluid_->c_sx_matrix(cond_name_pf_ps_), 1, C_pff_block,
        CouplingCommManager::row, scaling * dt, true, true);

    // 4// Add Block pf-pf(9)
    insert_matrix(-1, 0, *xfluid_->c_ss_matrix(cond_name_pf_pf_), 1, C_pfpfblock,
        CouplingCommManager::row_and_col, scaling * dt, true, true);
  }
  else
    FOUR_C_THROW(
        "XFPCoupling_Manager::AddCouplingMatrix: Not implemented for number of blocks = {}",
        idx_.size());
}

///*-----------------------------------------------------------------------------------------*
//| Add the coupling rhs                                                        ager 06/2016 |
//*-----------------------------------------------------------------------------------------*/
void XFEM::XfpCouplingManager::add_coupling_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const Core::LinAlg::MultiMapExtractor& me, double scaling)
{
  const double dt = poro_->fluid_field()->dt();
  if (idx_.size() == 2)  // assume that the poro field is not split and we just have a blockmatrix
                         // P/F
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_ps_ps =
        xfluid_->rhs_s_vec(cond_name_ps_ps_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_ps_pf =
        xfluid_->rhs_s_vec(cond_name_ps_pf_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_pf_ps =
        xfluid_->rhs_s_vec(cond_name_pf_ps_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_pf_pf =
        xfluid_->rhs_s_vec(cond_name_pf_pf_);

    std::shared_ptr<Core::LinAlg::Vector<double>> prhs =
        std::make_shared<Core::LinAlg::Vector<double>>(*me.map(idx_[0]), true);

    insert_vector(0, rhs_C_ps_ps, 0, prhs, CouplingCommManager::partial_to_global, true, scaling);
    insert_vector(0, rhs_C_ps_pf, 0, prhs, CouplingCommManager::partial_to_global, true, scaling);

    insert_vector(
        0, rhs_C_pf_ps, 1, prhs, CouplingCommManager::partial_to_global, true, scaling * dt);
    insert_vector(
        0, rhs_C_pf_pf, 1, prhs, CouplingCommManager::partial_to_global, true, scaling * dt);

    // Add lambda contribution
    if (lambda_ps_ != nullptr && lambda_pf_ != nullptr)
    {
      /*----------------------------------------------------------------------*/
      // get time integration parameters of structure and fluid time integrators
      // to enable consistent time integration among the fields
      /*----------------------------------------------------------------------*/

      /*----------------------------------------------------------------------*/
      // this is the interpolation weight for quantities from last time step
      // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for
      // displacements) TimeIntegration for poro needs to be consistent!
      const double stiparam =
          poro_->structure_field()->tim_int_param();  // (1-theta) for OST and alpha_f for Genalpha

      // scale factor for the structure system matrix w.r.t the new time step
      const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) = 1/weight^S_np

      insert_vector(0, lambda_ps_, 0, prhs, CouplingCommManager::partial_to_global, true,
          stiparam * scaling_S);
      insert_vector(0, lambda_pf_, 1, prhs, CouplingCommManager::partial_to_global, true,
          stiparam * scaling_S);
    }

    me.add_vector(*prhs, idx_[0], *rhs);
  }
  else if (idx_.size() == 3)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_ps_ps =
        xfluid_->rhs_s_vec(cond_name_ps_ps_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_ps_pf =
        xfluid_->rhs_s_vec(cond_name_ps_pf_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_pf_ps =
        xfluid_->rhs_s_vec(cond_name_pf_ps_);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_C_pf_pf =
        xfluid_->rhs_s_vec(cond_name_pf_pf_);

    std::shared_ptr<Core::LinAlg::Vector<double>> srhs =
        std::make_shared<Core::LinAlg::Vector<double>>(*me.map(idx_[0]), true);
    std::shared_ptr<Core::LinAlg::Vector<double>> pfrhs =
        std::make_shared<Core::LinAlg::Vector<double>>(*me.map(idx_[2]), true);

    insert_vector(0, rhs_C_ps_ps, 0, srhs, CouplingCommManager::partial_to_full, true, scaling);
    insert_vector(0, rhs_C_ps_pf, 0, srhs, CouplingCommManager::partial_to_full, true, scaling);

    insert_vector(
        0, rhs_C_pf_ps, 1, pfrhs, CouplingCommManager::partial_to_full, true, scaling * dt);
    insert_vector(
        0, rhs_C_pf_pf, 1, pfrhs, CouplingCommManager::partial_to_full, true, scaling * dt);

    // Add lambda contribution
    if (lambda_ps_ != nullptr && lambda_pf_ != nullptr)
    {
      /*----------------------------------------------------------------------*/
      // get time integration parameters of structure and fluid time integrators
      // to enable consistent time integration among the fields
      /*----------------------------------------------------------------------*/

      /*----------------------------------------------------------------------*/
      // this is the interpolation weight for quantities from last time step
      // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for
      // displacements) TimeIntegration for poro needs to be consistent!
      const double stiparam =
          poro_->structure_field()->tim_int_param();  // (1-theta) for OST and alpha_f for Genalpha

      // scale factor for the structure system matrix w.r.t the new time step
      const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) = 1/weight^S_np

      insert_vector(
          0, lambda_ps_, 0, srhs, CouplingCommManager::partial_to_full, true, stiparam * scaling_S);
      insert_vector(0, lambda_pf_, 1, pfrhs, CouplingCommManager::partial_to_full, true,
          stiparam * scaling_S);
    }

    me.add_vector(*srhs, idx_[0], *rhs);
    me.add_vector(*pfrhs, idx_[2], *rhs);
  }
  else
    FOUR_C_THROW("XFPCoupling_Manager::AddCouplingRHS: Not implemented for number of blocks = {}",
        idx_.size());
}

/*----------------------------------------------------------------------*/
/* Store the Coupling RHS of the Old Timestep in lambda     ager 06/2016 |
 *----------------------------------------------------------------------*/
void XFEM::XfpCouplingManager::update(double scaling)
{
  /*----------------------------------------------------------------------*/
  // we directly store the fluid-unscaled rhs_C_s residual contribution from the fluid solver which
  // corresponds to the actual acting forces

  // scaling for the structural residual is done when it is added to the global residual vector
  // get the coupling rhs from the xfluid, this vector is based on the boundary dis which is part of
  // the structure dis
  lambda_ps_->update(scaling, *xfluid_->rhs_s_vec(cond_name_ps_ps_), 0.0);
  lambda_ps_->update(scaling, *xfluid_->rhs_s_vec(cond_name_ps_pf_), 1.0);

  const double dt = poro_->fluid_field()->dt();
  lambda_pf_->update(scaling * dt, *xfluid_->rhs_s_vec(cond_name_pf_ps_), 0.0);
  lambda_pf_->update(scaling * dt, *xfluid_->rhs_s_vec(cond_name_pf_pf_), 1.0);
  return;
}

/*----------------------------------------------------------------------*/
/* Write Output                                             ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfpCouplingManager::output(Core::IO::DiscretizationWriter& writer)
{
  //--------------------------------
  // output for Lagrange multiplier field (ie forces onto the structure, Robin-type forces
  // consisting of fluid forces and the Nitsche penalty term contribution)
  //--------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->full_map(), true);
  insert_vector(0, lambda_ps_, 0, lambdafull, CouplingCommManager::partial_to_full);
  writer.write_vector("fpilambda_ps", lambdafull);

  lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->full_map(), true);
  insert_vector(0, lambda_pf_, 0, lambdafull, CouplingCommManager::partial_to_full);
  writer.write_vector("fpilambda_pf", lambdafull);
  return;
}
/*----------------------------------------------------------------------*/
/* Read Restart on the interface                            ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfpCouplingManager::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->full_map(), true);
  reader.read_vector(lambdafull, "fpilambda_ps");
  insert_vector(0, lambdafull, 0, lambda_ps_, CouplingCommManager::full_to_partial);

  lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->full_map(), true);
  reader.read_vector(lambdafull, "fpilambda_pf");
  insert_vector(0, lambdafull, 0, lambda_pf_, CouplingCommManager::full_to_partial);
  return;
}

/*-----------------------------------------------------------------------------------------*
| Get Timeface on the interface (for OST this is 1/(theta dt))                ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
double XFEM::XfpCouplingManager::get_interface_timefac()
{
  FOUR_C_THROW("Check if you really want this!");
  /*
   * Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
   *
   *             / = 2 / dt   if interface time integration is second order
   * with fac = |
   *             \ = 1 / dt   if interface time integration is first order
   */
  const double dt = xfluid_->dt();
  if (interface_second_order_)
    return 2. / dt;
  else
    return 1. / dt;
}

FOUR_C_NAMESPACE_CLOSE
