// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_xfem_XFScoupling_manager.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"  //Todo: remove me finally
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
| Constructor                                                                 ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
XFEM::XfsCouplingManager::XfsCouplingManager(std::shared_ptr<ConditionManager> condmanager,
    std::shared_ptr<Adapter::Structure> structure, std::shared_ptr<FLD::XFluid> xfluid,
    std::vector<int> idx)
    : CouplingCommManager(structure->discretization(), "XFEMSurfFSIMono", 0, 3),
      struct_(structure),
      xfluid_(xfluid),
      cond_name_("XFEMSurfFSIMono"),
      idx_(idx),
      interface_second_order_(false)
{
  if (idx_.size() != 2)
    FOUR_C_THROW("XFSCoupling_Manager required two block ( 2 != {})", idx_.size());

  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  interface_second_order_ = fsidyn.get<bool>("SECONDORDER");

  // Coupling_Comm_Manager create all Coupling Objects now with Structure has idx = 0, Fluid has idx
  // = 1!
  mcfsi_ =
      std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(condmanager->get_mesh_coupling(cond_name_));
  if (mcfsi_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFSI for Structure!");

  mcfsi_->set_time_fac(1. / get_interface_timefac());

  // safety check
  if (!mcfsi_->i_dispnp()->get_map().same_as(*get_map_extractor(0)->map(1)))
    FOUR_C_THROW("XFSCoupling_Manager: Maps of Condition and Mesh Coupling do not fit!");

  // storage of the resulting Robin-type structural forces from the old timestep
  // Recovering of Lagrange multiplier happens on fluid field
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *mcfsi_->get_coupling_dis()->dof_row_map(), true);
}

/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 04/2017 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::init_coupling_states()
{
  // 1 Set Displacement on both mesh couplings ... we get them from the structure field!
  insert_vector(0, struct_->dispn(), 0, mcfsi_->i_dispn(), CouplingCommManager::full_to_partial);
  insert_vector(0, struct_->dispn(), 0, mcfsi_->i_dispnp(), CouplingCommManager::full_to_partial);

  // 2 Set Displacement on both mesh couplings ... we get them from the structure field!
  insert_vector(0, struct_->veln(), 0, mcfsi_->i_veln(), CouplingCommManager::full_to_partial);
  insert_vector(0, struct_->veln(), 0, mcfsi_->i_velnp(), CouplingCommManager::full_to_partial);
}

/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::set_coupling_states()
{
  // 1 update last increment, before we set new idispnp
  mcfsi_->update_displacement_iteration_vectors();

  // 2 Set Displacement on both mesh couplings ... we get them from the structure field!
  insert_vector(0, struct_->dispnp(), 0, mcfsi_->i_dispnp(), CouplingCommManager::full_to_partial);

  // get interface velocity at t(n)
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp =
      std::make_shared<Core::LinAlg::Vector<double>>(mcfsi_->i_velnp()->get_map(), true);
  velnp->update(1.0, *mcfsi_->i_dispnp(), -1.0, *mcfsi_->i_dispn(), 0.0);

  // inverse of FSI (1st order, 2nd order) scaling
  const double scaling_FSI = get_interface_timefac();  // 1/(theta_FSI * dt) =  1/weight^FSI_np
  const double dt = xfluid_->dt();

  // v^{n+1} = -(1-theta)/theta * v^{n} - 1/(theta*dt)*(d^{n+1}-d^{n0})
  velnp->update(-(dt - 1 / scaling_FSI) * scaling_FSI, *mcfsi_->i_veln(), scaling_FSI);

  // 3 Set Structural Velocity onto ps mesh coupling
  insert_vector(0, velnp, 0, mcfsi_->i_velnp(), CouplingCommManager::partial_to_partial);

  // 4 Set Structural Velocity onto the structural discretization
  if (mcfsi_->get_averaging_strategy() != Inpar::XFEM::Xfluid_Sided)
  {
    // Set Dispnp (used to calc local coord of gausspoints)
    struct_->discretization()->set_state("dispnp", *struct_->dispnp());
    // Set Velnp (used for interface integration)
    Core::LinAlg::Vector<double> fullvelnp(struct_->velnp()->get_map(), true);
    fullvelnp.update(1.0, *struct_->dispnp(), -1.0, *struct_->dispn(), 0.0);
    fullvelnp.update(-(dt - 1 / scaling_FSI) * scaling_FSI, *struct_->veln(), scaling_FSI);
    struct_->discretization()->set_state("velaf", fullvelnp);
  }
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling matrixes to the global systemmatrix                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::add_coupling_matrix(
    Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling)
{
  /*----------------------------------------------------------------------*/
  // Coupling blocks C_sf, C_fs and C_ss
  /*----------------------------------------------------------------------*/
  Core::LinAlg::SparseMatrix& C_ss_block = (systemmatrix)(idx_[0], idx_[0]);
  /*----------------------------------------------------------------------*/
  // scaling factor for displacement <-> velocity conversion (FSI)
  // inverse of FSI (1st order, 2nd order) scaling
  const double scaling_FSI = get_interface_timefac();  // 1/(theta_FSI * dt) =  1/weight^FSI_np

  // * all the coupling matrices are scaled with the weighting of the fluid w.r.t new time step np
  //    -> Unscale the blocks with (1/(theta_f*dt) = 1/weight(t^f_np))
  // * additionally the C_*s blocks (C_ss and C_fs) have to include the conversion from structural
  // displacements to structural velocities
  //    -> Scale these blocks with (1/(theta_FSI*dt) = 1/weight(t^FSI_np))
  //
  // REMARK that Scale() scales the original coupling matrix in xfluid

  // C_ss_block scaled with 1/(theta_f*dt) * 1/(theta_FSI*dt) = 1/weight(t^f_np) *
  // 1/weight(t^FSI_np) add the coupling block C_ss on the already existing diagonal block
  Core::LinAlg::matrix_add(
      *xfluid_->c_ss_matrix(cond_name_), false, scaling * scaling_FSI, C_ss_block, 1.0);


  Core::ProblemType probtype = Global::Problem::instance()->get_problem_type();

  // Todo: Need to either split fluid matrixes in the fsi algo or change the maps of the coupling
  // matrixes(merged)
  bool is_xfluidfluid = std::dynamic_pointer_cast<FLD::XFluidFluid>(xfluid_) != nullptr;

  if (probtype == Core::ProblemType::fsi_xfem &&
      !is_xfluidfluid)  // use assign for off diagonal blocks
  {
    // scale the off diagonal coupling blocks
    xfluid_->c_sx_matrix(cond_name_)
        ->scale(scaling);  //<   1/(theta_f*dt)                    = 1/weight(t^f_np)
    xfluid_->c_xs_matrix(cond_name_)
        ->scale(scaling * scaling_FSI);  //<   1/(theta_f*dt) * 1/(theta_FSI*dt) = 1/weight(t^f_np)
                                         //* 1/weight(t^FSI_np)

    systemmatrix.assign(
        idx_[0], idx_[1], Core::LinAlg::DataAccess::Share, *xfluid_->c_sx_matrix(cond_name_));
    systemmatrix.assign(
        idx_[1], idx_[0], Core::LinAlg::DataAccess::Share, *xfluid_->c_xs_matrix(cond_name_));
  }
  else if (probtype == Core::ProblemType::fpsi_xfem || is_xfluidfluid)
  {
    Core::LinAlg::SparseMatrix& C_fs_block = (systemmatrix)(idx_[1], idx_[0]);
    Core::LinAlg::SparseMatrix& C_sf_block = (systemmatrix)(idx_[0], idx_[1]);

    Core::LinAlg::matrix_add(*xfluid_->c_sx_matrix(cond_name_), false, scaling, C_sf_block, 1.0);
    Core::LinAlg::matrix_add(
        *xfluid_->c_xs_matrix(cond_name_), false, scaling * scaling_FSI, C_fs_block, 1.0);
  }
  else
  {
    FOUR_C_THROW("XFSCoupling_Manager: Want to use me for other problemtype --> check and add me!");
  }
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling rhs                                                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::add_coupling_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const Core::LinAlg::MultiMapExtractor& me, double scaling)
{
  Core::LinAlg::Vector<double> coup_rhs_sum(*xfluid_->rhs_s_vec(
      cond_name_));  // REMARK: Copy this vector to store the correct lambda_ in update!
  /// Lagrange multiplier \lambda_\Gamma^n at the interface (ie forces onto the structure,
  /// Robin-type forces consisting of fluid forces and the Nitsche penalty term contribution)
  if (lambda_ != nullptr)
  {
    /*----------------------------------------------------------------------*/
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    /*----------------------------------------------------------------------*/

    /*----------------------------------------------------------------------*/
    // this is the interpolation weight for quantities from last time step
    // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for
    // displacements)
    const double stiparam = struct_->tim_int_param();  // (1-theta) for OST and alpha_f for Genalpha

    // scale factor for the structure system matrix w.r.t the new time step
    const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) = 1/weight^S_np
    // add Lagrange multiplier (structural forces from t^n)
    coup_rhs_sum.update(stiparam * scaling_S, *lambda_, scaling);
  }
  else
  {
    coup_rhs_sum.scale(scaling);
  }

  Core::LinAlg::Vector<double> coup_rhs(*me.map(idx_[0]), true);
  Core::LinAlg::export_to(coup_rhs_sum, coup_rhs);  // use this command as long as poro is not split
                                                    // into two bocks in the monolithic algorithm!
  // insert_vector(0,coup_rhs_sum,0,coup_rhs,Coupling_Comm_Manager::partial_to_full);
  me.add_vector(coup_rhs, idx_[0], *rhs);
}

/*----------------------------------------------------------------------*/
/* Store the Coupling RHS of the Old Timestep in lambda     ager 06/2016 |
 *----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::update(double scaling)
{
  /*----------------------------------------------------------------------*/
  // we directly store the fluid-unscaled rhs_C_s residual contribution from the fluid solver which
  // corresponds to the actual acting forces

  // scaling for the structural residual is done when it is added to the global residual vector
  // get the coupling rhs from the xfluid, this vector is based on the boundary dis which is part of
  // the structure dis
  lambda_->update(scaling, *xfluid_->rhs_s_vec(cond_name_), 0.0);
  return;
}

/*----------------------------------------------------------------------*/
/* Write Output                                             ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::output(Core::IO::DiscretizationWriter& writer)
{
  //--------------------------------
  // output for Lagrange multiplier field (ie forces onto the structure, Robin-type forces
  // consisting of fluid forces and the Nitsche penalty term contribution)
  //--------------------------------
  auto multi_map_extractor_block_0 = get_map_extractor(0);
  if (!multi_map_extractor_block_0) FOUR_C_THROW("get_map_extractor(0) returned nullptr.");

  auto fullmap = multi_map_extractor_block_0->full_map();
  if (!fullmap) FOUR_C_THROW("extractor->full_map() is null.");

  auto lambdafull = std::make_shared<Core::LinAlg::Vector<double>>(*fullmap, true);

  // only do the costly insert if maps differ
  if (lambda_->get_map().same_as(*fullmap))
  {
    lambdafull->update(1.0, *lambda_, 0.0);
  }
  else
  {
    insert_vector(0, lambda_, 0, lambdafull, CouplingCommManager::partial_to_full);
  }

  // finally write the result
  writer.write_vector("fsilambda", lambdafull);
}
/*----------------------------------------------------------------------*/
/* Read Restart on the interface                            ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*get_map_extractor(0)->full_map(), true);
  reader.read_vector(lambdafull, "fsilambda");
  insert_vector(0, lambdafull, 0, lambda_, CouplingCommManager::full_to_partial);
}

/*-----------------------------------------------------------------------------------------*
| Get Timeface on the interface (for OST this is 1/(theta dt))                ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
double XFEM::XfsCouplingManager::get_interface_timefac()
{
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
