// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_xfem_XFFcoupling_manager.hpp"

#include "4C_fluid_xfluid.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_discretization.hpp"


FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
| Constructor                                                                 ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
XFEM::XffCouplingManager::XffCouplingManager(std::shared_ptr<ConditionManager> condmanager,
    std::shared_ptr<FLD::XFluid> xfluid, std::shared_ptr<FLD::XFluid> fluid, std::vector<int> idx)
    : CouplingCommManager(fluid->discretization(), "XFEMSurfFluidFluid", 0, 3),
      fluid_(fluid),
      xfluid_(xfluid),
      cond_name_("XFEMSurfFluidFluid"),
      idx_(idx)
{
  if (idx_.size() != 2)
    FOUR_C_THROW("XFFCoupling_Manager required two block ( 2 != {})", idx_.size());

  // Coupling_Comm_Manager create all Coupling Objects now with Fluid has idx = 0, Fluid has idx =
  // 1!
  mcffi_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFluidFluid>(
      condmanager->get_mesh_coupling(cond_name_));
  if (mcffi_ == nullptr) FOUR_C_THROW(" Failed to get MeshCouplingFFI for embedded fluid!");
}

/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XffCouplingManager::init_coupling_states() {}


/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XffCouplingManager::set_coupling_states()
{
  std::cout << "SetCouplingStates in XFFCoupling_Manager" << std::endl;

  /// free the fluid-fluid interface
  mcffi_->set_interface_free();

  mcffi_->update_displacement_iteration_vectors();  // update last iteration interface displacements
  Core::LinAlg::export_to(*fluid_->dispnp(), *mcffi_->i_dispnp());
  Core::LinAlg::export_to(*fluid_->velnp(), *mcffi_->i_velnp());
  Core::LinAlg::export_to(*fluid_->veln(), *mcffi_->i_veln());

  std::shared_ptr<Core::LinAlg::Vector<double>> tmp_diff =
      std::make_shared<Core::LinAlg::Vector<double>>((*mcffi_->i_dispnp()).get_map());
  tmp_diff->update(1.0, *mcffi_->i_dispnp(), -1.0, *mcffi_->i_dispnpi(), 0.0);

  double norm = 0.0;
  tmp_diff->norm_inf(&norm);

  if (norm < 1e-12)
    std::cout << "No change in XFF interface position!!!" << std::endl;
  else
  {
    std::cout << "Change in XFF interface position??? with infnorm " << norm << std::endl;
  }


  //  std::cout << "mcffi-IDispnp()" << *mcffi_->IDispnp()  << std::endl;
  //  std::cout << "mcffi-IVelpnp()" << *mcffi_->IVelnp()   << std::endl;

  //  //1 update last increment, before we set new idispnp
  //  mcffi_->update_displacement_iteration_vectors();
  //
  //  //2 Set Displacement on both mesh couplings ... we get them from the embedded fluid field!
  //  insert_vector(0,fluid_->Dispnp(),0,mcffi_->IDispnp(),Coupling_Comm_Manager::full_to_partial);
  //
  //
  //  insert_vector(0,fluid_->Velnp(),0,mcffi_->IVelnp(),Coupling_Comm_Manager::full_to_partial);
  //

  return;
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling matrixes to the global systemmatrix                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XffCouplingManager::add_coupling_matrix(
    Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling)
{
  /*----------------------------------------------------------------------*/
  // Coupling blocks C_fxf, C_xff and C_ff
  /*----------------------------------------------------------------------*/
  Core::LinAlg::SparseMatrix& C_ff_block = (systemmatrix)(idx_[0], idx_[0]);
  /*----------------------------------------------------------------------*/

  // add the coupling block C_ss on the already existing diagonal block
  Core::LinAlg::matrix_add(*xfluid_->c_ss_matrix(cond_name_), false, scaling, C_ff_block, 1.0);

  Core::LinAlg::SparseMatrix& C_xff_block = (systemmatrix)(idx_[1], idx_[0]);
  Core::LinAlg::SparseMatrix& C_fxf_block = (systemmatrix)(idx_[0], idx_[1]);

  Core::LinAlg::matrix_add(*xfluid_->c_sx_matrix(cond_name_), false, scaling, C_fxf_block, 1.0);
  Core::LinAlg::matrix_add(*xfluid_->c_xs_matrix(cond_name_), false, scaling, C_xff_block, 1.0);
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling rhs                                                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XffCouplingManager::add_coupling_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const Core::LinAlg::MultiMapExtractor& me, double scaling)
{
  // REMARK: Copy this vector to store the correct lambda_ in update!
  Core::LinAlg::Vector<double> coup_rhs_sum(*xfluid_->rhs_s_vec(cond_name_));

  coup_rhs_sum.scale(scaling);

  Core::LinAlg::Vector<double> coup_rhs(*me.map(idx_[0]), true);
  Core::LinAlg::export_to(coup_rhs_sum, coup_rhs);
  me.add_vector(coup_rhs, idx_[0], *rhs);

  return;
}

FOUR_C_NAMESPACE_CLOSE
