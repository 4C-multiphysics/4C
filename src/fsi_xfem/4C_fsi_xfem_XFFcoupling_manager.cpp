/*----------------------------------------------------------------------*/
/*! \file
\brief Coupling Manager for eXtended Fluid Fluid Coupling

\level 3


*----------------------------------------------------------------------*/
#include "4C_fsi_xfem_XFFcoupling_manager.hpp"

#include "4C_fluid_xfluid.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
| Constructor                                                                 ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
XFEM::XffCouplingManager::XffCouplingManager(Teuchos::RCP<ConditionManager> condmanager,
    Teuchos::RCP<FLD::XFluid> xfluid, Teuchos::RCP<FLD::XFluid> fluid, std::vector<int> idx)
    : CouplingCommManager(fluid->discretization(), "XFEMSurfFluidFluid", 0, 3),
      fluid_(fluid),
      xfluid_(xfluid),
      cond_name_("XFEMSurfFluidFluid"),
      idx_(idx)
{
  if (idx_.size() != 2)
    FOUR_C_THROW("XFFCoupling_Manager required two block ( 2 != %d)", idx_.size());

  // Coupling_Comm_Manager create all Coupling Objects now with Fluid has idx = 0, Fluid has idx =
  // 1!
  mcffi_ = Teuchos::rcp_dynamic_cast<XFEM::MeshCouplingFluidFluid>(
      condmanager->get_mesh_coupling(cond_name_));
  if (mcffi_ == Teuchos::null) FOUR_C_THROW(" Failed to get MeshCouplingFFI for embedded fluid!");
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

  Teuchos::RCP<Core::LinAlg::Vector<double>> tmp_diff =
      Teuchos::rcp(new Core::LinAlg::Vector<double>((*mcffi_->i_dispnp()).Map()));
  tmp_diff->Update(1.0, *mcffi_->i_dispnp(), -1.0, *mcffi_->i_dispnpi(), 0.0);

  double norm = 0.0;
  tmp_diff->NormInf(&norm);

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
  C_ff_block.add(*xfluid_->c_ss_matrix(cond_name_), false, scaling, 1.0);

  Core::LinAlg::SparseMatrix& C_xff_block = (systemmatrix)(idx_[1], idx_[0]);
  Core::LinAlg::SparseMatrix& C_fxf_block = (systemmatrix)(idx_[0], idx_[1]);

  C_fxf_block.add(*xfluid_->c_sx_matrix(cond_name_), false, scaling, 1.0);
  C_xff_block.add(*xfluid_->c_xs_matrix(cond_name_), false, scaling, 1.0);
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling rhs                                                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XffCouplingManager::add_coupling_rhs(Teuchos::RCP<Core::LinAlg::Vector<double>> rhs,
    const Core::LinAlg::MultiMapExtractor& me, double scaling)
{
  // REMARK: Copy this vector to store the correct lambda_ in update!
  Teuchos::RCP<Core::LinAlg::Vector<double>> coup_rhs_sum =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*xfluid_->rhs_s_vec(cond_name_)));

  coup_rhs_sum->Scale(scaling);

  Teuchos::RCP<Core::LinAlg::Vector<double>> coup_rhs =
      Teuchos::rcp(new Core::LinAlg::Vector<double>(*me.Map(idx_[0]), true));
  Core::LinAlg::export_to(*coup_rhs_sum, *coup_rhs);
  me.add_vector(coup_rhs, idx_[0], rhs);

  return;
}

FOUR_C_NAMESPACE_CLOSE
