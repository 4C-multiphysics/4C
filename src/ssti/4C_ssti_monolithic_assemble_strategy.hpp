// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSTI_MONOLITHIC_ASSEMBLE_STRATEGY_HPP
#define FOUR_C_SSTI_MONOLITHIC_ASSEMBLE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_ssti_monolithic.hpp"
#include "4C_ssti_utils.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
  class MultiMapExtractor;
  class Solver;
  class SparseMatrix;
  enum class MatrixType;
}  // namespace Core::LinAlg


namespace SSTI
{

  /*!
  We have three options how the global system matrix and the sub matrices are arranged:
  1) System matrix: sparse
    ->Scatra + Thermo matrix sparse
    ->Structure matrix sparse
  2) System matrix: block
    2a) Scatra + Thermo matrix block
    ->Structure matrix sparse
    2b) Scatra + Thermo matrix sparse
    ->Structure matrix sparse

  The inheritance hierarchy is appropriate*/
  class AssembleStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~AssembleStrategyBase() = default;

    AssembleStrategyBase(std::shared_ptr<const SSTI::SSTIMono> ssti_mono);

    //! write 1.0 on main diagonal of slave side dofs
    virtual void apply_meshtying_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) = 0;

    //! apply structural Dirichlet boundary conditions on system matrix
    virtual void apply_structural_dbc_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) = 0;

    //! assemble RHS
    void assemble_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> RHS,
        std::shared_ptr<const Core::LinAlg::Vector<double>> RHSscatra,
        const Core::LinAlg::Vector<double>& RHSstructure,
        std::shared_ptr<const Core::LinAlg::Vector<double>> RHSthermo);

    //! assemble ScaTra-Block into system matrix
    virtual void assemble_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain) = 0;

    //! assemble ScaTra-Structure-Block (domain contributions) into system matrix
    virtual void assemble_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface) = 0;

    //! assemble ScaTra-Thermo-Block (domain contributions) into system matrix
    virtual void assemble_scatra_thermo_domain(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain) = 0;

    virtual void assemble_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface) = 0;

    //! assemble Structure-Block into system matrix
    virtual void assemble_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain) = 0;

    //! assemble Structure-ScaTra-Block (domain contributions) into system matrix
    virtual void assemble_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain) = 0;

    //! assemble Thermo-Block into system matrix
    virtual void assemble_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain) = 0;

    //! assemble Thermo-ScaTra-Block into system matrix
    virtual void assemble_thermo_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface) = 0;

    //! assemble Thermo-Structure-Block into system matrix
    virtual void assemble_thermo_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface) = 0;

    //! assemble Thermo-Block into system matrix
    virtual void assemble_structure_thermo(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain) = 0;

   protected:
    //! write 1.0 on main diagonal of slave side dofs
    void apply_meshtying_sys_mat(Core::LinAlg::SparseMatrix& systemmatrix_structure);

    //! assemble x-structure block into system matrix for meshtying
    void assemble_xxx_structure_meshtying(Core::LinAlg::SparseMatrix& systemmatrix_x_structure,
        const Core::LinAlg::SparseMatrix& x_structurematrix);

    //! assemble structure block  into system matrix for meshtying
    void assemble_structure_meshtying(Core::LinAlg::SparseMatrix& systemmatrix_structure,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain);

    //! assemble structure-x block into system matrix for meshtying
    void assemble_structure_xxx_meshtying(Core::LinAlg::SparseMatrix& systemmatrix_structure_x,
        const Core::LinAlg::SparseMatrix& structures_x_matrix);

    //! Meshtying adapters
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_thermo() const
    {
      return ssti_mono_->meshtying_thermo();
    }
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_scatra() const
    {
      return ssti_mono_->meshtying_scatra();
    }
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssti_structure_meshtying() const
    {
      return ssti_mono_->ssti_structure_mesh_tying();
    }
    //@}

    //! SSTI mono maps
    std::shared_ptr<SSTI::SSTIMapsMono> all_maps() const { return ssti_mono_->all_maps(); }

    std::shared_ptr<Adapter::SSIStructureWrapper> structure_field() const
    {
      return ssti_mono_->structure_field();
    }

    //! flag indicating meshtying
    bool interface_meshtying() const { return ssti_mono_->interface_meshtying(); }

   private:
    //! monolithic algorithm for scalar-structure-thermo interaction
    const std::shared_ptr<const SSTI::SSTIMono> ssti_mono_;
  };

  //======================================================================================================
  //! SSTI problem is organized in sub matrices
  class AssembleStrategyBlock : public AssembleStrategyBase
  {
   public:
    AssembleStrategyBlock(std::shared_ptr<const SSTI::SSTIMono> ssti_mono);

    void apply_meshtying_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override = 0;

    void apply_structural_dbc_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override;

    void assemble_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain) override = 0;

    void assemble_scatra_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface) override = 0;

    void assemble_scatra_thermo_domain(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain) override = 0;

    void assemble_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface) override = 0;

    void assemble_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain) override = 0;

    void assemble_structure_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain) override = 0;

    void assemble_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain) override = 0;

    void assemble_thermo_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface) override = 0;

    void assemble_thermo_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface) override = 0;

    void assemble_structure_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain) override = 0;

   protected:
    //! position of scatra blocks in system matrix
    std::vector<int> block_position_scatra() const { return block_position_scatra_; };

    //! position of thermo blocks in system matrix
    std::vector<int> block_position_thermo() const { return block_position_thermo_; };

    //! position of structure block in system matrix
    int position_structure() const { return position_structure_; };

   private:
    //! position of scatra blocks in system matrix
    std::vector<int> block_position_scatra_;

    //! position of thermo blocks in system matrix
    std::vector<int> block_position_thermo_;

    //! position of structure block in system matrix
    int position_structure_;
  };

  // *********************************************************************************************
  //! SSTI problem is organized in sparse structure sub matrix and block scatra sub matrix
  class AssembleStrategyBlockBlock : public AssembleStrategyBlock
  {
   public:
    AssembleStrategyBlockBlock(std::shared_ptr<const SSTI::SSTIMono> ssti_mono);

    void apply_meshtying_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override;

    void assemble_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain) override;

    void assemble_scatra_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface) override;

    void assemble_scatra_thermo_domain(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain) override;

    void assemble_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface) override;

    void assemble_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain) override;

    void assemble_structure_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain) override;

    void assemble_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain) override;

    void assemble_thermo_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface) override;

    void assemble_thermo_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface) override;

    void assemble_structure_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain) override;

   private:
    //! assemble interface contribution from thermo-scatra block
    void assemble_thermo_scatra_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface);
  };

  // *********************************************************************************************
  //! SSTI problem is organized in sparse sub matrices
  class AssembleStrategyBlockSparse : public AssembleStrategyBlock
  {
   public:
    AssembleStrategyBlockSparse(std::shared_ptr<const SSTI::SSTIMono> ssti_mono);

    void apply_meshtying_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override;

    void assemble_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain) override;

    void assemble_scatra_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface) override;

    void assemble_scatra_thermo_domain(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain) override;

    void assemble_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface) override;

    void assemble_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain) override;

    void assemble_structure_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain) override;

    void assemble_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain) override;

    void assemble_thermo_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface) override;

    void assemble_thermo_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface) override;

    void assemble_structure_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain) override;

   private:
    //! assemble interface contribution from thermo-scatra block
    void assemble_thermo_scatra_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface);
  };

  //======================================================================================================
  //! SSTI problem is organized in one sparse matrix
  class AssembleStrategySparse : public AssembleStrategyBase
  {
   public:
    AssembleStrategySparse(std::shared_ptr<const SSTI::SSTIMono> ssti_mono);

    void apply_meshtying_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override;

    void apply_structural_dbc_system_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix) override;

    void assemble_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain) override;

    void assemble_scatra_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface) override;

    void assemble_scatra_thermo_domain(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain) override;

    void assemble_scatra_thermo_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface) override;

    void assemble_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain) override;

    void assemble_structure_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain) override;

    void assemble_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain) override;

    void assemble_thermo_scatra(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface) override;

    void assemble_thermo_structure(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface) override;

    void assemble_structure_thermo(std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain) override;

   private:
    //! assemble interface contribution from thermo-scatra block
    void assemble_thermo_scatra_interface(
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface);
  };

  //! build specific assemble strategy
  std::shared_ptr<SSTI::AssembleStrategyBase> build_assemble_strategy(
      std::shared_ptr<const SSTI::SSTIMono> ssti_mono, Core::LinAlg::MatrixType matrixtype_ssti,
      Core::LinAlg::MatrixType matrixtype_scatra);

}  // namespace SSTI
FOUR_C_NAMESPACE_CLOSE

#endif
