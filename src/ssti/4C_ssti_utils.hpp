// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSTI_UTILS_HPP
#define FOUR_C_SSTI_UTILS_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_ssi_clonestrategy.hpp"
#include "4C_sti_clonestrategy.hpp"
#include "4C_utils_parameter_list.fwd.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class SSIStructureWrapper;
}  // namespace Adapter

namespace Inpar
{
  namespace ScaTra
  {
    enum class MatrixType;
  }
}  // namespace Inpar

namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
  enum class MatrixType;
  class MultiMapExtractor;
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace ScaTra
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace ScaTra

namespace SSTI
{
  class SSTIAlgorithm;
  class SSTIMono;

  //! holds all maps in context of SSTI simulations
  class SSTIMaps
  {
   public:
    SSTIMaps(const SSTI::SSTIMono& ssti_mono_algorithm);

    //! get maps of subproblems
    //@{
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_scatra() const
    {
      return block_map_scatra_;
    }
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure() const
    {
      return block_map_structure_;
    }
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo() const
    {
      return block_map_thermo_;
    }
    //@}

    /*!
     * @brief global map extractor
     * @note only access with GetProblemPosition method
     */
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> maps_sub_problems() const
    {
      return maps_subproblems_;
    }

    //! return map with dofs on both sides of interface
    std::shared_ptr<Core::LinAlg::Map> map_interface(
        std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtyingstrategy) const;

    //! return block map with dofs on both sides of interface
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> maps_interface_blocks(
        std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtyingstrategy,
        Core::LinAlg::MatrixType scatramatrixtype, unsigned nummaps) const;

    //! return block map with dofs on slave side of interface
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> maps_interface_blocks_slave(
        const ScaTra::MeshtyingStrategyS2I& meshtyingstrategy,
        Core::LinAlg::MatrixType scatramatrixtype, unsigned nummaps) const;

   private:
    //! map extractor associated with all degrees of freedom inside scatra field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_scatra_;

    //! map extractor associated with all degrees of freedom inside structural field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure_;

    //! map extractor associated with all degrees of freedom inside thermo field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_thermo_;

    //! global map extractor (0: scalar transport, 1: structure, 2: thermo)
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> maps_subproblems_;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  //! holds all maps in context of SSTI monolithic simulations
  class SSTIMapsMono : public SSTIMaps
  {
   public:
    SSTIMapsMono(const SSTI::SSTIMono& ssti_mono_algorithm);

    //! map extractor associated with blocks of global system matrix
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_system_matrix() const
    {
      return block_map_system_matrix_;
    };

   private:
    //! map extractor associated with blocks of global system matrix
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_system_matrix_;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  //! sets up and holds all sub blocks of system matrices and system matrix for SSTI simulations
  class SSTIMatrices
  {
   public:
    SSTIMatrices(std::shared_ptr<SSTI::SSTIMapsMono> ssti_maps_mono,
        const Core::LinAlg::MatrixType matrixtype_global,
        const Core::LinAlg::MatrixType matrixtype_scatra, bool interfacemeshtying);

    //! method that clears all ssi matrices
    void clear_matrices();

    //! call complete on all coupling matrices
    void complete_coupling_matrices();

    //! call uncomplete on all coupling matrices
    void un_complete_coupling_matrices();

    std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix() { return systemmatrix_; };

    //! return sub blocks of system matrix
    //@{
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_domain()
    {
      return scatrastructuredomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_interface()
    {
      return scatrastructureinterface_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_thermo_domain()
    {
      return scatrathermodomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_thermo_interface()
    {
      return scatrathermointerface_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_domain()
    {
      return structurescatradomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_thermo_domain()
    {
      return structurethermodomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> thermo_scatra_domain()
    {
      return thermoscatradomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> thermo_scatra_interface()
    {
      return thermoscatrainterface_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> thermo_structure_domain()
    {
      return thermostructuredomain_;
    };
    std::shared_ptr<Core::LinAlg::SparseOperator> thermo_structure_interface()
    {
      return thermostructureinterface_;
    };
    //@}

   private:
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> setup_block_matrix(
        const Core::LinAlg::MultiMapExtractor& row_map,
        const Core::LinAlg::MultiMapExtractor& col_map);

    std::shared_ptr<Core::LinAlg::SparseMatrix> setup_sparse_matrix(
        const Core::LinAlg::Map& row_map);

    //! scalar transport matrix type
    const Core::LinAlg::MatrixType matrixtype_scatra_;

    //! maps for monolithic treatment of scalar transport-structure-thermo-interaction
    std::shared_ptr<SSTI::SSTIMapsMono> ssti_maps_mono_;

    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix_;
    //! subblocks of system matrix
    //@{
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrastructuredomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrastructureinterface_;
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermointerface_;
    std::shared_ptr<Core::LinAlg::SparseOperator> structurescatradomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> structurethermodomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatradomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatrainterface_;
    std::shared_ptr<Core::LinAlg::SparseOperator> thermostructuredomain_;
    std::shared_ptr<Core::LinAlg::SparseOperator> thermostructureinterface_;
    //@}

    //! bool indicating if we have at least one ssi interface meshtying condition
    const bool interfacemeshtying_;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  class ConvCheckMono
  {
   public:
    ConvCheckMono(const Teuchos::ParameterList params);

    //! Is this Newton step converged
    bool converged(const SSTI::SSTIMono& ssti_mono);

   private:
    //! maximum number of Newton-Raphson iteration steps
    const unsigned itermax_;

    //! relative tolerance for Newton-Raphson iteration
    const double itertol_;

    //! absolute tolerance for residual vectors
    const double restol_;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  class SSTIScatraStructureCloneStrategy : public SSI::ScatraStructureCloneStrategy
  {
   public:
    /// returns condition names to be copied (source and target name)
    std::map<std::string, std::string> conditions_to_copy() const override;

   protected:
    //! provide cloned element with element specific data (material etc.)
    void set_element_data(std::shared_ptr<Core::Elements::Element>
                              newele,     //! current cloned element on target discretization
        Core::Elements::Element* oldele,  //! current element on source discretization
        const int matid,                  //! material of cloned element
        const bool isnurbs                //! nurbs flag
        ) override;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  class SSTIScatraThermoCloneStrategy : public STI::ScatraThermoCloneStrategy
  {
   protected:
    /// returns condition names to be copied (source and target name)
    std::map<std::string, std::string> conditions_to_copy() const override;
  };
}  // namespace SSTI

FOUR_C_NAMESPACE_CLOSE

#endif
