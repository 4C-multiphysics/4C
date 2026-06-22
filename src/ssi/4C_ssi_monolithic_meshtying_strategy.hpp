// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_MONOLITHIC_MESHTYING_STRATEGY_HPP
#define FOUR_C_SSI_MONOLITHIC_MESHTYING_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace Adapter
{
  class CouplingSourceConverter;
}

namespace SSI
{
  namespace Utils
  {
    class SSIMaps;
    class SSIMeshTying;
  }  // namespace Utils

  class SsiMono;

  //! base functionality for scatra structure interaction mesh tying
  class MeshtyingStrategyBase
  {
   public:
    //! constructor
    MeshtyingStrategyBase() = default;

    //! virtual destructor
    virtual ~MeshtyingStrategyBase() = default;

    /*!
     * @brief apply mesh tying to structure matrix
     *
     * @param[out] ssi_structure_matrix    structure matrix including mesh tying constraints
     * @param[in] structure_matrix         structure matrix from the structure problem
     * @param[in] ssi_structure_meshtying  ssi mesh tying object providing the relevant maps and
     *                                     converters for mesh tying
     * @param[in] do_uncomplete            flag indicating if we need to uncomplete the matrix
     *                                     before adding something
     */
    void apply_meshtying_to_structure_matrix(Core::LinAlg::SparseMatrix& ssi_structure_matrix,
        const Core::LinAlg::SparseMatrix& structure_matrix,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete);

    /*!
     * @brief apply mesh tying to scatra manifold structure matrix
     *
     * @param[in,out] manifold_structure_matrix  scalar transport on manifold structure matrix
     * @param[in] ssi_maps                       ssi maps object
     * @param[in] ssi_structure_meshtying        ssi mesh tying object providing the relevant maps
     *                                           and converters for mesh tying converters for mesh
     *                                           tying
     * @param[in]     do_uncomplete              flag indicating if we need to uncomplete the matrix
     *                                           before adding something
     */
    virtual void apply_meshtying_to_scatra_manifold_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> manifold_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) = 0;

    /*!
     * @brief apply mesh tying to scatra structure matrix
     *
     * @param[in,out] scatra_structure_matrix  scatra structure matrix
     * @param[in] ssi_maps                     ssi maps object
     * @param[in] ssi_structure_meshtying      ssi mesh tying object providing the relevant maps and
     *                                         converters for mesh tying converters for mesh tying
     * @param[in]     do_uncomplete            flag indicating if we need to uncomplete the matrix
     *                                         before adding something
     */
    virtual void apply_meshtying_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) = 0;

    /*!
     * @brief apply mesh tying to structure right hand side vector
     *
     * @param[in] structure_rhs  structure right hand side vector without mesh tying contributions
     * @param[in] ssi_maps                 ssi maps object
     * @param[in] ssi_structure_meshtying  ssi mesh tying object providing the relevant maps and
     *                                     converters for mesh tying converters for mesh tying
     * @return structure right hand side vector including mesh tying contributions
     */
    Core::LinAlg::Vector<double> apply_meshtying_to_structure_rhs(
        const Core::LinAlg::Vector<double>& structure_rhs, const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying);

    /*!
     * @brief apply mesh tying to the structure scatra matrix
     *
     * @param[in,out] structure_scatra_matrix  structure scatra matrix
     * @param[in] ssi_maps                     ssi maps object
     * @param[in] ssi_structure_meshtying      ssi mesh tying object providing the relevant maps and
     *                                         converters for mesh tying converters for mesh tying
     * @param[in]     do_uncomplete            flag indicating if we need to uncomplete the matrix
     *                                         before adding something
     */
    virtual void apply_meshtying_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) = 0;

   protected:
    /*!
     * @brief apply mesh tying to structure-xxx block
     *
     * @param[out] ssi_structure_xxx_matrix  structure xxx matrix block including mesh tying
     *                                       constraints
     * @param[in] structure_xxx_matrix       structure xxx matrix block
     * @param[in] ssi_structure_meshtying    ssi mesh tying object providing the relevant maps and
     *                                       converters for mesh tying converters for mesh tying
     */
    void apply_meshtying_to_structure_xxx(Core::LinAlg::SparseMatrix& ssi_structure_xxx_matrix,
        const Core::LinAlg::SparseMatrix& structure_xxx_matrix,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying);

    /*!
     * @brief apply mesh tying to xxx-structure block
     *
     * @param[out] ssi_xxx_structure_matrix  xxx structure matrix block including mesh tying
     *                                       constraints
     * @param[in] xxx_structure_matrix       xxx structure matrix block
     * @param[in] ssi_structure_meshtying    ssi mesh tying object providing the relevant maps and
     *                                       converters for mesh tying converters for mesh tying
     */
    void apply_meshtying_to_xxx_structure(Core::LinAlg::SparseMatrix& ssi_xxx_structure_matrix,
        const Core::LinAlg::SparseMatrix& xxx_structure_matrix,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying);

    //! scatra structure contribution matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> temp_scatra_struct_mat_;

    //! scatra-manifold structure system matrix used to apply mesh tying to this matrix block
    std::shared_ptr<Core::LinAlg::SparseOperator> temp_scatramanifold_struct_mat_;

    //! structure scatra system matrix used to apply mesh tying to this matrix block
    std::shared_ptr<Core::LinAlg::SparseOperator> temp_struct_scatra_mat_;

   private:
    /*!
     * @brief finalize mesh tying by applying pseudo dirichlet conditions to the slave side, i.e.
     * write 1.0 on main diagonal of slave side dofs
     *
     * @param[in,out] ssi_structure_matrix  structure matrix with mesh tying constraints
     * @param[in] ssi_structure_meshtying   ssi mesh tying object providing the relevant maps and
     *                                      converters for mesh tying converters for mesh tying
     */
    void finalize_meshtying_structure_matrix(Core::LinAlg::SparseMatrix& ssi_structure_matrix,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying);
  };

  //! SSI problem is represented by one sparse matrix
  class MeshtyingStrategySparse : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategySparse(bool is_scatra_manifold, const SSI::Utils::SSIMaps& ssi_maps);

    void apply_meshtying_to_scatra_manifold_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> manifold_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;

    void apply_meshtying_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;

    void apply_meshtying_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;
  };

  //! SSI problem is composed of sub matrices
  class MeshtyingStrategyBlock : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyBlock(bool is_scatra_manifold, const SSI::Utils::SSIMaps& ssi_maps);

    void apply_meshtying_to_scatra_manifold_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> manifold_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;

    void apply_meshtying_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;

    void apply_meshtying_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix,
        const SSI::Utils::SSIMaps& ssi_maps,
        const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, bool do_uncomplete) override;

   protected:
    //! position of scatra blocks in system matrix
    [[nodiscard]] const std::vector<int>& block_position_scatra() const
    {
      return block_position_scatra_;
    }

    //! position of scatra manifold blocks in system matrix
    [[nodiscard]] const std::vector<int>& block_position_scatra_manifold() const
    {
      return block_position_scatra_manifold_;
    }

    //! position of structure block in system matrix
    [[nodiscard]] int position_structure() const { return position_structure_; };

   private:
    //! position of scatra blocks in system matrix
    const std::vector<int> block_position_scatra_;

    //! position of scatra manifold blocks in system matrix
    std::vector<int> block_position_scatra_manifold_;

    //! position of structure block in system matrix
    const int position_structure_;
  };

  //! build specific mesh tying strategy
  std::unique_ptr<SSI::MeshtyingStrategyBase> build_meshtying_strategy(bool is_scatra_manifold,
      Core::LinAlg::MatrixType matrixtype_scatra, const SSI::Utils::SSIMaps& ssi_maps);
}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif
