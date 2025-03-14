// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_CONTACT_STRATEGY_HPP
#define FOUR_C_SSI_CONTACT_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class NitscheStrategySsi;
}

namespace Core::LinAlg
{
  enum class MatrixType;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace SSI
{
  class SsiMono;

  namespace Utils
  {
    class SSIMaps;
  }

  //! base functionality for scatra structure contact interaction
  class ContactStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ContactStrategyBase() = default;

    //! constructor
    explicit ContactStrategyBase(
        std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
        std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps);

    //! apply contact contributions to the scatra residual
    void apply_contact_to_scatra_residual(
        std::shared_ptr<Core::LinAlg::Vector<double>> scatra_residual);

    //! apply contact contributions to scatra sub matrix
    virtual void apply_contact_to_scatra_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_matrix) = 0;

    //! apply contact contributions to scatra-structure sub matrix
    virtual void apply_contact_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix) = 0;

    //! apply contact contributions to structure-scatra sub matrix
    virtual void apply_contact_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix) = 0;

   protected:
    //! return contact nitsche strategy for ssi problems
    std::shared_ptr<CONTACT::NitscheStrategySsi> nitsche_strategy_ssi() const
    {
      return contact_strategy_nitsche_;
    }

    //! this object holds all maps relevant to monolithic scalar transport - structure interaction
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps() const { return ssi_maps_; }

   private:
    //! store contact nitsche strategy for ssi problems
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_strategy_nitsche_;

    //! this object holds all maps relevant to monolithic/partitioning scalar transport - structure
    //! interaction
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps_;
  };

  //! SSI (sub) matrices are sparse matrices
  class ContactStrategySparse : public ContactStrategyBase
  {
   public:
    //! constructor
    explicit ContactStrategySparse(
        std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
        std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps);

    void apply_contact_to_scatra_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_matrix) override;

    void apply_contact_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix) override;

    void apply_contact_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix) override;
  };

  //! SSI (sub) matrices are block matrices
  class ContactStrategyBlock : public ContactStrategyBase
  {
   public:
    //! constructor
    explicit ContactStrategyBlock(
        std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
        std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps);

    void apply_contact_to_scatra_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_matrix) override;

    void apply_contact_to_scatra_structure(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix) override;

    void apply_contact_to_structure_scatra(
        std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix) override;
  };

  //! build specific contact strategy
  std::shared_ptr<SSI::ContactStrategyBase> build_contact_strategy(
      std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
      std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps,
      Core::LinAlg::MatrixType matrixtype_scatra);
}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif
