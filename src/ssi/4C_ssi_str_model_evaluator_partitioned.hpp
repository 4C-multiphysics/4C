// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_STR_MODEL_EVALUATOR_PARTITIONED_HPP
#define FOUR_C_SSI_STR_MODEL_EVALUATOR_PARTITIONED_HPP

#include "4C_config.hpp"

#include "4C_ssi_str_model_evaluator_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class Structure;
}  // namespace Adapter

namespace SSI
{
  class SSIPart;
}

namespace Solid
{
  namespace ModelEvaluator
  {
    class PartitionedSSI : public BaseSSI
    {
     public:
      //! constructor
      PartitionedSSI(const std::shared_ptr<const SSI::SSIPart>
              ssi_part  //!< partitioned algorithm for scalar-structure interaction
      );

      void setup() override;

      //! @name Functions which are derived from the base generic class
      //! @{
      [[nodiscard]] Solid::ModelType type() const override
      {
        return Solid::model_partitioned_coupling;
      }

      void assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      void assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      void determine_stress_strain() override {};

      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override;
      //! @}

     private:
      //! partitioned algorithm for scalar-structure interaction
      const std::shared_ptr<const SSI::SSIPart> ssi_part_;
    };  // class PartitionedSSI

  }  // namespace ModelEvaluator
}  // namespace Solid


FOUR_C_NAMESPACE_CLOSE

#endif
