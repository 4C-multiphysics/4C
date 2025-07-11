// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_ADAPTER_CONSTRAINTBRIDGE_PENALTY_HPP
#define FOUR_C_FBI_ADAPTER_CONSTRAINTBRIDGE_PENALTY_HPP

#include "4C_config.hpp"

#include "4C_fbi_adapter_constraintbridge.hpp"
#include "4C_linalg_fevector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  class BeamToFluidMeshtyingVtkOutputWriter;
}

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg
namespace Adapter
{
  /**
   *   \brief Implementation connecting the penalty constraint enforcement technique with a
   *   discretization approach for Fluid-beam interaction
   *
   */
  class FBIConstraintBridgePenalty : public FBIConstraintBridge
  {
    friend class ConstraintEnforcerFactory;
    friend class FBIPenaltyConstraintenforcer;
    friend class BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter;
    friend class FBIConstraintenforcer;

   public:
    /**
     * \brief Initializes all members of the class     *
     */
    void setup(const Core::LinAlg::Map* beam_map, const Core::LinAlg::Map* fluid_map,
        std::shared_ptr<Core::LinAlg::SparseOperator> fluidmatrix, bool fluidmeshtying) override;

    /**
     * \brief Computes the coupling matrices
     *
     * This is where the magic happens. The global meshtying contributions are integrated using
     * information of the beam elements, the fluid elements and their position relative to each
     * other.
     *
     */

    void evaluate(std::shared_ptr<const Core::FE::Discretization> discretization1,
        std::shared_ptr<const Core::FE::Discretization> discretization2,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
        std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel) override;

    /// resets the matrices and vectors to zero
    void reset_bridge() override;

    void prepare_fluid_solve() override { set_weak_dirichlet_flag(); };

    /// Matrix containing only structure side contributions \f$C_{ss}\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_css() const override { return css_; };

    /// Matrix containing only fluid side contributions \f$C_{ff}\f$
    std::shared_ptr<const Core::LinAlg::SparseOperator> get_cff() const override { return cff_; };

    /// Matrix containing mixed fluid side contributions \f$C_{fs}\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_cfs() const override { return cfs_; };

    /// Matrix containing mixed structure side contributions \f$C_{sf}\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_csf() const override { return csf_; };

    /// Negative RHS coupling contribution for the fluid partition \f$f_f\f$
    std::shared_ptr<const Core::LinAlg::FEVector<double>> get_fluid_coupling_residual()
        const override
    {
      return ff_;
    };

    /// Force vector acting on the structure side \f$f_s\f$
    std::shared_ptr<const Core::LinAlg::FEVector<double>> get_structure_coupling_residual()
        const override
    {
      return fs_;
    };

   protected:
    /** \brief You will have to use the Adapter::ConstraintEnforcerFactory
     *
     */
    FBIConstraintBridgePenalty()
        : css_(nullptr),
          cff_(nullptr),
          cfs_(nullptr),
          csf_(nullptr),
          ff_(nullptr),
          fs_(nullptr),
          fluid_scaled_(false),
          structure_scaled_(false) {};

    /**
     * \brief Sets the flag to compute only force contributions from the beam
     *
     * This allows for a more efficient implementation for the assembly of weak dirichlet
     * contributions to the fluid field, since it avoids a global multiplication of the stiffness
     * $\f C_sf \f$ matrix with the structure velocity
     */
    void set_weak_dirichlet_flag();

    /// Sets the flag to compute force contributions from beam and fluid
    void unset_weak_dirichlet_flag();

    /// Scales all structure vectors and martrices with penalty
    void scale_penalty_structure_contributions();

    /// Scales all fluid vectors and martrices with penalty
    void scale_penalty_fluid_contributions();

   private:
    /// Coupling matrix containing only structure side contributions \f$C_ss\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> css_;

    /// Coupling matrix containing only fluid side contributions \f$C_ff\f$
    std::shared_ptr<Core::LinAlg::SparseOperator> cff_;

    /// Coupling matrix containing mixed fluid side contributions \f$C_fs\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> cfs_;

    /// Coupling matrix containing mixed structure side contributions \f$C_sf\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> csf_;

    /// Force vector acting on the fluid side \f$f_f\f$
    std::shared_ptr<Core::LinAlg::FEVector<double>> ff_;

    /// Force vector acting on the structure side \f$f_s\f$
    std::shared_ptr<Core::LinAlg::FEVector<double>> fs_;

    /// Bool to keep track if the fluid coupling contributions were already scaled with the penalty
    /// parameter
    bool fluid_scaled_;

    /// Bool to keep track if the structure coupling contributions were already scaled with the
    /// penalty parameter
    bool structure_scaled_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
