// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_ADAPTER_CONSTRAINTBRIDGE_HPP
#define FOUR_C_FBI_ADAPTER_CONSTRAINTBRIDGE_HPP

#include "4C_config.hpp"

#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>
#include <vector>


FOUR_C_NAMESPACE_OPEN

// Forward declaration

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg
namespace BeamInteraction
{
  class BeamContactPair;
}

namespace GeometryPair
{
  class LineTo3DEvaluationData;
}

namespace FBI
{
  class BeamToFluidMeshtyingParams;
  namespace Utils
  {
    class FBIAssemblyStrategy;
  }
}  // namespace FBI

namespace Adapter
{
  class ConstraintEnforcerFactory;

  /**
   *   \brief Abstract class to be overloaded by different adapter implementations connecting a
   * constraint enforcement technique with a discretization approach for Fluid-beam interaction.
   *
   * The idea is, that this method should act a bit like a mini
   * BeamInteraction::SubmodelEvaluator::BeamContact and manage the fluid-beam interaction pairs and
   * their assembly into global contribution matrices
   */
  class FBIConstraintBridge
  {
    friend ConstraintEnforcerFactory;

   public:
    /// empty destructor
    virtual ~FBIConstraintBridge() = default;

    /**
     * \brief Initializes all members of the class
     *
     * \param[in] beam_map Row Map of the structure discretization
     * \param[in} fluid_map Row Map of the fluid discretization
     * \param[in] fluidmatrix system matrix of the fluid matrix in correct from
     * \param[in] fluidmeshtying bool indicating if fluid meshtying is included
     */
    virtual void setup(const Core::LinAlg::Map* beam_map, const Core::LinAlg::Map* fluid_map,
        std::shared_ptr<Core::LinAlg::SparseOperator> fluidmatrix, bool fluidmeshtying);

    /**
     * \brief Computes the coupling matrices
     *
     * This is where the magic happens. The global meshtying contributions are integrated using
     * information of the beam elements, the fluid elements and their position relative to each
     * other.
     *
     */
    virtual void evaluate(std::shared_ptr<const Core::FE::Discretization> discretization1,
        std::shared_ptr<const Core::FE::Discretization> discretization2,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
        std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel) = 0;

    /**
     * \brief Wraps the ResetState function of the pair
     *
     * Here, the current values of the single fields lie positions and velocities are handed
     * to/updated in the pair
     */
    virtual void reset_pair(const std::vector<double> beam_centerline_dofvec,
        const std::vector<double> fluid_nodal_dofvec,
        std::shared_ptr<BeamInteraction::BeamContactPair> interactionpair);

    /// Creates a fluid_beam_meshtying pair
    virtual void create_pair(std::vector<Core::Elements::Element const*> elements,
        std::vector<double> beam_centerline_dofvec, std::vector<double> fluid_nodal_dofvec);

    // Get function for the meshtying pairs meshtying_pairs_
    virtual std::shared_ptr<std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>>
    get_pairs() const final
    {
      return meshtying_pairs_;
    };

    /// returns data container holding all beam interaction related parameters
    virtual std::shared_ptr<FBI::BeamToFluidMeshtyingParams> get_params() const final
    {
      return beam_interaction_params_;
    };

    /// returns data container geometry_evaluation-data_ holding all geometry related evaluation
    /// data
    virtual std::shared_ptr<GeometryPair::LineTo3DEvaluationData> get_geometry_data() const final
    {
      return geometry_evaluation_data_;
    };

    /// Clears the pair vector and segmentation information
    virtual void clear();

    /// Resets class members of the bridge
    virtual void reset_bridge() = 0;

    /// Sets the fluid solve flag
    virtual void prepare_fluid_solve() = 0;

    /// Matrix containing only structure side contributions \f$C_{ss}\f$
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> get_css() const = 0;

    /// Matrix containing only fluid side contributions \f$C_{ff}\f$
    virtual std::shared_ptr<const Core::LinAlg::SparseOperator> get_cff() const = 0;

    /// Matrix containing mixed fluid side contributions \f$C_{fs}\f$
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> get_cfs() const = 0;

    /// Matrix containing mixed structure side contributions \f$C_{sf}\f$
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> get_csf() const = 0;

    /// Force vector acting on the fluid side \f$f_f\f$
    virtual std::shared_ptr<const Core::LinAlg::FEVector<double>> get_fluid_coupling_residual()
        const = 0;

    /// Force vector acting on the structure side \f$f_s\f$
    virtual std::shared_ptr<const Core::LinAlg::FEVector<double>> get_structure_coupling_residual()
        const = 0;

   protected:
    /** \brief You will have to use the Adapter::ConstraintEnforcerFactory
     *
     */
    FBIConstraintBridge();

    /// data container holding all beam interaction related parameters
    std::shared_ptr<FBI::BeamToFluidMeshtyingParams> beam_interaction_params_;

    /// Store the assembly strategy here to hand into the assembly manager
    std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy_;

   private:
    /// meshtying pairs
    std::shared_ptr<std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>>
        meshtying_pairs_;

    /// data container holding all geometry related evaluation data
    std::shared_ptr<GeometryPair::LineTo3DEvaluationData> geometry_evaluation_data_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
