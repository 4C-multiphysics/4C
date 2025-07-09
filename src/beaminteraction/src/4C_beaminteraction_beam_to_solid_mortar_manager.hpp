// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_MORTAR_MANAGER_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_MORTAR_MANAGER_HPP


#include "4C_config.hpp"

#include "4C_inpar_beaminteraction.hpp"
#include "4C_linalg_fevector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
  }
  namespace ModelEvaluator
  {
    class BeamInteractionDataState;
  }  // namespace ModelEvaluator
}  // namespace Solid
namespace BeamInteraction
{
  class BeamToSolidParamsBase;
  class BeamContactPair;
}  // namespace BeamInteraction

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg


namespace BeamInteraction
{
  /**
   * \brief In beam to solid interactions with mortar contact discretization, we need to create a
   * map with the Lagrange multiplier DOFs (in contrast to solid meshtying / mortar we do not create
   * a own discretization for the interface).
   *
   * The created DOF can be split into two groups:
   *   - Lagrange multiplier DOFs on  nodes that are physical nodes of the system. The do not need
   *     to have the same number of nodal values as the physical node or even the same dimension
   *     (although in most cases the Lagrange multiplier have 3 components for each nodal value).
   *   - Lagrange multiplier DOFs on elements. For example when we have a two noded beam element and
   *     we want a quadratic interpolation of the Lagrange multipliers, we 'give' the element
   *     additional DOFs that represent the values at the middle node.
   *
   * By defining the Lagrange multipliers like described above, each additional DOF can be
   * identified by either the global id of the physical node it is defined on or by the global id of
   * the element it is defined on.
   *
   * The start value for the Lagrange multiplier global IDs can be explicitly given. This is usually
   * the number of solid DOFs + beam DOFs + Lagrange multipliers from other beam-to-solid couplings
   * preceding this mortar manager in the model.
   * The Lagrange multiplier DOFs are then numbered the following way, and used in \ref
   * lambda_dof_rowmap_.
   *   - Lagrange multiplier DOFs on nodes of processor 0
   *   - Lagrange multiplier DOFs on elements of processor 0
   *   - Lagrange multiplier DOFs on nodes of processor 1
   *   - Lagrange multiplier DOFs on elements of processor 1
   *   - ...
   *
   * This class manages the connection between the created nodes and the global node / element DOFs.
   * For the created maps a offset can be chosen, so the new DOFs fit into a global saddle-point
   * system.
   */
  class BeamToSolidMortarManager
  {
   public:
    /**
     * \brief Standard Constructor
     *
     * @param discret (in) Pointer to the discretization.
     * @param params (in) Beam-to-solid parameters.
     * @param start_value_lambda_gid (in) Start value for the Lagrange multiplier global IDs.
     */
    BeamToSolidMortarManager(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<const BeamInteraction::BeamToSolidParamsBase>& params,
        int start_value_lambda_gid);

    /**
     * \brief Virtual Destructor
     */
    virtual ~BeamToSolidMortarManager() = default;

    /**
     * \brief This method builds the global maps for the global node / element IDs to the Lagrange
     * multiplier DOFs.
     *
     * Some nodes / elements in the discretization need additional Lagrange multiplier DOFs. We need
     * to be able to know which pair refers to which Lagrange multipliers. In this setup routine, a
     * multi vector is created, that maps all centerline nodes and beam elements, to a
     * Lagrange multiplier DOF.
     *
     * @param discret Pointer to the discretization.
     */
    void setup();

    /**
     * \brief Calculate the maps for the beam and solid dofs. The calculated maps are used in
     * Complete of the mortar matrices.
     */
    void set_global_maps();

    /**
     * \brief This method builds the local maps from the global multi vector created in Setup. The
     * global mortar matrices are also created.
     *
     * Since some nodes of this pair, that have Lagrange multipliers may not be on this processor,
     * we need to get the node ID to Lagrange multiplier ID form the processor that holds the
     * node. All relevant global node / element to global Lagrange multiplier maps for the given
     * contact pairs are stored in a standard maps in this object. The keys in those maps are the
     * global node / element id and the value is a vector with the corresponding Lagrange multiplier
     * gids. By doing so we only have to communicate between the ranks once per timestep (to be more
     * precise: only once for each set of contact pairs. If they do not change between timesteps and
     * do not switch rank, we can keep the created maps).
     *
     * @param discret Pointer to the discretization.
     * @param contact_pairs All contact pairs on this processor.
     */
    void set_local_maps(
        const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& contact_pairs);

    /**
     * \brief Get vectors with GIDs of the positional and rotational Lagrange multipliers for the
     * contact pair.
     *
     * First entry in the pair are the translational GIDs and second entry are the rotational GIDs.
     */
    [[nodiscard]] std::pair<std::vector<int>, std::vector<int>> location_vector(
        const BeamInteraction::BeamContactPair& contact_pair) const;

    /**
     * \brief Evaluate the force and stiffness contributions resulting from a penalty
     * regularization.
     * @param data_state (in) Beam interaction data state.
     * @param stiff (out) Global stiffness matrix. Mortar terms are added to it.
     * @param force (out) Global force vector. Mortar terms are added to it.
     */
    void evaluate_force_stiff_penalty_regularization(
        const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::FEVector<double>> force);

    /**
     * \brief Get the global vector of Lagrange multipliers.
     * @return Global vector of Lagrange multipliers.
     */
    [[nodiscard]] virtual std::shared_ptr<Core::LinAlg::Vector<double>> get_global_lambda() const;

    /**
     * \brief Get the global vector of Lagrange multipliers, with the maps being the column maps of
     * the Lagrange GID. on the ranks where they are used.
     * @return Global vector of Lagrange multipliers.
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_global_lambda_col() const;

    /**
     * \brief Return a const reference to the contact pairs in this mortar manager.
     * @return Reference to the pair vector.
     */
    const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& get_contact_pairs() const
    {
      return contact_pairs_;
    }

    /**
     * \brief Calculate the internal mortar penalty energy.
     * @return Global mortar energy.
     */
    double get_energy() const;

   protected:
    /**
     * \brief Throw an error if setup was not called on the object prior to this function call.
     */
    inline void check_setup() const
    {
      if (!is_setup_) FOUR_C_THROW("Setup not called on BeamToSolidMortarManager!");
    }

    /**
     * \brief Throw an error if the global maps were not build.
     */
    inline void check_global_maps() const
    {
      if (!is_global_maps_build_)
        FOUR_C_THROW("Global maps are not build in BeamToSolidMortarManager!");
    }

    /**
     * \brief Throw an error if the local maps were not build.
     */
    inline void check_local_maps() const
    {
      if (!is_local_maps_build_)
        FOUR_C_THROW("Local maps are not build in BeamToSolidMortarManager!");
    }

    /**
     * \brief Evaluate mortar coupling contributions on all pairs and assemble them into the global
     * matrices.
     * @param displacement_vector (in) global displacement vector.
     */
    void evaluate_and_assemble_global_coupling_contributions(
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector);

    /**
     * \brief Add the mortar penalty contributions to the global force vector and stiffness matrix.
     * @param data_state (in) Beam interaction data state.
     * @param stiff (out) Global stiffness matrix. Mortar terms are added to it.
     * @param force (out) Global force vector. Mortar terms are added to it.
     */
    virtual void add_global_force_stiffness_penalty_contributions(
        const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::FEVector<double>> force) const;

    /**
     * \brief Get the penalty regularization of the constraint vector
     * @param compute_linearization If the linearization of the regularization should be returned
     * @return The regularized Lagrange multipliers and (optionally) their derivative w.r.t. the
     * constraint vector and the scaling vector
     */
    [[nodiscard]] virtual std::tuple<std::shared_ptr<Core::LinAlg::Vector<double>>,
        std::shared_ptr<Core::LinAlg::Vector<double>>,
        std::shared_ptr<Core::LinAlg::Vector<double>>>
    get_penalty_regularization(const bool compute_linearization = false) const;

    /**
     * \brief invert the scaling vector \ref kappa_ vector.
     *
     * The inversion is performed with accounting for non active Lagrange multipliers. Furthermore,
     * the inverted matrix is scaled with the the corresponding penalty parameter.
     *
     * @return Inverted kappa_ vector.
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> penalty_invert_kappa() const;

   protected:
    //! Flag if setup was called.
    bool is_setup_;

    //! Flag if local maps were build.
    bool is_local_maps_build_;

    //! Flag if global maps were build.
    bool is_global_maps_build_;

    //! The start value for the Lagrange multiplier global IDs.
    int start_value_lambda_gid_;

    //! Number of Lagrange multiplier DOFs on a node.
    unsigned int n_lambda_node_;

    //! Number of Lagrange multiplier DOFs on an element.
    unsigned int n_lambda_element_;

    //! Number of Lagrange multiplier DOFs on a node for translational coupling.
    unsigned int n_lambda_node_translational_;

    //! Number of Lagrange multiplier DOFs on an element for translational coupling.
    unsigned int n_lambda_element_translational_;

    //! Number of Lagrange multiplier DOFs on a node for rotational coupling.
    unsigned int n_lambda_node_rotational_;

    //! Number of Lagrange multiplier DOFs on an element for rotational coupling.
    unsigned int n_lambda_element_rotational_;

    //! Pointer to the discretization containing the solid and beam elements.
    std::shared_ptr<const Core::FE::Discretization> discret_;

    //! Pointer to the beam contact parameters.
    std::shared_ptr<const BeamInteraction::BeamToSolidParamsBase> beam_to_solid_params_;

    //! Row map of the additional Lagrange multiplier DOFs for translations.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_rowmap_translations_;

    //! Row map of the additional Lagrange multiplier DOFs for rotations.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_rowmap_rotations_;

    //! Row map of the additional Lagrange multiplier DOFs.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_rowmap_;

    //! Column map of the additional Lagrange multiplier DOFs.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_colmap_;

    //! Row map of the beam DOFs.
    std::shared_ptr<Core::LinAlg::Map> beam_dof_rowmap_;

    //! Row map of the solid DOFs.
    std::shared_ptr<Core::LinAlg::Map> solid_dof_rowmap_;

    //! Multivector that connects the global node IDs with the Lagrange multiplier DOF IDs.
    //! The global row ID of the multi vector is the global ID of the node that a Lagrange
    //! multiplier is defined on. The columns hold the corresponding global IDs of the Lagrange
    //! multipliers.
    std::shared_ptr<Core::LinAlg::MultiVector<double>> node_gid_to_lambda_gid_;

    //! Multivector that connects the global element IDs with the Lagrange multiplier DOF IDs.
    //! The global row ID of the multi vector is the global ID of the element that a Lagrange
    //! multiplier is defined on. The columns hold the corresponding global IDs of the Lagrange
    //! multipliers.
    std::shared_ptr<Core::LinAlg::MultiVector<double>> element_gid_to_lambda_gid_;

    //! Standard map from global node ids to global Lagrange multiplier ids, for all
    //! nodes used on this rank.
    std::map<int, std::vector<int>> node_gid_to_lambda_gid_map_;

    //! Standard map from global element ids to global Lagrange multiplier ids, for all elements
    //! used on this rank.
    std::map<int, std::vector<int>> element_gid_to_lambda_gid_map_;

    //! Global constraint vector.
    std::shared_ptr<Core::LinAlg::FEVector<double>> constraint_;

    //! Derivative of constraint vector w.r.t the beam DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> constraint_lin_beam_;

    //! Derivative of constraint vector w.r.t the solid DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> constraint_lin_solid_;

    //! Derivative of the beam coupling forces w.r.t the Lagrange multipliers.
    std::shared_ptr<Core::LinAlg::SparseMatrix> force_beam_lin_lambda_;

    //! Derivative of the solid coupling forces w.r.t the Lagrange multipliers.
    std::shared_ptr<Core::LinAlg::SparseMatrix> force_solid_lin_lambda_;

    //! Global \f$\kappa\f$ vector. This vector is used to scale the mortar matrices. See Yang et
    //! al: Two dimensional mortar contact methods for large deformation frictional sliding (eq.
    //! 37). With this scaling correct units and pass patch tests are achieved (in the penalty
    //! case).
    std::shared_ptr<Core::LinAlg::FEVector<double>> kappa_;

    //! Derivative of the scaling vector w.r.t the beam DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> kappa_lin_beam_;

    //! Derivative of the scaling vector w.r.t the solid DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> kappa_lin_solid_;

    //! This vector keeps tack of all Lagrange multipliers that are active. This is needed when the
    //! kappa vector is inverted and some entries are zero, because no active contributions act on
    //! that Lagrange multiplier.
    std::shared_ptr<Core::LinAlg::FEVector<double>> lambda_active_;

    //! Vector with all contact pairs to be evaluated by this mortar manager.
    std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> contact_pairs_;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
