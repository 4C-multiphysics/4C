// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_HPP
#define FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_submodel_evaluator_generic.hpp"
#include "4C_binstrategy_utils.hpp"
#include "4C_io_visualization_manager.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace Core::GeometricSearch
{
  class geometric_search_params;
  class GeometricSearchVisualization;
}  // namespace Core::GeometricSearch
namespace BeamInteraction
{
  class BeamContactParams;
  class BeamContactPair;
  class BeamInteractionConditions;
  class BeamToSolidVolumeMeshtyingVisualizationOutputWriter;
  class BeamToSolidSurfaceVisualizationOutputWriter;
  class BeamToSolidSurfaceVisualizationOutputWriterContact;

  namespace SubmodelEvaluator
  {
    class BeamContactAssemblyManagerInDirect;
    class BeamContactAssemblyManager;

    class BeamContact : public Generic
    {
     public:
      //! constructor
      BeamContact();

      //! setup class variables
      void setup() override;

      //! derived
      void post_setup() override;

      //! Returns the type of the current submodel evaluator
      Inpar::BeamInteraction::SubModelType type() const override
      {
        return Inpar::BeamInteraction::submodel_beamcontact;
      }

      //! @name Derived public BeamInteraction::SubmodelEvaluator::Generic methods
      //! @{
      //! \brief reset submodel specific variables
      //! derived
      void reset() override;

      //! derived
      bool evaluate_force() override;

      //! derived
      bool evaluate_stiff() override;

      //! derived
      bool evaluate_force_stiff() override;

      /**
       * \brief Call pre_evaluate on each pair.
       *
       * This has to be done outside of the the assembly managers, as pre_evaluate should only be
       * called once per pair, and it is technically possible that the same pair is in multiple
       * assembly managers.
       */
      void pre_evaluate();

      //! derived
      void update_step_state(const double& timefac_n) override;

      //! derived
      bool pre_update_step_element(bool beam_redist) override;

      //! derived
      void update_step_element(bool repartition_was_done) override;

      //! derived
      void post_update_step_element() override;

      //! derived
      std::map<Solid::EnergyType, double> get_energy() const override;

      //! derived
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override;

      //! derived
      void runtime_output_step_state() const override;

      //! derived
      void reset_step_state() override;

      //! derived
      void write_restart(Core::IO::DiscretizationWriter& ia_writer,
          Core::IO::DiscretizationWriter& bin_writer) const override;

      //! derived
      void pre_read_restart() override;

      //! derived
      void read_restart(Core::IO::DiscretizationReader& ia_reader,
          Core::IO::DiscretizationReader& bin_reader) override;

      //! derived
      void post_read_restart() override;

      //! derived
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override;

      //! derived
      void init_submodel_dependencies(
          std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Map> const submodelmap) override;

      //! derived
      void add_bins_to_bin_col_map(std::set<int>& colbins) override;

      //! derived
      void add_bins_with_relevant_content_for_ia_discret_col_map(
          std::set<int>& colbins) const override;

      //! derived
      void get_half_interaction_distance(double& half_interaction_distance) override;

      /**
       * \brief Return the assembly managers in this submodel evaluator.
       */
      inline const std::vector<
          std::shared_ptr<BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManager>>&
      get_assembly_managers() const
      {
        return assembly_managers_;
      }

      /**
       * \brief Lagrange Multiplier specific function returning the first assembly manager.
       */
      std::shared_ptr<const BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect>
      get_lagrange_multiplier_assembly_manager() const;

      /**
       * \brief Return the geometry pairs in this submodel evaluator.
       */
      inline const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>&
      get_contact_pairs() const
      {
        return contact_elepairs_;
      }

      /**
       * \brief Return the conditions in this submodel evaluator.
       */
      inline std::shared_ptr<const BeamInteraction::BeamInteractionConditions> get_conditions()
          const
      {
        return beam_interaction_conditions_ptr_;
      }

      /**
       * \brief Return the dof rowmap of the lagrange multipliers.
       */
      std::shared_ptr<const FourC::Core::LinAlg::Map> get_lagrange_map() const override;

      /**
       * \brief Method used to assemble the force vector when using Lagrange Multipliers
       */
      void assemble_force(Core::LinAlg::Vector<double>& f) const override;

      /**
       * \brief Method used to assemble the stiffness matrix when using Lagrange Multipliers
       */
      void assemble_stiff(Core::LinAlg::SparseOperator& jac) const override;

      //! @}

     protected:
      //!@name routines that are not derived and handle beam to beam contact
      //! @{
      /// check which contact is considered
      bool have_contact_type(Core::Binstrategy::Utils::BinContentType const& contacttype) const;

      /// print
      void print_all_beam_contact_element_pairs(std::ostream& out) const;

      /// print
      void print_active_beam_contact_set(std::ostream& out) const;

      //! @}

     public:
      inline BeamInteraction::BeamContactParams const& beam_contact_params() const
      {
        check_init();
        return *beam_contact_params_ptr_;
      }

     private:
      inline BeamInteraction::BeamContactParams& beam_contact_params()
      {
        check_init();
        return *beam_contact_params_ptr_;
      }

     public:
      inline std::shared_ptr<BeamInteraction::BeamContactParams> beam_contact_params_ptr() const
      {
        check_init();
        return beam_contact_params_ptr_;
      }

      //!@name routines that handle visualization output for contact forces
      //! @{

      //! init output for contact forces in VTP format
      void init_output_runtime_beam_contact();

      //! writes visualization output for contact forces at the end of a time step
      void write_time_step_output_runtime_beam_contact() const;

      //! writes visualization output for contact forces at the end of a nonlinear iteration
      void write_iteration_output_runtime_beam_contact(int iteration_number) const;

      //! writes visualization output for contact forces
      void write_output_runtime_beam_contact(int timestep_number, double time) const;


      //! @}

      //!@name routines that are not derived and handle beam to beam contact
      //! @{

      // init element types considered for beam to ? contact
      void init_element_types_considered_for_contact();

      /// get neighbouring eles in discret
      void find_and_store_neighboring_elements();

      /// exclude certain neighbors from interaction evaluation
      /*
       * Here we remove pairs based on very trivial rules, e.g.:
       * - an element can not be in contact with itself, thus is removed if same global indice
       *   is found
       * - evaluate beam contact pairs only once i.e., pair(ele1, ele2) and pair(ele2, ele1),
       *   thus only beam-to-beam pairs where the GID of the first element is smaller than the
       *   second are considered
       * - elements sharing nodes don't interact
       */
      void select_eles_to_be_considered_for_contact_evaluation(
          const Core::Elements::Element* currele,
          std::set<Core::Elements::Element*>& neighbors) const;

      /// create instances of class BeamContactPair that will be evaluated
      ///  to get force and stiffness contributions from beam interactions
      void create_beam_contact_element_pairs();

      /// Add the restart displacement to the pairs, if the coupling should be evaluated with
      /// respect to the restart state.
      void set_restart_displacement_in_pairs();

      //! @}

      //! data container holding all beam contact related parameters
      std::shared_ptr<BeamInteraction::BeamContactParams> beam_contact_params_ptr_;

      //! data container holding all beam interaction related parameters
      std::shared_ptr<BeamInteraction::BeamInteractionParams> beam_interaction_params_ptr_;

      //! data container holding all beam interactions defined by conditions
      std::shared_ptr<BeamInteraction::BeamInteractionConditions> beam_interaction_conditions_ptr_;

      //! data container holding all geometric search related parameters
      std::shared_ptr<Core::GeometricSearch::GeometricSearchParams> geometric_search_params_ptr_;

      //! element types considered for beam to ? contact
      std::vector<Core::Binstrategy::Utils::BinContentType> contactelementtypes_;

      //! interacting pairs of beam elements that might exert forces on each other
      std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> contact_elepairs_;

      //! Objects to evaluate system contributions for stiffness and force terms.
      std::vector<std::shared_ptr<BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManager>>
          assembly_managers_;

      //! mapping beam ele (elegid) to set of spatially proximal eles (pointer to elements)
      std::map<int, std::set<Core::Elements::Element*>> nearby_elements_map_;

      //! runtime visualization writer for visualization of contact forces
      std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_ptr_;

      //! This object handles all beam to solid volume related visualization output.
      std::shared_ptr<BeamInteraction::BeamToSolidVolumeMeshtyingVisualizationOutputWriter>
          beam_to_solid_volume_meshtying_visualization_output_writer_ptr_;

      //! This object handles all beam to solid surface mesh tying related visualization output.
      std::shared_ptr<BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriter>
          beam_to_solid_surface_visualization_output_writer_ptr_;

      //! This object handles all beam to solid surface contact related visualization output.
      std::shared_ptr<BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact>
          beam_to_solid_surface_visualization_output_writer_contact_ptr_;

      //! This object handles all geometric search related visualization output.
      std::shared_ptr<Core::GeometricSearch::GeometricSearchVisualization>
          geometric_search_visualization_ptr_;
    };

  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
