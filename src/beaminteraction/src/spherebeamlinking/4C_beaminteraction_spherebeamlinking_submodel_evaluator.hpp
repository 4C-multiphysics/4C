// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_SPHEREBEAMLINKING_SUBMODEL_EVALUATOR_HPP
#define FOUR_C_BEAMINTERACTION_SPHEREBEAMLINKING_SUBMODEL_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_input.hpp"
#include "4C_beaminteraction_submodel_evaluator_generic.hpp"
#include "4C_binstrategy_utils.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN


namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace BeamInteraction
{
  class SphereBeamLinkingParams;
  class BeamLinkPinJointed;

  namespace SubmodelEvaluator
  {
    class SphereBeamLinking : public Generic
    {
     public:
      //! constructor
      SphereBeamLinking();

      //! setup class variables
      void setup() override;

      //! derived
      void post_setup() override;

      //! Returns the type of the current model evaluator
      BeamInteraction::SubModelType type() const override
      {
        return BeamInteraction::SubModelType::submodel_spherebeamlink;
      }

      //! derived
      void reset() override;

      //! derived
      bool evaluate_force() override;

      //! derived
      bool evaluate_stiff() override;

      //! derived
      bool evaluate_force_stiff() override;

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
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override {}

      //! derived
      void runtime_output_step_state() const override;

      //! derived
      void reset_step_state() override;

      //! derived
      void write_restart(Core::IO::DiscretizationWriter& bin_writer) const override;

      //! derived
      void pre_read_restart() override;

      //! derived
      void read_restart(Core::IO::DiscretizationReader& bin_reader) override;

      //! derived
      void post_read_restart() override;

      //! derived
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override { /*empty*/ };

      //! derived
      void init_submodel_dependencies(
          std::shared_ptr<Solid::ModelEvaluator::BeamInteractionModelEvaluator::Map> const
              submodelmap) override;

      //! derived
      void add_bins_to_bin_col_map(std::set<int>& colbins) override;

      //! derived
      void add_bins_with_relevant_content_for_ia_discret_col_map(
          std::set<int>& colbins) const override;

      //! derived
      void get_half_interaction_distance(double& half_interaction_distance) override;

      //! @}

     private:
      //! @name submodel specific, not derived member functions
      //! @{

      //! init output in VTP format
      void init_output_runtime();

      //! writes output for discretization structure in VTP format
      void write_output_runtime() const;

      /// get neighboring eles in discret
      virtual void find_and_store_neighboring_elements(
          std::map<int, std::vector<std::pair<int, int>>>& newlinks);

      /// get neighboring eles in discret
      virtual void check_feasibility_of_new_link(Core::Elements::Element const* currele,
          std::vector<Core::Elements::Element const*> const& neighbors,
          std::unordered_set<int>& tobebonded,
          std::map<int, std::vector<std::pair<int, int>>>& newlinks) const;

      /// create new beam to sphere joint object
      virtual void create_beam_to_sphere_joint(
          std::map<int, std::vector<std::pair<int, int>>> const& newlinks);

      /// update linker length to mimic contractive cells
      virtual void update_linker_length();

      /// check if bond needs to be dissolved
      virtual void unbind_sphere_beam_bonds(int& num_dissolved);

      /// compute force dependent off rate for a catch-slip bond
      virtual void calc_force_dependent_catch_slip_bond_unbind_probability(
          std::shared_ptr<BeamInteraction::BeamLinkPinJointed> linkelepairptr, double& p_unbind);

      //! @}

     private:
      //! cell discretization
      std::shared_ptr<BeamInteraction::SubmodelEvaluator::Crosslinking> sm_crosslinkink_ptr_;

      //! data container holding all beam contact related parameters
      std::shared_ptr<BeamInteraction::SphereBeamLinkingParams> spherebeamlinking_params_ptr_;

      //! runtime output for cell beam crosslinks (integrins)
      std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_ptr_;

      //! step number for random stuff concerning sphere beam linking
      int random_number_sphere_beam_linking_step_;
    };

  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
