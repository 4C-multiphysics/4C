// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_STR_MODEL_EVALUATOR_HPP
#define FOUR_C_BEAMINTERACTION_STR_MODEL_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"  // base class

FOUR_C_NAMESPACE_OPEN

// forward declaration ...

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::Binstrategy
{
  class BinningStrategy;
}

namespace BeamInteraction
{
  class BeamInteractionParams;

  class BeamCrosslinkerHandler;

  namespace SubmodelEvaluator
  {
    class Generic;
  }
}  // namespace BeamInteraction

namespace Solid
{
  namespace ModelEvaluator
  {
    // forward declaration
    class BeamInteractionDataState;

    class BeamInteraction : public Generic
    {
     public:
      using Map = std::map<enum Inpar::BeamInteraction::SubModelType,
          std::shared_ptr<FourC::BeamInteraction::SubmodelEvaluator::Generic>>;
      using Vector =
          std::vector<std::shared_ptr<FourC::BeamInteraction::SubmodelEvaluator::Generic>>;

      //! constructor
      BeamInteraction();

      void setup() override;

      void post_setup_submodels();

      /// print welcome to biopolymer network simulation
      virtual void logo() const;

      //! @name Derived public Solid::ModelEvaluator::Generic methods
      //! @{
      //! derived

      //! derived
      Inpar::Solid::ModelType type() const override { return Inpar::Solid::model_beaminteraction; }

      //! derived
      bool evaluate_force() override;

      //! derived
      bool evaluate_stiff() override;

      //! derived
      bool evaluate_force_stiff() override;

      //! derived
      void pre_evaluate() override { return; };

      //! derived
      void post_evaluate() override { /* currently unused */ };

      //! derived
      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! derived
      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      //! derived
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! derived
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! [derived]
      void predict(const Inpar::Solid::PredEnum& pred_type) override { return; };

      //! derived
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override;

      //! derived
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override;

      //! derived
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override;

      //! derived
      void update_step_state(const double& timefac_n) override;

      //! derived
      void update_step_element() override;

      //! derived
      void determine_stress_strain() override;

      //! derived
      void determine_energy() override;

      //! derived
      void determine_optional_quantity() override;

      //! derived
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override;

      //! derived
      void runtime_output_step_state() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

      //! derived
      void post_output() override;

      //! derived
      void reset_step_state() override;
      //! @}

     protected:
      //! derived
      void reset(const Core::LinAlg::Vector<double>& x) override;

      //!@name routines for submodel management
      //! @{

     public:
      /// check if the given model type is active.
      bool have_sub_model_type(Inpar::BeamInteraction::SubModelType const& submodeltype) const;

      //! check if lagrange formulation is active
      bool have_lagrange_dofs() const;

     private:
      void partition_problem();

      bool post_partition_problem();

      //! set beaminteraction sub models
      void set_sub_model_types();


      //! build, init and setup submodel evaluator
      void init_and_setup_sub_model_evaluators();

      //! give submodels a certain order in which they are evaluated
      virtual std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Vector> transform_to_vector(
          Solid::ModelEvaluator::BeamInteraction::Map submodel_map,
          std::vector<Inpar::BeamInteraction::SubModelType>& sorted_submodel_types) const;

      //! @}

      //!@name routines that manage to discretizations with distinct parallel distribution
      //! @{

      /// check if interaction discretization needs to be redistributed completely
      bool check_if_beam_discret_redistribution_needs_to_be_done();

      /// update coupling adapter and matrix transformation object with new maps
      void update_coupling_adapter_and_matrix_transformation();

      /// transform force vector from ia_discret_ to discret()
      virtual void transform_force();

      /// transform stiffness matrix from ia_discret_ to discret()
      virtual void transform_stiff();

      /// transform force vector and stiffness matrix from ia_discret_ to discret()
      virtual void transform_force_stiff();

      /// update states based on bindis after its redistribution
      virtual void update_maps();

      //! @}

      //!@name routines that manage binning strategy
      //! @{

      /// change parallel distribution of bindis and ia_discret and assign (beam) eles to bins
      virtual void extend_ghosting();

      /// build ele to bin map
      virtual void build_row_ele_to_bin_map();

      /// print some information about binning
      virtual void print_binning_info_to_screen() const;

      //! @}

     private:
      //! pointer to the problem discretization (cast of base class member)
      std::shared_ptr<Core::FE::Discretization> discret_ptr_;

      //! data container holding all beaminteraction related parameters
      std::shared_ptr<FourC::BeamInteraction::BeamInteractionParams> beaminteraction_params_ptr_;

      //!@name data for submodel management
      //! @{
      /// current active model types for the model evaluator
      std::shared_ptr<std::set<enum Inpar::BeamInteraction::SubModelType>> submodeltypes_;

      std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Map> me_map_ptr_;

      std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Vector> me_vec_ptr_;
      //! @}

      //!@name data for handling two distinct parallel distributed discretizations
      //! @{
      //! myrank
      int myrank_;

      //! coupling adapter to transfer vectors and matrices between discret() and intactids_
      std::shared_ptr<Coupling::Adapter::Coupling> coupsia_;

      //! transform object for structure stiffness matrix
      std::shared_ptr<Coupling::Adapter::MatrixRowTransform> siatransform_;
      //! @}


      //!@name data for beaminteraction with binning strategy
      //! @{
      //! interaction discretization handling all interactions (e.g. crosslinker to beam,
      //! beam to beam, potential ...)
      std::shared_ptr<Core::FE::Discretization> ia_discret_;

      /// map extractor for split of different element types
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> eletypeextractor_;

      //! pointer to the global state data container
      std::shared_ptr<Solid::ModelEvaluator::BeamInteractionDataState> ia_state_ptr_;

      //! force based on ia_discret at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> ia_force_beaminteraction_;

      //! global force based on discret() at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> force_beaminteraction_;

      //! structural stiffness matrix based on discret()
      std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_beaminteraction_;

      //! beam crosslinker handler
      std::shared_ptr<FourC::BeamInteraction::BeamCrosslinkerHandler> beam_crosslinker_handler_;

      //! binning strategy
      std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy_;

      //! crosslinker and bin discretization
      std::shared_ptr<Core::FE::Discretization> bindis_;

      //! elerowmap of bindis
      std::shared_ptr<Core::LinAlg::Map> rowbins_;

      //! displacement of nodes since last redistribution
      std::shared_ptr<Core::LinAlg::Vector<double>> dis_at_last_redistr_;

      //! depending on which submodels are active this variable has different values
      //! and determines how often a redistribution needs to be done
      double half_interaction_distance_;

      //! @}
    };
  }  // namespace ModelEvaluator
}  // namespace Solid


FOUR_C_NAMESPACE_CLOSE

#endif
