// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CROSSLINKING_SUBMODEL_EVALUATOR_HPP
#define FOUR_C_BEAMINTERACTION_CROSSLINKING_SUBMODEL_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_input.hpp"
#include "4C_beaminteraction_submodel_evaluator_generic.hpp"
#include "4C_binstrategy_utils.hpp"
#include "4C_comm_exporter.hpp"
#include "4C_linalg_fixedsizematrix.hpp"


FOUR_C_NAMESPACE_OPEN


// forward declaration
namespace Core::Communication
{
  class PackBuffer;
}

namespace Core::IO
{
  class DiscretizationVisualizationWriterNodes;
}

namespace Mat
{
  class CrosslinkerMat;
}
namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace CrossLinking
{
  class CrosslinkerNode;
}
namespace BeamInteraction
{
  namespace Data
  {
    struct CrosslinkerData;
    struct BeamData;
    struct BindEventData;
    struct UnBindEventData;
    struct BspotLinkerData;
  }  // namespace Data
  class CrosslinkingParams;
  class BeamLink;

  /// type of linker
  enum JointType
  {
    beam3r_line2_rigid,  ///< rigid joint
    beam3r_line2_pin,    ///< pin joint
    truss                ///< truss
  };

  /// type of filament
  enum FilamentType
  {
    filetype_arbitrary,  ///< no special type
    filetype_actin,      ///< actin type
    filetype_collagen,   ///< collagen type
    filetype_none        ///< filament which does not bind
  };

  /// type of crosslinker
  enum CrosslinkerType
  {
    linkertype_arbitrary,  ///< binds to all filament
    linkertype_actin,      ///< only binds to actin filaments
    linkertype_collagen,   ///< only binds to collagen filaments
    linkertype_integrin    ///< sphere to beam linker
  };

  //! Map type std::string to enum
  inline JointType string_to_joint_type(const std::string& name)
  {
    JointType type = beam3r_line2_rigid;
    if (name == "beam3rline2rigid")
      type = beam3r_line2_rigid;
    else if (name == "beam3rline2pin")
      type = beam3r_line2_pin;
    else if (name == "truss")
      type = truss;
    else
      FOUR_C_THROW("invalid filament type std::string ");

    return type;
  };

  //! Map type std::string to enum
  inline FilamentType string_to_filament_type(const std::string& name)
  {
    FilamentType type = filetype_arbitrary;
    if (name == "arbitrary")
      type = filetype_arbitrary;
    else if (name == "actin")
      type = filetype_actin;
    else if (name == "collagen")
      type = filetype_collagen;
    else if (name == "none")
      type = filetype_none;
    else
      FOUR_C_THROW("invalid filament type std::string ");

    return type;
  };

  //! Map type std::string to enum
  inline CrosslinkerType string_to_crosslinker_type(const std::string& name)
  {
    CrosslinkerType type = linkertype_arbitrary;
    if (name == "arbitrary")
      type = linkertype_arbitrary;
    else if (name == "actin")
      type = linkertype_actin;
    else if (name == "collagen")
      type = linkertype_collagen;
    else if (name == "integrin")
      type = linkertype_integrin;
    else
    {
      FOUR_C_THROW(
          "invalid crosslinker type {}. Possible values are arbitrary, actin, collagen and "
          "integrin.",
          name.c_str());
    }

    return type;
  };

  //! Map action type enum to std::string
  static inline std::string crosslinker_type_to_string(const CrosslinkerType type)
  {
    switch (type)
    {
      case linkertype_arbitrary:
        return "arbitrary";
      case linkertype_actin:
        return "actin";
      case linkertype_collagen:
        return "collagen";
      case linkertype_integrin:
        return "integrin";
      default:
        return "unknown";
    }
    return "";
  };

  namespace SubmodelEvaluator
  {
    class Crosslinking : public Generic
    {
     public:
      //! constructor
      Crosslinking();

      //! setup class variables
      void setup() override;

      //! derive
      bool post_partition_problem() override;

      //! derive
      void post_setup() override;

      //! Returns the type of the current model evaluator
      BeamInteraction::SubModelType type() const override
      {
        return BeamInteraction::SubModelType::submodel_crosslinking;
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

      //!@name routines that are not derived and handle crosslinking
      //! @{

      /// unbind all crosslinker residing in bingid and its neighborhood
      void unbind_crosslinker_in_bins_and_neighborhood(
          std::set<int> const& bingids, bool doubleunbind);

      /// unbind all crosslinker residing in bingid and its neighborhood
      void unbind_crosslinker_in_bins_and_neighborhood(std::set<int> const& bingids);

      /// determine which proc is responsible for forced crosslinker unbinding in certain bins
      /// by checking bin ownership
      void determine_responsilbe_procs_for_forced_crosslinker_unbinding(
          std::set<int> const& bingids, std::set<int>& binsonmyrank) const;

      void communicate_bin_ids(
          std::map<int, std::vector<int>> const& binstosend, std::set<int>& binsonmyrank) const;

      /// set all double bonds in bind and neighborhood
      void double_bind_crosslinker_in_bins_and_neighborhood(std::set<int> const& bingids);

      //! @}

     private:
      //! writes output for discretization structure in VTP format
      void write_output_runtime_structure() const;

      //! init output for discretization structure in VTP format
      void init_output_runtime_structure();

      //! @name Small data structs for faster access and better structure
      //! @{

      //! struct that stores all necessary data to handle the crosslinking between two elements on
      //! each proc
      struct NewDoubleBonds
      {
        int id;  // gid of crosslinker
        std::vector<std::pair<int, int>>
            eleids;  // elegid and local binding spot number of first element
        std::vector<Core::LinAlg::Matrix<3, 1>> bspotposs;
        std::vector<Core::LinAlg::Matrix<3, 3>> bspottriads;
      };

      //! @}


      //!@name routines that are not derived and handle crosslinking
      //! @{

      /// add crosslinker to bin discretization initially
      void add_crosslinker_to_bin_discretization();

      /// set filament types on elements
      void set_filament_types();

      /// set double bonded linker between all binding spots that match certain
      /// neigbhoring criteria
      void set_all_possible_initial_double_bonded_crosslinker(
          std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
          std::map<int, NewDoubleBonds>& mynewdbondcl);

      /// get all possible links between beam binding spots
      void get_all_possible_bspot_links(
          std::vector<BeamInteraction::Data::BspotLinkerData>& my_bspot_linker);

      /// communicate initial linker
      void communicate_initial_linker(
          std::vector<BeamInteraction::Data::BspotLinkerData> const& my_bspot_linker,
          std::map<int, std::vector<BeamInteraction::Data::BspotLinkerData>>& global_bspot_linker);

      /// all procs decide in the same way which bonds are valid
      void unambiguous_decisions_on_all_procs(
          std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
          std::map<int, std::vector<BeamInteraction::Data::BspotLinkerData>> const&
              global_bspot_linker,
          std::vector<int>& newlinkermatid);

      /// setup my initial double bonded linker
      void setup_my_initial_double_bonded_linker(
          std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>>& newlinker,
          std::map<int, NewDoubleBonds>& mynewdbondcl, std::vector<int> const& newlinkermatid);

      /// diffuse crosslinker depending on number of bonds they have
      void diffuse_crosslinker();

      /// diffuse unbound crosslinker according to brownian dynamics
      void diffuse_unbound_crosslinker(
          Core::Nodes::Node* crosslinker, BeamInteraction::Data::CrosslinkerData* cldata_i);

      /// get binding spot of crosslinker that is currently occupied
      int get_single_occupied_cl_bspot(std::vector<std::pair<int, int>> const& clbspots) const;

      void set_position_of_double_bonded_crosslinker_pb_cconsistent(
          Core::LinAlg::Matrix<3, 1>& clpos, Core::LinAlg::Matrix<3, 1> const& bspot1pos,
          Core::LinAlg::Matrix<3, 1> const& bspot2pos) const;

      /// new position after transition from single to not bonded
      void set_position_of_newly_free_crosslinker(CrossLinking::CrosslinkerNode* crosslinker,
          BeamInteraction::Data::CrosslinkerData* cldata);

      /// new position after transition from double to single bonded
      void set_position_of_newly_single_bonded_crosslinker(
          CrossLinking::CrosslinkerNode* crosslinker,
          BeamInteraction::Data::CrosslinkerData* cldata, int stayoccpotid);

      /// fill epetar vectors to write vtp output
      void fill_state_data_vectors_for_output(Core::LinAlg::Vector<double>& displacement,
          Core::LinAlg::Vector<double>& orientation, Core::LinAlg::Vector<double>& numberofbonds,
          Core::LinAlg::Vector<double>& owner, Core::LinAlg::Vector<double>& force) const;

      /// update maps
      void store_maps_prior_redistribution();

      /// get crosslink data before interaction evaluation
      void update_and_export_crosslinker_data();

      /// get beam data before interaction evaluation
      void update_and_export_beam_data(bool update_states = true);

      /// bind and unbind crosslinker
      void bind_and_unbind_crosslinker();

      /// bind crossslinker
      int bind_crosslinker();

      /// search and set crosslinker
      /*! -------------------------------------------------------------------------
       note: only the owner of a beam element is allowed to change the status of
       of a binding spot. Therefore we utilize the one layer ghosting around bins
       containing a crosslinker and the ghosting around bins that are touched
       by a row element (this can lead to two layer ghosting) of a proc. Thus we
       exclude the binding of two crosslinker on different procs on the same
       binding spot without loosing any potential interaction.
       To ensure that no crosslinker is bonded to often but still totally random over
       all procs, each binding event of a col crosslinker to a row element needs to
       be communicated to the crosslinker owner, he randomly decides who is allowed
       to bind, sets the according stuff for the cl and  informs back the
       requesting procs so they can set the stuff for the elements.
       As no proc on its own can decide whether a crosslink should be set, two
       binding events for one crosslinker in one time step are excluded (for this
       the proc must be sure that a crosslink is set as the binding range is
       different for a single bonded crosslinker compared to a free one)
       -------------------------------------------------------------------------*/
      void find_potential_binding_events(
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              undecidedbonds);

      /// find potential binding events in one bin
      void find_potential_binding_events_in_bin_and_neighborhood(Core::Elements::Element* bin,
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              undecidedbonds,
          std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
          bool checklinkingprop);

      /// check if sphere should prohibit binding if double bond would be to close
      bool check_if_sphere_prohibits_binding(
          std::set<Core::Elements::Element*> const& neighboring_col_spheres,
          Core::Nodes::Node* node_i) const;

      /// search for binding events on each proc separately (i.e. pretending myrank is alone)
      /// communication to ensure correct binding over all procs is done afterwards
      void prepare_binding(Core::Nodes::Node* node_i,
          std::set<Core::Elements::Element*> const& neighboring_beams,
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              undecidedbonds,
          std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
          bool checklinkingprop);

      /// check criteria if binding event is feasible
      bool check_bind_event_criteria(CrossLinking::CrosslinkerNode const* const crosslinker_i,
          Core::Elements::Element const* const potbeampartner,
          BeamInteraction::Data::CrosslinkerData* cldata_i,
          BeamInteraction::Data::BeamData const* beamdata_i, int locnbspot,
          std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
          bool checklinkingprop) const;

      // check if identical bond already exists
      bool return_false_if_identical_bond_already_exists(
          CrossLinking::CrosslinkerNode const* const crosslinker_i,
          BeamInteraction::Data::CrosslinkerData* cldata_i,
          std::map<int, std::vector<std::map<int, std::set<int>>>>& intendedbeambonds,
          BeamInteraction::Data::BeamData const* beamdata_i, int locnbspot,
          int potbeampartnerrowlid) const;

      /// check if crosslinke and filament type are compatible
      bool check_linker_and_filament_type_compatibility(BeamInteraction::CrosslinkerType linkertype,
          BeamInteraction::FilamentType filamenttype) const;

      /// if crosslinker is singly bound, we fetch the orientation vector of the
      /// filament axis at the already occupied binding spot for the orientation
      /// criterion (enclosed angle) to be checked later on
      void get_occupied_cl_b_spot_beam_tangent(
          CrossLinking::CrosslinkerNode const* const crosslinker_i,
          BeamInteraction::Data::CrosslinkerData* cldata_i,
          Core::LinAlg::Matrix<3, 1>& occ_bindingspot_beam_tangent, int clgid) const;

      /// decide by asking other procs who is allowed to set specific crosslinker,
      /// this is necessary to avoid setting crosslinker more than once per time step
      void manage_binding_in_parallel(
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>&
              mybonds,  // clgid to cldata
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              undecidedbonds,  // owner of cldatas in vector to be requested
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds) const;

      /// communicate requests
      void communicate_undecided_bonds(
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              undecidedbonds,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              requestedcl) const;

      /*
       now myrank needs to decide which proc is allowed to set the requested
       link, add it to its own list as row owner of cl sets stuff for cls, send
       back the answers to the row ele owner and receive the decisions made for
       its own requests:
       - if only one proc is requesting, the link can be set
       - if two procs are requesting or the current proc wants to set a link with
         a requested crosslinker, a random decision who is allowed to set the link
         has to be made
       -------------------------------------------------------------------------*/
      void decide_binding_in_parallel(
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              requestedcl,
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              decidedbonds) const;

      /// communicate decisions for binding events
      void communicate_decided_bonds(
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::BindEventData>>>&
              decidedbonds,
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds) const;

      /* now have two distinct maps of binding events on each proc, depending
         on ownership of crosslinker and elements myrank has different tasks:
          - mybonds: myrank takes care of crosslinker and (most) elements
          - myelebonds: myrank takes care of elements
         within those maps, different treatment is necessary for free and single
         bonded linker
                                                                              */
      int update_my_crosslinker_and_element_binding_states(
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& mybonds,
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>>& myelebonds);

      /// bind row linker of myrank
      void update_my_crosslinker_binding_states(
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> const& mybonds,
          std::map<int, NewDoubleBonds>& mynewdbondcl);

      /// bind row elements of myrank
      void update_my_element_binding_states(
          std::map<int, std::shared_ptr<BeamInteraction::Data::BindEventData>> const& myelebonds);

      /// setup new double bonds
      void create_new_double_bonded_crosslinker_element_pairs(
          const std::map<int, NewDoubleBonds>& mynewdbondcl);

      /// unbind crosslinker if criteria are met
      int un_bind_crosslinker();

      /// calculate force dependent unbind probability for double bonded crosslinker
      /// according to Bell's equation (Howard, eq 5.10, p.89)
      void calc_bells_force_dependent_unbind_probability(CrossLinking::CrosslinkerNode* linker,
          BeamInteraction::BeamLink& elepairptr, std::vector<double>& punlinkforcedependent) const;

      /// communicate crosslinker unbinding event data
      void communicate_crosslinker_unbinding(
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>&
              sendunbindevent,
          std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>& myrankunbindevent)
          const;

      /// update binding status of beams after unbinding
      void update_beam_binding_status_after_unbinding(
          std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>> const& unbindevent);

      /// -------------------------------------------------------------------------
      /// in case we have double bonded crosslinker on myrank we have to check if
      /// myrank is still owner of all its crosslinker (if not, set up double bond on
      /// other proc that is now responsible)
      /// -------------------------------------------------------------------------
      void update_my_double_bonds_after_redistribution();

      /// in case char vector containing double bonds is read by proc != proc
      /// that has written
      void update_my_double_bonds_remote_id_list();

      /// dissolve certain bonds
      void dissolve_bond(Core::Nodes::Node* linker, int freedbspotid, int numbondsold,
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>>&
              sendunbindevents,
          std::vector<std::shared_ptr<BeamInteraction::Data::UnBindEventData>>& myrankunbindevents);

      /// send double bonds to new owner if crosslinker ownership change
      /// in the course of redistribution
      void communicate_beam_link_after_redistribution(
          std::map<int, std::vector<std::shared_ptr<BeamInteraction::BeamLink>>>& dbondcltosend);

      /// send data T to rank= mapkey
      template <typename T>
      void i_send(Core::Communication::Exporter& exporter, std::vector<MPI_Request>& request,
          std::map<int, std::vector<std::shared_ptr<T>>> const& send) const;

      /// get number of request for each proc
      template <typename T>
      void prepare_receiving_procs(
          std::map<int, std::vector<std::shared_ptr<T>>> const& datasenttorank,
          std::vector<int>& summedtargets) const;

      /// receive "receivesize" number of T and store in vector recv
      template <typename T>
      void recv_any(Core::Communication::Exporter& exporter, int receivesize,
          std::vector<std::shared_ptr<T>>& recv) const;

      /// unblocking send and blocking recvany
      template <typename T>
      void i_send_recv_any(std::map<int, std::vector<std::shared_ptr<T>>> const& send,
          std::vector<std::shared_ptr<T>>& recv) const;

      // wait for all communication to finish
      void wait(Core::Communication::Exporter& exporter, std::vector<MPI_Request>& request,
          int length) const;

      /// debug feature to check bindevent structs
      void print_and_check_bind_event_data(
          BeamInteraction::Data::BindEventData& bindeventdata) const;

      //! @}

     private:
      //!@name crosslinking member variables
      //! @{

      //! data container holding all beam contact related parameters
      std::shared_ptr<BeamInteraction::CrosslinkingParams> crosslinking_params_ptr_;

      //! temporary storage for all relevant crosslinker data
      //! (vector key is col lid of crosslinker)
      std::vector<std::shared_ptr<BeamInteraction::Data::CrosslinkerData>> crosslinker_data_;

      //! crosslinker exporter for crosslinker data container
      std::shared_ptr<Core::Communication::Exporter> cl_exporter_;

      //! beam exporter for beam data container
      std::shared_ptr<Core::Communication::Exporter> beam_exporter_;

      //! temporary storage for all relevant beam data during crosslinking
      //  (vector index is col lid of beamele)
      std::vector<std::shared_ptr<BeamInteraction::Data::BeamData>> beam_data_;

      //! double bonded crosslinker that exert forces on network (map key is crosslinker gid)
      std::map<int, std::shared_ptr<BeamInteraction::BeamLink>> doublebondcl_;

      //! linker, i.e. crosslinker molecule discretization runtime vtp writer
      std::shared_ptr<Core::IO::DiscretizationVisualizationWriterNodes>
          visualization_output_writer_ptr_;

      //! current linker displacement
      std::shared_ptr<Core::LinAlg::Vector<double>> linker_disnp_;

      //! summarized displacement of nodes since last redistribution
      std::shared_ptr<Core::LinAlg::Vector<double>> dis_at_last_redistr_;

      //! half interaction distance considering largest linker + tolerance
      double half_interaction_distance_;

      //! store node row map before current redistribution
      std::shared_ptr<Core::LinAlg::Map> cl_noderowmap_prior_redistr_;

      //! store node row map before current redistribution
      std::shared_ptr<Core::LinAlg::Map> cl_nodecolmap_prior_redistr_;

      //! store node row map before current redistribution
      std::shared_ptr<Core::LinAlg::Map> beam_elerowmap_prior_redistr_;

      //! store node row map before current redistribution
      std::shared_ptr<Core::LinAlg::Map> beam_elecolmap_prior_redistr_;

      //! @}
    };

  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
