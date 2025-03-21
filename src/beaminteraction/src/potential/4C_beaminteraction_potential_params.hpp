// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_POTENTIAL_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_POTENTIAL_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_potential_input.hpp"

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace BeamInteraction
{
  class BeamToBeamPotentialRuntimeOutputParams;

  /*!
   *  */
  class BeamPotentialParams
  {
   public:
    //! constructor
    BeamPotentialParams();

    //! destructor
    virtual ~BeamPotentialParams() = default;

    //! initialize with the stuff coming from input file
    void init(double restart_time);

    //! setup member variables
    void setup();

    //! returns the isinit_ flag
    inline bool is_init() const { return isinit_; }

    //! returns the issetup_ flag
    inline bool is_setup() const { return issetup_; }

    //! asserts the init and setup status
    void throw_error_if_not_init_and_setup() const;

    //! asserts the init status
    void throw_error_if_not_init() const;

    inline std::vector<double> const& potential_law_exponents() const
    {
      throw_error_if_not_init_and_setup();
      return *pot_law_exponents_;
    }

    inline std::vector<double> const& potential_law_prefactors() const
    {
      throw_error_if_not_init_and_setup();
      return *pot_law_prefactors_;
    }

    inline enum BeamPotential::Type potential_type() const
    {
      throw_error_if_not_init_and_setup();
      return potential_type_;
    }

    inline enum BeamPotential::Strategy strategy() const
    {
      throw_error_if_not_init_and_setup();
      return strategy_;
    }

    inline double cutoff_radius() const
    {
      throw_error_if_not_init_and_setup();
      return cutoff_radius_;
    }

    inline enum BeamPotential::RegularizationType regularization_type() const
    {
      throw_error_if_not_init_and_setup();
      return regularization_type_;
    }

    inline double regularization_separation() const
    {
      throw_error_if_not_init_and_setup();
      return regularization_separation_;
    }

    inline int number_integration_segments() const
    {
      throw_error_if_not_init_and_setup();
      return num_integration_segments_;
    }

    inline int number_gauss_points() const
    {
      throw_error_if_not_init_and_setup();
      return num_gp_s_;
    }

    inline bool use_fad() const
    {
      throw_error_if_not_init_and_setup();
      return use_fad_;
    }

    inline enum BeamPotential::MasterSlaveChoice choice_master_slave() const
    {
      throw_error_if_not_init_and_setup();
      return choice_master_slave_;
    }

    //! whether to write visualization output for beam contact
    inline bool runtime_output() const
    {
      throw_error_if_not_init_and_setup();
      return visualization_output_;
    }

    //! get the data container for parameters regarding visualization output
    inline std::shared_ptr<const BeamInteraction::BeamToBeamPotentialRuntimeOutputParams>
    get_beam_potential_visualization_output_params() const
    {
      throw_error_if_not_init_and_setup();
      return params_runtime_visualization_output_btb_potential_;
    }

    inline double potential_reduction_length() const
    {
      throw_error_if_not_init_and_setup();
      return potential_reduction_length_;
    }

    //! data container for prior element lengths for potential reduction strategy
    //! first entry is left prior length and second entry is right prior length
    std::unordered_map<int, std::pair<double, double>> ele_gid_prior_length_map_ = {};

   private:
    bool isinit_;

    bool issetup_;

    //! exponents of the summands of a potential law in form of a power law
    // Todo maybe change to integer?
    std::shared_ptr<std::vector<double>> pot_law_exponents_;

    //! prefactors of the summands of a potential law in form of a power law
    std::shared_ptr<std::vector<double>> pot_law_prefactors_;

    //! type of applied potential (volume, surface)
    enum BeamPotential::Type potential_type_;

    //! strategy to evaluate interaction potential
    enum BeamPotential::Strategy strategy_;

    //! neglect all contributions at separation larger than this cutoff radius
    double cutoff_radius_;

    //! type of regularization to use for force law at separations below specified separation
    enum BeamPotential::RegularizationType regularization_type_;

    //! use specified regularization type for separations smaller than this value
    double regularization_separation_;

    //! number of integration segments to be used per beam element
    int num_integration_segments_;

    //! number of Gauss points to be used per integration segment
    int num_gp_s_;

    //! use automatic differentiation via FAD
    bool use_fad_;

    //! rule how to assign the role of master and slave to beam elements (if applicable)
    enum BeamPotential::MasterSlaveChoice choice_master_slave_;

    //! whether to write visualization output at runtime
    bool visualization_output_;

    //! data container for input parameters related to visualization output of beam contact at
    //! runtime
    std::shared_ptr<BeamInteraction::BeamToBeamPotentialRuntimeOutputParams>
        params_runtime_visualization_output_btb_potential_;

    //! within this length starting from the master beam end point the potential is smoothly
    //! reduced to zero to account for infinitely long master beam surrogates and enable an
    //! axial pull-off force.
    double potential_reduction_length_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
