// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"

#include <fstream>

namespace NOX
{
  namespace LineSearch
  {
    class Generic;
  }  // namespace LineSearch
}  // namespace NOX

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class DiscretizationWriter;
}  // namespace Core::IO

namespace Solid
{
  namespace TimeInt
  {
    class ParamsRuntimeOutput;
    class ParamsRuntimeVtpOutput;
    class ParamsMonitorDBC;

    /** \brief Input/output data container for the structural (time) integration
     *
     * This data container holds everything, which refers directly to the
     * input/output writer and the screen output.
     *
     * */
    class BaseDataIO
    {
     public:
      /// constructor
      BaseDataIO();

      /// destructor
      virtual ~BaseDataIO() = default;

      /// initialize the class variables
      void init(const Teuchos::ParameterList& IOParams, const Teuchos::ParameterList& sDynParams,
          const Teuchos::ParameterList& xParams,
          std::shared_ptr<Core::IO::DiscretizationWriter> output);

      /// setup new class variables
      void setup();

     protected:
      /// get the init indicator status
      virtual const bool& is_init() const { return isinit_; };

      /// get the setup indicator status
      virtual const bool& is_setup() const { return issetup_; };

      /// Check if init() and setup() have been called, yet.
      virtual void check_init_setup() const;

     public:
      /// get the binary output writer
      std::shared_ptr<Core::IO::DiscretizationWriter> get_output_ptr()
      {
        check_init_setup();
        return output_;
      };

      /// get the binary output writer
      std::shared_ptr<const Core::IO::DiscretizationWriter> get_output_ptr() const
      {
        check_init_setup();
        return output_;
      }

      /// get the data container for parameters regarding output at runtime
      std::shared_ptr<const ParamsRuntimeOutput> get_runtime_output_params() const
      {
        check_init_setup();
        return params_runtime_vtk_output_;
      };

      /// get the data container for parameters regarding output at runtime
      std::shared_ptr<const ParamsRuntimeVtpOutput> get_runtime_vtp_output_params() const
      {
        check_init_setup();
        return params_runtime_vtp_output_;
      };

      /// get the data container for parameters regarding output at runtime
      std::shared_ptr<const ParamsMonitorDBC> get_monitor_dbc_params() const
      {
        check_init_setup();
        return params_monitor_dbc_;
      };

      /// \brief return TRUE if the results shall be written for this load/time \c step
      bool write_results_for_this_step(const int step) const;

      [[nodiscard]] bool is_write_results_enabled() const;

      /// \brief return TRUE if runtime vtk results shall be written for this load/time \c step
      bool write_runtime_vtk_results_for_this_step(const int step) const;

      [[nodiscard]] bool is_runtime_output_enabled() const;

      /// \brief return TRUE if runtime vtp results shall be written for this load/time \c step
      bool write_runtime_vtp_results_for_this_step(const int step) const;

      /// \brief return TRUE if the restart state shall be written for this load/time step
      bool should_write_restart_for_step(int step) const;

      /// \brief return TRUE if reaction forces shall be written for this load/time step
      bool should_write_reaction_forces_for_this_step(int step) const;

      /// \brief return TRUE if energy data shall be written for this load/time step
      bool should_write_energy_for_this_step(int step) const;

      /// @name Get printing and output parameters/files
      ///@{
      /// get the output file for energy
      std::ostream& get_energy_output_stream()
      {
        check_init_setup();

        FOUR_C_ASSERT(energyfile_, "energy file stream uninitialized");

        return *energyfile_;
      };

      /// Is GMSH output of displacements required?
      const bool& is_gmsh() const
      {
        check_init_setup();
        return gmsh_out_;
      };

      /// Shall we print the logo?
      const bool& is_logo() const
      {
        check_init_setup();
        return printlogo_;
      };

      /// Shall we print intermediate iterations during solution?
      const bool& is_print_intermediate_iterations() const
      {
        check_init_setup();
        return printiter_;
      };

      /// Shall we write surfactant output?
      const bool& is_surfactant_output() const
      {
        check_init_setup();
        return writesurfactant_;
      };

      /// Shall we write the current state?
      const bool& is_write_state() const
      {
        check_init_setup();
        return writestate_;
      };

      /// Shall we write the jacobian to MATLAB?
      bool is_write_jacobian_to_matlab() const
      {
        check_init_setup();
        return writejac2matlab_;
      }

      /// Shall we compute and write the condition number?
      Inpar::Solid::ConditionNumber condition_number_type() const
      {
        check_init_setup();
        return conditionnumbertype_;
      }

      /// Is this the first output of the current run?
      const bool& is_first_output_of_run() const
      {
        check_init_setup();
        return firstoutputofrun_;
      };

      /// Whether to output evaluation times for each rank to a csv file
      const bool& output_per_rank_eval_time() const
      {
        check_init_setup();
        return output_per_rank_eval_time_;
      };

      /// Print infos to standard out every n step
      const int& get_print2_screen_every_n_step() const
      {
        check_init_setup();
        return printscreen_;
      };

      /// Get the output counter for OutputEveryIter
      const int& get_oei_output_counter() const
      {
        check_init_setup();
        return outputcounter_;
      };

      /// returns the offset added to the current step to shift the steps to be written
      int get_write_timestep_offset() const
      {
        check_init_setup();
        return writetimestepoffset_;
      }

      /// write restart every given step. if 0, restart is not written
      const int& get_write_restart_every_n_step() const
      {
        check_init_setup();
        return writerestartevery_;
      };

      /// write state/stress/strain every given step
      const int& get_write_results_every_n_step() const
      {
        check_init_setup();
        return writeresultsevery_;
      };

      /// write system energy every given step
      const int& get_write_energy_every_n_step() const
      {
        check_init_setup();
        return writeenergyevery_;
      }

      /// get stress output type
      const Inpar::Solid::StressType& get_stress_output_type() const
      {
        check_init_setup();
        return writestress_;
      }

      /// get strain output type
      const Inpar::Solid::StrainType& get_strain_output_type() const
      {
        check_init_setup();
        return writestrain_;
      }

      /// get plastic strain output type
      const Inpar::Solid::StrainType& get_plastic_strain_output_type() const
      {
        check_init_setup();
        return writeplstrain_;
      }
      ///@}

      /// set the flag indicator firstoutputofrun_
      void set_first_output_of_run(const bool& firstoutputofrun)
      {
        check_init_setup();
        firstoutputofrun_ = firstoutputofrun;
      }

      /// initialize the output of system energy
      void setup_energy_output_file();

     protected:
      /// @name variables for internal use only
      ///@{
      ///
      bool isinit_;

      bool issetup_;
      ///@}

     private:
      /// @name Printing and output
      ///@{

      /// binary output
      std::shared_ptr<Core::IO::DiscretizationWriter> output_;

      /// data container for input parameters related to VTK output at runtime
      std::shared_ptr<ParamsRuntimeOutput> params_runtime_vtk_output_;

      /// data container for input parameters related to VTP output at runtime
      std::shared_ptr<ParamsRuntimeVtpOutput> params_runtime_vtp_output_;

      /// data container for input parameters related to monitoring of reaction forces
      std::shared_ptr<ParamsMonitorDBC> params_monitor_dbc_;
      /// outputfile for energy
      std::shared_ptr<std::ofstream> energyfile_;

      /// Is GMSH output of displacements required?
      bool gmsh_out_;

      /// print the logo (or not)?
      bool printlogo_;

      /// print intermediate iterations during solution
      bool printiter_;

      /// write surfactant output
      bool writesurfactant_;

      /// write state on/off
      bool writestate_;

      /// write jacobian to MATLAB
      bool writejac2matlab_;

      /// flag whether this output step is the first one (restarted or not)
      bool firstoutputofrun_;

      /// output of evaluation times for each rank to a csv file
      bool output_per_rank_eval_time_;

      /// print infos to standard out every n steps
      int printscreen_;

      /// output counter for OutputEveryIter
      int outputcounter_;

      /// offset added on the current step to determine if output/restart should be written
      int writetimestepoffset_ = 0;

      /// write restart every given step. if 0, restart is not written
      int writerestartevery_;

      /// write state/stress/strain every given step
      int writeresultsevery_;

      /// write system energy every given step
      int writeenergyevery_;

      /// stress output type
      Inpar::Solid::StressType writestress_;

      /// strain output type
      Inpar::Solid::StrainType writestrain_;

      /// plastic strain output type
      Inpar::Solid::StrainType writeplstrain_;

      Inpar::Solid::ConditionNumber conditionnumbertype_;

      ///@}
    };  // class BaseDataIO
  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
