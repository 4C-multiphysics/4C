// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTK_OUTPUT_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTK_OUTPUT_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_parameters.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class StructureRuntimeOutputParams;
    class BeamRuntimeOutputParams;
  }  // namespace Elements
}  // namespace Discret


namespace Solid
{
  namespace TimeInt
  {
    /** \brief Input data container for output at runtime for the structural (time) integration
     *
     * */
    class ParamsRuntimeOutput
    {
     public:
      /// destructor
      virtual ~ParamsRuntimeOutput() = default;

      /// initialize the class variables
      void init(const Teuchos::ParameterList& IO_vtk_structure_paramslist);

      /// setup new class variables
      void setup();

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_in_steps() const
      {
        check_init_setup();
        return output_interval_steps_;
      };

      [[nodiscard]] int output_step_offset() const
      {
        check_init_setup();
        return output_step_offset_;
      }

      /// whether to write output in every iteration of the nonlinear solver
      bool output_every_iteration() const
      {
        check_init_setup();
        return output_every_iteration_;
      };

      /// whether to write special output for structure elements
      bool output_structure() const
      {
        check_init_setup();
        return output_structure_;
      };

      /// whether to write special output for structure elements
      bool output_beams() const
      {
        check_init_setup();
        return output_beams_;
      };

      /// whether to write element evaluation times to vtk output
      bool output_element_eval_times() const
      {
        check_init_setup();
        return output_element_eval_times_;
      }

      /// get the data container for parameters regarding beams
      std::shared_ptr<const Discret::Elements::StructureRuntimeOutputParams> get_structure_params()
          const
      {
        check_init_setup();
        return params_runtime_output_structure_;
      };

      /// get the data container for parameters regarding beams
      std::shared_ptr<const Discret::Elements::BeamRuntimeOutputParams> get_beam_params() const
      {
        check_init_setup();
        return params_runtime_output_beams_;
      };


     private:
      /// get the init indicator status
      const bool& is_init() const { return isinit_; };

      /// get the setup indicator status
      const bool& is_setup() const { return issetup_; };

      /// Check if init() and setup() have been called, yet.
      void check_init_setup() const;


     private:
      /// @name variables for internal use only
      /// @{
      ///
      bool isinit_ = false;

      bool issetup_ = false;
      /// @}

      /// @name variables controlling output
      /// @{

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_steps_ = -1;

      /// An offset added to the current step to shift the steps to be written
      int output_step_offset_ = 0;

      /// whether to write output in every iteration of the nonlinear solver
      bool output_every_iteration_ = false;

      /// whether to write output for structural elements
      bool output_structure_ = false;

      /// whether to write special output for beam elements
      bool output_beams_ = false;

      /// whether to write element evaluation times to vtk output
      bool output_element_eval_times_ = false;

      /// data container for input parameters related to output of structure at runtime
      std::shared_ptr<Discret::Elements::StructureRuntimeOutputParams>
          params_runtime_output_structure_ = nullptr;

      /// data container for input parameters related to output of beams at runtime
      std::shared_ptr<Discret::Elements::BeamRuntimeOutputParams> params_runtime_output_beams_ =
          nullptr;

      //@}
    };

  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
