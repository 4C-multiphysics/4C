// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_ELEMENTS_PARAMSINTERFACE_HPP
#define FOUR_C_STRUCTURE_NEW_ELEMENTS_PARAMSINTERFACE_HPP

#include "4C_config.hpp"

#include "4C_browniandyn_input.hpp"                     // enums
#include "4C_fem_general_elements_paramsinterface.hpp"  // base class
#include "4C_inpar_structure.hpp"                       // enums
#include "4C_linalg_multi_vector.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"  // enums
#include "4C_structure_new_enum_lists.hpp"

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo::MeshFree
{
  class BoundingBox;
}

// forward declaration
namespace BrownianDynamics
{
  class ParamsInterface;
}
namespace Solid
{
  namespace ModelEvaluator
  {
    class GaussPointDataOutputManager;
  }
  namespace Elements
  {
    class BeamParamsInterface;

    //! evaluation error flags
    enum EvalErrorFlag : int
    {
      ele_error_none = 0,                          //!< no error occurred (default)
      ele_error_negative_det_of_def_gradient = 1,  //!< negative determinant of deformation gradient
      ele_error_determinant_at_corner = 2,         /*!< invalid/negative jac determinant at the
                                                        element corner nodes */
      ele_error_material_failed = 3,               //!< material evaluation failed
      ele_error_determinant_analysis = 4 /*!< this flag is used to get an idea when the det
                                              analysis found an invalid element */
    };

    //! Map evaluation error flag to a std::string
    static inline std::string eval_error_flag_to_string(const enum EvalErrorFlag& errorflag)
    {
      switch (errorflag)
      {
        case ele_error_none:
          return "ele_error_none";
        case ele_error_negative_det_of_def_gradient:
          return "ele_error_negative_det_of_def_gradient";
        case ele_error_determinant_at_corner:
          return "ele_error_determinant_at_corner";
        case ele_error_material_failed:
          return "ele_error_material_failed";
        case ele_error_determinant_analysis:
          return "ele_error_determinant_analysis";
        default:
          return "unknown";
          break;
      }
      return "";
    };  // EvalErrorFlag2String

    /*! \brief Parameter interface for the structural elements and the Solid::Integrator data
     * exchange
     *
     *  This class is a special case of the Core::Elements::ParamsInterface class and gives you
     * all the basic function definitions which you can use to get access to the Solid::Integrator
     * and many more objects. Please consider to derive a special interface class, if you need
     * special parameters inside of your element. Keep the Evaluate call untouched and cast the
     * interface object to the desired specification.
     *
     *  ToDo Currently we set the interface in the elements via the Teuchos::ParameterList.
     *  Theoretically, the Teuchos::ParameterList can be replaced by the interface itself!
     *

     *  */
    class ParamsInterface : public Core::Elements::ParamsInterface
    {
     public:
      //! return the damping type
      virtual enum Inpar::Solid::DampKind get_damping_type() const = 0;

      //! return the predictor type
      virtual enum Inpar::Solid::PredEnum get_predictor_type() const = 0;

      /// Shall errors during the element evaluation be tolerated?
      virtual bool is_tolerate_errors() const = 0;

      //! @name General time integration parameters
      //! @{
      virtual double get_tim_int_factor_disp() const = 0;

      virtual double get_tim_int_factor_vel() const = 0;
      //! @}


      //! @name Model specific interfaces
      //! @{
      virtual std::shared_ptr<BrownianDynamics::ParamsInterface> get_brownian_dyn_param_interface()
          const = 0;

      //! get pointer to special parameter interface for beam elements
      virtual std::shared_ptr<BeamParamsInterface> get_beam_params_interface_ptr() const = 0;
      //! @}

      //! @name Access control parameters for the handling of element internal variables (e.g.
      //! EAS)
      //! @{

      //! get the current step length
      virtual double get_step_length() const = 0;

      //! Is the current step a default step, or e.g. a line search step?
      virtual bool is_default_step() const = 0;
      //! @}

      //! @name Accessors
      //! @{

      //! get the evaluation error flag
      virtual Solid::Elements::EvalErrorFlag get_ele_eval_error_flag() const = 0;

      //! @}

      //! @name Set functions
      //! @{

      /*! \brief set evaluation error flag
       *
       *  See the EvalErrorFlag enumerators for more information. */
      virtual void set_ele_eval_error_flag(const enum EvalErrorFlag& error_flag) = 0;
      //! @}

      //! @name output related functions
      //! @{
      virtual std::shared_ptr<std::vector<char>>& stress_data_ptr() = 0;

      virtual std::shared_ptr<std::vector<char>>& strain_data_ptr() = 0;

      virtual std::shared_ptr<std::vector<char>>& plastic_strain_data_ptr() = 0;

      virtual std::shared_ptr<std::vector<char>> opt_quantity_data_ptr() = 0;

      //! get the current stress type
      virtual enum Inpar::Solid::StressType get_stress_output_type() const = 0;

      //! get the current strain type
      virtual enum Inpar::Solid::StrainType get_strain_output_type() const = 0;

      //! get the current plastic strain type
      virtual enum Inpar::Solid::StrainType get_plastic_strain_output_type() const = 0;

      virtual std::shared_ptr<ModelEvaluator::GaussPointDataOutputManager>&
      gauss_point_data_output_manager_ptr() = 0;

      //! add contribution to energy of specified type
      virtual void add_contribution_to_energy_type(double value, enum Solid::EnergyType type) = 0;

      //! add the current partial update norm of the given quantity
      virtual void sum_into_my_update_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype,
          const int& numentries, const double* my_update_values, const double* my_new_sol_values,
          const double& step_length, const int& owner) = 0;

      /*! collects and calculates the solution norm of the previous accepted Newton
       *  step on the current proc */
      virtual void sum_into_my_previous_sol_norm(
          const enum NOX::Nln::StatusTest::QuantityType& qtype, const int& numentries,
          const double* my_old_values, const int& owner) = 0;
      //! @}
    };  // class ParamsInterface


    /*! \brief Parameter interface for the data exchange between beam elements and the
     * Solid::Integrator */
    class BeamParamsInterface
    {
     public:
      //! destructor
      virtual ~BeamParamsInterface() = default;

      /*! @name time integration parameters required for element-internal update of angular
       * velocity and acceleration (in combination with GenAlphaLieGroup) */
      //! @{
      virtual double get_beta() const = 0;
      virtual double get_gamma() const = 0;
      virtual double get_alphaf() const = 0;
      virtual double get_alpham() const = 0;
      //! @}
    };  // class BeamParamsInterface
  }  // namespace Elements

}  // namespace Solid

namespace BrownianDynamics
{
  /*! \brief Parameter interface for brownian dynamic data exchange between integrator and
   * structure (beam) elements */
  class ParamsInterface
  {
   public:
    //! destructor
    virtual ~ParamsInterface() = default;

    /// ~ 1e-3 / 2.27 according to cyron2011 eq 52 ff, viscosity of surrounding fluid
    virtual double const& get_viscosity() const = 0;

    /// the way how damping coefficient values for beams are specified
    virtual BrownianDynamics::BeamDampingCoefficientSpecificationType
    how_beam_damping_coefficients_are_specified() const = 0;

    /// get prefactors for damping coefficients of beams if they are specified via input file
    virtual std::array<double, 3> const& get_beam_damping_coefficient_prefactors_from_input_file()
        const = 0;

    //! get vector holding periodic bounding box object
    virtual std::shared_ptr<Core::Geo::MeshFree::BoundingBox> const& get_periodic_bounding_box()
        const = 0;

    //! get the current step length
    virtual const std::shared_ptr<Core::LinAlg::MultiVector<double>>& get_random_forces() const = 0;
  };
}  // namespace BrownianDynamics

FOUR_C_NAMESPACE_CLOSE

#endif
