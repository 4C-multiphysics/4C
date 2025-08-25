// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_DATA_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_DATA_HPP

#include "4C_config.hpp"

#include "4C_browniandyn_input.hpp"        // enums
#include "4C_contact_input.hpp"            // enums
#include "4C_contact_paramsinterface.hpp"  // base class of the ContactData class
#include "4C_fem_discretization.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"  // base class of the Data class
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_gauss_point_data_output_manager.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"  // base class of the Data class
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_pairedvector.hpp"

#include <NOX_Abstract_Vector.H>

#include <array>
#include <unordered_map>

// forward declarations

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo::MeshFree
{
  class BoundingBox;
}


namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
    class MeshFreeData;
    class BaseDataIO;
    class Base;
  }  // namespace TimeInt
  namespace ModelEvaluator
  {
    class BeamData;
    class ContactData;
    class BrownianDynData;
    class GaussPointDataOutputManager;


    /*! \brief Discrete implementation of the Solid::Elements::ParamsInterface
     *
     * This class represents an actual implementation of the Solid::Elements::ParamsInterface
     * class and gives you all the functionality to interchange data between the elements and the
     * structural time integrators.
     *
     * To add a new and more specialized data container or method
     * that is not directly linked to the pure structure problem,
     * but is linked to related problem classes such as contact, beam interaction or
     * Brownian dynamics instead, please refer to these sub-containers.
     * If these sub-containers are not sufficient for your purposes,
     * other sub-containers can be added following the provided templates.
     * To do so, the following basic steps should be considered:
     * - Add a pointer pointing to your new container as a member to this class (parent class).
     * - Initialize and setup your container in the Setup routine of this parent class.
     * - Add an accessor such as Contact() or BrownianDyn().
     * - Create your own data container class and use a call-back to the parent container to avoid
     *   code redundancy.
     *
     */
    class Data : public Solid::Elements::ParamsInterface
    {
      using quantity_norm_type_map =
          std::map<enum NOX::Nln::StatusTest::QuantityType, enum ::NOX::Abstract::Vector::NormType>;

     public:
      //! constructor
      Data();


      //! initialize the stuff coming from outside
      void init(const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr);

      //! setup member variables
      void setup();

      //! @name Derived Solid::Elements::ParamsInterface accessors
      //!@{

      //! get the desired action type [derived]
      [[nodiscard]] inline enum Core::Elements::ActionType get_action_type() const override
      {
        check_init_setup();
        return ele_action_;
      }

      //! get the total time for the element evaluation [derived]
      [[nodiscard]] inline double get_total_time() const override
      {
        check_init_setup();
        return total_time_;
      }

      //! get the current time step for the element evaluation [derived]
      [[nodiscard]] inline double get_delta_time() const override
      {
        check_init_setup();
        return delta_time_;
      }

      //! get function manager
      const Core::Utils::FunctionManager* get_function_manager() const override
      {
        return function_manager_;
      }

      //! get the current step length [derived]
      [[nodiscard]] inline double get_step_length() const override
      {
        check_init_setup();
        return step_length_;
      }

      //! get the is_default_step indicator [derived]
      [[nodiscard]] inline bool is_default_step() const override
      {
        check_init_setup();
        return is_default_step_;
      }

      //! get the current damping type [derived]
      [[nodiscard]] enum Inpar::Solid::DampKind get_damping_type() const override;

      //! get the tolerate errors indicator [derived]
      [[nodiscard]] inline bool is_tolerate_errors() const override
      {
        check_init_setup();
        return is_tolerate_errors_;
      }

      //! get the structural time integration factor for the displacement [derived]
      [[nodiscard]] inline double get_tim_int_factor_disp() const override
      {
        check_init_setup();
        return timintfactor_disp_;
      }

      //! get the structural time integration factor for the velocities [derived]
      [[nodiscard]] inline double get_tim_int_factor_vel() const override
      {
        check_init_setup();
        return timintfactor_vel_;
      }

      //! get the predictor type of the structural time integration
      [[nodiscard]] enum Inpar::Solid::PredEnum get_predictor_type() const override
      {
        check_init_setup();
        return predict_type_;
      }



      //! is the current state the predictor state?
      bool is_predictor_state() const;

      //! mutable access to the stress data vector
      std::shared_ptr<std::vector<char>>& stress_data_ptr() override;

      //! mutable access to the strain data vector
      std::shared_ptr<std::vector<char>>& strain_data_ptr() override;

      //! mutable access to the plastic strain data vector
      std::shared_ptr<std::vector<char>>& plastic_strain_data_ptr() override;

      //! mutable access to the stress data vector
      std::shared_ptr<std::vector<char>>& coupling_stress_data_ptr() override;

      //! mutable access to the optional quantity data vector
      std::shared_ptr<std::vector<char>> opt_quantity_data_ptr() override;

      //! get the current stress type [derived]
      [[nodiscard]] enum Inpar::Solid::StressType get_stress_output_type() const override;

      //! get the current strain type [derived]
      [[nodiscard]] enum Inpar::Solid::StrainType get_strain_output_type() const override;

      //! get the current plastic strain type [derived]
      [[nodiscard]] enum Inpar::Solid::StrainType get_plastic_strain_output_type() const override;

      //! get the current coupling stress type [derived]
      [[nodiscard]] enum Inpar::Solid::StressType get_coupling_stress_output_type() const override;

      //! get the current strain type [derived]
      virtual enum Inpar::Solid::OptQuantityType get_opt_quantity_output_type() const;

      //< get the manager of Gauss point data output
      std::shared_ptr<GaussPointDataOutputManager>& gauss_point_data_output_manager_ptr() override;

      //! register energy type to be computed and written to file
      void insert_energy_type_to_be_considered(enum Solid::EnergyType type);

      //! read-only access to energy data
      std::map<enum Solid::EnergyType, double> const& get_energy_data() const;

      //! read-only access to energy data
      double get_energy_data(enum Solid::EnergyType type) const;

      //! read-only access to energy data
      double get_energy_data(const std::string type) const;

      //! set value for a specific energy type
      void set_value_for_energy_type(double value, enum Solid::EnergyType type);

      //! set function manager
      void set_function_manager(const Core::Utils::FunctionManager& function_manager)
      {
        function_manager_ = &function_manager;
      }

      //! clear values for all energy types
      void clear_values_for_all_energy_types();

      /*! \brief Add contribution to energy of specified type [derived]
       *
       * @param value Value to be added
       * @param type Type of energy to be added to
       */
      void add_contribution_to_energy_type(
          const double value, const enum Solid::EnergyType type) override;

      //! get Interface to brownian dyn data [derived]
      [[nodiscard]] inline std::shared_ptr<BrownianDynamics::ParamsInterface>
      get_brownian_dyn_param_interface() const override
      {
        check_init_setup();
        return std::dynamic_pointer_cast<BrownianDynamics::ParamsInterface>(browniandyn_data_ptr_);
      }

      //! get special parameter interface for beam elements [derived]
      [[nodiscard]] inline std::shared_ptr<Solid::Elements::BeamParamsInterface>
      get_beam_params_interface_ptr() const override
      {
        FOUR_C_ASSERT(beam_data_ptr_, "pointer to beam data container not set!");
        return std::dynamic_pointer_cast<Solid::Elements::BeamParamsInterface>(beam_data_ptr_);
      }

      /** \brief get reference to the set model evaluator
       *
       *  \note Currently only used in the contact data container and therefore
       *  not part of the params_interface. Feel free to add. */
      const Generic& get_model_evaluator() const
      {
        FOUR_C_ASSERT(model_ptr_, "No reference to the model evaluator available!");

        return *model_ptr_;
      }

      //!@name set routines which can be called inside of the element [derived]
      //! @{

      /*! \brief Set the element evaluation error flag inside the element
       *
       * @param[in] error_flag Error flag to be set
       */
      inline void set_ele_eval_error_flag(const Elements::EvalErrorFlag& error_flag) override
      {
        ele_eval_error_flag_ = error_flag;
      }

      /*! \brief Collects and calculates the update norm of the current processor
       *
       * These methods are used to calculate the norms for a status test
       * w.r.t. internally, elementwise stored quantities.
       *
       * This is supported by the EAS formulation of the HEX8 element.
       * This specific method is used for the NormUpdate status test that tests the increments
       * of the solution variables.
       *
       * @param[in] qtype Quantity type which is tested
       * @param[in] numentries Length/size of the value arrays
       * @param[in] my_update_values Local part of increment/direction vector (with default step
       *                             length)
       * @param[in] my_new_sol_values Local part of the already updated solution vector
       * @param[in] step_length Step length of a possible active globalization strategy
       * @param[in] owner Owner of the corresponding element (used to avoid summing up ghost
       *                  entries)
       *
       * \sa sum_into_my_previous_sol_norm
       */
      void sum_into_my_update_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype,
          const int& numentries, const double* my_update_values, const double* my_new_sol_values,
          const double& step_length, const int& owner) override;

      /*! brief Collect and calculate solution norm of previous accepted Newton step on current
       * proc
       *
       * @param[in] qtype Quantity type which is tested
       * @param[in] numentries Length/size of the value arrays
       * @param[in] my_old_sol_values Local part of the previous solution vector
       * @param[in] owner Owner of the corresponding element (used to avoid summing up ghost
       *                  entries)
       *
       * \sa sum_into_my_update_norm
       */
      void sum_into_my_previous_sol_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype,
          const int& numentries, const double* my_old_sol_values, const int& owner) override;

      //!@}

      /*! \brief Returns the partial update norm of the given quantity on the current processor
       *
       * \todo Complete documentation of return parameters.
       *
       * @param[in] qtype Quantity type which is tested
       * @return
       */
      inline double get_my_update_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype) const
      {
        check_init_setup();
        std::map<enum NOX::Nln::StatusTest::QuantityType, double>::const_iterator c_it;
        c_it = my_update_norm_.find(qtype);
        // not on this proc
        if (c_it == my_update_norm_.end()) return 0.0;
        return c_it->second;
      }

      /*! \brief Return partial root-mean-squared norm of given quantity on current processor
       *
       * \todo Complete documentation of return parameters.
       *
       * @param[in] qtype Quantity type which is tested
       * @return
       */
      inline double get_my_rms_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype) const
      {
        check_init_setup();
        std::map<enum NOX::Nln::StatusTest::QuantityType, double>::const_iterator c_it;
        c_it = my_rms_norm_.find(qtype);
        // not on this proc
        if (c_it == my_rms_norm_.end()) return 0.0;
        return c_it->second;
      }

      /*! \brief Return partial solution norm of previous accepted Newton step of given quantity
       * on current processor
       *
       * \todo Complete documentation of return parameters.
       *
       * @param[in] qtype Quantity type which is tested
       * @return
       */
      inline double get_my_previous_sol_norm(
          const enum NOX::Nln::StatusTest::QuantityType& qtype) const
      {
        check_init_setup();
        std::map<enum NOX::Nln::StatusTest::QuantityType, double>::const_iterator c_it;
        c_it = my_prev_sol_norm_.find(qtype);
        // not on this proc
        if (c_it == my_prev_sol_norm_.end()) return 0.0;
        return c_it->second;
      }

      /*! brief Returns the update norm type
       *
       * \todo Complete documentation of return parameters.
       *
       * @param[in] qtype Quantity type which is tested
       * @return
       */
      inline enum ::NOX::Abstract::Vector::NormType get_update_norm_type(
          const enum NOX::Nln::StatusTest::QuantityType& qtype) const
      {
        check_init_setup();
        // collect the norm types only once
        static bool iscollected = false;
        if (not iscollected)
        {
          collect_norm_types_over_all_procs(normtype_update_);
          iscollected = true;
        }

        std::map<enum NOX::Nln::StatusTest::QuantityType,
            enum ::NOX::Abstract::Vector::NormType>::const_iterator c_it;
        c_it = normtype_update_.find(qtype);
        if (c_it == normtype_update_.end())
          FOUR_C_THROW("The corresponding norm type could not be found! (quantity: {})",
              NOX::Nln::StatusTest::quantity_type_to_string(qtype));
        return c_it->second;
      }

      /*! \brief Returns the dof number
       *
       * \todo Complete documentation of return parameters.
       *
       * @param[in] qtype Quantity type which is tested
       * @return
       */
      inline int get_my_dof_number(const enum NOX::Nln::StatusTest::QuantityType& qtype) const
      {
        check_init_setup();
        std::map<enum NOX::Nln::StatusTest::QuantityType, std::size_t>::const_iterator c_it;
        c_it = my_dof_number_.find(qtype);
        // not on this proc
        if (c_it == my_dof_number_.end()) return 0;
        return static_cast<int>(c_it->second);
      }

      /*! \brief Did an element evaluation error occur?
       *
       * @return Boolean flag to indicate occurrence error during element evaluation
       */
      bool is_ele_eval_error() const;

      /*! brief Access the element evaluation error flag
       *
       * @return Flag describing errors during element evaluation
       */
      inline Solid::Elements::EvalErrorFlag get_ele_eval_error_flag() const override
      {
        return ele_eval_error_flag_;
      }

      /*! @name Set routines which are used to set the parameters of the data container
       *
       *  \warning These functions are not allowed to be called by the elements!
       */
      //!@{

      /*! \brief Set the action type
       *
       * @param[in] actiontype Action type
       */
      inline void set_action_type(const enum Core::Elements::ActionType& actiontype)
      {
        ele_action_ = actiontype;
      }

      /*! \brief Set the tolerate errors flag
       *
       * @param[in] is_tolerate_errors Boolean flag to indicate error tolerance
       */
      inline void set_is_tolerate_error(const bool& is_tolerate_errors)
      {
        is_tolerate_errors_ = is_tolerate_errors;
      }

      /*! \brief Set the current step length
       *
       * @param[in] step_length Value for current step length to be set
       */
      inline void set_step_length(const double& step_length) { step_length_ = step_length; }

      //! set the default step flag
      inline void set_is_default_step(const bool& is_default_step)
      {
        is_default_step_ = is_default_step;
      }

      //! set the total time for the evaluation call
      inline void set_total_time(const double total_time) { total_time_ = total_time; }

      //! set the current time step for the evaluation call
      inline void set_delta_time(const double dt) { delta_time_ = dt; }

      //! set the time integration factor for the displacements
      inline void set_tim_int_factor_disp(const double timintfactor_disp)
      {
        timintfactor_disp_ = timintfactor_disp;
      }

      //! set the time integration factor for the velocities
      inline void set_tim_int_factor_vel(const double timintfactor_vel)
      {
        timintfactor_vel_ = timintfactor_vel;
      }

      //! set the predictor type of the structural time integration
      inline void set_predictor_type(const Inpar::Solid::PredEnum predictor_type)
      {
        predict_type_ = predictor_type;
      }

      //! set stress data vector
      inline void set_stress_data(const std::shared_ptr<std::vector<char>>& stressdata)
      {
        stressdata_ptr_ = stressdata;
      }

      /*!
       * \brief Set the pointer to the manager of gauss point data output
       *
       * \param data_manager Manager of gauss point data output
       */
      inline void set_gauss_point_data_output_manager_ptr(
          const std::shared_ptr<GaussPointDataOutputManager> data_manager)
      {
        gauss_point_data_manager_ptr_ = data_manager;
      }

      /// Return constant manager of gauss point data output
      inline const std::shared_ptr<GaussPointDataOutputManager>&
      get_gauss_point_data_output_manager_ptr() const
      {
        check_init_setup();
        return gauss_point_data_manager_ptr_;
      }

      //! get stress data vector
      inline const std::shared_ptr<std::vector<char>>& get_stress_data() const
      {
        return stressdata_ptr_;
      }

      //! get nodal postprocessed stress data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_stress_data_node_postprocessed() const
      {
        return stressdata_postprocessed_nodal_ptr_;
      }

      //! get nodal postprocessed stress data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_stress_data_node_postprocessed()
      {
        return stressdata_postprocessed_nodal_ptr_;
      }

      //! get element postprocessed stress data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_stress_data_element_postprocessed() const
      {
        return stressdata_postprocessed_element_ptr_;
      }

      //! get element postprocessed stress data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_stress_data_element_postprocessed()
      {
        return stressdata_postprocessed_element_ptr_;
      }

      //! get nodal postprocessed coupling stress data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_coupling_stress_data_node_postprocessed() const
      {
        return coupling_stressdata_postprocessed_nodal_ptr_;
      }

      //! get nodal postprocessed coupling stress data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_coupling_stress_data_node_postprocessed()
      {
        return coupling_stressdata_postprocessed_nodal_ptr_;
      }

      //! get element postprocessed coupling stress data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_coupling_stress_data_element_postprocessed() const
      {
        return coupling_stressdata_postprocessed_element_ptr_;
      }

      //! get element postprocessed coupling stress data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_coupling_stress_data_element_postprocessed()
      {
        return coupling_stressdata_postprocessed_element_ptr_;
      }

      //! set coupling stress data vector
      inline void set_coupling_stress_data(
          const std::shared_ptr<std::vector<char>>& coupling_stressdata)
      {
        coupling_stressdata_ptr_ = coupling_stressdata;
      }

      //! set strain data vector
      inline void set_strain_data(const std::shared_ptr<std::vector<char>>& straindata)
      {
        straindata_ptr_ = straindata;
      }

      //! get strain data vector
      inline const std::shared_ptr<std::vector<char>>& get_strain_data() const
      {
        return straindata_ptr_;
      }

      //! get nodal postprocessed strain data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_strain_data_node_postprocessed() const
      {
        return straindata_postprocessed_nodal_ptr_;
      }

      //! get nodal postprocessed strain data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_strain_data_node_postprocessed()
      {
        return straindata_postprocessed_nodal_ptr_;
      }

      //! get element postprocessed strain data vector
      inline const std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_strain_data_element_postprocessed() const
      {
        return straindata_postprocessed_element_ptr_;
      }

      //! get element postprocessed strain data vector
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>>&
      get_strain_data_element_postprocessed()
      {
        return straindata_postprocessed_element_ptr_;
      }

      //! set plastic strain data vector
      inline void set_plastic_strain_data(
          const std::shared_ptr<std::vector<char>>& plastic_straindata)
      {
        plastic_straindata_ptr_ = plastic_straindata;
      }

      //! set optional quantity data vector
      inline void set_opt_quantity_data(std::shared_ptr<std::vector<char>> opt_quantity_data)
      {
        opt_quantity_data_ptr_ = opt_quantity_data;
      }

      //! get stress data vector
      inline const std::shared_ptr<std::vector<char>>& get_opt_quantity_data() const
      {
        return opt_quantity_data_ptr_;
      }

      //! get nodal post processed optional quantity data vector
      [[nodiscard]] const Core::LinAlg::MultiVector<double>&
      get_opt_quantity_data_node_postprocessed() const
      {
        return *opt_quantity_data_postprocessed_nodal_ptr_;
      }

      //! get nodal post processed optional quantity data vector
      void set_opt_quantity_data_node_postprocessed(
          std::shared_ptr<Core::LinAlg::MultiVector<double>>
              opt_quantity_data_postprocessed_nodal_ptr)
      {
        opt_quantity_data_postprocessed_nodal_ptr_ = opt_quantity_data_postprocessed_nodal_ptr;
      }


      //! set model evaluator ptr
      inline void set_model_evaluator(Generic* model_ptr) { model_ptr_ = model_ptr; }

      //! reset the partial update norm value of the current processor
      void reset_my_norms(const bool& isdefaultstep);

      //! return the stress data (read-only)
      const std::vector<char>& stress_data() const;

      //! return the strain data (read-only)
      const std::vector<char>& strain_data() const;

      //! return the plastic strain data (read-only)
      const std::vector<char>& plastic_strain_data() const;

      //! return the coupling stress data (read-only)
      const std::vector<char>& coupling_stress_data() const;

      //! return the optional quantity data (read-only)
      [[nodiscard]] const std::vector<char>& opt_quantity_data() const;

      //!@}

      /*! @name Accessors to the remaining data containers
       *
       * \warning You are not allowed to call these functions on element level.
       */
      //!@{

      //! access the beam data container, if applicable
      inline BeamData& get_beam_data()
      {
        FOUR_C_ASSERT(beam_data_ptr_, "pointer to beam data container not set!");
        return *beam_data_ptr_;
      }
      inline const std::shared_ptr<BeamData>& get_beam_data_ptr()
      {
        FOUR_C_ASSERT(beam_data_ptr_, "pointer to beam data container not set!");
        return beam_data_ptr_;
      }

      //! access the contact data container, if the contact model is active
      inline ContactData& contact()
      {
        FOUR_C_ASSERT(contact_data_ptr_, "The contact model is not active!");
        return *contact_data_ptr_;
      }
      inline const std::shared_ptr<ContactData>& contact_ptr() const
      {
        FOUR_C_ASSERT(contact_data_ptr_, "The contact model is not active!");
        return contact_data_ptr_;
      }

      //! access the brownian dynamic data container
      inline BrownianDynData& brownian_dyn()
      {
        FOUR_C_ASSERT(browniandyn_data_ptr_, "The brownian dynamic model is not active!");
        return *browniandyn_data_ptr_;
      }
      inline const std::shared_ptr<BrownianDynData>& brownian_dyn_ptr()
      {
        FOUR_C_ASSERT(browniandyn_data_ptr_, "The brownian dynamic model is not active!");
        return browniandyn_data_ptr_;
      }
      //!@}

      /*! @name Accessors to some important member variables
       *  (necessary for possible other model evaluator containers, etc.) */
      //!@{

      //! Time integration strategy
      inline const Solid::TimeInt::Base& tim_int() const
      {
        check_init();
        return *timint_ptr_;
      }

      //! Structural dynamic data
      inline const Solid::TimeInt::BaseDataSDyn& sdyn() const
      {
        check_init();
        return *sdyn_ptr_;
      }

      //! input/ouput parameters
      inline const Solid::TimeInt::BaseDataIO& in_output() const
      {
        check_init();
        return *io_ptr_;
      }

      //! global state variables
      inline const Solid::TimeInt::BaseDataGlobalState& global_state() const
      {
        check_init();
        return *gstate_ptr_;
      }

      //! get the nonlinear iteration number
      int get_nln_iter() const;

      //! get the current step counter \f$(n+1)\f$
      int get_step_np() const;

      //! get the predictor indicator
      bool is_predictor() const;

      /*! Get the step number from which the current simulation has been
       *  restarted. Equal to 0 if no restart has been performed. */
      int get_restart_step() const;

      //!@}

     protected:
      //! returns the isinit_ flag
      inline const bool& is_init() const { return isinit_; };

      //! returns the issetup_ flag
      inline const bool& is_setup() const { return issetup_; };

      //! Checks the init and setup status
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

      //! Checks the init status
      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");
      }

     private:
      //! fill the normtype maps
      void fill_norm_type_maps();

      /*! \brief Get the norm type of the desired quantity.
       *
       *  If the norm type can be found, the function returns true,
       *  otherwise false. */
      bool get_update_norm_type(const enum NOX::Nln::StatusTest::QuantityType& qtype,
          enum ::NOX::Abstract::Vector::NormType& normtype);

      /*! \brief Get the WRMS absolute and relative tolerances of the desired quantity.
       *
       *  If the tolerances can be found, the function returns true,
       *  otherwise false. */
      bool get_wrms_tolerances(
          const enum NOX::Nln::StatusTest::QuantityType& qtype, double& atol, double& rtol);

      /*! \brief Sum locally values into a norm of the desired type
       *
       * \todo Complete documentation of input parameters.
       *
       * @param[in] numentries
       * @param[in] my_values
       * @param[in] normtype
       * @param[in] step_length
       * @param[in/out] my_norm
       */
      void sum_into_my_norm(const int& numentries, const double* my_values,
          const enum ::NOX::Abstract::Vector::NormType& normtype, const double& step_length,
          double& my_norm) const;

      /*! \brief Calculate a local relative mean square sum for the global WRMS status test
       *
       * \todo Complete documentation of input parameters.
       *
       * (1) \f$ v_i = x_{i}^{k-1} = x_{i}^{k} - sl* \Delta x_{i}^{k} \f$
       * (2) \f$ my_rms_norm = \sum_{i} [(x_i^{k}-x_{i}^{k-1}) / (RTOL * |x_{i}^{k-1}| +
       * ATOL)]^{2} \f$
       *
       * \param[in] atol Absolute tolerance
       * \param[in] rtol Relative tolerance
       * \param[in] step_length Step length of a possible active globalization strategy
       * \param[in] numentries Length/size of the value arrays
       * \param[in] my_update_values Local part of increment/direction vector (with default step
       *                             length)
       * \param[in] my_new_sol_values Local part of the already updated solution vector
       * \param[in/out] my_rms Root mean squared norm (to be summed into)
       */
      void sum_into_my_relative_mean_square(const double& atol, const double& rtol,
          const double& step_length, const int& numentries, const double* my_update_values,
          const double* my_new_sol_values, double& my_rms) const;

      void collect_norm_types_over_all_procs(const quantity_norm_type_map& normtypes) const;

     private:
      //! indicator if the init() routine has been called, yet.
      bool isinit_;

      //! indicator if the setup() routine has been called, yet.
      bool issetup_;

      /*! \brief Indicator for the norm type maps
       *
       * If true, the norm type maps have already been initialized successfully.
       */
      bool isntmaps_filled_;

      //! @name General element control parameters
      //!@{

      //! Current action type
      enum Core::Elements::ActionType ele_action_;

      //! Current predictor type
      enum Inpar::Solid::PredEnum predict_type_;

      //! element evaluation error flag
      enum Solid::Elements::EvalErrorFlag ele_eval_error_flag_;

      //! tolerate errors flag
      bool is_tolerate_errors_;

      //! total time for the evaluation
      double total_time_;

      //! function manager
      const Core::Utils::FunctionManager* function_manager_;

      //! current time step for the evaluation
      double delta_time_;
      //!@}

      //! @name Control parameters for the handling of element internal variables (e.g. EAS)
      //!@{

      //! Current step length of the nonlinear solver
      double step_length_;

      /*! \brief Indicator if the current step is a default step
       *
       *  Only important for the internal elementwise update. */
      bool is_default_step_;

      //!@}

      //! @name time integration parameters
      //!@{

      //! time integration factor for the displacements
      double timintfactor_disp_;

      //! time integration factor for the velocities
      double timintfactor_vel_;

      //!@}

      //! @name references to output data container
      //!@{
      //! stress data vector
      std::shared_ptr<std::vector<char>> stressdata_ptr_;

      //! postprocessed nodal stress data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> stressdata_postprocessed_nodal_ptr_;

      //! postprocessed element stress data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> stressdata_postprocessed_element_ptr_;

      //! strain data vector
      std::shared_ptr<std::vector<char>> straindata_ptr_;

      //! postprocessed nodal strain data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> straindata_postprocessed_nodal_ptr_;

      //! postprocessed element strain data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> straindata_postprocessed_element_ptr_;

      //! strain data vector
      std::shared_ptr<std::vector<char>> plastic_straindata_ptr_;

      //! coupling stress data vector
      //! e.g. in TSI: coupling stress corresponds to thermal stresses
      std::shared_ptr<std::vector<char>> coupling_stressdata_ptr_;

      //! postprocessed nodal coupling stress data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>>
          coupling_stressdata_postprocessed_nodal_ptr_;

      //! postprocessed element coupling stress data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>>
          coupling_stressdata_postprocessed_element_ptr_;

      //! optional quantity data vector
      std::shared_ptr<std::vector<char>> opt_quantity_data_ptr_;

      //! post processed nodal optional quantity data vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> opt_quantity_data_postprocessed_nodal_ptr_;

      //! system energy, stored separately by type
      std::map<enum Solid::EnergyType, double> energy_data_;

      //! Manager of gauss point data output
      std::shared_ptr<GaussPointDataOutputManager> gauss_point_data_manager_ptr_;

      //!@}

      //! map holding the force/rhs norm type of the active quantities
      quantity_norm_type_map normtype_force_;

      //! map holding the update norm type of the active quantities
      quantity_norm_type_map normtype_update_;

      //! map holding the dof number of the the active quantities on the current processor
      std::map<enum NOX::Nln::StatusTest::QuantityType, std::size_t> my_dof_number_;

      /*! map holding the absolute tolerance for the wrms status test of the active
       *  quantities on the current processor */
      std::map<enum NOX::Nln::StatusTest::QuantityType, double> atol_wrms_;

      /*! map holding the relative tolerance for the wrms status test of the active
       *  quantities on the current processor */
      std::map<enum NOX::Nln::StatusTest::QuantityType, double> rtol_wrms_;

      //! partial update norm of the current processor
      std::map<enum NOX::Nln::StatusTest::QuantityType, double> my_update_norm_;

      //! partial relative mean square norm of the current processor
      std::map<enum NOX::Nln::StatusTest::QuantityType, double> my_rms_norm_;

      //! global partial solution norm of the previous step
      std::map<enum NOX::Nln::StatusTest::QuantityType, double> my_prev_sol_norm_;

      //! read-only access to the structural dynamic parameters
      std::shared_ptr<const Solid::TimeInt::BaseDataSDyn> sdyn_ptr_;

      //! read-only access to the input/output parameters
      std::shared_ptr<const Solid::TimeInt::BaseDataIO> io_ptr_;

      //! read-only access to the global state data container
      std::shared_ptr<const Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

      //! read-only access to the timint object
      std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

      //! read-only access to the epetra communicator
      MPI_Comm comm_;

      //! beam data container pointer
      std::shared_ptr<BeamData> beam_data_ptr_;

      //! contact data container
      std::shared_ptr<ContactData> contact_data_ptr_;

      //! brownian dynamic data container
      std::shared_ptr<BrownianDynData> browniandyn_data_ptr_;

      //! pointer to a model evaluator object
      const Generic* model_ptr_;
    };  // class Data


    /*! data container holding special parameters required for the evaluation of beam elements
     *
     */
    class BeamData : public Solid::Elements::BeamParamsInterface
    {
     public:
      //! constructor
      BeamData();

      //! initialize the stuff coming from outside
      void init();

      //! setup member variables
      void setup();

      //! @name Derived Solid::Elements::BeamParamsInterface accessors
      //!@{

      //! get the Lie group GenAlpha time integration parameters [derived]
      [[nodiscard]] inline double get_beta() const override
      {
        check_init_setup();
        return beta_;
      }

      [[nodiscard]] inline double get_gamma() const override
      {
        check_init_setup();
        return gamma_;
      }

      [[nodiscard]] inline double get_alphaf() const override
      {
        check_init_setup();
        return alphaf_;
      }

      [[nodiscard]] inline double get_alpham() const override
      {
        check_init_setup();
        return alpham_;
      }

      //!@}

      /*! @name set routines which are used to set the parameters of the data container
       *
       *  These functions are not allowed to be called by the elements! */
      //!@{

      //! set the Lie group GenAlpha time integration parameters
      inline void set_beta(const double& beta) { beta_ = beta; }
      inline void set_gamma(const double& gamma) { gamma_ = gamma; }
      inline void set_alphaf(const double& alphaf) { alphaf_ = alphaf; }
      inline void set_alpham(const double& alpham) { alpham_ = alpham; }

      //!@}

     protected:
      //! returns the #isinit_ flag
      inline const bool& is_init() const { return isinit_; };

      //! returns the #issetup_ flag
      inline const bool& is_setup() const { return issetup_; };

      //! Checks the init and setup status
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

      //! Checks the init status
      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");
      }

     private:
      bool isinit_;

      bool issetup_;

      //! @name time integration parameters
      //!@{

      /*! \brief Lie-group Generalized-\f$\alpha\f$ parameters
       *
       * These parameters are needed for element-internal updates of angular velocities and
       * accelerations in case of non-additive rotation vector DOFs.
       *
       * See Lie-group Generalized-\f$\alpha\f$ time integration for details.
       *
       * \sa Solid::IMPLICIT::GenAlphaLieGroup
       */
      double beta_;
      double gamma_;
      double alphaf_;
      double alpham_;

      //!@}

    };  // class BeamData


    /*--------------------------------------------------------------------------*/
    /*! Contact data container for the contact model evaluation procedure.
     *
     */
    class ContactData : public CONTACT::ParamsInterface
    {
     public:
      //! constructor
      ContactData();

      //! initialize the stuff coming from outside
      void init(const std::shared_ptr<const Solid::ModelEvaluator::Data>& str_data_ptr);

      //! setup member variables
      void setup();

      //! returns the mortar/contact action type
      [[nodiscard]] inline enum Mortar::ActionType get_action_type() const override
      {
        check_init_setup();
        return mortar_action_;
      };

      //! get the nonlinear iteration number
      [[nodiscard]] int get_nln_iter() const override
      {
        check_init();
        return str_data_ptr_->get_nln_iter();
      };

      //! get the current step counter \f$(n+1)\f$
      [[nodiscard]] int get_step_np() const override
      {
        check_init();
        return str_data_ptr_->get_step_np();
      };

      [[nodiscard]] bool is_predictor() const override
      {
        check_init();
        return str_data_ptr_->is_predictor();
      };

      /*! \brief Get the current active predictor type
       *
       * If no predictor is active, \c pred_vague will be returned.
       *
       * @return Type of predictor
       *
       * */
      [[nodiscard]] enum Inpar::Solid::PredEnum get_predictor_type() const override
      {
        check_init();
        return str_data_ptr_->get_predictor_type();
      }

      //! get the current step length [derived]
      [[nodiscard]] inline double get_step_length() const override
      {
        check_init();
        return str_data_ptr_->get_step_length();
      };

      //! get the is_default_step indicator [derived]
      [[nodiscard]] inline bool is_default_step() const override
      {
        check_init();
        return str_data_ptr_->is_default_step();
      };

      //! is the current state the predictor state?
      inline bool is_predictor_state() const override
      {
        check_init();
        return str_data_ptr_->is_predictor_state();
      }

      //! get the current time step [derived]
      [[nodiscard]] inline double get_delta_time() const override
      {
        check_init();
        return str_data_ptr_->get_delta_time();
      }

      //! get reference to the set model evaluator
      [[nodiscard]] const Generic& get_model_evaluator() const override
      {
        check_init();
        return str_data_ptr_->get_model_evaluator();
      }

      //! get output file name
      [[nodiscard]] std::string get_output_file_path() const override;

      //! set coupling mode enumerator
      [[nodiscard]] enum CONTACT::CouplingScheme get_coupling_scheme() const override
      {
        return coupling_scheme_;
      }

      //! set coupling mode enumerator
      void set_coupling_scheme(const enum CONTACT::CouplingScheme scheme) override
      {
        coupling_scheme_ = scheme;
      }

      /*! \brief Get time step number from which the current simulation has been restarted
       *
       * Equal to 0 if no restart has been performed.
       */
      [[nodiscard]] int get_restart_step() const override
      {
        check_init();
        return str_data_ptr_->get_restart_step();
      }

      /*! @name set routines which are used to set the parameters of the data container
       *
       *  These functions are not allowed to be called by the elements! */
      //! @{

      //! set the action type
      inline void set_action_type(const enum Mortar::ActionType& actiontype)
      {
        mortar_action_ = actiontype;
      }

      void set_user_data(const std::any& user_data)
      {
        check_init();
        user_data_ = user_data;
      }

      void clear_user_data()
      {
        check_init();
        user_data_.reset();
      }

      //! @}

     protected:
      //! returns the isinit_ flag
      inline const bool& is_init() const { return isinit_; };

      //! returns the issetup_ flag
      inline const bool& is_setup() const { return issetup_; };

      //! Checks the init and setup status
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

      //! Checks the init status
      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");
      }

      //! Time integration strategy
      inline const Solid::TimeInt::Base& tim_int() const
      {
        check_init();
        return str_data_ptr_->tim_int();
      }

      //! Structural dynamic data
      inline const Solid::TimeInt::BaseDataSDyn& sdyn() const
      {
        check_init();
        return str_data_ptr_->sdyn();
      }

      //! input/ouput parameters
      inline const Solid::TimeInt::BaseDataIO& in_output() const
      {
        check_init();
        return str_data_ptr_->in_output();
      }

      //! global state variables
      inline const Solid::TimeInt::BaseDataGlobalState& global_state() const
      {
        check_init();
        return str_data_ptr_->global_state();
      }

      [[nodiscard]] const std::any& get_user_data() const override
      {
        check_init();
        return user_data_;
      }



     private:
      bool isinit_;

      bool issetup_;

      enum Mortar::ActionType mortar_action_;

      enum CONTACT::CouplingScheme coupling_scheme_;

      std::shared_ptr<const Solid::ModelEvaluator::Data> str_data_ptr_;

      std::any user_data_;
    };

    /*! Brownian dynamic data container for the model evaluation procedure.
     *
     */
    class BrownianDynData : public BrownianDynamics::ParamsInterface
    {
     public:
      //! constructor
      BrownianDynData();

      //! initialize the stuff coming from outside
      void init(std::shared_ptr<const Solid::ModelEvaluator::Data> const& str_data_ptr);

      //! setup member variables
      void setup();

      //! Structural dynamic data
      inline Solid::TimeInt::BaseDataSDyn const& sdyn() const
      {
        check_init();
        return str_data_ptr_->sdyn();
      }

      /// thermal energy
      double const& kt() const
      {
        check_init_setup();
        return kt_;
      };

      //! get specified time curve number of imposed Dirichlet BCs
      void resize_random_force_m_vector(
          std::shared_ptr<Core::FE::Discretization> discret_ptr, int maxrandnumelement);

      //! get mutable random force vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& get_random_forces()
      {
        check_init_setup();
        return randomforces_;
      };

      /// ~ 1e-3 / 2.27 according to cyron2011 eq 52 ff, viscosity of surrounding fluid
      double const& max_rand_force() const
      {
        check_init_setup();
        return maxrandforce_;
      };

      /// thermal energy
      double const& time_step_const_rand_numb() const
      {
        check_init_setup();
        return timeintconstrandnumb_;
      };
      //! @}

      /*! @name set routines which are allowed to be called by the elements
       */
      //! @{
      std::shared_ptr<Core::LinAlg::MultiVector<double>> const& get_random_forces() const override
      {
        check_init_setup();
        return randomforces_;
      };

      /// ~ 1e-3 / 2.27 according to cyron2011 eq 52 ff, viscosity of surrounding fluid
      double const& get_viscosity() const override
      {
        check_init_setup();
        return viscosity_;
      };

      /// the way how damping coefficient values for beams are specified
      [[nodiscard]] BrownianDynamics::BeamDampingCoefficientSpecificationType
      how_beam_damping_coefficients_are_specified() const override
      {
        check_init_setup();
        return beam_damping_coeff_specified_via_;
      }

      /// get prefactors for damping coefficients of beams if they are specified via input file
      [[nodiscard]] std::array<double, 3> const&
      get_beam_damping_coefficient_prefactors_from_input_file() const override
      {
        check_init_setup();
        return beams_damping_coefficient_prefactors_perunitlength_;
      };

      //! get vector holding periodic bounding box object
      [[nodiscard]] std::shared_ptr<Core::Geo::MeshFree::BoundingBox> const&
      get_periodic_bounding_box() const override
      {
        check_init_setup();
        return str_data_ptr_->sdyn().get_periodic_bounding_box();
      }
      //! @}

     protected:
      //! returns the isinit_ flag
      inline const bool& is_init() const { return isinit_; };

      //! returns the issetup_ flag
      inline const bool& is_setup() const { return issetup_; };

      //! Checks the init and setup status
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

      //! Checks the init status
      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");
      }

     private:
      bool isinit_;

      bool issetup_;

      std::shared_ptr<const Solid::ModelEvaluator::Data> str_data_ptr_;

      /// ~ 1e-3 / 2.27 according to cyron2011 eq 52 ff, viscosity of surrounding fluid
      double viscosity_;
      /// thermal energy
      double kt_;
      /// any random force beyond MAXRANDFORCE*(standdev) will be omitted and redrawn. -1.0 means
      /// no bounds.
      double maxrandforce_;
      /// within this time interval the random numbers remain constant. -1.0 means no prescribed
      /// time interval
      double timeintconstrandnumb_;

      /// the way how damping coefficient values for beams are specified
      BrownianDynamics::BeamDampingCoefficientSpecificationType beam_damping_coeff_specified_via_;

      /// prefactors for damping coefficients of beams if they are specified via input file
      /// (per unit length, NOT yet multiplied by viscosity)
      std::array<double, 3> beams_damping_coefficient_prefactors_perunitlength_;

      /// multiVector holding random forces
      std::shared_ptr<Core::LinAlg::MultiVector<double>> randomforces_;
    };

  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
