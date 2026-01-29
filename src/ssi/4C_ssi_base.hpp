// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_BASE_HPP
#define FOUR_C_SSI_BASE_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_ssi_str_model_evaluator_base.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class Structure;
  class ScaTraBaseAlgorithm;
  class SSIStructureWrapper;
  class StructureBaseAlgorithmNew;
}  // namespace Adapter

namespace CONTACT
{
  class NitscheStrategySsi;
}

namespace SSI
{
  enum class FieldCoupling;
}  // namespace SSI

namespace Core::LinAlg
{
  class MultiMapExtractor;
}

namespace ScaTra
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace ScaTra

namespace SSI
{
  // forward declaration
  class SSICouplingBase;

  namespace Utils
  {
    class SSISlaveSideConverter;
    class SSIMeshTying;
  }  // namespace Utils

  enum class RedistributionType
  {
    none,     //!< unknown redistribution type
    binning,  //!< redistribute by binning
    match     //!< redistribute by node matching
  };

  //! Base class of all solid-scatra algorithms
  class SSIBase : public Adapter::AlgorithmBase
  {
   public:
    explicit SSIBase(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    //! return counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm)
    [[nodiscard]] int iteration_count() const { return iter_; }

    //! reset the counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm)
    void reset_iteration_count() { iter_ = 0; }

    //! increment the counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm) by 1
    void increment_iteration_count() { iter_ += 1; }

    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return void

    */
    virtual void init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool is_ale) = 0;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    virtual void setup();

    /*!
     * @brief Perform all necessary tasks after setting up the object.
     * Currently, this only calls the post_setup routine of the structure field.
     */
    void post_setup() const;

    //! returns true if setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    /*!
     * @brief checks whether simulation is restarted or not
     *
     * @return  flag indicating if simulation is restarted
     */
    [[nodiscard]] bool is_restart() const;

    [[nodiscard]] bool is_s2i_kinetics_with_pseudo_contact() const
    {
      return is_s2i_kinetic_with_pseudo_contact_;
    }

    /*! \brief Setup discretizations and dofsets

     Init coupling object \ref ssicoupling_ and
     other possible coupling objects in derived
     classes

    \return RedistributionType

    */
    virtual RedistributionType init_field_coupling(const std::string& struct_disname);

    /*! \brief Setup discretizations


    */
    virtual void init_discretizations(MPI_Comm comm, const std::string& struct_disname,
        const std::string& scatra_disname, bool redistribute_struct_dis);

    /// setup
    virtual void setup_system();

    /// timeloop of coupled problem
    virtual void timeloop() = 0;

    /// test results (if necessary)
    virtual void test_results(MPI_Comm comm) const;

    /// read restart
    void read_restart(int restart) override;

    //! access to structural field
    [[nodiscard]] std::shared_ptr<Adapter::SSIStructureWrapper> structure_field() const
    {
      return structure_;
    }

    /// pointer to the underlying structure problem base algorithm
    [[nodiscard]] std::shared_ptr<Adapter::StructureBaseAlgorithmNew> structure_base_algorithm()
        const
    {
      return struct_adapterbase_ptr_;
    }

    //! access the scalar transport base algorithm
    [[nodiscard]] std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_base_algorithm() const
    {
      return scatra_base_algorithm_;
    }

    //! access the scalar transport base algorithm on manifolds
    [[nodiscard]] std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_manifold_base_algorithm()
        const
    {
      return scatra_manifold_base_algorithm_;
    }

    //! access the scalar transport field
    [[nodiscard]] std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_field() const;

    //! access the scalar transport field on manifolds
    [[nodiscard]] std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_manifold() const;

    /// set structure solution on other fields
    void set_struct_solution(const Core::LinAlg::Vector<double>& disp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel, bool set_mechanical_stress);

    /// set scatra solution on other fields
    virtual void set_scatra_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const;

    /*!
     * @brief set contact states needed for evaluation of ssi contact
     *
     * @param[in] phi  scatra state to be set to contact nitsche strategy
     */
    void set_ssi_contact_states(std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const;

    /// set micro scatra solution on other fields
    virtual void set_micro_scatra_solution(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const;

    /// set temperature field  by evaluating time dependent function
    void evaluate_and_set_temperature_field();

    //! get bool indicating if we have at least one ssi interface meshtying condition
    [[nodiscard]] bool ssi_interface_meshtying() const { return ssi_interface_meshtying_; }

    //! return the scatra-scatra interface meshtying strategy
    [[nodiscard]] std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_s2i() const
    {
      return meshtying_strategy_s2i_;
    }

    //! returns whether calculation of the initial potential field is performed
    [[nodiscard]] bool do_calculate_initial_potential_field() const;

    //! returns if the scalar transport time integration is of type electrochemistry
    [[nodiscard]] bool is_elch_scatra_time_int_type() const;

    //! solve additional scatra field on manifolds
    [[nodiscard]] bool is_scatra_manifold() const { return is_scatra_manifold_; }

    //! activate mesh tying between overlapping manifold fields
    [[nodiscard]] bool is_scatra_manifold_meshtying() const { return is_manifold_meshtying_; }

    //! Redistribute nodes and elements on processors
    void redistribute(RedistributionType redistribution_type) const;

    //! get bool indicating if we have at least one ssi interface contact condition
    [[nodiscard]] bool ssi_interface_contact() const { return ssi_interface_contact_; }

    //! set up a pointer to the contact strategy of the structural field and store it
    void setup_contact_strategy();

    //! SSI structure meshtying object containing coupling adapters, converters and maps
    [[nodiscard]] std::shared_ptr<Utils::SSIMeshTying> ssi_structure_mesh_tying() const
    {
      return ssi_structure_meshtying_;
    }

    //! return contact nitsche strategy for ssi problems
    [[nodiscard]] std::shared_ptr<CONTACT::NitscheStrategySsi> nitsche_strategy_ssi() const
    {
      return contact_strategy_nitsche_;
    }

   protected:
    //! check if \ref setup() was called
    void check_is_setup() const
    {
      if (not is_setup()) FOUR_C_THROW("setup() was not called.");
    }

    //! check if \ref init() was called
    void check_is_init() const
    {
      if (not is_init()) FOUR_C_THROW("init(...) was not called.");
    }

    //! copy modified time step from scatra to scatra manifold field
    void set_dt_from_scatra_to_manifold() const;

    //! copy modified time step from scatra to this SSI algorithm
    void set_dt_from_scatra_to_ssi();

    //! copy modified time step from scatra to structure field
    void set_dt_from_scatra_to_structure() const;

    //! set structure stress state on scatra field
    void set_mechanical_stress_state(
        std::shared_ptr<const Core::LinAlg::Vector<double>> mechanical_stress_state) const;

    void set_modelevaluator_base_ssi(
        std::shared_ptr<Solid::ModelEvaluator::Generic> modelevaluator_ssi_base)
    {
      modelevaluator_ssi_base_ = std::move(modelevaluator_ssi_base);
    }

    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; }

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; }

    //! set up structural model evaluator for scalar-structure interaction
    virtual void setup_model_evaluator();

    //! macro-micro scatra problem?
    [[nodiscard]] bool macro_scale() const { return macro_scale_; }

    //! different time step size between scatra field and structure field
    [[nodiscard]] bool diff_time_step_size() const { return diff_time_step_size_; }

    //! store contact nitsche strategy for ssi problems
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_strategy_nitsche_;

   private:
    /*!
     * @brief Checks whether flags for adaptive time stepping in ssi have been set consistently
     *
     * @param[in] scatraparams  parameter list containing the SCALAR TRANSPORT DYNAMIC parameters
     * @param[in] structparams  parameter list containing the STRUCTURAL DYNAMIC parameters
     */
    static void check_adaptive_time_stepping(
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams);

    /*!
     * @brief Time integrators for the scalar and structure fields are instantiated and initialized
     *
     * @param[in] globaltimeparams  parameter list containing the SSI CONTROL parameters
     * @param[in] scatraparams      parameter list containing the SCALAR TRANSPORT DYNAMIC
     *                              parameters
     * @param[in] structparams      parameter list containing the STRUCTURAL DYNAMIC parameters
     * @param[in] struct_disname    name of structure discretization
     * @param[in] scatra_disname    name of scalar transport discretization
     * @param[in] is_ale            flag indicating if ALE is activated
     */
    void init_time_integrators(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool is_ale);

    /*!
     * @brief check whether pseudo contact is activated for at least one of the s2i kinetics
     * conditions
     *
     * @param[in] struct_disname  name of structure discretization
     */
    [[nodiscard]] bool check_s2i_kinetics_condition_for_pseudo_contact(
        const std::string& struct_disname) const;

    //! check whether scatra-structure interaction flags are set correctly
    void check_ssi_flags() const;

    /*!
     * @brief SSI interface condition definition is checked
     *
     * @param[in] struct_disname  name of structure discretization
     */
    void check_ssi_interface_conditions(const std::string& struct_disname) const;

    //! returns true if init(..) was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    /// set structure mesh displacement on scatra field
    void set_mesh_disp(const Core::LinAlg::Vector<double>& disp);

    /// set structure velocity field on scatra field
    void set_velocity_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> vel);

    //! different time step size between scatra field and structure field
    const bool diff_time_step_size_;

    //! Type of coupling strategy between the two fields of the SSI problems
    const SSI::FieldCoupling fieldcoupling_;

    //! flag indicating if class is initialized
    bool isinit_ = false;

    //! flag indicating if class is setup
    bool issetup_ = false;

    //! solve additional scatra field on manifolds
    const bool is_scatra_manifold_;

    //! activate mesh tying between overlapping manifold fields
    const bool is_manifold_meshtying_;

    //! flag indicating if an s2i kinetic condition with activated pseudo contact is available
    const bool is_s2i_kinetic_with_pseudo_contact_;

    //! counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm)
    int iter_ = 0;

    //! macro-micro scatra problem?
    const bool macro_scale_;

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_s2i_;

    //! structure model evaluator for ssi problems
    std::shared_ptr<Solid::ModelEvaluator::Generic> modelevaluator_ssi_base_;

    //! underlying scatra problem base algorithm
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_base_algorithm_;

    //! underlying scatra problem base algorithm on manifolds
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_manifold_base_algorithm_;

    //! SSI structure mesh tying object containing coupling adapters, converters and maps
    std::shared_ptr<Utils::SSIMeshTying> ssi_structure_meshtying_;

    /// helper class for applying SSI couplings
    std::shared_ptr<SSICouplingBase> ssicoupling_;

    //! bool indicating if we have at least one ssi interface contact condition
    const bool ssi_interface_contact_;

    //! bool indicating if we have at least one ssi interface meshtying condition
    const bool ssi_interface_meshtying_;

    /// ptr to underlying structure
    std::shared_ptr<Adapter::SSIStructureWrapper> structure_;

    /// ptr to the underlying structure problem base algorithm
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> struct_adapterbase_ptr_;

    //! number of function for prescribed temperature
    const int temperature_funct_num_;

    //! vector of temperatures
    std::shared_ptr<Core::LinAlg::Vector<double>> temperature_vector_;

    //! a zero vector of full length with structure dofs
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_structure_;
  };  // SSI_Base
}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif
