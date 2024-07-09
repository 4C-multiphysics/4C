/*----------------------------------------------------------------------*/
/*! \file

\brief internal representation of an input file

\level 1


*/
/*----------------------------------------------------------------------*/


#ifndef FOUR_C_GLOBAL_DATA_HPP
#define FOUR_C_GLOBAL_DATA_HPP

#include "4C_config.hpp"

#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io_walltime_based_restart.hpp"
#include "4C_legacy_enum_definitions_problem_type.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_random.hpp"
#include "4C_utils_result_test.hpp"
#include "4C_utils_std_cxx20_ranges.hpp"

#include <Epetra_Comm.h>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>
#include <Teuchos_RCP.hpp>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class OutputControl;
  class InputControl;
}  // namespace Core::IO

namespace Core::Communication
{
  class Communicators;
}

namespace Mat
{
  namespace PAR
  {
    class Bundle;
  }
}  // namespace Mat

namespace PARTICLEENGINE
{
  class ParticleObject;
}

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    class Bundle;
  }
}  // namespace CONTACT

namespace Core::IO
{
  class DatFileReader;
}
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Global
{
  /*!
   * The Problem class gathers various input parameters and provides access
   * from anywhere via the singleton instance() function.
   *
   * This class is an old attempt to deal with parameters. The fundamental problem lies in the
   * global nature of the singleton instance. This behavior makes it very difficult to follow the
   * flow of data in the code and introduces hidden dependencies between different modules.
   * Nowadays, we know better and do not like to write new code that uses this class. Instead, try
   * to pass whatever data is needed direclty to a class or function. We work on removing
   * functionality from this class.
   *
   * Nevertheless, here is the old documentation, for as long as we'll be using this class:
   *
   * Global problem instance that keeps the discretizations
   * The global problem represents the input file passed to 4C. This class organizes the reading of
   * a dat file (utilizing the DatFileReader of course). That is way, in all but the most eccentric
   * cases there will be exactly one object of this class during a 4C run. This object contains all
   * parameters read from the input file as well as any material definitions and even all the
   * discretizations.
   *
   * <h3>Input parameters</h3>
   *
   * All input parameters are known by the global problem object. These parameters are guaranteed
   * to be valid (because they passed the validation) and are guaranteed to be there (because
   * default values have been set for all parameters missing from the input file.) This is
   * Teuchos::ParameterList magic, that just requires the list of valid parameters in the file
   * validparameters.cpp to be complete.
   *
   * The algorithms are meant to ask the global problem object for their parameters and extract
   * them from the respective parameter list directly.
   *
   * <h3>Discretizations</h3>
   *
   * The global problem object knows the discretizations defined by the input file. In order to
   * access a particular discretization you get the global problem object and ask.
   *
   * <h3>Materials</h3>
   *
   * The global problem object knows the material descriptions from the input file. These are not to
   * be confused with the material classes the elements know and work with. The global problem
   * object does not keep track of gauss point material values, all that is known here are the
   * definitions from the input file.
   *
   * <h3>Singleton behavior</h3>
   *
   * \warning This is a guru only section!
   *
   * The global problem behaves like a singleton, so there is always one instance available. But you
   * can have more than one instance of Problem. In normal situations this will not be needed. So
   * don't bother. Just call the static instance() function to get the global instance and access
   * your discretizations.
   *
   * In the special case that you want to read more that one input file, however, you will need to
   * handle the fields from each file separately.
   *
   * One artefact that comes from using global variables together with multiple Problem objects is
   * the notion of activating of problem object. This translates to setting the global variable
   * pointers to this Problem object's internal variables. Normally, if there is just one Problem
   * object, this is done by default. If you need more that one, however, you will have to activate
   * the global problems yourself.
   */
  class Problem
  {
   public:
    /// @name Instances

    /// Disallow copying this class.
    Problem(const Problem&) = delete;

    /// Disallow copying this class.
    Problem& operator=(const Problem&) = delete;

    /// Disallow moving this class.
    Problem(Problem&&) = delete;

    /// Disallow moving this class.
    Problem& operator=(Problem&&) = delete;

    /// return an instance of this class
    static Problem* instance(int num = 0);

    /// return number of problem instances
    static unsigned num_instances() { return instances_.size(); }

    /// calculation done, clean up
    /*!
      There can be a variety of objects to a problem. Some of them might
      require proper cleanup. Make sure we always do it.
     */
    static void done();

    //@}

    /// @name Input

    /// set restart step which was read from the command line
    void set_restart_step(int r);

    void set_input_control_file(Teuchos::RCP<Core::IO::InputControl>& input)
    {
      inputcontrol_ = input;
    }

    /// manipulate problem type
    void set_problem_type(Core::ProblemType targettype);

    void set_spatial_approximation_type(Core::FE::ShapeFunctionType shape_function_type);

    /// @name General query methods
    /// Once and for all definitions

    /// give enum of my problem type
    Core::ProblemType get_problem_type() const;

    /// give string name of my problem type
    std::string problem_name() const;

    /// return restart step
    [[nodiscard]] int restart() const;

    /// number of space dimensions (as specified in the input file)
    int n_dim() const;

    //! Return type of the basis function encoded as enum
    Core::FE::ShapeFunctionType spatial_approximation_type() const { return shapefuntype_; }

    //! @}

    /// @name Control file

    /*!
    \brief Create control file for output and read restart data if required

    In addition, issue a warning to the screen, if no binary output will be written.

    @param[in] comm Communicator
    @param[in] inputfile File name of input file
    @param[in] prefix
    @param[in] restartkenner
    */
    void open_control_file(const Epetra_Comm& comm, const std::string& inputfile,
        std::string prefix, const std::string& restartkenner);

    /// control file for restart read
    Teuchos::RCP<Core::IO::InputControl> input_control_file() { return inputcontrol_; }

    /// control file for normal output
    Teuchos::RCP<Core::IO::OutputControl> output_control_file() { return outputcontrol_; }

    /// write parameters read from input file for documentation
    void write_input_parameters();

    //@}

    /// @name Parameters read from file

    /// Set parameters from a parameter list and return with default values.
    void set_parameter_list(Teuchos::RCP<Teuchos::ParameterList> const& parameter_list);

    /// Return a const parameter list of all of the valid parameters that
    /// this->setParameterList(...) will accept.
    Teuchos::RCP<const Teuchos::ParameterList> get_valid_parameters() const;

    Teuchos::RCP<const Teuchos::ParameterList> get_parameter_list() const;

    /// @name Communicators and their parallel groups

    /// set communicators
    void set_communicators(Teuchos::RCP<Core::Communication::Communicators> communicators);

    /// return communicators
    Teuchos::RCP<Core::Communication::Communicators> get_communicators() const;

    //@}

    /// @name Input parameter sections
    /// direct access to parameters from input file sections

    const Teuchos::ParameterList& binning_strategy_params() const
    {
      return parameters_->sublist("BINNING STRATEGY");
    }
    const Teuchos::ParameterList& geometric_search_params() const
    {
      return parameters_->sublist("BOUNDINGVOLUME STRATEGY");
    }
    const Teuchos::ParameterList& io_params() const { return parameters_->sublist("IO"); }
    const Teuchos::ParameterList& structural_dynamic_params() const
    {
      return parameters_->sublist("STRUCTURAL DYNAMIC");
    }
    const Teuchos::ParameterList& cardiovascular0_d_structural_params() const
    {
      return parameters_->sublist("CARDIOVASCULAR 0D-STRUCTURE COUPLING");
    }
    const Teuchos::ParameterList& mortar_coupling_params() const
    {
      return parameters_->sublist("MORTAR COUPLING");
    }
    const Teuchos::ParameterList& contact_dynamic_params() const
    {
      return parameters_->sublist("CONTACT DYNAMIC");
    }
    const Teuchos::ParameterList& beam_interaction_params() const
    {
      return parameters_->sublist("BEAM INTERACTION");
    }
    const Teuchos::ParameterList& rve_multi_point_constraint_params() const
    {
      return get_parameter_list()->sublist("MULTI POINT CONSTRAINTS");
    }
    const Teuchos::ParameterList& brownian_dynamics_params() const
    {
      return parameters_->sublist("BROWNIAN DYNAMICS");
    }
    const Teuchos::ParameterList& thermal_dynamic_params() const
    {
      return parameters_->sublist("THERMAL DYNAMIC");
    }
    const Teuchos::ParameterList& tsi_dynamic_params() const
    {
      return parameters_->sublist("TSI DYNAMIC");
    }
    const Teuchos::ParameterList& fluid_dynamic_params() const
    {
      return parameters_->sublist("FLUID DYNAMIC");
    }
    const Teuchos::ParameterList& lubrication_dynamic_params() const
    {
      return parameters_->sublist("LUBRICATION DYNAMIC");
    }
    const Teuchos::ParameterList& scalar_transport_dynamic_params() const
    {
      return parameters_->sublist("SCALAR TRANSPORT DYNAMIC");
    }
    const Teuchos::ParameterList& sti_dynamic_params() const
    {
      return parameters_->sublist("STI DYNAMIC");
    }
    const Teuchos::ParameterList& f_s3_i_dynamic_params() const
    {
      return parameters_->sublist("FS3I DYNAMIC");
    }
    const Teuchos::ParameterList& ale_dynamic_params() const
    {
      return parameters_->sublist("ALE DYNAMIC");
    }
    const Teuchos::ParameterList& fsi_dynamic_params() const
    {
      return parameters_->sublist("FSI DYNAMIC");
    }
    const Teuchos::ParameterList& fpsi_dynamic_params() const
    {
      return parameters_->sublist("FPSI DYNAMIC");
    }
    const Teuchos::ParameterList& immersed_method_params() const
    {
      return parameters_->sublist("IMMERSED METHOD");
    }
    const Teuchos::ParameterList& cut_general_params() const
    {
      return parameters_->sublist("CUT GENERAL");
    }
    const Teuchos::ParameterList& xfem_general_params() const
    {
      return parameters_->sublist("XFEM GENERAL");
    }
    const Teuchos::ParameterList& x_fluid_dynamic_params() const
    {
      return parameters_->sublist("XFLUID DYNAMIC");
    }
    const Teuchos::ParameterList& fbi_params() const
    {
      return parameters_->sublist("FLUID BEAM INTERACTION");
    }
    const Teuchos::ParameterList& loma_control_params() const
    {
      return parameters_->sublist("LOMA CONTROL");
    }
    const Teuchos::ParameterList& biofilm_control_params() const
    {
      return parameters_->sublist("BIOFILM CONTROL");
    }
    const Teuchos::ParameterList& elch_control_params() const
    {
      return parameters_->sublist("ELCH CONTROL");
    }
    const Teuchos::ParameterList& ep_control_params() const
    {
      return parameters_->sublist("CARDIAC MONODOMAIN CONTROL");
    }
    const Teuchos::ParameterList& arterial_dynamic_params() const
    {
      return parameters_->sublist("ARTERIAL DYNAMIC");
    }
    const Teuchos::ParameterList& reduced_d_airway_dynamic_params() const
    {
      return parameters_->sublist("REDUCED DIMENSIONAL AIRWAYS DYNAMIC");
    }
    const Teuchos::ParameterList& red_airway_tissue_dynamic_params() const
    {
      return parameters_->sublist("COUPLED REDUCED-D AIRWAYS AND TISSUE DYNAMIC");
    }
    const Teuchos::ParameterList& poroelast_dynamic_params() const
    {
      return parameters_->sublist("POROELASTICITY DYNAMIC");
    }
    const Teuchos::ParameterList& poro_fluid_multi_phase_dynamic_params() const
    {
      return parameters_->sublist("POROFLUIDMULTIPHASE DYNAMIC");
    }
    const Teuchos::ParameterList& poro_multi_phase_scatra_dynamic_params() const
    {
      return parameters_->sublist("POROMULTIPHASESCATRA DYNAMIC");
    }
    const Teuchos::ParameterList& poro_multi_phase_dynamic_params() const
    {
      return parameters_->sublist("POROMULTIPHASE DYNAMIC");
    }
    const Teuchos::ParameterList& poro_scatra_control_params() const
    {
      return parameters_->sublist("POROSCATRA CONTROL");
    }
    const Teuchos::ParameterList& elasto_hydro_dynamic_params() const
    {
      return parameters_->sublist("ELASTO HYDRO DYNAMIC");
    }
    const Teuchos::ParameterList& ssi_control_params() const
    {
      return parameters_->sublist("SSI CONTROL");
    }
    const Teuchos::ParameterList& ssti_control_params() const
    {
      return parameters_->sublist("SSTI CONTROL");
    }
    const Teuchos::ParameterList& searchtree_params() const
    {
      return parameters_->sublist("SEARCH TREE");
    }
    const Teuchos::ParameterList& structural_nox_params() const
    {
      return parameters_->sublist("STRUCT NOX");
    }
    const Teuchos::ParameterList& loca_params() const { return parameters_->sublist("LOCA"); }
    const Teuchos::ParameterList& particle_params() const
    {
      return parameters_->sublist("PARTICLE DYNAMIC");
    }
    const Teuchos::ParameterList& pasi_dynamic_params() const
    {
      return parameters_->sublist("PASI DYNAMIC");
    }
    const Teuchos::ParameterList& level_set_control() const
    {
      return parameters_->sublist("LEVEL-SET CONTROL");
    }
    const Teuchos::ParameterList& wear_params() const { return parameters_->sublist("WEAR"); }
    const Teuchos::ParameterList& tsi_contact_params() const
    {
      return parameters_->sublist("TSI CONTACT");
    }
    const Teuchos::ParameterList& beam_contact_params() const
    {
      return parameters_->sublist("BEAM CONTACT");
    }
    const Teuchos::ParameterList& beam_potential_params() const
    {
      return parameters_->sublist("BEAM POTENTIAL");
    }
    const Teuchos::ParameterList& semi_smooth_plast_params() const
    {
      return parameters_->sublist("SEMI-SMOOTH PLASTICITY");
    }
    const Teuchos::ParameterList& electromagnetic_params() const
    {
      return parameters_->sublist("ELECTROMAGNETIC DYNAMIC");
    }
    const Teuchos::ParameterList& volmortar_params() const
    {
      return parameters_->sublist("VOLMORTAR COUPLING");
    }
    const Teuchos::ParameterList& mor_params() const { return parameters_->sublist("MOR"); };
    const Teuchos::ParameterList& mesh_partitioning_params() const
    {
      return parameters_->sublist("MESH PARTITIONING");
    }

    const Teuchos::ParameterList& problem_type_params() const
    {
      return parameters_->sublist("PROBLEM TYP");
    }

    const Teuchos::ParameterList& problem_size_params() const
    {
      return parameters_->sublist("PROBLEM SIZE");
    }

    const Teuchos::ParameterList& solver_params(int solverNr) const;

    std::function<const Teuchos::ParameterList&(int)> solver_params_callback() const;

    const Teuchos::ParameterList& umfpack_solver_params();

    //@}

    /// @name Discretizations

    /// get access to a particular discretization
    Teuchos::RCP<Core::FE::Discretization> get_dis(const std::string& name) const;

    auto discretization_range() { return std_20::ranges::views::all(discretizationmap_); }

    auto discretization_range() const { return std_20::ranges::views::all(discretizationmap_); }

    /// tell number of known fields
    unsigned num_fields() const { return discretizationmap_.size(); }

    /// tell names of known fields
    std::vector<std::string> get_dis_names() const;

    /// check whether a certain discretization exists or not
    bool does_exist_dis(const std::string& name) const;

    /// add a discretization to the global problem
    void add_dis(const std::string& name, Teuchos::RCP<Core::FE::Discretization> dis);


    //@}

    /// @name Materials

    /// return pointer to materials bundled to the problem
    Teuchos::RCP<Mat::PAR::Bundle> materials() { return materials_; }

    // return pointer to contact constitutive law bundled to the problem
    Teuchos::RCP<CONTACT::CONSTITUTIVELAW::Bundle> contact_constitutive_laws()
    {
      return contactconstitutivelaws_;
    }

    //@}

    /// @name Particles

    /// return reference to read in particles
    std::vector<std::shared_ptr<PARTICLEENGINE::ParticleObject>>& particles() { return particles_; }

    //@}

    std::map<std::pair<std::string, std::string>, std::map<int, int>>& cloning_material_map()
    {
      return clonefieldmatmap_;
    }

    /// @name Spatial Functions

    /**
     * Get a function read from the input file by its ID @p num.
     *
     * @tparam T The type of function interface.
     */
    template <typename T>
    const T& function_by_id(int num)
    {
      return functionmanager_.template function_by_id<T>(num);
    }

    //@}

    /// @name Result Tests

    /// Do the testing
    void test_all(const Epetra_Comm& comm) { resulttest_.test_all(comm); }

    /// add field specific result test object
    void add_field_test(Teuchos::RCP<Core::UTILS::ResultTest> test)
    {
      resulttest_.add_field_test(test);
    }

    Core::UTILS::ResultTestManager& get_result_test_manager() { return resulttest_; }

    //@}

    /// Return the class that handles random numbers globally
    Core::UTILS::Random* random() { return &random_; }

    /// Return the class that handles restart initiating -> to be extended
    Core::IO::RestartManager* restart_manager() { return &restartmanager_; }

    /**
     * Set the @p function_manager which contains all parsed functions.
     *
     * @note The parsing of functions must take place before. This calls wants a filled
     * FunctionManager.
     */
    void set_function_manager(Core::UTILS::FunctionManager&& function_manager);

    const Core::UTILS::FunctionManager& function_manager() const { return functionmanager_; }

   private:
    /// private default constructor to disallow creation of instances
    Problem();

    /// the single instance
    static std::vector<Problem*> instances_;

    /// the problem type
    Core::ProblemType probtype_;

    /// Spatial approximation type
    Core::FE::ShapeFunctionType shapefuntype_;

    /// the restart step (given by command line or input file)
    int restartstep_;

    /// discretizations of this problem
    std::map<std::string, Teuchos::RCP<Core::FE::Discretization>> discretizationmap_;

    /// material bundle
    Teuchos::RCP<Mat::PAR::Bundle> materials_;

    /// bundle containing all read-in contact constitutive laws
    Teuchos::RCP<CONTACT::CONSTITUTIVELAW::Bundle> contactconstitutivelaws_;

    /// all particles that are read in
    std::vector<std::shared_ptr<PARTICLEENGINE::ParticleObject>> particles_;

    /// basket of spatial function
    Core::UTILS::FunctionManager functionmanager_;

    /// all test values we might have
    Core::UTILS::ResultTestManager resulttest_;

    /// map of coupled fields and corresponding material IDs (needed for cloning
    /// of discretizations)
    std::map<std::pair<std::string, std::string>, std::map<int, int>> clonefieldmatmap_;

    /// communicators
    Teuchos::RCP<Core::Communication::Communicators> communicators_;

    /// @name File IO

    Teuchos::RCP<Core::IO::InputControl> inputcontrol_;
    Teuchos::RCP<Core::IO::OutputControl> outputcontrol_;

    //@}

    /// handles all sorts of random numbers
    Core::UTILS::Random random_;

    /// handles restart
    Core::IO::RestartManager restartmanager_;

    //! The central list of all paramters read from input.
    Teuchos::RCP<Teuchos::ParameterList> parameters_;
  };

}  // namespace Global

FOUR_C_NAMESPACE_CLOSE

#endif
