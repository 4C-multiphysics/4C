// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_global_data_read.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <fstream>

namespace
{
  using namespace FourC;

  class ElectrodeTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      setup_global_problem();

      setup_anode_ocp_polynomial();
      setup_anode_ocp_redlich_kister();
      setup_anode_ocp_taralov();
      setup_cathode_ocp_csv();
      setup_cathode_ocp_polynomial();
      setup_cathode_ocp_redlich_kister();
      setup_cathode_ocp_taralov();
    }

    void setup_anode_ocp_polynomial()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // half-cell open-circuit equilibrium potential by function
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::function);
      // create the chosen ocp model 'Function'
      auto& ocp_function = ocp_model.group("Function");
      ocp_function.add("OCP_FUNCT_NUM", 2);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);

      // add parameters for lithium metal anode to container
      container.add("C_MAX", 100000.0);

      // initialize parameter class for anode material
      parameters_anode_polynomial_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize anode material
      anode_polynomial_ = std::make_shared<Mat::Electrode>(parameters_anode_polynomial_.get());

      // define fictitious sample concentration values for anode material
      concentrations_anode_polynomial_.resize(3, 0.0);
      concentrations_anode_polynomial_[0] = 500.;   // anode concentration at 10% state of charge
      concentrations_anode_polynomial_[1] = 2500.;  // anode concentration at 50% state of charge
      concentrations_anode_polynomial_[2] = 4500.;  // anode concentration at 90% state of charge
    }

    void setup_anode_ocp_redlich_kister()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // choose semi-empirical Redlich-Kister expansion to model half-cell open-circuit equilibrium
      // potential
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::redlichkister);
      // create the chosen ocp model 'Redlich-Kister'
      auto& ocp_redlich_kister = ocp_model.group("Redlich-Kister");
      ocp_redlich_kister.add("OCP_PARA_NUM", 16);
      std::vector<double> ocp_para(16, 0.0);
      ocp_para[0] = 1.1652e4;
      ocp_para[1] = -3.268e3;
      ocp_para[2] = 3.955e3;
      ocp_para[3] = -4.573e3;
      ocp_para[4] = 6.147e3;
      ocp_para[5] = -3.339e3;
      ocp_para[6] = 1.117e4;
      ocp_para[7] = 2.997e2;
      ocp_para[8] = -4.866e4;
      ocp_para[9] = 1.362e2;
      ocp_para[10] = 1.373e5;
      ocp_para[11] = -2.129e4;
      ocp_para[12] = -1.722e5;
      ocp_para[13] = 3.956e4;
      ocp_para[14] = 9.302e4;
      ocp_para[15] = -3.280e4;
      ocp_redlich_kister.add("OCP_PARA", ocp_para);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 16.1);

      // initialize parameter class for anode material
      parameters_anode_redlichkister_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize anode material
      anode_redlichkister_ =
          std::make_shared<Mat::Electrode>(parameters_anode_redlichkister_.get());

      // define sample concentration values for anode material
      // cf. Goldin et al., Electrochimica Acta 64 (2012) 118-129
      concentrations_anode_redlichkister_.resize(3, 0.0);
      concentrations_anode_redlichkister_[0] = 2.029;  // anode concentration at 0% state of charge
      concentrations_anode_redlichkister_[1] =
          8.05;  // anode concentration at 50% intercalation fraction (constitutes singularity in
                 // original form of Redlich-Kister expansion)
      concentrations_anode_redlichkister_[2] =
          10.88;  // anode concentration at 100% state of charge
    }

    void setup_anode_ocp_taralov()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // choose half-cell open-circuit equilibrium potential according to Taralov, Taralova, Popov,
      // Iliev, Latz, and Zausch (2012)
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::taralov);
      // create the chosen ocp model 'Taralov'
      auto& ocp_taralov = ocp_model.group("Taralov");
      std::vector<double> ocp_para(13, 0.0);
      ocp_para[0] = -0.132;
      ocp_para[10] = 1.41;
      ocp_para[11] = -3.52;
      ocp_taralov.add("OCP_PARA", ocp_para);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 24.681);

      // initialize parameter class for anode material
      parameters_anode_taralov_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize anode material
      anode_taralov_ = std::make_shared<Mat::Electrode>(parameters_anode_taralov_.get());

      // define sample concentration values for anode material
      // cf. Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
      concentrations_anode_taralov_.resize(3, 0.0);
      concentrations_anode_taralov_[0] = 2.4681;   // anode concentration at 10% state of charge
      concentrations_anode_taralov_[1] = 12.3405;  // anode concentration at 50% state of charge
      concentrations_anode_taralov_[2] = 22.2129;  // anode concentration at 90% state of charge
    }

    void setup_cathode_ocp_csv()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // obtain half-cell open-circuit equilibrium potential from cubic spline interpolation of
      // *.csv data points
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::function);
      // create the chosen ocp model 'Function'
      auto& ocp_function = ocp_model.group("Function");
      // add cathode parameters to container according to master thesis by Alexander Rupp (2017)
      ocp_function.add("OCP_FUNCT_NUM", 1);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 4793.3);

      // initialize parameter class for cathode material
      parameters_cathode_csv_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize cathode material
      cathode_csv_ = std::make_shared<Mat::Electrode>(parameters_cathode_csv_.get());

      // define sample concentration values for cathode material
      // cf. master thesis by Alexander Rupp (2017)
      concentrations_cathode_csv_.resize(3, 0.0);
      concentrations_cathode_csv_[0] = 1677.6;  // cathode concentration at 100% state of charge
      concentrations_cathode_csv_[1] =
          3115.6;  // cathode concentration at 50% intercalation fraction
      concentrations_cathode_csv_[2] = 4553.6;  // cathode concentration at 0% state of charge
    }

    void setup_cathode_ocp_polynomial()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // add cathode parameters to container according to Ji et al., Journal of The Electrochemical
      // Society 160 (4) (2013) A636-A649
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::function);
      // create the chosen ocp model 'Function'
      auto& ocp_function = ocp_model.group("Function");
      ocp_function.add("OCP_FUNCT_NUM", 3);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 4793.3);

      // initialize parameter class for cathode material
      parameters_cathode_polynomial_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize cathode material
      cathode_polynomial_ = std::make_shared<Mat::Electrode>(parameters_cathode_polynomial_.get());

      // define sample concentration values for cathode material
      // cf. Ji et al., Journal of The Electrochemical Society 160 (4) (2013) A636-A649
      concentrations_cathode_polynomial_.resize(3, 0.0);
      concentrations_cathode_polynomial_[0] =
          1677.6;  // cathode concentration at 100% state of charge
      concentrations_cathode_polynomial_[1] =
          3115.6;  // cathode concentration at 50% state of charge
      concentrations_cathode_polynomial_[2] =
          4553.6;  // cathode concentration at 0% state of charge
    }


    void setup_cathode_ocp_redlich_kister()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // add cathode parameters to container according to Goldin et al., Electrochimica Acta 64
      // (2012) 118-129
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::redlichkister);
      // create the chosen ocp model 'Redlich-Kister'
      auto& ocp_redlich_kister = ocp_model.group("Redlich-Kister");
      ocp_redlich_kister.add("OCP_PARA_NUM", 21);
      std::vector<double> ocp_para(21, 0.0);
      ocp_para[0] = 3.954616e5;
      ocp_para[1] = -7.676e4;
      ocp_para[2] = 3.799e4;
      ocp_para[3] = -2.873e4;
      ocp_para[4] = 1.169e4;
      ocp_para[5] = 1.451e4;
      ocp_para[6] = -8.938e4;
      ocp_para[7] = 1.671e5;
      ocp_para[8] = -7.236e4;
      ocp_para[9] = -1.746e5;
      ocp_para[10] = -4.067e5;
      ocp_para[11] = 9.534e5;
      ocp_para[12] = 5.897e5;
      ocp_para[13] = -7.455e5;
      ocp_para[14] = -1.102e6;
      ocp_para[15] = -2.927e5;
      ocp_para[16] = 7.214e5;
      ocp_para[17] = 9.029e5;
      ocp_para[18] = -1.599e5;
      ocp_para[19] = 6.658e5;
      ocp_para[20] = -1.084e6;
      ocp_redlich_kister.add("OCP_PARA", ocp_para);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 23.9);

      // initialize parameter class for cathode material
      parameters_cathode_redlichkister_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize cathode material
      cathode_redlichkister_ =
          std::make_shared<Mat::Electrode>(parameters_cathode_redlichkister_.get());

      // define sample concentration values for cathode material
      // cf. Goldin et al., Electrochimica Acta 64 (2012) 118-129
      concentrations_cathode_redlichkister_.resize(3, 0.0);
      concentrations_cathode_redlichkister_[0] =
          10.56;  // cathode concentration at 100% state of charge
      concentrations_cathode_redlichkister_[1] =
          11.95;  // cathode concentration at 50% intercalation fraction (constitutes singularity
                  // in original form of Redlich-Kister expansion)
      concentrations_cathode_redlichkister_[2] =
          22.37;  // cathode concentration at 0% state of charge
    }

    void setup_cathode_ocp_taralov()
    {
      // initialize container for material parameters
      Core::IO::InputParameterContainer container;
      set_dummy_parameters_to_container(container);

      // add cathode parameters to container according to Taralov, Taralova, Popov, Iliev, Latz,
      // and Zausch (2012)
      auto& ocp_model = container.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::taralov);
      // create the chosen ocp model 'Taralov'
      auto& ocp_taralov = ocp_model.group("Taralov");
      std::vector<double> ocp_para(13, 0.0);
      ocp_para[0] = 4.06279;
      ocp_para[1] = 0.0677504;
      ocp_para[2] = -21.8502;
      ocp_para[3] = 12.8268;
      ocp_para[4] = -0.045;
      ocp_para[5] = -71.69;
      ocp_para[6] = -0.105734;
      ocp_para[7] = 1.00167;
      ocp_para[8] = 0.379571;
      ocp_para[9] = -1.576;
      ocp_para[10] = 0.01;
      ocp_para[11] = -200.0;
      ocp_para[12] = -0.19;
      ocp_taralov.add("OCP_PARA", ocp_para);
      ocp_model.add("X_MIN", -1.0);
      ocp_model.add("X_MAX", -1.0);
      // add the ocp model
      container.add("OCP_MODEL", ocp_model);
      container.add("C_MAX", 23.671);

      // initialize parameter class for cathode material
      parameters_cathode_taralov_ = std::make_shared<Mat::PAR::Electrode>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize cathode material
      cathode_taralov_ = std::make_shared<Mat::Electrode>(parameters_cathode_taralov_.get());

      // define sample concentration values for cathode material
      // cf. Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
      concentrations_cathode_taralov_.resize(3, 0.0);
      concentrations_cathode_taralov_[0] = 4.02407;  // cathode concentration at 90% state of charge
      concentrations_cathode_taralov_[1] = 11.8355;  // cathode concentration at 50% state of charge
      concentrations_cathode_taralov_[2] = 21.3039;  // cathode concentration at 17% state of charge
    }

    void set_dummy_parameters_to_container(Core::IO::InputParameterContainer& container) const
    {
      // add dummy parameters to container
      container.add("CHI_MAX", 1.0);
      container.add("COND_CONC_DEP_FUNCT", 0);
      container.add("COND_PARA_NUM", 0);
      container.add("COND_PARA", std::vector<double>(0, 0.0));
      container.add("COND_TEMP_SCALE_FUNCT", 0);
      container.add("COND_TEMP_SCALE_FUNCT_PARA_NUM", 0);
      container.add("COND_TEMP_SCALE_FUNCT_PARA", std::vector<double>(0, 0.0));
      container.add("DIFF_COEF_CONC_DEP_FUNCT", 0);
      container.add("DIFF_COEF_TEMP_SCALE_FUNCT", 0);
      container.add("DIFF_PARA_NUM", 0);
      container.add("DIFF_PARA", std::vector<double>(0, 0.0));
      container.add("DIFF_COEF_TEMP_SCALE_FUNCT_PARA_NUM", 0);
      container.add("DIFF_COEF_TEMP_SCALE_FUNCT_PARA", std::vector<double>(0, 0.0));
    }

    void setup_global_problem() const
    {
      Global::Problem* problem = Global::Problem::instance();

      Core::Communication::CommConfig config;
      // create default communicators
      Core::Communication::Communicators communicators = Core::Communication::create_comm(config);
      problem->set_communicators(communicators);
      MPI_Comm comm = communicators.global_comm();

      std::string csv_template_file_name = "4C_electrode_test.csv";
      std::string file_name = "4C_electrode_test.4C.yaml";
      setup_template_csv_file(csv_template_file_name);
      setup_dummy_input_file(file_name, csv_template_file_name);

      Core::IO::InputFile input_file = Global::set_up_input_file(comm);
      input_file.read(file_name);

      Global::read_parameter(*problem, input_file);

      {
        Core::Utils::FunctionManager function_manager;
        global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);
        function_manager.read_input(input_file);
        problem->set_function_manager(std::move(function_manager));
      }
    }

    void setup_template_csv_file(const std::string& csv_template_file_name) const
    {
      std::ofstream test_csv_file{csv_template_file_name};

      // include header line
      test_csv_file << "#intercalation_fraction,half_cell_open_circuit_potential" << '\n';
      // include the actual data
      test_csv_file << "0.300000000000000,4.44000000000000" << '\n';
      test_csv_file << "0.322400000000000,4.34000000000000" << '\n';
      test_csv_file << "0.371400000000000,4.23000000000000" << '\n';
      test_csv_file << "0.430900000000000,4.13000000000000" << '\n';
      test_csv_file << "0.502300000000000,4.02500000000000" << '\n';
      test_csv_file << "0.566000000000000,3.94500000000000" << '\n';
      test_csv_file << "0.680100000000000,3.83500000000000" << '\n';
      test_csv_file << "0.842500000000000,3.71000000000000" << '\n';
      test_csv_file << "0.910400000000000,3.62000000000000" << '\n';
      test_csv_file << "0.947500000000000,3.51000000000000" << '\n';
      test_csv_file << "0.960100000000000,3.42000000000000" << '\n';
      test_csv_file << "0.969900000000000,3.30000000000000" << '\n';
      test_csv_file << "0.976200000000000,3.16500000000000" << '\n';
      test_csv_file << "0.979000000000000,3.02000000000000" << '\n';
      test_csv_file << "0.980400000000000,2.90000000000000" << '\n';
      test_csv_file << "0.982500000000000,2.68800000000000" << '\n';
    }



    void setup_dummy_input_file(
        const std::string& file_name, const std::string& csv_template_file_name) const
    {
      std::ofstream file{file_name};

      file << std::format(R"(PROBLEM TYPE:
  PROBLEMTYPE: Electrochemistry
ELCH CONTROL:
  GAS_CONSTANT: {}
FUNCT1:
  - CUBIC_SPLINE_FROM_CSV:
      CSV: {}
FUNCT2:
  - FASTPOLYNOMIAL:
      NUMCOEFF: 1
      COEFF: [0.0]
FUNCT3:
  - FASTPOLYNOMIAL:
      NUMCOEFF: 5
      COEFF: [4.563, 2.595, -16.77, 23.88, -10.72])",
          gasconstant_, csv_template_file_name);
    }

    //! cathode material based on half cell open circuit potential obtained from cubic spline
    //! interpolation of *.csv data points
    std::shared_ptr<const Mat::Electrode> cathode_csv_;

    //! anode material based on half cell open circuit potential according to Redlich-Kister
    //! expansion
    std::shared_ptr<const Mat::Electrode> anode_redlichkister_;

    //! cathode material based on half cell open circuit potential according to Redlich-Kister
    //! expansion
    std::shared_ptr<const Mat::Electrode> cathode_redlichkister_;

    //! anode material based on half cell open circuit potential according to Taralov, Taralova,
    //! Popov, Iliev, Latz, and Zausch (2012)
    std::shared_ptr<const Mat::Electrode> anode_taralov_;

    //! cathode material based on half cell open circuit potential according to Taralov, Taralova,
    //! Popov, Iliev, Latz, and Zausch (2012)
    std::shared_ptr<const Mat::Electrode> cathode_taralov_;

    //! anode material based on polynomial half cell open circuit potential
    std::shared_ptr<const Mat::Electrode> anode_polynomial_;

    //! cathode material based on polynomial half cell open circuit potential
    std::shared_ptr<const Mat::Electrode> cathode_polynomial_;

    //! parameters for cathode material based on half cell open circuit potential obtained from
    //! cubic spline interpolation of *.csv data points
    std::shared_ptr<Mat::PAR::Electrode> parameters_cathode_csv_;

    //! parameters for anode material based on half cell open circuit potential according to
    //! Redlich-Kister expansion
    std::shared_ptr<Mat::PAR::Electrode> parameters_anode_redlichkister_;

    //! parameters for cathode material based on half cell open circuit potential according to
    //! Redlich-Kister expansion
    std::shared_ptr<Mat::PAR::Electrode> parameters_cathode_redlichkister_;

    //! parameters for anode material based on half cell open circuit potential according to
    //! Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
    std::shared_ptr<Mat::PAR::Electrode> parameters_anode_taralov_;

    //! parameters for cathode material based on half cell open circuit potential according to
    //! Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
    std::shared_ptr<Mat::PAR::Electrode> parameters_cathode_taralov_;

    //! parameters for anode material based on polynomial half cell open circuit potential
    std::shared_ptr<Mat::PAR::Electrode> parameters_anode_polynomial_;

    //! parameters for cathode material based on polynomial half cell open circuit potential
    std::shared_ptr<Mat::PAR::Electrode> parameters_cathode_polynomial_;

    //! sample concentration values for cathode material based on half cell open circuit potential
    //! obtained from cubic spline interpolation of *.csv data points
    std::vector<double> concentrations_cathode_csv_;

    //! sample concentration values for anode material based on half cell open circuit potential
    //! according to Redlich-Kister expansion
    std::vector<double> concentrations_anode_redlichkister_;

    //! sample concentration values for cathode material based on half cell open circuit potential
    //! according to Redlich-Kister expansion
    std::vector<double> concentrations_cathode_redlichkister_;

    //! sample concentration values for anode material based on half cell open circuit potential
    //! according to Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
    std::vector<double> concentrations_anode_taralov_;

    //! sample concentration values for cathode material based on half cell open circuit potential
    //! according to Taralov, Taralova, Popov, Iliev, Latz, and Zausch (2012)
    std::vector<double> concentrations_cathode_taralov_;

    //! sample concentration values for anode material based on polynomial half cell open circuit
    //! potential
    std::vector<double> concentrations_anode_polynomial_;

    //! sample concentration values for cathode material based on polynomial half cell open circuit
    //! potential
    std::vector<double> concentrations_cathode_polynomial_;

    //! detF
    const double detF_ = 1.0;

    //! faraday constant
    const double faraday_ = 9.64853399e4;

    //! universal gas constant
    const double gasconstant_ = 8.314472;

    //! factor F/(RT)
    const double frt_ = faraday_ / (gasconstant_ * 298.0);

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  TEST_F(ElectrodeTest, TestComputeOpenCircuitPotential)
  {
    // define results and tolerances for anode materials
    const std::array<double, 3> results_anode_redlichkister = {
        1.78149102529067321354e-01, 1.00269118707846299765e-01, 8.08307883078566852264e-02};
    const std::array<double, 3> tolerances_anode_redlichkister = {1.0e-15, 1.0e-15, 1.0e-16};
    const std::array<double, 3> results_anode_taralov = {
        8.59624971986640673549e-01, 1.10583257990501226953e-01, -7.26563582809017227682e-02};
    const std::array<double, 3> tolerances_anode_taralov = {1.0e-15, 1.0e-15, 1.0e-16};
    const std::array<double, 3> results_anode_polynomial = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_anode_polynomial = {1.0e-20, 1.0e-20, 1.0e-20};

    // test member function using sample concentration values for anode materials
    for (unsigned i = 0; i < concentrations_anode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(anode_redlichkister_->compute_open_circuit_potential(
                      concentrations_anode_redlichkister_[i], faraday_, frt_, detF_),
          results_anode_redlichkister[i], tolerances_anode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(anode_taralov_->compute_open_circuit_potential(
                      concentrations_anode_taralov_[i], faraday_, frt_, detF_),
          results_anode_taralov[i], tolerances_anode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(anode_polynomial_->compute_open_circuit_potential(
                      concentrations_anode_polynomial_[i], faraday_, frt_, detF_),
          results_anode_polynomial[i], tolerances_anode_polynomial[i]);
    }

    // define results and tolerances for cathode materials
    const std::array<double, 3> results_cathode_csv = {
        4.26568197738244947459e+00, 3.85982096426498033637e+00, 3.49646582304368758187e+00};
    const std::array<double, 3> tolerances_cathode_csv = {1.0e-14, 1.0e-14, 1.0e-14};
    const std::array<double, 3> results_cathode_redlichkister = {
        3.97322289583641552468e+00, 3.90180104449214848472e+00, 3.55923291247763096123e+00};
    const std::array<double, 3> tolerances_cathode_redlichkister = {1.0e-14, 1.0e-14, 1.0e-14};
    const std::array<double, 3> results_cathode_taralov = {
        4.68476462172226959524e+00, 4.12283197556701885844e+00, 3.90987668507063856893e+00};
    const std::array<double, 3> tolerances_cathode_taralov = {1.0e-14, 1.0e-14, 1.0e-14};
    const std::array<double, 3> results_cathode_polynomial = {
        4.27993831912628497349e+00, 3.80888970352517386431e+00, 3.63594305532142847426e+00};
    const std::array<double, 3> tolerances_cathode_polynomial = {1.0e-14, 1.0e-14, 1.0e-14};

    // test member function using sample concentration values for cathode materials
    for (unsigned i = 0; i < concentrations_cathode_csv_.size(); ++i)
    {
      EXPECT_NEAR(cathode_csv_->compute_open_circuit_potential(
                      concentrations_cathode_csv_[i], faraday_, frt_, detF_),
          results_cathode_csv[i], tolerances_cathode_csv[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(cathode_redlichkister_->compute_open_circuit_potential(
                      concentrations_cathode_redlichkister_[i], faraday_, frt_, detF_),
          results_cathode_redlichkister[i], tolerances_cathode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(cathode_taralov_->compute_open_circuit_potential(
                      concentrations_cathode_taralov_[i], faraday_, frt_, detF_),
          results_cathode_taralov[i], tolerances_cathode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(cathode_polynomial_->compute_open_circuit_potential(
                      concentrations_cathode_polynomial_[i], faraday_, frt_, detF_),
          results_cathode_polynomial[i], tolerances_cathode_polynomial[i]);
    }
  }

  TEST_F(ElectrodeTest, TestComputeFirstDerivOpenCircuitPotentialConc)
  {
    // define results and tolerances for anode materials
    const std::array<double, 3> results_anode_redlichkister = {
        -4.67265915814377499893e-02, -4.69987444027383099998e-03, -8.94809228199507436519e-03};
    const std::array<double, 3> tolerances_anode_redlichkister = {1.0e-16, 1.0e-17, 1.0e-17};
    const std::array<double, 3> results_anode_taralov = {
        -1.41425383954984579260e-01, -3.45971827773009360518e-02, -8.46358003529945959742e-03};
    const std::array<double, 3> tolerances_anode_taralov = {1.0e-15, 1.0e-16, 1.0e-17};
    const std::array<double, 3> results_anode_polynomial = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_anode_polynomial = {1.0e-20, 1.0e-20, 1.0e-20};

    // test member function using sample concentration values for anode materials
    for (unsigned i = 0; i < concentrations_anode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(anode_redlichkister_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_anode_redlichkister_[i], faraday_, frt_, detF_),
          results_anode_redlichkister[i], tolerances_anode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(anode_taralov_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_anode_taralov_[i], faraday_, frt_, detF_),
          results_anode_taralov[i], tolerances_anode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(anode_polynomial_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_anode_polynomial_[i], faraday_, frt_, detF_),
          results_anode_polynomial[i], tolerances_anode_polynomial[i]);
    }

    // define results and tolerances for cathode materials
    const std::array<double, 3> results_cathode_csv = {
        -3.97916196311376825587e-04, -1.82254204470957764054e-04, -1.19179495232888516508e-03};
    const std::array<double, 3> tolerances_cathode_csv = {1.0e-18, 1.0e-18, 1.0e-17};
    const std::array<double, 3> results_cathode_redlichkister = {
        -5.87374545823974372749e-02, -4.59544093325152322449e-02, -2.40034452721541273490e-02};
    const std::array<double, 3> tolerances_cathode_redlichkister = {1.0e-16, 1.0e-16, 1.0e-16};
    const std::array<double, 3> results_cathode_taralov = {
        -4.61526542535566619563e+00, -3.29033139167962890639e-03, -3.97138702703261764482e-02};
    const std::array<double, 3> tolerances_cathode_taralov = {1.0e-14, 1.0e-17, 1.0e-16};
    const std::array<double, 3> results_cathode_polynomial = {
        -4.60348767050483748987e-04, -1.48972359281208444927e-04, -2.87284599311867232425e-04};
    const std::array<double, 3> tolerances_cathode_polynomial = {1.0e-18, 1.0e-18, 1.0e-18};

    // test member function using sample concentration values for cathode materials
    for (unsigned i = 0; i < concentrations_cathode_csv_.size(); ++i)
    {
      EXPECT_NEAR(cathode_csv_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_cathode_csv_[i], faraday_, frt_, detF_),
          results_cathode_csv[i], tolerances_cathode_csv[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(cathode_redlichkister_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_cathode_redlichkister_[i], faraday_, frt_, detF_),
          results_cathode_redlichkister[i], tolerances_cathode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(cathode_taralov_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_cathode_taralov_[i], faraday_, frt_, detF_),
          results_cathode_taralov[i], tolerances_cathode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(cathode_polynomial_->compute_d_open_circuit_potential_d_concentration(
                      concentrations_cathode_polynomial_[i], faraday_, frt_, detF_),
          results_cathode_polynomial[i], tolerances_cathode_polynomial[i]);
    }
  }

  TEST_F(ElectrodeTest, TestComputeFirstDerivOpenCircuitPotentialTemp)
  {
    // define results and tolerances for cathode materials
    const std::array<double, 3> results_cathode_csv = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_cathode_csv = {1.0e-20, 1.0e-20, 1.0e-20};
    const std::array<double, 3> results_cathode_redlichkister = {
        2.0138191402737653e-05, 0.0, -2.3115616026561951e-04};
    const std::array<double, 3> tolerances_cathode_redlichkister = {1.0e-20, 1.0e-20, 1.0e-20};
    const std::array<double, 3> results_cathode_taralov = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_cathode_taralov = {1.0e-20, 1.0e-20, 1.0e-20};
    const std::array<double, 3> results_cathode_polynomial = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_cathode_polynomial = {1.0e-20, 1.0e-20, 1.0e-20};

    // test member function using sample concentration values for cathode materials
    for (unsigned i = 0; i < concentrations_cathode_csv_.size(); ++i)
    {
      EXPECT_NEAR(cathode_csv_->compute_d_open_circuit_potential_d_temperature(
                      concentrations_cathode_csv_[i], faraday_, gasconstant_),
          results_cathode_csv[i], tolerances_cathode_csv[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(cathode_redlichkister_->compute_d_open_circuit_potential_d_temperature(
                      concentrations_cathode_redlichkister_[i], faraday_, gasconstant_),
          results_cathode_redlichkister[i], tolerances_cathode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(cathode_taralov_->compute_d_open_circuit_potential_d_temperature(
                      concentrations_cathode_taralov_[i], faraday_, gasconstant_),
          results_cathode_taralov[i], tolerances_cathode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(cathode_polynomial_->compute_d_open_circuit_potential_d_temperature(
                      concentrations_cathode_polynomial_[i], faraday_, gasconstant_),
          results_cathode_polynomial[i], tolerances_cathode_polynomial[i]);
    }
  }

  TEST_F(ElectrodeTest, TestComputeSecondDerivOpenCircuitPotentialConc)
  {
    // define results and tolerances for anode materials
    const std::array<double, 3> results_anode_redlichkister = {
        3.04927774052812586292e-02, -1.05174077799887128251e-03, -7.35165984480485711766e-04};
    const std::array<double, 3> tolerances_anode_redlichkister = {1.0e-16, 1.0e-17, 1.0e-18};
    const std::array<double, 3> results_anode_taralov = {
        2.01700640785035384406e-02, 4.93424429221260389677e-03, 1.20707433751687906627e-03};
    const std::array<double, 3> tolerances_anode_taralov = {1.0e-16, 1.0e-17, 1.0e-17};
    const std::array<double, 3> results_anode_polynomial = {0.0, 0.0, 0.0};
    const std::array<double, 3> tolerances_anode_polynomial = {1.0e-20, 1.0e-20, 1.0e-20};

    // test member function using sample concentration values for anode materials
    for (unsigned i = 0; i < concentrations_anode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(
          anode_redlichkister_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
              concentrations_anode_redlichkister_[i], faraday_, frt_, detF_),
          results_anode_redlichkister[i], tolerances_anode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(anode_taralov_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
                      concentrations_anode_taralov_[i], faraday_, frt_, detF_),
          results_anode_taralov[i], tolerances_anode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_anode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(
          anode_polynomial_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
              concentrations_anode_polynomial_[i], faraday_, frt_, detF_),
          results_anode_polynomial[i], tolerances_anode_polynomial[i]);
    }

    // define results and tolerances for cathode materials
    const std::array<double, 3> results_cathode_csv = {
        1.63562986543888766237e-06, 1.43228888999912318160e-07, -1.10872570647637550212e-05};
    const std::array<double, 3> tolerances_cathode_csv = {1.0e-20, 1.0e-20, 1.0e-19};
    const std::array<double, 3> results_cathode_redlichkister = {
        1.41256487214221678611e-02, 5.72637600860917590773e-03, -7.56499159049272597299e-03};
    const std::array<double, 3> tolerances_cathode_redlichkister = {1.0e-16, 1.0e-17, 1.0e-17};
    const std::array<double, 3> results_cathode_taralov = {
        3.89765224056106092121e+01, -7.34373323389212427637e-03, -2.27647861985425976894e-02};
    const std::array<double, 3> tolerances_cathode_taralov = {1.0e-13, 1.0e-17, 1.0e-16};
    const std::array<double, 3> results_cathode_polynomial = {
        3.69515732169165666976e-08, 2.28146222921396267486e-07, -5.88484654599405069383e-07};
    const std::array<double, 3> tolerances_cathode_polynomial = {1.0e-20, 1.0e-20, 1.0e-20};

    // test member function using sample concentration values for cathode materials
    for (unsigned i = 0; i < concentrations_cathode_csv_.size(); ++i)
    {
      EXPECT_NEAR(cathode_csv_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
                      concentrations_cathode_csv_[i], faraday_, frt_, detF_),
          results_cathode_csv[i], tolerances_cathode_csv[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_redlichkister_.size(); ++i)
    {
      EXPECT_NEAR(
          cathode_redlichkister_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
              concentrations_cathode_redlichkister_[i], faraday_, frt_, detF_),
          results_cathode_redlichkister[i], tolerances_cathode_redlichkister[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_taralov_.size(); ++i)
    {
      EXPECT_NEAR(
          cathode_taralov_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
              concentrations_cathode_taralov_[i], faraday_, frt_, detF_),
          results_cathode_taralov[i], tolerances_cathode_taralov[i]);
    }
    for (unsigned i = 0; i < concentrations_cathode_polynomial_.size(); ++i)
    {
      EXPECT_NEAR(
          cathode_polynomial_->compute_d2_open_circuit_potential_d_concentration_d_concentration(
              concentrations_cathode_polynomial_[i], faraday_, frt_, detF_),
          results_cathode_polynomial[i], tolerances_cathode_polynomial[i]);
    }
  }
}  // namespace