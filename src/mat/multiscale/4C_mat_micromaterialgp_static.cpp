// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_micromaterialgp_static.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_solid_ele.hpp"
#include "4C_solid_ele_interface_serializable.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <filesystem>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct GlobalMicroState
  {
    //! map between number of micro-scale discretization and micro-scale time integrator
    std::map<int, std::shared_ptr<MultiScale::MicroStatic>> microstaticmap_;

    //! map between number of micro-scale discretization and number of associated macro-scale
    //! Gauss points
    std::map<int, int> microstaticcounter_;
  };

  // Manage a global state within a singleton
  GlobalMicroState& global_micro_state()
  {
    static auto global_micro_state =
        Core::Utils::make_singleton_owner([]() { return std::make_unique<GlobalMicroState>(); });

    return *global_micro_state.instance(Core::Utils::SingletonAction::create);
  }

}  // namespace

/// construct an instance of MicroMaterial for a given Gauss point and
/// microscale discretization

Mat::MicroMaterialGP::MicroMaterialGP(const int gp, const int ele_ID, const bool eleowner,
    const int microdisnum, const double V0, const bool initialize_runtime_output_writer)
    : gp_(gp),
      ele_id_(ele_ID),
      microdisnum_(microdisnum),
      history_data_(std::unordered_map<int, std::vector<char>>())
{
  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis("structure");
  dis_ = std::make_shared<Core::LinAlg::Vector<double>>(*microdis->dof_row_map(), true);
  disn_ = std::make_shared<Core::LinAlg::Vector<double>>(*microdis->dof_row_map(), true);

  // data must be consistent between micro and macro input file
  const Teuchos::ParameterList& sdyn_macro =
      Global::Problem::instance()->structural_dynamic_params();
  const Teuchos::ParameterList& sdyn_micro = microproblem->structural_dynamic_params();

  dt_ = sdyn_macro.get<double>("TIMESTEP");
  Core::Communication::broadcast(&dt_, 1, 0, microdis->get_comm());
  step_ = 0;
  stepn_ = step_ + 1;
  time_ = 0.;
  timen_ = time_ + dt_;

  // if class handling microscale simulations is not yet initialized
  // -> set up

  if (global_micro_state().microstaticmap_.find(microdisnum_) ==
          global_micro_state().microstaticmap_.end() or
      global_micro_state().microstaticmap_[microdisnum_] == nullptr)
  {
    // create "time integration" class for this microstructure
    global_micro_state().microstaticmap_[microdisnum_] =
        std::make_shared<MultiScale::MicroStatic>(microdisnum_, V0);
    // create a counter of macroscale GP associated with this "time integration" class
    // note that the counter is immediately updated afterwards!
    global_micro_state().microstaticcounter_[microdisnum_] = 0;
  }

  global_micro_state().microstaticcounter_[microdisnum] += 1;
  density_ = (global_micro_state().microstaticmap_[microdisnum_])->density();

  std::string newfilename;
  new_result_file(eleowner, newfilename, initialize_runtime_output_writer);

  // check whether we are using modified Newton as a nonlinear solver
  // on the macroscale or not
  if (Teuchos::getIntegralValue<Solid::NonlinSolTech>(sdyn_micro, "NLNSOL") ==
      Solid::soltech_newtonmod)
    mod_newton_ = true;
  else
    mod_newton_ = false;

  build_stiff_ = true;
}

/// destructor

Mat::MicroMaterialGP::~MicroMaterialGP()
{
  global_micro_state().microstaticcounter_[microdisnum_] -= 1;
  if (global_micro_state().microstaticcounter_[microdisnum_] == 0)
    global_micro_state().microstaticmap_[microdisnum_] = nullptr;
}


/// Read restart

void Mat::MicroMaterialGP::read_restart()
{
  step_ = Global::Problem::instance()->restart();
  global_micro_state().microstaticmap_[microdisnum_]->read_restart(
      step_, dis_, history_data_, restartname_);

  disn_->update(1.0, *dis_, 0.0);
}


/// New resultfile

void Mat::MicroMaterialGP::new_result_file(
    bool eleowner, std::string& newfilename, bool initialize_runtime_output_writer)
{
  // set up micro output
  //
  // Get the macro output prefix and insert element and gauss point
  // identifier. We use the original name here and rely on our (micro)
  // OutputControl object below to act just like the macro (default)
  // OutputControl. In particular we assume that there are always micro and
  // macro control files on restart.
  std::shared_ptr<Core::IO::OutputControl> macrocontrol =
      Global::Problem::instance(0)->output_control_file();
  std::string microprefix = macrocontrol->restart_name();
  std::string micronewprefix = macrocontrol->new_output_file_name();

  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis("structure");

  if (Core::Communication::my_mpi_rank(microdis->get_comm()) == 0)
  {
    // figure out prefix of micro-scale restart files
    restartname_ = new_result_file_path(microprefix);

    // figure out new prefix for micro-scale output files
    newfilename = new_result_file_path(micronewprefix);
  }

  // restart file name and new output file name are sent to supporting procs
  if (Core::Communication::num_mpi_ranks(microdis->get_comm()) > 1)
  {
    {
      // broadcast restartname_ for micro scale
      int length = restartname_.length();
      std::vector<int> name(restartname_.begin(), restartname_.end());
      Core::Communication::broadcast(&length, 1, 0, microdis->get_comm());
      name.resize(length);
      Core::Communication::broadcast(name.data(), length, 0, microdis->get_comm());
      restartname_.assign(name.begin(), name.end());
    }

    {
      // broadcast newfilename for micro scale
      int length = newfilename.length();
      std::vector<int> name(newfilename.begin(), newfilename.end());
      Core::Communication::broadcast(&length, 1, 0, microdis->get_comm());
      name.resize(length);
      Core::Communication::broadcast(name.data(), length, 0, microdis->get_comm());
      newfilename.assign(name.begin(), name.end());
    }
  }

  if (eleowner)
  {
    const int ndim = Global::Problem::instance()->n_dim();
    const int restart = Global::Problem::instance()->restart();
    bool adaptname = true;
    // in case of restart, the new output file name is already adapted
    if (restart) adaptname = false;

    micro_output_control_.emplace(microdis->get_comm(), "Structure",
        microproblem->spatial_approximation_type(), "micro-input-file-not-known", restartname_,
        newfilename, ndim, restart, macrocontrol->file_steps(),
        Global::Problem::instance()->io_params().get<bool>("OUTPUT_BIN"), adaptname);

    // initialize writer for restart output
    micro_output_ = std::make_shared<Core::IO::DiscretizationWriter>(
        *microdis, *micro_output_control_, microproblem->spatial_approximation_type());
    micro_output_->write_mesh(step_, time_);

    if (initialize_runtime_output_writer)
    {
      micro_visualization_writer_ =
          std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
              microdis, Core::IO::visualization_parameters_factory(
                            Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                            *micro_output_control_, time_));
    }
  }
}

std::string Mat::MicroMaterialGP::new_result_file_path(const std::string& newprefix) const
{
  std::string newfilename;

  // create path from string to extract only filename prefix
  const std::filesystem::path path(newprefix);
  const std::string newfileprefix = path.filename().string();

  const size_t posn = newfileprefix.rfind('-');
  if (posn != std::string::npos)
  {
    std::string number = newfileprefix.substr(posn + 1);
    std::string prefix = newfileprefix.substr(0, posn);

    // recombine path and file
    const std::filesystem::path parent_path(path.parent_path());
    const std::filesystem::path filen_name(prefix);
    const std::filesystem::path recombined_path = parent_path / filen_name;

    std::ostringstream s;
    s << recombined_path.string() << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp" << gp_
      << "-" << number;
    newfilename = s.str();
  }
  else
  {
    std::ostringstream s;
    s << newprefix << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp" << gp_;
    newfilename = s.str();
  }
  return newfilename;
}

/// Post setup routine which will be called after the end of the setup
void Mat::MicroMaterialGP::post_setup()
{
  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis("structure");

  if (Core::Communication::my_mpi_rank(microdis->get_comm()) == 0)
  {
    step_ = Global::Problem::instance()->restart();
    if (step_ > 0)
    {
      std::shared_ptr<MultiScale::MicroStatic> microstatic =
          global_micro_state().microstaticmap_[microdisnum_];
      time_ = microstatic->get_time_to_step(step_, restartname_);
    }
    else
    {
      time_ = 0.0;
    }
  }

  Core::Communication::broadcast(&step_, 1, 0, microdis->get_comm());
  Core::Communication::broadcast(&time_, 1, 0, microdis->get_comm());

  stepn_ = step_ + 1;
  timen_ = time_ + dt_;
}

void Mat::MicroMaterialGP::extract_and_store_history_data()
{
  std::shared_ptr<Core::FE::Discretization> discret =
      (Global::Problem::instance(microdisnum_))->get_dis("structure");

  for (auto ele : discret->my_col_element_range())
  {
    auto* actele = ele.user_element();
    // get the solid evaluator which holds potential internal variables
    const Discret::Elements::SolidCalcVariant<3>& solid_evaluator =
        dynamic_cast<Discret::Elements::Solid<3>*>(actele)->get_solid_element_evaluator();

    // pack evaluator and material with its potential internal variables
    Core::Communication::PackBuffer data;
    Discret::Elements::pack(solid_evaluator, data);
    actele->material()->pack(data);
    history_data_[actele->id()] = data();
  }
}

void Mat::MicroMaterialGP::fill_history_data_into_elements()
{
  if (!history_data_.empty())
  {
    std::shared_ptr<Core::FE::Discretization> discret =
        (Global::Problem::instance(microdisnum_))->get_dis("structure");

    for (auto ele : discret->my_col_element_range())
    {
      auto* actele = ele.user_element();
      // get the solid evaluator
      Discret::Elements::SolidCalcVariant<3>& solid_evaluator =
          dynamic_cast<Discret::Elements::Solid<3>*>(actele)->get_solid_element_evaluator();

      // unpack all potential internal variables
      Core::Communication::UnpackBuffer buffer(history_data_[actele->id()]);
      Discret::Elements::unpack(solid_evaluator, buffer);
      actele->material()->unpack(buffer);
    }
  }
}

void Mat::MicroMaterialGP::perform_micro_simulation(const Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat)
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set latest displacements and internal history data
  microstatic->set_state(dis_, disn_, stress_data_node_postprocessed_,
      stress_data_element_postprocessed_, strain_data_node_postprocessed_,
      strain_data_element_postprocessed_, plstrain_, nullptr);
  fill_history_data_into_elements();

  // set current time, time step size and step number
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);

  microstatic->predictor(defgrd);
  microstatic->full_newton();
  microstatic->static_homogenization(stress, cmat, defgrd, mod_newton_, build_stiff_);

  // store matrix for output
  macro_cmat_output_ = std::make_shared<Core::LinAlg::Matrix<6, 6>>(*cmat);
  // save current history data of micro scale elements
  extract_and_store_history_data();

  // clear displacements on micro scale
  microstatic->clear_state();
}

void Mat::MicroMaterialGP::update()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  time_ = timen_;
  timen_ += dt_;
  step_ = stepn_;
  stepn_++;

  dis_->update(1.0, *disn_, 0.0);

  microstatic->set_state(dis_, disn_, stress_data_node_postprocessed_,
      stress_data_element_postprocessed_, strain_data_node_postprocessed_,
      strain_data_element_postprocessed_, plstrain_, nullptr);
  fill_history_data_into_elements();
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);
  // update internal variables on micro scale
  microstatic->update_step_element();
  extract_and_store_history_data();

  // in case of modified Newton, the stiffness matrix needs to be rebuilt at
  // the beginning of the new time step
  build_stiff_ = true;
}

void Mat::MicroMaterialGP::runtime_pre_output_step_state()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  microstatic->set_state(dis_, disn_, stress_data_node_postprocessed_,
      stress_data_element_postprocessed_, strain_data_node_postprocessed_,
      strain_data_element_postprocessed_, plstrain_, nullptr);
  fill_history_data_into_elements();
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);
  microstatic->runtime_pre_output_step_state();
}

void Mat::MicroMaterialGP::runtime_output_step_state_microscale(
    const std::pair<double, int>& output_time_and_step, const std::string& section_name)
{
  // select corresponding "time integration class" for this microstructure
  const std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and internal history data of latest converged state
  microstatic->set_state(dis_, disn_, stress_data_node_postprocessed_,
      stress_data_element_postprocessed_, strain_data_node_postprocessed_,
      strain_data_element_postprocessed_, plstrain_, macro_cmat_output_);
  microstatic->runtime_output_step_state_microscale(
      micro_visualization_writer_, output_time_and_step, section_name);

  stress_data_node_postprocessed_ = nullptr;
  stress_data_element_postprocessed_ = nullptr;
  strain_data_node_postprocessed_ = nullptr;
  strain_data_element_postprocessed_ = nullptr;
  plstrain_ = nullptr;
  macro_cmat_output_ = nullptr;
}

void Mat::MicroMaterialGP::write_restart()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and history data of last step
  microstatic->set_state(dis_, disn_, stress_data_node_postprocessed_,
      stress_data_element_postprocessed_, strain_data_node_postprocessed_,
      strain_data_element_postprocessed_, plstrain_, nullptr);
  microstatic->write_restart(micro_output_, time_, step_, dt_);

  // add element history data to restart output
  if (!history_data_.empty())
  {
    std::shared_ptr<Core::FE::Discretization> discret =
        (Global::Problem::instance(microdisnum_))->get_dis("structure");

    Core::Communication::PackBuffer data;
    for (auto ele : discret->my_row_element_range())
    {
      add_to_pack(data, history_data_[ele.global_id()]);
    }
    micro_output_->write_vector(
        "history_data", data(), *discret->element_row_map(), Core::IO::VectorType::elementvector);
  }

  // we don't need these containers anymore
  stress_data_node_postprocessed_ = nullptr;
  stress_data_element_postprocessed_ = nullptr;
  strain_data_node_postprocessed_ = nullptr;
  strain_data_element_postprocessed_ = nullptr;
  plstrain_ = nullptr;
}

FOUR_C_NAMESPACE_CLOSE
