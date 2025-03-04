// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_micromaterialgp_static.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_stru_multi_microstatic.hpp"
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

Mat::MicroMaterialGP::MicroMaterialGP(
    const int gp, const int ele_ID, const bool eleowner, const int microdisnum, const double V0)
    : gp_(gp), ele_id_(ele_ID), microdisnum_(microdisnum)
{
  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis("structure");
  dis_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  disn_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  lastalpha_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
  oldalpha_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
  oldfeas_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
  old_kaainv_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();
  old_kda_ = std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();

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

  // create and initialize "empty" EAS history map (if necessary)
  eas_init();

  std::string newfilename;
  new_result_file(eleowner, newfilename);

  // check whether we are using modified Newton as a nonlinear solver
  // on the macroscale or not
  if (Teuchos::getIntegralValue<Inpar::Solid::NonlinSolTech>(sdyn_micro, "NLNSOL") ==
      Inpar::Solid::soltech_newtonmod)
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
      step_, dis_, lastalpha_, restartname_);

  *oldalpha_ = *lastalpha_;

  disn_->Update(1.0, *dis_, 0.0);
}


/// New resultfile

void Mat::MicroMaterialGP::new_result_file(bool eleowner, std::string& newfilename)
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

    std::shared_ptr<Core::IO::OutputControl> microcontrol =
        std::make_shared<Core::IO::OutputControl>(microdis->get_comm(), "Structure",
            microproblem->spatial_approximation_type(), "micro-input-file-not-known", restartname_,
            newfilename, ndim, restart, macrocontrol->file_steps(),
            Global::Problem::instance()->io_params().get<bool>("OUTPUT_BIN"), adaptname);

    micro_output_ = std::make_shared<Core::IO::DiscretizationWriter>(
        microdis, microcontrol, microproblem->spatial_approximation_type());
    micro_output_->set_output(microcontrol);

    micro_output_->write_mesh(step_, time_);
  }
}

std::string Mat::MicroMaterialGP::new_result_file_path(const std::string& newprefix)
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
    s << recombined_path.string() << "_el" << ele_id_ << "_gp" << gp_ << "-" << number;
    newfilename = s.str();
  }
  else
  {
    std::ostringstream s;
    s << newprefix << "_el" << ele_id_ << "_gp" << gp_;
    newfilename = s.str();
  }
  return newfilename;
}

void Mat::MicroMaterialGP::eas_init()
{
  std::shared_ptr<Core::FE::Discretization> discret =
      (Global::Problem::instance(microdisnum_))->get_dis("structure");

  for (int lid = 0; lid < discret->element_row_map()->NumMyElements(); ++lid)
  {
    Core::Elements::Element* actele = discret->l_row_element(lid);

    if (actele->element_type() == Discret::Elements::SoHex8Type::instance())
    {
      // create the parameters for the discretization
      Teuchos::ParameterList p;
      // action for elements
      p.set("action", "multi_eas_init");
      p.set("lastalpha", lastalpha_);
      p.set("oldalpha", oldalpha_);
      p.set("oldfeas", oldfeas_);
      p.set("oldKaainv", old_kaainv_);
      p.set("oldKda", old_kda_);

      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;
      std::vector<int> lm;

      actele->evaluate(p, *discret, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
    }
  }

  return;
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

/// perform microscale simulation
void Mat::MicroMaterialGP::perform_micro_simulation(Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat)
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);

  // set current time, time step size and step number
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);

  microstatic->predictor(defgrd);
  microstatic->full_newton();
  microstatic->static_homogenization(stress, cmat, defgrd, mod_newton_, build_stiff_);

  // note that it is not necessary to save displacements and EAS data
  // explicitly since we dealt with std::shared_ptr's -> any update in class
  // microstatic and the elements, respectively, inherently updates the
  // micromaterialgp_static data!

  // clear displacements in MicroStruGenAlpha for next usage
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

  dis_->Update(1.0, *disn_, 0.0);

  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis("structure");
  const Epetra_Map* elemap = microdis->element_row_map();

  for (int i = 0; i < elemap->NumMyElements(); ++i) (*lastalpha_)[i] = (*oldalpha_)[i];

  // in case of modified Newton, the stiffness matrix needs to be rebuilt at
  // the beginning of the new time step
  build_stiff_ = true;
}


void Mat::MicroMaterialGP::prepare_output()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  stress_ = std::make_shared<std::vector<char>>();
  strain_ = std::make_shared<std::vector<char>>();
  plstrain_ = std::make_shared<std::vector<char>>();

  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);
  microstatic->prepare_output();
}

void Mat::MicroMaterialGP::output_step_state_microscale()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->output(*micro_output_, time_, step_, dt_);

  // we don't need these containers anymore
  stress_ = nullptr;
  strain_ = nullptr;
  plstrain_ = nullptr;
}

void Mat::MicroMaterialGP::runtime_output_step_state_microscale(
    const std::pair<double, int>& output_time_and_step, const std::string& section_name)
{
  // select corresponding "time integration class" for this microstructure
  const std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->runtime_output(output_time_and_step, section_name);

  stress_ = nullptr;
  strain_ = nullptr;
  plstrain_ = nullptr;
}

void Mat::MicroMaterialGP::write_restart()
{
  // select corresponding "time integration class" for this microstructure
  std::shared_ptr<MultiScale::MicroStatic> microstatic =
      global_micro_state().microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->write_restart(micro_output_, time_, step_, dt_);

  // we don't need these containers anymore
  stress_ = nullptr;
  strain_ = nullptr;
  plstrain_ = nullptr;
}

FOUR_C_NAMESPACE_CLOSE
