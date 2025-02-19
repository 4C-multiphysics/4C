// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_svd.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_micromaterialgp_static.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_stru_multi_microstatic.hpp"

FOUR_C_NAMESPACE_OPEN



// This function has to be separated from the remainder of the
// MicroMaterial class. MicroMaterialGP is NOT a member of
// FILTER_OBJECTS hence the MicroMaterial::Evaluate function that
// builds the connection to MicroMaterialGP is not either. In
// post_evaluation.cpp this function is defined to content the
// compiler. If during postprocessing the MicroMaterial::Evaluate
// function should be called, an error is invoked.
//
// -> see also Makefile.objects and setup-objects.sh
//
// In case of any changes of the function prototype make sure that the
// corresponding prototype in src/filter_common/filter_evaluation.cpp is adapted, too!!

// evaluate for master procs
void Mat::MicroMaterial::evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  if (eleGID == -1) FOUR_C_THROW("no element ID provided in material");

  Core::LinAlg::Matrix<3, 3>* defgrd_enh = const_cast<Core::LinAlg::Matrix<3, 3>*>(defgrd);

  if (params.get("EASTYPE", "none") != "none")
  {
    // In this case, we have to calculate the "enhanced" deformation gradient
    // from the enhanced GL strains with the help of two polar decompositions

    // First step: determine enhanced material stretch tensor U_enh from C_enh=U_enh^T*U_enh
    // -> get C_enh from enhanced GL strains
    Core::LinAlg::Matrix<3, 3> C_enh;
    for (int i = 0; i < 3; ++i) C_enh(i, i) = 2.0 * (*glstrain)(i) + 1.0;
    // off-diagonal terms are already twice in the Voigt-GLstrain-vector
    C_enh(0, 1) = (*glstrain)(3);
    C_enh(1, 0) = (*glstrain)(3);
    C_enh(1, 2) = (*glstrain)(4);
    C_enh(2, 1) = (*glstrain)(4);
    C_enh(0, 2) = (*glstrain)(5);
    C_enh(2, 0) = (*glstrain)(5);

    // -> polar decomposition of (U^mod)^2
    Core::LinAlg::Matrix<3, 3> Q;
    Core::LinAlg::Matrix<3, 3> S;
    Core::LinAlg::Matrix<3, 3> VT;
    Core::LinAlg::svd<3, 3>(C_enh, Q, S, VT);  // Singular Value Decomposition
    Core::LinAlg::Matrix<3, 3> U_enh;
    Core::LinAlg::Matrix<3, 3> temp;
    for (int i = 0; i < 3; ++i) S(i, i) = sqrt(S(i, i));
    temp.multiply_nn(Q, S);
    U_enh.multiply_nn(temp, VT);

    // Second step: determine rotation tensor R from F (F=R*U)
    // -> polar decomposition of displacement based F
    Core::LinAlg::svd<3, 3>(*(defgrd_enh), Q, S, VT);  // Singular Value Decomposition
    Core::LinAlg::Matrix<3, 3> R;
    R.multiply_nn(Q, VT);

    // Third step: determine "enhanced" deformation gradient (F_enh=R*U_enh)
    defgrd_enh->multiply_nn(R, U_enh);
  }

  // activate microscale material

  int microdisnum = micro_dis_num();
  double V0 = init_vol();
  Global::Problem::instance()->materials()->set_read_from_problem(microdisnum);

  // avoid writing output also for ghosted elements
  const bool eleowner =
      Global::Problem::instance(0)->get_dis("structure")->element_row_map()->MyGID(eleGID);

  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();

  // tell the supporting procs that the micro material will be evaluated
  int task[2] = {
      static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::evaluate), eleGID};
  Core::Communication::broadcast(task, 2, 0, subcomm);

  // container is filled with data for supporting procs
  std::map<int, std::shared_ptr<MultiScale::MicroStaticParObject>> condnamemap;
  condnamemap[0] = std::make_shared<MultiScale::MicroStaticParObject>();

  const auto convert_to_serial_dense_matrix = [](const auto& matrix)
  {
    using MatrixType = std::decay_t<decltype(matrix)>;
    constexpr int n_rows = MatrixType::num_rows();
    constexpr int n_cols = MatrixType::num_cols();
    Core::LinAlg::SerialDenseMatrix data(n_rows, n_cols);
    for (int i = 0; i < n_rows; i++)
      for (int j = 0; j < n_cols; j++) data(i, j) = matrix(i, j);
    return data;
  };

  MultiScale::MicroStaticParObject::MicroStaticData microdata{};
  microdata.defgrd_ = convert_to_serial_dense_matrix(*defgrd_enh);
  microdata.cmat_ = convert_to_serial_dense_matrix(*cmat);
  microdata.stress_ = convert_to_serial_dense_matrix(*stress);
  microdata.gp_ = gp;
  microdata.microdisnum_ = microdisnum;
  microdata.V0_ = V0;
  microdata.eleowner_ = eleowner;
  condnamemap[0]->set_micro_static_data(microdata);

  // maps are created and data is broadcast to the supporting procs
  int tag = 0;
  Epetra_Map oldmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
  Epetra_Map newmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
  Core::Communication::Exporter exporter(oldmap, newmap, subcomm);
  exporter.do_export<MultiScale::MicroStaticParObject>(condnamemap);

  // standard evaluation of the micro material
  if (matgp_.find(gp) == matgp_.end())
  {
    matgp_[gp] = std::make_shared<MicroMaterialGP>(gp, eleGID, eleowner, microdisnum, V0);

    /// save density of this micromaterial
    /// -> since we can assign only one material per element, all Gauss points have
    /// the same density -> arbitrarily ask micromaterialgp at gp=0
    if (gp == 0)
    {
      std::shared_ptr<MicroMaterialGP> actmicromatgp = matgp_[gp];
      density_ = actmicromatgp->density();
    }
  }

  std::shared_ptr<MicroMaterialGP> actmicromatgp = matgp_[gp];

  // perform microscale simulation and homogenization (if fint and stiff/mass or stress calculation
  // is required)
  actmicromatgp->perform_micro_simulation(defgrd_enh, stress, cmat);

  // reactivate macroscale material
  Global::Problem::instance()->materials()->reset_read_from_problem();
}

double Mat::MicroMaterial::density() const { return density_; }

void Mat::MicroMaterial::post_setup()
{
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will call post_setup
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::post_setup), eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->post_setup();
  }
}

// evaluate for supporting procs
void Mat::MicroMaterial::evaluate(Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 6>* cmat, Core::LinAlg::Matrix<6, 1>* stress, const int gp,
    const int ele_ID, const int microdisnum, double V0, bool eleowner)
{
  Global::Problem::instance()->materials()->set_read_from_problem(microdisnum);

  if (matgp_.find(gp) == matgp_.end())
  {
    matgp_[gp] = std::make_shared<MicroMaterialGP>(gp, ele_ID, eleowner, microdisnum, V0);
  }

  std::shared_ptr<MicroMaterialGP> actmicromatgp = matgp_[gp];

  // perform microscale simulation and homogenization (if fint and stiff/mass or stress calculation
  // is required)
  actmicromatgp->perform_micro_simulation(defgrd, stress, cmat);

  // reactivate macroscale material
  Global::Problem::instance()->materials()->reset_read_from_problem();
}

// update for all procs
void Mat::MicroMaterial::update()
{
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be evaluated for the element with id
    // eleID
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::update), eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->update();
  }
}

// prepare output for all procs
void Mat::MicroMaterial::prepare_output()
{
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be prepared for output
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::prepare_output), eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->prepare_output();
  }
}
// prepare output for all procs
void Mat::MicroMaterial::prepare_runtime_output() const
{
  if (params_->runtime_output_option_ == PAR::MicroMaterial::RuntimeOutputOption::none)
  {
    return;
  }
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be prepared for output
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::prepare_output), eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->prepare_output();

    if (params_->runtime_output_option_ == PAR::MicroMaterial::RuntimeOutputOption::gp1) break;
  }
}

void Mat::MicroMaterial::runtime_output_step_state(
    std::pair<double, int> output_time_and_step) const
{
  if (params_->runtime_output_option_ == PAR::MicroMaterial::RuntimeOutputOption::none)
  {
    return;
  }
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be output
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::output_step_state),
        eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::string section_name = "rve_elem_" + std::to_string(micromatgp.second->ele_id()) + "_gp_ " +
                               std::to_string(micromatgp.first);
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->runtime_output_step_state_microscale(output_time_and_step, section_name);

    if (params_->runtime_output_option_ == PAR::MicroMaterial::RuntimeOutputOption::gp1) break;
  }
}

// output for all procs
void Mat::MicroMaterial::output_step_state()
{
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be output
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::output_step_state),
        eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->output_step_state_microscale();
  }
}

// output for all procs
void Mat::MicroMaterial::write_restart()
{
  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();
  if (Core::Communication::my_mpi_rank(subcomm) == 0)
  {
    // tell the supporting procs that the micro material will be output
    int eleID = matgp_.begin()->second->ele_id();
    int task[2] = {
        static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::write_restart), eleID};
    Core::Communication::broadcast(task, 2, 0, subcomm);
  }

  for (const auto& micromatgp : matgp_)
  {
    std::shared_ptr<MicroMaterialGP> actmicromatgp = micromatgp.second;
    actmicromatgp->write_restart();
  }
}

// read restart for master procs
void Mat::MicroMaterial::read_restart(const int gp, const int eleID, const bool eleowner)
{
  int microdisnum = micro_dis_num();
  double V0 = init_vol();

  // get sub communicator including the supporting procs
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();

  // tell the supporting procs that the micro material will restart
  int task[2] = {
      static_cast<int>(MultiScale::MicromaterialNestedParallelismAction::read_restart), eleID};
  Core::Communication::broadcast(task, 2, 0, subcomm);

  // container is filled with data for supporting procs
  std::map<int, std::shared_ptr<MultiScale::MicroStaticParObject>> condnamemap;
  condnamemap[0] = std::make_shared<MultiScale::MicroStaticParObject>();

  MultiScale::MicroStaticParObject::MicroStaticData microdata{};
  microdata.gp_ = gp;
  microdata.microdisnum_ = microdisnum;
  microdata.V0_ = V0;
  microdata.eleowner_ = eleowner;
  condnamemap[0]->set_micro_static_data(microdata);

  // maps are created and data is broadcast to the supporting procs
  int tag = 0;
  Epetra_Map oldmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
  Epetra_Map newmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
  Core::Communication::Exporter exporter(oldmap, newmap, subcomm);
  exporter.do_export<MultiScale::MicroStaticParObject>(condnamemap);

  if (matgp_.find(gp) == matgp_.end())
  {
    matgp_[gp] = std::make_shared<MicroMaterialGP>(gp, eleID, eleowner, microdisnum, V0);
  }

  std::shared_ptr<MicroMaterialGP> actmicromatgp = matgp_[gp];
  actmicromatgp->read_restart();
}


// read restart for supporting procs
void Mat::MicroMaterial::read_restart(
    const int gp, const int eleID, const bool eleowner, int microdisnum, double V0)
{
  if (matgp_.find(gp) == matgp_.end())
  {
    matgp_[gp] = std::make_shared<MicroMaterialGP>(gp, eleID, eleowner, microdisnum, V0);
  }

  std::shared_ptr<MicroMaterialGP> actmicromatgp = matgp_[gp];
  actmicromatgp->read_restart();
}

FOUR_C_NAMESPACE_CLOSE
