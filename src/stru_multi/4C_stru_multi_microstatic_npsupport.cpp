// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_stru_multi_microstatic_npsupport.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_utils_enum.hpp"

#include <hdf5.h>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | "timeloop" for supporting procs                          ghamm 05/12 |
 *----------------------------------------------------------------------*/
void MultiScale::np_support_drt()
{
  // info:
  // Macro processors run on their macro problem (distributed or not)
  // and during run time supporting processors are available.
  // Dependent on the number of processors on the macro scale that need
  // support, the supporting procs are distributed. The distribution is performed
  // in Global::Problem::read_micro_fields() where a sub communicator is created
  // that contains one master proc and several supporting procs. The master
  // proc has always MyPID() = 0 in the subcomm. That's why all broadcast commands
  // send from proc 0 in subcomm.
  // Supporting procs always wait in front of Core::Communication::broadcast(task, 2, 0, "subcomm);"
  // until a master proc reaches the same broadcast command. Then all procs
  // in this subcomm go into the same routine. We need one dummy material that
  // is specified in the input file of the supporting procs. It is only used to
  // call the corresponding routines. All necessary information comes from the
  // master proc. Supporting processors always start their work in the routines
  // in this file.

  // one dummy micro material is specified in the supporting input file
  std::map<int, std::shared_ptr<Mat::MicroMaterial>> dummymaterials;

  // this call is needed in order to increment the unique ids that are distributed
  // by HDF5; the macro procs call output->write_mesh(0, 0.0) in Adapter::Structure
  const int someUniqueNumber = Core::Communication::my_mpi_rank(
      Global::Problem::instance(0)->get_communicators()->global_comm());
  std::string uniqueDummyName = &"dummyHDF5file_p"[someUniqueNumber];
  H5Fcreate(uniqueDummyName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // get sub communicator including the master proc
  MPI_Comm subcomm = Global::Problem::instance(0)->get_communicators()->sub_comm();

  // start std::endless loop
  while (true)
  {
    // receive what to do and which element is intended to work
    std::array<int, 2> task = {-1, -1};
    Core::Communication::broadcast(task.data(), 2, 0, subcomm);
    MultiScale::MicromaterialNestedParallelismAction whattodo =
        static_cast<MicromaterialNestedParallelismAction>(task[0]);
    int eleID = task[1];

    // every element needs one micromaterial
    if (dummymaterials[eleID] == nullptr)
      dummymaterials[eleID] = std::static_pointer_cast<Mat::MicroMaterial>(Mat::factory(1));

    // check what is the next task of the supporting procs
    switch (whattodo)
    {
      case MultiScale::MicromaterialNestedParallelismAction::evaluate:
      {
        // receive data from the master proc
        int tag = 0;
        Epetra_Map oldmap(1, 0, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
        Epetra_Map newmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
        // create an exporter object that will figure out the communication pattern
        Core::Communication::Exporter exporter(oldmap, newmap, subcomm);
        std::map<int, std::shared_ptr<MultiScale::MicroStaticParObject>> condnamemap;
        exporter.do_export<MultiScale::MicroStaticParObject>(condnamemap);

        const auto* micro_data = condnamemap[0]->get_micro_static_data_ptr();
        // extract received data from the container
        const Core::LinAlg::SerialDenseMatrix* defgrdcopy = &micro_data->defgrd_;
        Core::LinAlg::Matrix<3, 3> defgrd(
            *(const_cast<Core::LinAlg::SerialDenseMatrix*>(defgrdcopy)), true);
        const Core::LinAlg::SerialDenseMatrix* cmatcopy = &micro_data->cmat_;
        Core::LinAlg::Matrix<6, 6> cmat(
            *(const_cast<Core::LinAlg::SerialDenseMatrix*>(cmatcopy)), true);
        const Core::LinAlg::SerialDenseMatrix* stresscopy = &micro_data->stress_;
        Core::LinAlg::Matrix<6, 1> stress(
            *(const_cast<Core::LinAlg::SerialDenseMatrix*>(stresscopy)), true);
        int gp = micro_data->gp_;
        int microdisnum = micro_data->microdisnum_;
        double V0 = micro_data->V0_;
        bool eleowner = (bool)micro_data->eleowner_;

        // dummy material is used to evaluate the micro material
        dummymaterials[eleID]->evaluate(
            &defgrd, &cmat, &stress, gp, eleID, microdisnum, V0, eleowner);
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::prepare_output:
      {
        // dummy material is used to prepare the output of the micro material
        dummymaterials[eleID]->prepare_output();
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::update:
      {
        // dummy material is used to update the micro material
        dummymaterials[eleID]->update();
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::output_step_state:
      {
        // dummy material is used to output the micro material
        dummymaterials[eleID]->output_step_state();
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::write_restart:
      {
        // dummy material is used to write restart of the micro material
        dummymaterials[eleID]->write_restart();
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::read_restart:
      {
        // receive data from the master proc for restart
        int tag = 0;
        Epetra_Map oldmap(1, 0, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
        Epetra_Map newmap(1, 1, &tag, 0, Core::Communication::as_epetra_comm(subcomm));
        // create an exporter object that will figure out the communication pattern
        Core::Communication::Exporter exporter(oldmap, newmap, subcomm);
        std::map<int, std::shared_ptr<MultiScale::MicroStaticParObject>> condnamemap;
        exporter.do_export<MultiScale::MicroStaticParObject>(condnamemap);
        const auto* micro_data = condnamemap[0]->get_micro_static_data_ptr();

        // extract received data from the container
        int gp = micro_data->gp_;
        int microdisnum = micro_data->microdisnum_;
        double V0 = micro_data->V0_;
        bool eleowner = micro_data->eleowner_;

        // new dummy material is created if necessary
        if (dummymaterials[eleID] == nullptr)
          dummymaterials[eleID] = std::static_pointer_cast<Mat::MicroMaterial>(Mat::factory(1));

        // dummy material is used to restart the micro material
        dummymaterials[eleID]->read_restart(gp, eleID, eleowner, microdisnum, V0);
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::post_setup:
      {
        // dummy material is used to call post setup on the micro material
        dummymaterials[eleID]->post_setup();
        break;
      }
      case MultiScale::MicromaterialNestedParallelismAction::stop_multiscale:
      {
        // end of simulation after deleting all dummy materials
        dummymaterials.clear();
        return;
      }
      default:
        FOUR_C_THROW("Supporting processors do not know what to do ({})!", whattodo);
        break;
    }

  }  // end while(true) loop
}

FOUR_C_NAMESPACE_CLOSE
