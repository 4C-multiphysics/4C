// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_algorithm.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fsi_str_model_evaluator_partitioned.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
// Note: The order of calling the three BaseAlgorithm-constructors is
// important here! In here control file entries are written. And these entries
// define the order in which the filters handle the Discretizations, which in
// turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*/
FSI::Algorithm::Algorithm(MPI_Comm comm)
    : AlgorithmBase(comm, Global::Problem::instance()->fsi_dynamic_params()),
      adapterbase_ptr_(nullptr),
      use_old_structure_(false)
{
  // empty constructor
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::setup()
{
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis("structure");

  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();

  // access the fsi dynamic params
  Teuchos::ParameterList& fsidyn =
      const_cast<Teuchos::ParameterList&>(Global::Problem::instance()->fsi_dynamic_params());

  // build and register fsi model evaluator
  std::shared_ptr<Solid::ModelEvaluator::Generic> fsi_model_ptr =
      std::make_shared<Solid::ModelEvaluator::PartitionedFSI>();

  // todo FIX THIS !!!!
  // Decide whether to use old structural time integration or new structural time integration.
  // This should be removed as soon as possible! We need to clean up crack fsi first!
  // Also all structural elements need to be adapted first!
  // Then, we can switch the 3 remaining fsi tests using the old time integration to the new one,
  // i.e.: crfsi_inclDomain_suppFluid_interfaceBuild
  //       fsi_dc3D_part_ait_ga_ost_xwall
  //       fsi_ow3D_mtr_drt
  // build structure
  if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(sdyn, "INT_STRATEGY") ==
      Inpar::Solid::IntegrationStrategy::int_standard)
  {
    adapterbase_ptr_ = Adapter::build_structure_algorithm(sdyn);
    adapterbase_ptr_->init(fsidyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
    adapterbase_ptr_->register_model_evaluator("Partitioned Coupling Model", fsi_model_ptr);
    adapterbase_ptr_->setup();
    structure_ = std::dynamic_pointer_cast<Adapter::FSIStructureWrapper>(
        adapterbase_ptr_->structure_field());

    // set pointer in FSIStructureWrapper
    structure_->set_model_evaluator_ptr(
        std::dynamic_pointer_cast<Solid::ModelEvaluator::PartitionedFSI>(fsi_model_ptr));

    if (structure_ == nullptr)
      FOUR_C_THROW("cast from Adapter::Structure to Adapter::FSIStructureWrapper failed");
  }
  else if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(sdyn, "INT_STRATEGY") ==
           Inpar::Solid::IntegrationStrategy::int_old)  // todo this is the part that should be
                                                        // removed !
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n"
                << " USING OLD STRUCTURAL TIME INEGRATION! FIX THIS! THIS IS ONLY SUPPOSED TO BE "
                   "TEMPORARY!"
                   "\n"
                << std::endl;

    Adapter::StructureBaseAlgorithm structure(Global::Problem::instance()->fsi_dynamic_params(),
        const_cast<Teuchos::ParameterList&>(sdyn), structdis);
    structure_ =
        std::dynamic_pointer_cast<Adapter::FSIStructureWrapper>(structure.structure_field());
    structure_->setup();

    if (structure_ == nullptr)
      FOUR_C_THROW("cast from Adapter::Structure to Adapter::FSIStructureWrapper failed");

    use_old_structure_ = true;
  }
  else
    FOUR_C_THROW(
        "Unknown time integration requested!\n"
        "Set parameter INT_STRATEGY to Standard in ---STRUCTURAL DYNAMIC section!\n"
        "If you want to use yet unsupported elements or you want to do crack simulation,\n"
        "set INT_STRATEGY to Old in ---STRUCTURAL DYNAMIC section!");

  Adapter::FluidMovingBoundaryBaseAlgorithm MBFluidbase(
      Global::Problem::instance()->fsi_dynamic_params(), "FSICoupling");
  fluid_ = MBFluidbase.mb_fluid_field();

  coupsf_ = std::make_shared<Coupling::Adapter::Coupling>();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::post_setup()
{
  // call post_setup routine of the structural field
  structure_->post_setup();
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::read_restart(int step)
{
  structure_field()->read_restart(step);
  double time = mb_fluid_field()->read_restart(step);
  set_time_step(time, step);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::prepare_time_step()
{
  increment_time_and_step();

  print_header();

  structure_field()->prepare_time_step();
  mb_fluid_field()->prepare_time_step();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::update()
{
  structure_field()->update();
  mb_fluid_field()->update();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::prepare_output(bool force_prepare)
{
  structure_field()->prepare_output(force_prepare);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::output()
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the Discretizations, which in turn defines the dof number ordering of the
  // Discretizations.
  structure_field()->output();
  mb_fluid_field()->output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Algorithm::struct_to_fluid(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  return coupsf_->master_to_slave(*iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Algorithm::fluid_to_struct(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  return coupsf_->slave_to_master(*iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Coupling::Adapter::Coupling& FSI::Algorithm::structure_fluid_coupling() { return *coupsf_; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Coupling::Adapter::Coupling& FSI::Algorithm::structure_fluid_coupling() const
{
  return *coupsf_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Algorithm::struct_to_fluid(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsf_->master_to_slave(*iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Algorithm::fluid_to_struct(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsf_->slave_to_master(*iv);
}

FOUR_C_NAMESPACE_CLOSE
