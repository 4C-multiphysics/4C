/*----------------------------------------------------------------------*/
/*! \file

\brief Implements the constraint enforcement technique of a penalty approach (Mortar and GPTS) (for
fluid-beam interaction)

\level 2

*----------------------------------------------------------------------*/

#include "4C_fbi_constraintenforcer_penalty.hpp"

#include "4C_adapter_fld_fbi_movingboundary.hpp"
#include "4C_adapter_str_fbiwrapper.hpp"
#include "4C_fbi_adapter_constraintbridge_penalty.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_constraintenforcer.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIPenaltyConstraintenforcer::setup(
    Teuchos::RCP<Adapter::FSIStructureWrapper> structure,
    Teuchos::RCP<Adapter::FluidMovingBoundary> fluid)
{
  Adapter::FBIConstraintenforcer::setup(structure, fluid);
  std::ofstream log;
  if ((get_discretizations()[1]->get_comm().MyPID() == 0) &&
      (bridge()
              ->get_params()
              ->get_visualization_ouput_params_ptr()
              ->get_constraint_violation_output_flag()))
  {
    std::string s = Global::Problem::instance()->output_control_file()->file_name();
    s.append(".penalty");
    log.open(s.c_str(), std::ofstream::out);
    log << "Time \t Step \t ViolationNorm \t FluidViolationNorm \t StructureViolationNorm"
        << std::endl;
    log.close();
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::SparseOperator>
Adapter::FBIPenaltyConstraintenforcer::assemble_fluid_coupling_matrix() const
{
  // Get coupling contributions to the fluid stiffness matrix

  return bridge()->get_cff();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Core::LinAlg::SparseMatrix>
Adapter::FBIPenaltyConstraintenforcer::assemble_structure_coupling_matrix() const
{
  // For the classical partitioned algorithm we do not have any contributions to the stiffness
  // matrix of the structure field
  return Teuchos::null;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
Adapter::FBIPenaltyConstraintenforcer::assemble_fluid_coupling_residual() const
{
  Teuchos::rcp_dynamic_cast<Adapter::FBIConstraintBridgePenalty>(bridge(), true)
      ->scale_penalty_fluid_contributions();
  // Get the force acting on the fluid field, scale it with -1 to get the
  // correct direction
  Teuchos::RCP<Epetra_Vector> f =
      Teuchos::rcp(new Epetra_Vector((bridge()->get_fluid_coupling_residual())->Map()));
  f->Update(-1.0, *(bridge()->get_fluid_coupling_residual()), 0.0);
  return f;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
Adapter::FBIPenaltyConstraintenforcer::assemble_structure_coupling_residual() const
{
  Teuchos::rcp_dynamic_cast<Adapter::FBIConstraintBridgePenalty>(bridge(), true)
      ->scale_penalty_structure_contributions();
  // Get the force acting on the structure field, scale it with the penalty factor and -1 to get the
  // correct direction
  Teuchos::RCP<Epetra_Vector> f =
      Teuchos::rcp(new Epetra_Vector(bridge()->get_structure_coupling_residual()->Map()));
  f->Update(-1.0, *(bridge()->get_structure_coupling_residual()), 0.0);

  return f;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIPenaltyConstraintenforcer::prepare_fluid_solve()
{
  bridge()->prepare_fluid_solve();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIPenaltyConstraintenforcer::output(double time, int step)
{
  print_violation(time, step);
}
/*----------------------------------------------------------------------*/

void Adapter::FBIPenaltyConstraintenforcer::print_violation(double time, int step)
{
  if (bridge()
          ->get_params()
          ->get_visualization_ouput_params_ptr()
          ->get_constraint_violation_output_flag())
  {
    double penalty_parameter = bridge()->get_params()->get_penalty_parameter();

    Teuchos::RCP<Epetra_Vector> violation = Core::LinAlg::create_vector(
        Teuchos::rcp_dynamic_cast<Adapter::FBIFluidMB>(get_fluid(), true)->velnp()->Map());

    int err =
        Teuchos::rcp_dynamic_cast<const Adapter::FBIConstraintBridgePenalty>(get_bridge(), true)
            ->get_cff()
            ->multiply(false,
                *(Teuchos::rcp_dynamic_cast<Adapter::FBIFluidMB>(get_fluid(), true)->velnp()),
                *violation);

    if (err != 0) FOUR_C_THROW(" Matrix vector product threw error code %i ", err);

    err = violation->Update(1.0, *assemble_fluid_coupling_residual(), -1.0);
    if (err != 0) FOUR_C_THROW(" Epetra_Vector update threw error code %i ", err);

    double norm = 0.0, normf = 0.0, norms = 0.0, norm_vel = 0.0;

    get_velocity_pressure_splitter()
        ->extract_other_vector(
            Teuchos::rcp_dynamic_cast<Adapter::FBIFluidMB>(get_fluid(), true)->velnp())
        ->MaxValue(&norm_vel);

    violation->MaxValue(&norm);
    if (norm_vel > 1e-15) normf = norm / norm_vel;

    Teuchos::rcp_dynamic_cast<const Adapter::FBIStructureWrapper>(get_structure(), true)
        ->velnp()
        ->MaxValue(&norm_vel);
    if (norm_vel > 1e-15) norms = norm / norm_vel;

    std::ofstream log;
    if (get_discretizations()[1]->get_comm().MyPID() == 0)
    {
      std::string s = Global::Problem::instance()->output_control_file()->file_name();
      s.append(".penalty");
      log.open(s.c_str(), std::ofstream::app);
      log << time << "\t" << step << "\t" << norm / penalty_parameter << "\t"
          << normf / penalty_parameter << "\t" << norms / penalty_parameter << std::endl;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
