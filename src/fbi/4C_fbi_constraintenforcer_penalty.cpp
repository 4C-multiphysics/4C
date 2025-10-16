// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
#include "4C_linalg_vector.hpp"

#include <fstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIPenaltyConstraintenforcer::setup(
    std::shared_ptr<Adapter::FSIStructureWrapper> structure,
    std::shared_ptr<Adapter::FluidMovingBoundary> fluid)
{
  Adapter::FBIConstraintenforcer::setup(structure, fluid);
  std::ofstream log;
  if ((Core::Communication::my_mpi_rank(get_discretizations()[1]->get_comm()) == 0) &&
      (bridge()
              ->get_params()
              ->get_visualization_output_params_ptr()
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
std::shared_ptr<const Core::LinAlg::SparseOperator>
Adapter::FBIPenaltyConstraintenforcer::assemble_fluid_coupling_matrix() const
{
  // Get coupling contributions to the fluid stiffness matrix

  return bridge()->get_cff();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix>
Adapter::FBIPenaltyConstraintenforcer::assemble_structure_coupling_matrix() const
{
  // For the classical partitioned algorithm we do not have any contributions to the stiffness
  // matrix of the structure field
  return nullptr;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIPenaltyConstraintenforcer::assemble_fluid_coupling_residual() const
{
  std::dynamic_pointer_cast<Adapter::FBIConstraintBridgePenalty>(bridge())
      ->scale_penalty_fluid_contributions();
  // Get the force acting on the fluid field, scale it with -1 to get the
  // correct direction
  std::shared_ptr<Core::LinAlg::Vector<double>> f = std::make_shared<Core::LinAlg::Vector<double>>(
      (bridge()->get_fluid_coupling_residual())->get_map());
  f->update(-1.0, *(bridge()->get_fluid_coupling_residual()), 0.0);
  return f;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIPenaltyConstraintenforcer::assemble_structure_coupling_residual() const
{
  std::dynamic_pointer_cast<Adapter::FBIConstraintBridgePenalty>(bridge())
      ->scale_penalty_structure_contributions();
  // Get the force acting on the structure field, scale it with the penalty factor and -1 to get the
  // correct direction
  std::shared_ptr<Core::LinAlg::Vector<double>> f = std::make_shared<Core::LinAlg::Vector<double>>(
      bridge()->get_structure_coupling_residual()->get_map());
  f->update(-1.0, *(bridge()->get_structure_coupling_residual()), 0.0);

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
          ->get_visualization_output_params_ptr()
          ->get_constraint_violation_output_flag())
  {
    double penalty_parameter = bridge()->get_params()->get_penalty_parameter();

    std::shared_ptr<Core::LinAlg::Vector<double>> violation = Core::LinAlg::create_vector(
        std::dynamic_pointer_cast<Adapter::FBIFluidMB>(get_fluid())->velnp()->get_map());

    int err = std::dynamic_pointer_cast<const Adapter::FBIConstraintBridgePenalty>(get_bridge())
                  ->get_cff()
                  ->multiply(false,
                      *(std::dynamic_pointer_cast<Adapter::FBIFluidMB>(get_fluid())->velnp()),
                      *violation);

    if (err != 0) FOUR_C_THROW(" Matrix vector product threw error code {} ", err);

    err = violation->update(1.0, *assemble_fluid_coupling_residual(), -1.0);
    if (err != 0) FOUR_C_THROW(" Core::LinAlg::Vector<double> update threw error code {} ", err);

    double norm = 0.0, normf = 0.0, norms = 0.0, norm_vel = 0.0;

    get_velocity_pressure_splitter()
        ->extract_other_vector(
            *std::dynamic_pointer_cast<Adapter::FBIFluidMB>(get_fluid())->velnp())
        ->max_value(&norm_vel);

    violation->max_value(&norm);
    if (norm_vel > 1e-15) normf = norm / norm_vel;

    std::dynamic_pointer_cast<const Adapter::FBIStructureWrapper>(get_structure())
        ->velnp()
        ->max_value(&norm_vel);
    if (norm_vel > 1e-15) norms = norm / norm_vel;

    std::ofstream log;
    if (Core::Communication::my_mpi_rank(get_discretizations()[1]->get_comm()) == 0)
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
