// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_ParameterList.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

FLD::TimInt::TimInt(const std::shared_ptr<Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output)
    : discret_(discret),
      solver_(solver),
      params_(params),
      output_(output),
      runtime_output_writer_(nullptr),
      time_(0.0),
      step_(0),
      dta_(params_->get<double>("time step size")),
      stepmax_(params_->get<int>("max number timesteps")),
      maxtime_(params_->get<double>("total time")),
      itemax_(params_->get<int>("max nonlin iter steps")),
      uprestart_(params_->get("write restart every", -1)),
      upres_(params_->get("write solution every", -1)),
      timealgo_(Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(
          *params_, "time int algo")),
      physicaltype_(
          Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(*params_, "Physical Type")),
      myrank_(Core::Communication::my_mpi_rank(discret_->get_comm())),
      updateprojection_(false),
      projector_(nullptr),
      kspsplitter_(nullptr)
{
  // check for special fluid output which is to be handled by an own writer object
  const Teuchos::ParameterList fluid_runtime_output_list(
      Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT").sublist("FLUID"));

  bool output_fluid = fluid_runtime_output_list.get<bool>("OUTPUT_FLUID");

  // create and initialize parameter container object for fluid specific runtime output
  if (output_fluid)
  {
    runtime_output_params_.init(fluid_runtime_output_list);
    runtime_output_params_.setup();

    // TODO This does not work for restarted simulations as the time_ is not yet correctly set.
    // However, this is called before the restart is read and someone with knowledge on the module
    // has to refactor the code. The only implication is that in restarted simulations the .pvd file
    // does not contain the steps of the simulation that is restarted from
    runtime_output_writer_ = std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
        discret_, Core::IO::visualization_parameters_factory(
                      Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                      *Global::Problem::instance()->output_control_file(), time_));
  }
}

std::shared_ptr<const Core::LinAlg::Map> FLD::TimInt::dof_row_map(unsigned nds)
{
  return Core::Utils::shared_ptr_from_ref(*discretization()->dof_row_map(nds));
}


void FLD::TimInt::increment_time_and_step()
{
  step_ += 1;
  time_ += dta_;
}

FOUR_C_NAMESPACE_CLOSE
