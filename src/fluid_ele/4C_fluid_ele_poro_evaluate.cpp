// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_factory.hpp"
#include "4C_fluid_ele_interface.hpp"
#include "4C_fluid_ele_parameter_poro.hpp"
#include "4C_fluid_ele_poro.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

int Discret::Elements::FluidPoro::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  // get material
  std::shared_ptr<Core::Mat::Material> mat = material();

  // switch between different physical types as used below
  std::string impltype = "poro";
  switch (
      params.get<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::physicaltype_undefined))
  {
    case Inpar::FLUID::poro:
      impltype = "poro";
      break;
    case Inpar::FLUID::poro_p1:
      impltype = "poro_p1";
      break;
    default:
      FOUR_C_THROW("invalid physical type for porous fluid!");
      break;
  }

  switch (act)
  {
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case FLD::calc_fluid_systemmat_and_residual:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate(
              this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    // for the particular case of porous flow
    //-----------------------------------------------------------------------
    case FLD::calc_porousflow_fluid_coupling:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate(this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2,
              elevec3, true);
    }
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    // for evaluation of off-diagonal matrix block for monolithic
    // porous flow with scalar transport
    //-----------------------------------------------------------------------
    case FLD::calc_poroscatra_mono_odblock:
    {
      switch (params.get<Inpar::FLUID::PhysicalType>(
          "Physical Type", Inpar::FLUID::physicaltype_undefined))
      {
        case Inpar::FLUID::poro:
        {
          // no coupling
          return 0;
        }
        default:
          FOUR_C_THROW(
              "Invalid physical type for monolithic poroelasticity with scalar transport\n");
          break;
      }
    }
    break;
    case FLD::calc_volume:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate_service(
              this, params, mat, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    default:
      // call evaluate of standard fluid
      return Fluid::evaluate(
          params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
  }

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
