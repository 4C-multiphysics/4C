// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_fld_fluid_xfsi.hpp"

#include "4C_adapter_fld_fluid.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_discretization.hpp"

#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*======================================================================*/
/* constructor */
Adapter::XFluidFSI::XFluidFSI(std::shared_ptr<Fluid> fluid,  // the XFluid object
    const std::string coupling_name,                         // name of the FSI coupling condition
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : FluidWrapper(fluid),  // the XFluid object is set as fluid_ in the FluidWrapper
      fpsiinterface_(std::make_shared<FLD::Utils::MapExtractor>()),
      coupling_name_(coupling_name),
      solver_(solver),
      params_(params)
{
  // make sure
  if (fluid_ == nullptr) FOUR_C_THROW("Failed to create the underlying fluid adapter");
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::init()
{
  // call base class init
  FluidWrapper::init();

  // cast fluid to fluidimplicit
  xfluid_ = std::dynamic_pointer_cast<FLD::XFluid>(fluid_);
  if (xfluid_ == nullptr) FOUR_C_THROW("Failed to cast Adapter::Fluid to FLD::XFluid.");

  // NOTE: currently we are using the XFluidFSI adapter also for pure ALE-fluid problems with
  // level-set boundary in this case no mesh coupling object is available and no interface objects
  // can be created
  std::shared_ptr<XFEM::MeshCoupling> mc = xfluid_->get_mesh_coupling(coupling_name_);

  if (mc != nullptr)  // classical mesh coupling case for FSI
  {
    // get the mesh coupling object
    mesh_coupling_fsi_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(mc);

    structinterface_ = std::make_shared<FLD::Utils::MapExtractor>();

    // the solid mesh has to match the interface mesh
    // so we have to compute a interface true residual vector itrueresidual_
    structinterface_->setup(*mesh_coupling_fsi_->get_cutter_dis());
  }

  interface_ = std::make_shared<FLD::Utils::MapExtractor>();

  interface_->setup(
      *xfluid_->discretization(), false, true);  // Always Create overlapping FSI/FPSI Interface

  fpsiinterface_->setup(
      *xfluid_->discretization(), true, true);  // Always Create overlapping FSI/FPSI Interface

  meshmap_ = std::make_shared<Core::LinAlg::MapExtractor>();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Adapter::XFluidFSI::time_scaling() const
{
  // second order (OST(0.5) except for the first starting step, otherwise 1st order BackwardEuler
  if (params_->get<bool>("interface second order"))
    return 2. / xfluid_->dt();
  else
    return 1. / xfluid_->dt();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization> Adapter::XFluidFSI::boundary_discretization()
{
  return mesh_coupling_fsi_->get_cutter_dis();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::XFluidFSI::extract_struct_interface_forces()
{
  // the trueresidual vector has to match the solid dis
  // it contains the forces acting on the structural surface
  return structinterface_->extract_fsi_cond_vector(*mesh_coupling_fsi_->i_true_residual());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::XFluidFSI::extract_struct_interface_veln()
{
  // it depends, when this method is called, and when velnp is updated
  // the FSI algorithm expects first an time update and then asks for the old time step velocity
  // meaning that it gets the velocity from the new time step
  // not clear? exactly! thats why the FSI time update should be more clear about it
  // needs discussion with the FSI people
  return structinterface_->extract_fsi_cond_vector(*mesh_coupling_fsi_->i_veln());
}


/*----------------------------------------------------------------------*/
// apply the interface velocities to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_struct_interface_velocities(
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel)
{
  structinterface_->insert_fsi_cond_vector(*ivel, *mesh_coupling_fsi_->i_velnp());
}


/*----------------------------------------------------------------------*/
//  apply the interface displacements to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_struct_mesh_displacement(
    std::shared_ptr<const Core::LinAlg::Vector<double>> interface_disp)
{
  // update last increment, before we set new idispnp
  mesh_coupling_fsi_->update_displacement_iteration_vectors();

  // set new idispnp
  structinterface_->insert_fsi_cond_vector(*interface_disp, *mesh_coupling_fsi_->i_dispnp());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::XFluidFSI::set_mesh_map(
    std::shared_ptr<const Core::LinAlg::Map> mm, const int target_dofset_number)
{
  // check target_dofset_number
  if (target_dofset_number != 0) FOUR_C_THROW("target_dofset_number is supposed to be 0 here");

  meshmap_->setup(*xfluid_->discretisation_xfem()->initial_dof_row_map(), mm,
      Core::LinAlg::split_map(*xfluid_->discretisation_xfem()->initial_dof_row_map(), *mm));
}

/*----------------------------------------------------------------------*/
//  apply the ale displacements to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_mesh_displacement(
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp)
{
  meshmap_->insert_cond_vector(*fluiddisp, *xfluid_->write_access_dispnp());

  // new grid velocity
  xfluid_->update_gridv();
}


/*----------------------------------------------------------------------*
 * convert increment of displacement to increment in velocity
 * Delta d = d^(n+1,i+1)-d^n is converted to the interface velocity increment
 * Delta u = u^(n+1,i+1)-u^n
 * via first order or second order OST-discretization of d/dt d(t) = u(t)
 *----------------------------------------------------------------------*/
void Adapter::XFluidFSI::displacement_to_velocity(
    std::shared_ptr<Core::LinAlg::Vector<double>> fcx  /// Delta d = d^(n+1,i+1)-d^n
)
{
  // get interface velocity at t(n)
  const std::shared_ptr<const Core::LinAlg::Vector<double>> veln =
      structinterface_->extract_fsi_cond_vector(*mesh_coupling_fsi_->i_veln());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check, whether maps are the same
  if (!fcx->get_map().point_same_as(veln->get_map()))
  {
    FOUR_C_THROW("Maps do not match, but they have to.");
  }
#endif

  /*
   * Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
   *
   *             / = 2 / dt   if interface time integration is second order
   * with fac = |
   *             \ = 1 / dt   if interface time integration is first order
   */
  const double timescale = time_scaling();
  fcx->update(-timescale * xfluid_->dt(), *veln, timescale);
}


/// return xfluid coupling matrix between structure and fluid as sparse matrices
std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_struct_fluid_matrix()
{
  return xfluid_->c_sx_matrix(coupling_name_);
}

/// return xfluid coupling matrix between fluid and structure as sparse matrices
std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_fluid_struct_matrix()
{
  return xfluid_->c_xs_matrix(coupling_name_);
}

/// return xfluid coupling matrix between structure and structure as sparse matrices
std::shared_ptr<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_struct_struct_matrix()
{
  return xfluid_->c_ss_matrix(coupling_name_);
}

/// return xfluid coupling matrix between structure and structure as sparse matrices
std::shared_ptr<const Core::LinAlg::Vector<double>> Adapter::XFluidFSI::rhs_struct_vec()
{
  return xfluid_->rhs_s_vec(coupling_name_);
}

/// GmshOutput for background mesh and cut mesh
void Adapter::XFluidFSI::gmsh_output(const std::string& name,  ///< name for output file
    const int step,                                            ///< step number
    const int count,                    ///< counter for iterations within a global time step
    Core::LinAlg::Vector<double>& vel,  ///< vector holding velocity and pressure dofs
    std::shared_ptr<Core::LinAlg::Vector<double>> acc  ///< vector holding accelerations
)
{
  // TODO (kruse): find a substitute!
  // xfluid_->GmshOutput(name, step, count, vel, acc);
  FOUR_C_THROW("Gmsh output for XFSI during Newton currently not available.");
}

FOUR_C_NAMESPACE_CLOSE
