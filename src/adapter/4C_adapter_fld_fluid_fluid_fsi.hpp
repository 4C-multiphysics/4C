// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_FLUID_FSI_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_FLUID_FSI_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_xfem_input.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace FLD
{
  class XFluidFluid;
  namespace Utils
  {
    class MapExtractor;
    class XFluidFluidMapExtractor;
  }  // namespace Utils
}  // namespace FLD

namespace Adapter
{
  class FluidFluidFSI : public FluidFSI
  {
   public:
    /// constructor
    FluidFluidFSI(Global::Problem& problem, std::shared_ptr<Fluid> xfluidfluid,
        std::shared_ptr<Fluid> embfluid, std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params, bool isale, bool dirichletcond);

    /// initialize and prepare maps
    void init() override;

    /// prepare time step
    void prepare_time_step() override;

    /// save results of current time step, do XFEM cut and refresh the
    /// merged fluid map extractor
    void update() override;

    /// solve for pure fluid-fluid-ale problem
    void solve() override;

    std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override
    {
      FOUR_C_THROW("Do not call RexationSolve for XFFSI.");
      return nullptr;
    }

    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() override
    {
      FOUR_C_THROW("Do not call extract_interface_forces for XFFSI.");
      return nullptr;
    }

    /// @name Accessors
    //@{

    // get merged xfluid-fluid dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override;

    /// communication object at the interface
    std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const override
    {
      return mergedfluidinterface_;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> grid_vel() override;
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_grid_vel();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() override;
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp();
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() override;

    /// get the velocity row map of the embedded fluid
    std::shared_ptr<const Core::LinAlg::Map> velocity_row_map() override;

    /// get block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override;

    // access to embedded discretization
    const std::shared_ptr<Core::FE::Discretization>& discretization() override;

    // return discretization writer of embedded fluid discretization (for special purpose output)
    const std::shared_ptr<Core::IO::DiscretizationWriter>& disc_writer() override
    {
      return output_;
    }

    /// get map extractor for background/embedded fluid
    std::shared_ptr<FLD::Utils::XFluidFluidMapExtractor> const& x_fluid_fluid_map_extractor();

    //@}

    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp) override
    {
      FOUR_C_THROW("Not implemented, yet!");
    }

    // apply ALE-mesh displacements to embedded fluid
    void apply_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp) override;

    /// evaluate the fluid and update the merged fluid/FSI DOF-map extractor in case of a change in
    /// the DOF-maps
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            stepinc  ///< solution increment between time step n and n+1
        ) override;

    /// request fluid system matrix & shapederivatives as blockmatrices when called with true
    /// (indicated monolithic XFFSI with fluidsplit)
    void use_block_matrix(bool split_fluidsysmat = false) override;

    /// determine, whether the ALE-mesh should be relaxed at current time step
    bool is_ale_relaxation_step(int step) const;

    /// get type of monolithic XFFSI approach
    XFEM::MonolithicXffsiApproach monolithic_xffsi_approach() const { return monolithic_approach_; }

    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> shape_derivatives() override;

   private:
    /// setup of map extractor to distinguish between FSI DOF-map and
    /// merged inner embedded fluid and background fluid DOF-map
    void setup_interface(const int nds_master = 0) override;

    /// prepare underlying extended shape derivatives matrix, that is based
    /// on the merged fluid dof-map (with background fluid dof set to zero),
    /// as it may change
    void prepare_shape_derivatives();

    /// type cast pointer to XFluidFluid
    std::shared_ptr<FLD::XFluidFluid> xfluidfluid_;

    /// fsi map extractor for merged fluid maps (to keep fsi interface-DOF apart from
    /// merged inner DOF (inner embedded fluid together with background fluid)
    std::shared_ptr<FLD::Utils::MapExtractor> mergedfluidinterface_;

    /// type of monolithic XFluid-Fluid approach (decides whether ALE-mesh is fixed during
    /// Newton iteration)
    XFEM::MonolithicXffsiApproach monolithic_approach_;

    /// flag, that indicates, whether ALE-relaxation is activated
    bool relaxing_ale_;

    /// no. of timesteps, after which ALE-mesh should be relaxed
    int relaxing_ale_every_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
