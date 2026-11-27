// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TIMINT_RED_HPP
#define FOUR_C_FLUID_TIMINT_RED_HPP


#include "4C_config.hpp"

#include "4C_fluid_implicit_integration.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class ArtNet;
}
namespace FLD
{
  namespace Utils
  {
    class MapExtractor;
    class FluidImpedanceWrapper;
    class FluidWkOptimizationWrapper;
    class FluidVolumetricSurfaceFlowWrapper;
    class TotalTractionCorrector;
    class FluidCouplingWrapperBase;
    class VolumetricFlowMapExtractor;
    template <class RedDTimeInt>
    class FluidCouplingWrapper;
  }  // namespace Utils

  class TimIntRedModels : public virtual FluidImplicitTimeInt
  {
   public:
    /// Standard Constructor
    TimIntRedModels(const std::shared_ptr<Core::FE::Discretization>& actdis,
        const std::shared_ptr<Core::LinAlg::Solver>& solver,
        const std::shared_ptr<Teuchos::ParameterList>& params,
        const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid = false);


    /*!
    \brief initialization

    */
    void init() override;

    /*!
    \brief evaluate problem-dependent boundary consitions: update the 3D-to-reduced_D coupling data
    for dirichlet bc in this context

    */
    void do_problem_specific_boundary_conditions() override;

    /*!
    \brief update the 3D-to-reduced_D coupling data in assemble_mat_and_rhs

    */
    virtual void update_3d_to_reduced_mat_and_rhs();

    /*!
    \brief read restart data
    */
    void read_restart(int step) override;

    /*!
    \brief read restart (some more RedModels-specific data)

    */
    virtual void read_restart_reduced_d(int step);

    /*!
    \brief update configuration and output to file/screen

    */
    virtual void output_reduced_d();

    /*!
    \brief Setup meshtying

    */
    void setup_meshtying() override;

    /*!
    \brief update configuration and output to file/screen

    */
    void output() override;

    //! write restart information for each step as defined in the input file
    void write_restart() const;
    /*!
    \brief Insert Womersley condition

    */
    void insert_volumetric_surface_flow_cond_vector(
        std::shared_ptr<Core::LinAlg::Vector<double>> vel,
        std::shared_ptr<Core::LinAlg::Vector<double>> res) override;

    /// prepare AVM3-based scale separation
    void avm3_preparation() override;

    /// prepare time step
    void prepare_time_step() override;

    /*!
    \brief Additional function for RedModels in linear_relaxation_solve

    */
    void custom_solve(std::shared_ptr<Core::LinAlg::Vector<double>> relax) override;

    /*!
    \brief Set custom parameters in the respective time integration class (Loma, RedModels...)

    */
    void set_custom_ele_params_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams) override;

    /*!
    \brief call elements to calculate system matrix/rhs and assemble

    */
    void assemble_mat_and_rhs() override;

    /*!
    \brief apply Dirichlet boundary conditions to system of equations

    */
    void apply_dirichlet_to_system() override;


   protected:
    /// bio related special (in/outflow) traction velocity component adder
    std::shared_ptr<Utils::TotalTractionCorrector> traction_vel_comp_adder_bc_;

    /// bio related, 3D to reduced-D coupling
    std::shared_ptr<Utils::FluidCouplingWrapperBase> coupled3D_redDbc_art_;

    /// 1D arterial network time integration
    std::shared_ptr<Adapter::ArtNet> ART_timeInt_;

    /// bio related special (in/outflow) boundaries
    std::shared_ptr<Utils::FluidVolumetricSurfaceFlowWrapper> vol_surf_flow_bc_;

    /// maps for womersley flow profile which is applied as a Dirichlet condition
    std::shared_ptr<Core::LinAlg::Map> vol_surf_flow_bc_maps_;

    /// maps for extracting Dirichlet and free DOF sets
    std::shared_ptr<FLD::Utils::VolumetricFlowMapExtractor> vol_flow_rates_bc_extractor_;

    /// flag for potential 3D Reduced_D coupling
    bool strong_redD_3d_coupling_;

   private:
  };  // class TimIntRedModels

}  // namespace FLD


FOUR_C_NAMESPACE_CLOSE

#endif
