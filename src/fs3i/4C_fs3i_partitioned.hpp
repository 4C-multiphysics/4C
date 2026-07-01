// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FS3I_PARTITIONED_HPP
#define FOUR_C_FS3I_PARTITIONED_HPP


#include "4C_config.hpp"

#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fs3i.hpp"
#include "4C_fs3i_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Adapter
{
  class MortarVolCoupl;
  class ScaTraBaseAlgorithm;
}  // namespace Adapter

namespace FSI
{
  class Monolithic;

  namespace Utils
  {
    class MatrixRowTransform;
    class MatrixColTransform;
    class MatrixRowColTransform;
  }  // namespace Utils
}  // namespace FSI

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class BlockSparseMatrixBase;
  class SparseMatrix;
  class Solver;
}  // namespace Core::LinAlg


namespace FS3I
{
  class PartFS3I : public FS3IBase
  {
   public:
    //! constructor of base class for partitioned FS3I
    PartFS3I(MPI_Comm comm);

    //! initialize this class
    void init() override;

    //! setup this class
    void setup() override;

    //! @name overall FS3I system
    //@{

    //! time loop to be defined in inherited classes (structure depends on
    //! considered coupling, i.e. one-way or two-way)
    void timeloop() override = 0;

    //! flag whether time loop should be finished
    bool not_finished() const { return ((step_ < numstep_) and ((time_ + 1e-14) < timemax_)); };

    //! read and set fields needed for restart
    void read_restart() override;

    /// redistribute the  FPSI interface, if running on parallel. Just needed in the case of FPS3I
    void redistribute_interface() override { return; };

    /// create a volmortar object
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> create_vol_mortar_object(
        std::shared_ptr<Core::FE::Discretization> masterdis,
        std::shared_ptr<Core::FE::Discretization> slavedis);

    //! set-up of FSI and ScaTra systems
    void setup_system() override;

    //! test results for individual fields
    void test_results(MPI_Comm comm) override;

    //! information transfer FSI -> ScaTra
    void set_fsi_solution();

    /// set scatra solution on structure field
    void set_struct_scatra_solution() const;

    //! check convergence of monolithic ScaTra problem (depends on which
    // coupling is considered)
    virtual bool scatra_convergence_check(int itnum) = 0;

    //! return communicator
    MPI_Comm get_comm() const { return comm_; }

    /// extract fluid convective and structure convective velocities
    void extract_vel(std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& vel,
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& convel) const;

    void set_velocity_fields() const;

    //! routine for preparing time step to be defined in inherited classes
    //! (structure depends on coupling, that is, either one- or two-way)
    virtual void prepare_time_step() = 0;

    void set_mesh_disp() const;

    /// provide wall shear stresses from FS3I subproblem for scatra subproblem
    virtual void set_wall_shear_stresses() const;

    /// extract Wall Shear Stresses at the interface
    void extract_wss(std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& wss) const;

    Teuchos::ParameterList& manipulate_fsi_time_params(const Teuchos::ParameterList& fs3idyn) const;

    //@}

    /// transport quantity from fluid to fluid-scalar
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_to_fluid_scalar(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> fluidvector) const;

    /// transport quantity from fluid-scalar to fluid
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_scalar_to_fluid(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> fluidscalarvector) const;

    /// transport quantity from structure to structure-scalar
    std::shared_ptr<const Core::LinAlg::Vector<double>> structure_to_structure_scalar(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> structurevector) const;

    /// transport quantity from structure-scalar to structure
    std::shared_ptr<const Core::LinAlg::Vector<double>> structure_scalar_to_structure(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> structurescalavector) const;

   private:
    /// transport quantity from i-th volmortar master to i-th volmortar slave
    std::shared_ptr<const Core::LinAlg::Vector<double>> vol_mortar_master_to_slavei(
        const int i, const std::shared_ptr<const Core::LinAlg::Vector<double>> mastervector) const;

    /// transport quantity from i-th volmortar slave to i-th volmortar master
    std::shared_ptr<const Core::LinAlg::Vector<double>> vol_mortar_slave_to_masteri(
        const int i, const std::shared_ptr<const Core::LinAlg::Vector<double>> slavevector) const;

   protected:
    /// fsi algorithm
    std::shared_ptr<FSI::Monolithic> fsi_;

    /// vector of scatra volume couplings (i.e. fluid to fluid-scalar and structure to
    /// structure-scalar)
    std::vector<FS3I::VolumeCoupling> volume_fieldcouplings_;

    virtual void validate_structure_scatra_clone_configuration(
        const Core::FE::Discretization& structdis, const Core::FE::Discretization& structscatradis,
        const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
        const
    { /* do nothing in base class */
    }

   private:
    //! volume coupling (using mortar) adapter
    std::vector<std::shared_ptr<Coupling::Adapter::MortarVolCoupl>> volume_coupling_objects_;

    /// communication (mainly for screen output)
    MPI_Comm comm_;

    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> fluidscatra_;

    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> structscatra_;
  };
}  // namespace FS3I

FOUR_C_NAMESPACE_CLOSE

#endif
