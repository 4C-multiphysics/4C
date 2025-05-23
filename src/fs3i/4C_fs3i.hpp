// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FS3I_HPP
#define FOUR_C_FS3I_HPP


#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace Adapter
{
  class ScaTraBaseAlgorithm;
}  // namespace Adapter

namespace FSI
{
  class Monolithic;
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
  class FS3IBase
  {
   public:
    /// constructor of base class
    FS3IBase();

    /// destructor of base class
    virtual ~FS3IBase() = default;

    /// initialize this class
    virtual void init();

    /// setup this class
    virtual void setup();

    /// setup
    virtual void setup_system() = 0;

    /// timeloop of coupled problem
    virtual void timeloop() = 0;

    /// test results (if necessary)
    virtual void test_results(MPI_Comm comm) = 0;

    /// read restart
    virtual void read_restart() = 0;

    /// needed for redistribution of FPS3I interface, if running on parallel
    virtual void redistribute_interface() = 0;

    //! make sure potential Dirichlet conditions at the scatra coupling
    //! interface are defined on both discretizations
    void check_interface_dirichlet_bc();

    //! Check FS3I specific inputs
    void check_f_s3_i_inputs();

    //! output of scalars and mean scalars
    void scatra_output();

    //! increment step and time
    void increment_time_and_step();

    //! update ScaTra solution vectors (new time step)
    void update_scatra_fields();

    //! evaluate, solve and iteratively update coupled ScaTra problem
    void scatra_evaluate_solve_iter_update();

    //! @name monolithic ScaTra problem
    //@{

    //! evaluate ScaTra fields
    virtual void evaluate_scatra_fields();

    //! set Membrane concentration in scatra fields
    void set_membrane_concentration() const;

    //! set-up of global matrix and rhs of the monolithic ScaTra problem
    void setup_coupled_scatra_system();

    //! set-up of global rhs of the monolithic ScaTra problem
    void setup_coupled_scatra_vector(
        Core::LinAlg::Vector<double>& globalvec,   //!< resulting global vector
        const Core::LinAlg::Vector<double>& vec1,  //!< vector in fluid ScaTra map
        const Core::LinAlg::Vector<double>& vec2   //!< vector in solid ScaTra map
    );

    //! set-up of global rhs of the monolithic ScaTra problem
    void setup_coupled_scatra_rhs();

    //! set-up of global matrix of the monolithic ScaTra problem
    void setup_coupled_scatra_matrix() const;

    std::shared_ptr<Core::LinAlg::Vector<double>> scatra2_to_scatra1(
        const Core::LinAlg::Vector<double>& iv) const;

    std::shared_ptr<Core::LinAlg::Vector<double>> scatra1_to_scatra2(
        const Core::LinAlg::Vector<double>& iv) const;

    //! linear solution of monolithic ScaTra problem
    void linear_solve_scatra();

    //! iterative update of ScaTra solution vectors
    void scatra_iter_update();

    //! extraction of field-specific vectors from global ScaTra vector
    void extract_scatra_field_vectors(
        const Core::LinAlg::Vector<double>& globalvec,  //!< global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>>&
            vec1,  //!< resulting vector in fluid ScaTra map
        std::shared_ptr<const Core::LinAlg::Vector<double>>&
            vec2  //!< resulting vector in solid ScaTra map
    );

   private:
    /// extracts membrane concentration in membrane (interface)
    void extract_membrane_concentration(
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& MembraneConcentration) const;

    /// Calculation of membane concentration in the membrane between fluid-scatra and
    /// structure-scatra
    std::shared_ptr<Core::LinAlg::Vector<double>> calc_membrane_concentration() const;

   protected:
    /// vector of scatra algorithms
    std::vector<std::shared_ptr<Adapter::ScaTraBaseAlgorithm>> scatravec_;

    /// scatra rhs vector
    std::shared_ptr<Core::LinAlg::Vector<double>> scatrarhs_;

    /// scatra increment vector
    std::shared_ptr<Core::LinAlg::Vector<double>> scatraincrement_;

    /// dof row map of scatra problems split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> scatraglobalex_;

    /// vector of scatra field map extractors (coupled vs. uncoupled dofs)
    std::vector<std::shared_ptr<Core::LinAlg::MultiMapExtractor>> scatrafieldexvec_;

    /// coupling of dofs at the scatra interface
    std::shared_ptr<Coupling::Adapter::Coupling> scatracoup_;

    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> scatrasystemmatrix_;

    /// coupling forces (in case of surface permeability)
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> scatracoupforce_;

    /// coupling matrices (in case of surface permeability)
    std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>> scatracoupmat_;

    /// zero vector (needed for application of Dirichlet BC on coupling vector)
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> scatrazeros_;

    /// scatra solver
    std::shared_ptr<Core::LinAlg::Solver> scatrasolver_;

    /// flag for infinite surface permeability
    const bool infperm_;

    /// @name  control parameters for time-integration scheme

    /// maximal simulation time
    const double timemax_;

    /// number of steps to simulate
    const int numstep_;

    /// timestep
    const double dt_;

    /// current time
    double time_;

    /// current step
    int step_;

    //@}
   private:
    /// @name Matrix block transform objects
    /// Handle row and column map exchange for matrix blocks

    std::shared_ptr<Coupling::Adapter::MatrixRowColTransform> sbbtransform_;
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> sbitransform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> sibtransform_;
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> fbitransform_;
    ///@}

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    bool is_setup() { return issetup_; };

    //! returns true if init(..) was called and is still valid
    bool is_init() { return isinit_; };

    //! check if \ref setup() was called
    void check_is_setup();

    //! check if \ref init() was called
    void check_is_init();

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };
  };
}  // namespace FS3I

FOUR_C_NAMESPACE_CLOSE

#endif
