// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LOMA_ALGORITHM_HPP
#define FOUR_C_LOMA_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_scatra_fluid_coupling_algorithm.hpp"
#include "4C_linalg_mapextractor.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace LowMach
{
  /// LOMA algorithm base
  /*!

    Base class of LOMA algorithms. Derives from FluidBaseAlgorithm
    and ScatraBaseAlgorithm.
    There can (and will) be different subclasses that implement
    different coupling schemes.

   */
  class Algorithm : public Adapter::ScaTraFluidCouplingAlgorithm
  {
   public:
    /// constructor
    Algorithm(Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& prbdyn,
        const Teuchos::ParameterList& solverparams);


    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return void

    */
    void init() override;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

    /// LOMA time loop
    void time_loop() override;

    /// read restart for preceding turbulent inflow simulation
    void read_inflow_restart(int restart);

   protected:
    /// do initial calculations
    void initial_calculations();

    /// prepare time step
    void prepare_time_step() override;

    /// do (partitioned) outer iteration loop
    void outer_loop();

    /// do monolithic iteration loop
    void mono_loop();

    /// set fluid values required in scatra
    void set_fluid_values_in_scatra();

    /// set scatra values required in fluid
    void set_scatra_values_in_fluid();

    /// set up right-hand-side for monolithic low-Mach-number system
    void setup_mono_loma_matrix();

    /// evaluate off-diagonal block with fluid weighting functions
    void evaluate_loma_od_block_mat_fluid(std::shared_ptr<Core::LinAlg::SparseMatrix> mat_fs);

    /// evaluate off-diagonal block with scatra weighting functions
    // void EvaluateLomaODBlockMatScaTra(std::shared_ptr<Core::LinAlg::SparseMatrix> mat_sf);

    /// set up right-hand-side for monolithic low-Mach-number system
    void setup_mono_loma_rhs();

    /// solve monolithic low-Mach-number system
    void mono_loma_system_solve();

    /// update for next iteration step for monolithic low-Mach-number system
    void iter_update();

    /// convergence Check for present iteration step
    bool convergence_check(int itnum);

    /// update for next time step
    void time_update();

    /// write output
    void output() override;

    /// flag for monolithic solver
    bool monolithic_;

    /// dof row map split in (field) blocks for monolithic solver
    Core::LinAlg::MultiMapExtractor lomablockdofrowmap_;

    /// combined Dirichlet boundary condition map for monolithic solver
    /// (unique map of all dofs with Dirichlet boundary conditions)
    std::shared_ptr<Core::LinAlg::Map> lomadbcmap_;

    /// incremental vector for monolithic solver
    std::shared_ptr<Core::LinAlg::Vector<double>> lomaincrement_;

    /// rhs vector for monolithic solver
    std::shared_ptr<Core::LinAlg::Vector<double>> lomarhs_;

    /// vector of zeros for Dirichlet boundary conditions for monolithic solver
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    /// block matrix for monolithic solver
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> lomasystemmatrix_;

    /// monolithic solver
    std::shared_ptr<Core::LinAlg::Solver> lomasolver_;

    /// time-step length, maximum time and maximum number of steps
    double dt_;
    double maxtime_;
    int stepmax_;

    /// (preliminary) maximum number of iterations and tolerance for outer iteration
    int itmax_;
    int itmaxpre_;
    int itmaxbs_;
    double ittol_;

    /// flag for constant thermodynamic pressure
    std::string consthermpress_;

    /// flag for special flow
    std::string special_flow_;

    /// start of sampling period
    int samstart_;

    /// flag for turbulent inflow
    bool turbinflow_;
    /// number of inflow steps
    int numinflowsteps_;

   private:
    //! problem dynamic parameters
    Teuchos::ParameterList probdyn_;
  };

}  // namespace LowMach

FOUR_C_NAMESPACE_CLOSE

#endif
