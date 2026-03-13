// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_MONOLITHICSTRUCTURESPLIT_HPP
#define FOUR_C_FSI_MONOLITHICSTRUCTURESPLIT_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_inpar_fsi.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class Structure;
  class Fluid;
}  // namespace Adapter

namespace NOX
{
  namespace FSI
  {
    class AdaptiveNewtonNormF;
    class Group;
  }  // namespace FSI
}  // namespace NOX

namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
}

namespace FSI
{
  class OverlappingBlockMatrix;
}  // namespace FSI

namespace FSI
{
  /// monolithic FSI algorithm with overlapping interface equations
  /*!

    Here the structural matrix is split whereas the fluid matrix is taken as
    it is.

    \sa MonolithicOverlap
    */
  class MonolithicStructureSplit : public BlockMonolithic
  {
    friend class FSI::FSIResultTest;

   public:
    explicit MonolithicStructureSplit(MPI_Comm comm, const Teuchos::ParameterList& timeparams);

    /*! do the setup for the monolithic system


    1.) setup coupling; right now, we use matching meshes at the interface
    2.) create combined map
    3.) create block system matrix


    */
    void setup_system() override;

    //! @name Apply current field state to system

    /// setup composed system matrix from field solvers
    void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat) override;

    //@}

    /*! \brief Recover Lagrange multiplier \f$\lambda_\Gamma\f$
     *
     *  Recover Lagrange multiplier \f$\lambda_\Gamma\f$ at the interface at the
     *  end of each time step (i.e. condensed forces onto the structure) needed
     *  for rhs in next time step in order to guarantee temporal consistent
     *  exchange of coupling traction
     */
    void recover_lagrange_multiplier() override;

    /*! \brief Compute spurious interface energy increment due to temporal discretization
     *
     *  Due to the temporal discretization, spurious energy \f$\Delta E_\Gamma^{n\rightarrow n+1}\f$
     *  might be produced at the interface. It can be computed as
     *  \f[
     *  \Delta E_\Gamma^{n\rightarrow n+1}
     *  = \left((a-b)\lambda^n +
     * (b-a)\lambda^{n+1}\right)\left(d_\Gamma^{S,n+1}-d_\Gamma^{S,n}\right) \f] with the time
     * interpolation factors a and b.
     */
    void calculate_interface_energy_increment() override;

    /// the composed system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> system_matrix() const override;

    //! @name Methods for infnorm-scaling of the system

    /// apply infnorm scaling to linear block system
    void scale_system(
        Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b) override;

    /// undo infnorm scaling from scaled solution
    void unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& b) override;

    //@}

    /// Output routine accounting for Lagrange multiplier at the interface
    void output() override;

    /// Write Lagrange multiplier
    void output_lambda() override;

    //! take current results for converged and save for next time step
    void update() override;

    /// read restart data
    void read_restart(int step) override;

    /// return Lagrange multiplier \f$\lambda_\Gamma\f$ at the interface
    std::shared_ptr<Core::LinAlg::Vector<double>> get_lambda() override { return lambda_; };

    //! @name Time Adaptivity
    //@{

    /*! \brief Select \f$\Delta t_{min}\f$ of all proposed time step sizes based on error estimation
     *
     *  Depending on the chosen method (fluid or structure split), only 3 of the
     *  6 available norms are useful. Each of these three norms delivers a new
     *  time step size. Select the minimum of these three as the new time step size.
     */
    double select_dt_error_based() const override;

    /*! \brief Check whether time step is accepted or not
     *
     *  In case that the local truncation error is small enough, the time step is
     *  accepted.
     */
    bool set_accepted() const override;

    //@}

   protected:
    /// create the composed system matrix
    void create_system_matrix();

    /// setup of NOX convergence tests
    Teuchos::RCP<::NOX::StatusTest::Combo> create_status_test(
        Teuchos::ParameterList& nlParams, Teuchos::RCP<::NOX::Abstract::Group> grp) override;

    //! Extract the three field vectors from a given composed vector
    //!
    //! The condensed ale degrees of freedom have to be recovered
    //! from the fluid solution using a suitable velocity-displacement
    //! conversion. After that, the ale solution increment is projected onto the
    //! structure where a possible structural predictor has to
    //! be considered.
    //!
    //! We are dealing with NOX here, so we get absolute values. x is the sum of
    //! all increments up to this point.
    //!
    //! \sa  Adapter::FluidFSI::velocity_to_displacement()
    void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>>
                                   x,  ///< composed vector that contains all field vectors
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,  ///< structural displacements
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,  ///< fluid velocities and pressure
        std::shared_ptr<const Core::LinAlg::Vector<double>>& ax   ///< ale displacements
        ) override;

   protected:
    /*! \brief Create the combined DOF row map for the FSI problem
     *
     *  Combine the DOF row maps of structure, fluid and ALE to an global FSI
     *  DOF row map.
     */
    void create_combined_dof_row_map() override;

    /*! \brief Setup the Dirichlet map extractor
     *
     *  Create a map extractor #dbcmaps_ for the Dirichlet degrees of freedom
     *  for the entire FSI problem. This is done just by combining the
     *  condition maps and other maps from structure, fluid and ALE to a FSI-global
     *  condition map and other map.
     */
    void setup_dbc_map_extractor() override;

    /// setup RHS contributions based on single field residuals
    void setup_rhs_residual(Core::LinAlg::Vector<double>& f) override;

    /// setup RHS contributions based on the Lagrange multiplier field
    void setup_rhs_lambda(Core::LinAlg::Vector<double>& f) override;

    /// setup RHS contributions based on terms for first nonlinear iteration
    void setup_rhs_firstiter(Core::LinAlg::Vector<double>& f) override;

    void combine_field_vectors(Core::LinAlg::Vector<double>& v,
        std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fv,
        std::shared_ptr<const Core::LinAlg::Vector<double>> av,
        bool slave_vectors_contain_interface_dofs) final;

    /// check for dirichlet conditions on solid interface
    void check_dirichlet_boundary_conditions_on_interface();

    /// block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    /// coupling of fluid and ale at the free surface
    std::shared_ptr<Coupling::Adapter::Coupling> fscoupfa_;

    /// @name Matrix block transform objects
    /// Handle row and column map exchange for matrix blocks

    std::shared_ptr<Coupling::Adapter::MatrixRowColTransform> sggtransform_;
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> sgitransform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> sigtransform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> aigtransform_;

    std::shared_ptr<Coupling::Adapter::MatrixColTransform> fmiitransform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> fmgitransform_;

    std::shared_ptr<Coupling::Adapter::MatrixColTransform> fsaigtransform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> fsmgitransform_;

    ///@}

    /// @name infnorm scaling

    std::shared_ptr<Core::LinAlg::Vector<double>> srowsum_;
    std::shared_ptr<Core::LinAlg::Vector<double>> scolsum_;
    std::shared_ptr<Core::LinAlg::Vector<double>> arowsum_;
    std::shared_ptr<Core::LinAlg::Vector<double>> acolsum_;

    //@}

    /// @name Some quantities to recover the Lagrange multiplier at the end of each time step

    //! Lagrange multiplier \f$\lambda_\Gamma^n\f$ at the interface (ie condensed forces onto the
    //! structure) evaluated at old time step \f$t_n\f$ but needed for next time step \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda_;

    //! Lagrange multiplier of previous time step
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaold_;

    //! inner structural displacement increment \f$\Delta(\Delta d_{I,i+1}^{n+1})\f$ at current NOX
    //! iteration \f$i+1\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> ddiinc_;

    //! inner displacement solution of the structure at previous NOX iteration
    std::shared_ptr<const Core::LinAlg::Vector<double>> soliprev_;

    //! structural interface displacement increment \f$\Delta(\Delta d_{\Gamma,i+1}^{n+1})\f$ at
    //! current NOX iteration \f$i+1\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> ddginc_;

    //! fluid interface velocity increment \f$\Delta(\Delta u_{\Gamma,i+1}^{n+1})\f$ at current NOX
    //! iteration \f$i+1\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> duginc_;

    //! interface displacement solution of the structure at previous NOX iteration
    std::shared_ptr<const Core::LinAlg::Vector<double>> disgprev_;

    //! interface velocity solution of the fluid at previous NOX iteration
    std::shared_ptr<const Core::LinAlg::Vector<double>> velgprev_;

    //! block \f$S_{\Gamma I,i+1}\f$ of structural matrix at current NOX iteration \f$i+1\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> sgicur_;

    //! block \f$S_{\Gamma I,i}\f$ of structural matrix at previous NOX iteration \f$i\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> sgiprev_;

    //! block \f$S_{\Gamma\Gamma,i+1}\f$ of structural matrix at current NOX iteration \f$i+1\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> sggcur_;

    //! block \f$S_{\Gamma\Gamma,i}\f$ of structural matrix at previous NOX iteration \f$i\f$
    std::shared_ptr<const Core::LinAlg::SparseMatrix> sggprev_;

    //@}

    //! summation of amount of artificial interface energy due to temporal discretization
    double energysum_;

    /// additional ale residual to avoid incremental ale errors
    std::shared_ptr<Core::LinAlg::Vector<double>> aleresidual_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
