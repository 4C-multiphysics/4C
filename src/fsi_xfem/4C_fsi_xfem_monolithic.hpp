// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_XFEM_MONOLITHIC_HPP
#define FOUR_C_FSI_XFEM_MONOLITHIC_HPP


#include "4C_config.hpp"

#include "4C_fsi_input.hpp"
#include "4C_fsi_xfem_algorithm.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_xfem_input.hpp"

#include <limits>

FOUR_C_NAMESPACE_OPEN



// forward declarations

namespace Core::LinAlg
{
  class Solver;
}

namespace XFEM
{
  class CouplingManager;
  class XFluidContactComm;
}  // namespace XFEM

namespace FSI
{
  /// monolithic XFSI algorithm
  /*!
       The monolithic system which consists of structural dofs and a varying number of fluid dofs
    (based on XFEM) is solved via a (possibly multiple restarting) Newton-Raphson scheme. When the
    fluid dofsets change during the Newton iterations, the Newton scheme has to be restarted with a
    good prediction of last Newton iteration. The Newton scheme has to be restarted as long as the
    fluid dofsets change between iterations. When convergence is reached, also the dofsets are
    expected not to change anymore. In case of permanently activating/deactivating fluid dofs, the
    fluid has to be solved on a slightly larger modified fluid dofsets where additional dofs are
    controlled via fluid stabilization (ghost-penalty) terms.

  */
  class MonolithicXFEM : public AlgorithmXFEM
  {
    // TODO adapt the FSIResultTest
    friend class FSIResultTest;

   public:
    //! constructor
    explicit MonolithicXFEM(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
        const Adapter::FieldWrapper::Fieldtype type = Adapter::FieldWrapper::type_StructureField);

    //--------------------------------------------------------------------------//
    //! @name Setup routine

    //! setup of the monolithic XFSI system, setup a new combined block row map and a new block
    //! matrix
    void setup_system() override;

    virtual void create_system_matrix();
    //@}

    //--------------------------------------------------------------------------//
    //! @name Restart routine

    /// read restart data for monolithic XFSI system
    void read_restart(int step) override;

    //@}


    //--------------------------------------------------------------------------//
    //! @name Time loop building blocks

    //! time loop of the monolithic system
    void timeloop() override;

    //! prepare the time step for fluid and structure
    void prepare_time_step() override;

    //! recover Lagrange multiplier (structural forces) needed for rhs in next time step and update
    //! single fields
    void update() override;

    //! write output
    void output() override;

   protected:
    //--------------------------------------------------------------------------//
    //! @name Safety routines

    //! validate the input parameter combinations
    void validate_parameters();

    //@}



    //@}

    //--------------------------------------------------------------------------//
    //! @name Coupling methods

    //! setup all coupling objects
    void setup_coupling_objects();

    //@}

    //--------------------------------------------------------------------------//
    //! @name dof_row_map access methods

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const { return extractor().full_map(); }

    //! extractor to communicate between full monolithic map and block maps of single fields
    const Core::LinAlg::MultiMapExtractor& extractor() const { return blockrowdofmap_; }

    //! extractor to communicate between full monolithic map and block maps of single fields
    //! this extractor considers poro as one field
    const Core::LinAlg::MultiMapExtractor& extractor_merged_poro() const
    {
      return blockrowdofmap_mergedporo_;
    }

    //@}

   private:
    //--------------------------------------------------------------------------//
    //! @name Setup of SYSMAT and RHS vector

    //! setup composed system matrix from field solvers, complete the global system matrix
    void setup_system_matrix();

    //! setup composed right hand side from field solvers
    void setup_rhs();

    //! setup RHS contributions based on single field residuals
    //! \sa setup_rhs()
    void setup_rhs_residual(Core::LinAlg::Vector<double>& f);

    //! Apply Dirichlet BCs to the whole system
    void apply_dbc();

    //! Extract initial guess from fields
    void initial_guess(Core::LinAlg::Vector<double>& ig);

    //! Create the combined DOF row map for the FSI problem; row maps of structure and xfluid to an
    //! global FSI DOF row map
    void create_combined_dof_row_map();

    //! set full monolithic dof row map
    //! The block maps must be row maps by themselves and must not contain identical GIDs.
    void set_dof_row_maps(const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps,
        const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps_mergedporo);

    //! Put two field vectors together to a monolithic vector
    //!
    //! As usual, the ordering is: structure -- fluid
    void combine_field_vectors(
        Core::LinAlg::Vector<double>& v,         ///< composed vector containing all field vectors
        const Core::LinAlg::Vector<double>& sv,  ///< structural DOFs
        const Core::LinAlg::Vector<double>& fv   ///< fluid DOFs
    );


    //! extract the two/three field vectors from a given composed vector
    /*!
      \param x  (i) composed vector that contains all field vectors
      \param sx (o) structural displacements and poro fluid velocity and pressure
      \param fx (o) fluid velocities and pressure
      \param ax (o) ale displacements
     */
    virtual void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& ax);

    //@}


    //--------------------------------------------------------------------------//
    //! @name Newton solve building blocks

    //! outer iteration loop, restarts inner Newton-Raphson iterations when fluid dofsets changes
    void solve();

    //! inner iteration loop (Newton-Raphson scheme) - return "true" if converged or "false" if
    //! unconverged or in case of changing fluid dof maps
    bool newton();

    //! update the global step-increment, evaluate the single fields with
    //! x^n+1 with x^n+1 = x_n + stepinc and return if the fluid dofsets
    //! between the two last iterations changed and a Newton restart is necessary
    bool evaluate();

    //! compute all norms used for convergence check
    void build_convergence_norms();

    //! check convergence of Newton iteration
    bool converged();

    //! apply damping on newton scheme
    void apply_newton_damping();

    //@}

    //--------------------------------------------------------------------------//
    //! @name Fluid-DOF permutation routines used for permuting fluid dofs during the Newton-solve

    //! update the permutation map between using the recent permutations between the last two Newton
    //! iterations
    void update_permutation_map(
        std::map<int, int> permutation_map  /// permutation map between last two Newton iterations,
                                            /// call by copy, do not call by reference!
    );

    //! build the new permutation cycles used for permuting the fluid dofs forward and backward
    void build_fluid_permutation();

    //! forward permutation of fluid dofs - transform vectors (based on dofsets) w.r.t old interface
    //! position forward to a vector (based on dofsets) w.r.t. new interface position
    void permute_fluid_dofs_forward(Core::LinAlg::Vector<double>& fx);

    //! backward permutation of fluid dofs - transform vectors (based on dofsets) w.r.t new
    //! interface position backward to a vector (based on dofsets) w.r.t. old interface position
    void permute_fluid_dofs_backward(Core::LinAlg::Vector<double>& fx);

    //@}


    //--------------------------------------------------------------------------//
    //! @name Linear Solve routines

    //! create linear solver (setup of parameter lists, etc...)
    void create_linear_solver();

    //! solve the linear FSI system
    void linear_solve();

    //! apply infnorm scaling to linear block system
    void scale_system(Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b);

    //! undo infnorm scaling from scaled solution
    void unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& b);

    //! create combined Dirichlet boundary condition map, map containing the dofs with Dirichlet BC
    std::shared_ptr<Core::LinAlg::Map> combined_dbc_map();

    //@}


    //--------------------------------------------------------------------------//
    //! @name Print Newton Iteration info

    //! print to screen information about residual forces and displacements
    void print_newton_iter();

    //! contains header to print_newton_iter
    void print_newton_iter_header();

    //! contains text to print_newton_iter
    void print_newton_iter_text();

    //@}

    //--------------------------------------------------------------------------//
    //! @name Parameter lists

    const Teuchos::ParameterList& fsidyn_;        //!< FSI dynamic parameter list
    const Teuchos::ParameterList& fsimono_;       //!< FSI-monolithic sublist
    const Teuchos::ParameterList& xfluidparams_;  //!< XFLUID DYNAMIC parameter list
    const Teuchos::ParameterList& xfpsimono_;     //!< XFPSI Monolithic parameter list

    //@}

    //--------------------------------------------------------------------------//
    //! @name General solver parameters

    bool solveradapttol_;                           //!< adapt solver tolerance
    double solveradaptolbetter_;                    //!< tolerance to which is adapted
    std::shared_ptr<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver

    //@}


    //--------------------------------------------------------------------------//
    //! @name Linear direct solver

    bool merge_fsi_blockmatrix_;  //!< merged blockmatrix, used for solver setup

    //@}


    //--------------------------------------------------------------------------//
    //! @name inf-norm scaling

    const bool scaling_infnorm_;  //!< inf-norm scaling for blockmatrix for iterative solvers

    std::shared_ptr<Core::LinAlg::Vector<double>> srowsum_;  //!< structural row sum
    std::shared_ptr<Core::LinAlg::Vector<double>> scolsum_;  //!< structural column sum

    //@}


    //--------------------------------------------------------------------------//
    //! @name Fluid dofset permutations during Newton

    std::map<int, int>
        permutation_map_;  //!< map of dof permutations (key=gid before, value=gid after)
    std::vector<std::vector<int>> permutation_;  //!< permutation cycles

    //@}


    //--------------------------------------------------------------------------//
    //! @name Global setup attributes

    //! dofrowmap split in (field) blocks with merged poro (to avoid splitting and merging vector)
    Core::LinAlg::MultiMapExtractor blockrowdofmap_mergedporo_;

    //! dofrowmap split in (field) blocks
    Core::LinAlg::MultiMapExtractor blockrowdofmap_;

    //! block systemmatrix for structural and fluid dofs
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //--------------------------------------------------------------------------//
    //! @name vectors used within the Newton scheme

    //! global sum of increments (step-increment), increment w.r.t the old timestep t^n
    std::shared_ptr<Core::LinAlg::Vector<double>> x_sum_;

    //! intermediate step increment for structure based on predictor solution from beginning of time
    //! step
    std::shared_ptr<const Core::LinAlg::Vector<double>> sx_sum_;

    //! intermediate step increment for fluid based on solution from t^n mapped/transformed to
    //! current interface position and permuted to current dofset ordering
    std::shared_ptr<Core::LinAlg::Vector<double>> fx_sum_;

    //! intermediate step increment for ale based on predictor solution from beginning of time step
    std::shared_ptr<Core::LinAlg::Vector<double>> ax_sum_;

    //! global Newton increment = iteration increment Delta x = x^n+1_i+1 - x^n+1_i
    std::shared_ptr<Core::LinAlg::Vector<double>> iterinc_;

    //! global residual vector
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;

    //! global vector for combined fluid and structure system filled with zeros used for DBCs
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    //@}


    //--------------------------------------------------------------------------//
    //! @name output streams

    //! output stream
    std::shared_ptr<std::ofstream> log_;

    //@}


    //--------------------------------------------------------------------------//
    //! @name Norms for convergence criterion and linear solver

    const double tolrhs_;  //!< rhs-tolerance for iterative linear solver

    // number of DOFS
    //--------------------------------------------------------------------------//
    int ns_;    //!< length of structural dofs
    int nf_;    //!< length of fluid dofs
    int nfv_;   //!< length of fluid velocity dofs
    int nfp_;   //!< length of fluid pressure dofs
    int nall_;  //!< length of all dofs (structure + fluid)

    // L2-NORMS (full)
    //--------------------------------------------------------------------------//
    double normrhs_ = std::numeric_limits<double>::max();  //!< L2-norm of full residual
    double norminc_ = std::numeric_limits<double>::max();  //!< L2-norm of full increment

    // L2-NORMS (split)
    //--------------------------------------------------------------------------//
    double normstrrhs_l2_ = std::numeric_limits<double>::max();  //!< L2-norm of structural residual
    double normflvelrhs_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of fluid velocity residual
    double normflpresrhs_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of fluid pressure residual
    double normpflvelrhs_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of poro fluid velocity residual
    double normpflpresrhs_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of poro fluid pressure residual

    double normstrinc_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of structural increment
    double normflvelinc_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of fluid velocity increment
    double normflpresinc_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of fluid pressure increment
    double normpflvelinc_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of poro fluid velocity increment
    double normpflpresinc_l2_ =
        std::numeric_limits<double>::max();  //!< L2-norm of poro fluid pressure increment
    //--------------------------------------------------------------------------//

    // Inf-NORMS (split)
    //--------------------------------------------------------------------------//
    double normstrrhs_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of structural residual
    double normflvelrhs_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of fluid velocity residual
    double normflpresrhs_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of fluid pressure residual
    double normpflvelrhs_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of poro fluid velocity residual
    double normpflpresrhs_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of poro fluid pressure residual

    //--------------------------------------------------------------------------//
    double normstrinc_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of structural increment
    double normstrincdisp_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of structural displacement increment
    double normflvelinc_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of fluid velocity increment
    double normflpresinc_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of fluid pressure increment
    double normpflvelinc_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of poro fluid velocity residual
    double normpflpresinc_inf_ =
        std::numeric_limits<double>::max();  //!< Inf-norm of poro fluid pressure residual

    //@}


    //--------------------------------------------------------------------------//
    //! @name Iteration counter

    int iter_{};        //!< inner iteration step
    int iter_outer_{};  //!< iteration counter for outer Newton loop

    const int itermin_;        //!< minimally requested inner iteration
    const int itermax_;        //!< maximally permitted inner iterations
    const int itermax_outer_;  //!< maximally permitted outer iterations (used for newton restart)
    //@}


    //--------------------------------------------------------------------------//
    //! @name Convergence criterion and convergence tolerances for Newton scheme

    const FSI::ConvNorm normtypeinc_;   //!< convergence check for increment
    const FSI::ConvNorm normtypefres_;  //!< convergence check for residual forces
    const enum FSI::BinaryOp
        combincfres_;  //!< binary operator to check for increment plus residual convergence or not

    const double tolinc_;   //!< tolerance full increment (structure + fluid)
    const double tolfres_;  //!< tolerance full residual (structure + fluid)

    // tolerances for structure block (displacements)
    const double tol_dis_res_l2_;
    const double tol_dis_res_inf_;
    const double tol_dis_inc_l2_;
    const double tol_dis_inc_inf_;
    // tolerances for fluid block (pressure)
    const double tol_pre_res_l2_;
    const double tol_pre_res_inf_;
    const double tol_pre_inc_l2_;
    const double tol_pre_inc_inf_;
    // tolerances for fluid block (velocity)
    const double tol_vel_res_l2_;
    const double tol_vel_res_inf_;
    const double tol_vel_inc_l2_;
    const double tol_vel_inc_inf_;
    //@}

    //--------------------------------------------------------------------------//
    //! @name Newton damping members
    bool nd_newton_damping_;
    bool nd_newton_incmax_damping_;
    int nd_levels_;
    double nd_reduction_fac_;
    double nd_increase_fac_;
    std::vector<double> nd_normrhs_old_;
    double nd_maxscaling_;
    /// limiting value for maximal increment
    std::vector<double> nd_max_incnorm_;
    /// scaling member: residual based actual scaling
    double nd_act_scaling_;
    /// scaling member: incremental based scaling
    double nd_inc_scaling_;

    /// Minimal tolerance compared to max. structural displacement increment for evaluation of the
    /// cut
    double cut_evaluate_mintol_;
    /// Minimal number of iteration before evaluation of the CUT is stopped
    int cut_evaluate_miniter_;
    /// Specify CUT evaluate dynamic
    bool cut_evaluate_dynamic_;

    //! flag to indicate if we have contact
    bool have_contact_;

    //! Xfluid-Contact-Communicator
    std::shared_ptr<XFEM::XFluidContactComm> xf_c_comm_;
    //@}

    // Map of Coupling Object for FS, FP, FA coupling, ...(Fluid, Structure, Poro, Ale)
    std::map<int, std::shared_ptr<XFEM::CouplingManager>> coup_man_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
