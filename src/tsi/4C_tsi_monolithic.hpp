// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TSI_MONOLITHIC_HPP
#define FOUR_C_TSI_MONOLITHIC_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 11/10 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_structure_new_input.hpp"
#include "4C_thermo_input.hpp"
#include "4C_timestepping_time_step_control.hpp"
#include "4C_tsi_algorithm.hpp"
#include "4C_tsi_input.hpp"

#include <Teuchos_Time.hpp>

#include <deque>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           dano 11/10 |
 *----------------------------------------------------------------------*/
// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
  class MultiMapExtractor;

  class BlockSparseMatrixBase;
  class Solver;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}

namespace Adapter
{
  class MortarVolCoupl;
}

namespace TSI
{
  namespace Utils
  {
    // forward declaration of clone strategy
    class ThermoStructureCloneStrategy;
  }  // namespace Utils

  //! monolithic TSI algorithm
  //!
  //!  Base class of TSI algorithms. Derives from structure_base_algorithm and
  //!  Thermo::BaseAlgorithm with temperature field.
  //!  There can (and will) be different subclasses that implement different
  //!  coupling schemes.
  //!
  //!  \warning The order of calling the two BaseAlgorithm-constructors (that
  //!  is the order in which we list the base classes) is important here! In the
  //!  constructors control file entries are written. And these entries define
  //!  the order in which the filters handle the Discretizations, which in turn
  //!  defines the dof number ordering of the Discretizations... Don't get
  //!  confused. Just always list structure, thermo. In that order.
  //!
  //!  \note There is the Algorithm class for general purpose TSI algorithms.
  //!  This simplifies the monolithic implementation.
  class Monolithic final : public Algorithm
  {
   public:
    explicit Monolithic(MPI_Comm comm, const Teuchos::ParameterList& sdynparams);



    /*! do the setup for the monolithic system


    1.) setup coupling
    2.) get maps for all blocks in the system (and for the whole system as well)
        create combined map
    3.) create system matrix


    \note We want to do this setup after reading the restart information, not
    directly in the constructor. This is necessary since during restart (if
    read_mesh is called), the dofmaps for the blocks might get invalid.
    */
    //! Setup the monolithic TSI system
    void setup_system() override;

    /// non-linear solve, i.e. (multiple) corrector
    [[nodiscard]] TSI::StepStatus solve() override;

    //! read restart data
    void read_restart(int step  //!< step number where the calculation is continued
        ) override;

    //! @name Apply current field state to system

    //! setup composed right hand side from field solvers
    void setup_rhs();

    //! setup composed system matrix from field solvers
    void setup_system_matrix();

    //! composed system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> system_matrix() const
    {
      return systemmatrix_;
    }

    //! solve linear TSI system
    void linear_solve();

    //! create linear solver (setup of parameter lists, etc...)
    void create_linear_solver();

    //! Evaluate mechanical-thermal system matrix
    void apply_str_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseMatrix> k_st  //!< mechanical-thermal stiffness matrix
    );

    //! Evaluate thermal-mechanical system matrix
    void apply_thermo_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseMatrix> k_ts  //!< thermal-mechanical tangent matrix
    );

    //! Evaluate thermal-mechanical system matrix for geonln + heat convection BC
    void apply_thermo_coupl_matrix_conv_bc(
        std::shared_ptr<Core::LinAlg::SparseMatrix> k_ts  //!< thermal-mechanical tangent matrix
    );

    //@}

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    void evaluate(std::shared_ptr<Core::LinAlg::Vector<double>>
            stepinc  //!< increment between time step n and n+1
    );

    //! extract initial guess from fields
    //! returns \f$\Delta x_{n+1}^{<k>}\f$
    void initial_guess(std::shared_ptr<Core::LinAlg::Vector<double>> ig);

    //! is convergence reached of iterative solution technique?
    //! keep your fingers crossed...
    bool converged();

    //! outer iteration loop
    void newton_full();

    //! apply DBC to all blocks
    void apply_dbc();

    //! do pseudo-transient continuation nonlinear iteration
    //!
    //! Pseudo-transient continuation is a variant of a full newton which has a
    //! larger convergence radius than newton and is therefore more stable
    //! and/or can do larger time steps
    //!

    void ptc();

    //! @name Output

    //! print to screen information about residual forces and displacements
    void print_newton_iter();

    //! contains text to print_newton_iter
    void print_newton_iter_text(FILE* ofile  //!< output file handle
    );

    //! contains header to print_newton_iter
    void print_newton_iter_header(FILE* ofile  //!< output file handle
    );

    //! Determine norm of force residual
    double calculate_vector_norm(const TSI::VectorNorm norm,  //!< norm to use
        const Core::LinAlg::Vector<double>& vect              //!< the vector of interest
    );

    //@}

    //! apply infnorm scaling to linear block system
    void scale_system(Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b);

    //! undo infnorm scaling from scaled solution
    void unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& b);

   protected:
    //! @name Time loop building blocks

    //! start a new time step
    void prepare_time_step() final;

    //! start a new time step and report whether the predictors succeeded
    [[nodiscard]] TSI::StepStatus prepare_time_step_with_status() override;

    //! convert the step status into an action, preparing a retry with reduced time-step size
    TSI::StepAction perform_error_action(TSI::StepStatus step_status) override;

    //! possibly increase the time-step size after an accepted step
    void finalize_successful_step() override;

    //! take current results for converged and save for next time step
    void update() override;

    //! calculate stresses, strains, energies
    void prepare_output(bool force_prepare) override;

    //! convergence check for Newton solver
    bool convergence_check(int itnum, int itmax, double ittol);

    //! extract the three field vectors from a given composed vector
    /*!
      x is the sum of all increments up to this point.
      \param x  (i) composed vector that contains all field vectors
      \param sx (o) structural vector (e.g. displacements)
      \param tx (o) thermal vector (e.g. temperatures)
      */
    void extract_field_vectors(std::shared_ptr<Core::LinAlg::Vector<double>> x,
        std::shared_ptr<Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<Core::LinAlg::Vector<double>>& tx);

    //! @name Access methods for subclasses

    //! full monolithic dof row map
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const;

    //! set full monolithic dof row map
    /*!
     A subclass calls this method (from its constructor) and thereby
     defines the number of blocks, their maps and the block order. The block
     maps must be row maps by themselves and must not contain identical GIDs.
    */
    void set_dof_row_maps();

    //! extractor to communicate between full monolithic map and block maps
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor() const { return blockrowdofmap_; }

    //! setup list with default parameters
    void set_default_parameters();

    //! recover structural and thermal Lagrange multipliers
    // this takes into account the dependence on off diagonal blocks
    void recover_struct_therm_lm();

    //@}

    //! @name General purpose algorithm members
    //@{

    bool solveradapttol_;                           //!< adapt solver tolerance
    double solveradaptolbetter_;                    //!< tolerance to which is adapted ????
    std::shared_ptr<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver

    //@}

    //! @name Printing and output
    //@{

    bool printiter_;  //!< print intermediate iterations during solution

    //@}

    //! @name Global vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  //!< a zero vector of full length
    //@}

    //! enum for STR time integartion
    Solid::DynamicType strmethodname_;

   private:
    //! if just rho_inf is specified for genAlpha, the other parameters in the global parameter
    //! list need to be adapted accordingly
    void fix_time_integration_params();

    //! reset all fields to the beginning of the step and reduce the time-step size
    void prepare_retry_with_reduced_time_step();

    //! set the time-step size of the algorithm and all fields for the current step
    void set_new_time_step_size(double new_dt);

    const Teuchos::ParameterList& tsidyn_;      //!< TSI dynamic parameter list
    const Teuchos::ParameterList& tsidynmono_;  //!< monolithic TSI dynamic parameter list

    //! dofrowmap split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_;

    //! build block vector from field vectors, e.g. rhs, increment vector
    void setup_vector(Core::LinAlg::Vector<double>& f,  //!< vector of length of all dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            sv,  //!< vector containing only structural dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            tv  //!< vector containing only thermal dofs
    );

    //! check if step is admissible for line search
    bool l_sadmissible();

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //! off diagonal matrixes
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_st_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_ts_;

    bool merge_tsi_blockmatrix_;  //!< bool whether TSI block matrix is merged

    //! @name iterative solution technique

    TSI::NlnSolTech soltech_;  //!< kind of iteration technique or
                               //!< nonlinear solution technique

    TSI::ConvNorm normtypeinc_;        //!< convergence check for increments
    TSI::ConvNorm normtyperhs_;        //!< convergence check for residual forces
    Solid::ConvNorm normtypedisi_;     //!< convergence check for residual displacements
    Solid::ConvNorm normtypestrrhs_;   //!< convergence check for residual forces
    Thermo::ConvNorm normtypetempi_;   //!< convergence check for residual temperatures
    Thermo::ConvNorm normtypethrrhs_;  //!< convergence check for residual thermal forces

    TSI::BinaryOp combincrhs_;  //!< binary operator to combine increments and forces

    TSI::VectorNorm iternorm_;     //!< vector norm to check TSI values with
    TSI::VectorNorm iternormstr_;  //!< vector norm to check structural values with
    TSI::VectorNorm iternormthr_;  //!< vector norm to check thermal values with

    double tolinc_;     //!< tolerance for increment
    double tolrhs_;     //!< tolerance for rhs
    double toldisi_;    //!< tolerance for displacement increments
    double tolstrrhs_;  //!< tolerance for structural rhs
    double toltempi_;   //!< tolerance for temperature increments
    double tolthrrhs_;  //!< tolerance for thermal rhs

    double normrhs_;          //!< norm of residual forces
    double normrhsiter0_;     //!< norm of residual force of 1st iteration
    double norminc_;          //!< norm of residual unknowns
    double norminciter0_;     //!< norm of residual unknowns of 1st iteration
    double normdisi_;         //!< norm of residual displacements
    double normdisiiter0_;    //!< norm of residual displacements of 1st iteration
    double normstrrhs_;       //!< norm of structural residual forces
    double normstrrhsiter0_;  //!< norm of structural residual forces of 1st iteration
    double normtempi_;        //!< norm of residual temperatures
    double normtempiiter0_;   //!< norm of residual temperatures of 1st iteration
    double normthrrhs_;       //!< norm of thermal residual forces
    double normthrrhsiter0_;  //!< norm of thermal residual forces of 1st iteration

    int iter_;     //!< iteration step
    int itermax_;  //!< maximally permitted iterations
    int itermin_;  //!< minimally requested iteration

    const Teuchos::ParameterList& sdyn_;  //!< structural dynamic parameter list

    //! @name time-step control
    const TimeStepping::TimeStepControlSettings time_step_control_settings_;
    const bool allow_material_time_step_reduction_;
    size_t num_steps_at_current_dt_ = 0;
    std::deque<size_t> nonlinear_iterations_per_step_;
    //@}

    Teuchos::Time timernewton_;  //!< timer for measurement of solution time of newton iterations
    double dtsolve_;             //!< linear solver time
    double dtcmt_;               //!< contact evaluation time
    //@}

    //! @name Pseudo-transient continuation parameters

    double ptcdt_;  //!< pseudo time step size for PTC
    double dti_;    //!< scaling factor for PTC (initially 1/ptcdt_, then adapted)

    //@}

    //! @name line search parameters
    TSI::LineSearch ls_strategy_;
    double ls_step_length_;
    std::pair<double, double> last_iter_res_;
    //@}

    //! @name Various global forces

    //! rhs of TSI system
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;

    //! increment between Newton steps k and k+1 \f$\Delta{x}^{<k>}_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> iterinc_;

    //! global velocities \f${V}_{n+1}\f$ at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel_;

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_;

    //@}

    //! @name infnorm scaling

    std::shared_ptr<Core::LinAlg::Vector<double>>
        srowsum_;  //!< sum of absolute values of the rows of the structural block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        scolsum_;  //!< sum of absolute values of the column of the structural block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        trowsum_;  //!< sum of absolute values of the rows of the thermal block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        tcolsum_;  //!< sum of absolute values of the column of the thermal block

    //@}

  };  // Monolithic

}  // namespace TSI


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
