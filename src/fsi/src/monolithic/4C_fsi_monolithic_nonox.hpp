/*----------------------------------------------------------------------*/
/*! \file
\brief Base class for monolithic fluid-fluid-fsi algorithm
 using XFEM (without NOX)

\level 2

*----------------------------------------------------------------------*/

#ifndef FOUR_C_FSI_MONOLITHIC_NONOX_HPP
#define FOUR_C_FSI_MONOLITHIC_NONOX_HPP

#include "4C_config.hpp"

#include "4C_fsi_monolithic.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_xfem.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class FluidFluidFSI;
  class AleXFFsiWrapper;
}  // namespace Adapter

namespace FSI
{
  namespace UTILS
  {
    class DebugWriter;
    class MonolithicDebugWriter;
  }  // namespace UTILS

  class MonolithicNoNOX : public FSI::MonolithicBase, public FSI::MonolithicInterface
  {
    friend class FSI::UTILS::MonolithicDebugWriter;

   public:
    explicit MonolithicNoNOX(const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams);

    ///
    /*! do the setup for the monolithic system


    1.) setup coupling
    2.) get maps for all blocks in the system (and for the whole system as well)
    3.) if necessary, define system block matrix


    \note We want to do this setup after reading the restart information, not
    directly in the constructor. This is necessary since during restart (if
    read_mesh is called), the dofmaps for the blocks might get invalid.

    */
    virtual void SetupSystem();

    /// outer level FSI time loop
    void Timeloop();

   protected:
    /// time update
    /// recover the Lagrange multiplier and relax ALE (if requested)
    void update() override;

    /// start a new time step
    void prepare_time_step() override;

    /// Prepare preconditioner for a new time step
    void prepare_time_step_preconditioner() override{};

    /// output of fluid, structure & ALE-quantities and Lagrange multiplier
    void output() override = 0;

    /// Evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    void evaluate(
        Teuchos::RCP<const Epetra_Vector> step_increment  ///< increment between time step n and n+1
    );

    //! @name Apply current field state to system

    //! setup composed right hand side from field solvers
    //!
    //! The RHS consists of three contributions from:
    //! 1) the single fields residuals
    //! 2) the Lagrange multiplier field lambda_
    //! 3) terms in the first nonlinear iteration
    void setup_rhs(Epetra_Vector& f,  ///< empty rhs vector (to be filled)
        bool firstcall  ///< indicates whether this is the first nonlinear iteration or not
        ) override = 0;

    /// setup composed system matrix from field solvers
    void setup_system_matrix() override = 0;
    //@}

    /// create merged map of DOF in the final system from all fields
    virtual void create_combined_dof_row_map() = 0;

    /// Newton Raphson
    virtual void newton();

    /// test for convergence
    bool converged();

    /// compute the residual and incremental norms required for convergence check
    virtual void build_convergence_norms() = 0;

    void linear_solve();

    /// create merged map with Dirichlet-constrained DOF from all fields
    virtual Teuchos::RCP<Epetra_Map> combined_dbc_map() = 0;

    //! Extract the three field vectors from a given composed vector
    //!
    //! In analogy to NOX, x is step increment \f$\Delta x\f$
    //! that brings us from \f$t^{n}\f$ to \f$t^{n+1}\f$:
    //! \f$x^{n+1} = x^{n} + \Delta x\f$
    //!
    //! Iteration increments, that are needed internally in the single fields,
    //! have to be computed somewhere else.
    //!
    //! \param x  (i) composed vector that contains all field vectors
    //! \param sx (o) structural displacements
    //! \param fx (o) fluid velocities and pressure
    //! \param ax (o) ale displacements
    virtual void extract_field_vectors(Teuchos::RCP<const Epetra_Vector> x,
        Teuchos::RCP<const Epetra_Vector>& sx, Teuchos::RCP<const Epetra_Vector>& fx,
        Teuchos::RCP<const Epetra_Vector>& ax) = 0;

    /// compute the Lagrange multiplier (FSI stresses) for the current time step
    virtual void recover_lagrange_multiplier() = 0;

    /// Extract initial guess from fields
    virtual void initial_guess(Teuchos::RCP<Epetra_Vector> ig) = 0;

    //! @name Methods for infnorm-scaling of the system

    /// apply infnorm scaling to linear block system
    void scale_system(Epetra_Vector& b) override {}

    /// undo infnorm scaling from scaled solution
    void unscale_solution(Epetra_Vector& x, Epetra_Vector& b) override {}

    //@}

    //! @name Output

    //! print to screen information about residual forces and displacements
    void print_newton_iter();

    //! contains text to print_newton_iter
    void print_newton_iter_text();

    //! contains header to print_newton_iter
    void print_newton_iter_header();

    //! print statistics of converged Newton-Raphson iteration
    // void print_newton_conv();

    //@}

    //! @name Access methods for subclasses

    /// get full monolithic dof row map
    Teuchos::RCP<const Epetra_Map> dof_row_map() const { return blockrowdofmap_.FullMap(); }

    //@}

    /// set full monolithic dof row map
    /*!
      A subclass calls this method (from its constructor) and thereby
      defines the number of blocks, their maps and the block order. The block
      maps must be row maps by themselves and must not contain identical GIDs.
     */
    void set_dof_row_maps(const std::vector<Teuchos::RCP<const Epetra_Map>>& maps);

    /// extractor to communicate between full monolithic map and block maps
    const Core::LinAlg::MultiMapExtractor& extractor() const { return blockrowdofmap_; }

    /// setup list with default parameters
    void set_default_parameters(const Teuchos::ParameterList& fsidyn, Teuchos::ParameterList& list);

    /*!
     * In case of a change in the fluid DOF row maps during the Newton loop (full Newton approach),
     * reset vectors accordingly.
     * \author kruse
     * \date 05/14
     */
    virtual void handle_fluid_dof_map_change_in_newton() = 0;

    /*!
     * Determine a change in fluid DOF map
     * \param (in) : DOF map of fluid increment vector
     * \return : true, in case of a mismatch between map of increment vector
     * and inner fluid DOF map after evaluation
     * \author kruse
     * \date 05/14
     */
    virtual bool has_fluid_dof_map_changed(const Epetra_BlockMap& fluidincrementmap) = 0;

    /// access type-cast pointer to problem-specific ALE-wrapper
    const Teuchos::RCP<Adapter::AleXFFsiWrapper>& ale_field() { return ale_; }

    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;  //!< block system matrix

    bool firstcall_;

    // sum of increments
    Teuchos::RCP<Epetra_Vector> x_sum_;

    int iter_;     //!< iteration step
    int itermax_;  //!< maximally permitted iterations

    int ns_;   //!< length of structural dofs
    int nf_;   //!< length of fluid dofs
    int ni_;   //!< length of fsi interface dofs
    int nfv_;  //!< length of fluid velocity dofs
    int nfp_;  //!< length of fluid pressure dofs
    int na_;   //!< length of ale dofs
    int nall_;

    double normrhs_;  //!< norm of residual forces
    double norminc_;  //!< norm of solution increment

    // L2-NORMS
    //--------------------------------------------------------------------------//
    double normstrrhsL2_;        //!< norm of structural residual
    double normflvelrhsL2_;      //!< norm of inner fluid velocity residual
    double normflpresrhsL2_;     //!< norm of fluid pressure residual
    double normalerhsL2_;        //!< norm of ale residual
    double norminterfacerhsL2_;  //!< norm of interface residual

    //--------------------------------------------------------------------------//
    double normstrincL2_;        //!< norm of inner structural increment
    double normflvelincL2_;      //!< norm of inner fluid velocity residual forces
    double normflpresincL2_;     //!< norm of fluid pressure residual forces
    double normaleincL2_;        //!< norm of ale residual forces
    double norminterfaceincL2_;  //!< norm of interface residual forces
    //--------------------------------------------------------------------------//

    // Inf-NORMS
    //--------------------------------------------------------------------------//
    double normstrrhsInf_;        //!< norm of structural residual
    double normflvelrhsInf_;      //!< norm of inner fluid velocity residual
    double normflpresrhsInf_;     //!< norm of fluid pressure residual
    double normalerhsInf_;        //!< norm of ale residual
    double norminterfacerhsInf_;  //!< norm of interface residual

    //--------------------------------------------------------------------------//
    double normstrincInf_;        //!< norm of inner structural increment
    double normflvelincInf_;      //!< norm of inner fluid velocity residual forces
    double normflpresincInf_;     //!< norm of fluid pressure residual forces
    double normaleincInf_;        //!< norm of ale residual forces
    double norminterfaceincInf_;  //!< norm of interface residual forces
    //--------------------------------------------------------------------------//

    Teuchos::RCP<Epetra_Vector> iterinc_;        //!< increment between Newton steps k and k+1
    Teuchos::RCP<Epetra_Vector> rhs_;            //!< rhs of FSI system
    Teuchos::RCP<Epetra_Vector> zeros_;          //!< a zero vector of full length
    Teuchos::RCP<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver

    /// type-cast pointer to problem-specific fluid-wrapper
    Teuchos::RCP<Adapter::FluidFluidFSI> fluid_;

   private:
    /*!
     * Check whether input parameters are appropriate
     * \author kruse
     * \date 05/14
     */
    void validate_parameters();

    /// dof row map splitted in (field) blocks
    Core::LinAlg::MultiMapExtractor blockrowdofmap_;

    /// output stream
    Teuchos::RCP<std::ofstream> log_;

    //! @name special debugging output
    //@{
    Teuchos::RCP<UTILS::DebugWriter> sdbg_;
    Teuchos::RCP<UTILS::DebugWriter> fdbg_;

    //@}

    //! @name Iterative solution technique
    //@{
    enum Inpar::FSI::ConvNorm normtypeinc_;   //!< convergence check for increment
    enum Inpar::FSI::ConvNorm normtypefres_;  //!< convergence check for residual forces
    enum Inpar::FSI::BinaryOp combincfres_;  //!< binary operator to combine temperatures and forces

    double tolinc_;   //!< tolerance residual temperatures
    double tolfres_;  //!< tolerance force residual

    double tol_dis_res_l2_;
    double tol_dis_res_inf_;
    double tol_dis_inc_l2_;
    double tol_dis_inc_inf_;
    double tol_fsi_res_l2_;
    double tol_fsi_res_inf_;
    double tol_fsi_inc_l2_;
    double tol_fsi_inc_inf_;
    double tol_pre_res_l2_;
    double tol_pre_res_inf_;
    double tol_pre_inc_l2_;
    double tol_pre_inc_inf_;
    double tol_vel_res_l2_;
    double tol_vel_res_inf_;
    double tol_vel_inc_l2_;
    double tol_vel_inc_inf_;
    //@}

    /// type-cast pointer to problem-specific ALE-wrapper
    Teuchos::RCP<Adapter::AleXFFsiWrapper> ale_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
