// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROELAST_MONOLITHIC_HPP
#define FOUR_C_POROELAST_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_poroelast_base.hpp"
#include "4C_poroelast_input.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_structure_new_input.hpp"

namespace Teuchos
{
  class Time;
}

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;

  class BlockSparseMatrixBase;
  class Solver;

  class Equilibration;
  enum class EquilibrationMethod;
}  // namespace Core::LinAlg

namespace PoroElast
{
  //! base class of all monolithic Poroelasticity algorithms
  class Monolithic : public PoroBase
  {
   public:
    Monolithic(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
        std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter);

    /*! do the setup for the monolithic system


     1.) setup coupling
     2.) get maps for all blocks in the system (and for the whole system as well)
     create combined map
     3.) create system matrix


     \note We want to do this setup after reading the restart information, not
     directly in the constructor. This is necessary since during restart (if
     read_mesh is called), the dofmaps for the blocks might get invalid.
     */
    //! Setup the monolithic Poroelasticity system
    void setup_system() override;

    //! setup composed right hand side from field solvers
    void setup_rhs(bool firstcall = false) override;

    //! start a new time step
    void prepare_time_step() override;

    //! setup composed system matrix from field solvers
    virtual void setup_system_matrix() { setup_system_matrix(*systemmatrix_); }

    //! setup composed system matrix from field solvers
    virtual void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat);

    //! setup equilibration of system matrix
    void setup_equilibration();

    //! setup newton solver
    virtual void setup_newton();


    //! build the combined dirichletbcmap
    void build_combined_dbc_map() override;

    //! @name Access methods for subclasses

    //! extractor to communicate between full monolithic map and block maps
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const override
    {
      return blockrowdofmap_;
    }

    //!@}

    //! @name Access methods

    //! composed system matrix
    // remove this method!
    // this method merges the block matrix when called.
    // As this is very expensive this,this method is not meant to be used any more.
    // Use block_system_matrix() instead and assemble the blocks separately, if necessary.
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override;

    //! block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return systemmatrix_;
    }

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override;

    //! dof row map of Structure field
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map_structure() override;

    //! dof row map of Fluid field
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map_fluid() override;

    //! unique map of all dofs that should be constrained with DBC
    std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const override
    {
      return combinedDBCMap_;
    }

    //! right hand side vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override { return rhs_; }

    //! zero all entries in iterinc vector
    void clear_poro_iterinc();

    //! replaces the iterinc with poroinc
    void update_poro_iterinc(const Core::LinAlg::Vector<double>& poroinc);

    //! iter_ += 1
    void increment_poro_iter();

    //! fluid_field()->system_matrix()->RangeMap()
    const Core::LinAlg::Map& fluid_range_map();

    //! fluid_field()->system_matrix()->DomainMap()
    const Core::LinAlg::Map& fluid_domain_map();

    //! structure_field()->system_matrix()->DomainMap()
    const Core::LinAlg::Map& structure_domain_map();

    //!@}

    //! solve linear system
    void linear_solve();

    //! create linear solver (setup of parameter lists, etc...)
    void create_linear_solver();

    //! update all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    void update_state_incrementally(std::shared_ptr<const Core::LinAlg::Vector<double>>
            iterinc  //!< increment between iteration i and i+1
        ) override;

    //! update all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc (with structural and fluid
    //! increment separately)
    void update_state_incrementally(std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc,
        std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc);

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    //! and assemble systemmatrix and rhs-vector
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
                      iterinc,  //!< increment between iteration i and i+1
        bool firstiter) override;

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    //! and assemble systemmatrix and rhs-vector
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
                      s_iterinc,  //!< structural increment between iteration i and i+1
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            f_iterinc,  //!< fluid increment between iteration i and i+1
        bool firstiter) override;

    //! evaluate fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc);

    //! evaluate fields separately at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc,
        std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc);

    //! extract initial guess from fields
    //! returns \f$\Delta x_{n+1}^{<k>}\f$
    virtual void initial_guess(std::shared_ptr<Core::LinAlg::Vector<double>> ig);

    //! is convergence reached of iterative solution technique?
    //! keep your fingers crossed...
    virtual bool converged();

    //! inner newton iteration
    void solve() override;

    //! perform one time step (setup + solve + output)
    void do_time_step() override;

    //! @name Output

    //! print to screen information about residual forces and displacements
    virtual void print_newton_iter();

    //! contains text to print_newton_iter
    virtual void print_newton_iter_text(FILE* ofile  //!< output file handle
    );

    //! contains text to print_newton_iter
    virtual void print_newton_iter_text_stream(std::ostringstream& oss);

    //! contains header to print_newton_iter
    virtual void print_newton_iter_header(FILE* ofile  //!< output file handle
    );

    //! contains header to print_newton_iter
    virtual void print_newton_iter_header_stream(std::ostringstream& oss);

    //! print statistics of converged Newton-Raphson iteration
    void print_newton_conv();

    //!@}

    //! finite difference check of stiffness matrix
    [[maybe_unused]] void poro_fd_check();

    //! Evaluate no penetration condition
    void evaluate_condition(Core::LinAlg::SparseOperator& Sysmat,
        PoroElast::Coupltype coupltype = PoroElast::fluidfluid);

    //! recover Lagrange multiplier \f$\lambda_\Gamma\f$ at the interface at the end of each time
    //! step (i.e. condensed forces onto the structure) needed for rhs in next time step
    virtual void recover_lagrange_multiplier_after_time_step() {}

    //! recover Lagrange multiplier \f$\lambda_\Gamma\f$ at the interface at the end of each
    //! iteration step (i.e. condensed forces onto the structure) needed for rhs in next time step
    virtual void recover_lagrange_multiplier_after_newton_step(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc);

    //! Setup solver for monolithic system
    bool setup_solver() override;

    //! read restart data
    void read_restart(const int step) override;

    void post_setup() override;

   protected:
    //! Aitken
    void aitken();

    //! Aitken Reset
    [[maybe_unused]] void aitken_reset();

    //! @name Apply current field state to system

    //! Evaluate mechanical-fluid system matrix
    virtual void apply_str_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_sf  //!< mechanical-fluid stiffness matrix
    );

    //! Evaluate fluid-mechanical system matrix
    virtual void apply_fluid_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_fs  //!< fluid-mechanical tangent matrix
    );

    //!@}

    //! convergence check for Newton solver
    virtual void build_convergence_norms();

    //! extract the field vectors from a given composed vector. Different for fluid and structure
    //! split
    /*!
     x is the sum of all increments up to this point.
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) structural vector (e.g. displacements)
     \param fx (o) fluid vector (e.g. velocities and pressure)
     */
    virtual void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx, bool firstcall = false);

    //! @name General purpose algorithm members
    //!@{

    bool solveradapttol_;                           //!< adapt solver tolerance
    double solveradaptolbetter_;                    //!< tolerance to which is adapted ????
    std::shared_ptr<Core::LinAlg::Solver> solver_;  //!< linear algebraic solver

    //!@}

    //! @name Printing and output
    //!@{

    int printscreen_;  //!< print infos to standard out every printscreen_ steps
    bool printiter_;   //!< print intermediate iterations during solution

    //!@}

    //! @name Global vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  //!< a zero vector of full length

    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;  //!< rhs of Poroelasticity system

    //!@}

    Solid::DynamicType strmethodname_;  //!< enum for STR time integration

    //! @name Global matrixes

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //! structure-fluid coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sf_;
    //! fluid-structure coupling matrix
    std::shared_ptr<Core::LinAlg::SparseOperator> k_fs_;

    //!@}

    //! dof row map (not split)
    std::shared_ptr<Core::LinAlg::Map> fullmap_;

    //! dof row map split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_;

    //! dirichlet map of monolithic system
    std::shared_ptr<Core::LinAlg::Map> combinedDBCMap_;

    //! return structure fluid coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> struct_fluid_coupling_matrix();

    //! return fluid structure coupling sparse matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> fluid_struct_coupling_matrix();

    //! return structure fluid coupling block sparse matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> struct_fluid_coupling_block_matrix();

    //! return fluid structure coupling block sparse matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fluid_struct_coupling_block_matrix();


    //! @name poro-contact

    //! setup flags associated with contact
    void setup_poro_contact_flags();

    //! apply current velocity of fluid  to ContactManager if contact problem
    void set_poro_contact_states();

    //! assemble relevant matrixes for porocontact and meshtying
    void eval_poro_mortar();

    //! flag activation poro contact no penetration condition
    bool no_penetration_;

    //!@}

    //! build block vector from field vectors, e.g. rhs, increment vector
    virtual void setup_vector(Core::LinAlg::Vector<double>& f,  //!< vector of length of all dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            sv,  //!< vector containing only structural dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            fv  //!< vector containing only fluid dofs
    );

    //! @name Iterative solution technique

    PoroElast::ConvNorm normtypeinc_;       //!< convergence check for residual temperatures
    PoroElast::ConvNorm normtypefres_;      //!< convergence check for residual forces
    PoroElast::BinaryOp combincfres_;       //!< binary operator to combine temperatures and forces
    PoroElast::VectorNorm vectornormfres_;  //!< type of norm for residual
    PoroElast::VectorNorm vectornorminc_;   //!< type of norm for increments

    double tolinc_;   //!< tolerance residual increment
    double tolfres_;  //!< tolerance force residual

    double tolinc_struct_;   //!< tolerance residual increment for structure displacements
    double tolfres_struct_;  //!< tolerance force residual for structure displacements

    double tolinc_velocity_;   //!< tolerance residual increment for fluid velocity field
    double tolfres_velocity_;  //!< tolerance force residual for fluid velocity field

    double tolinc_pressure_;   //!< tolerance residual increment for fluid pressure field
    double tolfres_pressure_;  //!< tolerance force residual for fluid pressure field

    double tolinc_porosity_;   //!< tolerance residual increment for porosity field
    double tolfres_porosity_;  //!< tolerance force residual for porosity field

    int itermax_;     //!< maximally permitted iterations
    int itermin_;     //!< minimally requested iteration
    double normrhs_;  //!< norm of residual forces
    double norminc_;  //!< norm of residual unknowns

    double normrhsfluidvel_;   //!< norm of residual forces (fluid velocity)
    double normincfluidvel_;   //!< norm of residual unknowns (fluid velocity)
    double normrhsfluidpres_;  //!< norm of residual forces (fluid pressure)
    double normincfluidpres_;  //!< norm of residual unknowns (fluid pressure)
    double normrhsfluid_;      //!< norm of residual forces (fluid )
    double normincfluid_;      //!< norm of residual unknowns (fluid )

    double normrhsstruct_;  //!< norm of residual forces (structure)
    double normincstruct_;  //!< norm of residual unknowns (structure)

    double normrhsporo_;  //!< norm of residual forces (porosity)
    double normincporo_;  //!< norm of residual unknowns (porosity)

    std::shared_ptr<Teuchos::Time> timer_;  //!< timer for solution technique

    int iter_;  //!< iteration step

    //!@}

    //! @name Various global forces

    std::shared_ptr<Core::LinAlg::Vector<double>>
        iterinc_;  //!< increment between Newton steps k and k+1
    //!< \f$\Delta{x}^{<k>}_{n+1}\f$

    //!@}

    //! flag for direct solver
    bool directsolve_;

    //! @name Aitken relaxation

    //! difference of last two solutions
    // del = r^{i+1}_{n+1} = d^{i+1}_{n+1} - d^i_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> del_;
    //! difference of difference of last two pair of solutions
    // delhist = ( r^{i+1}_{n+1} - r^i_{n+1} )
    std::shared_ptr<Core::LinAlg::Vector<double>> delhist_;
    //! Aitken factor
    double mu_;
    //!@}

    //! @name matrix equilibration

    //! all equilibration of global system matrix and RHS is done in here
    std::shared_ptr<Core::LinAlg::Equilibration> equilibration_;

    //! equilibration method applied to system matrix
    Core::LinAlg::EquilibrationMethod equilibration_method_;
    //!@}

    //!@}
  };

}  // namespace PoroElast


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
