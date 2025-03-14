// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_EHL_MONOLITHIC_HPP
#define FOUR_C_EHL_MONOLITHIC_HPP


/*----------------------------------------------------------------------*
 | headers                                                  wirtz 01/16 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_ehl_base.hpp"
#include "4C_ehl_input.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_lubrication_input.hpp"

#include <Epetra_FEVector.h>
#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                          wirtz 01/16 |
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


namespace FSI
{
  namespace Utils
  {
    class MatrixRowTransform;
    class MatrixColTransform;
    class MatrixRowColTransform;
  }  // namespace Utils
}  // namespace FSI

namespace Mortar
{
  class IntCell;
  class Element;
}  // namespace Mortar

namespace CONTACT
{
  class Element;
}

namespace EHL
{
  //! monolithic EHL algorithm
  //!
  //!  Base class of EHL algorithms. Derives from structure_base_algorithm and
  //!  LubricationBaseAlgorithm with pressure field.
  //!  There can (and will) be different subclasses that implement different
  //!  coupling schemes.
  //!
  //!  \warning The order of calling the two BaseAlgorithm-constructors (that
  //!  is the order in which we list the base classes) is important here! In the
  //!  constructors control file entries are written. And these entries define
  //!  the order in which the filters handle the Discretizations, which in turn
  //!  defines the dof number ordering of the Discretizations... Don't get
  //!  confused. Just always list structure, lubrication. In that order.
  //!
  //!  \note There is the Algorithm class for general purpose EHL algorithms.
  //!  This simplifies the monolithic implementation.
  class Monolithic : public Base
  {
   public:
    explicit Monolithic(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& lubricationparams, const Teuchos::ParameterList& structparams,
        const std::string struct_disname, const std::string lubrication_disname);



    /*! do the setup for the monolithic system


    1.) setup coupling
    2.) get maps for all blocks in the system (and for the whole system as well)
        create combined map
    3.) create system matrix


    \note We want to do this setup after reading the restart information, not
    directly in the constructor. This is necessary since during restart (if
    read_mesh is called), the dofmaps for the blocks might get invalid.
    */
    //! Setup the monolithic EHL system
    void setup_system() override;

    /// non-linear solve, i.e. (multiple) corrector
    virtual void solve();

    //! outer level EHL time loop
    void timeloop() override;

    //! @name Apply current field state to system

    //! setup composed right hand side from field solvers
    void setup_rhs();

    //! setup composed system matrix from field solvers
    void setup_system_matrix();

    //! apply all Dirichlet boundary conditions
    void apply_dbc();

    //! composed system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> system_matrix() const
    {
      return systemmatrix_;
    }

    //! solve linear EHL system
    void linear_solve();

    //! Evaluate lubrication-mechanical system matrix
    void apply_lubrication_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseMatrix>
            matheight,  //!< lubrication matrix associated with linearization wrt height
        std::shared_ptr<Core::LinAlg::SparseMatrix>
            matvel  //!< lubrication matrix associated with linearization wrt velocities
    );

    void lin_pressure_force_disp(
        Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd);
    void lin_poiseuille_force_disp(
        Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd);
    void lin_couette_force_disp(
        Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd);

    void lin_pressure_force_pres(
        Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp);
    void lin_poiseuille_force_pres(
        Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp);
    void lin_couette_force_pres(
        Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp);
    //@}

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    virtual void evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> stepinc);

    //! is convergence reached of iterative solution technique?
    //! keep your fingers crossed...
    bool converged();

    //! outer iteration loop
    void newton_full();

    //! @name Output

    //! print to screen information about residual forces and displacements
    void print_newton_iter();

    //! contains text to print_newton_iter
    void print_newton_iter_text(FILE* ofile  //!< output file handle
    );

    //! contains header to print_newton_iter
    void print_newton_iter_header(FILE* ofile  //!< output file handle
    );

    //! print statistics of converged Newton-Raphson iteration
    void print_newton_conv();

    //! Determine norm of force residual
    double calculate_vector_norm(const enum EHL::VectorNorm norm,  //!< norm to use
        Core::LinAlg::Vector<double>& vect                         //!< the vector of interest
    );

    //@}

    //! apply infnorm scaling to linear block system
    virtual void scale_system(
        Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b);

    //! undo infnorm scaling from scaled solution
    virtual void unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
        Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b);

   protected:
    //! @name Time loop building blocks

    //! start a new time step
    void prepare_time_step() override;

    //! calculate stresses, strains, energies
    virtual void prepare_output(bool force_prepare);
    //@}

    //! convergence check for Newton solver
    bool convergence_check(int itnum, int itmax, double ittol);

    //! extract the three field vectors from a given composed vector
    /*!
      x is the sum of all increments up to this point.
      \param x  (i) composed vector that contains all field vectors
      \param sx (o) structural vector (e.g. displacements)
      \param lx (o) lubrication vector (e.g. pressures)
      */
    virtual void extract_field_vectors(std::shared_ptr<Core::LinAlg::Vector<double>> x,
        std::shared_ptr<Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<Core::LinAlg::Vector<double>>& lx);

    //! @name Access methods for subclasses

    //! full monolithic dof row map
    std::shared_ptr<const Epetra_Map> dof_row_map() const;

    //! set full monolithic dof row map
    /*!
     A subclass calls this method (from its constructor) and thereby
     defines the number of blocks, their maps and the block order. The block
     maps must be row maps by themselves and must not contain identical GIDs.
    */
    void set_dof_row_maps(const std::vector<std::shared_ptr<const Epetra_Map>>& maps);

    //! combined DBC map
    //! unique map of all dofs that should be constrained with DBC
    std::shared_ptr<Epetra_Map> combined_dbc_map();

    //! extractor to communicate between full monolithic map and block maps
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor() const { return blockrowdofmap_; }

    //! setup list with default parameters
    void set_default_parameters();

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
    enum Inpar::Solid::DynamicType strmethodname_;

   private:
    const Teuchos::ParameterList& ehldyn_;      //!< EHL dynamic parameter list
    const Teuchos::ParameterList& ehldynmono_;  //!< monolithic EHL dynamic parameter list

    //! dofrowmap split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_;

    //! build block vector from field vectors, e.g. rhs, increment vector
    void setup_vector(Core::LinAlg::Vector<double>& f,  //!< vector of length of all dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            sv,  //!< vector containing only structural dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            lv  //!< vector containing only lubrication dofs
    );

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;

    //! off diagonal matrixes
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_sl_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_ls_;

    //! @name iterative solution technique

    enum EHL::ConvNorm normtypeinc_;              //!< convergence check for increments
    enum EHL::ConvNorm normtyperhs_;              //!< convergence check for residual forces
    enum Inpar::Solid::ConvNorm normtypedisi_;    //!< convergence check for residual displacements
    enum Inpar::Solid::ConvNorm normtypestrrhs_;  //!< convergence check for residual forces
    enum Lubrication::ConvNorm normtypeprei_;     //!< convergence check for residual pressures
    enum Lubrication::ConvNorm
        normtypelubricationrhs_;  //!< convergence check for residual lubrication forces

    enum EHL::BinaryOp combincrhs_;  //!< binary operator to combine increments and forces

    enum EHL::VectorNorm iternorm_;             //!< vector norm to check EHL values with
    enum EHL::VectorNorm iternormstr_;          //!< vector norm to check structural values with
    enum EHL::VectorNorm iternormlubrication_;  //!< vector norm to check lubrication values with

    double tolinc_;             //!< tolerance for increment
    double tolrhs_;             //!< tolerance for rhs
    double toldisi_;            //!< tolerance for displacement increments
    double tolstrrhs_;          //!< tolerance for structural rhs
    double tolprei_;            //!< tolerance for pressure increments
    double tollubricationrhs_;  //!< tolerance for lubrication rhs

    double normrhs_;                  //!< norm of residual forces
    double normrhsiter0_;             //!< norm of residual force of 1st iteration
    double norminc_;                  //!< norm of residual unknowns
    double norminciter0_;             //!< norm of residual unknowns of 1st iteration
    double normdisi_;                 //!< norm of residual displacements
    double normdisiiter0_;            //!< norm of residual displacements of 1st iteration
    double normstrrhs_;               //!< norm of structural residual forces
    double normstrrhsiter0_;          //!< norm of structural residual forces of 1st iteration
    double normprei_;                 //!< norm of residual pressures
    double normpreiiter0_;            //!< norm of residual pressures of 1st iteration
    double normlubricationrhs_;       //!< norm of lubrication residual forces
    double normlubricationrhsiter0_;  //!< norm of lubrication residual forces of 1st iteration

    int iter_;     //!< iteration step
    int itermax_;  //!< maximally permitted iterations
    int itermin_;  //!< minimally requested iteration

    const Teuchos::ParameterList& sdyn_;  //!< structural dynamic parameter list

    Teuchos::Time timernewton_;  //!< timer for measurement of solution time of newton iterations
    double dtsolve_;             //!< linear solver time
    //@}

    //! @name Various global forces

    //! rhs of EHL system
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;

    //! increment between Newton steps k and k+1 \f$\Delta{x}^{<k>}_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> iterinc_;

    //@}

    //! @name infnorm scaling

    std::shared_ptr<Core::LinAlg::Vector<double>>
        srowsum_;  //!< sum of absolute values of the rows of the structural block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        scolsum_;  //!< sum of absolute values of the column of the structural block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        lrowsum_;  //!< sum of absolute values of the rows of the lubrication block
    std::shared_ptr<Core::LinAlg::Vector<double>>
        lcolsum_;  //!< sum of absolute values of the column of the lubrication block

    //@}

  };  // Monolithic

}  // namespace EHL


FOUR_C_NAMESPACE_CLOSE

#endif
