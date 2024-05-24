/*----------------------------------------------------------------------*/
/*! \file

\brief general Field adapter - In the end should be the base class for all
'fields' like structure, fluid, ale, poro,... (in principle every coupled
problem can be seen as field, if it is coupled into a bigger system).


This base class makes it easier to use one algorithm for different
subfields! (e.g. basically same algorithm for fsi_xfem and fpsi_xfem)

At the moment this class is on the same level as AlgorithmBase for Algorithms
... It would be an option to use this class also as Base Class for AlgorithmBase as most of the
functions there exist also for fields.
(e.g. Dt(),....) --> Should be discussed!


\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_ADAPTER_FIELD_HPP
#define FOUR_C_ADAPTER_FIELD_HPP

// includes
#include "4C_config.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations:
namespace DRT
{
  class Discretization;
}

namespace CORE::LINALG
{
  class SparseMatrix;
  class BlockSparseMatrixBase;
}  // namespace CORE::LINALG

namespace ADAPTER
{
  /// general field interface

  class Field
  {
   public:
    /*!
    \brief Type of Field hold by the adapter

    */

    //! @name Destruction
    //@{

    /// virtual to get polymorph destruction
    virtual ~Field() = default;
    //! @name Vector access

    /// Return the already evaluated RHS of Newton's method
    virtual Teuchos::RCP<const Epetra_Vector> RHS() = 0;

    //@}

    //! @name Misc
    //@{

    /// dof map of vector of unknowns
    virtual Teuchos::RCP<const Epetra_Map> dof_row_map() = 0;

    /// direct access to system matrix
    virtual Teuchos::RCP<CORE::LINALG::SparseMatrix> SystemMatrix() = 0;

    /// direct access to system matrix
    virtual Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> BlockSystemMatrix() = 0;

    //@}

    //! @name Time step helpers
    //@{

    /// start new time step
    virtual void prepare_time_step() = 0;

    /// Update state with prescribed increment vector
    /*!
    \brief update dofs

    There are two dof increments possible

    \f$x^n+1_i+1 = x^n+1_i + disiterinc\f$  (sometimes referred to as residual increment), and

    \f$x^n+1_i+1 = x^n     + disstepinc\f$

    with \f$n\f$ and \f$i\f$ being time and Newton iteration step

    Note: Fields expect an iteration increment.
    In case the StructureNOXCorrectionWrapper is applied, the step increment is expected
    which is then transformed into an iteration increment
    */
    virtual void update_state_incrementally(
        Teuchos::RCP<const Epetra_Vector> disi  ///< iterative solution increment
        ) = 0;

    /*!
    \brief update dofs and evaluate elements

    There are two dof increments possible

    \f$x^n+1_i+1 = x^n+1_i + disiterinc\f$  (sometimes referred to as residual increment), and

    \f$x^n+1_i+1 = x^n     + disstepinc\f$

    with \f$n\f$ and \f$i\f$ being time and Newton iteration step

    Note: Field Expects an iteration increment.
    In case the StructureNOXCorrectionWrapper is applied, the step increment is expected
    which is then transformed into an iteration increment
    */
    virtual void Evaluate(
        Teuchos::RCP<const Epetra_Vector> iterinc  ///< dof increment between Newton iteration i and
                                                   ///< i+1 or between timestep n and n+1
        ) = 0;

    /// Evaluate with different eval. for first iteration, has to be overload by relevant fields
    /// (coupled fields)
    virtual void Evaluate(
        Teuchos::RCP<const Epetra_Vector> iterinc,  ///< dof increment between Newton iteration i
                                                    ///< and i+1 or between timestep n and n+1
        bool firstiter)
    {
      Evaluate(iterinc);
    }

    /// update at time step end
    virtual void Update() = 0;

    /// prepare output (i.e. calculate stresses, strains, energies)
    virtual void prepare_output(bool force_prepare_timestep) = 0;

    /// output results
    virtual void Output(bool forced_writerestart = false) = 0;

    /// read restart information for given time step
    virtual void read_restart(const int step) = 0;

    //@}
  };

}  // namespace ADAPTER

FOUR_C_NAMESPACE_CLOSE

#endif
