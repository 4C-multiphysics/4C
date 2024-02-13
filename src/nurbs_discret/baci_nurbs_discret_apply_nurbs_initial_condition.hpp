/*----------------------------------------------------------------------*/
/*! \file

\brief A service method allowing the application of initial conditions
       for nurbs discretisations.

Since nurbs shape functions are not interpolating, it is not as
straightforward to apply initial conditions to the degrees of freedom.
(dofs are always associated with control points, i.e. the location
associated with the 'node'=control point is not the physical location
and the value at the control point is not the prescribed value at this
position since dofs associated with neighbouring control points influence
the function value as well)



\level 2
*/
/*----------------------------------------------------------------------*/
#ifndef BACI_NURBS_DISCRET_APPLY_NURBS_INITIAL_CONDITION_HPP
#define BACI_NURBS_DISCRET_APPLY_NURBS_INITIAL_CONDITION_HPP

#include "baci_config.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN

// forward declarations
namespace DRT
{
  class Discretization;
}
namespace CORE::LINALG
{
  class Solver;
}

namespace DRT
{
  namespace NURBS
  {
    /*----------------------------------------------------------------------*/
    /*!
    \brief A service method allowing the application of initial conditions
           for nurbs discretisations. Recommended version with separate
           solver allocation

    \param dis          (i) the discretisation
    \param solverparams (i) a list with solver parameters
    \param startfuncno  (i) the number of the startfunction defining
                           the initial field (i.e. u_0(x))
    \param initialvals  (o) the initial field on output (i.e. u_cp)

    \date 08/11
    */
    void apply_nurbs_initial_condition(DRT::Discretization& dis,
        const Teuchos::ParameterList& solverparams, const int startfuncno,
        Teuchos::RCP<Epetra_Vector> initialvals);

    /*----------------------------------------------------------------------*/
    /*!
    \brief A service method allowing the application of initial conditions
           for nurbs discretisations.

    This method provides the following:

    Given an initial distribution u_0(x) of initial values (for example by a
    spatial function), we compute control point values u_cp such that they
    minimize the following least-squares problem:

                          ||                                   || 2
                      || +----+                            ||
                      ||  \                                ||
                 min  ||   +    N   (x) * u     -  u   (x) ||
                      ||  /      cp       - cp     - 0     ||
                 u    || +----+                            ||
                 - cp ||   cp                              ||


    This is equivalent to the solution of the following linear system:


             /                         \               /                         \
            |    /                      |             |    /                      |
     +----+ |   |                       |             |   |                       |
      \     |   |                       |    dim      |   |            dim        |
       +    |   | N   (x) * N   (x) dx  | * u     =   |   | N   (x) * u   (x) dx  |
      /     |   |  cp        cp         |    cp       |   |  cp        0          |
     +----+ |   |    j         i        |      j      |   |    j                  |
       cp   |  /                        |             |  /                        |
         j   \                         /               \                         /

            |                           |             |                           |
            +---------------------------+             +---------------------------+

                   M(assmatrix)                                   rhs



    \param dis         (i) the discretisation
    \param solver      (i) a solver object for the least-squares problem
    \param startfuncno (i) the number of the startfunction defining
                           the initial field (i.e. u_0(x))
    \param initialvals (o) the initial field on output (i.e. u_cp)

    \date 04/09
    */
    void apply_nurbs_initial_condition_solve(DRT::Discretization& dis, CORE::LINALG::Solver& solver,
        const int startfuncno, Teuchos::RCP<Epetra_Vector> initialvals);


  }  // namespace NURBS

}  // namespace DRT

BACI_NAMESPACE_CLOSE

#endif  // NURBS_DISCRET_APPLY_NURBS_INITIAL_CONDITION_H
