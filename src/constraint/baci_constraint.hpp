/*----------------------------------------------------------------------*/
/*! \file

\brief Basic constraint class, dealing with constraints living on boundaries, code originally by
Thomas Kloeppel


\level 2

*----------------------------------------------------------------------*/

#ifndef BACI_CONSTRAINT_HPP
#define BACI_CONSTRAINT_HPP

#include "baci_config.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN

// forward declarations
namespace DRT
{
  class Condition;
  class Discretization;
}  // namespace DRT

namespace CORE::LINALG
{
  class SparseOperator;
}

namespace CONSTRAINTS
{
  /*!
  \brief Basic constraint class, dealing with constraint and monitor boundary conditions.
  This class cannot handle multi point constraints, they will be dealt with by a derived class.
  */
  class Constraint

  {
   public:
    //! Constraint types
    enum ConstrType
    {
      none,
      volconstr3d,
      areaconstr3d,
      areaconstr2d,
      mpcnodeonplane3d,
      mpcnormalcomp3d,
      mpcnodeonline2d
    };

    /*!
    \brief Constructor of a constraint based on a conditions with a given name. It also
    takes care of the constraint IDs.
    */

    Constraint(Teuchos::RCP<DRT::Discretization> discr,  ///< Discretization constraint lives on
        const std::string& conditionname,  ///< Name of condition to creat constraint from
        int& minID,                        ///< minimum constraint or monitor ID so far
        int& maxID                         ///< maximum constraint or monitor ID so far
    );

    /*!
    \brief Constructor of a constraint based on a conditions with a given name.
    */

    Constraint(Teuchos::RCP<DRT::Discretization> discr,  ///< Discretization constraint lives on
        const std::string& conditionname  ///< Name of condition to create constraints from
    );


    /*!
     \brief Return if there are constraints
    */
    bool HaveConstraint() { return constrtype_ != none; };

    /// initialization routine called by the manager ctor to get correct reference base values and
    /// activating the right conditions at the beginning
    void Initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Epetra_Vector> systemvector3  ///< distributed vector that may be filled by
                                                   ///< assembly of element contributions
    );

    /// initialization routine called at restart to activate the right conditions
    void Initialize(const double& time  ///< current time
    );

    //! Evaluate routine to call from outside. In here the right action is determined and the
    //! #EvaluateConstraint routine is called
    void Evaluate(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<CORE::LINALG::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        Teuchos::RCP<CORE::LINALG::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        Teuchos::RCP<Epetra_Vector> systemvector1,  ///< distributed vector that may be filled by
                                                    ///< assembly of element contributions
        Teuchos::RCP<Epetra_Vector> systemvector2,  ///< distributed vector that may be filled by
                                                    ///< assembly of element contributions
        Teuchos::RCP<Epetra_Vector> systemvector3   ///< distributed vector that may be filled by
                                                    ///< assembly of element contributions
    );

    /// Return type of constraint
    ConstrType Type() { return constrtype_; }

    /// Return vector with IDs of active conditions
    std::vector<int> GetActiveCondID();

   protected:
    Teuchos::RCP<DRT::Discretization> actdisc_;  ///< standard discretization
    std::vector<DRT::Condition*>
        constrcond_;         ///< conditions, that define the constraint (all of the same kind)
    ConstrType constrtype_;  ///< constraint type
    std::map<int, double>
        inittimes_;  ///< map with times at which constraint is supposed to become active
    std::map<int, bool> activecons_;  ///< map with indicator if constraints are active

   private:
    // don't want = operator, cctor and destructor

    Constraint operator=(const Constraint& old);
    Constraint(const Constraint& old);

    //! Return the ConstrType based on the condition name
    ConstrType GetConstrType(const std::string& Name  ///< condition name
    );

    //! Evaluate constraint conditions and assemble the results
    void EvaluateConstraint(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<CORE::LINALG::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        Teuchos::RCP<CORE::LINALG::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        Teuchos::RCP<Epetra_Vector> systemvector1,  ///< distributed vector that may be filled by
                                                    ///< aasembly of element contributions
        Teuchos::RCP<Epetra_Vector> systemvector2,  ///< distributed vector that may be filled by
                                                    ///< aasembly of element contributions
        Teuchos::RCP<Epetra_Vector> systemvector3   ///< distributed vector that may be filled by
                                                    ///< aasembly of element contributions
    );

    //! Compute and assemble initial constraint values (depending on user specific activation times)
    void InitializeConstraint(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Epetra_Vector> systemvector  ///< distributed vector that may be filled by
                                                  ///< aasembly of element contributions
    );
  };  // class
}  // namespace CONSTRAINTS

BACI_NAMESPACE_CLOSE

#endif  // CONSTRAINT_H
