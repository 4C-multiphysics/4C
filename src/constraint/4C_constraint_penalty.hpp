/*----------------------------------------------------------------------*/
/*! \file

\brief Basic constraint class, dealing with constraints living on boundaries


\level 2

*----------------------------------------------------------------------*/

#ifndef FOUR_C_CONSTRAINT_PENALTY_HPP
#define FOUR_C_CONSTRAINT_PENALTY_HPP

#include "4C_config.hpp"

#include "4C_constraint.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Operator.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace CONSTRAINTS
{
  /*!
  \brief Basic constraint class, dealing with constraint and monitor boundary conditions.
  This class cannot handle multi point constraints, they will be dealt with by a derived class.
  */
  class ConstraintPenalty : public Constraint

  {
   public:
    /*!
    \brief Constructor of a constraint based on a conditions with a given name.
    */

    ConstraintPenalty(
        Teuchos::RCP<Core::FE::Discretization> discr,  ///< discretization constraint lives on
        const std::string& conditionname  ///< Name of condition to create constraints from
    );


    /// unused
    void initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::Vector> systemvector3  ///< distributed vector that may be filled
                                                          ///< by assembly of element contributions
    );

    /// initialization routine called by the manager ctor
    void initialize(Teuchos::ParameterList&
            params  ///< parameter list to communicate between elements and discretization
    );

    /// initialization routine called at restart to activate the right conditions
    void initialize(const double& time  ///< current time
    );

    //! Evaluate routine to call from outside. In here the right action is determined and the
    //! #evaluate_constraint routine is called
    void evaluate(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        Teuchos::RCP<Core::LinAlg::Vector>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector> systemvector3  ///< distributed vector that may be filled
                                                          ///< by assembly of element contributions
    );

   protected:
    std::map<int, double> penalties_;          ///< map containing penalty values
    std::map<int, double> rho_;                ///< map containing rhos for augmented Lagrange
    Teuchos::RCP<Epetra_Export> errorexport_;  ///< exporter for redundant and non-overlapping maps
    Teuchos::RCP<Epetra_Import> errorimport_;  ///< importer for redundant and non-overlapping maps
    Teuchos::RCP<Epetra_Map> rederrormap_;     ///< redundant map of errors
    Teuchos::RCP<Epetra_Map> errormap_;        ///< non-overlapping map of errors
    Teuchos::RCP<Core::LinAlg::Vector> initerror_;  ///< initial value of bc
    Teuchos::RCP<Core::LinAlg::Vector> acterror_;   ///< current value of bc
    Teuchos::RCP<Core::LinAlg::Vector>
        lagrvalues_;  ///< value of Lagrange multiplier in augmented Lagrange
    Teuchos::RCP<Core::LinAlg::Vector>
        lagrvalues_force_;  ///< value of Lagrange multiplier in augmented Lagrange


   private:
    // don't want = operator, cctor and destructor

    ConstraintPenalty operator=(const ConstraintPenalty& old);
    ConstraintPenalty(const ConstraintPenalty& old);


    //! Evaluate constraint conditions and assemble the results
    void evaluate_constraint(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        Teuchos::RCP<Core::LinAlg::Vector>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< aasembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< aasembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector> systemvector3  ///< distributed vector that may be filled
                                                          ///< by aasembly of element contributions
    );

    //! Compute and assemble initial constraint values (depending on user specific activation times)
    void evaluate_error(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::Vector> systemvector  ///< distributed vector that may be filled
                                                         ///< by aasembly of element contributions
    );
  };  // class
}  // namespace CONSTRAINTS

FOUR_C_NAMESPACE_CLOSE

#endif
