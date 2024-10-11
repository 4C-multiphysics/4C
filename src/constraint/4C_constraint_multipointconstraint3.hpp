/*----------------------------------------------------------------------*/
/*! \file
\brief Basic constraint class, dealing with multi point constraints
\level 2


*----------------------------------------------------------------------*/

#ifndef FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT3_HPP
#define FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT3_HPP

#include "4C_config.hpp"

#include "4C_constraint_multipointconstraint.hpp"

FOUR_C_NAMESPACE_OPEN



namespace CONSTRAINTS
{
  /*!
  \brief This pure virtual class can handle multi point constraints in 3D.
  It is derived from the basic multipointconstraint class.
  */
  class MPConstraint3 : public CONSTRAINTS::MPConstraint
  {
   public:
    /*!
    \brief Standard Constructor
    */
    MPConstraint3(
        Teuchos::RCP<Core::FE::Discretization> discr,  ///< discretization constraint lives on
        const std::string& conditionname,  ///< Name of condition to create constraint from
        int& offsetID,                     ///< minimum constraint or monitor ID so far
        int& maxID                         ///< maximum constraint or monitor ID so far
    );

    /// initialization routine called by the manager ctor to get correct reference base values and
    /// activating the right conditions at the beginning
    void initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector3  ///< distributed vector that may be filled
                           ///< by assembly of element contributions
        ) override;

    /// initialization routine called at restart to activate the right conditions
    void initialize(const double& time  ///< current time
        ) override;

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
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector3  ///< distributed vector that may be filled
                           ///< by assembly of element contributions
        ) override;

   private:
    // don't want = operator, cctor
    MPConstraint3 operator=(const MPConstraint3& old);
    MPConstraint3(const MPConstraint3& old);

    //! Return the ConstrType based on the condition name
    ConstrType get_constr_type(const std::string& Name);  ///< condition name

    //! Evaluate constraint discretization and assemble the results
    void evaluate_constraint(
        Teuchos::RCP<Core::FE::Discretization> disc,  ///< discretization to evaluate
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        Teuchos::RCP<Core::LinAlg::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector<double>>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector3)
        override;  ///< distributed vector that may be filled by
                   ///< assembly of element contributions

    //! Initialize constraint discretization and assemble the results to the refbasevector
    void initialize_constraint(Core::FE::Discretization& disc,  ///< discretization to evaluate
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Core::LinAlg::Vector<double>& systemvector3  ///< distributed vector that may be filled
                                                     ///< by aasembly of element contributions
    );

    //! creating a new discretization based on conditions containing constraint elements
    std::map<int, Teuchos::RCP<Core::FE::Discretization>> create_discretization_from_condition(
        Teuchos::RCP<Core::FE::Discretization> actdisc,
        std::vector<Core::Conditions::Condition*>
            constrcond,                   ///< conditions as discretization basis
        const std::string& discret_name,  ///< name of new discretization
        const std::string& element_name,  ///< name of element type to create
        int& startID                      ///< ID to start with
        ) override;

    // projected attributes
    std::map<int, bool> absconstraint_;  ///< maps condition ID to indicator if absolute values are
                                         ///< to use for controlling
    std::map<int, int>
        eletocond_id_;  ///< maps element ID to condition ID, to allow use of other maps
    std::map<int, int>
        eletocondvecindex_;  ///< maps element ID to condition index in vector #constrcond_

  };  // class
}  // namespace CONSTRAINTS
FOUR_C_NAMESPACE_CLOSE

#endif
