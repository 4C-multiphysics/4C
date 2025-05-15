// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT2_HPP
#define FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT2_HPP

#include "4C_config.hpp"

#include "4C_constraint_multipointconstraint.hpp"

FOUR_C_NAMESPACE_OPEN



namespace Constraints
{
  /*!
  \brief This class can handle twodimensional multi point constraints. It is derived from the basic
  constraint class and reimplements the evaluate routine.
  */
  class MPConstraint2 : public MPConstraint
  {
   public:
    /*!
    \brief Standard Constructor
    */
    MPConstraint2(
        std::shared_ptr<Core::FE::Discretization> discr,  ///< discretization constraint lives on
        const std::string& conditionname,  ///< Name of condition to creat constraint from
        int& minID,                        ///< minimum constraint or monitor ID so far
        int& maxID                         ///< maximum constraint or monitor ID so far
    );

    /// initialization routine called by the manager ctor to get correct reference base values and
    /// activating the right conditions at the beginning
    void initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector  ///< distributed vector that may be filled
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
        std::shared_ptr<Core::LinAlg::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        std::shared_ptr<Core::LinAlg::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector3  ///< distributed vector that may be filled
                           ///< by assembly of element contributions
        ) override;

   private:
    // don't want = operator, cctor
    MPConstraint2 operator=(const MPConstraint2& old);
    MPConstraint2(const MPConstraint2& old);

    //! Private Member Functions

    //! Return the ConstrType based on the condition name
    ConstrType get_constr_type(const std::string& Name);  ///< condition name

    //! Evaluate constraint discretization and assemble the results
    void evaluate_constraint(
        std::shared_ptr<Core::FE::Discretization> disc,  ///< discretization to evaluate
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        std::shared_ptr<Core::LinAlg::SparseOperator>
            systemmatrix1,  ///< sparse matrix that may be filled by assembly of element
                            ///< contributions
        std::shared_ptr<Core::LinAlg::SparseOperator>
            systemmatrix2,  ///< sparse (rectangular) matrix that may be filled by assembly of
                            ///< element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector1,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector2,  ///< distributed vector that may be filled by
                            ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
        override;  ///< distributed vector that may be filled by
                   ///< assembly of element contributions


    //! creating a new discretization based on conditions containing constraint elements
    std::map<int, std::shared_ptr<Core::FE::Discretization>> create_discretization_from_condition(
        std::shared_ptr<Core::FE::Discretization> actdisc,
        std::span<const Core::Conditions::Condition*>
            constrcond,                   ///< conditions as discretization basis
        const std::string& discret_name,  ///< name of new discretization
        const std::string& element_name,  ///< name of element type to create
        int& startID) override;

    //! Reorder MPC nodes based on condition input
    void reorder_constraint_nodes(std::vector<int>& nodeids,  ///< reordered node ids
        const Core::Conditions::Condition* condname);         ///< condition to deal with


  };  // class
}  // namespace Constraints
FOUR_C_NAMESPACE_CLOSE

#endif
