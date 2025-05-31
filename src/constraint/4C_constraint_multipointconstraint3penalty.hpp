// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT3PENALTY_HPP
#define FOUR_C_CONSTRAINT_MULTIPOINTCONSTRAINT3PENALTY_HPP

#include "4C_config.hpp"

#include "4C_constraint_multipointconstraint.hpp"

FOUR_C_NAMESPACE_OPEN



namespace Constraints
{
  /*!
  \brief This class can handle multi point constraints in 3D.
  It is derived from the basic multipointconstraint class.
  */
  class MPConstraint3Penalty : public Constraints::MPConstraint
  {
   public:
    /*!
    \brief Standard Constructor
    */
    MPConstraint3Penalty(
        std::shared_ptr<Core::FE::Discretization> discr,  ///< discretization constraint lives on
        const std::string& CondName  ///< Name of condition to create constraint from
    );

    /// unused
    void initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>
            systemvector3  ///< distributed vector that may be filled
                           ///< by assembly of element contributions
        ) override;

    /// initialization routine called by the manager ctor
    void initialize(Teuchos::ParameterList&
            params  ///< parameter list to communicate between elements and discretization
    );

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
    MPConstraint3Penalty operator=(const MPConstraint3Penalty& old);
    MPConstraint3Penalty(const MPConstraint3Penalty& old);

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

    //! Initialize constraint discretization and assemble the results to the refbasevector
    void evaluate_error(Core::FE::Discretization& disc,  ///< discretization to evaluate
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Core::LinAlg::Vector<double>& systemvector3,  ///< distributed vector that may be filled by
                                                      ///< aasembly of element contributions
        bool init = false);

    //! creating a new discretization based on conditions containing constraint elements
    std::map<int, std::shared_ptr<Core::FE::Discretization>> create_discretization_from_condition(
        std::shared_ptr<Core::FE::Discretization> actdisc,
        std::span<const Core::Conditions::Condition*>
            constrcond,                   ///< conditions as discretization basis
        const std::string& discret_name,  ///< name of new discretization
        const std::string& element_name,  ///< name of element type to create
        int& startID) override;

    // projected attributes
    std::map<int, bool> absconstraint_;  ///< maps condition ID to indicator if absolute values are
                                         ///< to use for controlling
    std::map<int, int>
        eletocond_id_;  ///< maps element ID to condition ID, to allow use of other maps
    std::map<int, int>
        eletocondvecindex_;  ///< maps element ID to condition index in vector #constrcond_
    std::map<int, double> penalties_;  ///< maps condition ID to penalty factor
    std::shared_ptr<Epetra_Export> errorexport_;
    std::shared_ptr<Core::LinAlg::Import> errorimport_;
    std::shared_ptr<Core::LinAlg::Map> rederrormap_;
    std::shared_ptr<Core::LinAlg::Map> errormap_;
    std::shared_ptr<Core::LinAlg::Vector<double>> initerror_;
    std::shared_ptr<Core::LinAlg::Vector<double>> acterror_;


  };  // class
}  // namespace Constraints
FOUR_C_NAMESPACE_CLOSE

#endif
