// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CARDIOVASCULAR0D_HPP
#define FOUR_C_CARDIOVASCULAR0D_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace Utils
{
  class Cardiovascular0D

  {
   public:
    //! 0D cardiovascular types
    enum Cardiovascular0DType
    {
      none,
      cardvasc0d_4elementwindkessel,
      cardvasc0d_arterialproxdist,
      cardvasc0d_syspulcirculation,
      cardvascrespir0d_syspulperiphcirculation
    };

    /*!
    \brief Constructor of a Cardiovascular0D based on conditions with a given name. It also
    takes care of the Cardiovascular0D IDs.
    */

    Cardiovascular0D(std::shared_ptr<Core::FE::Discretization>
                         discr,            ///< discretization where Cardiovascular0D lives on
        const std::string& conditionname,  ///< Name of condition to create Cardiovascular0D from
        std::vector<int>& curID            ///< current ID
    );

    /*!
    \brief Constructor of a Cardiovascular0D based on a conditions with a given name.
    */

    Cardiovascular0D(std::shared_ptr<Core::FE::Discretization>
                         discr,  ///< discretization where Cardiovascular0D function lives on
        const std::string&
            CondName  ///< Name of condition to create Cardiovascular0D functions from
    );

    /*!
        \brief Destructor

     */
    virtual ~Cardiovascular0D() = default;

    /*!
     \brief Return if there are Cardiovascular0D functions
    */
    bool have_cardiovascular0_d() { return cardiovascular0dtype_ != none; };

    /// Set state of the underlying discretization
    void set_state(const std::string& state,  ///< name of state to set
        Core::LinAlg::Vector<double>& V       ///< values to set
    );


    /// initialization routine called by the manager ctor to get correct reference base values and
    /// activating the right conditions at the beginning
    virtual void initialize(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>
            sysvec1,  ///< distributed vector that may be filled by
                      ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>>
            sysvec2  ///< distributed vector that may be filled by assembly of element contributions
    );

    //! Evaluate routine to call from outside. In here the right action is determined and the
    //! #EvaluateCardiovascular0D routine is called
    virtual void evaluate(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat1,  ///< Cardiovascular0D stiffness matrix
        std::shared_ptr<Core::LinAlg::SparseOperator>
            sysmat2,  ///< Cardiovascular0D offdiagonal matrix dV/dd
        std::shared_ptr<Core::LinAlg::SparseOperator>
            sysmat3,  ///< Cardiovascular0D offdiagonal matrix dfext/dp
        std::shared_ptr<Core::LinAlg::Vector<double>>
            sysvec1,  ///< distributed vectors that may be filled by
                      ///< assembly of element contributions
        std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2,
        std::shared_ptr<Core::LinAlg::Vector<double>> sysvec3,
        const std::shared_ptr<Core::LinAlg::Vector<double>> sysvec4,
        std::shared_ptr<Core::LinAlg::Vector<double>> sysvec5);

    /// Return type of Cardiovascular0D function
    Cardiovascular0DType type() { return cardiovascular0dtype_; }

    std::vector<const Core::Conditions::Condition*> get_cardiovascular0_d_condition()
    {
      return cardiovascular0dcond_;
    }

    std::vector<const Core::Conditions::Condition*>
    get_cardiovascular0_d_structure_coupling_condition()
    {
      return cardiovascular0dstructcoupcond_;
    }

    Inpar::Cardiovascular0D::Cardvasc0DRespiratoryModel get_respiratory_model()
    {
      return respiratory_model_;
    }

    //! Evaluate Cardiovascular0D conditions and assemble the results
    void evaluate_d_struct_dp(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Core::LinAlg::SparseOperator& sysmat  ///< Cardiovascular0D offdiagonal matrix dfext/dp
    );

   protected:
    std::shared_ptr<Core::FE::Discretization> actdisc_;  ///< standard discretization
    std::vector<const Core::Conditions::Condition*>
        cardiovascular0dcond_;  ///< 0D cardiovascular conditions
    std::vector<const Core::Conditions::Condition*>
        cardiovascular0dstructcoupcond_;  ///< 0D cardiovascular structure coupling conditions
    Cardiovascular0DType cardiovascular0dtype_;  ///< Cardiovascular0D type
    const Inpar::Cardiovascular0D::Cardvasc0DAtriumModel atrium_model_;
    const Inpar::Cardiovascular0D::Cardvasc0DVentricleModel ventricle_model_;
    const Inpar::Cardiovascular0D::Cardvasc0DRespiratoryModel respiratory_model_;
    //! gaussian integration to be used
    Core::FE::GaussRule2D gaussrule_;

   private:
    // don't want = operator, cctor and destructor

    Cardiovascular0D operator=(const Cardiovascular0D& old);
    Cardiovascular0D(const Cardiovascular0D& old);

    //! Return the Cardiovascular0DType based on the condition name
    Cardiovascular0DType get_cardiovascular0_d_type(const std::string& Name  ///< condition name
    );

  };  // class
}  // namespace Utils

FOUR_C_NAMESPACE_CLOSE

#endif
