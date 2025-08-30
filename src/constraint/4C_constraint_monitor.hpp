// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_MONITOR_HPP
#define FOUR_C_CONSTRAINT_MONITOR_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseOperator;
}

namespace Constraints
{
  /*!
   */
  class Monitor

  {
   public:
    //! Monitor types
    enum MoniType
    {
      none,
      volmonitor3d,
      areamonitor3d,
      areamonitor2d
    };

    /*!
    \brief Constructor of a monitor based on a conditions with a given name. It also
    takes care of the monitor IDs.
    */

    Monitor(std::shared_ptr<Core::FE::Discretization> discr,  ///< discretization monitor lives on
        const std::string& conditionname,  ///< Name of condition to creat monitor from
        int& minID,                        ///< minimum monitor ID so far
        int& maxID                         ///< maximum monitor ID so far
    );


    /*!
     \brief Return if there are monitors
    */
    bool have_monitor() { return montype_ != none; };

    /// Set state of the underlying discretization
    void set_state(const std::string& state,  ///< name of state to set
        Core::LinAlg::Vector<double>& V       ///< values to set
    );

    //! Evaluate routine to call from outside. In here the right action is determined and the
    //! #evaluate_monitor routine is called
    void evaluate(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Core::LinAlg::Vector<double>& systemvector1  ///< distributed vector that may be filled
                                                     ///< by assembly of element contributions
    );


    /// Return type of monitor
    MoniType type() { return montype_; }


   protected:
    std::shared_ptr<Core::FE::Discretization> actdisc_;  ///< standard discretization
    std::vector<const Core::Conditions::Condition*>
        moncond_;       ///< conditions, that define the monitor (all of the same kind)
    MoniType montype_;  ///< monitor type
    std::map<int, double>
        inittimes_;  ///< map with times at which monitor is supposed to become active
    std::map<int, bool> activemons_;  ///< map with indicator if monitors are active

   private:
    // don't want = operator, cctor and destructor

    Monitor operator=(const Monitor& old);
    Monitor(const Monitor& old);

    //! Return the MoniType based on the condition name
    MoniType get_moni_type(const std::string& Name  ///< condition name
    );


    //! Evaluate monitor values and assemble the results
    void evaluate_monitor(
        Teuchos::ParameterList&
            params,  ///< parameter list to communicate between elements and discretization
        Core::LinAlg::Vector<double>& systemvector  ///< distributed vector that may be filled
                                                    ///< by aasembly of element contributions
    );
  };  // class
}  // namespace Constraints

FOUR_C_NAMESPACE_CLOSE

#endif
