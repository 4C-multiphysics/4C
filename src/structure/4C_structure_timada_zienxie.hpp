// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMADA_ZIENXIE_HPP
#define FOUR_C_STRUCTURE_TIMADA_ZIENXIE_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timada.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* belongs to structure namespace */
namespace Solid
{
  /*====================================================================*/
  /*!
   * \brief Time step size adaptivity with Zienkiewicz-Xie error indicator
   *        only for lower or equal than second order accurate marching
   *        scheme
   *
   * References
   * - [1] OC Zienkiewicz and YM Xie, A simple error estimator and adaptive
   *   time stepping procedure for dynamic analysis, Earthquake Engrg.
   *   and Structural Dynamics, 20:871-887, 1991.
   *

   */
  class TimAdaZienXie : public TimAda
  {
   public:
    //! Constructor
    TimAdaZienXie(const Teuchos::ParameterList& timeparams,  //!< TIS input parameters
        const Teuchos::ParameterList& adaparams,             //!< adaptive input flags
        std::shared_ptr<TimInt> tis                          //!< marching time integrator
    );

    //! @name Actions
    //@{

    //! Finalize the class initialization (nothing to do here)
    void init(std::shared_ptr<TimInt>& sti) override {}

    /*! \brief Make one step with auxiliary scheme
     *
     *  Afterwards, the auxiliary solutions are stored in the local error
     *  vectors, ie:
     *  - \f$D_{n+1}^{AUX}\f$ in #locdiserrn_
     *  - \f$V_{n+1}^{AUX}\f$ in #locvelerrn_
     */
    void integrate_step_auxiliary() override;

    //@}

    //! @name Attributes
    //@{

    //! Provide the name
    enum Inpar::Solid::TimAdaKind method_name() const override
    {
      return Inpar::Solid::timada_kind_zienxie;
    }

    //! Provide local order of accuracy
    int method_order_of_accuracy_dis() const override { return 3; }

    //! Provide local order of accuracy
    int method_order_of_accuracy_vel() const override { return 2; }

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override { return -1.0 / 24.0; }

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override { return -1.0 / 12.0; }

    //! Provide type of algorithm
    enum AdaEnum method_adapt_dis() const override { return ada_upward; }

    //@}

   protected:
    //! not wanted: = operator
    TimAdaZienXie operator=(const TimAdaZienXie& old);

    //! not wanted: copy constructor
    TimAdaZienXie(const TimAdaZienXie& old);
  };

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
