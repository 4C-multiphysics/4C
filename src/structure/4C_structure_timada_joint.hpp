/*======================================================================*/
/*! \file
\brief Indicator based on general STR::TimInt object

\level 1

*/

/*----------------------------------------------------------------------*/
/* definitions */
#ifndef FOUR_C_STRUCTURE_TIMADA_JOINT_HPP
#define FOUR_C_STRUCTURE_TIMADA_JOINT_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timada.hpp"
#include "4C_structure_timint.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace DRT
{
  class Discretization;
}

namespace CORE::LINALG
{
  class Solver;
}

namespace IO
{
  class DiscretizationWriter;
}

/*----------------------------------------------------------------------*/
/* belongs to structure namespace */
namespace STR
{
  /*====================================================================*/
  /*!
   * \brief Time step size adaptivity with general STR::TimInt scheme
   *
   * \author bborn
   * \date 10/07
   */
  template <typename T>
  class TimAdaJoint : public TimAda
  {
   public:
    //! Map STR::TimInt::NameEnum to STR::TimAda::NameEnum
    enum INPAR::STR::TimAdaKind map_name_tim_int_to_tim_ada(
        const enum INPAR::STR::DynamicType term  //!< input enum term
    ) const
    {
      switch (term)
      {
        case INPAR::STR::dyna_ab2:
          return INPAR::STR::timada_kind_ab2;
          break;
        case INPAR::STR::dyna_expleuler:
          return INPAR::STR::timada_kind_expleuler;
          break;
        case INPAR::STR::dyna_centrdiff:
          return INPAR::STR::timada_kind_centraldiff;
          break;
        default:
          FOUR_C_THROW("Cannot handle requested time integrator");
          return INPAR::STR::timada_kind_none;
          break;
      }
    }

    //! @name Life
    //@{

    //! Full-fledged constructor
    TimAdaJoint(const Teuchos::ParameterList& ioparams,  //!< ioflags
        const Teuchos::ParameterList& timeparams,        //!< TIS input parameters
        const Teuchos::ParameterList& sdyn,              //!< structural dynamic
        const Teuchos::ParameterList& xparams,           //!< extra flags
        const Teuchos::ParameterList& adaparams,         //!< adaptive input flags
        Teuchos::RCP<TimInt>& sti                        //!< marching time integrator
        )
        : TimAda(timeparams, adaparams, sti), ada_(ada_vague), sta_(Teuchos::null)
    {
      // allocate auxiliary integrator
      sta_ = Teuchos::rcp(
          new T(timeparams, ioparams, sdyn, xparams, sti->discretization(), sti->Solver(),
              Teuchos::null,  // no contact solver
              sti->DiscWriter()));
      sta_->Init(timeparams, sdyn, xparams, sti->discretization(), sti->Solver());

      // check explicitness
      if (sta_->MethodImplicit())
      {
        FOUR_C_THROW("Implicit might work, but please check carefully");
      }

      // check order
      if (sta_->method_order_of_accuracy_dis() > sti_->method_order_of_accuracy_dis())
      {
        ada_ = ada_upward;
      }
      else if (sta_->method_order_of_accuracy_dis() < sti_->method_order_of_accuracy_dis())
      {
        ada_ = ada_downward;
      }
      else if (sta_->MethodName() == sti_->MethodName())
      {
        ada_ = ada_ident;
      }
      else
      {
        ada_ = ada_orderequal;
      }

      // setup
      sta_->Setup();

      // Actually, we would like to call Merge() and ResizeMStep() now and in the
      // past we have done this right here. However, this requires that both Init()
      // and Setup() have already been called on both(!) the marching time integrator
      // and the auxiliary time integrator. Since Setup() has not yet been called
      // for the marching time integrator, we must postpone Merge() and ResizeMStep()
      // to a later point. They can now be found in the new Init() function below,
      // which is called at the beginning of Integrate(). [popp 01/2017]
      // sta_->Merge(*sti);
      // sta_->ResizeMStep();

      return;
    }

    //@}

    //! @name Actions
    //@{

    /*! Finalize the class initialization
     * Merge() and ResizeMStep() need to be called after(!) both Init()
     * and Setup() have been called on both the marching time integrator
     * and the auxiliary time integrator (popp 01/2017).
     */
    void Init(Teuchos::RCP<TimInt>& sti) override
    {
      // merge
      sta_->Merge(*sti);

      // resize multi-step quantities
      sta_->ResizeMStep();

      return;
    }

    /*! \brief Make one step with auxiliary scheme
     *
     *  Afterwards, the auxiliary solutions are stored in the local error
     *  vectors, ie:
     *  - \f$D_{n+1}^{AUX}\f$ in #locdiserrn_
     *  - \f$V_{n+1}^{AUX}\f$ in #locvelerrn_
     */
    void integrate_step_auxiliar() override
    {
      // integrate the auxiliary time integrator one step in time
      sta_->IntegrateStep();

      // copy onto target
      locerrdisn_->Update(1.0, *(sta_->disn_), 0.0);

      // reset
      // remember: sta_ and sti_ are merged and work on the same vectors
      sta_->reset_step();

      return;
    }

    //@}

    //! @name Attributes
    //@{

    //! Provide the name
    enum INPAR::STR::TimAdaKind MethodName() const override
    {
      return map_name_tim_int_to_tim_ada(sti_->MethodName());
    }

    //! Provide local order of accuracy of displacements
    int method_order_of_accuracy_dis() const override
    {
      return sta_->method_order_of_accuracy_dis();
    }

    //! Provide local order of accuracy of velocities
    int method_order_of_accuracy_vel() const override
    {
      return sta_->method_order_of_accuracy_vel();
    }

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override { return sta_->method_lin_err_coeff_dis(); }

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override { return sta_->method_lin_err_coeff_vel(); }

    //! Provide type of algorithm
    enum AdaEnum MethodAdaptDis() const override { return ada_; }

    //@}

   protected:
    //! not wanted: = operator
    TimAdaJoint operator=(const TimAdaJoint& old);

    //! not wanted: copy constructor
    TimAdaJoint(const TimAdaJoint& old);

    //! type of adaptivity algorithm
    enum AdaEnum ada_;

    //! The auxiliary integrator
    Teuchos::RCP<T> sta_;
  };

}  // namespace STR



/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif