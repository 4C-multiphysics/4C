/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_convcheck_strategies.cpp

\brief strategies for Newton-Raphson convergence check for scalar transport problems

To keep the scalar transport time integrator class and derived classes as plain as possible,
the convergence check for the Newton-Raphson iteration has been encapsulated within separate
strategy classes. Every specific convergence check strategy (e.g., for standard scalar transport
or electrochemistry problems with or without scatra-scatra interface coupling involving Lagrange
multipliers) computes, checks, and outputs different relevant vector norms and is implemented
in a subclass derived from an abstract, purely virtual interface class.

<pre>
Maintainer: Rui Fang
            fang@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/
            089-289-15251
</pre>
 */
/*----------------------------------------------------------------------*/
#include "scatra_timint_convcheck_strategies.H"
#include "scatra_timint_implicit.H"

#include "../drt_lib/drt_discret.H"

#include "../linalg/linalg_mapextractor.H"

#include <Epetra_Vector.h>

/*----------------------------------------------------------------------*
 | constructor                                               fang 02/16 |
 *----------------------------------------------------------------------*/
SCATRA::ConvCheckStrategyBase::ConvCheckStrategyBase(
    const Teuchos::ParameterList&   parameters   //!< parameter list for Newton-Raphson iteration
    ) :
itmax_(parameters.get<int>("ITEMAX")),
ittol_(parameters.get<double>("CONVTOL")),
abstolres_(parameters.get<double>("ABSTOLRES"))
{
  return;
}


/*----------------------------------------------------------------------*
 | perform convergence check for Newton-Raphson iteration    fang 02/16 |
 *----------------------------------------------------------------------*/
bool SCATRA::ConvCheckStrategyStd::AbortNonlinIter(
    const ScaTraTimIntImpl&   scatratimint,   //!< scalar transport time integrator
    double&                   actresidual     //!< return maximum current residual value
    ) const
{
  // extract processor ID
  const int mypid = scatratimint.Discretization()->Comm().MyPID();

  // extract current Newton-Raphson iteration step
  const int itnum = scatratimint.IterNum();

  // compute L2 norm of concentration state vector
  double conc_state_L2(0.0);
  scatratimint.Phinp()->Norm2(&conc_state_L2);

  // compute L2 norm of concentration residual vector
  double conc_res_L2(0.);
  scatratimint.Residual()->Norm2(&conc_res_L2);

  // compute infinity norm of concentration residual vector
  double conc_res_inf(0.);
  scatratimint.Residual()->NormInf(&conc_res_inf);

  // compute L2 norm of concentration increment vector
  double conc_inc_L2(0.);
  scatratimint.Increment()->Norm2(&conc_inc_L2);

  // safety checks
  if(std::isnan(conc_state_L2) or std::isnan(conc_res_L2) or std::isnan(conc_inc_L2))
    dserror("Calculated vector norm for concentration is not a number!");
  if(std::isinf(conc_state_L2) or std::isinf(conc_res_L2) or std::isinf(conc_inc_L2))
    dserror("Calculated vector norm for concentration is infinity!");

  // care for the case that nothing really happens in the concentration field
  if(conc_state_L2 < 1.e-5)
    conc_state_L2 = 1.;

  // special case: very first iteration step --> solution increment is not yet available
  if(itnum == 1)
  {
    if(mypid == 0)
    {
      // print header of convergence table to screen
      std::cout << "+------------+-------------------+--------------+--------------+------------------+" << std::endl;
      std::cout << "|- step/max -|- tol      [norm] -|-- con-res ---|-- con-inc ---|-- con-res-inf ---|" << std::endl;

      // print first line of convergence table to screen
      std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itmax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << ittol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_L2 << "   |      --      | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_inf << "       | (      --     ,te="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtEle() << ")" << std::endl;
    }
  }

  // ordinary case: later iteration steps --> solution increment can be printed and convergence check should be done
  else
  {
    if(mypid == 0)
      // print current line of convergence table to screen
      std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itmax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << ittol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_inc_L2/conc_state_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_inf << "       | (ts="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtSolve() << ",te="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtEle() << ")" << std::endl;

    // convergence check
    if(conc_res_L2 <= ittol_ and conc_inc_L2/conc_state_L2 <= ittol_)
    {
      if(mypid == 0)
        // print finish line of convergence table to screen
        std::cout << "+------------+-------------------+--------------+--------------+------------------+" << std::endl << std::endl;

      return true;
    }
  }

  // abort iteration when there is nothing more to do --> better robustness
  // absolute tolerance determines whether residual is already zero
  // prevents additional solver calls that will not improve the solution anymore
  if(conc_res_L2 < abstolres_)
  {
    if(mypid == 0)
      // print finish line of convergence table to screen
      std::cout << "+------------+-------------------+--------------+--------------+------------------+" << std::endl << std::endl;

    return true;
  }

  // output warning in case maximum number of iteration steps is reached without convergence, and proceed to next time step
  if(itnum == itmax_)
  {
    if (mypid == 0)
    {
      std::cout << "+---------------------------------------------------------------+" << std::endl;
      std::cout << "|       >>>>>> Newton-Raphson iteration did not converge!       |" << std::endl;
      std::cout << "+---------------------------------------------------------------+" << std::endl << std::endl;
    }

    return true;
  }

  // return maximum residual value for adaptivity of linear solver tolerance
  actresidual = std::max(conc_res_L2,conc_inc_L2/conc_state_L2);

  // proceed with next iteration step
  return false;
} // SCATRA::ConvCheckStrategyStd::AbortNonlinIter()


/*----------------------------------------------------------------------*
 | perform convergence check for Newton-Raphson iteration    fang 02/16 |
 *----------------------------------------------------------------------*/
bool SCATRA::ConvCheckStrategyStdElch::AbortNonlinIter(
    const ScaTraTimIntImpl&   scatratimint,   //!< scalar transport time integrator
    double&                   actresidual     //!< return maximum current residual value
    ) const
{
  // extract processor ID
  const int mypid = scatratimint.Discretization()->Comm().MyPID();

  // extract current Newton-Raphson iteration step
  const int itnum = scatratimint.IterNum();

  // compute L2 norm of concentration state vector
  Teuchos::RCP<Epetra_Vector> conc_vector = scatratimint.Splitter()->ExtractOtherVector(scatratimint.Phinp());
  double conc_state_L2(0.0);
  conc_vector->Norm2(&conc_state_L2);

  // compute L2 norm of concentration residual vector
  scatratimint.Splitter()->ExtractOtherVector(scatratimint.Residual(),conc_vector);
  double conc_res_L2(0.);
  conc_vector->Norm2(&conc_res_L2);

  // compute infinity norm of concentration residual vector
  double conc_res_inf(0.);
  conc_vector->NormInf(&conc_res_inf);

  // compute L2 norm of concentration increment vector
  scatratimint.Splitter()->ExtractOtherVector(scatratimint.Increment(),conc_vector);
  double conc_inc_L2(0.);
  conc_vector->Norm2(&conc_inc_L2);

  // compute L2 norm of electric potential state vector
  Teuchos::RCP<Epetra_Vector> pot_vector = scatratimint.Splitter()->ExtractCondVector(scatratimint.Phinp());
  double pot_state_L2(0.0);
  pot_vector->Norm2(&pot_state_L2);

  // compute L2 norm of electric potential residual vector
  scatratimint.Splitter()->ExtractCondVector(scatratimint.Residual(),pot_vector);
  double pot_res_L2(0.);
  pot_vector->Norm2(&pot_res_L2);

  // compute L2 norm of electric potential increment vector
  scatratimint.Splitter()->ExtractCondVector(scatratimint.Increment(),pot_vector);
  double pot_inc_L2(0.);
  pot_vector->Norm2(&pot_inc_L2);

  // safety checks
  if(std::isnan(conc_state_L2) or std::isnan(conc_res_L2) or std::isnan(conc_inc_L2))
    dserror("Calculated vector norm for concentration is not a number!");
  if(std::isinf(conc_state_L2) or std::isinf(conc_res_L2) or std::isinf(conc_inc_L2))
    dserror("Calculated vector norm for concentration is infinity!");
  if(std::isnan(pot_state_L2) or std::isnan(pot_res_L2) or std::isnan(pot_inc_L2))
    dserror("Calculated vector norm for electric potential is not a number!");
  if(std::isinf(pot_state_L2) or std::isinf(pot_res_L2) or std::isinf(pot_inc_L2))
    dserror("Calculated vector norm for electric potential is infinity!");

  // care for the case that nothing really happens in the concentration or electric potential fields
  if(conc_state_L2 < 1.e-5)
    conc_state_L2 = 1.;
  if(pot_state_L2 < 1.e-5)
    pot_state_L2 = 1.;

  // special case: very first iteration step --> solution increment is not yet available
  if(itnum == 1)
  {
    if(mypid == 0)
    {
      // print header of convergence table to screen
      std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+------------------+" << std::endl;
      std::cout << "|- step/max -|- tol      [norm] -|-- con-res ---|-- pot-res ---|-- con-inc ---|-- pot-inc ---|-- con-res-inf ---|" << std::endl;

      // print first line of convergence table to screen
      std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itmax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << ittol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << pot_res_L2 << "   |      --      |      --      | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_inf << "       | (      --     ,te="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtEle() << ")" << std::endl;
    }
  }

  // ordinary case: later iteration steps --> solution increment can be printed and convergence check should be done
  else
  {
    if(mypid == 0)
      // print current line of convergence table to screen
      std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itmax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << ittol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << pot_res_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_inc_L2/conc_state_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << pot_inc_L2/pot_state_L2 << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << conc_res_inf << "       | (ts="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtSolve() << ",te="
                << std::setw(10) << std::setprecision(3) << std::scientific << scatratimint.DtEle() << ")" << std::endl;

    // convergence check
    if(conc_res_L2 <= ittol_ and conc_inc_L2/conc_state_L2 <= ittol_ and pot_res_L2 <= ittol_ and pot_inc_L2/pot_state_L2 <= ittol_)
    {
      if(mypid == 0)
        // print finish line of convergence table to screen
        std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+------------------+" << std::endl << std::endl;

      return true;
    }
  }

  // abort iteration when there is nothing more to do --> better robustness
  // absolute tolerance determines whether residual is already zero
  // prevents additional solver calls that will not improve the solution anymore
  if(conc_res_L2 < abstolres_ and pot_res_L2 < abstolres_)
  {
    if(mypid == 0)
      // print finish line of convergence table to screen
      std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+------------------+" << std::endl << std::endl;

    return true;
  }

  // output warning in case maximum number of iteration steps is reached without convergence, and proceed to next time step
  if(itnum == itmax_)
  {
    if (mypid == 0)
    {
      std::cout << "+---------------------------------------------------------------+" << std::endl;
      std::cout << "|       >>>>>> Newton-Raphson iteration did not converge!       |" << std::endl;
      std::cout << "+---------------------------------------------------------------+" << std::endl << std::endl;
    }

    return true;
  }

  // return maximum residual value for adaptivity of linear solver tolerance
  actresidual = std::max(conc_res_L2,conc_inc_L2/conc_state_L2);
  actresidual = std::max(actresidual,pot_res_L2);
  actresidual = std::max(actresidual,pot_inc_L2/pot_state_L2);

  // proceed with next iteration step
  return false;
} // SCATRA::ConvCheckStrategyStdElch::AbortNonlinIter()


/*----------------------------------------------------------------------*
 | perform convergence check for Newton-Raphson iteration    fang 02/16 |
 *----------------------------------------------------------------------*/
bool SCATRA::ConvCheckStrategyS2ILM::AbortNonlinIter(
    const ScaTraTimIntImpl&   scatratimint,   //!< scalar transport time integrator
    double&                   actresidual     //!< return maximum current residual value
    ) const
{
  dserror("Not yet implemented!");

  return false;
} // SCATRA::ConvCheckStrategyS2ILM::AbortNonlinIter()


/*----------------------------------------------------------------------*
 | perform convergence check for Newton-Raphson iteration    fang 02/16 |
 *----------------------------------------------------------------------*/
bool SCATRA::ConvCheckStrategyS2ILMElch::AbortNonlinIter(
    const ScaTraTimIntImpl&   scatratimint,   //!< scalar transport time integrator
    double&                   actresidual     //!< return maximum current residual value
    ) const
{
  dserror("Not yet implemented!");

  return false;
} // SCATRA::ConvCheckStrategyS2ILMElch::AbortNonlinIter()
