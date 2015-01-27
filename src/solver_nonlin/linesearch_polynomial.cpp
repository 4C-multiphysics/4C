/*----------------------------------------------------------------------------*/
/*!
\file linesearch_polynomial.cpp

<pre>
Maintainer: Matthias Mayr
            mayr@mhpc.mw.tum.de
            089 - 289-10362
</pre>
*/

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* headers */

// standard
#include <iostream>

// Epetra
#include <Epetra_Comm.h>
#include <Epetra_MultiVector.h>

// Teuchos
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

// baci
#include "linesearch_polynomial.H"

#include "../drt_io/io_pstream.H"
#include "../drt_lib/drt_dserror.H"

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
NLNSOL::LineSearchPolynomial::LineSearchPolynomial()
 : NLNSOL::LineSearchBase(),
   itermax_(0)
{
  return;
}

/*----------------------------------------------------------------------------*/
void NLNSOL::LineSearchPolynomial::Setup()
{
  // make sure that Init() has been called
  if (not IsInit()) { dserror("Init() has not been called, yet."); }

  // fill member variables
  itermax_ = Params().sublist("Polynomial2").get<int>("max number of recursive polynomials");

  // SetupLineSearch() has been called
  SetIsSetup();

  return;
}

/*----------------------------------------------------------------------------*/
const double NLNSOL::LineSearchPolynomial::ComputeLSParam() const
{
  // time measurements
  Teuchos::RCP<Teuchos::Time> time = Teuchos::TimeMonitor::getNewCounter(
      "NLNSOL::LineSearchPolynomial::ComputeLSParam");
  Teuchos::TimeMonitor monitor(*time);

  int err = 0;

  // make sure that Init() and Setup() has been called
  if (not IsInit()) { dserror("Init() has not been called, yet."); }
  if (not IsSetup()) { dserror("Setup() has not been called, yet."); }

  // the line search parameter
  double lsparam = 1.0;
  double lsparamold = 1.0;

  // try a full step first
  Teuchos::RCP<Epetra_MultiVector> xnew =
      Teuchos::rcp(new Epetra_MultiVector(GetXOld().Map(), true));
  err = xnew->Update(1.0, GetXOld(), lsparam, GetXInc(), 0.0);
  if (err != 0) { dserror("Failed."); }

  Teuchos::RCP<Epetra_MultiVector> residual =
      Teuchos::rcp(new Epetra_MultiVector(GetXOld().Map(), true));
  ComputeF(*xnew, *residual);

  double fnorm2fullstep = 1.0e+12;
  bool converged = ConvergenceCheck(*residual, fnorm2fullstep);

  if (IsSufficientDecrease(fnorm2fullstep, lsparam))
    return lsparam;
  else
  {
    // try half step
    lsparamold = lsparam;
    lsparam = 0.5;

    err = xnew->Update(1.0, GetXOld(), lsparam, GetXInc(), 0.0);
    if (err != 0) { dserror("Failed."); }

    ComputeF(*xnew, *residual);

    double fnorm2halfstep = 1.0e+12;
    converged = ConvergenceCheck(*residual, fnorm2halfstep);

    if (converged or IsSufficientDecrease(fnorm2halfstep, lsparam))
      return lsparam;
    else
    {
      // build polynomial model
      int iter = 0;

      // define three data points for a quadratic model
      double l1 = 0.0;
      double l2 = 1.0;
      double l3 = 0.5;

      double y1 = GetFNormOld(); // value at l1
      double y2 = fnorm2fullstep; // value at l2
      double y3 = fnorm2halfstep; // value at l3

//      std::cout << "x_i\ty_i" << std::endl
//                << l1 << "\t" << y1 << std::endl
//                << l2 << "\t" << y2 << std::endl
//                << l3 << "\t" << y3 << std::endl;

      double fnorm2 = y3;
      double a = 0.0;
      double b = 0.0;

      while (not converged and not IsSufficientDecrease(fnorm2, lsparam)
          and iter < itermax_)
      {
        ++iter;

        // compute coefficients of 2nd order polynomial
        a = y3 - (y2-y1)/(l2-l1)*l3 - y1 + l1*(y2-y1)/(l2-l1);
        a = a / (l3*l3 - l3*(l2*l2-l1*l1)/(l2-l1) - l1*l1 +
            l1*(l2*l2-l1*l1)/(l2-l1));

        b = (y2-y1)/(l2-l1) - a*(l2*l2-l1*l1)/(l2-l1);

//        const double c = y1 - a*l1*l1 - b*l1;
//
//        std::cout << "a = " << a
//                  << "\tb = " << b
//                  << "\tc = " << c
//                  << std::endl;

        lsparamold = lsparam;
        lsparam = - b / (2*a);

        if (a < 0.0) // cf. [Kelley1995a, p. 143]
          lsparam = 0.5*lsparamold;

        if (lsparam <= 0.0) // line search parameter has to be strictly positive
          lsparam = 0.5*lsparamold;

//        std::cout << "lsparam = " << lsparam << std::endl;

        // safeguard strategy
        Safeguard(lsparam, lsparamold);

//        std::cout << "lsparam = " << lsparam << std::endl;

        err = xnew->Update(1.0, GetXOld(), lsparam, GetXInc(), 0.0);
        if (err != 0) { dserror("Failed."); }
        ComputeF(*xnew, *residual);

        converged = ConvergenceCheck(*residual, fnorm2);

        // update interpolation points
        l1 = l2;
        l2 = l3;
        l3 = lsparam;
        y1 = y2;
        y2 = y3;
        y3 = fnorm2;
      }

//      if (not converged
//          and (not IsSufficientDecrease(fnorm2, lsparam) or iter > itermax_))
//        dserror("Polynomial line search cannot satisfy sufficient decrease "
//            "condition within %d iterations.", itermax_);

    }
  }

  return lsparam;
}
