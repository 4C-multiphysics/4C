/*----------------------------------------------------------------------*/
/*!
\file fluid_timint_topopt_bdf2.cpp
\brief TimIntTopOptBDF2

<pre>
Maintainers: Benjamin Krank & Martin Kronbichler
             {krank,kronbichler}@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289-15252/-235
</pre>
*/
/*----------------------------------------------------------------------*/

#include "fluid_timint_topopt_bdf2.H"
#include "../drt_io/io.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntTopOptBDF2::TimIntTopOptBDF2(
        const Teuchos::RCP<DRT::Discretization>&      actdis,
        const Teuchos::RCP<LINALG::Solver>&           solver,
        const Teuchos::RCP<Teuchos::ParameterList>&   params,
        const Teuchos::RCP<IO::DiscretizationWriter>& output,
        bool                                          alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis,solver,params,output,alefluid),
      TimIntBDF2(actdis,solver,params,output,alefluid),
      TimIntTopOpt(actdis,solver,params,output,alefluid)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntTopOptBDF2::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntBDF2::Init();
  TimIntTopOpt::Init();

  // write output
  Output();

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                    bk 11/13 |
*----------------------------------------------------------------------*/
FLD::TimIntTopOptBDF2::~TimIntTopOptBDF2()
{
  return;
}

