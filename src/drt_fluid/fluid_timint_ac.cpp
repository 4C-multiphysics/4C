/*!----------------------------------------------------------------------
\file fluid_timint_ac.cpp
\brief Fluid time integrator for FS3I-AC problems

\date 2015-07-29

\maintainer Moritz Thon
            thon@mhpc.mw.tum.de
            089/289-10364

\level 3
----------------------------------------------------------------------*/

#include "fluid_timint_ac.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_io/io.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                                     Thon 12/14 |
 *----------------------------------------------------------------------*/
FLD::TimIntAC::TimIntAC(
        const Teuchos::RCP<DRT::Discretization>&      actdis,
        const Teuchos::RCP<LINALG::Solver>&           solver,
        const Teuchos::RCP<Teuchos::ParameterList>&   params,
        const Teuchos::RCP<IO::DiscretizationWriter>& output,
        bool                                          alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis,solver,params,output,alefluid)
{
  return;
}

/*----------------------------------------------------------------------*
| Destructor (public)                                        Thon 12/14 |
*-----------------------------------------------------------------------*/
FLD::TimIntAC::~TimIntAC()
{
  return;
}

/*----------------------------------------------------------------------*
 | output of solution vector to binio                        Thon 12/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntAC::ReadRestart(int step)
{
  const Teuchos::ParameterList& fs3idynac = DRT::Problem::Instance()->FS3IDynamicParams().sublist("AC");
  const bool restartfrompartfsi = DRT::INPUT::IntegralValue<int>(fs3idynac,"RESTART_FROM_PART_FSI");

  if (not restartfrompartfsi) //standard restart
  {
    IO::DiscretizationReader reader(discret_,step);

    reader.ReadVector(trueresidual_,"trueresidual");
  }

  return;
}

/*----------------------------------------------------------------------*
 | output of solution vector to binio                        Thon 12/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntAC::Output()
{
  FluidImplicitTimeInt::Output();

  // output of solution
  if (step_%upres_ == 0 or (uprestart_ > 0 and step_%uprestart_ == 0) )
  {
    output_->WriteVector("trueresidual", trueresidual_);
  }
  return;
}
