/*----------------------------------------------------------------------*/
/*!
\file strtimada.cpp

\brief Time step adaptivity front-end for structural dynamics

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*/
/* definitions */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include <iostream>

#include "../drt_io/io_ostream0.H"
#include "../drt_inpar/inpar_structure.H"

#include "strtimada.H"

/*----------------------------------------------------------------------*/
/* Constructor */
STR::TimAda::TimAda
(
  const Teuchos::ParameterList& sdyn,  //!< TIS input parameters
  const Teuchos::ParameterList& tap,  //!< adaptive input flags
  Teuchos::RCP<TimInt> tis  //!< marching time integrator
)
: sti_(tis),
  discret_(tis->Discretization()),
  myrank_(discret_->Comm().MyPID()),
  solver_(tis->Solver()),
  output_(tis->DiscWriter()),
  cout0_(discret_->Comm(), std::cout),
  //
  timeinitial_(0.0),
  timefinal_(sdyn.get<double>("MAXTIME")),
  timedirect_(Sign(timefinal_-timeinitial_)),
  timestepinitial_(0),
  timestepfinal_(sdyn.get<int>("NUMSTEP")),
  stepsizeinitial_(sdyn.get<double>("TIMESTEP")),
  //
  stepsizemax_(tap.get<double>("STEPSIZEMAX")),
  stepsizemin_(tap.get<double>("STEPSIZEMIN")),
  sizeratiomax_(tap.get<double>("SIZERATIOMAX")),
  sizeratiomin_(tap.get<double>("SIZERATIOMIN")),
  sizeratioscale_(tap.get<double>("SIZERATIOSCALE")),
  errctrl_(ctrl_dis),  // PROVIDE INPUT PARAMETER
  errnorm_(Teuchos::getIntegralValue<INPAR::STR::VectorNorm>(tap,"LOCERRNORM")),
  errtol_(tap.get<double>("LOCERRTOL")),
  errorder_(1),  // CHANGE THIS CONSTANT
  adaptstepmax_(tap.get<int>("ADAPTSTEPMAX")),
  //
  time_(timeinitial_),
  timestep_(0),
  stepsizepre_(0),
  stepsize_(sdyn.get<double>("TIMESTEP")),
  locerrdisn_(Teuchos::null),
  adaptstep_(0),
  //
  outsys_(false),
  outstr_(false),
  outene_(false),
  outrest_(false),
  outsysperiod_(tap.get<double>("OUTSYSPERIOD")),
  outstrperiod_(tap.get<double>("OUTSTRPERIOD")),
  outeneperiod_(tap.get<double>("OUTENEPERIOD")),
  outrestperiod_(tap.get<double>("OUTRESTPERIOD")),
  outsizeevery_(tap.get<int>("OUTSIZEEVERY")),
  outsystime_(timeinitial_+outsysperiod_),
  outstrtime_(timeinitial_+outstrperiod_),
  outenetime_(timeinitial_+outeneperiod_),
  outresttime_(timeinitial_+outrestperiod_),
  outsizefile_(NULL)
{
  // allocate displacement local error vector
  locerrdisn_ = LINALG::CreateVector(*(discret_->DofRowMap()), true);

  // check wether energyout_ file handle was attached
  if ( (not sti_->AttachedEnergyFile()) 
       and (outeneperiod_ != 0.0) 
       and (myrank_ == 0) )
  {
    sti_->AttachEnergyFile();
  }

  // check if step size file is wanted and attach
  if ( (outsizeevery_ != 0) and (myrank_ == 0) )
  {
    AttachFileStepSize();
  }

  // hallelujah
  return;
}

/*----------------------------------------------------------------------*/
/* Integrate adaptively in time */
void STR::TimAda::Integrate()
{
  // Richardson extrapolation to no avail
  if (MethodAdaptDis() == ada_ident)
    dserror("This combination is not implemented ... Richardson's extrapolation ... Yoshida technique ...");

  // initialise time loop
  time_ = timeinitial_;
  timestep_ = timestepinitial_;
  stepsize_ = stepsizeinitial_;
  stepsizepre_ = stepsize_;

  // time loop
  while ( (time_ < timefinal_) and (timestep_ < timestepfinal_) )
  {
    // time step size adapting loop
    adaptstep_ = 0;
    //double err = 2.0*errtol_;
    bool accepted = false;
    double stpsiznew;
    while ( (not accepted) and (adaptstep_ < adaptstepmax_) )
    {

      // modify step-size #stepsize_ according to output period
      // and store output type on #outstep_
      SizeForOutput();

      // set current stepsize
      sti_->dt_->SetStep(0, stepsize_);
      //*(sti_->dt_(0)) = stepsize_;

      // integrate system with auxiliar TIS
      // we hold \f$D_{n+1}^{AUX}\f$ on #locdiserrn_
      // and \f$V_{n+1}^{AUX}\f$ on #locvelerrn_
      IntegrateStepAuxiliar();

      // integrate system with marching TIS and 
      sti_->IntegrateStep();

      // get local error vector on #locerrdisn_
      EvaluateLocalErrorDis();

      // check wether step passes
      Indicate(accepted, stpsiznew);

      // adjust step-size
      if (not accepted)
      {
        cout0_ << "Repeating step with stepsize = " << stpsiznew
               << std::endl;
        cout0_ << "- - - - - - - - - - - - - - - - - - - - - - - - -"
               << " - - - - - - - - - - - - - - -"
               << std::endl;
        stepsize_ = stpsiznew;
        outrest_ = outsys_ = outstr_ = outene_ = false;
        sti_->ResetStep();
      }

      // increment number of adapted step sizes in a row
      adaptstep_ += 1;
    }

    // update or break
    if (accepted)
    {
      cout0_ << "Step size accepted" << std::endl;
    }
    else if (adaptstep_ >= adaptstepmax_)
    {
      cout0_ << "Could not find acceptable time step size"
             << " ... continuing" << std::endl;
    }
    else 
    {
      dserror("Do not know what to do");
    }
    
    // sti_->time_ = time_ + stepsize_;
    sti_->time_->UpdateSteps(time_ + stepsize_);
    sti_->step_ = timestep_ + 1;
    // sti_->dt_ = stepsize_;
    sti_->dt_->UpdateSteps(stepsize_);

    // printing and output
    sti_->UpdateStepState();
    sti_->PrintStep();
    OutputPeriod();
    OutputStepSize();
    
    // update
    sti_->stepn_ = timestep_ += 1;
    sti_->timen_ = time_ += stepsize_;
    stepsizepre_ = stepsize_;
    stepsize_ = stpsiznew;
    //
    UpdatePeriod();
    outrest_ = outsys_ = outstr_ = outene_ = false;
    
    // the user reads but rarely listens
    if (myrank_ == 0)
    {
      std::cout << "Step " << timestep_ 
                << ", Time " << time_ 
                << ", StepSize " << stepsize_ 
                << std::endl;
    }
  }

  // leave for good
  return;
}

/*----------------------------------------------------------------------*/
/* Evaluate local error vector */
void STR::TimAda::EvaluateLocalErrorDis()
{
  if (MethodAdaptDis() == ada_orderequal)
  {
    const double coeffmarch = sti_->MethodLinErrCoeffDis();
    const double coeffaux = MethodLinErrCoeffDis();
    locerrdisn_->Update(-1.0, *(sti_->disn_), 1.0);
    locerrdisn_->Scale(coeffmarch/(coeffaux-coeffmarch));
  }
  else 
  {
    // schemes do not have the same order of accuracy
    locerrdisn_->Update(-1.0, *(sti_->disn_), 1.0);
  }
}

/*----------------------------------------------------------------------*/
/* Indicate error and determine new step size */
void STR::TimAda::Indicate
(
  bool& accepted,
  double& stpsiznew
)
{
  // norm of local discretisation error vector
  double norm = TimIntVector::CalculateNorm(errnorm_, locerrdisn_); 

  // check if acceptable
  accepted = (norm < errtol_);

  // debug
  if (myrank_ == 0)
  {
    std::cout << "LocErrNorm " << std::scientific << norm 
              << ", LocErrTol " << errtol_ 
              << ", Accept " << std::boolalpha << accepted 
              << std::endl;
  }

  // get error order
  if (MethodAdaptDis() == ada_upward)
    errorder_ = sti_->MethodOrderOfAccuracyDis();
  else
    errorder_ = MethodOrderOfAccuracyDis();

  // optimal size ration with respect to given tolerance
  double sizrat = pow(errtol_/norm, 1.0/(errorder_+1.0));

  // debug
  if (myrank_ == 0)
  {
    printf("sizrat %g, stepsize %g, stepsizepre %g\n",
           sizrat, stepsize_, stepsizepre_);
  }

  // scaled by safety parameter
  sizrat *= sizeratioscale_;
  // optimal new step size
  stpsiznew = sizrat * stepsize_;
  // redefine sizrat to be dt*_{n}/dt_{n-1}, ie true optimal ratio
  sizrat = stpsiznew/stepsizepre_;

  // limit #sizrat by maximum and minimum
  if (sizrat > sizeratiomax_)
  {
    stpsiznew = sizeratiomax_ * stepsizepre_;
  }
  else if (sizrat < sizeratiomin_)
  {
    stpsiznew = sizeratiomin_ * stepsizepre_;
  }

  // new step size subject to safety measurements 
  if (stpsiznew > stepsizemax_)
  {
    stpsiznew = stepsizemax_;
  }
  else if (stpsiznew < stepsizemin_)
  {
    stpsiznew = stepsizemin_;
  }

  // get away from here
  return;
}

/*----------------------------------------------------------------------*/
/*  Modify step size to hit precisely output period */
void STR::TimAda::SizeForOutput()
{
  // check output of restart data first
  if ( (fabs(time_ + stepsize_) >= fabs(outresttime_))
       and (outrestperiod_ != 0.0) )

  {
    stepsize_ = outresttime_ - time_;
    outrest_ = true;
  }

  // check output of system vectors
  if ( (fabs(time_ + stepsize_) >= fabs(outsystime_))
       and (outsysperiod_ != 0.0) )
  {
    stepsize_ = outsystime_ - time_;
    outsys_ = true;
    if (fabs(outsystime_) < fabs(outresttime_)) outrest_ = false;
  }

  // check output of stress/strain
  if ( (fabs(time_ + stepsize_) >= fabs(outstrtime_))
       and (outstrperiod_ != 0.0) )
  {
    stepsize_ = outstrtime_ - time_;
    outstr_ = true;
    if (fabs(outstrtime_) < fabs(outresttime_)) outrest_ = false;
    if (fabs(outstrtime_) < fabs(outsystime_)) outsys_ = false;
  }

  // check output of energy
  if ( (fabs(time_ + stepsize_) >= fabs(outenetime_))
       and (outeneperiod_ != 0.0) )
  {
    stepsize_ = outenetime_ - time_;
    outene_ = true;
    if (fabs(outenetime_) < fabs(outresttime_)) outrest_ = false;
    if (fabs(outenetime_) < fabs(outsystime_)) outsys_ = false;
    if (fabs(outenetime_) < fabs(outstrtime_)) outstr_ = false;
  }

  // give a lift
  return;
}

/*----------------------------------------------------------------------*/
/* Output to file(s) */
void STR::TimAda::OutputPeriod()
{
  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if (outrest_)
  {
    sti_->OutputRestart(datawritten);
  }

  // output results (not necessary if restart in same step)
  if (outsys_ and (not datawritten) )
  {
    sti_->OutputState(datawritten);
  }

  // output stress & strain
  if (outstr_)
  {
    sti_->OutputStressStrain(datawritten);
  }

  // output energy
  if (outene_)
  {
    sti_->OutputEnergy();
  }

  // flag down the cab
  return;
}

/*----------------------------------------------------------------------*/
/* Update output periods */
void STR::TimAda::UpdatePeriod()
{
  if (outrest_) outresttime_ += outrestperiod_;
  if (outsys_) outsystime_ += outsysperiod_;
  if (outstr_) outstrtime_ += outstrperiod_;
  if (outene_) outenetime_ += outeneperiod_;
  // freedom
  return;
}

/*----------------------------------------------------------------------*/
/* Write step size */
void STR::TimAda::OutputStepSize()
{
  if ( (outsizeevery_ != 0)
       and (timestep_%outsizeevery_ == 0)
       and (myrank_ == 0) )
  {
    *outsizefile_ << " " << std::setw(12) << timestep_
                  << std::scientific << std::setprecision(8)
                  << " " << time_
                  << " " << stepsize_
                  << " " << std::setw(2) << adaptstep_
                  << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Print constants */
void STR::TimAda::PrintConstants
(
  std::ostream& str
) const
{
  str << "TimAda:  Constants" << std::endl
      << "   Initial time = " << timeinitial_ << std::endl
      << "   Final time = " << timefinal_ << std::endl
      << "   Initial Step = " << timestepinitial_ << std::endl
      << "   Final Step = " << timestepfinal_ << std::endl
      << "   Initial step size = " << stepsizeinitial_ << std::endl
      << "   Max step size = " << stepsizemax_ << std::endl
      << "   Min step size = " << stepsizemin_ << std::endl
      << "   Max size ratio = " << sizeratiomax_ << std::endl
      << "   Min size ratio = " << sizeratiomin_ << std::endl
      << "   Size ratio scale = " << sizeratioscale_ << std::endl
      << "   Error norm = " << INPAR::STR::VectorNormString(errnorm_) << std::endl
      << "   Error order = " << errorder_ << std::endl
      << "   Error tolerance = " << errtol_ << std::endl
      << "   Max adaptations = " << adaptstepmax_ << std::endl;
  return;
}

/*----------------------------------------------------------------------*/
/* Print variables */
void STR::TimAda::PrintVariables
(
  std::ostream& str
) const
{
  str << "TimAda:  Variables" << std::endl
      << "   Current time = " << time_ << std::endl
      << "   Previous step size = " << stepsizepre_ << std::endl
      << "   Current step size = " << stepsize_ << std::endl
      << "   Current adaptive step = " << adaptstep_ << std::endl;
  return;
}


/*----------------------------------------------------------------------*/
/* Print */
void STR::TimAda::Print
(
  std::ostream& str
) const
{
  str << "TimAda" << std::endl;
  PrintConstants(str);
  PrintVariables(str);
  // step aside
  return;
}


/*======================================================================*/
/* Out stream */
std::ostream& operator<<
(
  std::ostream& str, 
  const STR::TimAda::TimAda& ta
)
{
  ta.Print(str);
  return str;
}

/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
