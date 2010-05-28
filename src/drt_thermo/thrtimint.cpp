/*----------------------------------------------------------------------*/
/*!
\file thrtimint.cpp
\brief Time integration for spatially discretised thermal dynamics

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237

</pre>
*/

/*----------------------------------------------------------------------*
 |  definitions                                             bborn 08/09 |
 *----------------------------------------------------------------------*/
#ifdef CCADISCRET

/*----------------------------------------------------------------------*
 |  headers                                                 bborn 08/09 |
 *----------------------------------------------------------------------*/
#include <iostream>
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Teuchos_TimeMonitor.hpp"

#include "thrtimint_mstep.H"
#include "thrtimint.H"

#include "../drt_io/io_control.H"
#include "../drt_fluid/fluid_utils.H"

/*----------------------------------------------------------------------*
 |  print thermal time logo                                  dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimInt::Logo()
{
  std::cout << "Welcome to Thermal Time Integration " << std::endl;
  std::cout << "      _______________________________" << std::endl;
  std::cout << "  ===(_________|_|_|_|_|_37°C_|_|____)" << std::endl;
  std::cout << std::endl;
}

/*----------------------------------------------------------------------*
 |  constructor                                             bborn 08/09 |
 *----------------------------------------------------------------------*/
THR::TimInt::TimInt(
  const Teuchos::ParameterList& ioparams,
  const Teuchos::ParameterList& tdynparams,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization> actdis,
  Teuchos::RCP<LINALG::Solver> solver,
  Teuchos::RCP<IO::DiscretizationWriter> output
  )
: discret_(actdis),
  myrank_(actdis->Comm().MyPID()),
  dofrowmap_(actdis->Filled() ? actdis->DofRowMap() : NULL),
  solver_(solver),
  solveradapttol_(Teuchos::getIntegralValue<int>(tdynparams,"ADAPTCONV")==1),
  solveradaptolbetter_(tdynparams.get<double>("ADAPTCONV_BETTER")),
  dbcmaps_(Teuchos::rcp(new LINALG::MapExtractor())),
  output_(output),
  printlogo_(true),  // DON'T EVEN DARE TO SET THIS TO FALSE
  printscreen_(true),  // ADD INPUT PARAMETER
  errfile_(xparams.get<FILE*>("err file")),
  printerrfile_(true and errfile_),  // ADD INPUT PARAMETER FOR 'true'
  printiter_(true),  // ADD INPUT PARAMETER
  writerestartevery_(tdynparams.get<int>("RESTARTEVRY")),
  writeglob_((bool) Teuchos::getIntegralValue<int>(ioparams,"THERM_TEMPERATURE")),
  writeglobevery_(tdynparams.get<int>("RESEVRYGLOB")),
  writeelemevery_(tdynparams.get<int>("RESEVRYELEM")),
  writeheatflux_(Teuchos::getIntegralValue<INPAR::THR::HeatFluxType>(ioparams,"THERM_HEATFLUX")),
  writetempgrad_(Teuchos::getIntegralValue<INPAR::THR::TempGradType>(ioparams,"THERM_TEMPGRAD")),
  writeenergyevery_(0),//(tdynparams.get<int>("RESEVRYERGY")),
  energyfile_(NULL),
  time_(Teuchos::null),
  timen_(0.0),
  dt_(Teuchos::null),
  timemax_(tdynparams.get<double>("MAXTIME")),
  stepmax_(tdynparams.get<int>("NUMSTEP")),
  step_(Teuchos::null),
  stepn_(0),
  zeros_(Teuchos::null),
  temp_(Teuchos::null),
  rate_(Teuchos::null),
  tempn_(Teuchos::null),
  raten_(Teuchos::null),
  tang_(Teuchos::null)
//  capa_(Teuchos::null)
{
  // welcome user
  if ( (printlogo_) and (myrank_ == 0) )
  {
    Logo();
  }

  // check wether discretisation has been completed
  if (not discret_->Filled())
  {
    dserror("Discretisation is not complete!");
  }

  // time state
  time_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, 0.0));
  // HERE SHOULD BE SOMETHING LIKE (tdynparams.get<double>("TIMEINIT"))
  dt_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, tdynparams.get<double>("TIMESTEP")));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ( (writeenergyevery_ != 0) and (myrank_ == 0) )
    AttachEnergyFile();

  // a zero vector of full length
  zeros_ = LINALG::CreateVector(*dofrowmap_, true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    discret_->EvaluateDirichlet(p, zeros_, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0); // just in case of change
  }

  // temperatures T_{n}
  temp_ = Teuchos::rcp(new TimIntMStep<Epetra_Vector>(0, 0, dofrowmap_, true));
  // temperature rates R_{n}
  rate_ = Teuchos::rcp(new TimIntMStep<Epetra_Vector>(0, 0, dofrowmap_, true));

  // temperatures T_{n+1} at t_{n+1}
  tempn_ = LINALG::CreateVector(*dofrowmap_, true);
  // temperature rates R_{n+1} at t_{n+1}
  raten_ = LINALG::CreateVector(*dofrowmap_, true);

  // create empty matrices
  tang_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_, 81, true, true));
//  capa_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_, 81, true, true));

  // stay with us
  return;
}

/*----------------------------------------------------------------------*
 |  equilibrate system at initial state                     bborn 08/09 |
 |  and identify consistent temperature rate                            |
 *----------------------------------------------------------------------*/
void THR::TimInt::DetermineCapaConsistTempRate()
{
  // temporary force vectors in this routine
  Teuchos::RCP<Epetra_Vector> fext
    = LINALG::CreateVector(*dofrowmap_, true); //!< external force
  Teuchos::RCP<Epetra_Vector> fint
    = LINALG::CreateVector(*dofrowmap_, true); //!< internal force

  // overwrite initial state vectors with DirichletBCs
  ApplyDirichletBC((*time_)[0], (*temp_)(0), (*rate_)(0), false);

  // get external force
  ApplyForceExternal((*time_)[0], (*temp_)(0), fext);

  // initialise matrices
  tang_->Zero();
//  capa_->Zero();

  // get initial internal force, tangent and capacity
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action", "calc_thermo_fintcapa");
    // type of calling time integrator
    p.set("time integrator", MethodName());
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    // set vector values needed by elements
    discret_->ClearState();
    // SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0, "residual temperature", zeros_);
    discret_->SetState(0, "temperature", (*temp_)(0));
    // set displacements for the coupled TSI problem
    if(disn_!=Teuchos::null) discret_->SetState(1,"displacement",disn_);
    if(veln_!=Teuchos::null) discret_->SetState(1,"velocity",veln_);
    // calculate the capacity matrix onto tang_, instead of buildung 2 matrices
    discret_->Evaluate(p, Teuchos::null, tang_, fint, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  // finish capacity matrix
//  capa_->Complete();

  // close tangent matrix
  tang_->Complete();

  // calculate consistent initial temperature rates
  {
    Teuchos::RCP<Epetra_Vector> rhs = LINALG::CreateVector(*dofrowmap_, true);
    rhs->Update(-1.0, *fint, 1.0, *fext, -1.0);
    // blank RHS on DBC DOFs
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), rhs);
    solver_->Solve(tang_->EpetraMatrix(), (*rate_)(0), rhs, true, true);
  }

  // We need to reset the tangent matrix because its graph (topology)
  // is not finished yet in case of constraints and posssibly other side
  // effects (basically managers).
  tang_->Reset();

  // leave this hell
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate Dirichlet BC at t_{n+1}                       bborn 06/08  |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyDirichletBC(
  const double time,
  Teuchos::RCP<Epetra_Vector> temp,
  Teuchos::RCP<Epetra_Vector> rate,
  bool recreatemap
  )
{
  // apply DBCs
  // needed parameters
  ParameterList p;
  p.set("total time", time);  // target time

  // predicted Dirichlet values
  // \c temp then also holds prescribed new Dirichlet temperatures
  discret_->ClearState();
  if (recreatemap)
  {
    discret_->EvaluateDirichlet(p, temp, rate, Teuchos::null,
                                Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->EvaluateDirichlet(p, temp, rate, Teuchos::null,
                                Teuchos::null, Teuchos::null);
  }
  discret_->ClearState();

  // ciao
  return;
}

/*----------------------------------------------------------------------*
 |  Update time and step counter                            bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::UpdateStepTime()
{
  // update time and step
  time_->UpdateSteps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;  // n := n+1

  timen_ += (*dt_)[0];
  stepn_ += 1;

  // new deal
  return;
}

/*----------------------------------------------------------------------*
 |  Reset configuration after time step                     bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ResetStep()
{
  // reset state vectors
  tempn_->Update(1.0, (*temp_)[0], 0.0);
  raten_->Update(1.0, (*rate_)[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    ParameterList p;
    p.set("action", "calc_thermo_reset_istep");
    // go to elements
    discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                       Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  // I am gone
  return;
}

/*----------------------------------------------------------------------*
 |  Read and set restart values                             bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ReadRestart(const int step)
{
  IO::DiscretizationReader reader(discret_, step);
  if (step != reader.ReadInt("step"))
    dserror("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, reader.ReadDouble("time")));
  timen_ = (*time_)[0] + (*dt_)[0];

  ReadRestartState();
  ReadRestartForce();

  // fix pointer to #dofrowmap_, which has not really changed, but is
  // located at different place
  dofrowmap_ = discret_->DofRowMap();
}

/*----------------------------------------------------------------------*
 |  Read and set restart state                              bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ReadRestartState()
{
  IO::DiscretizationReader reader(discret_, step_);
  reader.ReadVector(tempn_, "temperature");
  temp_->UpdateSteps(*tempn_);
  reader.ReadVector(raten_, "rate");
  rate_->UpdateSteps(*raten_);
  reader.ReadMesh(step_);
  return;
}

/*----------------------------------------------------------------------*
 |  output to file                                          mwgee 03/07 |
 *----------------------------------------------------------------------*/
void THR::TimInt::OutputStep()
{
  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if (writerestartevery_ and (step_%writerestartevery_ == 0) )
  {
    OutputRestart(datawritten);
  }

  // output results (not necessary if restart in same step)
  if ( writeglob_
       and writeglobevery_ and (step_%writeglobevery_ == 0)
       and (not datawritten) )
  {
    OutputState(datawritten);
  }

  // output heatflux & tempgrad
  if ( writeelemevery_
       and ( (writeheatflux_ != INPAR::THR::heatflux_none)
             or (writetempgrad_ != INPAR::THR::tempgrad_none) )
       and (step_%writeelemevery_ == 0) )
  {
    OutputHeatfluxTempgrad(datawritten);
  }

  // output energy
  if ( writeenergyevery_ and (step_%writeenergyevery_ == 0) )
  {
    OutputEnergy();
  }

  // what's next?
  return;
}

/*----------------------------------------------------------------------*
 |  write restart                                           mwgee 03/07 |
 *----------------------------------------------------------------------*/
void THR::TimInt::OutputRestart(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write restart output, please
  output_->WriteMesh(step_, (*time_)[0]);
  output_->NewStep(step_, (*time_)[0]);
  output_->WriteVector("temperature", (*temp_)(0));
  output_->WriteVector("rate", (*rate_)(0));
  output_->WriteVector("fexternal", Fext());

  // info dedicated to user's eyes staring at standard out
  if ( (myrank_ == 0) and printscreen_)
  {
    printf("====== Restart written in step %d\n", step_);
    // print a beautiful line made exactly of 80 dashes
    printf("--------------------------------------------------------------"
            "------------------\n");
    fflush(stdout);
  }

  // info dedicated to processor error file
  if (printerrfile_)
  {
    fprintf(errfile_, "====== Restart written in step %d\n", step_);
    fprintf(errfile_,"--------------------------------------------------------------"
            "------------------\n");
    fflush(errfile_);
  }

  // we will say what we did
  return;
}

/*----------------------------------------------------------------------*
 |  output temperature,temperature rate                     bborn 06/08 |
 |  originally by mwgee 03/07                                           |
 *----------------------------------------------------------------------*/
void THR::TimInt::OutputState(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write now
  output_->NewStep(step_, (*time_)[0]);
  output_->WriteVector("temperature", (*temp_)(0));
  output_->WriteVector("rate", (*rate_)(0));
  output_->WriteVector("fexternal", Fext());
  output_->WriteElementData();

  // leave for good
  return;
}

/*----------------------------------------------------------------------*
 |  heatflux calculation and output                         bborn 06/08 |
 |  originally by lw                                                    |
 *----------------------------------------------------------------------*/
void THR::TimInt::OutputHeatfluxTempgrad(bool& datawritten)
{
  // create the parameters for the discretization
  ParameterList p;
  // action for elements
  p.set("action", "proc_thermo_heatflux");
  // other parameters that might be needed by the elements
  p.set("total time", (*time_)[0]);
  p.set("delta time", (*dt_)[0]);

  Teuchos::RCP<std::vector<char> > heatfluxdata
    = Teuchos::rcp(new std::vector<char>());
  p.set("heatflux", heatfluxdata);
  p.set("ioheatflux", writeheatflux_);

  Teuchos::RCP<std::vector<char> > tempgraddata
    = Teuchos::rcp(new std::vector<char>());
  p.set("tempgrad", tempgraddata);
  p.set("iotempgrad", writetempgrad_);

  // set vector values needed by elements
  discret_->ClearState();
  // SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0, "residual temperature", zeros_);
  discret_->SetState(0, "temperature", (*temp_)(0));
  discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                     Teuchos::null, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // Make new step
  if (not datawritten)
  {
    output_->NewStep(step_, (*time_)[0]);
  }
  datawritten = true;

  // write heatflux
  if (writeheatflux_ != INPAR::THR::heatflux_none)
  {
    std::string heatfluxtext = "";
    if (writeheatflux_ == INPAR::THR::heatflux_current)
    {
     heatfluxtext = "gauss_current_heatfluxes_xyz";
    }
    else if (writeheatflux_ == INPAR::THR::heatflux_initial)
    {
      heatfluxtext = "gauss_initial_heatfluxes_xyz";
    }
    else
    {
      dserror("requested heatflux type not supported");
    }
    output_->WriteVector(heatfluxtext, *heatfluxdata,
                         *(discret_->ElementColMap()));
  }

  // write temperature gradient
  if (writetempgrad_ != INPAR::THR::tempgrad_none)
  {
    std::string tempgradtext = "";
    if (writetempgrad_ == INPAR::THR::tempgrad_current)
    {
      tempgradtext = "gauss_current_tempgrad_xyz";
    }
    else if (writetempgrad_ == INPAR::THR::tempgrad_initial)
    {
      tempgradtext = "gauss_initial_tempgrad_xyz";
    }
    else
    {
      dserror("requested tempgrad type not supported");
    }
    output_->WriteVector(tempgradtext, *tempgraddata,
                         *(discret_->ElementColMap()));
  }

  // the way they do the output in SCATRA  10.11.09

/*  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // output heatflux
  if (writeheatflux_ != INPAR::THR::heatflux_none)
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action", "process_heatflux");
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    p.set("ioheatflux", writeheatflux_);

    // We have to treat each spatial direction separately
    Teuchos::RCP<Epetra_Vector> fluxx = LINALG::CreateVector(*dofrowmap,true);
    Teuchos::RCP<Epetra_Vector> fluxy = LINALG::CreateVector(*dofrowmap,true);
    Teuchos::RCP<Epetra_Vector> fluxz = LINALG::CreateVector(*dofrowmap,true);

    // set vector values needed by elements
    discret_->ClearState();
    //discret_->SetState("residual temperature", zeros_);
    discret_->SetState("temperature", (*temp_)(0));
    discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                       fluxx,fluxy,fluxz);
    discret_->ClearState();

    // empty vector for heat flux vectors (always 3D)
    Teuchos::RCP<Epetra_MultiVector> flux = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));
    // insert values into final flux vector for visualization
    for (int i = 0;i<flux->MyLength();++i)
    {
      flux->ReplaceMyValue(i,0,(*fluxx)[i]);
      flux->ReplaceMyValue(i,1,(*fluxy)[i]);
      flux->ReplaceMyValue(i,2,(*fluxz)[i]);
    }

    // post_drt_ensight does not support multivectors based on the dofmap
    // for now, I create single vectors that can be handled by the filter

    // get the noderowmap
    const Epetra_Map* noderowmap = discret_->NodeRowMap();
    Teuchos::RCP<Epetra_MultiVector> fluxk = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,3,true));

    for (int i = 0;i<fluxk->MyLength();++i)
    {
      DRT::Node* actnode = discret_->lRowNode(i);
      int dofgid = discret_->Dof(actnode,k-1);
      fluxk->ReplaceMyValue(i,0,((*flux)[0])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,1,((*flux)[1])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,2,((*flux)[2])[(flux->Map()).LID(dofgid)]);
    }

    output_->WriteVector("heatflux", fluxk, IO::DiscretizationWriter::nodevector);
  }

  // output temperature gradient
  if (writetempgrad_ != INPAR::THR::tempgrad_none)
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action", "process_tempgrad");
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    p.set("iotempgrad", writetempgrad_);

    // We have to treat each spatial direction separately
    Teuchos::RCP<Epetra_Vector> fluxx = LINALG::CreateVector(*dofrowmap,true);
    Teuchos::RCP<Epetra_Vector> fluxy = LINALG::CreateVector(*dofrowmap,true);
    Teuchos::RCP<Epetra_Vector> fluxz = LINALG::CreateVector(*dofrowmap,true);

    // set vector values needed by elements
    discret_->ClearState();
    //discret_->SetState("residual temperature", zeros_);
    discret_->SetState("temperature", (*temp_)(0));
    discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                       fluxx,fluxy,fluxz);
    discret_->ClearState();

    // empty vector for (normal) mass or heat flux vectors (always 3D)
    Teuchos::RCP<Epetra_MultiVector> flux = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));
    // insert values into final flux vector for visualization
    for (int i = 0;i<flux->MyLength();++i)
    {
      flux->ReplaceMyValue(i,0,(*fluxx)[i]);
      flux->ReplaceMyValue(i,1,(*fluxy)[i]);
      flux->ReplaceMyValue(i,2,(*fluxz)[i]);
    }

    // post_drt_ensight does not support multivectors based on the dofmap
    // for now, I create single vectors that can be handled by the filter

    // get the noderowmap
    const Epetra_Map* noderowmap = discret_->NodeRowMap();
    Teuchos::RCP<Epetra_MultiVector> fluxk = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,3,true));

    for (int i = 0;i<fluxk->MyLength();++i)
    {
      DRT::Node* actnode = discret_->lRowNode(i);
      int dofgid = discret_->Dof(actnode,k-1);
      fluxk->ReplaceMyValue(i,0,((*flux)[0])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,1,((*flux)[1])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,2,((*flux)[2])[(flux->Map()).LID(dofgid)]);
    }

    output_->WriteVector("tempgrad", fluxk, IO::DiscretizationWriter::nodevector);
  }
*/
  // leave me alone
  return;
}

/*----------------------------------------------------------------------*
 |  output system energies                                  bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::OutputEnergy()
{
  // internal/tempgrad energy
  double intergy = 0.0;  // total internal energy
  {
    ParameterList p;
    // other parameters needed by the elements
    p.set("action", "calc_thermo_energy");

    // set vector values needed by elements
    discret_->ClearState();
    // SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0, "temperature", (*temp_)(0));
    // get energies
    Teuchos::RCP<Epetra_SerialDenseVector> energies
      = Teuchos::rcp(new Epetra_SerialDenseVector(1));
    discret_->EvaluateScalars(p, energies);
    discret_->ClearState();
    intergy = (*energies)(0);
  }

  // global calculation of kinetic energy
  double kinergy = 0.0;  // total kinetic energy
//  {
//    Teuchos::RCP<Epetra_Vector> linmom
//      = LINALG::CreateVector(*dofrowmap_, true);
//    capa_->Multiply(false, (*rate_)[0], *linmom);
//    linmom->Dot((*rate_)[0], &kinergy);
//    kinergy *= 0.5;
//  }

  // external energy
  double extergy = 0.0;  // total external energy
  {
    // WARNING: This will only work with dead loads!!!
    Teuchos::RCP<Epetra_Vector> fext = Fext();
    fext->Dot((*temp_)[0], &extergy);
  }

  // total energy
  double totergy = kinergy + intergy - extergy;

  // the output
  if (myrank_ == 0)
  {
    *energyfile_ << " " << std::setw(9) << step_
                 << std::scientific  << std::setprecision(16)
                 << " " << (*time_)[0]
                 << " " << totergy
                 << " " << kinergy
                 << " " << intergy
                 << " " << extergy
                 << std::endl;
  }

  // in God we trust
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate external forces at t_{n+1}                     bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyForceExternal(
  const double time,  //!< evaluation time
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  Teuchos::RCP<Epetra_Vector>& fext  //!< external force
  )
{
  Teuchos::ParameterList p;
  // action for elements
  const std::string action = "calc_thermo_fext";
  p.set("action", action);
  // type of calling time integrator
  p.set("time integrator", MethodName());
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->ClearState();
  // SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0, "temperature", temp);
  // get load vector
  discret_->EvaluateNeumann(p, *fext);
  discret_->ClearState();

  // go away
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate ordinary internal force, its tangent at state  bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyForceTangInternal(
  Teuchos::ParameterList& p,
  const double time,
  const double dt,
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  const Teuchos::RCP<Epetra_Vector> tempi,  //!< residual temperature
  Teuchos::RCP<Epetra_Vector> fint,  //!< internal force
  Teuchos::RCP<LINALG::SparseMatrix> tang  //!< tangent matrix
  )
{
  // type of calling time integrator
  p.set<INPAR::THR::DynamicType>("time integrator", MethodName());
  // action for elements
  const std::string action = "calc_thermo_fintcond";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->ClearState();
  // SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0, "residual temperature", tempi);
  discret_->SetState(0, "temperature", temp);
  // set displacements and velocities for the coupled TSI problem
  if(disn_!=Teuchos::null)
  {
    discret_->SetState(1,"displacement",disn_);
  }
  if(veln_!=Teuchos::null)
  {
    discret_->SetState(1,"velocity",veln_);
  }
  discret_->Evaluate(p, tang, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // that's it
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate ordinary internal force, its tangent at state  bborn 10/09 |
 |  overloaded function specified for ost time integration              |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyForceTangInternal(
  Teuchos::ParameterList& p,
  const double time,
  const double dt,
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  const Teuchos::RCP<Epetra_Vector> tempi,  //!< residual temperature
  Teuchos::RCP<Epetra_Vector> fcap,  //!< stored force
  Teuchos::RCP<Epetra_Vector> fint,  //!< internal force
  Teuchos::RCP<LINALG::SparseMatrix> tang  //!< tangent matrix
  )
{
  // type of calling time integrator
  p.set<INPAR::THR::DynamicType>("time integrator", MethodName());
  // action for elements
  const std::string action = "calc_thermo_finttang";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->ClearState();
  // SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0,"residual temperature", tempi);
  discret_->SetState(0,"temperature", temp);
  // set displacements and velocities for the coupled TSI problem
  if(disn_!=Teuchos::null)
  {
    discret_->SetState(1,"displacement",disn_);
  }
  if(veln_!=Teuchos::null)
  {
    discret_->SetState(1,"velocity",veln_);
  }

  discret_->Evaluate(p, tang, Teuchos::null, fint, Teuchos::null, fcap);
  discret_->ClearState();

  // that's it
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate ordinary internal force                        bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyForceInternal(
  Teuchos::ParameterList& p,
  const double time,
  const double dt,
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  const Teuchos::RCP<Epetra_Vector> tempi,  //!< incremental temperature
  Teuchos::RCP<Epetra_Vector> fint  //!< internal force
  )
{
  // type of calling time integrator
  p.set("time integrator", MethodName());
  // action for elements
  const std::string action = "calc_thermo_fint";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  // set vector values needed by elements
  discret_->ClearState();
  // SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0, "residual temperature", tempi);  // these are incremental
  discret_->SetState(0, "temperature", temp);
  // set displacements and velocities for the coupled TSI problem
  if(disn_!=Teuchos::null)
  {
    discret_->SetState(1,"displacement",disn_);
  }
  if(veln_!=Teuchos::null)
  {
    discret_->SetState(1,"velocity",veln_);
  }
  discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                     fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // where the fun starts
  return;
}


/*----------------------------------------------------------------------*
 |  get current displacements and  velocities from the       dano 05/10 |
 |  structure discretization                                            |
 *----------------------------------------------------------------------*/
void THR::TimInt::ApplyStructVariables(
  Teuchos::RCP<Epetra_Vector> idisp,  ///< the current velocities
  Teuchos::RCP<Epetra_Vector> ivel  ///< the current velocities
  )
{
  disn_ = idisp;
  veln_ = ivel;
  // where the fun starts
  return;
}


/*----------------------------------------------------------------------*
 |  integrate                                               bborn 06/08 |
 *----------------------------------------------------------------------*/
void THR::TimInt::Integrate()
{
  dserror("Integrate() will not be used but ADAPTER::Thermo::Integrate()!");

  /*
  // target time #timen_ and step #stepn_ already set

  // time loop
  while ( ((timen_+((1e-10)*((*dt_)[0]))) < timemax_) and (stepn_ <= stepmax_) )
  {
    // integrate time step
    // after this step we hold tempn_, etc
    IntegrateStep();

    // update temperature and temperature rate
    // after this call we will have tempn_==temp_, etc
    UpdateStepState();

    // update time and step
    UpdateStepTime();

    // print info about finished time step
    PrintStep();

    // write output
    OutputStep();
  }

  // print monitoring of time consumption
  TimeMonitor::summarize();

  // that's it
  return;
  */
}

/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
