/*!----------------------------------------------------------------------
\file
\brief

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "stru_static_drt.H"
#include "../io/io_drt.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/linalg_systemmatrix.H"
#include "stru_resulttest.H"


/*----------------------------------------------------------------------*
 |                                                         maf 05/07    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*!----------------------------------------------------------------------
\brief file pointers

<pre>                                                           maf 05/07
This structure struct _FILES allfiles is defined in input_control_global.c
and the type is in standardtypes.h
It holds all file pointers and some variables needed for the FRSYSTEM
</pre>
*----------------------------------------------------------------------*/
extern struct _FILES  allfiles;

/*----------------------------------------------------------------------*
 | global variable *solv, vector of lenght numfld of structures SOLVAR  |
 | defined in solver_control.c                                          |
 |                                                                      |
 |                                                         maf 05/07    |
 *----------------------------------------------------------------------*/
extern struct _SOLVAR  *solv;

/*----------------------------------------------------------------------*
 |                                                         maf 05/07    |
 | pointer to allocate static variables if needed                       |
 | defined in global_control.c                                          |
 *----------------------------------------------------------------------*/
extern struct _STATIC_VAR  *statvar;

/*----------------------------------------------------------------------*
  | structural nonlinear static solution routine             maf 05/07  |
 *----------------------------------------------------------------------*/
void stru_static_drt()
{
  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  RefCountPtr<DRT::Discretization> actdis = null;
  actdis = DRT::Problem::Instance()->Dis(genprob.numsf,0);

  // set degrees of freedom in the discretization
  if (!actdis->Filled()) actdis->FillComplete();
  
  // -------------------------------------------------------------------
  // check for contact case
  // -------------------------------------------------------------------
  const Teuchos::ParameterList& scontact = DRT::Problem::Instance()->StructuralContactParams();
  bool contact = Teuchos::getIntegralValue<int>(scontact,"CONTACT");
  bool initialcontact = Teuchos::getIntegralValue<int>(scontact,"INIT_CONTACT");
  
  if (contact)
  {
    // leave and call special routine for statics with contact
    CONTACT::contact_stru_static_drt(initialcontact);
    return;
  } 
  
  // -------------------------------------------------------------------
  // get a communicator and myrank
  // -------------------------------------------------------------------
  const Epetra_Comm& Comm = actdis->Comm();
  const int myrank = Comm.MyPID();

  //----------------------------------------------------- get error file
  FILE* errfile = allfiles.out_err;

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  SOLVAR*         actsolv  = &solv[0];
  const Teuchos::ParameterList& ioflags  = DRT::Problem::Instance()->IOParams();

  //-----------------------------------------------------create a solver
  RefCountPtr<ParameterList> solveparams = rcp(new ParameterList());
  LINALG::Solver solver(solveparams,actdis->Comm(),allfiles.out_err);
  solver.TranslateSolverParameters(*solveparams,actsolv);
  actdis->ComputeNullSpaceIfNecessary(*solveparams);

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = actdis->DofRowMap();

  // -------------------------------------------------------------------
  // create empty stiffness matrix
  // -------------------------------------------------------------------
  // `81' is an initial guess for the bandwidth of the matrices
  // A better guess will be determined later.
  RefCountPtr<LINALG::SparseMatrix> stiff_mat = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,81));

  // -------------------------------------------------------------------
  // create empty vectors
  // -------------------------------------------------------------------
  // a zero vector of full length
  RefCountPtr<Epetra_Vector> zeros = LINALG::CreateVector(*dofrowmap,true);
  // vector of full length; for each component
  //                /  1   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  0   i-th DOF is free
  RefCountPtr<Epetra_Vector> dirichtoggle = LINALG::CreateVector(*dofrowmap,true);
  // opposite of dirichtoggle vector, ie for each component
  //                /  0   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  1   i-th DOF is free
  RefCountPtr<Epetra_Vector> invtoggle = LINALG::CreateVector(*dofrowmap,false);
  // displacements D_{n} at last time
  RefCountPtr<Epetra_Vector> dis = LINALG::CreateVector(*dofrowmap,true);

  // displacements D_{n+1} at new time
  RefCountPtr<Epetra_Vector> disn = LINALG::CreateVector(*dofrowmap,true);

  // iterative displacement increments IncD_{n+1}
  // also known as residual displacements
  RefCountPtr<Epetra_Vector> disi = LINALG::CreateVector(*dofrowmap,true);

  // internal force vector F_int at different times
  RefCountPtr<Epetra_Vector> fint = LINALG::CreateVector(*dofrowmap,true);
  // external force vector F_ext at last times
  RefCountPtr<Epetra_Vector> fext = LINALG::CreateVector(*dofrowmap,true);
  // external force vector F_{n+1} at new time
  RefCountPtr<Epetra_Vector> fextn = LINALG::CreateVector(*dofrowmap,true);

  // dynamic force residual at mid-time R_{n+1-alpha}
  // also known at out-of-balance-force
  RefCountPtr<Epetra_Vector> fresm = LINALG::CreateVector(*dofrowmap,false);

  if (statvar->nr_controltyp != control_load) dserror("Only load control implemented");

  /*
  ** solution control parameters are inherited from dynamic routine:
  ** dt     = stepsize
  ** istep  = load step index
  ** time   = redundant, equals istep*dt
  */
  //------------------------------------------ time integration parameters
  const double dt = statvar->stepsize;
  int istep = 0;
  double time = 0.0;  // we should add an input parameter
  double timen;

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  IO::DiscretizationWriter output(actdis);
  if (genprob.restart){
    int restartstep = genprob.restart;
    RefCountPtr<DRT::Discretization> rcpdiscret(actdis);
    rcpdiscret.release();
    IO::DiscretizationReader reader(rcpdiscret,restartstep);
    double rtime  = reader.ReadDouble("time");
    int    rstep = reader.ReadInt("step");
    if (rstep != restartstep) dserror("Time step on file not equal to given step");

    reader.ReadVector(dis, "displacement");
    //reader.ReadVector(fext, "fexternal");
    //reader.ReadMesh(restartstep);

    // override current time and step with values from file
    time = rtime;
    istep = rstep;
  }

  // write mesh always at beginning of calc or restart
  output.WriteMesh(istep,time);


  //-------------------------------- calculate external force distribution
  //---- which is scaled by the load factor lambda (itself stays constant)
  {
    ParameterList params;
    // action for elements
    params.set("action","calc_struct_eleload");

    //other parameters needed by the elements
    params.set("total time",time);
    params.set("delta time",dt);

    // set vector values needed by elements
    actdis->ClearState();
    actdis->SetState("displacement",dis);
    // predicted dirichlet values
    // dis then also holds prescribed new dirichlet displacements
    actdis->EvaluateDirichlet(params,dis,null,null,dirichtoggle);
    actdis->ClearState();
    actdis->SetState("displacement",dis);
    // predicted rhs
    actdis->EvaluateNeumann(params,*fext); // *fext holds external force vector
    actdis->ClearState();
  }

  //----------------------- compute an inverse of the dirichtoggle vector
  invtoggle->PutScalar(1.0);
  invtoggle->Update(-1.0,*dirichtoggle,1.0);

  //------------------------------------------------- output initial state
  output.NewStep(istep, time);
  output.WriteVector("displacement", dis);
  output.WriteElementData();

  //---------------------------------------------- do "stress" calculation
  int mod_stress = istep % statvar->resevry_stress;
  string iostress;
  switch (Teuchos::getIntegralValue<STRUCT_STRESS_TYP>(ioflags,"STRUCT_STRESS"))
  {
  case struct_stress_none:
    iostress = "none";
    break;
  case struct_stress_cauchy:
    iostress = "cauchy";
    break;
  case struct_stress_pk:
    iostress = "2PK";
    break;
  default:
    iostress = "none";
    break;
  }
  if (!mod_stress && iostress!="none")
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_stress");
    // other parameters that might be needed by the elements
    p.set("total time",timen);
    p.set("delta time",dt);
    Teuchos::RCP<std::vector<char> > stress = Teuchos::rcp(new std::vector<char>());
    Teuchos::RCP<std::vector<char> > strain = Teuchos::rcp(new std::vector<char>());
    p.set("stress", stress);
    p.set("strain", strain);
    if (iostress == "cauchy")   // output of Cauchy stresses instead of 2PK stresses
    {
      p.set("cauchy", true);
    }
    else
    {
      p.set("cauchy", false);
    }
    // set vector values needed by elements
    actdis->ClearState();
    actdis->SetState("residual displacement",zeros);
    actdis->SetState("displacement",dis);
    actdis->Evaluate(p,null,null,null,null,null);
    actdis->ClearState();
    if (iostress == "cauchy")
    {
      output.WriteVector("gauss_cauchy_stresses_xyz",*stress,*(actdis->ElementColMap()));
    }
    else
    {
      output.WriteVector("gauss_2PK_stresses_xyz",*stress,*(actdis->ElementColMap()));
    }
    if (Teuchos::getIntegralValue<int>(ioflags,"STRUCT_STRAIN")==1)
    {
      output.WriteVector("gauss_GL_strains_xyz",*strain,*(actdis->ElementColMap()));
    }
  }

  //---------------------------------------------end of output initial state

  //========================================== start of time/loadstep loop
  while ( istep < statvar->nstep)
  {
    //------------------------------------------------------- current time
    // we are at t_{n} == time; the new time is t_{n+1} == time+dt
    timen = time + dt;

    //--------------------------------------------------- predicting state
    // constant predictor : displacement in domain
    disn->Update(1.0, *dis, 0.0);

    // eval fint and stiffness matrix at current istep
    // and apply new displacements at DBCs
    {
      // destroy and create new matrix
      stiff_mat->Zero();
      // create the parameters for the discretization
      ParameterList params;
      // action for elements
      params.set("action","calc_struct_nlnstiff");
      // other parameters needed by the elements
      params.set("total time",timen);  // load factor (pseudo time)
      params.set("delta time",dt);  // load factor increment (pseudo time increment)
      // set vector values needed by elements
      actdis->ClearState();
      actdis->SetState("residual displacement",disi);
      // predicted dirichlet values
      // disn then also holds prescribed new dirichlet displacements
      actdis->EvaluateDirichlet(params,disn,null,null,dirichtoggle);
      actdis->SetState("displacement",disn);
      fint->PutScalar(0.0);  // initialise internal force vector
      actdis->Evaluate(params,stiff_mat,null,fint,null,null);
      // predicted rhs
      fextn->PutScalar(0.0);  // initialize external force vector (load vect)
      actdis->EvaluateNeumann(params,*fextn); // *fext holds external force vector at current step

      actdis->ClearState();
      }
    // complete stiffness matrix
    stiff_mat->Complete();

    double stiffnorm;
    stiffnorm = stiff_mat->NormFrobenius();

    // evaluate residual at current istep
    // R{istep,numiter=0} = F_int{istep,numiter=0} - F_ext{istep}
    fresm->Update(1.0,*fint,-1.0,*fextn,0.0);

    // blank residual at DOFs on Dirichlet BC
    {
      Epetra_Vector fresmcopy(*fresm);
      fresm->Multiply(1.0,*invtoggle,fresmcopy,0.0);
    }

    //------------------------------------------------ build residual norm
    double norm;
    fresm->Norm2(&norm);
    if (!myrank) cout << " Predictor residual forces " << norm << endl; fflush(stdout);
   //=================================================== equilibrium loop
    int numiter=0;
    while ((norm > statvar->toldisp) && numiter < statvar->maxiter)
    {
      //----------------------- apply dirichlet BCs to system of equations
      fresm->Scale(-1.0);     // rhs = -R = -fresm
      disi->PutScalar(0.0);   // Useful? depends on solver and more
      LINALG::ApplyDirichlettoSystem(stiff_mat,disi,fresm,zeros,dirichtoggle);

      //Do usual newton step
      // solve for disi
      // Solve K . IncD = -R  ===>  IncD_{n+1}
      if (numiter==0)
      {
        solver.Solve(stiff_mat->EpetraMatrix(),disi,fresm,true,true);
      }
      else
      {
        solver.Solve(stiff_mat->EpetraMatrix(),disi,fresm,true,false);
      }
      // update displacements
      // D_{istep,numiter+1} := D_{istep,numiter} + IncD_{numiter}
      disn->Update(1.0, *disi, 1.0);

      // compute internal forces and stiffness at current iterate numiter
      {
        // zero out stiffness
        stiff_mat->Zero();
        // create the parameters for the discretization
        ParameterList params;
        // action for elements
        params.set("action","calc_struct_nlnstiff");
        // other parameters needed by the elements
        params.set("total time",timen);  // load factor (pseudo time)
        params.set("delta time",dt);  // load factor increment (pseudo time increment)
        // set vector values needed by elements
        actdis->ClearState();
        actdis->SetState("residual displacement",disi);
        actdis->SetState("displacement",disn);
        fint->PutScalar(0.0);  // initialise internal force vector
        actdis->Evaluate(params,stiff_mat,null,fint,null,null);

        actdis->ClearState();
      }
      // complete stiffness matrix
      stiff_mat->Complete();

      // evaluate new residual fresm at current iterate numiter
      // R{istep,numiter} = F_int{istep,numiter} - F_ext{istep}
      fresm->Update(1.0,*fint,-1.0,*fextn,0.0);

      // blank residual DOFs which are on Dirichlet BC
      {
        Epetra_Vector fresmcopy(*fresm);
        fresm->Multiply(1.0,*invtoggle,fresmcopy,0.0);
      }

      //---------------------------------------------- build residual norm
      double disinorm;
      disi->Norm2(&disinorm);

      fresm->Norm2(&norm);
      // a short message
      if (!myrank)
      {
        printf("numiter %d res-norm %e dis-norm %e \n",numiter+1, norm, disinorm);
        fprintf(errfile,"numiter %d res-norm %e dis-norm %e\n",numiter+1, norm, disinorm);
        fflush(stdout);
        fflush(errfile);
      }
      // criteria to stop Newton iteration
      norm = disinorm;

      //--------------------------------- increment equilibrium loop index
      ++numiter;
    } //
    //============================================= end equilibrium loop

    //-------------------------------- test whether max iterations was hit
    if (statvar->maxiter == 1 && statvar->nstep == 1)
      printf("computed 1 step with 1 iteration: STATIC LINEAR SOLUTION\n");
    else if (numiter==statvar->maxiter)
      dserror("Newton unconverged in %d iterations",numiter);

    //---------------------------- determine new end-quantities and update
    // new displacements at t_{n+1} -> t_n
    // D_{n} := D_{n+1}
    dis->Update(1.0, *disn, 0.0);

    //----- update anything that needs to be updated at the element level
    {
      // create the parameters for the discretization
      ParameterList params;
      // action for elements
      params.set("action","calc_struct_update_istep");
      // other parameters that might be needed by the elements
      params.set("total time",timen);
      params.set("delta time",dt);
      actdis->Evaluate(params,null,null,null,null,null);
    }

    //------------------------------------------ increment time/load step
    ++istep;      // load step n := n + 1
    time += dt;   // load factor / pseudo time  t_n := t_{n+1} = t_n + Delta t

    //------------------------------------------------- write restart step
    bool isdatawritten = false;
    if (istep % statvar->resevery_restart==0)
    {
      output.WriteMesh(istep,time);
      output.NewStep(istep, time);
      output.WriteVector("displacement",dis);
      //output.WriteVector("fexternal", fext);
      isdatawritten = true;

      if (!myrank)
      {
        cout << "====== Restart written in step " << istep << endl;
        fflush(stdout);
        fprintf(errfile,"====== Restart written in step %d\n",istep);
        fflush(errfile);
      }
    }

    //----------------------------------------------------- output results
    int mod_disp   = istep % statvar->resevry_disp;
    if (!mod_disp && Teuchos::getIntegralValue<int>(ioflags,"STRUCT_DISP")==1 && !isdatawritten)
    {
      output.NewStep(istep, time);
      output.WriteVector("displacement", dis);
      output.WriteElementData();
      isdatawritten = true;
    }

    //---------------------------------------------- do stress calculation
    int mod_stress = istep % statvar->resevry_stress;
    string iostress;
    switch (Teuchos::getIntegralValue<STRUCT_STRESS_TYP>(ioflags,"STRUCT_STRESS"))
    {
    case struct_stress_none:
      iostress = "none";
      break;
    case struct_stress_cauchy:
      iostress = "cauchy";
      break;
    case struct_stress_pk:
      iostress = "2PK";
      break;
    default:
      iostress = "none";
      break;
    }
    if (!mod_stress && iostress!="none")
    {
      // create the parameters for the discretization
      ParameterList p;
      // action for elements
      p.set("action","calc_struct_stress");
      // other parameters that might be needed by the elements
      p.set("total time",timen);
      p.set("delta time",dt);
      Teuchos::RCP<std::vector<char> > stress = Teuchos::rcp(new std::vector<char>());
      Teuchos::RCP<std::vector<char> > strain = Teuchos::rcp(new std::vector<char>());
      p.set("stress", stress);
      p.set("strain", strain);
      if (iostress=="cauchy")   // output of Cauchy stresses instead of 2PK stresses
      {
        p.set("cauchy", true);
      }
      else
      {
        p.set("cauchy", false);
      }
      // set vector values needed by elements
      actdis->ClearState();
      actdis->SetState("residual displacement",zeros);
      actdis->SetState("displacement",dis);
      actdis->Evaluate(p,null,null,null,null,null);
      actdis->ClearState();
      if (!isdatawritten) output.NewStep(istep, timen);
      isdatawritten = true;
      if (iostress=="cauchy")
      {
        output.WriteVector("gauss_cauchy_stresses_xyz",*stress,*(actdis->ElementColMap()));
      }
      else
      {
        output.WriteVector("gauss_2PK_stresses_xyz",*stress,*(actdis->ElementColMap()));
      }
      if (Teuchos::getIntegralValue<int>(ioflags,"STRUCT_STRAIN")==1)
      {
        output.WriteVector("gauss_GL_strains_xyz",*strain,*(actdis->ElementColMap()));
      }
    }

//
//
//    ofstream f_system("stresses.gmsh");
//    stringstream gmshfilecontent;
//    gmshfilecontent << "View \" Solid Elements stresses \" {" << endl;
//    // plot elements
//    for (int i=0; i < actdis->NumMyColElements(); ++i)
//    {
//      if (actdis->lColElement(i)->Type() != DRT::Element::element_sosh8) continue;
//      DRT::ELEMENTS::So_sh8* actele = dynamic_cast<DRT::ELEMENTS::So_sh8*>(actdis->lColElement(i));
//      if (!actele) dserror("cast to So_sh8* failed");
//      // plot elements
//      gmshfilecontent << IO::GMSH::elementToString(actele->thickdir_, actele) << endl;
//      Epetra_SerialDenseMatrix* mystresses(8,8);
//      mystresses = actele->data_.Get<Epetra_SerialDenseMatrix>("Stresses");
//      cout << *mystresses;
//    }
//    gmshfilecontent << "};" << endl;

    //---------------------------------------------------------- print out
    if (!myrank)
    {
      printf("step %6d | nstep %6d | time %-14.8E | dt %-14.8E | numiter %3d\n",
             istep,statvar->nstep,timen,dt,numiter);
      fprintf(errfile,"step %6d | nstep %6d | time %-14.8E | dt %-14.8E | numiter %3d\n",
              istep,statvar->nstep,timen,dt,numiter);
      printf("----------------------------------------------------------------------------------\n");
      fprintf(errfile,"----------------------------------------------------------------------------------\n");
      fflush(stdout);
      fflush(errfile);
    }

  }  //=============================================end time/loadstep loop

  // Structure Resulttests
  DRT::ResultTestManager testmanager(actdis->Comm());
  testmanager.AddFieldTest(rcp(new StruResultTest(actdis,dis,null,null)));
  testmanager.TestAll();

  //----------------------------- this is the end my lonely friend the end
  return;
} // end of stru_static_drt()



#endif  // #ifdef CCADISCRET
