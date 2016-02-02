/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_implicit_service.cpp
\brief Service routines of the scalar transport time integration class

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_timint_implicit.H"
#include "scatra_timint_meshtying_strategy_base.H"

#include "../drt_adapter/adapter_coupling.H"

#include "../drt_fluid/fluid_rotsym_periodicbc_utils.H"
#include "../drt_fluid_turbulence/dyn_smag.H"
#include "../drt_fluid_turbulence/dyn_vreman.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_timecurve.H"

#include "../drt_nurbs_discret/drt_nurbs_discret.H"

#include "../drt_scatra_ele/scatra_ele_action.H"

#include "../linalg/linalg_solver.H"

#include "turbulence_hit_scalar_forcing.H"

// for AVM3 solver:
#include <MLAPI_Workspace.h>
#include <MLAPI_Aggregation.h>

// for printing electrode status to file
#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io_gmsh.H"
#include "../drt_io/io_pstream.H"


/*==========================================================================*
 |                                                                          |
 | public:                                                                  |
 |                                                                          |
 *==========================================================================*/

/*==========================================================================*/
//! @name general framework
/*==========================================================================*/

/*----------------------------------------------------------------------*
 |  calculate mass / heat flux vector                        gjb   04/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFlux(const bool writetofile, const int num)
{
  switch(writeflux_)
  {
  case INPAR::SCATRA::flux_total_domain:
  case INPAR::SCATRA::flux_diffusive_domain:
  {
    return CalcFluxInDomain(writeflux_);
    break;
  }
  case INPAR::SCATRA::flux_total_boundary:
  case INPAR::SCATRA::flux_diffusive_boundary:
  case INPAR::SCATRA::flux_convective_boundary:
  {
    // calculate normal flux vector field only for the user-defined boundary conditions:
    std::vector<std::string> condnames;
    condnames.push_back("ScaTraFluxCalc");

    return CalcFluxAtBoundary(condnames, writetofile, num);
    break;
  }
  default:
    break;
  }
  // else: we just return a zero vector field (needed for result testing)
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  return Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));
}

/*----------------------------------------------------------------------*
 |  calculate mass / heat flux vector field in comp. domain    gjb 06/09|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFluxInDomain
(const INPAR::SCATRA::FluxType fluxtype)
{
  // get a vector layout from the discretization to construct matching
  // vectors and matrices    local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // empty vector for (normal) mass or heat flux vectors (always 3D)
  Teuchos::RCP<Epetra_MultiVector> flux = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));

  // We have to treat each spatial direction separately
  Teuchos::RCP<Epetra_Vector> fluxx = LINALG::CreateVector(*dofrowmap,true);
  Teuchos::RCP<Epetra_Vector> fluxy = LINALG::CreateVector(*dofrowmap,true);
  Teuchos::RCP<Epetra_Vector> fluxz = LINALG::CreateVector(*dofrowmap,true);

  // we need a vector for the integrated shape functions
  Teuchos::RCP<Epetra_Vector> integratedshapefcts = LINALG::CreateVector(*dofrowmap,true);

  {
    Teuchos::ParameterList eleparams;
    eleparams.set<int>("action",SCATRA::integrate_shape_functions);
    // we integrate shape functions for the first numscal_ dofs per node!!
    Epetra_IntSerialDenseVector dofids(7); // make it big enough!
    for(int rr=0;rr<7;rr++)
    {
      dofids(rr) = -1; // do not integrate shape functions for these dofs
    }
    for (std::vector<int>::iterator it = writefluxids_->begin(); it!=writefluxids_->end(); ++it)
    {
      dofids((*it)-1) = (*it); // do not integrate shape functions for these dofs
    }
    eleparams.set("dofids",dofids);

    if (isale_)
      eleparams.set<int>("ndsdisp",nds_disp_);

    // evaluate fluxes in the whole computational domain
    // (e.g., for visualization of particle path-lines) or L2 projection for better consistency
    discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,integratedshapefcts,Teuchos::null,Teuchos::null);
  }

  // set action for elements
  Teuchos::ParameterList params;
  params.set<int>("action",SCATRA::calc_flux_domain);

  // provide velocity field and potentially acceleration/pressure field
  // (export to column map necessary for parallel evaluation)
  params.set<int>("ndsvel",nds_vel_);

  //provide displacement field in case of ALE
  if (isale_)
    params.set<int>("ndsdisp",nds_disp_);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);

  // evaluate fluxes in the whole computational domain (e.g., for visualization of particle path-lines)
  discret_->Evaluate(params,Teuchos::null,Teuchos::null,fluxx,fluxy,fluxz);

  // insert values into final flux vector for visualization
  // we do not solve a global equation system for the flux values here
  // but perform a lumped mass matrix approach, i.e., dividing by the values of
  // integrated shape functions
  for (int i = 0;i<flux->MyLength();++i)
  {
    const double intshapefct = (*integratedshapefcts)[i];
    // is zero at electric potential dofs
    if (abs(intshapefct) > EPS13)
    {
      flux->ReplaceMyValue(i,0,((*fluxx)[i])/intshapefct);
      flux->ReplaceMyValue(i,1,((*fluxy)[i])/intshapefct);
      flux->ReplaceMyValue(i,2,((*fluxz)[i])/intshapefct);
    }
  }

  // clean up
  discret_->ClearState();

  return flux;
} // SCATRA::ScaTraTimIntImpl::CalcFluxInDomain


/*----------------------------------------------------------------------*
 |  calculate mass / heat normal flux at specified boundaries  gjb 06/09|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFluxAtBoundary(
    std::vector<std::string>&   condnames,     //!< ?
    const bool                  writetofile,   //!< ?
    const int                   num            //!< field number
    )
{
  // The normal flux calculation is based on the idea proposed in
  // GRESHO ET AL.,
  // "THE CONSISTENT GALERKIN FEM FOR COMPUTING DERIVED BOUNDARY
  // QUANTITIES IN THERMAL AND/OR FLUIDS PROBLEMS",
  // INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN FLUIDS, VOL. 7, 371-394 (1987)
  // For the moment, we are lumping the 'boundary mass matrix' instead of solving
  // a small linear system!

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // empty vector for (normal) mass or heat flux vectors (always 3D)
  Teuchos::RCP<Epetra_MultiVector> flux = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));

  // determine the averaged normal vector field for indicated boundaries
  // used for the output of the normal flux as a vector field
  // is computed only once; for ALE formulation recalculation is necessary
  if ((normals_ == Teuchos::null) or (isale_== true))
    normals_ = ComputeNormalVectors(condnames);

  if (writeflux_==INPAR::SCATRA::flux_convective_boundary)
  {
    // zero out trueresidual vector -> we do not need this info
    trueresidual_->PutScalar(0.0);
  }
  else
  {
    // was the residual already prepared?
    if ((solvtype_!=INPAR::SCATRA::solvertype_nonlinear) and (lastfluxoutputstep_ != step_))
    {
      lastfluxoutputstep_ = step_;

      // For nonlinear problems we already have the actual residual vector
      // from the last convergence test!
      // For linear problems we have to compute this information first, since
      // the residual (w.o. Neumann boundary) has not been computed after the last solve!

      // zero out matrix entries
      sysmat_->Zero();

      // zero out residual vector
      residual_->PutScalar(0.0);

      Teuchos::ParameterList eleparams;
      // action for elements
      eleparams.set<int>("action",SCATRA::calc_mat_and_rhs);

      // other parameters that might be needed by the elements
      // the resulting system has to be solved incrementally
      SetElementTimeParameter(true);

      // provide velocity field and potentially acceleration/pressure field
      // (export to column map necessary for parallel evaluation)
      eleparams.set<int>("ndsvel",nds_vel_);

      //provide displacement field in case of ALE
      if (isale_)
        eleparams.set<int>("ndsdisp",nds_disp_);

      // clear state
      discret_->ClearState();

      // AVM3 separation for incremental solver: get fine-scale part of scalar
      if (incremental_ and
          (fssgd_ != INPAR::SCATRA::fssugrdiff_no or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales))
       AVM3Separation();

      // add element parameters according to time-integration scheme
      // we want to have an incremental solver!!
      AddTimeIntegrationSpecificVectors(true);

      // add element parameters according to specific problem
      AddProblemSpecificParametersAndVectors(eleparams);

      {
        // call standard loop over elements
        discret_->Evaluate(eleparams,sysmat_,Teuchos::null,residual_,Teuchos::null,Teuchos::null);
        discret_->ClearState();
      }

      // scaling to get true residual vector for all time integration schemes
      trueresidual_->Update(ResidualScaling(),*residual_,0.0);

      // undo potential changes
      SetElementTimeParameter();

    } // if ((solvtype_!=INPAR::SCATRA::solvertype_nonlinear) && (lastfluxoutputstep_ != step_))
  } // if (writeflux_==INPAR::SCATRA::flux_convective_boundary)

  // if desired add the convective flux contribution
  // to the trueresidual_ now.
  if((writeflux_==INPAR::SCATRA::flux_total_boundary)
      or (writeflux_==INPAR::SCATRA::flux_convective_boundary))
  {
    // following screen output removed, for the time being
    /*if(myrank_ == 0)
    {
      std::cout << "Convective flux contribution is added to trueresidual_ vector." << std::endl;
      std::cout << "Be sure not to address the same boundary part twice!" << std::endl;
      std::cout << "Two flux calculation boundaries should also not share a common node!" << std::endl;
    }*/

    // now we evaluate the conditions and separate via ConditionID
    for (unsigned int i=0; i < condnames.size(); i++)
    {
      std::vector<DRT::Condition*> cond;
      discret_->GetCondition(condnames[i],cond);

      discret_->ClearState();
      Teuchos::ParameterList params;

      params.set<int>("action",SCATRA::bd_add_convective_mass_flux);

      // add element parameters according to time-integration scheme
      AddTimeIntegrationSpecificVectors();

      // provide velocity field
      // (export to column map necessary for parallel evaluation)
      params.set<int>("ndsvel",nds_vel_);

      //provide displacement field in case of ALE
      if (isale_)
        params.set<int>("ndsdisp",nds_disp_);

      // call loop over boundary elements and add integrated fluxes to trueresidual_
      discret_->EvaluateCondition(params,trueresidual_,condnames[i]);
      discret_->ClearState();
    }
  }

  // vector for effective flux over all defined boundary conditions
  // maximal number of fluxes  (numscal+1 -> ionic species + potential) is generated
  // for OUTPUT standard -> last entry is not used
  std::vector<double> normfluxsum(numdofpernode_);

  for (unsigned int i=0; i < condnames.size(); i++)
  {
    std::vector<DRT::Condition*> cond;
    discret_->GetCondition(condnames[i],cond);

    // safety check
    if(!cond.size())
      dserror("Flux output requested without corresponding boundary condition specification!");

    if (myrank_ == 0)
    {
      std::cout << "Normal fluxes at boundary '" << condnames[i] << "' on discretization '" << discret_->Name() << "':" << std::endl;
      std::cout <<"+----+-----+-------------------------+------------------+--------------------------+" << std::endl;
      std::cout << "| ID | DOF | Integral of normal flux | Area of boundary | Mean normal flux density |" << std::endl;
    }

    // first, add to all conditions of interest a ConditionID
    for (int condid = 0; condid < (int) cond.size(); condid++)
    {
      // is there already a ConditionID?
      const std::vector<int>*    CondIDVec  = cond[condid]->Get<std::vector<int> >("ConditionID");
      if (CondIDVec)
      {
        if ((*CondIDVec)[0] != condid)
          dserror("Condition %s has non-matching ConditionID",condnames[i].c_str());
      }
      else
      {
        // let's add a ConditionID
        cond[condid]->Add("ConditionID",condid);
      }
    }

    // now we evaluate the conditions and separate via ConditionID
    for (int condid = 0; condid < (int) cond.size(); condid++)
    {
      Teuchos::ParameterList params;

      // calculate integral of shape functions over indicated boundary and it's area
      params.set("area",0.0);
      params.set<int>("action",SCATRA::bd_integrate_shape_functions);

      //provide displacement field in case of ALE
      if (isale_)
        params.set<int>("ndsdisp",nds_disp_);

      // create vector (+ initialization with zeros)
      Teuchos::RCP<Epetra_Vector> integratedshapefunc = LINALG::CreateVector(*dofrowmap,true);

      // call loop over elements
      discret_->ClearState();
      discret_->EvaluateCondition(params,integratedshapefunc,condnames[i],condid);
      discret_->ClearState();

      // maximal number of fluxes
      std::vector<double> normfluxintegral(numdofpernode_);

      // insert values into final flux vector for visualization
      int numrownodes = discret_->NumMyRowNodes();
      for (int lnodid = 0; lnodid < numrownodes; ++lnodid )
      {
        DRT::Node* actnode = discret_->lRowNode(lnodid);
        for (int idof = 0; idof < numdofpernode_; ++idof)
        {
          int dofgid = discret_->Dof(0,actnode,idof);
          int doflid = dofrowmap->LID(dofgid);

          if ((*integratedshapefunc)[doflid] != 0.0)
          {
            // this is the value of the normal flux density
            double normflux = ((*trueresidual_)[doflid])/(*integratedshapefunc)[doflid];
            // compute integral value for every degree of freedom
            normfluxintegral[idof] += (*trueresidual_)[doflid];

            // care for the slave nodes of rotationally symm. periodic boundary conditions
            double rotangle(0.0);
            bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(actnode,rotangle);

            // do not insert slave node values here, since they would overwrite the
            // master node values owning the same dof
            // (rotation of slave node vectors is performed later during output)
            if (not havetorotate)
            {
              // for visualization, we plot the normal flux with
              // outward pointing normal vector
              for (int idim = 0; idim < 3; idim++)
              {
                Epetra_Vector* normalcomp = (*normals_)(idim);
                double normalveccomp =(*normalcomp)[lnodid];
                int err=flux->ReplaceMyValue(doflid,idim,normflux*normalveccomp);
                if (err!=0) dserror("Detected error in ReplaceMyValue");
              }
            }
          }
        }
      }

      // get area of the boundary on this proc
      double boundaryint = params.get<double>("area");

      // care for the parallel case
      std::vector<double> parnormfluxintegral(numdofpernode_);
      discret_->Comm().SumAll(&normfluxintegral[0],&parnormfluxintegral[0],numdofpernode_);
      double parboundaryint = 0.0;
      discret_->Comm().SumAll(&boundaryint,&parboundaryint,1);

      for (int idof = 0; idof < numdofpernode_; ++idof)
      {
        // print out results
        if (myrank_ == 0)
        {
          printf("| %2d | %2d  |       %+10.4E       |    %10.4E    |        %+10.4E       |\n",
              condid,idof,parnormfluxintegral[idof],parboundaryint,parnormfluxintegral[idof]/parboundaryint);
        }
        normfluxsum[idof]+=parnormfluxintegral[idof];
      }

      // statistics section for normfluxintegral
      if (DoBoundaryFluxStatistics())
      {
        // add current flux value to the sum!
        (*sumnormfluxintegral_)[condid] += parnormfluxintegral[0]; // only first scalar!
        int samstep = step_-samstart_+1;

        // dump every dumperiod steps (i.e., write to screen)
        bool dumpit(false);
        if (dumperiod_==0)
          {dumpit=true;}
        else
          {if(samstep%dumperiod_==0) dumpit=true;}

        if(dumpit)
        {
          double meannormfluxintegral = (*sumnormfluxintegral_)[condid]/samstep;
          // dump statistical results
          if (myrank_ == 0)
          {
            printf("| %2d | Mean normal-flux integral (step %5d -- step %5d) :   %12.5E |\n", condid,samstart_,step_,meannormfluxintegral);
          }
        }
      }

      // print out results to file as well (only if really desired)
      if ((myrank_ == 0) and (writetofile==true))
      {
        std::ostringstream temp;
        temp << condid;
        temp << num;
        const std::string fname
        = DRT::Problem::Instance()->OutputControlFile()->FileName()+".boundaryflux_"+condnames[i]+"_"+temp.str()+".txt";

        std::ofstream f;
        if (Step() <= 1)
        {
          f.open(fname.c_str(),std::fstream::trunc);
          f << "#| ID | Step | Time | Area of boundary |";
          for(int idof = 0; idof < numdofpernode_; ++idof)
          {
            f<<" Integral of normal flux "<<idof<<" | Mean normal flux density "<<idof<<" |";
          }
          f<<"\n";
        }
        else
          f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

        f << condid << " " << Step() << " " << Time() << " "<< parboundaryint<< " ";
        for (int idof = 0; idof < numdofpernode_; ++idof)
        {
          f << parnormfluxintegral[idof] << " "<< parnormfluxintegral[idof]/parboundaryint<< " ";
        }
        f << "\n";
        f.flush();
        f.close();
      } // write to file

    } // loop over condid

    if (myrank_==0)
      std::cout << "+----+-----+-------------------------+------------------+--------------------------+" << std::endl;
  }

  // print out the accumulated normal flux over all indicated boundaries
  // the accumulated normal flux over all indicated boundaries is not printed for numscal+1
  if (myrank_ == 0)
  {
    for (int idof = 0; idof < numscal_; ++idof)
    {
      printf("| Sum of all normal flux boundary integrals for scalar %d: %+10.5E             |\n"
          ,idof,normfluxsum[idof]);
    }
    std::cout << "+----------------------------------------------------------------------------------+" << std::endl << std::endl;
  }
  // clean up
  discret_->ClearState();

  return flux;
} // SCATRA::ScaTraTimIntImpl::CalcFluxAtBoundary


/*--------------------------------------------------------------------*
 | calculate initial time derivatives of state variables   fang 03/15 |
 *--------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcInitialTimeDerivative()
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + calculate initial time derivative");

  // initial screen output
  if(myrank_ == 0)
    std::cout << "SCATRA: calculating initial time derivative of state variables on discretization \"" << discret_->Name().c_str() << "\" (step " << Step() <<", time " << Time() << ") ... ... ";

  // evaluate Dirichlet and Neumann boundary conditions at time t = 0 to ensure consistent computation of initial time derivative vector
  // Dirichlet boundary conditions should be consistent with initial field
  ApplyDirichletBC(time_,phinp_,Teuchos::null);
  ComputeIntermediateValues();
  ApplyNeumannBC(neumann_loads_);

  // In most cases, the history vector is entirely zero at this point, since this routine is called before the first time step.
  // However, in levelset simulations with reinitialization, this routine might be called before every single time step.
  // For this reason, the history vector needs to be manually set to zero here and restored at the end of this routine.
  Teuchos::RCP<Epetra_Vector> hist = hist_;
  hist_ = zeros_;

  // In a first step, we assemble the standard global system of equations.
  AssembleMatAndRHS();

  // In a second step, we need to modify the assembled system of equations, since we want to solve
  // M phidt^0 = f^n - K\phi^n - C(u_n)\phi^n
  // In particular, we need to replace the global system matrix by a global mass matrix,
  // and we need to remove all transient contributions associated with time discretization from the global residual vector.

  // reset global system matrix
  sysmat_->Zero();

  // create and fill parameter list for elements
  Teuchos::ParameterList eleparams;
  eleparams.set<int>("action",SCATRA::calc_initial_time_deriv);
  eleparams.set<int>("ndsvel",nds_vel_);
  if(isale_)
    eleparams.set<int>("ndsdisp",nds_disp_);
  AddProblemSpecificParametersAndVectors(eleparams);

  // add state vectors according to time integration scheme
  discret_->ClearState();
  AddTimeIntegrationSpecificVectors();

  // modify global system of equations as explained above
  discret_->Evaluate(eleparams,sysmat_,residual_);
  discret_->ClearState();

  // apply condensation to global system of equations if necessary
  strategy_->CondenseMatAndRHS(sysmat_,residual_,true);

  // finalize assembly of global mass matrix
  sysmat_->Complete();

  // solve global system of equations for initial time derivative of state variables
  solver_->Solve(sysmat_->EpetraOperator(),phidtnp_,residual_,true,true);

  // copy solution
  phidtn_->Update(1.0,*phidtnp_,0.0);

  // reset global system matrix and its graph, since we solved a very special problem with a special sparsity pattern
  sysmat_->Reset();

  // reset solver
  solver_->Reset();

  // reset true residual vector computed during assembly of the standard global system of equations, since not yet needed
  trueresidual_->Scale(0.);

  // restore history vector as explained above
  hist_ = hist;

  // final screen output
  if(myrank_ == 0)
    std::cout << "done!" << std::endl;

  return;
} // SCATRA::ScaTraTimIntImpl::CalcInitialTimeDerivative


/*---------------------------------------------------------------------------*
 | compute nodal density values from nodal concentration values   fang 12/14 |
 *---------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeDensity()
{
  // loop over all nodes owned by current processor
  for(int lnodeid=0; lnodeid<discret_->NumMyRowNodes(); ++lnodeid)
  {
    // get current node
    DRT::Node* lnode = discret_->lRowNode(lnodeid);

    // get associated degrees of freedom
    std::vector<int> nodedofs = discret_->Dof(0,lnode);
    const int numdof = nodedofs.size();

    // initialize nodal density value
    double density = 1.;

    // loop over all transported scalars
    for(int k=0; k<numscal_; ++k)
    {
      /*
        //                  k=numscal_-1
        //          /       ----                         \
        //         |        \                            |
        // rho_0 * | 1 +    /      alpha_k * (c_k - c_0) |
        //         |        ----                         |
        //          \       k=0                          /
        //
        // For use of molar mass M_k:  alpha_k = M_k/rho_0
       */

      // global and local dof ID
      const int globaldofid = nodedofs[k];
      const int localdofid = Phiafnp()->Map().LID(globaldofid);
      if (localdofid < 0)
        dserror("Local dof ID not found in dof map!");

      // add contribution of scalar k to nodal density value
      density += densific_[k] * ((*(Phiafnp()))[localdofid] - c0_[k]);
    }

    // insert nodal density value into global density vector
    // note that the density vector has been initialized with the dofrowmap of the discretization
    // in case there is more than one dof per node, the nodal density value is inserted into the position of the last dof
    // this way, all nodal density values will be correctly extracted in the fluid algorithm
    const int globaldofid = nodedofs[numdof-1];
    const int localdofid = Phiafnp()->Map().LID(globaldofid);
    if (localdofid < 0)
      dserror("Local dof ID not found in dof map!");

    int err = densafnp_->ReplaceMyValue(localdofid,0,density);

    if(err)
      dserror("Error while inserting nodal density value into global density vector!");
  } // loop over all local nodes
  return;
} // SCATRA::ScaTraTimIntImpl::ComputeDensity


/*----------------------------------------------------------------------*
 | output mean values of scalar(s)                             vg 05/15 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputMeanScalars(const int num)
{
  if(outmean_)
  {
    // set scalar values needed by elements
    discret_->ClearState();
    discret_->SetState("phinp",phinp_);
    // set action for elements
    Teuchos::ParameterList eleparams;
    eleparams.set<int>("action",SCATRA::calc_mean_scalars);
    eleparams.set("inverting",false);

    // provide displacement field in case of ALE
    if (isale_)
      eleparams.set<int>("ndsdisp",nds_disp_);

    eleparams.set<int>("ndsvel",nds_vel_);

    // evaluate integrals of scalar(s) and domain
    Teuchos::RCP<Epetra_SerialDenseVector> scalars
    = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
    discret_->EvaluateScalars(eleparams, scalars);
    discret_->ClearState();   // clean up

    // extract domain integral
    const double domint = (*scalars)[numscal_];

    // print out results to screen and file
    if (myrank_ == 0)
    {
      // screen output
      std::cout << "Mean scalar values:" << std::endl;
      std::cout << "+-------------------------------+" << std::endl;
      for(int k = 0; k < numscal_; k++)
        std::cout << "| Mean scalar " << k+1 << ":   " << std::scientific << std::setprecision(6) << (*scalars)[k]/domint << " |" << std::endl;
      std::cout << "+-------------------------------+" << std::endl << std::endl;

      // file output
      std::stringstream number;
      number << num;
      const std::string fname = DRT::Problem::Instance()->OutputControlFile()->FileName()+number.str()+".meanvalues.txt";

      std::ofstream f;
      if (Step() <= 1)
      {
        f.open(fname.c_str(),std::fstream::trunc);
        f << "#| Step | Time | Domain integral |";
        for (int k = 0; k < numscal_; k++)
        {
          f << " Total scalar " << k+1 << " |";
          f << " Mean scalar " << k+1 << " |";
        }
        f << "\n";
      }
      else f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      f << Step() << " " << Time() << " " << std::setprecision (9) << domint;
      for (int k = 0; k < numscal_; k++)
      {
        f << " " << std::setprecision (9) << (*scalars)[k];
        f << " " << std::setprecision (9) << (*scalars)[k]/domint;
      }
      f << "\n";
      f.flush();
      f.close();
    }

  } // if(outmean_)

  return;
} // SCATRA::ScaTraTimIntImpl::OutputMeanScalars


/*--------------------------------------------------------------------------------------------------------*
 | output domain or boundary integrals, i.e., surface areas or volumes of specified nodesets   fang 06/15 |
 *--------------------------------------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputDomainOrBoundaryIntegrals(const std::string condstring)
{
  // check whether output is applicable
  if((computeintegrals_ == INPAR::SCATRA::computeintegrals_initial and Step() == 0) or computeintegrals_ == INPAR::SCATRA::computeintegrals_repeated)
  {
    // initialize label
    std::string label;

    // create parameter list
    Teuchos::ParameterList condparams;

    // determine label and element action depending on whether domain or boundary integrals are relevant
    if(condstring == "BoundaryIntegral")
    {
      label = "Boundary";
      condparams.set<int>("action",SCATRA::bd_calc_boundary_integral);
    }
    else if(condstring == "DomainIntegral")
    {
      label = "Domain";
      condparams.set<int>("action",SCATRA::calc_domain_integral);
    }
    else
      dserror("Invalid condition name!");

    // extract conditions for computation of domain or boundary integrals
    std::vector<DRT::Condition*> conditions;
    discret_->GetCondition(condstring,conditions);

    // print header
    if(conditions.size() > 0 and myrank_ == 0)
    {
      std::cout << label+" integrals:" << std::endl;
      std::cout << "+----+-------------------------+" << std::endl;
      std::cout << "| ID | value of integral       |" << std::endl;
    }

    // loop over all conditions
    for(unsigned condid=0; condid<conditions.size(); ++condid)
    {
      // clear state vectors on discretization
      discret_->ClearState();

      // initialize one-component result vector for value of current domain or boundary integral
      Teuchos::RCP<Epetra_SerialDenseVector> integralvalue = Teuchos::rcp(new Epetra_SerialDenseVector(1));

      // compute value of current domain or boundary integral
      discret_->EvaluateScalars(condparams,integralvalue,condstring,condid);
      discret_->ClearState();

      // output results to screen and file
      if(myrank_ == 0)
      {
        // print results to screen
         std::cout << "| " << std::setw(2) << condid << " |         " << std::setw(6) << std::setprecision(3) << std::fixed << (*integralvalue)(0) << "          |" << std::endl;

         // set file name
         const std::string filename(DRT::Problem::Instance()->OutputControlFile()->FileName()+"."+label+"_integrals.txt");

         // open file in appropriate mode and write header at beginning
         std::ofstream file;
         if(Step() == 0 and condid == 0)
         {
           file.open(filename.c_str(),std::fstream::trunc);
           file << "Step,Time,"+label+"ID,"+label+"Integral" << std::endl;
         }
         else
           file.open(filename.c_str(),std::fstream::app);

         // write value of current domain or boundary integral to file
         file << Step() << "," << Time() << "," << condid << ',' << std::setprecision(16) << std::fixed << (*integralvalue)(0) << std::endl;

         // close file
         file.close();
      } // if(myrank_ == 0)
    } // loop over all conditions

    // print finish line to screen
    if(myrank_ == 0)
      std::cout << "+----+-------------------------+" << std::endl << std::endl;
  } // check whether output is applicable

  return;
} // SCATRA::ScaTraTimIntImpl::OutputDomainOrBoundaryIntegrals


/*----------------------------------------------------------------------*
 | Evaluate surface/interface permeability for FS3I          Thon 11/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SurfacePermeability(
    Teuchos::RCP<LINALG::SparseOperator> matrix,
    Teuchos::RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'SurfacePermeability'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ScaTraCoupling'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_fs3i_surface_permeability);

  // provide displacement field in case of ALE
  if (isale_)
    condparams.set<int>("ndsdisp",nds_disp_);

  condparams.set<int>("ndswss",nds_wss_);

  // set vector values needed by elements
  discret_->ClearState();

  // add element parameters according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  // replace phinp by the mean of phinp if this has been set
  if ( meanconc_!=Teuchos::null )
  {
    // TODO: (thon) this is not a nice way of using the mean concentration instead of phinp!
    // Note: meanconc_ is not cleared by calling 'discret_->ClearState()', hence if you
    // don't want to do this replacement here any more call 'ClearMeanConcentration()'
    discret_->SetState("phinp",meanconc_);

    if (myrank_ == 0)
      std::cout<<"Replacing 'phinp' by 'meanconc_' in the evaluation of the surface permeability" <<std::endl;
  }

  // test if all necessary ingredients had been set
  Teuchos::RCP<const Epetra_Vector> wss = discret_->GetState(nds_wss_,"WallShearStress");
  if ( wss == Teuchos::null )
    dserror("WSS must already been set into one of the secondary dofset before calling this function!");

  //Evaluate condition
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,"ScaTraCoupling");
  discret_->ClearState();

  matrix->Complete();

  return;
} // SCATRA::ScaTraTimIntImpl::SurfacePermeability

/*----------------------------------------------------------------------------*
 |  Kedem & Katchalsky second membrane equation for FPS3i       hemmler 07/14 |
 |  see e.g. Kedem, O. T., and A. Katchalsky. "Thermodynamic analysis of the permeability of biological membranes to non-electrolytes." Biochimica et biophysica Acta 27 (1958): 229-246.
 *----------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::KedemKatchalsky(
    Teuchos::RCP<LINALG::SparseOperator> matrix,
    Teuchos::RCP<Epetra_Vector> rhs
    )
{
  // time measurement: evaluate condition 'SurfacePermeability'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ScaTraCoupling'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_fps3i_surface_permeability);

  // provide displacement field in case of ALE
  if (isale_)
    condparams.set<int>("ndsdisp",nds_disp_);

  condparams.set<int>("ndswss",nds_wss_);
  condparams.set<int>("ndspres",nds_pres_);

  // set vector values needed by elements
  discret_->ClearState();

  // add element parameters according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  // test if all necessary ingredients for the second Kedem-Katchalsky equations had been set
  Teuchos::RCP<const Epetra_Vector> wss = discret_->GetState(nds_wss_,"WallShearStress");
  if ( wss == Teuchos::null )
    dserror("WSS must already been set into one of the secondary dofset before calling this function!");

  Teuchos::RCP<const Epetra_Vector> pressure = discret_->GetState(nds_pres_,"Pressure");
  if ( pressure == Teuchos::null )
    dserror("Pressure must already been set into one of the secondary dofset before calling this function!");

  if ( meanconc_==Teuchos::null )
    dserror("Mean concentration must already been saved before calling this function!");
  discret_->SetState("MeanConcentration",meanconc_);

  //Evaluate condition
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,"ScaTraCoupling");
  discret_->ClearState();

  matrix->Complete();

  return;
} // SCATRA::ScaTraTimIntImpl::KedemKatchalsky


/*==========================================================================*
 |                                                                          |
 | protected:                                                               |
 |                                                                          |
 *==========================================================================*/

/*==========================================================================*/
// general framework
/*==========================================================================*/

/*----------------------------------------------------------------------*
 | add approximation to flux vectors to a parameter list      gjb 05/10 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AddFluxApproxToParameterList(
    Teuchos::ParameterList& p,
    const enum INPAR::SCATRA::FluxType fluxtype
)
{
  Teuchos::RCP<Epetra_MultiVector> flux
    = CalcFluxInDomain(fluxtype);

  // post_drt_ensight does not support multivectors based on the dofmap
  // for now, I create single vectors that can be handled by the filters

  // get the noderowmap
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  Teuchos::RCP<Epetra_MultiVector> fluxk = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,3,true));
  for(int k=0;k<numscal_;++k)
  {
    std::ostringstream temp;
    temp << k;
    std::string name = "flux_phi_"+temp.str();
    for (int i = 0;i<fluxk->MyLength();++i)
    {
      DRT::Node* actnode = discret_->lRowNode(i);
      int dofgid = discret_->Dof(0,actnode,k);
      fluxk->ReplaceMyValue(i,0,((*flux)[0])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,1,((*flux)[1])[(flux->Map()).LID(dofgid)]);
      fluxk->ReplaceMyValue(i,2,((*flux)[2])[(flux->Map()).LID(dofgid)]);
    }
    discret_->AddMultiVectorToParameterList(p,name,fluxk);
  }
  return;
} // SCATRA::ScaTraTimIntImpl::AddFluxApproxToParameterList


/*----------------------------------------------------------------------*
 | compute outward pointing unit normal vectors at given b.c.  gjb 01/09|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::ComputeNormalVectors(
    const std::vector<std::string>& condnames
)
{
  // create vectors for x,y and z component of average normal vector field
  // get noderowmap of discretization
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  Teuchos::RCP<Epetra_MultiVector> normal = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,3,true));

  discret_->ClearState();

  // set action for elements
  Teuchos::ParameterList eleparams;
  eleparams.set<int>("action",SCATRA::bd_calc_normal_vectors);
  eleparams.set<Teuchos::RCP<Epetra_MultiVector> >("normal vectors",normal);

  //provide displacement field in case of ALE
  if (isale_)
    eleparams.set<int>("ndsdisp",nds_disp_);

  // loop over all intended types of conditions
  for (unsigned int i=0; i < condnames.size(); i++)
  {
    discret_->EvaluateCondition(eleparams,condnames[i]);
  }

  // clean up
  discret_->ClearState();

  // the normal vector field is not properly scaled up to now. We do this here
  int numrownodes = discret_->NumMyRowNodes();
  Epetra_Vector* xcomp = (*normal)(0);
  Epetra_Vector* ycomp = (*normal)(1);
  Epetra_Vector* zcomp = (*normal)(2);
  for (int i=0; i<numrownodes; ++i)
  {
    double x = (*xcomp)[i];
    double y = (*ycomp)[i];
    double z = (*zcomp)[i];
    double norm = sqrt(x*x + y*y + z*z);
    // form the unit normal vector
    if (norm > EPS15)
    {
      normal->ReplaceMyValue(i,0,x/norm);
      normal->ReplaceMyValue(i,1,y/norm);
      normal->ReplaceMyValue(i,2,z/norm);
    }
  }

  return normal;
} // SCATRA:: ScaTraTimIntImpl::ComputeNormalVectors

/*----------------------------------------------------------------------*
 | evaluate Neumann inflow boundary condition                  vg 03/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeNeumannInflow(
    Teuchos::RCP<LINALG::SparseOperator> matrix,
    Teuchos::RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'Neumann inflow'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'TransportNeumannInflow'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_Neumann_inflow);

  // provide velocity field and potentially acceleration/pressure field
  // as well as displacement field in case of ALE
  condparams.set<int>("ndsvel",nds_vel_);
  if (isale_)
    condparams.set<int>("ndsdisp",nds_disp_);

  // clear state
  discret_->ClearState();

  // add element parameters and vectors according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  std::string condstring("TransportNeumannInflow");
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  return;
} // SCATRA::ScaTraTimIntImpl::ComputeNeumannInflow

/*----------------------------------------------------------------------*
 | evaluate boundary cond. due to convective heat transfer     vg 10/11 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateConvectiveHeatTransfer(
    Teuchos::RCP<LINALG::SparseOperator> matrix,
    Teuchos::RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'ThermoConvections'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ThermoConvections'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_convective_heat_transfer);

  // clear state
  discret_->ClearState();

  // add element parameters and vectors according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  std::string condstring("ThermoConvections");
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  return;
} // SCATRA::ScaTraTimIntImpl::EvaluateConvectiveHeatTransfer

/*----------------------------------------------------------------------*
 | write state vectors to Gmsh postprocessing files        henke   12/09|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputToGmsh(
    const int step,
    const double time
    ) const
{
  // turn on/off screen output for writing process of Gmsh postprocessing file
  const bool screen_out = true;

  // create Gmsh postprocessing file
  const std::string filename = IO::GMSH::GetNewFileNameAndDeleteOldFiles("solution_field_scalar", step, 500, screen_out, discret_->Comm().MyPID());
  std::ofstream gmshfilecontent(filename.c_str());
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phin \" {" << std::endl;
//    // draw scalar field 'Phindtp' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phin_,gmshfilecontent);
//    gmshfilecontent << "};" << std::endl;
//  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "Phinp \" {" << std::endl;
    // draw scalar field 'Phinp' for every element
    IO::GMSH::ScalarFieldToGmsh(discret_,phinp_,gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phidtn \" {" << std::endl;
//    // draw scalar field 'Phindtn' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phidtn_,gmshfilecontent);
//    gmshfilecontent << "};" << std::endl;
//  }
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phidtnp \" {" << std::endl;
//    // draw scalar field 'Phindtp' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phidtnp_,gmshfilecontent);
//    gmshfilecontent << "};" << std::endl;
//  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "Convective Velocity \" {" << std::endl;

    // extract convective velocity from discretization
    Teuchos::RCP<const Epetra_Vector> convel = discret_->GetState(nds_vel_,"convective velocity field");
    if (convel == Teuchos::null)
      dserror("Cannot extract convective velocity field from discretization");

    // draw vector field 'Convective Velocity' for every element
    IO::GMSH::VectorFieldDofBasedToGmsh(discret_,convel,gmshfilecontent,nds_vel_);
    gmshfilecontent << "};" << std::endl;
  }
  gmshfilecontent.close();
  if (screen_out) std::cout << " done" << std::endl;
} // ScaTraTimIntImpl::OutputToGmsh

/*----------------------------------------------------------------------*
 |  write mass / heat flux vector to BINIO                   gjb   08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputFlux(Teuchos::RCP<Epetra_MultiVector> flux)
{
  //safety check
  if (flux == Teuchos::null)
    dserror("Null pointer for flux vector output. Output() called before Update() ??");

  // WORK-AROUND FOR NURBS DISCRETIZATIONS
  // using noderowmap is problematic. Thus, we do not add normal vectors
  // to the scalar result field (scalar information is enough anyway)
  DRT::NURBS::NurbsDiscretization* nurbsdis
  = dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*discret_));
  if(nurbsdis!=NULL)
  {
    Teuchos::RCP<Epetra_Vector> normalflux = Teuchos::rcp(((*flux)(0)),false);
    output_->WriteVector("normalflux", normalflux, IO::DiscretizationWriter::dofvector);
    return; // leave here
  }

  // post_drt_ensight does not support multivectors based on the dofmap
  // for now, I create single vectors that can be handled by the filter

  // get the noderowmap
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  Teuchos::RCP<Epetra_MultiVector> fluxk = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,3,true));
  for (std::vector<int>::iterator it = writefluxids_->begin(); it!=writefluxids_->end(); ++it)
  {
    int k=(*it);

    std::ostringstream temp;
    temp << k;
    std::string name = "flux_phi_"+temp.str();
    for (int i = 0;i<fluxk->MyLength();++i)
    {
      DRT::Node* actnode = discret_->lRowNode(i);
      int dofgid = discret_->Dof(0,actnode,k-1);
      // get value for each component of flux vector
      double xvalue = ((*flux)[0])[(flux->Map()).LID(dofgid)];
      double yvalue = ((*flux)[1])[(flux->Map()).LID(dofgid)];
      double zvalue = ((*flux)[2])[(flux->Map()).LID(dofgid)];
      // care for the slave nodes of rotationally symm. periodic boundary conditions
      double rotangle(0.0); //already converted to radians
      bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(actnode,rotangle);
      if (havetorotate)
      {
        double xvalue_rot = (xvalue*cos(rotangle)) - (yvalue*sin(rotangle));
        double yvalue_rot = (xvalue*sin(rotangle)) + (yvalue*(cos(rotangle)));
        xvalue = xvalue_rot;
        yvalue = yvalue_rot;
      }
      // insert values
      int err = fluxk->ReplaceMyValue(i,0,xvalue);
      err += fluxk->ReplaceMyValue(i,1,yvalue);
      err += fluxk->ReplaceMyValue(i,2,zvalue);
      if (err!=0) dserror("Detected error in ReplaceMyValue");
    }
    output_->WriteVector(name, fluxk, IO::DiscretizationWriter::nodevector);
  }
  // that's it
  return;
} // ScaTraTimIntImpl::OutputFlux


//TODO: SCATRA_ELE_CLEANING: BIOFILM
// This action is not called at element level!!
// Can it be integrated in CalcFlux_domain?
/*----------------------------------------------------------------------*
 |  output of integral reaction                               mc   03/13|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputIntegrReac(const int num)
{
  if (outintegrreac_)
  {
    if (!discret_->Filled()) dserror("FillComplete() was not called");
    if (!discret_->HaveDofs()) dserror("AssignDegreesOfFreedom() was not called");

    // set scalar values needed by elements
    discret_->ClearState();
    discret_->SetState("phinp",phinp_);
    // set action for elements
    Teuchos::ParameterList eleparams;
    eleparams.set<int>("action",SCATRA::calc_integr_reaction);
    Teuchos::RCP<std::vector<double> > myreacnp = Teuchos::rcp(new std::vector<double>(numscal_,0.0));
    eleparams.set<Teuchos::RCP<std::vector<double> > >("local reaction integral",myreacnp);

    discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

    myreacnp = eleparams.get<Teuchos::RCP<std::vector<double> > >("local reaction integral");
    // global integral of reaction terms
    std::vector<double> intreacterm(numscal_,0.0);
    for (int k=0; k<numscal_; ++k)
      phinp_->Map().Comm().SumAll(&((*myreacnp)[k]),&intreacterm[k],1);
    discret_->ClearState();   // clean up

    // print out values
    if (myrank_ == 0)
    {
      for (int k = 0; k < numscal_; k++)
      {
        std::cout << "Total reaction (r_"<<k<<"): "<< std::setprecision (9) << intreacterm[k] << std::endl;
      }
    }

    // print out results to file as well
    if (myrank_ == 0)
    {
      std::stringstream number;
      number << num;
      const std::string fname
      = DRT::Problem::Instance()->OutputControlFile()->FileName()+number.str()+".integrreacvalues.txt";

      std::ofstream f;
      if (Step() <= 1)
      {
        f.open(fname.c_str(),std::fstream::trunc);
        f << "#| Step | Time ";
        for (int k = 0; k < numscal_; k++)
        {
          f << "| Total reaction (r_"<<k<<") ";
        }
        f << "\n";

      }
      else
        f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      f << Step() << " " << Time() << " ";
      for (int k = 0; k < numscal_; k++)
      {
        f << std::setprecision (9) << intreacterm[k] << " ";
      }
      f << "\n";
      f.flush();
      f.close();
    }

  } // if(outintegrreac_)

  return;
} // SCATRA::ScaTraTimIntImpl::OutputIntegrReac


/*==========================================================================*/
// ELCH
/*==========================================================================*/



/*----------------------------------------------------------------------*
 |  prepare AVM3-based scale separation                        vg 10/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AVM3Preparation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // create normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->Zero();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements, time factor and stationary flag
  eleparams.set<int>("action",SCATRA::calc_subgrid_diffusivity_matrix);

  //provide displacement field in case of ALE
  if (isale_)
    eleparams.set<int>("ndsdisp",nds_disp_);

  // add element parameters according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  // call loop over elements
  discret_->Evaluate(eleparams,sysmat_sd_,residual_);
  discret_->ClearState();

  // finalize the normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->Complete();

  // apply DBC to normalized all-scale subgrid-diffusivity matrix
  LINALG::ApplyDirichlettoSystem(sysmat_sd_,phinp_,residual_,phinp_,*(dbcmaps_->CondMap()));

  // get normalized fine-scale subgrid-diffusivity matrix
  {
    // this is important to have!!!
    // MLAPI::Init() without arguments uses internally MPI_COMM_WOLRD
    MLAPI::Init();

    // extract the ML parameters
    Teuchos::ParameterList&  mlparams = solver_->Params().sublist("ML Parameters");
    // remark: we create a new solver with ML preconditioner here, since this allows for also using other solver setups
    // to solve the system of equations
    // get the solver number used form the multifractal subgrid-scale model parameter list
    const int scale_sep_solvernumber = extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES").get<int>("ML_SOLVER");
    if (scale_sep_solvernumber != (-1))     // create a dummy solver
    {
      Teuchos::RCP<LINALG::Solver> solver = Teuchos::rcp(new LINALG::Solver(DRT::Problem::Instance()->SolverParams(scale_sep_solvernumber),
                                            discret_->Comm(),
                                            DRT::Problem::Instance()->ErrorFile()->Handle()));
      // compute the null space,
      discret_->ComputeNullSpaceIfNecessary(solver->Params(),true);
      // and, finally, extract the ML parameters
      mlparams = solver->Params().sublist("ML Parameters");
    }

    // get toggle vector for Dirchlet boundary conditions
    const Teuchos::RCP<const Epetra_Vector> dbct = DirichletToggle();

    // get nullspace parameters
    double* nullspace = mlparams.get("null space: vectors",(double*)NULL);
    if (!nullspace) dserror("No nullspace supplied in parameter list");
    int nsdim = mlparams.get("null space: dimension",1);

    // modify nullspace to ensure that DBC are fully taken into account
    if (nullspace)
    {
      const int length = sysmat_sd_->OperatorRangeMap().NumMyElements();
      for (int i=0; i<nsdim; ++i)
        for (int j=0; j<length; ++j)
          if ((*dbct)[j]!=0.0) nullspace[i*length+j] = 0.0;
    }

    // get plain aggregation Ptent
    Teuchos::RCP<Epetra_CrsMatrix> crsPtent;
    MLAPI::GetPtent(*sysmat_sd_->EpetraMatrix(),mlparams,nullspace,crsPtent);
    LINALG::SparseMatrix Ptent(crsPtent,LINALG::View);

    // compute scale-separation matrix: S = I - Ptent*Ptent^T
    Sep_ = LINALG::Multiply(Ptent,false,Ptent,true);
    Sep_->Scale(-1.0);
    Teuchos::RCP<Epetra_Vector> tmp = LINALG::CreateVector(Sep_->RowMap(),false);
    tmp->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> diag = LINALG::CreateVector(Sep_->RowMap(),false);
    Sep_->ExtractDiagonalCopy(*diag);
    diag->Update(1.0,*tmp,1.0);
    Sep_->ReplaceDiagonalValues(*diag);

    //complete scale-separation matrix and check maps
    Sep_->Complete(Sep_->DomainMap(),Sep_->RangeMap());
    if (!Sep_->RowMap().SameAs(sysmat_sd_->RowMap())) dserror("rowmap not equal");
    if (!Sep_->RangeMap().SameAs(sysmat_sd_->RangeMap())) dserror("rangemap not equal");
    if (!Sep_->DomainMap().SameAs(sysmat_sd_->DomainMap())) dserror("domainmap not equal");

    // precomputation of unscaled diffusivity matrix:
    // either two-sided S^T*M*S: multiply M by S from left- and right-hand side
    // or one-sided M*S: only multiply M by S from left-hand side
    if (not incremental_)
    {
      Mnsv_ = LINALG::Multiply(*sysmat_sd_,false,*Sep_,false);
      //Mnsv_ = LINALG::Multiply(*Sep_,true,*Mnsv_,false);
    }
  }

  return;
} // ScaTraTimIntImpl::AVM3Preparation

/*----------------------------------------------------------------------*
 |  scaling of AVM3-based subgrid-diffusivity matrix           vg 10/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AVM3Scaling(Teuchos::ParameterList& eleparams)
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // some necessary definitions
  int ierr;
  double* sgvsqrt = 0;
  int length = subgrdiff_->MyLength();

  // square-root of subgrid-viscosity-scaling vector for left and right scaling
  sgvsqrt = (double*)subgrdiff_->Values();
  for (int i = 0; i < length; ++i)
  {
    sgvsqrt[i] = sqrt(sgvsqrt[i]);
    int err = subgrdiff_->ReplaceMyValues(1,&sgvsqrt[i],&i);
    if (err != 0) dserror("index not found");
  }

  // get unscaled S^T*M*S from Sep
  sysmat_sd_ = Teuchos::rcp(new LINALG::SparseMatrix(*Mnsv_));

  // left and right scaling of normalized fine-scale subgrid-viscosity matrix
  ierr = sysmat_sd_->LeftScale(*subgrdiff_);
  if (ierr) dserror("Epetra_CrsMatrix::LeftScale returned err=%d",ierr);
  ierr = sysmat_sd_->RightScale(*subgrdiff_);
  if (ierr) dserror("Epetra_CrsMatrix::RightScale returned err=%d",ierr);

  // add the subgrid-viscosity-scaled fine-scale matrix to obtain complete matrix
  Teuchos::RCP<LINALG::SparseMatrix> sysmat = SystemMatrix();
  sysmat->Add(*sysmat_sd_,false,1.0,1.0);

  // set subgrid-diffusivity vector to zero after scaling procedure
  subgrdiff_->PutScalar(0.0);

  return;
} // ScaTraTimIntImpl::AVM3Scaling

/*----------------------------------------------------------------------*
 | construct toggle vector for Dirichlet dofs                  gjb 11/08|
 | assures backward compatibility for avm3 solver; should go away once  |
 *----------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Vector> SCATRA::ScaTraTimIntImpl::DirichletToggle()
{
  if (dbcmaps_ == Teuchos::null)
    dserror("Dirichlet map has not been allocated");
  Teuchos::RCP<Epetra_Vector> dirichones = LINALG::CreateVector(*(dbcmaps_->CondMap()),false);
  dirichones->PutScalar(1.0);
  Teuchos::RCP<Epetra_Vector> dirichtoggle = LINALG::CreateVector(*(discret_->DofRowMap()),true);
  dbcmaps_->InsertCondVector(dirichones, dirichtoggle);
  return dirichtoggle;
} // ScaTraTimIntImpl::DirichletToggle


/*========================================================================*/
// turbulence and related
/*========================================================================*/


/*----------------------------------------------------------------------*
 | provide access to the dynamic Smagorinsky filter     rasthofer 08/12 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AccessDynSmagFilter(
  Teuchos::RCP<FLD::DynSmagFilter> dynSmag
)
{
  DynSmag_ = Teuchos::rcp(dynSmag.get(), false);

  // access to the dynamic Smagorinsky class is provided
  // by the fluid scatra coupling algorithm
  // therefore, we only have to add some scatra specific parameters here
  if (turbmodel_ == INPAR::FLUID::dynamic_smagorinsky)
  {
    if (myrank_ == 0)
    {
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << "Dynamic Smagorinsky model: provided access for ScaTra       " << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
    DynSmag_->AddScatra(discret_);
  }

  return;
}

/*----------------------------------------------------------------------*
 | provide access to the dynamic Vreman class               krank 09/13 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AccessVreman(
  Teuchos::RCP<FLD::Vreman> vrem
)
{
  Vrem_ = Teuchos::rcp(vrem.get(), false);

  // access to the dynamic Vreman class is provided
  // by the fluid scatra coupling algorithm
  // therefore, we only have to add some scatra specific parameters here
  if (turbmodel_ == INPAR::FLUID::dynamic_vreman)
  {
    if (myrank_ == 0)
    {
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << "Dynamic Vreman model: provided access for ScaTra            " << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
    Vrem_->AddScatra(discret_);
  }

  return;
}


/*-------------------------------------------------------------------------*
 | calculate mean CsgsB to estimate CsgsD                                  |
 | for multifractal subgrid-scale model                    rasthofer 08/12 |
 *-------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::RecomputeMeanCsgsB()
{
  if (DRT::INPUT::IntegralValue<int>(extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES"),"ADAPT_CSGS_PHI"))
  {
    // mean Cai
    double meanCai = 0.0;

    // variables required for calculation
    // local sums
    double local_sumCai = 0.0;
    double local_sumVol = 0.0;
    // global sums
    double global_sumCai = 0.0;
    double global_sumVol = 0.0;

    // define element matrices and vectors --- dummies
    Epetra_SerialDenseMatrix emat1;
    Epetra_SerialDenseMatrix emat2;
    Epetra_SerialDenseVector evec1;
    Epetra_SerialDenseVector evec2;
    Epetra_SerialDenseVector evec3;

    // generate a parameterlist for communication and control
    Teuchos::ParameterList myparams;

    // action for elements
    myparams.set<int>("action",SCATRA::calc_mean_Cai);

    // add number of dof-set associated with velocity related dofs
    myparams.set<int>("ndsvel",nds_vel_);

    // add element parameters according to time-integration scheme
    AddTimeIntegrationSpecificVectors();

    // set thermodynamic pressure in case of loma
    AddProblemSpecificParametersAndVectors(myparams);

    // loop all elements on this proc (excluding ghosted ones)
    for (int nele=0;nele<discret_->NumMyRowElements();++nele)
    {
      // get the element
      DRT::Element* ele = discret_->lRowElement(nele);

      // get element location vector, dirichlet flags and ownerships
      DRT::Element::LocationArray la(discret_->NumDofSets());
      ele->LocationVector(*discret_,la,false);

      // call the element evaluate method to integrate functions
      int err = ele->Evaluate(myparams,*discret_,la,
                            emat1,emat2,
                            evec1,evec2,evec2);
      if (err) dserror("Proc %d: Element %d returned err=%d",myrank_,ele->Id(),err);

      // get contributions of this element and add it up
      local_sumCai += myparams.get<double>("Cai_int");
      local_sumVol += myparams.get<double>("ele_vol");
    }

    // gather contibutions of all procs
    discret_->Comm().SumAll(&local_sumCai,&global_sumCai,1);
    discret_->Comm().SumAll(&local_sumVol,&global_sumVol,1);

    // calculate mean Cai
    meanCai = global_sumCai/global_sumVol;

    //std::cout << "Proc:  " << myrank_ << "  local vol and Cai   "
    //<< local_sumVol << "   " << local_sumCai << "  global vol and Cai   "
    //<< global_sumVol << "   " << global_sumCai << "  mean   " << meanCai << std::endl;

    if (myrank_ == 0)
    {
      std::cout << "\n+--------------------------------------------------------------------------------------------+" << std::endl;
      std::cout << "Multifractal subgrid scales: adaption of CsgsD from near-wall limit of CsgsB:  " << std::setprecision (8) << meanCai << std::endl;
      std::cout << "+--------------------------------------------------------------------------------------------+\n" << std::endl;
    }

    // set meanCai via pre-evaluate call
    myparams.set<int>("action",SCATRA::set_mean_Cai);
    myparams.set<double>("meanCai",meanCai);
    // call standard loop over elements
    discret_->Evaluate(myparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);
  }

  return;
}


/*----------------------------------------------------------------------*
 | calculate intermediate solution                       rasthofer 05/13|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcIntermediateSolution()
{
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" and
      extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING")=="isotropic" and
      DRT::INPUT::IntegralValue<INPAR::FLUID::ForcingType>(extraparams_->sublist("TURBULENCE MODEL"),"FORCING_TYPE")
      == INPAR::FLUID::linear_compensation_from_intermediate_spectrum)
  {
    bool activate = true;

    if (activate)
    {
      if (homisoturb_forcing_ == Teuchos::null)
        dserror("Forcing expected!");

      if (myrank_ == 0)
      {
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|     calculate intermediate solution\n";
        std::cout << "|"<< std::endl;
      }

      // turn off forcing in NonlinearSolve()
      homisoturb_forcing_->ActivateForcing(false);

      // temporary store velnp_ since it will be modified in NonlinearSolve()
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      Teuchos::RCP<Epetra_Vector> tmp = LINALG::CreateVector(*dofrowmap,true);
      tmp->Update(1.0,*phinp_,0.0);

      // compute intermediate solution without forcing
      forcing_->PutScalar(0.0); // just to be sure
      NonlinearSolve();

      // calculate required forcing
      homisoturb_forcing_->CalculateForcing(step_);

      // reset velnp_
      phinp_->Update(1.0,*tmp,0.0);

      // recompute intermediate values, since they have been likewise overwritten
      // only for gen.-alpha
      ComputeIntermediateValues();

      homisoturb_forcing_->ActivateForcing(true);

      if (myrank_ == 0)
      {
        std::cout << "|\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|" << std::endl;
      }
    }
    else
      // set force to zero
      forcing_->PutScalar(0.0);
  }

  return;
}


/*==========================================================================*/
// Biofilm related
/*==========================================================================*/

/*----------------------------------------------------------------------*/
/* set scatra-fluid displacement vector due to biofilm growth          */
void SCATRA::ScaTraTimIntImpl::SetScFldGrDisp(Teuchos::RCP<Epetra_MultiVector> scatra_fluid_growth_disp)
{
  scfldgrdisp_= scatra_fluid_growth_disp;

  return;
}

/*----------------------------------------------------------------------*/
/* set scatra-structure displacement vector due to biofilm growth          */
void SCATRA::ScaTraTimIntImpl::SetScStrGrDisp(Teuchos::RCP<Epetra_MultiVector> scatra_struct_growth_disp)
{
  scstrgrdisp_= scatra_struct_growth_disp;

  return;
}

/*----------------------------------------------------------------------*
 | Calculate the reconstructed nodal gradient of phi        winter 04/14|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector>  SCATRA::ScaTraTimIntImpl::ReconstructGradientAtNodes(
    const Teuchos::RCP<const Epetra_Vector> phi)
{

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<int>("action",SCATRA::recon_gradients_at_nodes);

//  ///=========================================
//  // Need to change call in scatra_ele_calc_general_service.cpp CalcGradientAtNodes(..,..,..)
//  //  if these lines are uncommented.
//  Teuchos::RCP<Epetra_MultiVector> gradPhi = Teuchos::rcp(new Epetra_MultiVector(*(discret_->DofRowMap()), 3, true));
//
//  // zero out matrix entries
//  sysmat_->Zero();
//
//  // set vector values needed by elements
//  discret_->ClearState();
//  discret_->SetState("phinp",phi);
//
//  Teuchos::RCP<Epetra_MultiVector> rhs_vectors = Teuchos::rcp(new Epetra_MultiVector(*(discret_->DofRowMap()), 3, true));
//
//  discret_->Evaluate(eleparams,sysmat_,Teuchos::null,Teuchos::rcp((*rhs_vectors)(0),false),Teuchos::rcp((*rhs_vectors)(1),false),Teuchos::rcp((*rhs_vectors)(2),false));
//
//  discret_->ClearState();
//
//  // finalize the complete matrix
//  sysmat_->Complete();
//
//  //TODO Remove for-loop and add Belos solver instead
//  for (int idim=0; idim<3 ; idim++)
//    solver_->Solve(sysmat_->EpetraOperator(),Teuchos::rcp((*gradPhi)(idim),false),Teuchos::rcp((*rhs_vectors)(idim),false),true,true);
//
//  ///=========================================

  ///**********************************

  // Probably not the nicest way...
  // Might want to have L2_projection incorporated into two-phase surface tension parameters
  const Teuchos::ParameterList& scatradyn =
    DRT::Problem::Instance()->ScalarTransportDynamicParams();
  const int lstsolver = scatradyn.get<int>("LINEAR_SOLVER");

  const int dim = DRT::Problem::Instance()->NDim();

  Teuchos::RCP<Epetra_MultiVector> gradPhi = SCATRA::ScaTraTimIntImpl::ComputeNodalL2Projection(phi,"phinp",dim,eleparams,lstsolver);

  //As the object is a MultiVector, this has to be done to keep the structure from before...
  //  not the nicest way but better than nothing.
  gradPhi->ReplaceMap(*(discret_->DofRowMap()));
  Teuchos::rcp((*gradPhi)(0),false)->ReplaceMap(*(discret_->DofRowMap()));
  Teuchos::rcp((*gradPhi)(1),false)->ReplaceMap(*(discret_->DofRowMap()));
  Teuchos::rcp((*gradPhi)(2),false)->ReplaceMap(*(discret_->DofRowMap()));

  return gradPhi;
}

/*----------------------------------------------------------------------*
 | Calculate the L2-projection of a scalar                  winter 04/14|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::ComputeNodalL2Projection(
    Teuchos::RCP<const Epetra_Vector> state,
    const std::string statename,
    const int numvec,
    Teuchos::ParameterList& eleparams,
    const int solvernumber)
{
  //Warning, this is only tested so far for 1 scalar field!!!

  // dependent on the desired projection, just remove this line
  if(not state->Map().SameAs(*discret_->DofRowMap()))
    dserror("input map is not a dof row map of the fluid");

  // set given state for element evaluation
  discret_->ClearState();
  discret_->SetState(statename,state);

  return DRT::UTILS::ComputeNodalL2Projection(discret_,statename,numvec,eleparams,solvernumber);;
}


/*--------------------------------------------------------------------------------------------------------------------*
 | convergence check (only for two-way coupled problems, e.g., low-Mach-number flow, two phase flow, ...)    vg 09/11 |
 *--------------------------------------------------------------------------------------------------------------------*/
bool SCATRA::ScaTraTimIntImpl::ConvergenceCheck(int          itnum,
                                                int          itmax,
                                                const double ittol)
{
  // declare bool variable for potentially stopping nonlinear iteration
  bool stopnonliniter = false;

  // declare L2-norm of (two) residual, incremental scalar and scalar
  double res1norm_L2(0.0);
  double res2norm_L2(0.0);
  double phi1incnorm_L2(0.0);
  double phi2incnorm_L2(0.0);
  double phi1norm_L2(0.0);
  double phi2norm_L2(0.0);

  // distinguish whether one or two scalars are considered
  if (numscal_ == 2)
  {
    Teuchos::RCP<Epetra_Vector> vec1 = splitter_->ExtractOtherVector(residual_);
    Teuchos::RCP<Epetra_Vector> vec2 = splitter_->ExtractCondVector(residual_);
    vec1->Norm2(&res1norm_L2);
    vec2->Norm2(&res2norm_L2);

    vec1 = splitter_->ExtractOtherVector(increment_);
    vec2 = splitter_->ExtractCondVector(increment_);
    vec1->Norm2(&phi1incnorm_L2);
    vec2->Norm2(&phi2incnorm_L2);

    vec1 = splitter_->ExtractOtherVector(phinp_);
    vec2 = splitter_->ExtractCondVector(phinp_);
    vec1->Norm2(&phi1norm_L2);
    vec2->Norm2(&phi2norm_L2);

    // check for any INF's and NaN's
    if (std::isnan(res1norm_L2) or
        std::isnan(phi1incnorm_L2) or
        std::isnan(phi1norm_L2) or
        std::isnan(res2norm_L2) or
        std::isnan(phi2incnorm_L2) or
        std::isnan(phi2norm_L2))
      dserror("At least one of the calculated vector norms is NaN.");

    if (std::isinf(res1norm_L2) or
        std::isinf(phi1incnorm_L2) or
        std::isinf(phi1norm_L2) or
        std::isinf(res2norm_L2) or
        std::isinf(phi2incnorm_L2) or
        std::isinf(phi2norm_L2))
      dserror("At least one of the calculated vector norms is INF.");

    // for scalar norm being (close to) zero, set to one
    if (phi1norm_L2 < 1e-5) phi1norm_L2 = 1.0;
    if (phi2norm_L2 < 1e-5) phi2norm_L2 = 1.0;

    if (myrank_==0)
    {
      printf("+------------+-------------------+---------------+---------------+---------------+---------------+\n");
      printf("|- step/max -|- tol      [norm] -|- residual1   -|- scalar1-inc -|- residual2   -|- scalar2-inc -|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  |  %10.3E   |  %10.3E   |  %10.3E   |  %10.3E   |",
           itnum,itmax,ittol,res1norm_L2,phi1incnorm_L2/phi1norm_L2,res2norm_L2,phi2incnorm_L2/phi2norm_L2);
      printf("\n");
      printf("+------------+-------------------+---------------+---------------+---------------+---------------+\n");
    }

     if ((res1norm_L2 <= ittol) and
        (phi1incnorm_L2/phi1norm_L2 <= ittol) and
         (res2norm_L2 <= ittol) and
        (phi2incnorm_L2/phi2norm_L2 <= ittol))
       stopnonliniter=true;

    // warn if itemax is reached without convergence, but proceed to next timestep
    if ((itnum == itmax) and
        ((res1norm_L2 > ittol) or (phi1incnorm_L2/phi1norm_L2 > ittol) or
         (res2norm_L2 > ittol) or (phi2incnorm_L2/phi2norm_L2 > ittol)))
    {
      stopnonliniter=true;
      if (myrank_==0)
      {
        printf("|            >>>>>> not converged in itemax steps!             |\n");
        printf("+--------------------------------------------------------------+\n");
      }
    }
  }
  else if (numscal_ == 1)
  {
    // compute L2-norm of residual, incremental scalar and scalar
    residual_ ->Norm2(&res1norm_L2);
    increment_->Norm2(&phi1incnorm_L2);
    phinp_    ->Norm2(&phi1norm_L2);

    // check for any INF's and NaN's
    if (std::isnan(res1norm_L2) or
        std::isnan(phi1incnorm_L2) or
        std::isnan(phi1norm_L2))
      dserror("At least one of the calculated vector norms is NaN.");

    if (std::isinf(res1norm_L2) or
        std::isinf(phi1incnorm_L2) or
        std::isinf(phi1norm_L2))
      dserror("At least one of the calculated vector norms is INF.");

    // for scalar norm being (close to) zero, set to one
    if (phi1norm_L2 < 1e-5) phi1norm_L2 = 1.0;

    if (myrank_==0)
    {
      printf("+------------+-------------------+--------------+--------------+\n");
      printf("|- step/max -|- tol      [norm] -|- residual   -|- scalar-inc -|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   |",
           itnum,itmax,ittol,res1norm_L2,phi1incnorm_L2/phi1norm_L2);
      printf("\n");
      printf("+------------+-------------------+--------------+--------------+\n");
    }

     if ((res1norm_L2 <= ittol) and
        (phi1incnorm_L2/phi1norm_L2 <= ittol)) stopnonliniter=true;

    // warn if itemax is reached without convergence, but proceed to next timestep
    if ((itnum == itmax) and
        ((res1norm_L2 > ittol) or (phi1incnorm_L2/phi1norm_L2 > ittol)))
    {
      stopnonliniter=true;
      if (myrank_==0)
      {
        printf("|            >>>>>> not converged in itemax steps!             |\n");
        printf("+--------------------------------------------------------------+\n");
      }
    }
  }
  else dserror("ScaTra convergence check for number of scalars other than one or two not yet supported!");

  return stopnonliniter;
} // SCATRA::ScaTraTimIntImplicit::ConvergenceCheck


/*----------------------------------------------------------------------------------------------*
 | finite difference check for scalar transport system matrix (for debugging only)   fang 10/14 |
 *----------------------------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::FDCheck()
{
  // initial screen output
  if(myrank_ == 0)
    std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR SCATRA SYSTEM MATRIX" << std::endl;

  // make a copy of state variables to undo perturbations later
  Teuchos::RCP<Epetra_Vector> phinp_original = Teuchos::rcp(new Epetra_Vector(*phinp_));

  // make a copy of system matrix as Epetra_CrsMatrix
  Teuchos::RCP<Epetra_CrsMatrix> sysmat_original = Teuchos::null;
  if(Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_) != Teuchos::null)
    sysmat_original = (new LINALG::SparseMatrix(*(Teuchos::rcp_static_cast<LINALG::SparseMatrix>(sysmat_))))->EpetraMatrix();
  else if(Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(sysmat_) != Teuchos::null)
    sysmat_original = (new LINALG::SparseMatrix(*(Teuchos::rcp_static_cast<LINALG::BlockSparseMatrixBase>(sysmat_)->Merge())))->EpetraMatrix();
  else
    dserror("Type of system matrix unknown!");
  sysmat_original->FillComplete();

  // make a copy of system right-hand side vector
  Teuchos::RCP<Epetra_Vector> rhs_original = Teuchos::rcp(new Epetra_Vector(*residual_));

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  for (int colgid=0; colgid<=sysmat_original->ColMap().MaxAllGID(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original->ColMap().LID(colgid));
    int maxcollid(-1);
    discret_->Comm().MaxAll(&collid,&maxcollid,1);
    if(maxcollid < 0)
      continue;

    // fill state vector with original state variables
    phinp_->Update(1.,*phinp_original,0.);

    // impose perturbation
    if(phinp_->Map().MyGID(colgid))
      if(phinp_->SumIntoGlobalValue(colgid,0,fdcheckeps_))
        dserror("Perturbation could not be imposed on state vector for finite difference check!");

    // carry perturbation over to state vectors at intermediate time stages if necessary
    ComputeIntermediateValues();

    // calculate element right-hand side vector for perturbed state
    AssembleMatAndRHS();

    // Now we compare the difference between the current entries in the system matrix
    // and their finite difference approximations according to
    // entries ?= (residual_perturbed - residual_original) / epsilon

    // Note that the residual_ vector actually denotes the right-hand side of the linear
    // system of equations, i.e., the negative system residual.
    // To account for errors due to numerical cancellation, we additionally consider
    // entries + residual_original / epsilon ?= residual_perturbed / epsilon

    // Note that we still need to evaluate the first comparison as well. For small entries in the system
    // matrix, the second comparison might yield good agreement in spite of the entries being wrong!
    for(int rowlid=0; rowlid<discret_->DofRowMap()->NumMyElements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original->RowMap().GID(rowlid);
      if(rowgid < 0)
        dserror("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original->NumMyEntries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original->ExtractMyRowCopy(rowlid,length,numentries,&values[0],&indices[0]);
      for(int ientry=0; ientry<length; ++ientry)
      {
        if(sysmat_original->ColMap().GID(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval = -(*residual_)[rowlid] / fdcheckeps_ + (*rhs_original)[rowlid] / fdcheckeps_;

      // confirm accuracy of first comparison
      if(abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
        dserror("Finite difference check involves values too close to numerical zero!");

      // absolute and relative errors in first comparison
      const double abserr1 = entry - fdval;
      if(abs(abserr1) > maxabserr)
        maxabserr = abs(abserr1);
      double relerr1(0.);
      if(abs(entry) > 1.e-17)
        relerr1 = abserr1 / abs(entry);
      else if(abs(fdval) > 1.e-17)
        relerr1 = abserr1 / abs(fdval);
      if(abs(relerr1) > maxrelerr)
        maxrelerr = abs(relerr1);

      // evaluate first comparison
      if(abs(relerr1) > fdchecktol_)
      {
        std::cout << "sysmat[" << rowgid << "," << colgid << "]:  " << entry << "   ";
        std::cout << "finite difference suggestion:  " << fdval << "   ";
        std::cout << "absolute error:  " << abserr1 << "   ";
        std::cout << "relative error:  " << relerr1 << std::endl;

        counter++;
      }

      // first comparison OK
      else
      {
        // left-hand side in second comparison
        const double left  = entry - (*rhs_original)[rowlid] / fdcheckeps_;

        // right-hand side in second comparison
        const double right = -(*residual_)[rowlid] / fdcheckeps_;

        // confirm accuracy of second comparison
        if(abs(right) > 1.e-17 and abs(right) < 1.e-15)
          dserror("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in second comparison
        const double abserr2 = left - right;
        if(abs(abserr2) > maxabserr)
          maxabserr = abs(abserr2);
        double relerr2(0.);
        if(abs(left) > 1.e-17)
          relerr2 = abserr2 / abs(left);
        else if(abs(right) > 1.e-17)
          relerr2 = abserr2 / abs(right);
        if(abs(relerr2) > maxrelerr)
          maxrelerr = abs(relerr2);

        // evaluate second comparison
        if(abs(relerr2) > fdchecktol_)
        {
          std::cout << "sysmat[" << rowgid << "," << colgid << "]-rhs[" << rowgid << "]/eps:  " << left << "   ";
          std::cout << "-rhs_perturbed[" << rowgid << "]/eps:  " << right << "   ";
          std::cout << "absolute error:  " << abserr2 << "   ";
          std::cout << "relative error:  " << relerr2 << std::endl;

          counter++;
        }
      }
    }
  }

  // communicate tracking variables
  int counterglobal(0);
  discret_->Comm().SumAll(&counter,&counterglobal,1);
  double maxabserrglobal(0.);
  discret_->Comm().MaxAll(&maxabserr,&maxabserrglobal,1);
  double maxrelerrglobal(0.);
  discret_->Comm().MaxAll(&maxrelerr,&maxrelerrglobal,1);

  // final screen output
  if(myrank_ == 0)
  {
    if(counterglobal)
    {
      printf("--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n",counterglobal);
      dserror("Finite difference check failed for scalar transport system matrix!");
    }
    else
      printf("--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",maxabserrglobal,maxrelerrglobal);
  }

  // undo perturbations of state variables
  phinp_->Update(1.,*phinp_original,0.);
  ComputeIntermediateValues();

  // recompute system matrix and right-hand side vector based on original state variables
  AssembleMatAndRHS();

  return;
}

/*----------------------------------------------------------------------*
 |  calculate error compared to analytical solution            gjb 10/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateErrorComparedToAnalyticalSol()
{
  const INPAR::SCATRA::CalcError calcerr
    = DRT::INPUT::IntegralValue<INPAR::SCATRA::CalcError>(*params_,"CALCERROR");

  if(calcerr == INPAR::SCATRA::calcerror_no) // do nothing (the usual case))
    return;

  // create the parameters for the error calculation
  Teuchos::ParameterList eleparams;
  eleparams.set<int>("action",SCATRA::calc_error);
  eleparams.set("total time",time_);
  eleparams.set<int>("calcerrorflag",calcerr);

  switch (calcerr)
  {
  case INPAR::SCATRA::calcerror_byfunction:
  {
    const int errorfunctnumber = params_->get<int>("CALCERRORNO");
    if(errorfunctnumber<1)
      dserror("invalid value of paramter CALCERRORNO for error function evaluation!");

    eleparams.set<int>("error function number",errorfunctnumber);
  break;
  }
  case INPAR::SCATRA::calcerror_spherediffusion:
    break;
  default:
    dserror("Cannot calculate error. Unknown type of analytical test problem"); break;
  }

  //provide displacement field in case of ALE
  if (isale_)
    eleparams.set<int>("ndsdisp",nds_disp_);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);

  // get (squared) error values
  Teuchos::RCP<Epetra_SerialDenseVector> errors
    = Teuchos::rcp(new Epetra_SerialDenseVector(4*numscal_));
  discret_->EvaluateScalars(eleparams, errors);
  discret_->ClearState();

  // std::vector containing
  // [0]: relative L2 scalar error
  // [1]: relative H1 scalar error
  Teuchos::RCP<std::vector<double> > relerror = Teuchos::rcp(new std::vector<double>(2*numscal_));

  for(int k=0;k<numscal_;k++)
  {
    if( std::abs((*errors)[k*numscal_+2])>1e-14 )
      (*relerror)[k*numscal_+0] = sqrt((*errors)[k*numscal_+0])/sqrt((*errors)[k*numscal_+2]);
    else
      (*relerror)[k*numscal_+0] = sqrt((*errors)[k*numscal_+0]);
    if( std::abs((*errors)[k*numscal_+2])>1e-14 )
      (*relerror)[k*numscal_+1] = sqrt((*errors)[k*numscal_+1])/sqrt((*errors)[k*numscal_+3]);
    else
      (*relerror)[k*numscal_+1] = sqrt((*errors)[k*numscal_+1]);

    if (myrank_ == 0)
    {

      // print last error in a separate file

      // append error of the last time step to the error file
//        if ((step_==stepmax_) or (time_==maxtime_))// write results to file
//        {
//          std::ostringstream temp;
//          temp << k;
//          const std::string simulation = DRT::Problem::Instance()->OutputControlFile()->FileName();
//          const std::string fname = simulation+"_c"+temp.str()+".relerror";
//
//          std::ofstream f;
//          f.open(fname.c_str(),std::fstream::ate | std::fstream::app);
//          f << "#| " << simulation << "\n";
//          f << "#| Step | Time | rel. L2-error scalar | rel. H1-error scalar  |\n";
//          f << step_ << " " << time_ << " " << (*relerror)[k*numscal_+0] << " " << (*relerror)[k*numscal_+1] << "\n";
//          f.flush();
//          f.close();
//        }


      std::ostringstream temp;
      temp << k;
      const std::string simulation = DRT::Problem::Instance()->OutputControlFile()->FileName();
      const std::string fname = simulation+"_c"+temp.str()+"_time.relerror";

      if(step_==0)
      {
        std::ofstream f;
        f.open(fname.c_str());
        f << "#| Step | Time | rel. L2-error  | rel. H1-error  |\n";
        f << std::setprecision(10) << step_ << " " << std::setw(1)<< std::setprecision(5)
          << time_ << std::setw(1) << std::setprecision(6) << " "
          << (*relerror)[k*numscal_+0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[k*numscal_+1] << "\n";

        f.flush();
        f.close();
      }
      else
      {
        std::ofstream f;
        f.open(fname.c_str(),std::fstream::ate | std::fstream::app);
        f << std::setprecision(10) << step_ << " " << std::setw(3)<< std::setprecision(5)
        << time_ << std::setw(1) << std::setprecision(6) << " "
        << (*relerror)[k*numscal_+0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[k*numscal_+1] <<"\n";

        f.flush();
        f.close();
      }
    }
  }

  return;
} // SCATRA::ScaTraTimIntImpl::EvaluateErrorComparedToAnalyticalSol
