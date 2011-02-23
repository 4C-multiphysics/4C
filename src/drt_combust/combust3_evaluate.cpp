/*!
\file combust3_evaluate.cpp
\brief

<pre>
Maintainer: Florian Henke
            henke@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15265
</pre>
*/
#ifdef D_FLUID3
#ifdef CCADISCRET

#include <Teuchos_TimeMonitor.hpp>

#include "combust3.H"
#include "combust3_sysmat.H"
#include "combust3_interpolation.H"
#include "combust_defines.H"

#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_xfem/dof_management.H"
#include "../drt_xfem/xdofmapcreation_combust.H"
#include "../drt_xfem/enrichment_utils.H"
#include "../drt_inpar/inpar_fluid.H"
#include "../drt_mat/newtonianfluid.H"
#include "../drt_mat/matlist.H"
#include "../drt_f3/fluid3_stabilization.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>


// converts a string into an Action for this element
DRT::ELEMENTS::Combust3::ActionType DRT::ELEMENTS::Combust3::convertStringToActionType(
              const string& action) const
{
  DRT::ELEMENTS::Combust3::ActionType act = Combust3::none;
  if (action == "calc_fluid_systemmat_and_residual")
    act = Combust3::calc_fluid_systemmat_and_residual;
  else if (action == "calc_fluid_stationary_systemmat_and_residual")
    act = Combust3::calc_fluid_stationary_systemmat_and_residual;
  else if (action == "calc_fluid_beltrami_error")
    act = Combust3::calc_fluid_beltrami_error;
  else if (action == "calc_nitsche_error")
    act = Combust3::calc_nitsche_error;
  else if (action == "calc_turbulence_statistics")
    act = Combust3::calc_turbulence_statistics;
  else if (action == "calc_fluid_box_filter")
    act = Combust3::calc_fluid_box_filter;
  else if (action == "calc_smagorinsky_const")
    act = Combust3::calc_smagorinsky_const;
  else if (action == "store_xfem_info")
    act = Combust3::store_xfem_info;
  else if (action == "get_density")
    act = Combust3::get_density;
  else if (action == "reset")
    act = Combust3::reset;
  else if (action == "set_output_mode")
    act = Combust3::set_output_mode;
  else
    dserror("Unknown type of action for Combust3");
  return act;
}

/*----------------------------------------------------------------------*
 // converts a string into an stabilisation action for this element
 //                                                          gammi 08/07
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::StabilisationAction DRT::ELEMENTS::Combust3::ConvertStringToStabAction(
  const string& action) const
{
  DRT::ELEMENTS::Combust3::StabilisationAction act = stabaction_unspecified;

  map<string,StabilisationAction>::const_iterator iter=stabstrtoact_.find(action);

  if (iter != stabstrtoact_.end())
  {
    act = (*iter).second;
  }
  else
  {
    dserror("looking for stab action (%s) not contained in map",action.c_str());
  }
  return act;
}


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                           g.bau 03/07 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Combust3::Evaluate(ParameterList& params,
                                     DRT::Discretization&      discretization,
                                     std::vector<int>&         lm,
                                     Epetra_SerialDenseMatrix& elemat1,
                                     Epetra_SerialDenseMatrix&,
                                     Epetra_SerialDenseVector& elevec1,
                                     Epetra_SerialDenseVector&,
                                     Epetra_SerialDenseVector&)
{
  // get the action required
  const string action(params.get<string>("action","none"));
  const DRT::ELEMENTS::Combust3::ActionType act = convertStringToActionType(action);

  // get the list of materials
  const Teuchos::RCP<MAT::Material> material = Material();

  switch(act)
  {
    case get_density:
    {
      std::cout << "Warning! The density is set to 1.0!" << std::endl;
      params.set("density", 1.0);
    }
    break;
    case reset:
    {
      // Reset all information and make element unusable (e.g. it can't answer the numdof question
      // anymore). This way, one can see, if all information is generated correctly or whether
      // something was left from the last nonlinear iteration.
      eleDofManager_ = Teuchos::null;
      eleDofManager_uncondensed_ = Teuchos::null;
      ih_ = NULL;
      DLM_info_ = Teuchos::null;
    }
    break;
    case set_output_mode:
    {
      output_mode_ = true;
      // reset element dof manager if present
      eleDofManager_ = Teuchos::null;
      eleDofManager_uncondensed_ = Teuchos::null;
      ih_ = NULL;
    }
    break;
    case store_xfem_info:
    {
      TEUCHOS_FUNC_TIME_MONITOR("COMBUST3 - evaluate - store_xfem_info");

      // now the element can answer how many (XFEM) dofs it has
      output_mode_ = false;

      // store pointer to interface handle
      ih_ = &*params.get< Teuchos::RCP< COMBUST::InterfaceHandleCombust > >("interfacehandle",Teuchos::null);

      //--------------------------------------------------
      // find out whether an element is intersected or not
      //--------------------------------------------------
      // remark: initialization call of fluid time integration scheme will also end up here: The initial
      //         flame front has not been incorporated into the fluid field -> no XFEM dofs, yet!
      this->bisected_      = false;
      this->touched_plus_  = false;
      this->touched_minus_ = false;

      if (ih_->FlameFront() != Teuchos::null) // not the initial call
      {
        // more than one domain integration cell -> element bisected
        if(ih_->ElementBisected(this))
          this->bisected_ = true;
        // one domain and one plus boundary integration cell -> element touched plus
        else if(ih_->ElementTouchedPlus(this))
          this->touched_plus_ = true;
        // one domain and one minus boundary integration cell -> element touched minus
        else if(ih_->ElementTouchedMinus(this))
          this->touched_minus_ = true;
        else // regular element (numdomaincells==1 and numboundarycells==0)
        {
// TODO @Florian uncomment DEBUG flag
//#if DEBUG
          std::size_t numDomainCells = ih_->GetNumDomainIntCells(this);
          if (numDomainCells<1) // 'numcells' is zero or negative number
            // impossible, something went wrong!
            dserror ("unknown number of DomainIntCells for element %d ", this->Id());
          std::size_t numBoundaryCells = ih_->GetNumBoundaryIntCells(this);
          if(numBoundaryCells!=0)
            // impossible, something went wrong!
            dserror ("unknown number of BoundaryIntCells for element %d ", this->Id());
        }
//#endif
      }

      //----------------------------------
      // hand over global dofs to elements
      //----------------------------------
      // get access to global dofman
      const Teuchos::RCP<const XFEM::DofManager> globaldofman = params.get< Teuchos::RCP< XFEM::DofManager > >("dofmanager");

#ifdef COMBUST_STRESS_BASED
#ifdef COMBUST_EPSPRES_BASED
      Teuchos::RCP<XFEM::ElementAnsatz> elementAnsatz = rcp(new COMBUST::EpsilonPressureAnsatz());
#endif
#ifdef COMBUST_SIGMA_BASED
      Teuchos::RCP<XFEM::ElementAnsatz> elementAnsatz = rcp(new COMBUST::CauchyStressAnsatz());
#endif
#endif
      // create an empty element ansatz map to be filled in the following
      map<XFEM::PHYSICS::Field, DRT::Element::DiscretizationType> element_ansatz_filled;
      // create an empty element ansatz map
      const map<XFEM::PHYSICS::Field, DRT::Element::DiscretizationType> element_ansatz_empty;
#ifdef COMBUST_STRESS_BASED
      // ask for appropriate element ansatz (shape functions) for this type of element and fill the map
      element_ansatz_filled = elementAnsatz->getElementAnsatz(this->Shape());
#endif
      //-------------------------------------------------------------------------
      // build element dof manager according to global dof manager
      // remark: this procedure is closely related to XFEM::createDofMapCombust()
      //-------------------------------------------------------------------------
#ifdef COMBUST_STRESS_BASED
      if (params.get<bool>("DLM_condensation")) // DLM condensation turned on
      {
        // add node dofs, but no element dofs (stress unknowns) for this element
        eleDofManager_ = rcp(new XFEM::ElementDofManager(*this, element_ansatz_empty, *globaldofman));
      }
      else // DLM condensation turned off
      {
        // add node dofs and element dofs (stress unknowns) for this element
        eleDofManager_ = rcp(new XFEM::ElementDofManager(*this, element_ansatz_filled, *globaldofman));
      }
#else
      // add node dofs, but no element dofs (stress unknowns) for this element
      eleDofManager_ = rcp(new XFEM::ElementDofManager(*this, element_ansatz_empty, *globaldofman));
#endif

      //----------------------------------------------------------------------------
      // build element dof manager holding element unknowns for intersected elements
      // remark: this procedure is closely related to XFEM::createDofMapCombust()
      //         condensation for uncut elements is not possible and unneccessary
      //----------------------------------------------------------------------------
      // TODO @Florian implement eleDofs for touched elements
      // schott Aug 3, 2010
      if (this->bisected_ || this->touched_plus_ || this->touched_minus_)
      {
        // create empty set of enrichment fields
        std::set<XFEM::FieldEnr> elementFieldEnrSet;

#ifdef COMBUST_STRESS_BASED
        // control boolean not used here
        bool skipped_elem_enr = false;
        // apply element enrichments (fill elementFieldEnrSet)
        // remark: This procedure must give the same result as the element enrichment procedure in
        //         createDofMapCombust(). The element dof manager has to be consistent with the
        //         global dof manager!
        skipped_elem_enr = ApplyElementEnrichmentCombust(this, element_ansatz_filled,
                                                         elementFieldEnrSet, *ih_,
                                                         params.get<double>("boundaryRatioLimit"));

        // add node dofs and element dofs (stress unknowns) for this element
        eleDofManager_uncondensed_ = rcp(new XFEM::ElementDofManager(*this, eleDofManager_->getNodalDofSet(), elementFieldEnrSet, element_ansatz_filled));
#else
        // add node dofs and element dofs (stress unknowns) for this element
        eleDofManager_uncondensed_ = rcp(new XFEM::ElementDofManager(*this, eleDofManager_->getNodalDofSet(), elementFieldEnrSet, element_ansatz_empty));
#endif
        DLM_info_ = rcp(new DLMInfo(eleDofManager_uncondensed_->NumNodeDof(), eleDofManager_uncondensed_->NumElemDof()));
      }
      else // element not intersected
      {
        eleDofManager_uncondensed_ = Teuchos::null;
        DLM_info_ = Teuchos::null;
      }
    }
    break;
    case calc_fluid_systemmat_and_residual:
    {
      TEUCHOS_FUNC_TIME_MONITOR("COMBUST3 - evaluate - calc_fluid_systemmat_and_residual");

      // do no calculation, if not needed
      if (lm.empty())
        break;

      const INPAR::COMBUST::CombustionType combusttype = DRT::INPUT::get<INPAR::COMBUST::CombustionType>(params, "combusttype");
      const INPAR::COMBUST::VelocityJumpType veljumptype = DRT::INPUT::get<INPAR::COMBUST::VelocityJumpType>(params, "veljumptype");
      const INPAR::COMBUST::FluxJumpType fluxjumptype = DRT::INPUT::get<INPAR::COMBUST::FluxJumpType>(params, "fluxjumptype");
      const INPAR::COMBUST::SmoothGradPhi smoothgradphi = DRT::INPUT::get<INPAR::COMBUST::SmoothGradPhi>(params, "smoothgradphi");

      // instationary formulation
      const bool instationary = true;
      // smoothed gradient of phi required (surface tension application)
      double gradphi = true;
      if (combusttype == INPAR::COMBUST::combusttype_twophaseflow or
          smoothgradphi == INPAR::COMBUST::smooth_grad_phi_none)
      {
        gradphi = false;
      }

      // extract local (element level) vectors from global state vectors
      DRT::ELEMENTS::Combust3::MyState mystate(discretization, lm, instationary, gradphi, this, ih_);

      const bool newton = params.get<bool>("include reactive terms for linearisation",false);

      const double flamespeed = params.get<double>("flamespeed");
      const double nitschevel = params.get<double>("nitschevel");
      const double nitschepres = params.get<double>("nitschepres");

      // stabilization terms
      const bool pstab = true;
      const bool supg  = true;
      const bool cstab = true;
      // stabilization parameters
      const INPAR::FLUID::TauType tautype = DRT::INPUT::IntegralValue<INPAR::FLUID::TauType>(params.sublist("STABILIZATION"),"DEFINITION_TAU");
      // check if stabilization parameter definition can be handled by combust3 element
      if (!(tautype == INPAR::FLUID::tau_taylor_hughes_zarins or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_scaled or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt or
            tautype == INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall or
            tautype == INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt or
            tautype == INPAR::FLUID::tau_shakib_hughes_codina or
            tautype == INPAR::FLUID::tau_shakib_hughes_codina_wo_dt))
        dserror("unknown type of stabilization parameter definition");

      // time integration parameters
      const INPAR::FLUID::TimeIntegrationScheme timealgo = DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params, "timealgo");
      const double            dt       = params.get<double>("dt");
      const double            theta    = params.get<double>("theta");
#ifdef SUGRVEL_OUTPUT
      //const int               step     = params.get<int>("step");
#endif

      // parameters for two-phase flow problems with surface tension
      // type of surface tension approximation
      const INPAR::COMBUST::SurfaceTensionApprox surftensapprox = DRT::INPUT::get<INPAR::COMBUST::SurfaceTensionApprox>(params, "surftensapprox");
      const bool connected_interface = params.get<bool>("connected_interface");
      const bool smoothed_boundary_integration = params.get<bool>("smoothed_bound_integration");

#ifdef COMBUST_STRESS_BASED
      // integrate and assemble all unknowns
      if (not this->bisected_ or
          not params.get<bool>("DLM_condensation"))
      {
        const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
            *eleDofManager_, NumNode(), NodeIds());

        if (ih_->GetNumBoundaryIntCells(this) > 0)
          cout << "/!\\ warning === element " << this->Id() << " is not intersected, but has boundary integration cells!" << endl;

        // calculate element coefficient matrix and rhs
        COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_, mystate, elemat1, elevec1,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface, veljumptype, fluxjumptype,smoothed_boundary_integration);
      }
      // create bigger element matrix and vector, assemble, condense and copy to small matrix provided by discretization
      else
      {
        if (eleDofManager_uncondensed_ == Teuchos::null)
          dserror("Intersected element %d has no element dofs", this->Id());

        UpdateOldDLMAndDLMRHS(discretization, lm, mystate);

        // create uncondensed element matrix and vector
        const int numdof_uncond = eleDofManager_uncondensed_->NumDofElemAndNode();
        Epetra_SerialDenseMatrix elemat1_uncond(numdof_uncond,numdof_uncond);
        Epetra_SerialDenseVector elevec1_uncond(numdof_uncond);

        const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
            *eleDofManager_uncondensed_, NumNode(), NodeIds());

        // calculate element coefficient matrix and rhs
        COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_uncondensed_, mystate, elemat1_uncond, elevec1_uncond,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface, veljumptype, fluxjumptype,smoothed_boundary_integration);

        // condensation
        CondenseElementStressAndStoreOldIterationStep(
            elemat1_uncond, elevec1_uncond,
            elemat1, elevec1
        );
      }
#else
      const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
          *eleDofManager_, NumNode(), NodeIds());

      // schott Jun 16, 2010
      // calculate element coefficient matrix and rhs
      COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_, mystate, elemat1, elevec1,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface,veljumptype,fluxjumptype,smoothed_boundary_integration);
#endif
    }
    break;
    case calc_fluid_stationary_systemmat_and_residual:
    {
      TEUCHOS_FUNC_TIME_MONITOR("COMBUST3 - evaluate - calc_fluid_stationary_systemmat_and_residual");
      // do no calculation, if not needed
      if (lm.empty())
        break;

      const INPAR::COMBUST::CombustionType combusttype = DRT::INPUT::get<INPAR::COMBUST::CombustionType>(params, "combusttype");
      const INPAR::COMBUST::VelocityJumpType veljumptype = DRT::INPUT::get<INPAR::COMBUST::VelocityJumpType>(params, "veljumptype");
      const INPAR::COMBUST::FluxJumpType fluxjumptype = DRT::INPUT::get<INPAR::COMBUST::FluxJumpType>(params, "fluxjumptype");
      const INPAR::COMBUST::SmoothGradPhi smoothgradphi = DRT::INPUT::get<INPAR::COMBUST::SmoothGradPhi>(params, "smoothgradphi");

      // stationary formulation
      const bool instationary = false;
      // smoothed gradient of phi required (surface tension application)
      double gradphi = true;
      if (combusttype == INPAR::COMBUST::combusttype_twophaseflow or
          smoothgradphi == INPAR::COMBUST::smooth_grad_phi_none)
      {
        gradphi = false;
      }

      // extract local (element level) vectors from global state vectors
      DRT::ELEMENTS::Combust3::MyState mystate(discretization, lm, instationary, gradphi, this, ih_);

      const bool newton = params.get<bool>("include reactive terms for linearisation",false);

      const double flamespeed = params.get<double>("flamespeed");
      const double nitschevel = params.get<double>("nitschevel");
      const double nitschepres = params.get<double>("nitschepres");

      // parameters for two-phase flow problems with surface tension
      // type of surface tension approximation
      const INPAR::COMBUST::SurfaceTensionApprox surftensapprox = DRT::INPUT::get<INPAR::COMBUST::SurfaceTensionApprox>(params, "surftensapprox");
      const bool connected_interface = params.get<bool>("connected_interface");
      const bool smoothed_boundary_integration = params.get<bool>("smoothed_bound_integration");

      // stabilization terms
      const bool pstab = true;
      const bool supg  = true;
      const bool cstab = true;
      // stabilization parameters
      const INPAR::FLUID::TauType tautype = DRT::INPUT::IntegralValue<INPAR::FLUID::TauType>(params.sublist("STABILIZATION"),"DEFINITION_TAU");
      // check if stabilization parameter definition can be handled by combust3 element
      if (!(tautype == INPAR::FLUID::tau_taylor_hughes_zarins or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_wo_dt or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_whiting_jansen_wo_dt or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_scaled or
            tautype == INPAR::FLUID::tau_taylor_hughes_zarins_scaled_wo_dt or
            tautype == INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall or
            tautype == INPAR::FLUID::tau_franca_barrenechea_valentin_frey_wall_wo_dt or
            tautype == INPAR::FLUID::tau_shakib_hughes_codina or
            tautype == INPAR::FLUID::tau_shakib_hughes_codina_wo_dt))
        dserror("unknown type of stabilization parameter definition");

      // time integration factors
      const INPAR::FLUID::TimeIntegrationScheme timealgo = DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(params, "timealgo");
      dsassert(timealgo == INPAR::FLUID::timeint_stationary, "must be stationary!");
      const double            dt       = 1.0;
      const double            theta    = 1.0;
#ifdef SUGRVEL_OUTPUT
      const int               step     = 0;
#endif

#ifdef COMBUST_STRESS_BASED
      // integrate and assemble all unknowns
      if (not this->bisected_ or
          not params.get<bool>("DLM_condensation"))
      {
        if (ih_->GetNumBoundaryIntCells(this) > 0)
          cout << "/!\\ warning === element " << this->Id() << " is not intersected, but has boundary integration cells!" << endl;

        const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
            *eleDofManager_, NumNode(), NodeIds());

        // calculate element coefficient matrix and rhs
        COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_, mystate, elemat1, elevec1,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface, veljumptype, fluxjumptype,smoothed_boundary_integration);
      }
      // create bigger element matrix and vector, assemble, condense and copy to small matrix provided by discretization
      else
      {
        if (eleDofManager_uncondensed_ == Teuchos::null)
          dserror("Intersected element %d has no element dofs", this->Id());

        UpdateOldDLMAndDLMRHS(discretization, lm, mystate);

        // create uncondensed element matrix and vector
        const int numdof_uncond = eleDofManager_uncondensed_->NumDofElemAndNode();
        //cout << "element " <<  this->Id() << "number of node dofs " << eleDofManager_uncondensed_->NumNodeDof() << endl;
        //cout << "element " <<  this->Id() << "number of element dofs " << eleDofManager_uncondensed_->NumElemDof() << endl;
        //cout << "element " <<  this->Id() << "total number of dofs " << numdof_uncond << endl;
        Epetra_SerialDenseMatrix elemat1_uncond(numdof_uncond,numdof_uncond);
        Epetra_SerialDenseVector elevec1_uncond(numdof_uncond);

        const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
            *eleDofManager_uncondensed_, NumNode(), NodeIds());

//        if (assembly_type == XFEM::standard_assembly) cout << "element " << this->Id() << " standard assembly " << endl;
//        if (assembly_type == XFEM::xfem_assembly) cout << "element " << this->Id() << " xfem assembly " << endl;

//        const int numnodes = this->NumNode();
//        const int* nodeidptrs = this->NodeIds();
//        for (int inode = 0; inode<numnodes; ++inode)
//        {
//          const int nodeid = nodeidptrs[inode];
//          cout << "num dof per node " << eleDofManager_->NumDofPerNode(nodeid) << endl;
//        }
//        cout << "num dof per field velx " << eleDofManager_uncondensed_->NumParamsPerField().find(XFEM::PHYSICS::Velx)->second << endl;
//        cout << "num dof per field vely " << eleDofManager_uncondensed_->NumParamsPerField().find(XFEM::PHYSICS::Vely)->second << endl;
//        cout << "num dof per field velz " << eleDofManager_uncondensed_->NumParamsPerField().find(XFEM::PHYSICS::Velz)->second << endl;
//        cout << "num dof per field veln " << eleDofManager_uncondensed_->NumParamsPerField().find(XFEM::PHYSICS::Veln)->second << endl;
//        cout << "num dof per field pres " << eleDofManager_uncondensed_->NumParamsPerField().find(XFEM::PHYSICS::Pres)->second << endl;

        // calculate element coefficient matrix and rhs
        COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_uncondensed_, mystate, elemat1_uncond, elevec1_uncond,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface, veljumptype, fluxjumptype,smoothed_boundary_integration);

        // condensation
        CondenseElementStressAndStoreOldIterationStep(
            elemat1_uncond, elevec1_uncond,
            elemat1, elevec1
        );
      }
#else
      const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
          *eleDofManager_, NumNode(), NodeIds());

      // calculate element coefficient matrix and rhs
      COMBUST::callSysmat(assembly_type,
          this, ih_, *eleDofManager_, mystate, elemat1, elevec1,
          material, timealgo, dt, theta, newton, pstab, supg, cstab, tautype, instationary,
          combusttype, flamespeed, nitschevel, nitschepres, surftensapprox,
          connected_interface,veljumptype,fluxjumptype,smoothed_boundary_integration);
#endif

#if 0
          const XFEM::BoundaryIntCells&  boundaryIntCells(ih_->GetBoundaryIntCells(this->Id()));
          if ((assembly_type == XFEM::xfem_assembly) and (not boundaryIntCells.empty()))
          {
              const int entry = 4; // line in stiffness matrix to compare
              const double disturbance = 1.0e-4;

              // initialize locval
              for (std::size_t i = 0;i < locval.size(); ++i)
              {
                  locval[i] = 0.0;
                  locval_hist[i] = 0.0;
              }
              // R_0
              // calculate element coefficient matrix and rhs
              XFLUID::callSysmat4(assembly_type,
                      this, ih_, eleDofManager_, locval, locval_hist, ivelcol, iforcecol, estif, eforce,
                      mat, pseudotime, 1.0, newton, pstab, supg, cstab, false);

              LINALG::SerialDensevector eforce_0(locval.size());
              for (std::size_t i = 0;i < locval.size(); ++i)
              {
                  eforce_0(i) = eforce(i);
              }

              // create disturbed vector
              vector<double> locval_disturbed(locval.size());
              for (std::size_t i = 0;i < locval.size(); ++i)
              {
                  if (i == entry)
                  {
                      locval_disturbed[i] = locval[i] + disturbance;
                  }
                  else
                  {
                      locval_disturbed[i] = locval[i];
                  }
                  std::cout << locval[i] <<  " " << locval_disturbed[i] << endl;
              }


              // R_0+dx
              // calculate element coefficient matrix and rhs
              XFLUID::callSysmat4(assembly_type,
                      this, ih_, eleDofManager_, locval_disturbed, locval_hist, ivelcol, iforcecol, estif, eforce,
                      mat, pseudotime, 1.0, newton, pstab, supg, cstab, false);



              // compare
              std::cout << "sekante" << endl;
              for (std::size_t i = 0;i < locval.size(); ++i)
              {
                  //cout << i << endl;
                  const double matrixentry = (eforce_0(i) - eforce(i))/disturbance;
                  printf("should be %+12.8E, is %+12.8E, factor = %5.2f, is %+12.8E, factor = %5.2f\n", matrixentry, estif(i, entry), estif(i, entry)/matrixentry, estif(entry,i), estif(entry,i)/matrixentry);
                  //cout << "should be: " << std::scientific << matrixentry << ", is: " << estif(entry, i) << " " << estif(i, entry) << endl;
              }

              exit(0);
          }
          else
#endif
    }
    break;
    case calc_fluid_beltrami_error:
    {
      // add error only for elements which are not ghosted
      if(this->Owner() == discretization.Comm().MyPID())
      {
        // need current velocity and history vector
        RefCountPtr<const Epetra_Vector> vel_pre_np = discretization.GetState("u and p at time n+1 (converged)");
        if (vel_pre_np==null)
          dserror("Cannot get state vectors 'velnp'");

        // extract local values from the global vectors
        std::vector<double> my_vel_pre_np(lm.size());
        DRT::UTILS::ExtractMyValues(*vel_pre_np,my_vel_pre_np,lm);

        // split "my_vel_pre_np" into velocity part "myvelnp" and pressure part "myprenp"
        const int numnode = NumNode();
        vector<double> myprenp(numnode);
        vector<double> myvelnp(3*numnode);

        for (int i=0;i<numnode;++i)
        {
          myvelnp[0+(i*3)]=my_vel_pre_np[0+(i*4)];
          myvelnp[1+(i*3)]=my_vel_pre_np[1+(i*4)];
          myvelnp[2+(i*3)]=my_vel_pre_np[2+(i*4)];

          myprenp[i]=my_vel_pre_np[3+(i*4)];
        }

        // integrate beltrami error
        f3_int_beltrami_err(myvelnp,myprenp,material,params);
      }
    }
    break;
    case calc_nitsche_error:
    {
      TEUCHOS_FUNC_TIME_MONITOR("COMBUST3 - evaluate - calc Nitsche errors");

      // add error only for elements which are not ghosted
      if(this->Owner() == discretization.Comm().MyPID())
      {
        // stationary formulation
        const bool instationary = false;
        // smoothed gradient of phi required (surface tension application)
        const bool gradphi = true;
        const bool smoothed_boundary_integration = params.get<bool>("smoothed_bound_integration");
        const INPAR::COMBUST::NitscheError NitscheErrorType = DRT::INPUT::get<INPAR::COMBUST::NitscheError>(params, "Nitsche_Compare_Analyt");

        // extract local (element level) vectors from global state vectors
        DRT::ELEMENTS::Combust3::MyState mystate(discretization, lm, instationary, gradphi, this, ih_);

        // get assembly type
        const XFEM::AssemblyType assembly_type = XFEM::ComputeAssemblyType(
            *eleDofManager_, NumNode(), NodeIds());

        // calculate Nitsche norms
        COMBUST::callNitscheErrors(params, NitscheErrorType, assembly_type, this, ih_, *eleDofManager_, mystate, material,smoothed_boundary_integration);
      }
    }
    break;
    default:
      dserror("Unknown type of action for Combust3");
  } // end of switch(act)

  return 0;
}


/*----------------------------------------------------------------------*
 |  do nothing (public)                                      gammi 04/07|
 |                                                                      |
 |  The function is just a dummy. For the fluid elements, the           |
 |  integration of the volume neumann (body forces) loads takes place   |
 |  in the element. We need it there for the stabilisation terms!       |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Combust3::EvaluateNeumann(ParameterList& params,
                                            DRT::Discretization&      discretization,
                                            DRT::Condition&           condition,
                                            std::vector<int>&         lm,
                                            Epetra_SerialDenseVector& elevec1,
                                            Epetra_SerialDenseMatrix* elemat1)
{
  return 0;
}

// get optimal gaussrule for discretization type
DRT::UTILS::GaussRule3D DRT::ELEMENTS::Combust3::getOptimalGaussrule(const DiscretizationType& distype)
{
  DRT::UTILS::GaussRule3D rule = DRT::UTILS::intrule3D_undefined;
    switch (distype)
    {
    case hex8:
        rule = DRT::UTILS::intrule_hex_8point;
        break;
    case hex20: case hex27:
        rule = DRT::UTILS::intrule_hex_27point;
        break;
    case tet4:
        rule = DRT::UTILS::intrule_tet_4point;
        break;
    case tet10:
        rule = DRT::UTILS::intrule_tet_5point;
        break;
    default:
        dserror("unknown number of nodes for gaussrule initialization");
  }
  return rule;
}

/*---------------------------------------------------------------------*
 |  calculate error for beltrami test problem               gammi 04/07|
 *---------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::f3_int_beltrami_err(
    std::vector<double>&      evelnp,
    std::vector<double>&      eprenp,
    Teuchos::RCP<const MAT::Material> material,
    ParameterList&            params
    )
{
  const int NSD = 3;

  // add element error to "integrated" error
  double velerr = params.get<double>("L2 integrated velocity error");
  double preerr = params.get<double>("L2 integrated pressure error");

  // set element data
  const int iel = NumNode();
  const DiscretizationType distype = this->Shape();

  Epetra_SerialDenseVector  funct(iel);
  Epetra_SerialDenseMatrix  xjm(3,3);
  Epetra_SerialDenseMatrix  deriv(3,iel);

  // get node coordinates of element
  Epetra_SerialDenseMatrix xyze(3,iel);
  for(int inode=0;inode<iel;inode++)
  {
    xyze(0,inode)=Nodes()[inode]->X()[0];
    xyze(1,inode)=Nodes()[inode]->X()[1];
    xyze(2,inode)=Nodes()[inode]->X()[2];
  }

  // set constants for analytical solution
  const double t = params.get("total time",-1.0);
  dsassert (t >= 0.0, "beltrami: no total time for error calculation");

  const double a      = M_PI/4.0;
  const double d      = M_PI/2.0;

  // get viscosity
  double  visc = 0.0;
  // just to be sure - actually this has already been checked in Combust3::Evaluate, before
  dsassert(material->MaterialType() == INPAR::MAT::m_matlist, "material is not of type m_matlist");
  const MAT::MatList* matlist = static_cast<const MAT::MatList*>(material.get());
  // use material MAT 3 (first in the material list) for Beltrami flow
  Teuchos::RCP<const MAT::Material> matptr = matlist->MaterialById(3);
  dsassert(matptr->MaterialType() == INPAR::MAT::m_fluid, "material is not of type m_fluid");
  std::cout << "material MAT 3 is used for Beltrami flow" << std::endl;
  const MAT::NewtonianFluid* mat = static_cast<const MAT::NewtonianFluid*>(matptr.get());
  visc = mat->Viscosity();

  double         preint;
  vector<double> velint  (3);
  vector<double> xint    (3);

  vector<double> u       (3);

  double         deltap;
  vector<double> deltavel(3);

  // gaussian points
  const DRT::UTILS::GaussRule3D gaussrule = getOptimalGaussrule(distype);
  const DRT::UTILS::IntegrationPoints3D  intpoints(gaussrule);

  // start loop over integration points
  for (int iquad=0;iquad<intpoints.nquad;iquad++)
  {
    // declaration of gauss point variables
    const double e1 = intpoints.qxg[iquad][0];
    const double e2 = intpoints.qxg[iquad][1];
    const double e3 = intpoints.qxg[iquad][2];
    DRT::UTILS::shape_function_3D(funct,e1,e2,e3,distype);
    DRT::UTILS::shape_function_3D_deriv1(deriv,e1,e2,e3,distype);

    /*----------------------------------------------------------------------*
      | calculate Jacobian matrix and it's determinant (private) gammi  07/07|
      | Well, I think we actually compute its transpose....
      |
      |     +-            -+ T      +-            -+
      |     | dx   dx   dx |        | dx   dy   dz |
      |     | --   --   -- |        | --   --   -- |
      |     | dr   ds   dt |        | dr   dr   dr |
      |     |              |        |              |
      |     | dy   dy   dy |        | dx   dy   dz |
      |     | --   --   -- |   =    | --   --   -- |
      |     | dr   ds   dt |        | ds   ds   ds |
      |     |              |        |              |
      |     | dz   dz   dz |        | dx   dy   dz |
      |     | --   --   -- |        | --   --   -- |
      |     | dr   ds   dt |        | dt   dt   dt |
      |     +-            -+        +-            -+
      |
      *----------------------------------------------------------------------*/
    LINALG::Matrix<NSD,NSD>    xjm;

    for (int isd=0; isd<NSD; isd++)
    {
      for (int jsd=0; jsd<NSD; jsd++)
      {
        double dum = 0.0;
        for (int inode=0; inode<iel; inode++)
        {
          dum += deriv(isd,inode)*xyze(jsd,inode);
        }
        xjm(isd,jsd) = dum;
      }
    }

    // determinant of jacobian matrix
    const double det = xjm.Determinant();

    if(det < 0.0)
    {
        printf("\n");
        printf("GLOBAL ELEMENT NO.%i\n",Id());
        printf("NEGATIVE JACOBIAN DETERMINANT: %f\n", det);
        dserror("Stopped not regulary!\n");
    }

    const double fac = intpoints.qwgt[iquad]*det;

    // get velocity sol at integration point
    for (int i=0;i<3;i++)
    {
      velint[i]=0.0;
      for (int j=0;j<iel;j++)
      {
        velint[i] += funct[j]*evelnp[i+(3*j)];
      }
    }

    // get pressure sol at integration point
    preint = 0;
    for (int inode=0;inode<iel;inode++)
    {
      preint += funct[inode]*eprenp[inode];
    }

    // get velocity sol at integration point
    for (int isd=0;isd<3;isd++)
    {
      xint[isd]=0.0;
      for (int inode=0;inode<iel;inode++)
      {
        xint[isd] += funct[inode]*xyze(isd,inode);
      }
    }

    // compute analytical pressure
    const double p = -a*a/2.0 *
        ( exp(2.0*a*xint[0])
        + exp(2.0*a*xint[1])
        + exp(2.0*a*xint[2])
        + 2.0 * sin(a*xint[0] + d*xint[1]) * cos(a*xint[2] + d*xint[0]) * exp(a*(xint[1]+xint[2]))
        + 2.0 * sin(a*xint[1] + d*xint[2]) * cos(a*xint[0] + d*xint[1]) * exp(a*(xint[2]+xint[0]))
        + 2.0 * sin(a*xint[2] + d*xint[0]) * cos(a*xint[1] + d*xint[2]) * exp(a*(xint[0]+xint[1]))
        )* exp(-2.0*visc*d*d*t);

    // compute analytical velocities
    u[0] = -a * ( exp(a*xint[0]) * sin(a*xint[1] + d*xint[2]) +
                  exp(a*xint[2]) * cos(a*xint[0] + d*xint[1]) ) * exp(-visc*d*d*t);
    u[1] = -a * ( exp(a*xint[1]) * sin(a*xint[2] + d*xint[0]) +
                  exp(a*xint[0]) * cos(a*xint[1] + d*xint[2]) ) * exp(-visc*d*d*t);
    u[2] = -a * ( exp(a*xint[2]) * sin(a*xint[0] + d*xint[1]) +
                  exp(a*xint[1]) * cos(a*xint[2] + d*xint[0]) ) * exp(-visc*d*d*t);

    // compute difference between analytical solution and numerical solution
    deltap = preint - p;

    for (int isd=0;isd<NSD;isd++)
    {
      deltavel[isd] = velint[isd]-u[isd];
    }

    // add square to L2 error
    for (int isd=0;isd<NSD;isd++)
    {
      velerr += deltavel[isd]*deltavel[isd]*fac;
    }
    preerr += deltap*deltap*fac;

  } // end of loop over integration points


  // we use the parameterlist as a container to transport the calculated
  // errors from the elements to the dynamic routine

  params.set<double>("L2 integrated velocity error",velerr);
  params.set<double>("L2 integrated pressure error",preerr);

  return;
}


/*------------------------------------------------------------------------------------------------*
 | evaluate element stresses and pressure and update element solution vector          henke 04/10 |
 *------------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::UpdateOldDLMAndDLMRHS(
    const DRT::Discretization&      discretization,
    const std::vector<int>&         lm,
    MyState&                        mystate
    ) const
{
  const int numnodedof = eleDofManager_uncondensed_->NumNodeDof();
  const int numeledof = eleDofManager_uncondensed_->NumElemDof();

  // check if element dofs really exist
  if (numeledof > 0)
  {
    // add Kda . inc_velnp to feas
    // new alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas

    // extract local (element) increment from global vector
    vector<double> inc_velnp(lm.size());
    DRT::UTILS::ExtractMyValues(*discretization.GetState("velpres nodal iterinc"),inc_velnp,lm);

    static const Epetra_BLAS blas;

    //-------------------------------------------------------
    // update old iteration residual of stresses and pressure
    //-------------------------------------------------------
    // DLM_info_->oldfa_(i) += DLM_info_->oldKad_(i,j)*inc_velnp[j];
    blas.GEMV('N', numeledof, numnodedof,-1.0, DLM_info_->oldKad_.A(), DLM_info_->oldKad_.LDA(), &inc_velnp[0], 1.0, DLM_info_->oldfa_.A());

    //--------------------------------------
    // compute element stresses and pressure
    //--------------------------------------
    // DLM_info_->stressdofs_(i) -= DLM_info_->oldKaainv_(i,j)*DLM_info_->oldfa_(j);
    blas.GEMV('N', numeledof, numeledof,1.0, DLM_info_->oldKaainv_.A(), DLM_info_->oldKaainv_.LDA(), DLM_info_->oldfa_.A(), 1.0, DLM_info_->stressdofs_.A());

    //----------------------------------------------------------------------
    // paste element dofs (stresses and pressure) in lovcal (element) vector
    //----------------------------------------------------------------------
    // increase size of element vectors (old values stay and zeros are added)
    const int numdof_uncond = eleDofManager_uncondensed_->NumDofElemAndNode();
    mystate.velnp_.resize(numdof_uncond,0.0);
    if (mystate.instationary_)
    {
      mystate.veln_ .resize(numdof_uncond,0.0);
      mystate.velnm_.resize(numdof_uncond,0.0);
      mystate.accn_ .resize(numdof_uncond,0.0);
    }
    for (int ieledof=0;ieledof<numeledof;ieledof++)
    {
      mystate.velnp_[numnodedof+ieledof] = DLM_info_->stressdofs_(ieledof);
    }
  }
  else
    dserror("You should never have come here in the first place!");
}


/*------------------------------------------------------------------------------------------------*
 | condense element dofs             henke 04/10 |
 *------------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::CondenseElementStressAndStoreOldIterationStep(
    const Epetra_SerialDenseMatrix& elemat1_uncond,
    const Epetra_SerialDenseVector& elevec1_uncond,
    Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseVector& elevec1
) const
{
  const size_t numnodedof = eleDofManager_uncondensed_->NumNodeDof();
  const size_t numeledof = eleDofManager_uncondensed_->NumElemDof();

  // copy nodal dof entries
  // TODO why is this done? henke 09/04/10
  for (size_t i = 0; i < numnodedof; ++i)
  {
    elevec1(i) = elevec1_uncond(i);
    for (size_t j = 0; j < numnodedof; ++j)
    {
      elemat1(i,j) = elemat1_uncond(i,j);
    }
  }

  // check if element dofs really exist
  if (numeledof > 0)
  {
    // note: the full (u,p,sigma) matrix is asymmetric,
    // hence we need both rectangular matrices Kda and Kad
    LINALG::SerialDenseMatrix Gus   (numnodedof,numeledof);
    LINALG::SerialDenseMatrix Kssinv(numeledof,numeledof);
    LINALG::SerialDenseMatrix KGsu  (numeledof,numnodedof);
    LINALG::SerialDenseVector fs    (numeledof);

//    cout << elemat1_uncond << endl;

    // copy data of uncondensed matrix into submatrices
    for (size_t i=0;i<numnodedof;i++)
      for (size_t j=0;j<numeledof;j++)
        Gus(i,j) = elemat1_uncond(i,numnodedof+j);

    for (size_t i=0;i<numeledof;i++)
      for (size_t j=0;j<numeledof;j++)
        Kssinv(i,j) = elemat1_uncond(numnodedof+i,numnodedof+j);

    for (size_t i=0;i<numeledof;i++)
      for (size_t j=0;j<numnodedof;j++)
        KGsu(i,j) = elemat1_uncond(numnodedof+i,j);

    for (size_t i=0;i<numeledof;i++)
      fs(i) = elevec1_uncond(numnodedof+i);

    // DLM-stiffness matrix is: Kdd - Kda . Kaa^-1 . Kad
    // DLM-internal force is: fint - Kda . Kaa^-1 . feas

    // we need the inverse of Kaa
    Epetra_SerialDenseSolver solve_for_inverseKaa;
    solve_for_inverseKaa.SetMatrix(Kssinv);
    solve_for_inverseKaa.Invert();

    static const Epetra_BLAS blas;
    {
      LINALG::SerialDenseMatrix GusKssinv(numnodedof,numeledof); // temporary Gus.Kss^{-1}

      // GusKssinv(i,j) = Gus(i,k)*Kssinv(k,j);
      blas.GEMM('N','N',numnodedof,numeledof,numeledof,1.0,Gus.A(),Gus.LDA(),Kssinv.A(),Kssinv.LDA(),0.0,GusKssinv.A(),GusKssinv.LDA());

      // elemat1(i,j) += - GusKssinv(i,k)*KGsu(k,j);   // note that elemat1 = Cuu below
      blas.GEMM('N','N',numnodedof,numnodedof,numeledof,-1.0,GusKssinv.A(),GusKssinv.LDA(),KGsu.A(),KGsu.LDA(),1.0,elemat1.A(),elemat1.LDA());

      // elevec1(i) += - GusKssinv(i,j)*fs(j);
      blas.GEMV('N', numnodedof, numeledof,-1.0, GusKssinv.A(), GusKssinv.LDA(), fs.A(), 1.0, elevec1.A());
    }

    // store current DLM data in iteration history
    //DLM_info_->oldKaainv_.Update(1.0,Kaa,0.0);
    blas.COPY(DLM_info_->oldKaainv_.M()*DLM_info_->oldKaainv_.N(), Kssinv.A(), DLM_info_->oldKaainv_.A());
    //DLM_info_->oldKad_.Update(1.0,Kad,0.0);
    blas.COPY(DLM_info_->oldKad_.M()*DLM_info_->oldKad_.N(), KGsu.A(), DLM_info_->oldKad_.A());
    //DLM_info_->oldfa_.Update(1.0,fa,0.0);
    blas.COPY(DLM_info_->oldfa_.M()*DLM_info_->oldfa_.N(), fs.A(), DLM_info_->oldfa_.A());
  }
  else
    dserror("You should never have come here in the first place!");
}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_FLUID3
