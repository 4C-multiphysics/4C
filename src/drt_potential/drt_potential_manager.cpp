/*!-------------------------------------------------------------------
\file drt_potential_manager.cpp

\brief  Class controlling surface stresses due to potential forces
        between interfaces of mesoscopic structures

<pre>
Maintainer: Ursula Mayer
            mayer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>

*--------------------------------------------------------------------*/


#include "drt_potential_manager.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/inpar_potential.H"
#include "../drt_inpar/inpar_searchtree.H"
#include <cstdlib>


/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;


/*-------------------------------------------------------------------*
 |  ctor (public)                                          umay 06/08|
 *-------------------------------------------------------------------*/
POTENTIAL::PotentialManager::PotentialManager(
    const Teuchos::RCP<DRT::Discretization>   discretRCP,
    DRT::Discretization&                      discret):
    discretRCP_(discretRCP),
    discret_(discret),
    surfacePotential_(Teuchos::null),
    volumePotential_(Teuchos::null),
    surface_(false),
    volume_(false)
{

  ReadParameter();
  string pot_type = params_.get<string>("potential type");
  // construct surface and contact potential
  if( pot_type =="Surface" ||
      pot_type =="Surfacevolume")
      surface_ = true;

  if( pot_type =="Volume" ||
      pot_type =="Surfacevolume")
      volume_ = true;

  if(surface_)
    surfacePotential_ = rcp(new POTENTIAL::SurfacePotential(discretRCP,discret,treetype_));

  if(volume_)
    volumePotential_ = rcp(new POTENTIAL::VolumePotential(discretRCP,discret,treetype_));

  cout << "Potential manager constructed" << endl;
  return;
}



/*-------------------------------------------------------------------*
 |  ReadParameter (private)                                umay 06/09|
 *-------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::ReadParameter()
{
  const Teuchos::ParameterList& intpot   = DRT::Problem::Instance()->InteractionPotentialParams();
  // parameters for interaction potential
  
  switch(DRT::INPUT::IntegralValue<INPAR::POTENTIAL::PotentialType>(intpot,"POTENTIAL_TYPE"))
  {
    case INPAR::POTENTIAL::potential_surface:
      params_.set<string>("potential type","Surface");
    break;
    case INPAR::POTENTIAL::potential_volume:
      params_.set<string>("potential type","Volume");
    break;
    case INPAR::POTENTIAL::potential_surfacevolume:
      params_.set<string>("potential type","Surfacevolume");
    break;
    case INPAR::POTENTIAL::potential_surface_fsi:
      params_.set<string>("potential type","Surface_fsi");
    break;
    case INPAR::POTENTIAL::potential_volume_fsi:
      params_.set<string>("potential type","Volume_fsi");
    break;
    case INPAR::POTENTIAL::potential_surfacevolume_fsi:
      params_.set<string>("potential type","Surfacevolume_fsi");
    break;
    default:
      params_.set<string>("potential type","Surface");
    break;
  }
 
  // set approximation method for volume potentials
  switch(DRT::INPUT::IntegralValue<INPAR::POTENTIAL::ApproximationType>(intpot,"APPROXIMATION_TYPE"))
  {
    case INPAR::POTENTIAL::approximation_none:
      params_.set<string>("approximation type","None");
    break;
    case INPAR::POTENTIAL::approximation_surface:
      params_.set<string>("approximation type","Surface_approx");
    break;
    case INPAR::POTENTIAL::approximation_point:
      params_.set<string>("approximation type","Point_approx");
    break;
    default:
      params_.set<string>("approximation type","None");
    break;
  }
  
  // check if analytical solution should be computed
  switch(DRT::INPUT::IntegralValue<INPAR::POTENTIAL::SolutionType>(intpot,"ANALYTICALSOLUTION"))
  {
    case INPAR::POTENTIAL::solution_none:
      params_.set<string>("solution type","None");
    break;
    case INPAR::POTENTIAL::solution_sphere:
      params_.set<string>("solution type","Sphere");
    break;
    case INPAR::POTENTIAL::solution_membrane:
      params_.set<string>("solution type","Membrane");
    break;
    default:
      params_.set<string>("solution type","None");
    break;
  }
  
  // read radius of a sphere
  params_.set<double>("vdw_radius",intpot.get<double>("VDW_RADIUS"));
  
  // read number of atoms offset to account for spatial discretization errors
  params_.set<double>("n_offset",intpot.get<double>("N_OFFSET"));
  
  // read membrane thickness
   params_.set<double>("thickness",intpot.get<double>("THICKNESS"));
  
  // parameters for search tree
  const Teuchos::ParameterList& search_tree   = DRT::Problem::Instance()->SearchtreeParams();

  switch(DRT::INPUT::IntegralValue<INPAR::GEO::TreeType>(search_tree,"TREE_TYPE"))
  {
    case INPAR::GEO::Octree3D:
      treetype_ = GEO::OCTTREE;
    break;
    case INPAR::GEO::Quadtree3D:
      treetype_ = GEO::QUADTREE;
    break;
    case INPAR::GEO::Quadtree2D:
      treetype_ = GEO::QUADTREE;
    break;
    default:
      dserror("please specify search tree type");
    break;
  }
  
  return;
}



/*-------------------------------------------------------------------*
| (public)                                                 umay 06/08|
|                                                                    |
| Call discretization to evaluate additional contributions due to    |
| potential forces                                                   |
*--------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::EvaluatePotential(  ParameterList&                    p,
                                                      RefCountPtr<Epetra_Vector>        disp,
                                                      RefCountPtr<Epetra_Vector>        fint,
                                                      RefCountPtr<LINALG::SparseMatrix> stiff)
{
  if(surface_)
    surfacePotential_->EvaluatePotential(p, disp, fint, stiff);
  if(volume_)
    volumePotential_->EvaluatePotential(p, disp, fint, stiff);
  return;
}


/*-------------------------------------------------------------------*
| (public)                                                 umay 06/08|
|                                                                    |
| Call discretization to evaluate additional contributions due to    |
| potential forces                                                   |
*--------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::TestEvaluatePotential(  ParameterList&                    p,
                                                          RefCountPtr<Epetra_Vector>        disp,
                                                          RefCountPtr<Epetra_Vector>        fint,
                                                          RefCountPtr<LINALG::SparseMatrix> stiff,
                                                          const double                      time,
                                                          const int                         step)
{
  cout << "EVALUATE" << endl;
  const double vdw_radius = params_.get<double>("vdw_radius",0.0);
  const double n_offset = params_.get<double>("n_offset",0.0);
  const double thickness = params_.get<double>("thickness",0.0);
  p.set("vdw_radius", vdw_radius);
  p.set("n_offset", n_offset);
  p.set("thickness", thickness);
  
  const Teuchos::ParameterList& intpot   = DRT::Problem::Instance()->InteractionPotentialParams();
  switch(DRT::INPUT::IntegralValue<INPAR::POTENTIAL::SolutionType>(intpot,"ANALYTICALSOLUTION"))
  {
    case INPAR::POTENTIAL::solution_none:
      p.set<string>("solution type","None");
    break;
    case INPAR::POTENTIAL::solution_sphere:
      p.set<string>("solution type","Sphere");
    break;
    case INPAR::POTENTIAL::solution_membrane:
      p.set<string>("solution type","Membrane");
    break;
    default:
      p.set<string>("solution type","None");
    break;
  }
  
  if(surface_)
    surfacePotential_->TestEvaluatePotential(p, disp, fint, stiff, time, step);
  if(volume_)
    volumePotential_->TestEvaluatePotential(p, disp, fint, stiff, time, step);
  return;
}


/*-------------------------------------------------------------------*
| (public)                                                umay  06/08|
|                                                                    |
| Calculate additional internal forces and corresponding stiffness   |
| on element level for Lennard-Jones potential interaction forces    |
*--------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::StiffnessAndInternalForcesPotential(
    const DRT::Element*             element,
    const DRT::UTILS::GaussRule2D&  gaussrule,
    ParameterList&                  eleparams,
    vector<int>&                    lm,
    Epetra_SerialDenseMatrix&       K_stiff,
    Epetra_SerialDenseVector&       F_int)
{
  if( params_.get<string>("approximation type") == "None" )
  {	
  	int prob_dim = DRT::Problem::Instance()->NDim();
  	// due to the Gaussrule 2D
  	if(prob_dim == 2)
  	  volumePotential_->StiffnessAndInternalForcesPotential(element, gaussrule, eleparams, lm, K_stiff, F_int);
  	else if(prob_dim == 3)
  		surfacePotential_->StiffnessAndInternalForcesPotential(element, gaussrule, eleparams, lm, K_stiff, F_int);
  	else
  	 dserror("problem dimension not correct");
  }
  else if( params_.get<string>("approximation type")== "Surface_approx" )
    surfacePotential_->StiffnessAndInternalForcesPotentialApprox1(element, gaussrule, eleparams, lm, K_stiff, F_int);
  else if( params_.get<string>("approximation type")== "Point_approx" )
    surfacePotential_->StiffnessAndInternalForcesPotentialApprox2(element, gaussrule, eleparams, lm, K_stiff, F_int);
  else
    dserror("no approximation type specified");
      
  return;
}


/*-------------------------------------------------------------------*
| (public)                                                umay  09/09|
|                                                                    |
| Calculate additional internal forces and corresponding stiffness   |
| on element level for Lennard-Jones potential interaction forces    |
*--------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::StiffnessAndInternalForcesPotential(
    const DRT::Element*             element,
    const DRT::UTILS::GaussRule3D&  gaussrule,
    ParameterList&                  eleparams,
    vector<int>&                    lm,
    Epetra_SerialDenseMatrix&       K_stiff,
    Epetra_SerialDenseVector&       F_int)
{ 
  //TODO
  // check in solid hex 8 if elemat and elevec are properly filled !!!
  if( params_.get<string>("approximation type")== "None" )
    volumePotential_->StiffnessAndInternalForcesPotential(element, gaussrule, eleparams, lm, K_stiff, F_int);
  else
    dserror("no approximation allowed");
  return;
}


/*-------------------------------------------------------------------*
| (public)                                                umay  06/08|
|                                                                    |
| Calculate additional internal forces and corresponding stiffness   |
| for line elements                                                  |
*--------------------------------------------------------------------*/
void POTENTIAL::PotentialManager::StiffnessAndInternalForcesPotential(
    const DRT::Element*             element,
    const DRT::UTILS::GaussRule1D&  gaussrule,
    ParameterList&                  eleparams,
    vector<int>&                    lm,
    Epetra_SerialDenseMatrix&       K_stiff,
    Epetra_SerialDenseVector&       F_int)
{
  //if( params_.get<string>("approximation type")== "none" )
  //  surfacePotential_->StiffnessAndInternalForcesPotential(element, gaussrule, eleparams, lm, K_stiff, F_int);
    
  return;
}


/*-------------------------------------------------------------------*
| (public)                                                umay  04/10|
|                                                                    |
| Check if analytical solution should be computed                    |
*--------------------------------------------------------------------*/
bool POTENTIAL::PotentialManager::ComputeAnalyticalSolution() 
{
  if( params_.get<string>("solution type") == "None" ) 
    return false;

  return true;
}





