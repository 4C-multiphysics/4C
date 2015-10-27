/*!
\file xfem_coupling_levelset.cpp

\brief manages the different types of level-set based coupling conditions and thereby builds the bridge between the
xfluid class and the cut-library

<pre>
Maintainer: Benedikt Schott & Magnus Winter
            {schott, winter}@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
*/

#include <Teuchos_TimeMonitor.hpp>

#include "xfem_coupling_levelset.H"
#include "xfem_utils.H"

#include "../linalg/linalg_utils.H"

#include "../drt_inpar/inpar_xfem.H"
#include "../drt_inpar/inpar_fluid.H"

#include "../drt_io/io.H"
#include "../drt_io/io_pstream.H"

XFEM::LevelSetCoupling::LevelSetCoupling(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name, ///< name of the condition, by which the derived cutter discretization is identified
    const int                           coupling_id,///< id of composite of coupling conditions
    const double                        time,      ///< time
    const int                           step       ///< time step
) : CouplingBase(bg_dis, cond_name, bg_dis,coupling_id,time,step)
{
  /// level-set field is given w.r.t background mesh
  /// NOTE: more generally it would be possible cutterdis != bg_dis for the single LevelSetCoupling,
  /// however, the unique bg_phinp vector stored in the ConditionManager has to be given w.r.t bgdis
  cutter_dis_ = bg_dis;

  SetConditionsToCopy();

  SetElementConditions();

  // set the averaging strategy
  SetAveragingStrategy();

  // set coupling discretization
  SetCouplingDiscretization();

  // create node-based vector with level-set values
  phinp_ = Teuchos::rcp(new Epetra_Vector(*cutter_dis_->NodeRowMap()));

  // read initial level-set field
  SetLevelSetField(time_);

  // set level-boolean type (may be overwritten in constructors of derived class
  SetLevelSetBooleanType();

  //For output:
  ls_output_ = cutter_dis_->Writer();

}


void XFEM::LevelSetCoupling::SetConditionsToCopy()
{
  // set only the unique given condition name
  conditions_to_copy_.push_back(cond_name_);

  // additional conditions required for the levelset field based on the cutter (background) mesh
  conditions_to_copy_.push_back("XFEMSurfDisplacement");
}

// set level-boolean type
void XFEM::LevelSetCoupling::SetLevelSetBooleanType()
{

  if(cutterele_conds_.size() == 0)
    dserror("no element condition for LevelSetCouplingBC set. Not possible to extract BOOLEANTYPE!");

  DRT::Condition* cond = (cutterele_conds_[0]).second;
  const std::string* booleantype = cond->Get<std::string>("booleantype");

  if(*booleantype == "none")
    ls_boolean_type_ = ls_none;
  else if(*booleantype == "cut")
    ls_boolean_type_ = ls_cut;
  else if(*booleantype == "union")
    ls_boolean_type_ = ls_union;
  else if(*booleantype == "difference")
    ls_boolean_type_ = ls_difference;
  else if(*booleantype == "sym_difference")
    ls_boolean_type_ = ls_sym_difference;
  else
    dserror("not a valid boolean type %s: ", booleantype->c_str());
}

bool XFEM::LevelSetCoupling::ApplyComplementaryOperator()
{
  if(cutterele_conds_.size() == 0)
    dserror("no element condition for LevelSetCouplingBC set. Not possible to extract BOOLEANTYPE!");

  DRT::Condition* cond = (cutterele_conds_[0]).second;
  bool complementary = (bool)cond->GetInt("complementary");

  return complementary;
}


void XFEM::LevelSetCoupling::Output(
    const int step,
    const double time,
    const bool write_restart_data,
    const int lsc_idx
)
{
  // output for level-set interface
  //ls_output_->NewStep(step,time); // not required, as already called for the bgdis when output is written for fluid fields

  std::ostringstream temp;
  temp << lsc_idx;
  std::string name = "phinp_"+temp.str();

  ls_output_->WriteVector(name, phinp_);

  // write restart
  if (write_restart_data)
  {
    std::ostringstream temp2;
    temp2 << lsc_idx;
    std::string name_restart = "phinp_res_"+temp.str();

    ls_output_->WriteVector(name_restart, phinp_);
  }

}

// -------------------------------------------------------------------
// Read Restart data for cutter discretization
// -------------------------------------------------------------------
void XFEM::LevelSetCoupling::ReadRestart(
    const int step,
    const int lsc_idx
)
{

//  dserror("Not tested Level Set restart from file. Should most likely work though if this dserror is removed.");

  //-------- boundary discretization
  IO::DiscretizationReader boundaryreader(cutter_dis_, step);

  const double time = boundaryreader.ReadDouble("time");

  if(myrank_ == 0)
  {
    IO::cout << "            RESTART IS PERFORMED FROM FUNCTION IN INPUT FILE!                  " << IO::endl;
    IO::cout << "ReadRestart for Level Set Cut in Xfluid (time="<< time <<" ; step="<< step <<")" << IO::endl;
  }

  SetLevelSetField(time);

}

/*----------------------------------------------------------------------*
 | ... |
 *----------------------------------------------------------------------*/
bool XFEM::LevelSetCoupling::SetLevelSetField(const double time)
{

  // make a copy of last time step
  Teuchos::RCP<Epetra_Vector> delta_phi = Teuchos::rcp(new Epetra_Vector(phinp_->Map(),true));
  delta_phi->Update(1.0, *phinp_, 0.0);

  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

  // get the function from the first element
  const int lid=0;
  DRT::Condition* cond = cutterele_conds_[lid].second;
  const int func_no = cond->GetInt("levelsetfieldno");

  // check for potential time curve
  const int curvenum  = cond->GetInt("levelsetcurve");

  // initialization of time-curve factor
  double curvefac = 0.0;

  // compute potential time curve or set time-curve factor to one
  if (curvenum >= 0)
  {
    // time factor (negative time indicating error)
    if (time >= 0.0)
      curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
    else dserror("Negative time in function evaluation: time = %f", time);
  }
  else curvefac = 1.0;

  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lnode = cutter_dis_->lRowNode(lnodeid);

    // get value
    if(func_no < 0)
      value = FunctImplementation(func_no, lnode->X(),time);
    else if(func_no >= 1)
      value=DRT::Problem::Instance()->Funct(func_no-1).Evaluate(0,lnode->X(),time,NULL);
    else
      dserror("invalid function no. to set level-set field!");

    double final_val = curvefac*value;

    // now copy the values
    err = phinp_->ReplaceMyValue(lnodeid,0,final_val);
    if (err != 0) dserror("error while inserting value into phinp_");
  }

  delta_phi->Update(1.0, *phinp_, -1.0); // phinp - phin

  double norm = 0.0;
  delta_phi->Norm2(&norm);

  return (norm > 1e-14); // did interface change?
}


/*----------------------------------------------------------------------*
 | set interface level set field at current time           schott 02/15 |
 *----------------------------------------------------------------------*/
double XFEM::LevelSetCoupling::FunctImplementation(
    const int      func_no,
    const double * coords,
    const double t
)
{
//  dserror("you try to evaluate an implemented function for level-set field! Which one?");
  // WARNING!

  double x = coords[0];
  double y = coords[1];
  double z = coords[2];

  double val = 0.0;

  double R = 0.2;
  double r = 0.1;

  const double alpha = 0.6;


  if(func_no == -1)
  {


    // level set field for a helical pipe
    // sqrt( (x-Rcos(2 pi t))^2 + (y-Rsin(2 pi t))^2 + (z-alpha t)^2 -r = 0
    // with t(x,y,z) solution of minimization problem
    // dist((x,y,z), curve(t(x,y,z))) = min!

    // with curve(t) a parametrized curve (e.g. a helix)



    // NEWTON SYSTEM FOR SOLVING FOR t
    // d''(t) Delta_t = -d'(t)
    // t_n+1 = t_n + Delta_t

    // HELICAL CURVE z=alpha*t


    const double two_alpha_squared = 2.0*alpha*alpha;
    double two_PI = 2.0*PI;

    double t_0 = z/alpha;

    double Jac = 0.0;
    double rhs = 1.0;

    int armijo_steps = 50;

    int maxiter = 50;

    for(int i=0; i< maxiter; i++)
    {
      if(fabs(rhs)<1e-13) break;

      double arc= two_PI*t_0;
      double cosine = cos(arc);
      double sine   = sin(arc);
      Jac = 4.0*PI*R*(two_PI*x*cosine + two_PI*y*sine)+two_alpha_squared;
      rhs = 4.0*PI*R*(x*sine-y*cosine) + two_alpha_squared * t_0 - 2.0*alpha*z;


      double dt = -rhs/Jac;


      double armijo = 1.0;

      if(i<armijo_steps)
      {
        // it may happen, that the Newton direction is not a descent direction, then change the search direction
        // grad(f(x))^T * searchdir < 0 !   <=>  d'(t)*dt < 0   <=>  rhs*dt < 0
        if(dt*rhs > 0.0)
          dt*=-1.0;

        for(int l=0; l< 5; ++l)
        {
          if( l>0)
            armijo *= 0.5;

          // d(t+armijo*dt) < d(t) !!! and armijo (0,1] möglichst nahe an 1
          double t_new = t_0+armijo*dt;
          double arc_new = two_PI*t_new ;
          double cosine_new  = cos(arc_new);
          double sine_new    = sin(arc_new);

          double tmpx_new = x-R*cosine_new;
          double tmpy_new = y-R*sine_new;
          double tmpz_new = z-alpha*t_new;
          double norm1_squared = tmpx_new*tmpx_new+tmpy_new*tmpy_new+tmpz_new*tmpz_new;

          double tmpx = x-R*cosine;
          double tmpy = y-R*sine;
          double tmpz = z-alpha*t_0;
          double norm2_squared = tmpx*tmpx+tmpy*tmpy+tmpz*tmpz;

          if(norm1_squared < norm2_squared)
            break;
        }
      }

      t_0 += dt*armijo;

      if(i > maxiter-1)
      {
        std::cout << "Jac: " << Jac << std::endl;
        std::cout << "i: " << i << " rhs " << rhs << std::endl;
        std::cout << "armijo: " << armijo << std::endl;

        dserror("did not converge properly, intial guess not good enough - increase helixal height alpha!");
      }

    }


    double curve = alpha*t_0;

    double angle = two_PI*t_0;

    double cosine = cos(angle);
    double sine   = sin(angle);

    double tmp1 = x-R*cosine;
    double tmp2 = y-R*sine;
    double tmp3 = z-curve;

    double val_helix = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3)-r;

    return val_helix;
  }
  else if(func_no == -2)
  {
    double n1 = 0.0;
    double n2 = 2.0*PI*R;
    double n3 = alpha;

    double norm = sqrt(n1*n1+n2*n2+n3*n3);
    n1/=norm;
    n2/=norm;
    n3/=norm;

    // inflow region
    // point_on_plane_x
    double pop_x = 0.0; // here arbitrary
    double pop_y = 0.0;
    double pop_z = -2.0*alpha;

    double dist =  n1*pop_x + n2*pop_y + n3*pop_z;

    double val_plane_inflow = n1*x + n2*y + n3*z - dist;

    double val_inflow = std::max(val_plane_inflow, z-(pop_z+r*1.1));
    val_inflow = std::min(val_inflow, (z-(pop_z-r*1.1)));

    return -val_inflow;
  }
  else if(func_no == -3)
  {
    // outflow region

    double n1_out = 0.0;
    double n2_out = -2.0*PI*R;
    double n3_out = -alpha;

    double norm_out = sqrt(n1_out*n1_out+n2_out*n2_out+n3_out*n3_out);
    n1_out/=norm_out;
    n2_out/=norm_out;
    n3_out/=norm_out;

    // point_on_plane_x
    double pop_x_out = 0.0; // here arbitrary
    double pop_y_out = 0.0;
    double pop_z_out = +2.0*alpha;

    double dist_out =  n1_out*pop_x_out + n2_out*pop_y_out + n3_out*pop_z_out;

    double val_plane_outflow = n1_out*x + n2_out*y + n3_out*z - dist_out;

    double val_outflow = std::max(val_plane_outflow, -(z-(pop_z_out-r*1.1)));
    val_outflow = std::min(val_outflow, -(z-(pop_z_out+r*1.1)));

    return -val_outflow;
  }
  else if(func_no == -4)
  {
    double val_inner_ring_cyl = sqrt((x+0.2)*(x+0.2)+y*y)-0.14;
    double val_z_limit_inner = z+0.9;
    return std::max(val_inner_ring_cyl,val_z_limit_inner);
  }
  else if(func_no == -5)
  {
    double val_outer_ring_cyl = sqrt((x+0.2)*(x+0.2)+y*y)-0.22;
    double val_inner_ring_cyl = sqrt((x+0.2)*(x+0.2)+y*y)-0.14;
    double val_ring = std::max(val_outer_ring_cyl,-val_inner_ring_cyl);
    double val_z_limit_inner = z+0.9;
    double val_cylinder_ring_half = std::max(val_ring,val_z_limit_inner);
    return val_cylinder_ring_half;
  }
  else if(func_no == -6) // cylinder at inflow of a helix
  {
    double n1 = 0.0;
    double n2 = 2.0*PI*R;
    double n3 = alpha;

    double norm = sqrt(n1*n1+n2*n2+n3*n3);

    n1/=norm;
    n2/=norm;
    n3/=norm;

    // inflow region
    // point_on_plane_x
    double pop_x = 0.2; // here arbitrary
    double pop_y = 0.0;
    double pop_z = -2.0*alpha;

    double dist =  n1*pop_x + n2*pop_y + n3*pop_z;

    double val_plane_inflow = n1*x + n2*y + n3*z - dist;

    double coord_x_center = x-0.2;
    double coord_y_center = y;
    double coord_z_center = z+alpha*2.0;

    double coord_dot_n = coord_x_center*n1+coord_y_center*n2+coord_z_center*n3;

    double tmp1 = (coord_x_center-n1*coord_dot_n);
    tmp1*=tmp1;
    double tmp2 = (coord_y_center-n2*coord_dot_n);
    tmp2*=tmp2;
    double tmp3 = (coord_z_center-n3*coord_dot_n);
    tmp3*=tmp3;

    double val_cylinder = sqrt(tmp1+tmp2+tmp3)-r;
    val_cylinder = std::max(val_cylinder, val_plane_inflow);

    return val_cylinder;
  }
  else if(func_no == -7) // box for Oseen
  {
    //return -(std::max( (fabs(x-0.5+0.013))/0.3, (fabs(y-0.5+0.013))/0.3)-1.0);
    return -(std::max( (fabs(x-1.0))/0.45, std::max((fabs(y-0.5))/0.45, (fabs(z-0.5))/0.45) )-1.0);
  }


  //val = std::max(val_helix, std::max(val_inflow, val_outflow) );


////  double z_centerline = 1.0;
//  double alpha = 0.21;
//
//  double arc=0.0;
//  int n=0;
//
//  double z_max_at_arczero = r;
//
////  if(fabs(x)<1e-14 or fabs(y)<1e-14)
//////    if(fabs(x)<1e-14 and fabs(y)<1e-14)
////  {
////    val = 0.5;
////    return val;
////  }
//
//  double sgn_y = 1.0;
//
//  if(y>1e-14)
//    sgn_y= 1.0;
//  else if(y<1e-14)
//    sgn_y= -1.0;
//  else
//    sgn_y = 0.0;
//
//  double sgn_x = 1.0;
//
//  if(x>1e-14)
//    sgn_x= 1.0;
//  else if(x<1e-14)
//    sgn_x= -1.0;
//  else
//    sgn_x = 0.0;
//
//  n = 0;
//
//  if(z>=0.0)
//  {
//    // look for the first
//    for(int i = 0; ; i++)
//    {
//      if(i*alpha <= z and z<(i+1)*alpha)
//      {
//        n=i;
//        break;
//      }
//    }
//  }
//  else // z<0.0
//  {
//    // look for the first
//    for(int i = 0; ; i--)
//    {
//      if(i*alpha <= z and z<(i+1)*alpha)
//      {
//        n=i;
//        break;
//      }
//    }
//  }
//
//  // three possible i's
////  if(fabs(z-(n-1)*alpha) < fabs(z-n*alpha))
////    n--;
//
//  double arc_1 = 0.0;
//  double arc_2 = 0.0;
//  double arc_3 = 0.0;
//
//
//  if(fabs(x)>1e-14 and fabs(y)>1e-14)
//  {
//    arc_1 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +n;
//    arc_2 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n-1);
////    arc_3 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n-2);
//
//    //arc_3 = sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +(n+1);
//  }
//  arc= std::max(arc_1,arc_2);
// //   arc= std::max(arc_1,std::max(arc_2,arc_3)); //sgn_y* 1.0/(2.0*PI)*acos(sgn_x/(sqrt(1+yy/xx))) + 0.5*(1.0 - sgn_y) +n;
////  else
////  {
////    if(fabs(x)>1e-14 and fabs(y)<1e-14)
////    {
////      if(x>0)
////        arc = n;
////      else
////        arc = 0.5+n;
////    }
////    else if(fabs(x)<1e-14 and fabs(y)>1e-14)
////    {
////      if(y>0)
////        arc = 0.25+n;
////      else
////        arc = 0.75+n;
////    }
////    else
////    {
////      arc = 0.5 + n;
////    }
////  }
//
////  if(y>1e-14)
////  {
////    arc= sgn* 1.0/(2.0*PI)*acos(1.0/(sqrt(1+yy/xx))) +n;
////  }
////  else if(y < 1e-14)
////  {
////    arc= 1.0-1.0/(2.0*PI)*acos(1.0/(sqrt(1+yy/xx))) +n-1;
////  }
////  else
////  {
////    if(x > 0.0)
////      arc=n;
////    else if(x< 0.0)
////      arc=0.5+n;
////    else
////      arc=0.0;
////
////  }
//
//  double z_centerline = alpha * arc;
//
//  double tmp_1 = sqrt(xx+yy)-R;
//  double tmp_2 = z-z_centerline;
//
//  val = tmp_1*tmp_1 + tmp_2*tmp_2-r*r;

  return val;
}


XFEM::LevelSetCouplingBC::LevelSetCouplingBC(
     Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
     const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
     const int                           coupling_id,///< id of composite of coupling conditions
     const double                        time,      ///< time
     const int                           step       ///< time step
 ) : LevelSetCoupling(bg_dis, cond_name, coupling_id, time, step)
{
  has_interface_moved_ = true;
}

/*----------------------------------------------------------------------*
 | set interface level set field at current time           schott 02/15 |
 *----------------------------------------------------------------------*/
void XFEM::LevelSetCouplingBC::PrepareSolve()
{

  if(myrank_ == 0) IO::cout << "\t set level-set field, time " << time_ << IO::endl;

  has_interface_moved_ = SetLevelSetField(time_);
  return;
}



bool XFEM::LevelSetCouplingBC::HasMovingInterface()
{
  return has_interface_moved_;
}



void XFEM::LevelSetCouplingWeakDirichlet::EvaluateCouplingConditions(
    LINALG::Matrix<3,1>& ivel,
    LINALG::Matrix<3,1>& itraction,
    const LINALG::Matrix<3,1>& x,
    const DRT::Condition* cond
)
{
  // evaluate interface velocity (given by weak Dirichlet condition)
  EvaluateDirichletFunction(ivel, x, cond, time_);

  // no interface traction to be evaluated
  itraction.Clear();
}

void XFEM::LevelSetCouplingWeakDirichlet::EvaluateCouplingConditionsOldState(
    LINALG::Matrix<3,1>& ivel,
    LINALG::Matrix<3,1>& itraction,
    const LINALG::Matrix<3,1>& x,
    const DRT::Condition* cond
)
{
  // evaluate interface velocity (given by weak Dirichlet condition)
  EvaluateDirichletFunction(ivel, x, cond, time_-dt_);

  // no interface traction to be evaluated
  itraction.Clear();
}

void XFEM::LevelSetCouplingNeumann::EvaluateCouplingConditions(
    LINALG::Matrix<3,1>& ivel,
    LINALG::Matrix<3,1>& itraction,
    const LINALG::Matrix<3,1>& x,
    const DRT::Condition* cond
)
{
  // no interface velocity to be evaluated
  ivel.Clear();

  // evaluate interface traction (given by Neumann condition)
  EvaluateNeumannFunction(itraction, x, cond, time_);
}

void XFEM::LevelSetCouplingNeumann::EvaluateCouplingConditionsOldState(
    LINALG::Matrix<3,1>& ivel,
    LINALG::Matrix<3,1>& itraction,
    const LINALG::Matrix<3,1>& x,
    const DRT::Condition* cond
)
{
  // no interface velocity to be evaluated
  ivel.Clear();

  // evaluate interface traction (given by Neumann condition)
  EvaluateNeumannFunction(itraction, x, cond, time_-dt_);
}

/*----------------------------------------------------------------------*
 | Set the LevelSet Field from a two phase algorithm.                   |
 *----------------------------------------------------------------------*/
void XFEM::LevelSetCouplingTwoPhase::SetLevelSetField(
   Teuchos::RCP<const Epetra_Vector> scalaraf,
   Teuchos::RCP<const Epetra_Vector> curvatureaf,
   Teuchos::RCP<Epetra_MultiVector>  smoothed_gradphiaf,
   Teuchos::RCP<DRT::Discretization> scatradis
   )
{

  //Has the settings for surface tension been set from the Algorithm.
  // WARNING!
  //   This is not nice programming practice. In the future this info might be beneficial to put in
  //   DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS in the input file.
  if(not surfacetension_init_)
    dserror("You can't set a LevelSetField without specifying the surface tension specifications.");

  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

// CUT INFORMATION FROM LEVEL SET
  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->NumDof(0,lscatranode);
    const int globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);
    const int localscatradofid = scalaraf->Map().LID(globalscatradofid);
    if (localscatradofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    // now copy the values
    value = (*scalaraf)[localscatradofid];
    err = phinp_->ReplaceMyValue(lnodeid,0,value);
    if (err != 0) dserror("error while inserting value into phinp_");
  }

// NODAL CURVATURE!!!!!!
//----------------------------------------------
  //Transfer the vectors onto the NodeColMap.
  if(surftensapprox_==INPAR::TWOPHASE::surface_tension_approx_nodal_curvature)
  {
    //SAFETY check
    if(curvatureaf==Teuchos::null)
      dserror("Nodal curvature chosen and empty curvatureaf provided.");

      Teuchos::RCP<Epetra_Vector> curvaturenp_rownode = Teuchos::rcp(new Epetra_Vector(*cutter_dis_->NodeRowMap()));

    // loop all column nodes on the processor
    for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
    {
      // get the processor's local scatra node
      DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

      // find out the global dof id of the last(!) dof at the scatra node
      const int numscatradof = scatradis->NumDof(0,lscatranode);
      const int globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);

      const int localscatradofid = curvatureaf->Map().LID(globalscatradofid);
      if (localscatradofid < 0)
        dserror("localdofid not found in map for given globaldofid");

      // now copy the values
      value = (*curvatureaf)[localscatradofid];
      err = curvaturenp_rownode->ReplaceMyValue(lnodeid,0,value);
      if (err != 0) dserror("error while inserting value into curvaturenp_rownode");
    }

  curvaturenp_node_ = Teuchos::rcp(new Epetra_Vector(*cutter_dis_->NodeColMap()));
  LINALG::Export(*curvaturenp_rownode,*curvaturenp_node_);
  }
//---------------------------------------------- // NODAL CURVATURE END


// SMOOTHED GRAD PHI!!!!!!
//----------------------------------------------
  // SMoothed gradphi needed for divgrad option or LB with smoothed Projection matrix.
  if(surftensapprox_==INPAR::TWOPHASE::surface_tension_approx_divgrad_normal or
      (surftensapprox_==INPAR::TWOPHASE::surface_tension_approx_laplacebeltrami and
          (laplacebeltrami_==INPAR::TWOPHASE::matrix_smoothed or laplacebeltrami_==INPAR::TWOPHASE::matrix_mixed_smoothed)))
  {
    //SAFETY check
    if(smoothed_gradphiaf==Teuchos::null)
      dserror("A smoothed grad phi is required, but an empty one is provided!");

    Teuchos::RCP<Epetra_MultiVector> gradphinp_smoothed_rownode = Teuchos::rcp(new Epetra_MultiVector(*cutter_dis_->NodeRowMap(),smoothed_gradphiaf->NumVectors()));
    int numvec = smoothed_gradphiaf->NumVectors();

    // loop all column nodes on the processor
    for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
    {
      // get the processor's local scatra node
      DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

      // find out the global dof id of the last(!) dof at the scatra node
      const int numscatradof = scatradis->NumDof(0,lscatranode);
      const int globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);

      const int localscatradofid = smoothed_gradphiaf->Map().LID(globalscatradofid);
      if (localscatradofid < 0)
        dserror("localdofid not found in map for given globaldofid");

      // now copy the values
      for(int i=0; i<numvec; i++)
      {
        value = smoothed_gradphiaf->Pointers()[i][localscatradofid]; //Somehow it is turned around?
        err = gradphinp_smoothed_rownode->ReplaceMyValue(lnodeid,i,value);
        if (err != 0) dserror("error while inserting value into gradphinp_smoothed_rownode");
      }
    }

    gradphinp_smoothed_node_ = Teuchos::rcp(new Epetra_MultiVector(*cutter_dis_->NodeColMap(),smoothed_gradphiaf->NumVectors()));
    LINALG::Export(*gradphinp_smoothed_rownode,*gradphinp_smoothed_node_);
  }
//---------------------------------------------- // SMOOTHED GRAD PHI END

  //SAFETY CHECK
//----------------------------------------------
  //Both empty vectors sent.
  if (curvatureaf!=Teuchos::null and smoothed_gradphiaf!=Teuchos::null)
    if(not (surftensapprox_== INPAR::TWOPHASE::surface_tension_approx_laplacebeltrami
            and laplacebeltrami_ == INPAR::TWOPHASE::matrix_non_smoothed )
            and !(surftensapprox_==INPAR::TWOPHASE::surface_tension_approx_fixed_curvature)
            and !(surftensapprox_==INPAR::TWOPHASE::surface_tension_approx_none)
            )
    {
      dserror("You can not both have a nodal curvature and a smoothed_gradphinp empty at once with this configuration.");
    }

//---------------------------------------------- // SAFETY CHECK

  //  //Transfer the vectors onto the DofColMap.
//  if(curvatureaf!=Teuchos::null)
//  {
//    curvaturenp_ = Teuchos::rcp(new Epetra_Vector(*scatradis->DofColMap()));
//    LINALG::Export(*curvatureaf,*curvaturenp_);
//  }
//  if(smoothed_gradphiaf!=Teuchos::null)
//  {
//    gradphinp_smoothed_ = Teuchos::rcp(new Epetra_MultiVector(*scatradis->DofColMap(),smoothed_gradphiaf->NumVectors()));
//    LINALG::Export(*smoothed_gradphiaf,*gradphinp_smoothed_);
//  }
//  if (curvaturenp_!=Teuchos::null and gradphinp_smoothed_!=Teuchos::null)
//    dserror("You can not have both a nodal curvature and a gradphinp prescribed at once.");


  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::LevelSetCouplingTwoPhase::GetInterfaceSlaveMaterial(
  DRT::Element* actele,
  Teuchos::RCP<MAT::Material> & mat
)
{
  XFEM::UTILS::GetVolumeCellMaterial(actele,mat,GEO::CUT::Point::inside);
}


// -------------------------------------------------------------------
// Read Restart data for ScaTra coupled level set
// -------------------------------------------------------------------
void XFEM::LevelSetCouplingTwoPhase::ReadRestart(
    const int step,
    const int lsc_idx
)
{

  //-------- boundary discretization
  IO::DiscretizationReader boundaryreader(cutter_dis_, step);

  const double time = boundaryreader.ReadDouble("time");

  if(myrank_ == 0)
  {
    IO::cout << "           RESTART IS PERFORMED FROM STORED VALUES!                            " << IO::endl;
    IO::cout << "ReadRestart for Level Set Cut in Xfluid (time="<< time <<" ; step="<< step <<")" << IO::endl;
  }

  std::ostringstream temp;
    temp << lsc_idx;
    std::string name = "phinp_"+temp.str();

  boundaryreader.ReadVector(phinp_,   name);

  if (not (cutter_dis_->NodeRowMap())->SameAs(phinp_->Map()))
    dserror("Global node numbering in maps does not match");

}
