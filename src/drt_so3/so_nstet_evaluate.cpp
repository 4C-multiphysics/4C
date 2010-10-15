/*!----------------------------------------------------------------------*
\file so_nstet_evaluate.cpp

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"
#include "../linalg/linalg_utils.H"
#include "Epetra_SerialDenseSolver.h"

#include "../drt_mat/micromaterial.H"
#include "../drt_mat/stvenantkirchhoff.H"
#include "../drt_mat/lung_penalty.H"
#include "../drt_mat/lung_ogden.H"
#include "../drt_mat/neohooke.H"
#include "../drt_mat/anisotropic_balzani.H"
#include "../drt_mat/aaaneohooke.H"
#include "../drt_mat/mooneyrivlin.H"

#include "so_nstet.H"

using namespace std;

/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)              gee 05/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet::InitElement()
{
  LINALG::Matrix<4,3> xrefe;
  LINALG::Matrix<4,3+1> J;
  {
    // compute element volume
    DRT::Node** nodes = Nodes();
    for (int i=0; i<4; ++i)
    {
      const double* x = nodes[i]->X();
      J(i,0) = 1.0;
      J(i,1) = xrefe(i,0) = x[0];
      J(i,2) = xrefe(i,1) = x[1];
      J(i,3) = xrefe(i,2) = x[2];
    }
    V_ = J.Determinant()/6.0;
    if (V_==0.0)     dserror("Element volume is zero");
    else if (V_<0.0) dserror("Element volume is negative");
  }
  // compute derivatives of shape functions w.r.t. to material coords.
  {
    // one gauss point at 0.25/0.25/0.25
    // gauss point weight is 1.0, so skip it
    const double gploc = 0.25;
    LINALG::Matrix<4,1> funct;
    ShapeFunction(funct,gploc,gploc,gploc,gploc);
    LINALG::Matrix<4,4> deriv;
    ShapeFunctionDerivatives(deriv);
    LINALG::Matrix<4-1,4> tmp;
    tmp.MultiplyTN(xrefe,deriv);
    for (int i=0; i<4; i++) J(0,i)=1;
    for (int row=0;row<3;row++)
      for (int col=0;col<4;col++)
        J(row+1,col)=tmp(row,col);

    LINALG::Matrix<4,3> Iaug(true); // initialize to zero
    Iaug(1,0)=1;
    Iaug(2,1)=1;
    Iaug(3,2)=1;
    LINALG::Matrix<4,3> partials;
    LINALG::FixedSizeSerialDenseSolver<4,4,3> solver;
    solver.SetMatrix(J);
    solver.SetVectors(partials,Iaug);
    solver.FactorWithEquilibration(true);
    int err  = solver.Factor();
    int err2 = solver.Solve();
    if (err || err2) dserror("Inversion of Jacobian failed");
    /* structure of nxyz_:
    **             [   dN_1     dN_1     dN_1   ]
    **             [  ------   ------   ------  ]
    **             [    dX       dY       dZ    ]
    **    nxyz_ =  [     |        |        |    ]
    **             [                            ]
    **             [   dN_4     dN_4     dN_4   ]
    **             [  -------  -------  ------- ]
    **             [    dX       dY       dZ    ]
    */
    nxyz_.Multiply(deriv,partials);
  }


  return;
} // DRT::ELEMENTS::NStet::InitElement



/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              gee 05/08|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::NStet::Evaluate(ParameterList& params,
                                  DRT::Discretization&      discretization,
                                  vector<int>&              lm,
                                  Epetra_SerialDenseMatrix& elemat1_epetra,
                                  Epetra_SerialDenseMatrix& elemat2_epetra,
                                  Epetra_SerialDenseVector& elevec1_epetra,
                                  Epetra_SerialDenseVector& elevec2_epetra,
                                  Epetra_SerialDenseVector& elevec3_epetra)
{
  LINALG::Matrix<12,12> elemat1(elemat1_epetra.A(),true);
  LINALG::Matrix<12,12> elemat2(elemat2_epetra.A(),true);
  LINALG::Matrix<12, 1> elevec1(elevec1_epetra.A(),true);
  LINALG::Matrix<12, 1> elevec2(elevec2_epetra.A(),true);
  // start with "none"
  DRT::ELEMENTS::NStet::ActionType act = NStet::none;

  // get the required action
  string action = params.get<string>("action","none");
  if (action == "none") dserror("No action supplied");
  else if (action=="calc_struct_linstiff")                act = NStet::calc_struct_linstiff;
  else if (action=="calc_struct_nlnstiff")                act = NStet::calc_struct_nlnstiff;
  else if (action=="calc_struct_internalforce")           act = NStet::calc_struct_internalforce;
  else if (action=="calc_struct_linstiffmass")            act = NStet::calc_struct_linstiffmass;
  else if (action=="calc_struct_nlnstiffmass")            act = NStet::calc_struct_nlnstiffmass;
  else if (action=="calc_struct_nlnstifflmass")           act = NStet::calc_struct_nlnstifflmass;
  else if (action=="calc_struct_stress")                  act = NStet::calc_struct_stress;
  else if (action=="postprocess_stress")                  act = NStet::postprocess_stress;
  else if (action=="calc_struct_eleload")                 act = NStet::calc_struct_eleload;
  else if (action=="calc_struct_fsiload")                 act = NStet::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")            act = NStet::calc_struct_update_istep;
  else if (action=="calc_struct_update_imrlike")          act = NStet::calc_struct_update_imrlike;
  else if (action=="calc_struct_reset_istep")             act = NStet::calc_struct_reset_istep;
  else dserror("Unknown type of action for NStet");

  // what should the element do
  switch(act)
  {
    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass:
    {
      // need current displacement and residual forces
      RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      RCP<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      nstetnlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,
                        NULL,NULL,INPAR::STR::stress_none,INPAR::STR::strain_none);
      if (act==calc_struct_nlnstifflmass) nstetlumpmass(&elemat2);
    }
    break;

    // nonlinear stiffness and internal force vector
    case calc_struct_nlnstiff:
    {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      LINALG::Matrix<12,12>* elemat1ptr = NULL;
      if (elemat1.IsInitialized()) elemat1ptr = &elemat1;
      nstetnlnstiffmass(lm,mydisp,myres,elemat1ptr,NULL,&elevec1,
                        NULL,NULL,INPAR::STR::stress_none,INPAR::STR::strain_none);
    }
    break;

    // evaluate stresses and strains at gauss points
    case calc_struct_stress:
    {
      // nothing to do for ghost elements
      if (discretization.Comm().MyPID()==Owner())
      {
        //------------------------------- compute element stress from stabilization
        RCP<vector<char> > stressdata = params.get<RCP<vector<char> > >("stress", null);
        RCP<vector<char> > straindata = params.get<RCP<vector<char> > >("strain", null);
        if (stressdata==null) dserror("Cannot get stress 'data'");
        if (straindata==null) dserror("Cannot get strain 'data'");
        INPAR::STR::StressType iostress = params.get<INPAR::STR::StressType>("iostress",INPAR::STR::stress_none);
        INPAR::STR::StrainType iostrain = params.get<INPAR::STR::StrainType>("iostrain",INPAR::STR::strain_none);
        RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
        RCP<const Epetra_Vector> res  = discretization.GetState("residual displacement");
        if (disp==null) dserror("Cannot get state vectors 'displacement'");
        vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
        vector<double> myres(lm.size());
        DRT::UTILS::ExtractMyValues(*res,myres,lm);
        LINALG::Matrix<1,6> stress(true); 
        LINALG::Matrix<1,6> strain(true);
        LINALG::Matrix<1,6> elestress(true);
        LINALG::Matrix<1,6> elestrain(true);
        nstetnlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&elestress,&elestrain,iostress,iostrain);
        
        //--------------------------------- interpolate nodal stress from every node
        Teuchos::RCP<Epetra_MultiVector> nodestress = ElementType().nstress_;
        Teuchos::RCP<Epetra_MultiVector> nodestrain = ElementType().nstrain_;
        const int numnode = NumNode();
        for (int i=0; i<numnode; ++i)
        {
          const int gid = Nodes()[i]->Id();
          const int lid = nodestress->Map().LID(gid);
          if (lid==-1) dserror("Cannot find matching nodal stresses/strains");
          for (int j=0; j<6; ++j)
          {
            stress(0,j) += (*(*nodestress)(j))[lid];
            strain(0,j) += (*(*nodestrain)(j))[lid];
          }
        }
        for (int j=0; j<6; ++j) 
        {
          stress(0,j) /= numnode;
          strain(0,j) /= numnode;
        }
        
        //-------------------------------------------------------- add element stress
        for (int j=0; j<6; ++j) 
        {
          stress(0,j) += elestress(0,j);
          strain(0,j) += elestrain(0,j);
        }
        
        //------------------------------------------------- add stress from MIS nodes
        Teuchos::RCP<Epetra_MultiVector> mis_stress = ElementType().pstab_nstress_;
        Teuchos::RCP<Epetra_MultiVector> mis_strain = ElementType().pstab_nstrain_;
        map<int,vector<int> >::iterator ele = ElementType().pstab_cid_mis_.find(Id());
        if (ele == ElementType().pstab_cid_mis_.end()) dserror("Cannot find this element");
        map<int,vector<double> >::iterator elew = ElementType().pstab_cid_mis_weight_.find(Id());
        if (elew == ElementType().pstab_cid_mis_weight_.end()) dserror("Cannot find this element weight");
        vector<int>& mis = ele->second;
        const int nummis = (int)mis.size();
        if (nummis < 1) dserror("Element not associated with any mis node");
        vector<double>& misw = elew->second;
        const int nummisw = (int)misw.size();
        if (nummis != nummisw) dserror("Number of patches and weight mismatch");
        //double totweight = 0.0;
        for (int i=0; i<nummis; ++i)
        {
          const int gid = mis[i];
          const int lid = ElementType().pstab_nstress_->Map().LID(gid);
          const double weight = misw[i];
          //totweight += weight;
          for (int j=0; j<6; ++j)
          {
            stress(0,j) += weight * (*(*mis_stress)(j))[lid];
            strain(0,j) += weight * (*(*mis_strain)(j))[lid];
          }
        } // for (int i=0; i<nummis; ++i)
        //printf("Proc %d Ele %d TotWeight %15.10e\n",Owner(),Id(),totweight);
        
        //----------------------------------------------- add final stress to storage
        AddtoPack(*stressdata, stress);
        AddtoPack(*straindata, strain);
      } // if (discretization.Comm().MyPID()==Owner())
    }
    break;

    // postprocess stresses/strains at gauss points

    // note that in the following, quantities are always referred to as
    // "stresses" etc. although they might also apply to strains
    // (depending on what this routine is called for from the post filter)
    case postprocess_stress:
    {
      // nothing to do for ghost elements
      if (discretization.Comm().MyPID()==Owner())
      {
        const RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > gpstressmap=
          params.get<RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > >("gpstressmap",null);
        if (gpstressmap==null) dserror("no gp stress/strain map available for postprocessing");
        string stresstype = params.get<string>("stresstype","ndxyz");

        const int gid = Id();
        LINALG::Matrix<1,6> gpstress(((*gpstressmap)[gid])->A(),true);

        RCP<Epetra_MultiVector> poststress=params.get<RCP<Epetra_MultiVector> >("poststress",null);
        if (poststress==null) dserror("No element stress/strain vector available");

        if (stresstype=="ndxyz")
        {
          for (int i=0; i<NumNode(); ++i)
          {
            const int gid = Nodes()[i]->Id();
            const int lid = poststress->Map().LID(gid);
            if (lid==-1) dserror("Cannot find local id for global id");
            const int numadjele = Nodes()[i]->NumElement();
            for (int j=0; j<6; ++j)
              (*((*poststress)(j)))[lid] += gpstress(0,j) / numadjele;
          }
        }
        else if (stresstype=="cxyz")
        {
          const Epetra_BlockMap elemap = poststress->Map();
          int lid = elemap.LID(Id());
          if (lid!=-1)
          {
            for (int i=0; i<6; ++i) (*((*poststress)(i)))[lid] = gpstress(0,i);
          }
        }
        else
          dserror("unknown type of stress/strain output on element level");
      } // if (discretization.Comm().MyPID()==Owner())
    }
    break;

    case calc_struct_eleload:
      dserror("this class is not supposed to evaluate a load, use EvaluateNeumann(...)");
    break;

    case calc_struct_fsiload:
      dserror("Case not yet implemented");
    break;

    case calc_struct_update_istep:
    {
      ;// there is nothing to do here at the moment
    }
    break;

    case calc_struct_update_imrlike:
    {
      ;// there is nothing to do here at the moment
    }
    break;

    case calc_struct_reset_istep:
    {
      ;// there is nothing to do here at the moment
    }
    break;

    // internal force vector only
    case calc_struct_internalforce:
    {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      LINALG::Matrix<12,12> myemat(true);
      nstetnlnstiffmass(lm,mydisp,myres,&myemat,NULL,&elevec1,
                        NULL,NULL,INPAR::STR::stress_none,INPAR::STR::strain_none);
    }
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not implemented");
    break;

    // linear stiffness
    case calc_struct_linstiff:
      dserror("action calc_struct_linstiff currently not supported");
    break;

    default:
      dserror("Unknown type of action for NStet");
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  evaluate the element (private)                             gee 05/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet::nstetnlnstiffmass(
      vector<int>&                     lm,             // location matrix
      vector<double>&                  disp,           // current displacements
      vector<double>&                  residual,       // current residuum
      LINALG::Matrix<12,12>*           stiffmatrix,    // element stiffness matrix
      LINALG::Matrix<12,12>*           massmatrix,     // element mass matrix
      LINALG::Matrix<12, 1>*           force,          // stress output options
      LINALG::Matrix<1,6>*             elestress,      // stress output
      LINALG::Matrix<1,6>*             elestrain,      // strain output
      const INPAR::STR::StressType     iostress,       // type of stress
      const INPAR::STR::StrainType     iostrain)       // type of strain
{
  //--------------------------------------------------- geometry update
  LINALG::Matrix<3,3> defgrd(false);
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      defgrd(i,j) = F()[i][j].val();

  //--------------------------- Right Cauchy-Green tensor C = = F^T * F
  LINALG::Matrix<3,3> cauchygreen;
  cauchygreen.MultiplyTN(defgrd,defgrd);

  // --Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  LINALG::Matrix<6,1> glstrain;
  glstrain(0) = 0.5 * (cauchygreen(0,0) - 1.0);
  glstrain(1) = 0.5 * (cauchygreen(1,1) - 1.0);
  glstrain(2) = 0.5 * (cauchygreen(2,2) - 1.0);
  glstrain(3) = cauchygreen(0,1);
  glstrain(4) = cauchygreen(1,2);
  glstrain(5) = cauchygreen(2,0);

  //------------------------------------ B=operator ( same as in hex8 !)
  /*
  ** B = F : N,xyz
  **
  **      [ ... | F_11*N_{,1}^k  F_21*N_{,1}^k  F_31*N_{,1}^k | ... ]
  **      [ ... | F_12*N_{,2}^k  F_22*N_{,2}^k  F_32*N_{,2}^k | ... ]
  **      [ ... | F_13*N_{,3}^k  F_23*N_{,3}^k  F_33*N_{,3}^k | ... ]
  ** B =  [ ~~~   ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~   ~~~ ]
  **      [       F_11*N_{,2}^k+F_12*N_{,1}^k                       ]
  **      [ ... |          F_21*N_{,2}^k+F_22*N_{,1}^k        | ... ]
  **      [                       F_31*N_{,2}^k+F_32*N_{,1}^k       ]
  **      [                                                         ]
  **      [       F_12*N_{,3}^k+F_13*N_{,2}^k                       ]
  **      [ ... |          F_22*N_{,3}^k+F_23*N_{,2}^k        | ... ]
  **      [                       F_32*N_{,3}^k+F_33*N_{,2}^k       ]
  **      [                                                         ]
  **      [       F_13*N_{,1}^k+F_11*N_{,3}^k                       ]
  **      [ ... |          F_23*N_{,1}^k+F_21*N_{,3}^k        | ... ]
  **      [                       F_33*N_{,1}^k+F_31*N_{,3}^k       ]
  */

  // 6x12 n_stresses * number degrees of freedom per element
  LINALG::Matrix<6,12> bop;
  for (int i=0; i<4; i++)
  {
    bop(0,3*i+0) = defgrd(0,0)*nxyz_(i,0);
    bop(0,3*i+1) = defgrd(1,0)*nxyz_(i,0);
    bop(0,3*i+2) = defgrd(2,0)*nxyz_(i,0);
    bop(1,3*i+0) = defgrd(0,1)*nxyz_(i,1);
    bop(1,3*i+1) = defgrd(1,1)*nxyz_(i,1);
    bop(1,3*i+2) = defgrd(2,1)*nxyz_(i,1);
    bop(2,3*i+0) = defgrd(0,2)*nxyz_(i,2);
    bop(2,3*i+1) = defgrd(1,2)*nxyz_(i,2);
    bop(2,3*i+2) = defgrd(2,2)*nxyz_(i,2);
    /* ~~~ */
    bop(3,3*i+0) = defgrd(0,0)*nxyz_(i,1) + defgrd(0,1)*nxyz_(i,0);
    bop(3,3*i+1) = defgrd(1,0)*nxyz_(i,1) + defgrd(1,1)*nxyz_(i,0);
    bop(3,3*i+2) = defgrd(2,0)*nxyz_(i,1) + defgrd(2,1)*nxyz_(i,0);
    bop(4,3*i+0) = defgrd(0,1)*nxyz_(i,2) + defgrd(0,2)*nxyz_(i,1);
    bop(4,3*i+1) = defgrd(1,1)*nxyz_(i,2) + defgrd(1,2)*nxyz_(i,1);
    bop(4,3*i+2) = defgrd(2,1)*nxyz_(i,2) + defgrd(2,2)*nxyz_(i,1);
    bop(5,3*i+0) = defgrd(0,2)*nxyz_(i,0) + defgrd(0,0)*nxyz_(i,2);
    bop(5,3*i+1) = defgrd(1,2)*nxyz_(i,0) + defgrd(1,0)*nxyz_(i,2);
    bop(5,3*i+2) = defgrd(2,2)*nxyz_(i,0) + defgrd(2,0)*nxyz_(i,2);
  }

  //------------------------------------------------- call material law
#if 1 // dev stab on cauchy stresses
  LINALG::Matrix<6,6> cmat(true); // Views
  LINALG::Matrix<6,1> stress(true);
  LINALG::Matrix<6,1> glstrainbar(false);
  double density = -999.99;
  {
    // do deviatoric F, C, E
    const double J = defgrd.Determinant();
    LINALG::Matrix<3,3> Cbar(cauchygreen);
    Cbar.Scale(pow(J,-2./3.));
    glstrainbar(0) = 0.5 * (Cbar(0,0) - 1.0);
    glstrainbar(1) = 0.5 * (Cbar(1,1) - 1.0);
    glstrainbar(2) = 0.5 * (Cbar(2,2) - 1.0);
    glstrainbar(3) = Cbar(0,1);
    glstrainbar(4) = Cbar(1,2);
    glstrainbar(5) = Cbar(2,0);
    LINALG::Matrix<3,3> Fbar(false);
    Fbar.SetCopy(defgrd.A());
    Fbar.Scale(pow(J,-1./3.));

    SelectMaterial(stress,cmat,density,glstrainbar,Fbar,0);

    // define stuff we need to do the split
    LINALG::Matrix<6,6> cmatdev; // set to zero?
    LINALG::Matrix<6,1> stressdev;

    // do just the deviatoric components
    NStetType::DevStressTangent(stressdev,cmatdev,cmat,stress,cauchygreen);
    stress = stressdev;
    cmat = cmatdev;
    stress.Scale(ALPHA_NSTET);
    cmat.Scale(ALPHA_NSTET);
  }
#endif

#if 0 // original puso tet
  Epetra_SerialDenseMatrix cmat(6,6);
  Epetra_SerialDenseVector stress(6);
  double density = -999.99;
  SelectMaterial(stress,cmat,density,glstrain,defgrd,0);
  stress.Scale(ALPHA_NSTET);
  cmat.Scale(ALPHA_NSTET);
#endif

  //---------------------------------------------- output of stress and strain
  {
    switch (iostrain)
    {
    case INPAR::STR::strain_gl:
    {
      if (elestrain == NULL) dserror("no strain data available");
      for (int i = 0; i < 3; ++i)
        (*elestrain)(0,i) = ALPHA_NSTET * glstrainbar(i);
      for (int i = 0; i < 3; ++i)
        (*elestrain)(0,i) = ALPHA_NSTET * 0.5 * glstrainbar(i);
    }
    break;
    case INPAR::STR::strain_ea:
    {
      if (elestrain == NULL) dserror("no strain data available");
      LINALG::Matrix<3,3> gl;
      gl(0,0) = glstrainbar(0);      // divide off-diagonals by 2
      gl(0,1) = 0.5*glstrainbar(3);
      gl(0,2) = 0.5*glstrainbar(5);
      gl(1,0) = gl(0,1);
      gl(1,1) = glstrainbar(1);
      gl(1,2) = 0.5*glstrainbar(4);
      gl(2,0) = gl(0,2);
      gl(2,1) = gl(1,2);
      gl(2,2) = glstrainbar(2);
      
      LINALG::Matrix<3,3> Fbar(false);
      Fbar.SetCopy(defgrd.A());
      Fbar.Scale(pow(defgrd.Determinant(),-1./3.));
      LINALG::Matrix<3,3> invdefgrd;
      invdefgrd.Invert(Fbar);
      
      LINALG::Matrix<3,3> temp;
      LINALG::Matrix<3,3> euler_almansi;
      temp.Multiply(gl,invdefgrd);
      euler_almansi.MultiplyTN(invdefgrd,temp);
      
      (*elestrain)(0,0) = ALPHA_NSTET * euler_almansi(0,0);
      (*elestrain)(0,1) = ALPHA_NSTET * euler_almansi(1,1);
      (*elestrain)(0,2) = ALPHA_NSTET * euler_almansi(2,2);
      (*elestrain)(0,3) = ALPHA_NSTET * euler_almansi(0,1);
      (*elestrain)(0,4) = ALPHA_NSTET * euler_almansi(1,2);
      (*elestrain)(0,5) = ALPHA_NSTET * euler_almansi(0,2);
    }
    break;
    case INPAR::STR::strain_none:
    break;
    default:
      dserror("requested strain option not available");
    }


    switch (iostress)
    {
    case INPAR::STR::stress_2pk:
    {
      if (elestress == NULL) dserror("no stress data available");
      for (int i = 0; i<6; ++i)
        (*elestress)(0,i) = stress(i); // ALPHA_NSTET scaling is already in stress
    }
    break;
    case INPAR::STR::stress_cauchy:
    {
      if (elestress == NULL) dserror("no stress data available");

      LINALG::Matrix<3,3> pkstress;
      pkstress(0,0) = stress(0);        // ALPHA_NSTET scaling is already in stress
      pkstress(0,1) = stress(3);
      pkstress(0,2) = stress(5);
      pkstress(1,0) = pkstress(0,1);
      pkstress(1,1) = stress(1);
      pkstress(1,2) = stress(4);
      pkstress(2,0) = pkstress(0,2);
      pkstress(2,1) = pkstress(1,2);
      pkstress(2,2) = stress(2);

      LINALG::Matrix<3,3> Fbar(false);
      Fbar.SetCopy(defgrd.A());
      Fbar.Scale(pow(defgrd.Determinant(),-1./3.));

      LINALG::Matrix<3,3> temp;
      LINALG::Matrix<3,3> cauchystress;
      temp.Multiply(1.0/Fbar.Determinant(),Fbar,pkstress,0.);
      cauchystress.MultiplyNT(temp,Fbar);

      (*elestress)(0,0) = cauchystress(0,0);
      (*elestress)(0,1) = cauchystress(1,1);
      (*elestress)(0,2) = cauchystress(2,2);
      (*elestress)(0,3) = cauchystress(0,1);
      (*elestress)(0,4) = cauchystress(1,2);
      (*elestress)(0,5) = cauchystress(0,2);
    }
    break;
    case INPAR::STR::stress_none:
      break;
    default:
      dserror("requested stress type not available");
    }
  }


  //----------------------------------------------- internal force and tangent
  if (force)
  {
    // integrate internal force vector f = f + (B^T . sigma) * V_
    force->MultiplyTN(V_,bop,stress,1.0);
  }
  if (stiffmatrix)
  {
    // integrate elastic stiffness matrix
    // keu = keu + (B^T . C . B) * V_
    LINALG::Matrix<6,12> cb;
    cb.Multiply(cmat,bop);
    stiffmatrix->MultiplyTN(V_,bop,cb,1.0);
    // integrate `geometric' stiffness matrix and add to keu
    double sBL[3];
    const double V = Vol();
    for (int i=0; i<4; ++i)
    {
      sBL[0] = V*(stress(0) * nxyz_(i,0) + stress(3) * nxyz_(i,1) + stress(5) * nxyz_(i,2));
      sBL[1] = V*(stress(3) * nxyz_(i,0) + stress(1) * nxyz_(i,1) + stress(4) * nxyz_(i,2));
      sBL[2] = V*(stress(5) * nxyz_(i,0) + stress(4) * nxyz_(i,1) + stress(2) * nxyz_(i,2));
      for (int j=0; j<4; ++j)
      {
        double BsB = 0.0;
        for (int dim=0; dim<3; ++dim)
          BsB += nxyz_(j,dim) * sBL[dim];
        (*stiffmatrix)(3*i+0,3*j+0) += BsB;
        (*stiffmatrix)(3*i+1,3*j+1) += BsB;
        (*stiffmatrix)(3*i+2,3*j+2) += BsB;
      }
    }

  } // if (force && stiffmatrix)

  //------------------------------------------ do mass matrix if desired
  if (massmatrix)
  {
    // for mass matrix use a 4 gauss points integration:
    // ( 1 gauss point is not enough!)
    const double alpha  = (5.0 + 3.0*sqrt(5.0))/20.0;
    const double beta   = (5.0 - sqrt(5.0))/20.0;
    const double weight = 0.25;
    const double V = Vol();
    double xsi[4][4];
    xsi[0][0] = alpha;   xsi[0][1] = beta ;   xsi[0][2] = beta ;   xsi[0][3] = beta ;
    xsi[1][0] = beta ;   xsi[1][1] = alpha;   xsi[1][2] = beta ;   xsi[1][3] = beta ;
    xsi[2][0] = beta ;   xsi[2][1] = beta ;   xsi[2][2] = alpha;   xsi[2][3] = beta ;
    xsi[3][0] = beta ;   xsi[3][1] = beta ;   xsi[3][2] = beta ;   xsi[3][3] = alpha;
    for (int gp=0; gp<4; ++gp)
    {
      LINALG::Matrix<4,1> funct;
      ShapeFunction(funct,xsi[gp][0],xsi[gp][1],xsi[gp][2],xsi[gp][3]);
      const double f = density * V * weight;
      for (int i=0; i<4; ++i)
        for (int j=0; j<4; ++j)
        {
          const double fac = funct(i) * funct(j) * f;
          (*massmatrix)(3*i+0,3*j+0) += fac;
          (*massmatrix)(3*i+1,3*j+1) += fac;
          (*massmatrix)(3*i+2,3*j+2) += fac;
        }
    }
  } // if (massmatrix)

  return;
} // DRT::ELEMENTS::NStet::nstetnlnstiffmass


/*----------------------------------------------------------------------*
 |  lump mass matrix                                         bborn 07/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet::nstetlumpmass(LINALG::Matrix<12,12>* emass)
{
  // lump mass matrix
  if (emass != NULL)
  {
    // we assume #elemat2 is a square matrix
    for (unsigned c=0; c<(*emass).N(); ++c)  // parse columns
    {
      double d = 0.0;
      for (unsigned r=0; r<(*emass).M(); ++r)  // parse rows
      {
        d += (*emass)(r,c);  // accumulate row entries
        (*emass)(r,c) = 0.0;
      }
      (*emass)(c,c) = d;  // apply sum of row entries on diagonal
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 | material laws for NStet (protected)                          gee 10/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet::SelectMaterial(
                      LINALG::Matrix<6,1>& stress,
                      LINALG::Matrix<6,6>& cmat,
                      double& density,
                      LINALG::Matrix<6,1>& glstrain,
                      LINALG::Matrix<3,3>& defgrd,
                      int gp)
{
  Epetra_SerialDenseVector stress_e(::View,stress.A(),stress.Rows());
  Epetra_SerialDenseMatrix cmat_e(::View,cmat.A(),cmat.Rows(),cmat.Rows(),cmat.Columns());
  const Epetra_SerialDenseVector glstrain_e(::View,glstrain.A(),glstrain.Rows());
  //Epetra_SerialDenseMatrix defgrd_e(View,defgrd.A(),defgrd.Rows(),defgrd.Rows(),defgrd.Columns());


  RCP<MAT::Material> mat = Material();
  switch (mat->MaterialType())
  {
    case INPAR::MAT::m_stvenant: /*------------------ st.venant-kirchhoff-material */
    {
      MAT::StVenantKirchhoff* stvk = static_cast<MAT::StVenantKirchhoff*>(mat.get());
      stvk->Evaluate(&glstrain_e,&cmat_e,&stress_e);
      density = stvk->Density();
    }
    break;
    case INPAR::MAT::m_neohooke: /*----------------- NeoHookean Material */
    {
      MAT::NeoHooke* neo = static_cast<MAT::NeoHooke*>(mat.get());
      neo->Evaluate(&glstrain_e,&cmat_e,&stress_e);
      density = neo->Density();
    }
    break;
    case INPAR::MAT::m_aaaneohooke: /*-- special case of generalised NeoHookean material see Raghavan, Vorp */
    {
      MAT::AAAneohooke* aaa = static_cast<MAT::AAAneohooke*>(mat.get());
      aaa->Evaluate(&glstrain_e,&cmat_e,&stress_e);
      density = aaa->Density();
    }
    break;
    case INPAR::MAT::m_lung_ogden: /* lung tissue material with Ogden for volumetric part */
    {
      MAT::LungOgden* lungog = static_cast <MAT::LungOgden*>(mat.get());
      lungog->Evaluate(&glstrain,&cmat,&stress);
      density = lungog->Density();
      return;
      break;
    }
    case INPAR::MAT::m_lung_penalty: /* lung tissue material with penalty function for incompressibility constraint */
    {
      MAT::LungPenalty* lungpen = static_cast <MAT::LungPenalty*>(mat.get());

      lungpen->Evaluate(&glstrain,&cmat,&stress);

      density = lungpen->Density();
      return;
      break;
    }
    default:
      dserror("Illegal type %d of material for element NStet tet4", mat->MaterialType());
    break;
  }

  /*--------------------------------------------------------------------*/
  return;
}  // DRT::ELEMENTS::NStet::SelectMaterial



/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)     gee 05/08|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::NStet::EvaluateNeumann(ParameterList& params,
                                         DRT::Discretization&      discretization,
                                         DRT::Condition&           condition,
                                         vector<int>&              lm,
                                         Epetra_SerialDenseVector& elevec1,
                                         Epetra_SerialDenseMatrix* elemat1)
{
  dserror("DRT::ELEMENTS::NStet::EvaluateNeumann not implemented");
  return -1;
} // DRT::ELEMENTS::NStet::EvaluateNeumann







#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3
