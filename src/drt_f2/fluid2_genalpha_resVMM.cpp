/*----------------------------------------------------------------------*/
/*!
\file fluid2_genalpha_resVMM.cpp

\brief Internal implementation of Fluid2 element with a generalised alpha
       time integration.

<pre>
Maintainer: Peter Gamnitzer
            gamnitzer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15235
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef D_FLUID2
#ifdef CCADISCRET

#include "fluid2_genalpha_resVMM.H"
#include "../drt_mat/newtonianfluid.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"

#include <Epetra_SerialDenseSolver.h>
#include <Epetra_LAPACK.h>


/*----------------------------------------------------------------------*
  |  constructor allocating arrays whose sizes may depend on the number |
  | of nodes of the element                                             |
  |                            (public)                      gammi 02/08|
  *----------------------------------------------------------------------*/
DRT::ELEMENTS::Fluid2GenalphaResVMM::Fluid2GenalphaResVMM(int iel)
  : iel_        (iel),
// nodal data
//-----------------------+------------+------------------------------------
//                  dim  | derivative | node
    xyze_         (  2   ,              iel_,blitz::ColumnMajorArray<2>()),
    weights_      (                     iel_                             ),
    edeadaf_      (  2   ,              iel_,blitz::ColumnMajorArray<2>()),
//-----------------------+------------+------------------------------------
// gausspoint data
//------------------------------------------------------------------------
//                  dim  | derivative | node
//-----------------------+------------+------------------------------------
    funct_        (                     iel_                             ),
    deriv_        (            2      , iel_,blitz::ColumnMajorArray<2>()),
    deriv2_       (            3      , iel_,blitz::ColumnMajorArray<2>()),
    derxy_        (            2      , iel_,blitz::ColumnMajorArray<2>()),
    derxy2_       (            3      , iel_,blitz::ColumnMajorArray<2>()),
    viscs2_       (  2   ,     2      , iel_,blitz::ColumnMajorArray<3>()),
    xjm_          (  2   ,     2            ,blitz::ColumnMajorArray<2>()),
    xji_          (  2   ,     2            ,blitz::ColumnMajorArray<2>()),
    // for xder2, dim and derivative are interchanged
    xder2_        (  3   ,     2            ,blitz::ColumnMajorArray<2>()),
    accintam_     (  2                                                   ),
    velintnp_     (  2                                                   ),
    velintaf_     (  2                                                   ),
    ugrid_af_     (  2                                                   ),
    pderxynp_     (  2                                                   ),
    vderxynp_     (  2   ,     2            ,blitz::ColumnMajorArray<2>()),
    vderxyaf_     (  2   ,     2            ,blitz::ColumnMajorArray<2>()),
    vderxy2af_    (  2   ,     3            ,blitz::ColumnMajorArray<2>()),
    bodyforceaf_  (  2                                                   ),
    conv_c_af_    (                     iel_                             ),
    conv_r_af_    (  2   ,     2      , iel_,blitz::ColumnMajorArray<3>()),
    conv_g_af_    (                     iel_                             ),
//----------------------+------------+------------------------------------
// element data
//------------------------------------------------------------------------
    tau_          (3),
    svelaf_       (2),
    convaf_old_   (2),
    convsubaf_old_(2),
    viscaf_old_   (2),
    resM_         (2),
    conv_resM_    (                      iel_),
    conv_subaf_   (                      iel_)
{
}


/*----------------------------------------------------------------------*
  |  calculate system matrix for a generalised alpha time integration   |
  |                            (public)                      gammi 02/08|
  *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid2GenalphaResVMM::Sysmat(
  Fluid2*                                               ele,
  std::vector<blitz::Array<double,1> >&                 myknots,
  Epetra_SerialDenseMatrix&                             elemat,
  Epetra_SerialDenseVector&                             elevec,
  const blitz::Array<double,2>&                         edispnp,
  const blitz::Array<double,2>&                         egridvaf,
  const blitz::Array<double,2>&                         evelnp,
  const blitz::Array<double,1>&                         eprenp,
  const blitz::Array<double,2>&                         eaccam,
  const blitz::Array<double,2>&                         evelaf,
  const struct _MATERIAL*                               material,
  const double                                          alphaM,
  const double                                          alphaF,
  const double                                          gamma,
  const double                                          dt,
  const double                                          time,
  const bool                                            newton,
  const enum Fluid2::StabilisationAction                tds,
  const enum Fluid2::StabilisationAction                inertia,
  const enum Fluid2::StabilisationAction                pspg,
  const enum Fluid2::StabilisationAction                supg,
  const enum Fluid2::StabilisationAction                vstab,
  const enum Fluid2::StabilisationAction                cstab,
  const enum Fluid2::StabilisationAction                cross,
  const enum Fluid2::StabilisationAction                reynolds,
  const bool                                            compute_elemat
  )
{

  //------------------------------------------------------------------
  //                     BLITZ CONFIGURATION
  //------------------------------------------------------------------
  //
  // We define the variables i,j,k to be indices to blitz arrays.
  // These are used for array expressions, that is matrix-vector
  // products in the following.

  blitz::firstIndex  i;   // Placeholder for the first index
  blitz::secondIndex j;   // Placeholder for the second index
  blitz::thirdIndex  k;   // Placeholder for the third index
  blitz::fourthIndex l;   // Placeholder for the fourth index

  blitz::Range       _ = blitz::Range::all();

  //------------------------------------------------------------------
  //           SET TIME INTEGRATION SCHEME RELATED DATA
  //------------------------------------------------------------------

  //         n+alpha_F     n+1
  //        t          = t     - (1-alpha_F) * dt

  const double timealphaF = time-(1-alphaF)*dt;

  //------------------------------------------------------------------
  //                      SET MATERIAL DATA
  //------------------------------------------------------------------
  // get viscosity
  // check here, if we really have a fluid !! 
  if( material->mattyp != m_carreauyasuda
      &&      material->mattyp != m_modpowerlaw
      && material->mattyp != m_fluid)
        dserror("Material law is not a fluid");

  // get viscosity
  double visc = 0.0;
  if(material->mattyp == m_fluid)
    visc = material->m.fluid->viscosity;

  //------------------------------------------------------------------
  //                      SET ELEMENT DATA
  //------------------------------------------------------------------
  // set element data
  const DRT::Element::DiscretizationType distype = ele->Shape();

  // get node coordinates
  DRT::Node** nodes = ele->Nodes();
  for (int inode=0; inode<iel_; inode++)
  {
    const double* x = nodes[inode]->X();
    xyze_(0,inode) = x[0];
    xyze_(1,inode) = x[1];
  }

  // add displacement, when fluid nodes move in the ALE case
  if (ele->is_ale_)
  {
    xyze_ += edispnp;
  }

  // get node weights for nurbs elements
  if(distype==DRT::Element::nurbs4 || distype==DRT::Element::nurbs9)
  {
    for (int inode=0; inode<iel_; inode++)
    {
      DRT::NURBS::ControlPoint* cp
        =
        dynamic_cast<DRT::NURBS::ControlPoint* > (nodes[inode]);
      
      weights_(inode) = cp->W();
    }
  }
  
  // dead load in element nodes
  GetNodalBodyForce(ele,timealphaF);

  // in case of viscous stabilization decide whether to use GLS or USFEM
  double vstabfac= 0.0;
  if (vstab == Fluid2::viscous_stab_usfem || vstab == Fluid2::viscous_stab_usfem_only_rhs)
  {
    vstabfac =  1.0;
  }
  else if(vstab == Fluid2::viscous_stab_gls || vstab == Fluid2::viscous_stab_gls_only_rhs)
  {
    vstabfac = -1.0;
  }

  //----------------------------------------------------------------------------
  //            STABILIZATION PARAMETER, SMAGORINSKY MODEL
  //      and everything else that is evaluated in the element center
  //
  // This has to be done before anything else is calculated because we use
  // the same arrays internally.
  //----------------------------------------------------------------------------

  // use one point gauss rule to calculate tau at element center
  DRT::UTILS::GaussRule2D integrationrule_stabili=DRT::UTILS::intrule2D_undefined;
  switch (distype)
  {
      case DRT::Element::quad4:
      case DRT::Element::quad8:
      case DRT::Element::quad9:
      case DRT::Element::nurbs4:
      case DRT::Element::nurbs9:
      {
        integrationrule_stabili = DRT::UTILS::intrule_quad_1point;
        break;
      }
      case DRT::Element::tri3:
      case DRT::Element::tri6:
      {
        integrationrule_stabili = DRT::UTILS::intrule_tri_1point;
        break;
      }
      default:
        dserror("invalid discretization type for fluid2");
  }


  // gaussian points
  const DRT::UTILS::IntegrationPoints2D intpoints_onepoint(integrationrule_stabili);

  // shape functions and derivs at element center
  const double wquad = intpoints_onepoint.qwgt[0];
        
  blitz::Array<double, 1> gp(2);
  gp(0)=intpoints_onepoint.qxg[0][0];
  gp(1)=intpoints_onepoint.qxg[0][1];

  if(distype == DRT::Element::nurbs4
          ||
     distype == DRT::Element::nurbs9)
  {
    DRT::NURBS::UTILS::nurbs_get_2D_funct_deriv
      (funct_  ,
       deriv_  ,
       gp      ,
       myknots ,
       weights_,
       distype );
  }
  else
  {
    // get values of shape functions and derivatives in the gausspoint
    DRT::UTILS::shape_function_2D(funct_,gp(0),gp(1),distype);
    DRT::UTILS::shape_function_2D_deriv1(deriv_,gp(0),gp(1),distype);
  }
  
  // get element type constant for tau
  double mk=0.0;
  switch (distype)
  {
      case DRT::Element::tri3:
      case DRT::Element::quad4:
      case DRT::Element::nurbs4:
        mk = 0.333333333333333333333;
        break;
      case DRT::Element::quad8:
      case DRT::Element::quad9:
      case DRT::Element::nurbs9:
      case DRT::Element::tri6:
        mk = 0.083333333333333333333;
        break;
      default:
        dserror("type unknown!\n");
  }

  // get Jacobian matrix and determinant
  xjm_ = blitz::sum(deriv_(i,k)*xyze_(j,k),k);
  const double det = xjm_(0,0)*xjm_(1,1) - xjm_(0,1)*xjm_(1,0);

  // check for degenerated elements
  if (det < 0.0)
  {
    dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);
  }

  // get a rough approximation of the element area by a one point gauss
  // integration
  area_ = wquad*det;

  // get element length for tau_M and tau_C:
  // it is chosen as the square root of the element area
  const double hk = sqrt(area_);

  //
  //             compute global first derivates
  //
  // this is necessary only for the calculation of the
  // streamlength (required by the quasistatic formulation)
  //
  /*
    Use the Jacobian and the known derivatives in element coordinate
    directions on the right hand side to compute the derivatives in
    global coordinate directions

          +-          -+     +-    -+      +-    -+
          |  dx    dy  |     | dN_k |      | dN_k |
          |  --    --  |     | ---- |      | ---- |
          |  dr    dr  |     |  dx  |      |  dr  |
          |            |  *  |      |   =  |      | for all k
          |  dx    dy  |     | dN_k |      | dN_k |
          |  --    --  |     | ---- |      | ---- |
          |  ds    ds  |     |  dy  |      |  ds  |
          +-          -+     +-    -+      +-    -+

          Matrix is inverted analytically
  */
  // inverse of jacobian
  xji_(0,0) = ( xjm_(1,1))/det;
  xji_(0,1) = (-xjm_(0,1))/det;
  xji_(1,0) = (-xjm_(1,0))/det;
  xji_(1,1) = ( xjm_(0,0))/det;

  // compute global derivates
  derxy_ = blitz::sum(xji_(i,k)*deriv_(k,j),k);

  // get velocities (n+alpha_F,i) at integration point
  //
  //                 +-----
  //       n+af       \                  n+af
  //    vel    (x) =   +      N (x) * vel
  //                  /        j         j
  //                 +-----
  //                 node j
  //
  velintaf_ = blitz::sum(funct_(j)*evelaf(i,j),j);

  // get velocity (n+alpha_F,i) derivatives at integration point
  //
  //       n+af      +-----  dN (x)
  //   dvel    (x)    \        k         n+af
  //   ----------- =   +     ------ * vel
  //       dx         /        dx        k
  //         j       +-----      j
  //                 node k
  //
  // j : direction of derivative x/y/z
  //
  vderxyaf_ = blitz::sum(derxy_(j,k)*evelaf(i,k),k);

  // get velocities (n+1,i)  at integration point
  //
  //                +-----
  //       n+1       \                  n+1
  //    vel   (x) =   +      N (x) * vel
  //                 /        j         j
  //                +-----
  //                node j
  //
  velintnp_    = blitz::sum(funct_(j)*evelnp(i,j),j);

  // get velocity norms
  const double vel_normaf = sqrt(blitz::sum(velintaf_*velintaf_));
  const double vel_normnp = sqrt(blitz::sum(velintnp_*velintnp_));

  /*------------------------------------------------------------------*/
  /*                                                                  */
  /*                 GET EFFECTIVE VISCOSITY IN GAUSSPOINT            */
  /*                                                                  */
  /* A cause for the necessity of an effective viscosity might        */
  /* be the use of a shear thinning Non-Newtonian fluid               */
  /*                                                                  */
  /*                            /         \                           */
  /*            visc    = visc | shearrate |                          */
  /*                eff         \         /                           */
  /*                                                                  */
  /*                                                                  */
  /* Mind that at the moment all stabilization (tau and viscous test  */
  /* functions if applied) are based on the effective viscosity.      */
  /* We do this since we do not evaluate the stabilisation parameter  */
  /* in the gausspoints but just once in the middle of the element.   */
  /*------------------------------------------------------------------*/

  // compute nonlinear viscosity according to the Carreau-Yasuda model
  if( material->mattyp != m_fluid )
      CalVisc( material, visc);
  
  double visceff = visc;

  if(tds == Fluid2::subscales_time_dependent)
  {
    // INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA, TIME DEPENDENT SUBSCALES
    //
    // tau_M: modification of
    //
    //    Franca, L.P. and Valentin, F.: On an Improved Unusual Stabilized
    //    Finite Element Method for the Advective-Reactive-Diffusive
    //    Equation. Computer Methods in Applied Mechanics and Enginnering,
    //    Vol. 190, pp. 1785-1800, 2000.
    //    http://www.lncc.br/~valentin/publication.htm                   */
    //
    // tau_Mp: modification of Barrenechea, G.R. and Valentin, F.
    //
    //    Barrenechea, G.R. and Valentin, F.: An unusual stabilized finite
    //    element method for a generalized Stokes problem. Numerische
    //    Mathematik, Vol. 92, pp. 652-677, 2002.
    //    http://www.lncc.br/~valentin/publication.htm
    //
    //
    // tau_C: kept Wall definition
    //
    // for the modifications see Codina, Principe, Guasch, Badia
    //    "Time dependent subscales in the stabilized finite  element
    //     approximation of incompressible flow problems"
    //
    //
    // see also: Codina, R. and Soto, O.: Approximation of the incompressible
    //    Navier-Stokes equations using orthogonal subscale stabilisation
    //    and pressure segregation on anisotropic finite element meshes.
    //    Computer methods in Applied Mechanics and Engineering,
    //    Vol 193, pp. 1403-1419, 2004.

    //---------------------------------------------- compute tau_Mu = tau_Mp
    /* convective : viscous forces (element reynolds number)*/
    const double re_convectaf = (vel_normaf * hk / visceff ) * (mk/2.0);

    const double xi_convectaf = DMAX(re_convectaf,1.0);

    /*
               xi_convect ^
                          |      /
                          |     /
                          |    /
                        1 +---+
                          |
                          |
                          |
                          +--------------> re_convect
                              1
    */

    /* the 4.0 instead of the Franca's definition 2.0 results from the viscous
     * term in the Navier-Stokes-equations, which is scaled by 2.0*nu         */

    tau_(0) = DSQR(hk) / (4.0 * visceff / mk + ( 4.0 * visceff/mk) * xi_convectaf);

    /*------------------------------------------------------ compute tau_C ---*/

    //-- stability parameter definition according to Wall Diss. 99
    /*
               xi_convect ^
                          |
                        1 |   +-----------
                          |  /
                          | /
                          |/
                          +--------------> Re_convect
                              1
    */
    const double re_convectnp = (vel_normnp * hk / visceff ) * (mk/2.0);

    const double xi_tau_c = DMIN(re_convectnp,1.0);

    tau_(2) = vel_normnp * hk * 0.5 * xi_tau_c;

  }
  else
  {
    // INSTATIONARY FLOW PROBLEM, GENERALISED ALPHA
    // tau_M: Barrenechea, G.R. and Valentin, F.
    // tau_C: Wall


    // this copy of velintaf_ will be used to store the normed velocity
    blitz::Array<double,1> normed_velintaf(2);
    normed_velintaf=velintaf_.copy();

    // normed velocity at element center (we use the copy for safety reasons!)
    if (vel_normaf>=1e-6)
    {
      normed_velintaf = velintaf_/vel_normaf;
    }
    else
    {
      normed_velintaf    = 0.;
      normed_velintaf(0) = 1.;
    }

    // get streamlength
    const double val = blitz::sum(blitz::abs(blitz::sum(normed_velintaf(j)*derxy_(j,i),j)));
    const double strle = 2.0/val;

    // time factor
    const double timefac = gamma*dt;

    /*----------------------------------------------------- compute tau_Mu ---*/
    /* stability parameter definition according to

              Barrenechea, G.R. and Valentin, F.: An unusual stabilized finite
              element method for a generalized Stokes problem. Numerische
              Mathematik, Vol. 92, pp. 652-677, 2002.
              http://www.lncc.br/~valentin/publication.htm
    and:
              Franca, L.P. and Valentin, F.: On an Improved Unusual Stabilized
              Finite Element Method for the Advective-Reactive-Diffusive
              Equation. Computer Methods in Applied Mechanics and Enginnering,
              Vol. 190, pp. 1785-1800, 2000.
              http://www.lncc.br/~valentin/publication.htm                   */


    const double re1 = 4.0 * timefac * visceff / (mk * DSQR(strle));   /* viscous : reactive forces   */
    const double re2 = mk * vel_normaf * strle / (2.0 * visceff);      /* convective : viscous forces */

    const double xi1 = DMAX(re1,1.0);
    const double xi2 = DMAX(re2,1.0);

    tau_(0) = timefac * DSQR(strle) / (DSQR(strle)*xi1+( 4.0 * timefac*visceff/mk)*xi2);

    // compute tau_Mp
    //    stability parameter definition according to Franca and Valentin (2000)
    //                                       and Barrenechea and Valentin (2002)
    const double re_viscous = 4.0 * timefac * visceff / (mk * DSQR(hk)); /* viscous : reactive forces   */
    const double re_convect = mk * vel_normaf * hk / (2.0 * visceff);    /* convective : viscous forces */

    const double xi_viscous = DMAX(re_viscous,1.0);
    const double xi_convect = DMAX(re_convect,1.0);

    /*
                  xi1,xi2 ^
                          |      /
                          |     /
                          |    /
                        1 +---+
                          |
                          |
                          |
                          +--------------> re1,re2
                              1
    */
    tau_(1) = timefac * DSQR(hk) / (DSQR(hk) * xi_viscous + ( 4.0 * timefac * visceff/mk) * xi_convect);

    // Wall Diss. 99
    /*
                      xi2 ^
                          |
                        1 |   +-----------
                          |  /
                          | /
                          |/
                          +--------------> Re2
                              1
    */
    const double xi_tau_c = DMIN(re2,1.0);
    tau_(2) = vel_normnp * hk * 0.5 * xi_tau_c;
  }

  //----------------------------------------------------------------------------
  //
  //    From here onwards, we are working on the gausspoints of the element
  //            integration, not on the element center anymore!
  //
  //----------------------------------------------------------------------------

  // flag for higher order elements
  const bool higher_order_ele = ele->isHigherOrderElement(distype);

  // gaussian points
  const DRT::UTILS::IntegrationPoints2D intpoints(ele->gaussrule_);

  // remember whether the subscale quantities have been allocated an set to zero.
  if(tds == Fluid2::subscales_time_dependent)
  {
    // if not available, the arrays for the subscale quantities have to
    // be resized and initialised to zero
    if(ele->sub_acc_old_.extent(blitz::firstDim) != 2
       ||
       ele->sub_acc_old_.extent(blitz::secondDim) != intpoints.nquad)
    {
      ele->sub_acc_old_ .resize(2,intpoints.nquad);
      ele->sub_acc_old_  = 0.;
    }
    if(ele->sub_vel_old_.extent(blitz::firstDim) != 2
       ||
       ele->sub_vel_old_.extent(blitz::secondDim) != intpoints.nquad)
    {
      ele->sub_vel_old_ .resize(2,intpoints.nquad);
      ele->sub_vel_old_  = 0.;

      ele->sub_vel_.resize(2,intpoints.nquad);
      ele->sub_vel_ = 0.;
    }
    if(ele->sub_pre_old_ .extent(blitz::firstDim) != intpoints.nquad)
    {
      ele->sub_pre_old_ .resize(intpoints.nquad);
      ele->sub_pre_old_ = 0.;

      ele->sub_pre_.resize(intpoints.nquad);
      ele->sub_pre_ = 0.;
    }
  }

  // get subscale information from element --- this is just a reference
  // to the element data
  blitz::Array<double,2> saccn (ele->sub_acc_old_);
  blitz::Array<double,2> sveln (ele->sub_vel_old_);
  blitz::Array<double,2> svelnp(ele->sub_vel_    );
  blitz::Array<double,1> spren (ele->sub_pre_old_);
  blitz::Array<double,1> sprenp(ele->sub_pre_    );

  // just define certain constants for conveniance
  const double afgdt  = alphaF * gamma * dt;


  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------
  for (int iquad=0;iquad<intpoints.nquad;++iquad)
  {

    // set gauss point coordinates
    blitz::Array<double, 1> gp(2);

    gp(0)=intpoints.qxg[iquad][0];
    gp(1)=intpoints.qxg[iquad][1];


    if(!(distype == DRT::Element::nurbs4
          ||
         distype == DRT::Element::nurbs9))
    {
      // get values of shape functions and derivatives in the gausspoint
      DRT::UTILS::shape_function_2D(funct_,gp(0),gp(1),distype);
      DRT::UTILS::shape_function_2D_deriv1(deriv_,gp(0),gp(1),distype);

      if (higher_order_ele)
      {
        // get values of shape functions and derivatives in the gausspoint
        DRT::UTILS::shape_function_2D_deriv2(deriv2_,gp(0),gp(1),distype);
      }
    }
    else
    {
      if (higher_order_ele)
      {
        DRT::NURBS::UTILS::nurbs_get_2D_funct_deriv_deriv2
          (funct_  ,
           deriv_  ,
           deriv2_ ,
           gp      ,
           myknots ,
           weights_,
           distype );
      }
      else
      {
        DRT::NURBS::UTILS::nurbs_get_2D_funct_deriv
          (funct_  ,
           deriv_  ,
           gp      ,
           myknots ,
           weights_,
           distype );
      }
    }
    
    // get transposed Jacobian matrix and determinant
    //
    //        +-       -+ T      +-       -+
    //        | dx   dx |        | dx   dy |
    //        | --   -- |        | --   -- |
    //        | dr   ds |        | dr   dr |
    //        |         |    =   |         |
    //        | dy   dy |        | dx   dy |
    //        | --   -- |        | --   -- |
    //        | dr   ds |        | ds   ds |
    //        +-       -+        +-       -+
    //
    // The Jacobian is computed using the formula
    //
    //            +-----
    //   dx_j(r)   \      dN_k(r)
    //   -------  = +     ------- * (x_j)_k
    //    dr_i     /       dr_i       |
    //            +-----    |         |
    //            node k    |         |
    //                  derivative    |
    //                   of shape     |
    //                   function     |
    //                           component of
    //                          node coordinate
    //
    xjm_ = blitz::sum(deriv_(i,k)*xyze_(j,k),k);

    // The determinant ist computed using Sarrus's rule
    const double det = xjm_(0,0)*xjm_(1,1)-xjm_(0,1)*xjm_(1,0);

    // check for degenerated elements
    if (det < 0.0)
    {
      dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);
    }

    // set total integration factor
    const double fac = intpoints.qwgt[iquad]*det;

    //--------------------------------------------------------------
    //             compute global first derivates
    //--------------------------------------------------------------
    //
    /*
      Use the Jacobian and the known derivatives in element coordinate
      directions on the right hand side to compute the derivatives in
      global coordinate directions

          +-          -+     +-    -+      +-    -+
          |  dx    dy  |     | dN_k |      | dN_k |
          |  --    --  |     | ---- |      | ---- |
          |  dr    dr  |     |  dx  |      |  dr  |
          |            |  *  |      |   =  |      | for all k
          |  dx    dy  |     | dN_k |      | dN_k |
          |  --    --  |     | ---- |      | ---- |
          |  ds    ds  |     |  dy  |      |  ds  |
          +-          -+     +-    -+      +-    -+

          Matrix is inverted analytically
    */
    // inverse of jacobian
    xji_(0,0) = ( xjm_(1,1))/det;
    xji_(0,1) = (-xjm_(0,1))/det;
    xji_(1,0) = (-xjm_(1,0))/det;
    xji_(1,1) = ( xjm_(0,0))/det;

    // compute global derivates
    derxy_ = blitz::sum(xji_(i,k)*deriv_(k,j),k);

    //--------------------------------------------------------------
    //             compute second global derivative
    //--------------------------------------------------------------

    /*----------------------------------------------------------------------*
     |  calculate second global derivatives w.r.t. x,y at point r,s
     |                                            (private)      gammi 02/08
     |
     | From the three equations
     |
     |              +-             -+
     |  d^2N     d  | dx dN   dy dN |
     |  ----   = -- | --*-- + --*-- |
     |  dr^2     dr | dr dx   dr dy |
     |              +-             -+
     |
     |              +-             -+
     |  d^2N     d  | dx dN   dy dN |
     |  ------ = -- | --*-- + --*-- |
     |  ds^2     ds | ds dx   ds dy |
     |              +-             -+
     |
     |              +-             -+
     |  d^2N     d  | dx dN   dy dN |
     | -----   = -- | --*-- + --*-- |
     | ds dr     ds | dr dx   dr dy |
     |              +-             -+
     |
     | the matrix (jacobian-bar matrix) system
     |
     | +-                                          -+   +-    -+
     | |   /dx\^2        /dy\^2         dy dx       |   | d^2N |
     | |  | -- |        | ---|        2*--*--       |   | ---- |
     | |   \dr/          \dr/           dr dr       |   | dx^2 |
     | |                                            |   |      |
     | |   /dx\^2        /dy\^2         dy dx       |   | d^2N |
     | |  | -- |        | ---|        2*--*--       |   | ---- |
     | |   \ds/          \ds/           ds ds       |   | dy^2 |
     | |                                            | * |      |
     | |   dx dx         dy dy      dx dy   dx dy   |   | d^2N |
     | |   --*--         --*--      --*-- + --*--   |   | ---- |
     | |   dr ds         dr ds      dr ds   ds dr   |   | dxdy |
     | +-                                          -+   +-    -+
     |
     |                  +-    -+     +-                 -+
     |                  | d^2N |     | d^2x dN   d^2y dN |
     |                  | ---- |     | ----*-- + ----*-- |
     |                  | dr^2 |     | dr^2 dx   dr^2 dy |
     |                  |      |     |                   |
     |                  | d^2N |     | d^2x dN   d^2y dN |
     |              =   | ---- |  -  | ----*-- + ----*-- |
     |                  | ds^2 |     | ds^2 dx   ds^2 dy |
     |                  |      |     |                   |
     |                  | d^2N |     | d^2x dN   d^2y dN |
     |                  | ---- |     | ----*-- + ----*-- |
     |                  | drds |     | drds dx   drds dy |
     |                  +-    -+     +-                 -+
     |
     |
     | is derived. This is solved for the unknown global derivatives.
     |
     |
     |             jacobian_bar * derxy2 = deriv2 - xder2 * derxy
     |                                              |           |
     |                                              +-----------+
     |                                              'chainrulerhs'
     |                                     |                    |
     |                                     +--------------------+
     |                                          'chainrulerhs'
     |
     *----------------------------------------------------------------------*/
    if (higher_order_ele)
    {
      // initialize and zero out everything
      blitz::Array<double,2> bm(3,3,blitz::ColumnMajorArray<2>());

      // calculate elements of jacobian_bar matrix
      bm(0,0) =                     xjm_(0,0)*xjm_(0,0);
      bm(0,1) =                     xjm_(0,1)*xjm_(0,1);
      bm(0,2) =                 2.0*xjm_(0,0)*xjm_(0,1);

      bm(1,0) =                     xjm_(1,0)*xjm_(1,0);
      bm(1,1) =                     xjm_(1,1)*xjm_(1,1);
      bm(1,2) =                 2.0*xjm_(1,1)*xjm_(1,0);

      bm(2,0) =                     xjm_(0,0)*xjm_(1,0);
      bm(2,1) =                     xjm_(0,1)*xjm_(1,1);
      bm(2,2) = xjm_(0,0)*xjm_(1,1)+xjm_(0,1)*xjm_(1,0);


      /*------------------ determine 2nd derivatives of coord.-functions */
      /*
       |                                             0 1
       |         0 1              0...iel-1         +-+-+
       |        +-+-+             +-+-+-+-+         | | | 0
       |        | | | 0           | | | | | 0       +-+-+
       |        +-+-+             +-+-+-+-+         | | | .
       |        | | | 1     =     | | | | | 1     * +-+-+ .
       |        +-+-+             +-+-+-+-+         | | | .
       |        | | | 2           | | | | | 2       +-+-+
       |        +-+-+             +-+-+-+-+         | | | iel-1
       |                                            +-+-+
       |
       |        xder2               deriv2          xyze^T
       |
       |
       |                                        +-           -+
       |  	   	    	    	        | d^2x   d^2y |
       |  	   	    	    	        | ----   ---- |
       | 	   	   	   	        | dr^2   dr^2 |
       | 	   	   	   	        |             |
       | 	   	   	   	        | d^2x   d^2y |
       |                    yields    xder2  =  | ----   ---- |
       | 	   	   	   	        | ds^2   ds^2 |
       | 	   	   	   	        |             |
       | 	   	   	   	        | d^2x   d^2y |
       | 	   	   	   	        | ----   ---- |
       | 	   	   	   	        | drds   drds |
       | 	   	   	   	        +-           -+
      */
      xder2_ = blitz::sum(deriv2_(i,k)*xyze_(j,k),k);


      /*
       |        0...iel-1             0 1
       |        +-+-+-+-+            +-+-+               0...iel-1
       |        | | | | | 0          | | | 0             +-+-+-+-+
       |        +-+-+-+-+            +-+-+               | | | | | 0
       |        | | | | | 1     =    | | | 1     *       +-+-+-+-+   * (-1)
       |        +-+-+-+-+            +-+-+               | | | | | 1
       |        | | | | | 2          | | | 2             +-+-+-+-+
       |        +-+-+-+-+            +-+-+
       |
       |       chainrulerhs          xder2                 derxy
      */
      derxy2_ = -blitz::sum(xder2_(i,k)*derxy_(k,j),k);

      /*
       |        0...iel-1             0...iel-1             0...iel-1
       |        +-+-+-+-+             +-+-+-+-+             +-+-+-+-+
       |        | | | | | 0           | | | | | 0           | | | | | 0
       |        +-+-+-+-+             +-+-+-+-+             +-+-+-+-+
       |        | | | | | 1     =     | | | | | 1     +     | | | | | 1
       |        +-+-+-+-+             +-+-+-+-+             +-+-+-+-+
       |        | | | | | 2           | | | | | 2           | | | | | 2
       |        +-+-+-+-+             +-+-+-+-+             +-+-+-+-+
       |
       |       chainrulerhs          chainrulerhs             deriv2
      */
      derxy2_ += deriv2_;

       /* make LU decomposition and solve system for all right hand sides
       * (i.e. the components of chainrulerhs)

       |
       |            0  1  2          i        i
       | 	   +--+--+--+       +-+      +-+
       | 	   |  |  |  | 0     | | 0    | | 0
       | 	   +--+--+--+       +-+	     +-+
       | 	   |  |  |  | 1  *  | | 1 =  | | 1  for i=0...iel-1
       | 	   +--+--+--+       +-+	     +-+
       | 	   |  |  |  | 2     | | 2    | | 2
       | 	   +--+--+--+       +-+	     +-+
       |                             |        |
       |                             |        |
       |                           derxy2[i]  |
       |                                      |
       |                                chainrulerhs[i]
       |
       |
       |
       |                      0...iel-1
       |		     +-+-+-+-+
       |		     | | | | | 0
       |		     +-+-+-+-+
       |	  yields     | | | | | 1
       |		     +-+-+-+-+
       |                     | | | | | 2
       | 		     +-+-+-+-+
       |
       |                      derxy2
       |
       */

      // Use LAPACK
      Epetra_LAPACK          solver;

      // a vector specifying the pivots (reordering)
      int pivot[3];

      // error code
      int ierr = 0;

      // Perform LU factorisation --- this call replaces bm with its factorisation
      solver.GETRF(3,3,bm.data(),3,&(pivot[0]),&ierr);

      if (ierr!=0)
      {
        dserror("Unable to perform LU factorisation during computation of derxy2");
      }

      // backward substitution. GETRS replaces the input (chainrulerhs, currently
      // stored on derxy2) with the result
      solver.GETRS('N',3,iel_,bm.data(),3,&(pivot[0]),derxy2_.data(),3,&ierr);

      if (ierr!=0)
      {
        dserror("Unable to perform backward substitution after factorisation of jacobian");
      }
    }
    else
    {
      derxy2_  = 0.;
    }

    //--------------------------------------------------------------
    //            interpolate nodal values to gausspoint
    //--------------------------------------------------------------

    // get intermediate accelerations (n+alpha_M,i) at integration point
    //
    //                 +-----
    //       n+am       \                  n+am
    //    acc    (x) =   +      N (x) * acc
    //                  /        j         j
    //                 +-----
    //                 node j
    //
    // i         : space dimension u/v
    //
    accintam_    = blitz::sum(funct_(j)*eaccam(i,j),j);

    // get velocities (n+alpha_F,i) at integration point
    //
    //                 +-----
    //       n+af       \                  n+af
    //    vel    (x) =   +      N (x) * vel
    //                  /        j         j
    //                 +-----
    //                 node j
    //
    velintaf_    = blitz::sum(funct_(j)*evelaf(i,j),j);

    if(ele->is_ale_)
    {
      // get velocities (n+alpha_F,i) at integration point
      //
      //                 +-----
      //       n+af       \                  n+af
      //    u_G    (x) =   +      N (x) * u_G
      //                  /        j         j
      //                 +-----
      //                 node j
      //
      ugrid_af_    = blitz::sum(funct_(j)*egridvaf(i,j),j);
    }

    // get bodyforce in gausspoint, time (n+alpha_F)
    //
    //                 +-----
    //       n+af       \                n+af
    //      f    (x) =   +      N (x) * f
    //                  /        j       j
    //                 +-----
    //                 node j
    //
    bodyforceaf_ = blitz::sum(funct_(j)*edeadaf_(i,j),j);

    // get velocities (n+1,i)  at integration point
    //
    //                +-----
    //       n+1       \                  n+1
    //    vel   (x) =   +      N (x) * vel
    //                 /        j         j
    //                +-----
    //                node j
    //
    velintnp_    = blitz::sum(funct_(j)*evelnp(i,j),j);

    // get pressure (n+1,i) at integration point
    //
    //                +-----
    //       n+1       \                  n+1
    //    pre   (x) =   +      N (x) * pre
    //                 /        i         i
    //                +-----
    //                node i
    //
    prenp_    = blitz::sum(funct_*eprenp);

    // get pressure gradient (n+1,i) at integration point
    //
    //       n+1      +-----  dN (x)
    //   dpre   (x)    \        j         n+1
    //   ---------- =   +     ------ * pre
    //       dx        /        dx        j
    //         i      +-----      i
    //                node j
    //
    // i : direction of derivative
    //
    pderxynp_ = blitz::sum(derxy_(i,j)*eprenp(j),j);


    // get velocity (n+alpha_F,i) derivatives at integration point
    //
    //       n+af      +-----  dN (x)
    //   dvel    (x)    \        k         n+af
    //   ----------- =   +     ------ * vel
    //       dx         /        dx        k
    //         j       +-----      j
    //                 node k
    //
    // j : direction of derivative x/y/z
    //
    vderxyaf_ = blitz::sum(derxy_(j,k)*evelaf(i,k),k);

    // get velocity (n+1,i) derivatives at integration point
    //
    //       n+1      +-----  dN (x)
    //   dvel   (x)    \        k         n+1
    //   ---------- =   +     ------ * vel
    //       dx        /        dx        k
    //         j      +-----      j
    //                node k
    //
    vderxynp_ = blitz::sum(derxy_(j,k)*evelnp(i,k),k);

    /*--- convective part u_old * grad (funct) --------------------------*/
    /* u_old_x * N,x  +  u_old_y * N,y + u_old_z * N,z
       with  N .. form function matrix                                   */
    conv_c_af_  = blitz::sum(derxy_(j,i)*velintaf_(j), j);


    /*--- convective grid part u_G * grad (funct) -----------------------*/
    /* u_old_x * N,x  +  u_old_y * N,y   with  N .. form function matrix */
    if (ele->is_ale_)
    {
      conv_g_af_ = blitz::sum(derxy_(j,i) * ugrid_af_(j), j);
    }
    else
    {
      conv_g_af_ = 0.0;
    }


    // calculate 2nd velocity derivatives at integration point, time(n+alpha_F)
    //
    //    2   n+af       +-----   dN (x)
    //   d vel    (x)     \         k          n+af
    //   ------------  =   +     -------- * vel
    //    dx  dx          /      dx  dx        k
    //      j1  j2       +-----    j1  j2
    //                   node k
    //
    // j=(j1,j2) : direction of derivative x/y
    if(higher_order_ele)
    {
      vderxy2af_ = blitz::sum(derxy2_(j,k)*evelaf(i,k),k);
    }
    else
    {
      vderxy2af_ = 0.;
    }

    /*--- reactive part funct * grad (u_old) ----------------------------*/
    /*        /                        \
              |  u_old_x,x   u_old_x,y |
              |                        | * N
              |  u_old_y,x   u_old_y,y |
              \                        /
       with  N .. form function matrix                                   */
    conv_r_af_ = vderxyaf_(i, j)*funct_(k);

    /*--- viscous term  grad * epsilon(u): ------------------------------*/
    /*

             /   2           2                  2          \ /    \
             | dN (x)       d N (x)            d N (x)     | |    |
             |   k      1      k           1      k        | | u  |
      +----- | ------ + - * ------         - * ------      | |  k |
       \     | dx*dx    2   dy*dy          2   dx*dy       | |    |
        +    |                                             |*|    |
       /     |          2                   2       2      | |    |
      +----- |         d N (x)            dN (x)   d N (x) | |    |
      node k |     1      k           1     k         k    | | v  |
             |     - * ------         - * ------ + ------  | |  k |
             |     2   dx*dy          2   dx*dx    dy*dy   | |    |
             \                                             / \    /

    */
    viscs2_(0,0,_) = derxy2_(0,_)+0.5*derxy2_(1,_);
    viscs2_(0,1,_) = 0.5 * derxy2_(2,_);
    viscs2_(1,0,_) = 0.5 * derxy2_(2,_);
    viscs2_(1,1,_) = 0.5*derxy2_(0,_)+derxy2_(1,_);


    /* divergence new time step n+1 */
    const double divunp = (vderxynp_(0,0)+vderxynp_(1,1));

    /* Convective term  u_old * grad u_old: */
    convaf_old_ = blitz::sum(vderxyaf_(i, j)*velintaf_(j), j);

    /* Viscous term  div epsilon(u_old) */
    /*        /                                 \
              |   n+af         n+af       n+af  |
              |  u      + 0.5*u    + 0.5*v      |
              |   xx(i)        yy(i)      xy(i) |
              |                                 |
              |       n+af        n+af    n+af  |
              |  0.5*u     + 0.5*v     + u      |
              |       xy(i)       xy(i)   xx(i) |
              \                                 /
    */
    viscaf_old_(0) = vderxy2af_(0,0) + 0.5 * (vderxy2af_(0,1) + vderxy2af_(1,2));
    viscaf_old_(1) = vderxy2af_(1,1) + 0.5 * (vderxy2af_(1,0) + vderxy2af_(0,2));

    /* compute residual in gausspoint --- the residual is based on the
                                                  effective viscosity! */
    resM_ = accintam_ + convaf_old_ - 2*visceff*viscaf_old_ + pderxynp_ - bodyforceaf_;

    if (ele->is_ale_)
    {
      // correct convection with grid velocity
      resM_ -= blitz::sum(vderxyaf_(i, j)*ugrid_af_(j), j);
    }

    /*
      This is the operator

                  /               \
                 | resM    o nabla |
                  \    (i)        /

      required for the cross and reynolds stress calculation

    */
    conv_resM_ =  blitz::sum(resM_(j)*derxy_(j,i),j);

    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //
    //    ELEMENT FORMULATION BASED ON TIME DEPENDENT SUBSCALES
    //
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    if(tds == Fluid2::subscales_time_dependent)
    {
      const double tauM   = tau_(0);
      const double tauC   = tau_(2);

      // update estimates for the subscale quantities

      const double factauC                  = tauC/(tauC+dt);
      const double facMtau                  = 1./(alphaM*tauM+afgdt);

      /*-------------------------------------------------------------------*
       *                                                                   *
       *                  update of SUBSCALE PRESSURE                      *
       *                                                                   *
       *-------------------------------------------------------------------*/

      /*
        ~n+1      tauC     ~n   tauC * dt            n+1
        p    = --------- * p  - --------- * nabla o u
         (i)   tauC + dt        tauC + dt            (i)
      */
      sprenp(iquad)=(spren(iquad)-dt*divunp)*factauC;

      /*-------------------------------------------------------------------*
       *                                                                   *
       *                  update of SUBSCALE VELOCITY                      *
       *                                                                   *
       *-------------------------------------------------------------------*/

      /*
        ~n+1                1.0
        u    = ----------------------------- *
         (i)   alpha_M*tauM+alpha_F*gamma*dt

                +-
                | +-                                  -+   ~n
               *| |alpha_M*tauM +gamma*dt*(alpha_F-1.0)| * u +
                | +-                                  -+
                +-


                    +-                      -+    ~ n
                  + | dt*tauM*(alphaM-gamma) | * acc -
                    +-                      -+

                                           -+
                                       n+1  |
                  - gamma*dt*tauM * res     |
                                       (i)  |
                                           -+
      */
      svelnp(_,iquad)=((alphaM*tauM+gamma*dt*(alphaF-1.0))*sveln(_,iquad)
                       +
                       (dt*tauM*(alphaM-gamma))           *saccn(_,iquad)
                       -
                       (gamma*dt*tauM)                    *resM_(_)
                      )*facMtau;

      /*-------------------------------------------------------------------*
       *                                                                   *
       *               update of intermediate quantities                   *
       *                                                                   *
       *-------------------------------------------------------------------*/

      /* compute the intermediate value of subscale velocity

              ~n+af            ~n+1                   ~n
              u     = alphaF * u     + (1.0-alphaF) * u
               (i)              (i)

      */
      svelaf_(_) = alphaF*svelnp(_,iquad)+(1.0-alphaF)*sveln(_,iquad);

      /* the intermediate value of subscale acceleration is not needed to be
       * computed anymore --- we use the governing ODE to replace it ....

             ~ n+am    alphaM     / ~n+1   ~n \    gamma - alphaM    ~ n
            acc     = -------- * |  u    - u   | + -------------- * acc
               (i)    gamma*dt    \  (i)      /         gamma

      */

      /*
        This is the operator

                  /~n+af         \
                 | u      o nabla |
                  \   (i)        /

                  required for the cross and reynolds stress calculation

      */
      conv_subaf_ =  blitz::sum(svelaf_(j)*derxy_(j,i),j);

      /* Most recent value for subgrid velocity convective term

                  /~n+af         \   n+af
                 | u      o nabla | u
                  \   (i)        /   (i)
      */

      convsubaf_old_ = blitz::sum(vderxyaf_(i, j)*svelaf_(j), j);

      //--------------------------------------------------------------
      //--------------------------------------------------------------
      //
      //                       SYSTEM MATRIX
      //
      //--------------------------------------------------------------
      //--------------------------------------------------------------
      if(compute_elemat)
      {
        //---------------------------------------------------------------
        //
        //   GALERKIN PART 1 AND SUBSCALE ACCELERATION STABILISATION
        //
        //---------------------------------------------------------------
        if(inertia == Fluid2::inertia_stab_keep)
        {
          const double fac_alphaM_tauM_facMtau                   = fac*alphaM*tauM*facMtau;
          const double fac_two_visceff_afgdt_alphaM_tauM_facMtau = fac*2.0*visceff*afgdt*alphaM*tauM*facMtau;
          const double fac_afgdt_afgdt_facMtau                   = fac*afgdt*afgdt*facMtau;
          const double fac_alphaM_afgdt_facMtau                  = fac*alphaM*afgdt*facMtau;


          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            const double fac_alphaM_afgdt_facMtau_funct_ui    = fac_alphaM_afgdt_facMtau*funct_(ui);
            const double fac_afgdt_afgdt_facMtau_conv_c_af_ui = fac_afgdt_afgdt_facMtau*conv_c_af_(ui);
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {

              /*
                inertia term (intermediate)

                factor:

                               alphaF*gamma*dt
                 alphaM*---------------------------
                        alphaM*tauM+alphaF*gamma*dt


                            /          \
                           |            |
                           |  Dacc , v  |
                           |            |
                            \          /
              */

              elemat(vi*3    , ui*3    ) += fac_alphaM_afgdt_facMtau_funct_ui*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM_afgdt_facMtau_funct_ui*funct_(vi) ;
              /* convection (intermediate)

              factor:

                                     alphaF*gamma*dt
               +alphaF*gamma*dt*---------------------------
                                alphaM*tauM+alphaF*gamma*dt


                          /                          \
                         |  / n+af       \            |
                         | | u    o nabla | Dacc , v  |
                         |  \            /            |
                          \                          /
              */
              elemat(vi*3    , ui*3    ) += fac_afgdt_afgdt_facMtau_conv_c_af_ui*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_afgdt_facMtau_conv_c_af_ui*funct_(vi) ;

              /* pressure (implicit) */

              /*  factor:
                             alphaM*tauM
                    ---------------------------
                    alphaM*tauM+alphaF*gamma*dt

                 /               \
                |                 |
                |  nabla Dp ,  v  |
                |                 |
                 \               /
              */
              elemat(vi*3    , ui*3 + 2) -= fac_alphaM_tauM_facMtau*derxy_(0,ui)*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 2) -= fac_alphaM_tauM_facMtau*derxy_(1,ui)*funct_(vi) ;

              /* viscous term (intermediate) */
              /*  factor:
                                       alphaM*tauM
           2*nu*alphaF*gamma*dt*---------------------------
                                alphaM*tauM+alphaF*gamma*dt


                  /                         \
                 |               /    \      |
                 |  nabla o eps | Dacc | , v |
                 |               \    /      |
                  \                         /

              */
              elemat(vi*3    , ui*3    ) += fac_two_visceff_afgdt_alphaM_tauM_facMtau*funct_(vi)*viscs2_(0,0,ui);
              elemat(vi*3    , ui*3 + 1) += fac_two_visceff_afgdt_alphaM_tauM_facMtau*funct_(vi)*viscs2_(0,1,ui);
              elemat(vi*3 + 1, ui*3    ) += fac_two_visceff_afgdt_alphaM_tauM_facMtau*funct_(vi)*viscs2_(0,1,ui);
              elemat(vi*3 + 1, ui*3 + 1) += fac_two_visceff_afgdt_alphaM_tauM_facMtau*funct_(vi)*viscs2_(1,1,ui);
            } // end loop rows (test functions for matrix)
          } // end loop rows (solution for matrix, test function for vector)


          if (newton) // if inertia and newton
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {
                /* convection (intermediate)

                factor:

                                     alphaF*gamma*dt
               +alphaF*gamma*dt*---------------------------
                                alphaM*tauM+alphaF*gamma*dt

                         /                            \
                        |  /            \   n+af       |
                        | | Dacc o nabla | u      , v  |
                        |  \            /              |
                         \                            /
                */
                elemat(vi*3    , ui*3    ) += fac_afgdt_afgdt_facMtau*funct_(vi)*conv_r_af_(0, 0, ui) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_afgdt_facMtau*funct_(vi)*conv_r_af_(0, 1, ui) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_afgdt_facMtau*funct_(vi)*conv_r_af_(1, 0, ui) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_afgdt_facMtau*funct_(vi)*conv_r_af_(1, 1, ui) ;
              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)
          } // end if inertia and newton
        } //   end if inertia stabilisation
        else
        { // if no inertia stabilisation
          const double fac_alphaM = fac*alphaM;
          const double fac_afgdt  = fac*afgdt;


          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            const double fac_afgdt_conv_c_af_ui = fac_afgdt*conv_c_af_(ui);
            const double fac_alphaM_funct_ui    = fac_alphaM*funct_(ui);
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {

              /*
                inertia term (intermediate)

                factor: +alphaM

                            /          \
                           |            |
                           |  Dacc , v  |
                           |            |
                            \          /
              */
              elemat(vi*3    , ui*3    ) += fac_alphaM_funct_ui*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM_funct_ui*funct_(vi) ;


              /*  factor:

               +alphaF*gamma*dt

                          /                          \
                         |  / n+af       \            |
                         | | u    o nabla | Dacc , v  |
                         |  \            /            |
                          \                          /
              */
              elemat(vi*3    , ui*3    ) += fac_afgdt_conv_c_af_ui*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_conv_c_af_ui*funct_(vi) ;
            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

          if (newton) // if no inertia and newton
          {

            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {

                /*  factor:

                +alphaF*gamma*dt

                         /                            \
                        |  /            \   n+af       |
                        | | Dacc o nabla | u      , v  |
                        |  \            /              |
                         \                            /
                */
                elemat(vi*3    , ui*3    ) += fac_afgdt*funct_(vi)*conv_r_af_(0, 0, ui) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt*funct_(vi)*conv_r_af_(0, 1, ui) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt*funct_(vi)*conv_r_af_(1, 0, ui) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt*funct_(vi)*conv_r_af_(1, 1, ui) ;
              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)
          } // end if no inertia and newton
        } // end if no inertia stabilisation


        const double fac_afgdt_visceff        = fac*visceff*afgdt;
        const double fac_gamma_dt             = fac*gamma*dt;

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          const double fac_funct_ui=fac*funct_(ui);

          for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
          {
            //---------------------------------------------------------------
            //
            //   GALERKIN PART 2 (REMAINING EXPRESSIONS)
            //
            //---------------------------------------------------------------
            /* pressure (implicit) */

            /*  factor: -1

                 /                \
                |                  |
                |  Dp , nabla o v  |
                |                  |
                 \                /
            */

            elemat(vi*3    , ui*3 + 2) -= fac_funct_ui*derxy_(0, vi) ;
            elemat(vi*3 + 1, ui*3 + 2) -= fac_funct_ui*derxy_(1, vi) ;

            /* viscous term (intermediate) */

            /*  factor: +2*nu*alphaF*gamma*dt

                 /                          \
                |       /    \         / \   |
                |  eps | Dacc | , eps | v |  |
                |       \    /         \ /   |
                 \                          /
            */

            elemat(vi*3    , ui*3    ) += fac_afgdt_visceff*(2.0*derxy_(0,ui)*derxy_(0,vi)
                                                             +
                                                             derxy_(1,ui)*derxy_(1,vi)) ;
            elemat(vi*3    , ui*3 + 1) += fac_afgdt_visceff*derxy_(0,ui)*derxy_(1,vi) ;
            elemat(vi*3 + 1, ui*3    ) += fac_afgdt_visceff*derxy_(1,ui)*derxy_(0,vi) ;
            elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_visceff*(derxy_(0,ui)*derxy_(0,vi)
                                                             +
                                                             2.0*derxy_(1,ui)*derxy_(1,vi)) ;


            /* continuity equation (implicit) */

            /*  factor: +gamma*dt

                 /                  \
                |                    |
                | nabla o Dacc  , q  |
                |                    |
                 \                  /
            */

            elemat(vi*3 + 2, ui*3    ) += fac_gamma_dt*derxy_(0,ui)*funct_(vi) ;
            elemat(vi*3 + 2, ui*3 + 1) += fac_gamma_dt*derxy_(1,ui)*funct_(vi) ;

          } // end loop rows (test functions for matrix)
        } // end loop columns (solution for matrix, test function for vector)
        // end remaining Galerkin terms


        if(pspg == Fluid2::pstab_use_pspg)
        {
          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //                    PRESSURE STABILISATION
          //
          //---------------------------------------------------------------

          const double fac_gamma_dt_tauM_facMtau                   = fac*gamma*dt*tauM*facMtau;

          const double fac_two_visceff_afgdt_gamma_dt_tauM_facMtau = fac*2.0*visceff*afgdt*gamma*dt*tauM*facMtau;

          const double fac_afgdt_gamma_dt_tauM_facMtau             = fac*afgdt*gamma*dt*tauM*facMtau;

          const double fac_alphaM_gamma_dt_tauM_facMtau            = fac*alphaM*gamma*dt*tauM*facMtau;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            const double fac_alphaM_gamma_dt_tauM_facMtau_funct_ui   =fac_alphaM_gamma_dt_tauM_facMtau*funct_(ui);
            const double fac_afgdt_gamma_dt_tauM_facMtau_conv_c_af_ui=fac_afgdt_gamma_dt_tauM_facMtau*conv_c_af_(ui);
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* pressure stabilisation --- inertia    */

              /*
                           gamma*dt*tau_M
            factor:  ------------------------------ * alpha_M
                     alpha_M*tau_M+alpha_F*gamma*dt


                                /                \
                               |                  |
                               |  Dacc , nabla q  |
                               |                  |
                                \                /
              */
              elemat(vi*3 + 2, ui*3    ) += fac_alphaM_gamma_dt_tauM_facMtau_funct_ui*derxy_(0,vi) ;
              elemat(vi*3 + 2, ui*3 + 1) += fac_alphaM_gamma_dt_tauM_facMtau_funct_ui*derxy_(1,vi) ;

              /* pressure stabilisation --- convection */

              /*
                           gamma*dt*tau_M
            factor:  ------------------------------ * alpha_F*gamma*dt
                     alpha_M*tau_M+alpha_F*gamma*dt


                        /                                \
                       |  / n+af       \                  |
                       | | u    o nabla | Dacc , nabla q  |
                       |  \            /                  |
                        \                                /


                       /                                  \
                      |  /            \   n+af             |
                      | | Dacc o nabla | u      , nabla q  |
                      |  \            /                    |
                       \                                  /

              */
              elemat(vi*3 + 2, ui*3    ) += fac_afgdt_gamma_dt_tauM_facMtau_conv_c_af_ui*derxy_(0,vi) ;
              elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_gamma_dt_tauM_facMtau_conv_c_af_ui*derxy_(1,vi) ;

              /* pressure stabilisation --- diffusion  */


            /*
                           gamma*dt*tau_M
            factor:  ------------------------------ * alpha_F*gamma*dt * 2 * nu
                     alpha_M*tau_M+alpha_F*gamma*dt


                    /                                \
                   |               /    \             |
                   |  nabla o eps | Dacc | , nabla q  |
                   |               \    /             |
                    \                                /
            */

              elemat(vi*3 + 2, ui*3    ) -= fac_two_visceff_afgdt_gamma_dt_tauM_facMtau*
                                            (derxy_(0,vi)*viscs2_(0,0,ui)
                                             +
                                             derxy_(1,vi)*viscs2_(0,1,ui));
              elemat(vi*3 + 2, ui*3 + 1) -= fac_two_visceff_afgdt_gamma_dt_tauM_facMtau*
                                            (derxy_(0,vi)*viscs2_(0,1,ui)
                                             +
                                             derxy_(1,vi)*viscs2_(1,1,ui));

              /* pressure stabilisation --- pressure   */

              /*
                          gamma*dt*tau_M
            factor:  ------------------------------
                     alpha_M*tau_M+alpha_F*gamma*dt



                    /                    \
                   |                      |
                   |  nabla Dp , nabla q  |
                   |                      |
                    \                    /
              */

              elemat(vi*3 + 2, ui*3 + 2) += fac_gamma_dt_tauM_facMtau*
                                            (derxy_(0,ui)*derxy_(0,vi)
                                             +
                                             derxy_(1,ui)*derxy_(1,vi)) ;

            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

          if (newton) // if pspg and newton
          {

            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {
                /* pressure stabilisation --- convection */

                /*
                                gamma*dt*tau_M
                factor:  ------------------------------ * alpha_F*gamma*dt
                         alpha_M*tau_M+alpha_F*gamma*dt

                       /                                  \
                      |  /            \   n+af             |
                      | | Dacc o nabla | u      , nabla q  |
                      |  \            /                    |
                       \                                  /

                */

                elemat(vi*3 + 2, ui*3    ) += fac_afgdt_gamma_dt_tauM_facMtau*
                                              (derxy_(0,vi)*conv_r_af_(0,0,ui)
                                               +
                                               derxy_(1,vi)*conv_r_af_(1,0,ui));
                elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_gamma_dt_tauM_facMtau*
                                              (derxy_(0,vi)*conv_r_af_(0,1,ui)
                                               +
                                               derxy_(1,vi)*conv_r_af_(1,1,ui));

              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)
          }// end if pspg and newton

        } // end pressure stabilisation

        if(supg == Fluid2::convective_stab_supg)
        {

          const double fac_alphaM_afgdt_tauM_facMtau            = fac*alphaM*afgdt*facMtau*tauM;
          const double fac_afgdt_tauM_afgdt_facMtau             = fac*afgdt*afgdt*facMtau*tauM;
          const double fac_afgdt_tauM_facMtau                   = fac*afgdt*tauM*facMtau;
          const double fac_two_visceff_afgdt_afgdt_tauM_facMtau = fac*2.0*visceff*afgdt*afgdt*tauM*facMtau;

          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //         SUPG STABILISATION FOR CONVECTION DOMINATED FLOWS
          //
          //---------------------------------------------------------------
          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            const double fac_alphaM_afgdt_tauM_facMtau_funct_ui    = fac_alphaM_afgdt_tauM_facMtau*funct_(ui);
            const double fac_afgdt_tauM_afgdt_facMtau_conv_c_af_ui = fac_afgdt_tauM_afgdt_facMtau*conv_c_af_(ui);
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* SUPG stabilisation --- inertia

               factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaM
                        alphaM*tauM+alphaF*gamma*dt


                    /                           \
                   |          / n+af       \     |
                   |  Dacc , | u    o nabla | v  |
                   |          \            /     |
                    \                           /
              */
              elemat(vi*3    , ui*3    ) += fac_alphaM_afgdt_tauM_facMtau_funct_ui*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM_afgdt_tauM_facMtau_funct_ui*conv_c_af_(vi) ;

              /* SUPG stabilisation --- convection


               factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaF * gamma * dt
                        alphaM*tauM+alphaF*gamma*dt

                    /                                               \
                   |    / n+af        \          / n+af        \     |
                   |   | u     o nabla | Dacc , | u     o nabla | v  |
                   |    \             /          \             /     |
                    \                                               /
              */

              elemat(vi*3    , ui*3    ) += fac_afgdt_tauM_afgdt_facMtau_conv_c_af_ui*conv_c_af_(vi);
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau_conv_c_af_ui*conv_c_af_(vi) ;

              /* SUPG stabilisation --- diffusion

               factor:
                               alphaF*gamma*tauM*dt
                  - 2 * nu  --------------------------- * alphaF * gamma * dt
                            alphaM*tauM+alphaF*gamma*dt


                    /                                            \
                   |               /     \    / n+af        \     |
                   |  nabla o eps | Dacc  |, | u     o nabla | v  |
                   |               \     /    \             /     |
                    \                                            /
              */
              elemat(vi*3    , ui*3    ) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscs2_(0, 0, ui)*conv_c_af_(vi) ;
              elemat(vi*3    , ui*3 + 1) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscs2_(0, 1, ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3    ) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscs2_(0, 1, ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscs2_(1, 1, ui)*conv_c_af_(vi) ;

              /* SUPG stabilisation --- pressure

               factor:
                               alphaF*gamma*tauM*dt
                            ---------------------------
                            alphaM*tauM+alphaF*gamma*dt


                    /                               \
                   |              / n+af       \     |
                   |  nabla Dp , | u    o nabla | v  |
                   |              \            /     |
                    \                               /
              */

              elemat(vi*3    , ui*3 + 2) += fac_afgdt_tauM_facMtau*derxy_(0,ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 2) += fac_afgdt_tauM_facMtau*derxy_(1,ui)*conv_c_af_(vi) ;

            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

          if (newton)
          {
            const double fac_afgdt_svelaf_x               = fac*afgdt*svelaf_(0);
            const double fac_afgdt_svelaf_y               = fac*afgdt*svelaf_(1);

            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {

                /* SUPG stabilisation --- convection


               factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaF * gamma * dt
                        alphaM*tauM+alphaF*gamma*dt

                    /                                               \
                   |    /            \   n+af    / n+af        \     |
                   |   | Dacc o nabla | u     , | u     o nabla | v  |
                   |    \            /           \             /     |
                    \                                               /

                */
                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*
                                              (conv_c_af_(vi)*conv_r_af_(0, 0, ui)) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*
                                              (conv_c_af_(vi)*conv_r_af_(0, 1, ui)) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*
                                              (conv_c_af_(vi)*conv_r_af_(1, 0, ui)) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*
                                              (conv_c_af_(vi)*conv_r_af_(1, 1, ui)) ;

                /* SUPG stabilisation --- subscale velocity, nonlinear part from testfunction

                factor:
                          alphaF * gamma * dt


                    /                            \
                   |  ~n+af    /            \     |
                   |  u     , | Dacc o nabla | v  |
                   |   (i)     \            /     |
                    \                            /

                */

                elemat(vi*3    , ui*3    ) -= fac_afgdt_svelaf_x*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) -= fac_afgdt_svelaf_x*funct_(ui)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3    ) -= fac_afgdt_svelaf_y*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_svelaf_y*funct_(ui)*derxy_(1,vi) ;

#if 0
                /* SUPG stabilisation --- inertia, lineariation of testfunction

                factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaF * gamma * dt
                        alphaM*tauM+alphaF*gamma*dt

                    /                               \
                   |     n+am     /            \     |
                   |  acc      , | Dacc o nabla | v  |
                   |              \            /     |
                    \                               /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(0)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(0)*derxy_(1,vi) ;
                elemat(vi*3    , ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(0)*derxy_(2,vi) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(1)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(1)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(1)*derxy_(2,vi) ;
                elemat(vi*3 + 2, ui*3)     += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(2)*derxy_(0,vi) ;
                elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(2)*derxy_(1,vi) ;
                elemat(vi*3 + 2, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*accintam_(2)*derxy_(2,vi) ;
#endif


#if 0
                /* SUPG stabilisation --- convection, lineariation of testfunction

                factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaF * gamma * dt
                        alphaM*tauM+alphaF*gamma*dt

                    /                                               \
                   |    / n+af        \   n+af    /            \     |
                   |   | u     o nabla | u     , | Dacc o nabla | v  |
                   |    \             /           \            /     |
                    \                                               /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(0)*funct_(ui)*derxy_(0,vi);
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(0)*funct_(ui)*derxy_(1,vi);
                elemat(vi*3    , ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(0)*funct_(ui)*derxy_(2,vi);
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(1)*funct_(ui)*derxy_(0,vi);
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(1)*funct_(ui)*derxy_(1,vi);
                elemat(vi*3 + 1, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(1)*funct_(ui)*derxy_(2,vi);
                elemat(vi*3 + 2, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(2)*funct_(ui)*derxy_(0,vi);
                elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(2)*funct_(ui)*derxy_(1,vi);
                elemat(vi*3 + 2, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*convaf_old_(2)*funct_(ui)*derxy_(2,vi);
#endif



#if 0
                /* SUPG stabilisation ---  pressure, lineariation of testfunction

                factor:
                           alphaF*gamma*dt*tauM
                        --------------------------- * alphaF * gamma * dt
                        alphaM*tauM+alphaF*gamma*dt

                    /                                 \
                   |         n+1    /            \     |
                   |  nabla p    , | Dacc o nabla | v  |
                   |                \            /     |
                    \                                 /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(0)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(0)*derxy_(1,vi) ;
                elemat(vi*3    , ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(0)*derxy_(2,vi) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(1)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(1)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(1)*derxy_(2,vi) ;
                elemat(vi*3 + 2, ui*3    ) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(2)*derxy_(0,vi) ;
                elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(2)*derxy_(1,vi) ;
                elemat(vi*3 + 2, ui*3 + 2) += fac_afgdt_tauM_afgdt_facMtau*funct_(ui)*pderxynp_(2)*derxy_(2,vi) ;
#endif

#if 0
                /* SUPG stabilisation --- diffusion, lineariation of testfunction
               factor:
                               alphaF*gamma*tauM*dt
                  - 2 * nu  --------------------------- * alphaF * gamma * dt
                            alphaM*tauM+alphaF*gamma*dt

                    /                                            \
                   |               / n+af \    /            \     |
                   |  nabla o eps | u      |, | Dacc o nabla | v  |
                   |               \      /    \            /     |
                    \                                            /
                */
                elemat(vi*3, ui*3)         -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(0)*funct_(ui)*derxy_(0, vi) ;
                elemat(vi*3, ui*3 + 1)     -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(0)*funct_(ui)*derxy_(1, vi) ;
                elemat(vi*3, ui*3 + 2)     -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(0)*funct_(ui)*derxy_(2, vi) ;
                elemat(vi*3 + 1, ui*3)     -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(1)*funct_(ui)*derxy_(0, vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(1)*funct_(ui)*derxy_(1, vi) ;
                elemat(vi*3 + 1, ui*3 + 2) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(1)*funct_(ui)*derxy_(2, vi) ;
                elemat(vi*3 + 2, ui*3)     -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(2)*funct_(ui)*derxy_(0, vi) ;
                elemat(vi*3 + 2, ui*3 + 1) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(2)*funct_(ui)*derxy_(1, vi) ;
                elemat(vi*3 + 2, ui*3 + 2) -= fac_two_visceff_afgdt_afgdt_tauM_facMtau*viscaf_old_(2)*funct_(ui)*derxy_(2, vi) ;

#endif
              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)
          } // end if newton and supg

        } // end supg stabilisation

        if(vstab == Fluid2::viscous_stab_usfem || vstab == Fluid2::viscous_stab_gls)
        {
          const double fac_alphaM_two_visc_afgdt_tauM_facMtau         = vstabfac*fac*alphaM*2.0*visc*afgdt*tauM*facMtau;
          const double fac_afgdt_two_visc_afgdt_tauM_facMtau          = vstabfac*fac*afgdt*2.0*visc*afgdt*tauM*facMtau;
          const double fac_afgdt_four_visceff_visc_afgdt_tauM_facMtau = vstabfac*fac*afgdt*4.0*visceff*visc*afgdt*tauM*facMtau;
          const double fac_two_visc_afgdt_tauM_facMtau                = vstabfac*fac*2.0*visc*afgdt*tauM*facMtau;

          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //            VISCOUS STABILISATION TERMS FOR (A)GLS
          //
          //---------------------------------------------------------------
          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            const double fac_alphaM_two_visc_afgdt_tauM_facMtau_funct_ui   = fac_alphaM_two_visc_afgdt_tauM_facMtau*funct_(ui);
            const double fac_afgdt_two_visc_afgdt_tauM_facMtau_conv_c_af_ui= fac_afgdt_two_visc_afgdt_tauM_facMtau*conv_c_af_(ui);
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* viscous stabilisation --- inertia     */

              /* factor:

                             alphaF*gamma*tauM*dt
        +(-)alphaM*2*nu* ---------------------------
                         alphaM*tauM+alphaF*gamma*dt

                     /                    \
                    |                      |
                    |  Dacc , div eps (v)  |
                    |                      |
                     \                    /
              */
              elemat(vi*3    , ui*3    ) += fac_alphaM_two_visc_afgdt_tauM_facMtau_funct_ui*viscs2_(0,0,vi);
              elemat(vi*3    , ui*3 + 1) += fac_alphaM_two_visc_afgdt_tauM_facMtau_funct_ui*viscs2_(1,0,vi);
              elemat(vi*3 + 1, ui*3    ) += fac_alphaM_two_visc_afgdt_tauM_facMtau_funct_ui*viscs2_(0,1,vi);
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM_two_visc_afgdt_tauM_facMtau_funct_ui*viscs2_(1,1,vi);

              /* viscous stabilisation --- convection */
              /*  factor:
                                         alphaF*gamma*dt*tauM
            +(-)alphaF*gamma*dt*2*nu* ---------------------------
                                      alphaM*tauM+alphaF*gamma*dt

                       /                                  \
                      |  / n+af       \                    |
                      | | u    o nabla | Dacc, div eps (v) |
                      |  \            /                    |
                       \                                  /

              */

              elemat(vi*3    , ui*3    ) += fac_afgdt_two_visc_afgdt_tauM_facMtau_conv_c_af_ui*viscs2_(0, 0, vi) ;
              elemat(vi*3    , ui*3 + 1) += fac_afgdt_two_visc_afgdt_tauM_facMtau_conv_c_af_ui*viscs2_(1, 0, vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_afgdt_two_visc_afgdt_tauM_facMtau_conv_c_af_ui*viscs2_(0, 1, vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_two_visc_afgdt_tauM_facMtau_conv_c_af_ui*viscs2_(1, 1, vi) ;

              /* viscous stabilisation --- diffusion  */

              /* factor:

                                           alphaF*gamma*tauM*dt
            -(+)alphaF*gamma*dt*4*nu*nu ---------------------------
                                        alphaM*tauM+alphaF*gamma*dt

                    /                                   \
                   |               /    \                |
                   |  nabla o eps | Dacc | , div eps (v) |
                   |               \    /                |
                    \                                   /
              */
              elemat(vi*3    , ui*3    ) -= fac_afgdt_four_visceff_visc_afgdt_tauM_facMtau*
                                            (viscs2_(0, 0, ui)*viscs2_(0, 0, vi)
                                             +
                                             viscs2_(1, 0, ui)*viscs2_(1, 0, vi)) ;

              elemat(vi*3    , ui*3 + 1) -= fac_afgdt_four_visceff_visc_afgdt_tauM_facMtau*
                                            (viscs2_(0, 0, vi)*viscs2_(0, 1, ui)
                                             +
                                             viscs2_(1, 0, vi)*viscs2_(1, 1, ui)) ;

              elemat(vi*3 + 1, ui*3    ) -= fac_afgdt_four_visceff_visc_afgdt_tauM_facMtau*
                                            (viscs2_(0, 0, ui)*viscs2_(0, 1, vi)
                                             +
                                             viscs2_(1, 0, ui)*viscs2_(1, 1, vi)) ;

              elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_four_visceff_visc_afgdt_tauM_facMtau*
                                            (viscs2_(0, 1, ui)*viscs2_(0, 1, vi)
                                             +
                                             viscs2_(1, 1, ui)*viscs2_(1, 1, vi)) ;


              /* viscous stabilisation --- pressure   */

              /* factor:

                          alphaF*gamma*tauM*dt
            +(-)2*nu * ---------------------------
                       alphaM*tauM+alphaF*gamma*dt


                    /                        \
                   |                          |
                   |  nabla Dp , div eps (v)  |
                   |                          |
                    \                        /
              */
              elemat(vi*3    , ui*3 + 2) += fac_two_visc_afgdt_tauM_facMtau*
                                            (derxy_(0,ui)*viscs2_(0,0,vi)
                                             +
                                             derxy_(1,ui)*viscs2_(1,0,vi)) ;
              elemat(vi*3 + 1, ui*3 + 2) += fac_two_visc_afgdt_tauM_facMtau*
                                            (derxy_(0,ui)*viscs2_(0,1,vi)
                                             +
                                             derxy_(1,ui)*viscs2_(1,1,vi)) ;

            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

          if (newton)
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {

                /* viscous stabilisation --- convection
                   factor:
                                         alphaF*gamma*dt*tauM
            +(-)alphaF*gamma*dt*2*nu* ---------------------------
                                      alphaM*tauM+alphaF*gamma*dt

                     /                                     \
                    |   /            \   n+af               |
                    |  | Dacc o nabla | u     , div eps (v) |
                    |   \            /                      |
                     \                                     /


                */


                elemat(vi*3     , ui*3    )+= fac_afgdt_two_visc_afgdt_tauM_facMtau*
                                              (viscs2_(0, 0, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               viscs2_(1, 0, vi)*conv_r_af_(1, 0, ui)) ;
                elemat(vi*3     , ui*3 + 1)+= fac_afgdt_two_visc_afgdt_tauM_facMtau*
                                              (viscs2_(0, 0, vi)*conv_r_af_(0, 1, ui)
                                               +
                                               viscs2_(1, 0, vi)*conv_r_af_(1, 1, ui)) ;
                elemat(vi*3 + 1, ui*3     )+= fac_afgdt_two_visc_afgdt_tauM_facMtau*
                                              (viscs2_(0, 1, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               viscs2_(1, 1, vi)*conv_r_af_(1, 0, ui)) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_two_visc_afgdt_tauM_facMtau*
                                              (viscs2_(0, 1, vi)*conv_r_af_(0, 1, ui)
                                               +
                                               viscs2_(1, 1, vi)*conv_r_af_(1, 1, ui)) ;

              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)

          } // end if (a)gls and newton

        } // end (a)gls stabilisation

        if(cstab == Fluid2::continuity_stab_yes)
        {

          const double fac_gamma_dt_tauC = fac*gamma*dt*tauC;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /*  factor: +gamma*dt*tauC

                    /                          \
                   |                            |
                   | nabla o Dacc  , nabla o v  |
                   |                            |
                    \                          /
              */

              elemat(vi*3    , ui*3    ) += fac_gamma_dt_tauC*derxy_(0,ui)*derxy_(0,vi) ;
              elemat(vi*3    , ui*3 + 1) += fac_gamma_dt_tauC*derxy_(1,ui)*derxy_(0,vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_gamma_dt_tauC*derxy_(0,ui)*derxy_(1,vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_gamma_dt_tauC*derxy_(1,ui)*derxy_(1,vi) ;

            } // end loop rows vi (test functions for matrix)
          } // end loop columns ui (solution for matrix, test function for vector)

        } // end cstab
        else if(cstab == Fluid2::continuity_stab_td)
        {

          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //                  CONTINUITY STABILISATION
          //
          //---------------------------------------------------------------

          const double fac_gamma_dt_dt_factauC          = fac*gamma*dt*dt*factauC;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /*
                                 tauC * dt
            factor: +gamma* dt * ---------
                                 tauC + dt
                    /                          \
                   |                            |
                   | nabla o Dacc  , nabla o v  |
                   |                            |
                    \                          /
              */

              elemat(vi*3    , ui*3    ) += fac_gamma_dt_dt_factauC*derxy_(0,ui)*derxy_(0,vi) ;
              elemat(vi*3    , ui*3 + 1) += fac_gamma_dt_dt_factauC*derxy_(1,ui)*derxy_(0,vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_gamma_dt_dt_factauC*derxy_(0,ui)*derxy_(1,vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_gamma_dt_dt_factauC*derxy_(1,ui)*derxy_(1,vi) ;

            } // end loop rows (test functions for matrix)
          } // end loop rows (solution for matrix, test function for vector)

        }
        // end continuity stabilisation

        if(cross == Fluid2::cross_stress_stab)
        {
          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //       RESIDUAL BASED VMM STABILISATION --- CROSS STRESS
          //
          //---------------------------------------------------------------

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {

              /*  factor:

               +alphaF*gamma*dt

                          /                          \
                         |  /~n+af       \            |
                         | | u    o nabla | Dacc , v  |
                         |  \            /            |
                          \                          /
              */
              elemat(vi*3    , ui*3    ) += fac*afgdt*conv_subaf_(ui)*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac*afgdt*conv_subaf_(ui)*funct_(vi) ;
            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

        } // end cross
      } // end if compute_elemat

      //---------------------------------------------------------------
      //---------------------------------------------------------------
      //
      //                       RIGHT HAND SIDE
      //
      //---------------------------------------------------------------
      //---------------------------------------------------------------

      if(inertia == Fluid2::inertia_stab_keep)
      {

        const double fac_sacc_plus_resM_not_partially_integrated_x =fac*(-svelaf_(0)/tauM-pderxynp_(0)+2*visceff*viscaf_old_(0)) ;
        const double fac_sacc_plus_resM_not_partially_integrated_y =fac*(-svelaf_(1)/tauM-pderxynp_(1)+2*visceff*viscaf_old_(1)) ;

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          //---------------------------------------------------------------
          //
          //     GALERKIN PART I AND SUBSCALE ACCELERATION STABILISATION
          //
          //---------------------------------------------------------------
          /*  factor: +1

               /             \     /
              |   ~ n+am      |   |     n+am    / n+af        \   n+af
              |  acc     , v  | + |  acc     + | u     o nabla | u     +
              |     (i)       |   |     (i)     \ (i)         /   (i)
               \             /     \

                                                   \
                                        n+af        |
                                     - f       , v  |
                                                    |
                                                   /

             using
                                                        /
                        ~ n+am        1.0      ~n+af   |    n+am
                       acc     = - --------- * u     - | acc     +
                          (i)           n+af    (i)    |    (i)
                                   tau_M                \

                                    / n+af        \   n+af            n+1
                                 + | u     o nabla | u     + nabla o p    -
                                    \ (i)         /   (i)             (i)

                                                            / n+af \
                                 - 2 * nu * grad o epsilon | u      | -
                                                            \ (i)  /
                                         \
                                    n+af  |
                                 - f      |
                                          |
                                         /

          */

          elevec[ui*3    ] -= fac_sacc_plus_resM_not_partially_integrated_x*funct_(ui) ;
          elevec[ui*3 + 1] -= fac_sacc_plus_resM_not_partially_integrated_y*funct_(ui) ;
        }

        //---------------------------------------------------------------
        //
        //   GALERKIN PART 2 (REMAINING EXPRESSIONS)
        //
        //---------------------------------------------------------------
        {

          const double fac_divunp  = fac*divunp;
          const double fac_visceff = fac*visceff;
          const double fac_prenp_  = fac*prenp_ ;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            /* pressure */

            /*  factor: -1

               /                  \
              |   n+1              |
              |  p    , nabla o v  |
              |                    |
               \                  /
            */

            elevec[ui*3    ] += fac_prenp_*derxy_(0,ui) ;
            elevec[ui*3 + 1] += fac_prenp_*derxy_(1,ui) ;

            /* viscous term */

            /*  factor: +2*nu

               /                            \
              |       / n+af \         / \   |
              |  eps | u      | , eps | v |  |
              |       \      /         \ /   |
               \                            /
            */

            elevec[ui*3    ] -= fac_visceff*
                                (derxy_(0,ui)*vderxyaf_(0,0)*2.0
                                 +
                                 derxy_(1,ui)*vderxyaf_(0,1)
                                 +
                                 derxy_(1,ui)*vderxyaf_(1,0)
                                 +
                                 derxy_(2,ui)*vderxyaf_(0,2)
                                 +
                                 derxy_(2,ui)*vderxyaf_(2,0)) ;
            elevec[ui*3 + 1] -= fac_visceff*
                                (derxy_(0,ui)*vderxyaf_(0,1)
                                 +
                                 derxy_(0,ui)*vderxyaf_(1,0)
                                 +
                                 derxy_(1,ui)*vderxyaf_(1,1)*2.0
                                 +
                                 derxy_(2,ui)*vderxyaf_(1,2)
                                 +
                                 derxy_(2,ui)*vderxyaf_(2,1)) ;


            /* continuity equation */

            /*  factor: +1

               /                \
              |          n+1     |
              | nabla o u   , q  |
              |                  |
               \                /
            */

            elevec[ui*3 + 2] -= fac_divunp*funct_(ui);

          } // end loop rows (solution for matrix, test function for vector)
        }
      }
      else
      {
        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {

          //---------------------------------------------------------------
          //
          //                       GALERKIN PART
          //
          //---------------------------------------------------------------

          /* inertia terms */

          /*  factor: +1

               /             \
              |     n+am      |
              |  acc     , v  |
              |               |
               \             /
          */

          elevec[ui*3    ] -= fac*funct_(ui)*accintam_(0) ;
          elevec[ui*3 + 1] -= fac*funct_(ui)*accintam_(1) ;

          /* convection */

          /*  factor: +1

               /                             \
              |  / n+af       \    n+af       |
              | | u    o nabla |  u      , v  |
              |  \            /               |
               \                             /
          */

          elevec[ui*3    ] -= fac*(velintaf_(0)*conv_r_af_(0,0,ui)
                                   +
                                   velintaf_(1)*conv_r_af_(0,1,ui)) ;
          elevec[ui*3 + 1] -= fac*(velintaf_(0)*conv_r_af_(1,0,ui)
                                   +
                                   velintaf_(1)*conv_r_af_(1,1,ui)) ;

          /* pressure */

          /*  factor: -1

               /                  \
              |   n+1              |
              |  p    , nabla o v  |
              |                    |
               \                  /
          */

          elevec[ui*3    ] += fac*prenp_*derxy_(0,ui) ;
          elevec[ui*3 + 1] += fac*prenp_*derxy_(1,ui) ;

          /* viscous term */

          /*  factor: +2*nu

               /                            \
              |       / n+af \         / \   |
              |  eps | u      | , eps | v |  |
              |       \      /         \ /   |
               \                            /
          */

          elevec[ui*3    ] -= visceff*fac*
                              (derxy_(0,ui)*vderxyaf_(0,0)*2.0
                               +
                               derxy_(1,ui)*vderxyaf_(0,1)
                               +
                               derxy_(1,ui)*vderxyaf_(1,0));
          elevec[ui*3 + 1] -= visceff*fac*
                              (derxy_(0,ui)*vderxyaf_(0,1)
                               +
                               derxy_(0,ui)*vderxyaf_(1,0)
                               +
                               derxy_(1,ui)*vderxyaf_(1,1)*2.0);


          /* body force (dead load...) */

          /*  factor: -1

               /           \
              |   n+af      |
              |  f     , v  |
              |             |
               \           /
          */

          elevec[ui*3    ] += fac*funct_(ui)*bodyforceaf_(0);
          elevec[ui*3 + 1] += fac*funct_(ui)*bodyforceaf_(1);

          /* continuity equation */

          /*  factor: +1

               /                \
              |          n+1     |
              | nabla o u   , q  |
              |                  |
               \                /
          */

          elevec[ui*3 + 2] -= fac*funct_(ui)*divunp;

        } // end loop rows (solution for matrix, test function for vector)
      }


      if(pspg == Fluid2::pstab_use_pspg)
      {
        const double fac_svelnpx                      = fac*svelnp(0,iquad);
        const double fac_svelnpy                      = fac*svelnp(1,iquad);

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //                    PRESSURE STABILISATION
          //
          //---------------------------------------------------------------
          /* factor: -1

                       /                 \
                      |  ~n+1             |
                      |  u    , nabla  q  |
                      |   (i)             |
                       \                 /
          */

          elevec[ui*3 + 2] += fac_svelnpx*derxy_(0,ui)+fac_svelnpy*derxy_(1,ui);

        } // end loop rows (solution for matrix, test function for vector)

      }

      if(supg == Fluid2::convective_stab_supg)
      {

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
        //---------------------------------------------------------------
        //
        //                     STABILISATION PART
        //         SUPG STABILISATION FOR CONVECTION DOMINATED FLOWS
        //
        //---------------------------------------------------------------
          /*
                  /                             \
                 |  ~n+af    / n+af        \     |
                 |  u     , | u     o nabla | v  |
                 |           \             /     |
                  \                             /

          */

          elevec[ui*3    ] += fac*conv_c_af_(ui)*svelaf_(0);
          elevec[ui*3 + 1] += fac*conv_c_af_(ui)*svelaf_(1);

        } // end loop rows (solution for matrix, test function for vector)

      }

      if (vstab != Fluid2::viscous_stab_none)
      {

        const double fac_two_visc_svelaf_x = vstabfac*fac*2.0*visc*svelaf_(0);
        const double fac_two_visc_svelaf_y = vstabfac*fac*2.0*visc*svelaf_(1);

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //             VISCOUS STABILISATION (FOR (A)GLS)
          //
          //---------------------------------------------------------------

          /*
                 /                      \
                |  ~n+af                 |
                |  u      , div eps (v)  |
                |                        |
                 \                      /

          */
          elevec[ui*3    ] += fac_two_visc_svelaf_x*viscs2_(0, 0, ui)
                              +
                              fac_two_visc_svelaf_y*viscs2_(1, 0, ui);

          elevec[ui*3 + 1] += fac_two_visc_svelaf_x*viscs2_(0, 1, ui)
                              +
                              fac_two_visc_svelaf_y*viscs2_(1, 1, ui);

        } // end loop rows (solution for matrix, test function for vector)

      } // endif (a)gls

      if(cstab == Fluid2::continuity_stab_yes)
      {

        const double fac_tauC = fac*tauC;
        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /* factor: +tauC

                  /                          \
                 |           n+1              |
                 |  nabla o u    , nabla o v  |
                 |                            |
                  \                          /
          */

          elevec[ui*3    ] -= fac_tauC*divunp*derxy_(0,ui) ;
          elevec[ui*3 + 1] -= fac_tauC*divunp*derxy_(1,ui) ;
        } // end loop rows
      }
      else if (cstab == Fluid2::continuity_stab_td)
      {

        const double fac_sprenp                       = fac*sprenp(iquad);

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //                  CONTINUITY STABILISATION
          //
          //---------------------------------------------------------------


          /* factor: -1

                       /                  \
                      |  ~n+1              |
                      |  p    , nabla o v  |
                      |   (i)              |
                       \                  /
          */
          elevec[ui*3    ] += fac_sprenp*derxy_(0,ui) ;
          elevec[ui*3 + 1] += fac_sprenp*derxy_(1,ui) ;
        } // end loop rows (solution for matrix, test function for vector)
      }

      if(cross == Fluid2::cross_stress_stab_only_rhs || cross == Fluid2::cross_stress_stab)
      {
        //---------------------------------------------------------------
        //
        //                     STABILISATION PART
        //       RESIDUAL BASED VMM STABILISATION --- CROSS STRESS
        //
        //---------------------------------------------------------------
        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {
          /* factor:

                  /                           \
                 |   ~n+af           n+af      |
                 | ( u    o nabla ) u     , v  |
                 |    (i)            (i)       |
                  \                           /
          */
          elevec[ui*3    ] -= fac*convsubaf_old_(0)*funct_(ui);
          elevec[ui*3 + 1] -= fac*convsubaf_old_(1)*funct_(ui);
        }
      }

      if(reynolds == Fluid2::reynolds_stress_stab_only_rhs)
      {
        //---------------------------------------------------------------
        //
        //                     STABILISATION PART
        //     RESIDUAL BASED VMM STABILISATION --- REYNOLDS STRESS
        //
        //---------------------------------------------------------------

        for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
        {

          /* factor:

                  /                             \
                 |  ~n+af      ~n+af             |
                 |  u      , ( u    o nabla ) v  |
                 |                               |
                  \                             /
          */
          elevec[ui*3    ] += fac*(svelaf_(0)*derxy_(0,ui)
                                   +
                                   svelaf_(1)*derxy_(1,ui))*svelaf_(0);
          elevec[ui*3 + 1] += fac*(svelaf_(0)*derxy_(0,ui)
                                   +
                                   svelaf_(1)*derxy_(1,ui))*svelaf_(1);

        } // end loop rows (solution for matrix, test function for vector)
      }
    }
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//
//     ELEMENT FORMULATION BASED ON QUASISTATIC SUBSCALES
//
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
    else
    {
      const double tauM   = tau_(0);
      const double tauMp  = tau_(1);
      const double tauC   = tau_(2);

      //--------------------------------------------------------------
      //--------------------------------------------------------------
      //
      //                       SYSTEM MATRIX
      //
      //--------------------------------------------------------------
      //--------------------------------------------------------------
      if(compute_elemat)
      {

        //---------------------------------------------------------------
        //
        //                       GALERKIN PART
        //
        //---------------------------------------------------------------
        {

          const double fac_alphaM        = fac*alphaM;
          const double fac_afgdt         = fac*afgdt;
          const double fac_visceff_afgdt = fac*visceff*afgdt;
          const double fac_gamma_dt      = fac*gamma*dt;
          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {

              /*
                inertia term (intermediate)

                factor: +alphaM

                 /          \
                |            |
                |  Dacc , v  |
                |            |
                 \          /
              */
              elemat(vi*3    , ui*3    ) += fac_alphaM*funct_(vi)*funct_(ui) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM*funct_(vi)*funct_(ui) ;

              /* convection (intermediate)

               factor:

               +alphaF*gamma*dt

                          /                          \
                         |  / n+af       \            |
                         | | u    o nabla | Dacc , v  |
                         |  \            /            |
                          \                          /
              */
              elemat(vi*3    , ui*3    ) += fac_afgdt*funct_(vi)*conv_c_af_(ui) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt*funct_(vi)*conv_c_af_(ui) ;

              /* pressure (implicit) */

              /*  factor: -1

                 /                \
                |                  |
                |  Dp , nabla o v  |
                |                  |
                 \                /
              */

              elemat(vi*3    , ui*3 + 2) -= fac*funct_(ui)*derxy_(0, vi) ;
              elemat(vi*3 + 1, ui*3 + 2) -= fac*funct_(ui)*derxy_(1, vi) ;

              /* viscous term (intermediate) */

              /*  factor: +2*nu*alphaF*gamma*dt

                 /                          \
                |       /    \         / \   |
                |  eps | Dacc | , eps | v |  |
                |       \    /         \ /   |
                 \                          /
              */

              elemat(vi*3    , ui*3    ) += fac_visceff_afgdt*(2.0*derxy_(0,ui)*derxy_(0,vi)
                                                               +
                                                               derxy_(1,ui)*derxy_(1,vi)) ;
              elemat(vi*3    , ui*3 + 1) += fac_visceff_afgdt*derxy_(0,ui)*derxy_(1,vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_visceff_afgdt*derxy_(1,ui)*derxy_(0,vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_visceff_afgdt*(derxy_(0,ui)*derxy_(0,vi)
                                                               +
                                                               2.0*derxy_(1,ui)*derxy_(1,vi)) ;

              /* continuity equation (implicit) */

              /*  factor: +gamma*dt

                 /                  \
                |                    |
                | nabla o Dacc  , q  |
                |                    |
                 \                  /
              */

              elemat(vi*3 + 2, ui*3    ) += fac_gamma_dt*funct_(vi)*derxy_(0,ui) ;
              elemat(vi*3 + 2, ui*3 + 1) += fac_gamma_dt*funct_(vi)*derxy_(1,ui) ;

            } // end loop rows (test functions for matrix)
          } // end loop rows (solution for matrix, test function for vector)
          if (ele->is_ale_)
          {
            for (int ui=0; ui<iel_; ++ui)
            {
              for (int vi=0; vi<iel_; ++vi)
              {
                /*  reduced convection through grid motion

                factor:

                +alphaF*gamma*dt



                       /                      \
                      |  /          \          |
                    - | | u  o nabla | Du , v  |
                      |  \ G        /          |
                       \                      /

                */
                elemat(vi*3    , ui*3    ) -= fac_afgdt*funct_(vi)*conv_g_af_(ui) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt*funct_(vi)*conv_g_af_(ui) ;
              }

            }
          } // end if is_ale

          if (newton)
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {
                /* convection (intermediate)

                 factor:

                        +alphaF*gamma*dt


                         /                            \
                        |  /            \   n+af       |
                        | | Dacc o nabla | u      , v  |
                        |  \            /              |
                         \                            /
                */
                elemat(vi*3    , ui*3    ) += fac_afgdt*funct_(vi)*conv_r_af_(0, 0, ui) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt*funct_(vi)*conv_r_af_(0, 1, ui) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt*funct_(vi)*conv_r_af_(1, 0, ui) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt*funct_(vi)*conv_r_af_(1, 1, ui) ;
              } // end loop rows (test functions for matrix)
            } // end loop rows (solution for matrix, test function for vector)
          }
        }

        if(pspg == Fluid2::pstab_use_pspg)
        {
          const double fac_alphaM_tauMp            = fac*alphaM*tauMp;
          const double fac_afgdt_tauMp             = fac*afgdt*tauMp;
          const double fac_two_visceff_afgdt_tauMp = fac*2.0*visceff*afgdt*tauMp;
          const double fac_tauMp                   = fac*tauMp;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* pressure stabilisation --- inertia    */

              /* factor: +alphaM*tauMp

                                /                \
                               |                  |
                               |  Dacc , nabla q  |
                               |                  |
                                \                /
              */

              elemat(vi*3 + 2, ui*3    ) += fac_alphaM_tauMp*funct_(ui)*derxy_(0,vi) ;
              elemat(vi*3 + 2, ui*3 + 1) += fac_alphaM_tauMp*funct_(ui)*derxy_(1,vi) ;


              /* pressure stabilisation --- convection */

              /*  factor: +alphaF*gamma*dt*tauMp

                        /                                \
                       |  / n+af       \                  |
                       | | u    o nabla | Dacc , nabla q  |
                       |  \            /                  |
                        \                                /

              */

              elemat(vi*3 + 2, ui*3    ) += fac_afgdt_tauMp*conv_c_af_(ui)*derxy_(0,vi) ;
              elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_tauMp*conv_c_af_(ui)*derxy_(1,vi) ;

              /* pressure stabilisation --- diffusion  */

              /* factor: -2*nu*alphaF*gamma*dt*tauMp

                    /                                \
                   |               /    \             |
                   |  nabla o eps | Dacc | , nabla q  |
                   |               \    /             |
                    \                                /
              */

              elemat(vi*3 + 2, ui*3    ) -= fac_two_visceff_afgdt_tauMp*
                                            (derxy_(0,vi)*viscs2_(0,0,ui)
                                             +
                                             derxy_(1,vi)*viscs2_(0,1,ui)) ;
              elemat(vi*3 + 2, ui*3 + 1) -= fac_two_visceff_afgdt_tauMp*
                                            (derxy_(0,vi)*viscs2_(0,1,ui)
                                             +
                                             derxy_(1,vi)*viscs2_(1,1,ui)) ;

              /* pressure stabilisation --- pressure   */

              /* factor: +tauMp

                    /                    \
                   |                      |
                   |  nabla Dp , nabla q  |
                   |                      |
                    \                    /
              */

              elemat(vi*3 + 2, ui*3 + 2) += fac_tauMp*
                                            (derxy_(0,ui)*derxy_(0,vi)
                                             +
                                             derxy_(1,ui)*derxy_(1,vi));

            } // end loop rows (test functions for matrix)
          } // end loop rows (solution for matrix, test function for vector)
          if (ele->is_ale_)
          {
            for (int ui=0; ui<iel_; ++ui)
            {
              for (int vi=0; vi<iel_; ++vi)
              {
                /*  reduced convection through grid motion


                       /                          \
                      |  /          \              |
                    - | | u  o nabla | Du , grad q |
                      |  \ G        /              |
                       \                          /

              */
              elemat(vi*3 + 2, ui*3    ) -= fac_afgdt_tauMp*conv_g_af_(ui)*derxy_(0, vi) ;
              elemat(vi*3 + 2, ui*3 + 1) -= fac_afgdt_tauMp*conv_g_af_(ui)*derxy_(1, vi) ;
              }// ui
            }//vi
          } // end if is_ale

          if (newton)
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {

                /* pressure stabilisation --- convection */

                /*  factor: +alphaF*gamma*dt*tauMp

                       /                                  \
                      |  /            \   n+af             |
                      | | Dacc o nabla | u      , nabla q  |
                      |  \            /                    |
                       \                                  /
                */

                elemat(vi*3 + 2, ui*3    ) += fac_afgdt_tauMp*
                                              (derxy_(0,vi)*conv_r_af_(0,0,ui)
                                               +
                                               derxy_(1,vi)*conv_r_af_(1,0,ui)) ;
                elemat(vi*3 + 2, ui*3 + 1) += fac_afgdt_tauMp*
                                              (derxy_(0,vi)*conv_r_af_(0,1,ui)
                                               +
                                               derxy_(1,vi)*conv_r_af_(1,1,ui)) ;
              } // end loop rows (test functions for matrix)
            } // end loop rows (solution for matrix, test function for vector)
          }
        }

        if(supg == Fluid2::convective_stab_supg)
        {
          const double fac_alphaM_tauM            = fac*tauM*alphaM;
          const double fac_afgdt_tauM             = fac*tauM*afgdt;
          const double fac_two_visceff_afgdt_tauM = fac*tauM*afgdt*2.0*visceff;
          const double fac_tauM                   = fac*tauM;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* SUPG stabilisation --- inertia     */

              /* factor: +alphaM*tauM

                    /                           \
                   |          / n+af       \     |
                   |  Dacc , | u    o nabla | v  |
                   |          \            /     |
                    \                           /
              */

              elemat(vi*3    , ui*3    ) += fac_alphaM_tauM*funct_(ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_alphaM_tauM*funct_(ui)*conv_c_af_(vi) ;

              /* SUPG stabilisation --- convection  */

              /* factor: +alphaF*gamma*dt*tauM


                    /                                               \
                   |    / n+af        \          / n+af        \     |
                   |   | u     o nabla | Dacc , | u     o nabla | v  |
                   |    \             /          \             /     |
                    \                                               /

              */

              elemat(vi*3    , ui*3    ) += fac_afgdt_tauM*conv_c_af_(ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM*conv_c_af_(ui)*conv_c_af_(vi) ;

              /* SUPG stabilisation --- diffusion   */

              /* factor: -2*nu*alphaF*gamma*dt*tauM


                    /                                            \
                   |               /     \    / n+af        \     |
                   |  nabla o eps | Dacc  |, | u     o nabla | v  |
                   |               \     /    \             /     |
                    \                                            /

              */

              elemat(vi*3    , ui*3    ) -= fac_two_visceff_afgdt_tauM*conv_c_af_(vi)*viscs2_(0, 0, ui) ;
              elemat(vi*3    , ui*3 + 1) -= fac_two_visceff_afgdt_tauM*conv_c_af_(vi)*viscs2_(0, 1, ui) ;
              elemat(vi*3 + 1, ui*3    ) -= fac_two_visceff_afgdt_tauM*conv_c_af_(vi)*viscs2_(0, 1, ui) ;
              elemat(vi*3 + 1, ui*3 + 1) -= fac_two_visceff_afgdt_tauM*conv_c_af_(vi)*viscs2_(1, 1, ui) ;

              /* SUPG stabilisation --- pressure    */

              /* factor: +tauM

                    /                               \
                   |              / n+af       \     |
                   |  nabla Dp , | u    o nabla | v  |
                   |              \            /     |
                    \                               /
              */

              elemat(vi*3    , ui*3 + 2) += fac_tauM*derxy_(0,ui)*conv_c_af_(vi) ;
              elemat(vi*3 + 1, ui*3 + 2) += fac_tauM*derxy_(1,ui)*conv_c_af_(vi) ;

            } // end loop rows (test functions for matrix)
          } // end loop rows (solution for matrix, test function for vector)

          if (ele->is_ale_)
          {
            for (int ui=0; ui<iel_; ++ui)
            {
              for (int vi=0; vi<iel_; ++vi)
              {

                /* factor: +alphaF*gamma*dt*tauM


                    /                                               \
                   |    / n+af        \          / n+af        \     |
                   |   | u     o nabla | Dacc , | u     o nabla | v  |
                   |    \ G           /          \             /     |
                    \                                               /

                */
                elemat(vi*3    , ui*3    ) -= fac_afgdt_tauM*conv_g_af_(ui)*conv_c_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*conv_g_af_(ui)*conv_c_af_(vi) ;

                /*
                  factor: tauM*alphaM

                      /                         \
                     |          /          \     |
                  -  |  Dacc , | u  o nabla | v  |
                     |          \ G        /     |
                      \                         /
                */


                elemat(vi*3    , ui*3    ) -= fac_alphaM_tauM*funct_(ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_alphaM_tauM*funct_(ui)*conv_g_af_(vi) ;

                /*

                 /                                           \
                |    / n+1        \          /          \     |
                |   | u    o nabla | Dacc , | u  o nabla | v  |
                |    \ (i)        /          \ G        /     |
                 \                                           /

                */
                elemat(vi*3    , ui*3    ) -= fac_afgdt_tauM*conv_c_af_(ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*conv_c_af_(ui)*conv_g_af_(vi) ;

                /*
                      /                             \
                     |              /          \     |
                     |  nabla Dp , | u  o nabla | v  |
                     |              \ G        /     |
                      \                             /
                */
                elemat(vi*3    , ui*3 + 2) -= fac_tauM*derxy_(0, ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 2) -= fac_tauM*derxy_(1, ui)*conv_g_af_(vi) ;

                /*
                  /                                        \
                 |               /    \    /          \     |
                 |  nabla o eps | Dacc |, | u  o nabla | v  |
                 |               \    /    \ G        /     |
                  \                                        /
                */
                elemat(vi*3, ui*3)         += fac_two_visceff_afgdt_tauM*viscs2_(0, 0, ui)*conv_g_af_(vi) ;
                elemat(vi*3, ui*3 + 1)     += fac_two_visceff_afgdt_tauM*viscs2_(0, 1, ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3)     += fac_two_visceff_afgdt_tauM*viscs2_(0, 1, ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_two_visceff_afgdt_tauM*viscs2_(1, 1, ui)*conv_g_af_(vi) ;

                /*

                 /                                         \
                |    /          \          /          \     |
                |   | u  o nabla | Dacc , | u  o nabla | v  |
                |    \ G        /          \ G        /     |
                 \                                         /

                */
                elemat(vi*3, ui*3)         -= fac_afgdt_tauM*conv_g_af_(ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*conv_g_af_(ui)*conv_g_af_(vi) ;
                elemat(vi*3 + 2, ui*3 + 2) -= fac_afgdt_tauM*conv_g_af_(ui)*conv_g_af_(vi) ;
              }// ui
            }//vi


            if (newton)
            {
              for (int ui=0; ui<iel_; ++ui)
              {
                for (int vi=0; vi<iel_; ++vi)
                {
                  /*
                       /                                           \
                      |    /          \   n+1    /            \     |
                      |   | u  o nabla | u    , | Dacc o nabla | v  |
                      |    \ G        /   (i)    \            /     |
                       \                                           /

                  */
                  elemat(vi*3    , ui*3    ) -= fac_afgdt_tauM*( - ugrid_af_(0)*derxy_(0, vi)*conv_r_af_(0, 0, ui)
                                                                 - ugrid_af_(1)*derxy_(0, vi)*conv_r_af_(0, 1, ui)) ;
                  elemat(vi*3    , ui*3 + 1) -= fac_afgdt_tauM*( - ugrid_af_(0)*derxy_(1, vi)*conv_r_af_(0, 0, ui)
                                                                 - ugrid_af_(1)*derxy_(1, vi)*conv_r_af_(0, 1, ui)) ;
                  elemat(vi*3 + 1, ui*3    ) -= fac_afgdt_tauM*( - ugrid_af_(0)*derxy_(0, vi)*conv_r_af_(1, 0, ui)
                                                                 - ugrid_af_(1)*derxy_(0, vi)*conv_r_af_(1, 1, ui)) ;
                  elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*( - ugrid_af_(0)*derxy_(1, vi)*conv_r_af_(1, 0, ui)
                                                                 - ugrid_af_(1)*derxy_(1, vi)*conv_r_af_(1, 1, ui)) ;

                  /*
                       /                                           \
                      |    /            \   n+1    /          \     |
                      |   | Dacc o nabla | u    , | u  o nabla | v  |
                      |    \            /   (i)    \ G        /     |
                       \                                           /
                  */
                  elemat(vi*3    , ui*3    ) -= fac_afgdt_tauM*conv_r_af_(0, 0, ui)*conv_g_af_(vi) ;
                  elemat(vi*3    , ui*3 + 1) -= fac_afgdt_tauM*conv_r_af_(0, 1, ui)*conv_g_af_(vi) ;
                  elemat(vi*3 + 1, ui*3    ) -= fac_afgdt_tauM*conv_r_af_(1, 0, ui)*conv_g_af_(vi) ;
                  elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*conv_r_af_(1, 1, ui)*conv_g_af_(vi) ;
                }// ui
              }//vi
            } // end if newton
          } // end if is_ale


          if (newton)
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {
                /* SUPG stabilisation --- inertia     */

                /* factor: +alphaF*gamma*dt*tauM

                    /                               \
                   |     n+am     /            \     |
                   |  acc      , | Dacc o nabla | v  |
                   |              \            /     |
                    \                               /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM*funct_(ui)*accintam_(0)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM*funct_(ui)*accintam_(0)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM*funct_(ui)*accintam_(1)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM*funct_(ui)*accintam_(1)*derxy_(1,vi) ;


                /* SUPG stabilisation --- convection  */

                /* factor: +alphaF*gamma*dt*tauM

                    /                                               \
                   |    / n+af        \   n+af    /            \     |
                   |   | u     o nabla | u     , | Dacc o nabla | v  |
                   |    \             /           \            /     |
                    \                                               /

                    /                                               \
                   |    /            \   n+af    / n+af        \     |
                   |   | Dacc o nabla | u     , | u     o nabla | v  |
                   |    \            /           \             /     |
                    \                                               /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM*
                                              (conv_c_af_(vi)*conv_r_af_(0, 0, ui)
                                               +
                                               velintaf_(0)*derxy_(0, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               velintaf_(1)*derxy_(0, vi)*conv_r_af_(0, 1, ui)) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM*
                                              (conv_c_af_(vi)*conv_r_af_(0, 1, ui)
                                               +
                                               velintaf_(0)*derxy_(1, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               velintaf_(1)*derxy_(1, vi)*conv_r_af_(0, 1, ui)) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM*
                                              (conv_c_af_(vi)*conv_r_af_(1, 0, ui)
                                               +
                                               velintaf_(0)*derxy_(0, vi)*conv_r_af_(1, 0, ui)
                                               +
                                               velintaf_(1)*derxy_(0, vi)*conv_r_af_(1, 1, ui)) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM*
                                              (conv_c_af_(vi)*conv_r_af_(1, 1, ui)
                                               +
                                               velintaf_(0)*derxy_(1, vi)*conv_r_af_(1, 0, ui)
                                               +
                                               velintaf_(1)*derxy_(1, vi)*conv_r_af_(1, 1, ui)) ;


                /* SUPG stabilisation --- diffusion   */

                /* factor: -2*nu*alphaF*gamma*dt*tauM

                    /                                            \
                   |               / n+af \    /            \     |
                   |  nabla o eps | u      |, | Dacc o nabla | v  |
                   |               \      /    \            /     |
                    \                                            /
                */
                elemat(vi*3    , ui*3    ) -= fac_two_visceff_afgdt_tauM*funct_(ui)*viscaf_old_(0)*derxy_(0, vi) ;
                elemat(vi*3    , ui*3 + 1) -= fac_two_visceff_afgdt_tauM*funct_(ui)*viscaf_old_(0)*derxy_(1, vi) ;
                elemat(vi*3 + 1, ui*3    ) -= fac_two_visceff_afgdt_tauM*funct_(ui)*viscaf_old_(1)*derxy_(0, vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_two_visceff_afgdt_tauM*funct_(ui)*viscaf_old_(1)*derxy_(1, vi) ;

                /* SUPG stabilisation --- pressure    */

                /* factor: +alphaF*gamma*dt*tauM

                    /                                 \
                   |         n+1    /            \     |
                   |  nabla p    , | Dacc o nabla | v  |
                   |                \            /     |
                    \                                 /
                */

                elemat(vi*3    , ui*3    ) += fac_afgdt_tauM*pderxynp_(0)*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) += fac_afgdt_tauM*pderxynp_(0)*funct_(ui)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3    ) += fac_afgdt_tauM*pderxynp_(1)*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_afgdt_tauM*pderxynp_(1)*funct_(ui)*derxy_(1,vi) ;

                /* SUPG stabilisation --- body force, nonlinear part from testfunction */

                /*  factor: -tauM*alphaF*gamma*dt
                    /                            \
                   |   n+af    /            \     |
                   |  f     , | Dacc o nabla | v  |
                   |           \            /     |
                    \                            /

                */
                elemat(vi*3    , ui*3    ) -= fac_afgdt_tauM*edeadaf_(0)*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3    , ui*3 + 1) -= fac_afgdt_tauM*edeadaf_(0)*funct_(ui)*derxy_(1,vi) ;
                elemat(vi*3 + 1, ui*3    ) -= fac_afgdt_tauM*edeadaf_(1)*funct_(ui)*derxy_(0,vi) ;
                elemat(vi*3 + 1, ui*3 + 1) -= fac_afgdt_tauM*edeadaf_(1)*funct_(ui)*derxy_(1,vi) ;

              } // end loop rows (test functions for matrix)
            } // end loop rows (solution for matrix, test function for vector)
          }
        }

        if(vstab == Fluid2::viscous_stab_gls || vstab == Fluid2::viscous_stab_usfem)
        {
          const double fac_two_visc_tauMp                = vstabfac*fac*2.0*visc*tauMp;
          const double fac_two_visc_afgdt_tauMp          = vstabfac*fac*2.0*visc*afgdt*tauMp;
          const double fac_two_visc_alphaM_tauMp         = vstabfac*fac*2.0*visc*alphaM*tauMp;
          const double fac_four_visceff_visc_afgdt_tauMp = vstabfac*fac*4.0*visceff*visc*afgdt*tauMp;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /* viscous stabilisation --- inertia     */

              /* factor: +(-)alphaM*tauMp*2*nu

                    /                    \
                   |                      |
                   |  Dacc , div eps (v)  |
                   |                      |
                    \                    /
              */
              elemat(vi*3    , ui*3    ) += fac_two_visc_alphaM_tauMp*funct_(ui)*viscs2_(0,0,vi);
              elemat(vi*3    , ui*3 + 1) += fac_two_visc_alphaM_tauMp*funct_(ui)*viscs2_(0,1,vi);
              elemat(vi*3 + 1, ui*3    ) += fac_two_visc_alphaM_tauMp*funct_(ui)*viscs2_(0,1,vi);
              elemat(vi*3 + 1, ui*3 + 1) += fac_two_visc_alphaM_tauMp*funct_(ui)*viscs2_(1,1,vi);


              /* viscous stabilisation --- convection */

              /*  factor: +(-)2*nu*alphaF*gamma*dt*tauMp

                       /                                  \
                      |  / n+af       \                    |
                      | | u    o nabla | Dacc, div eps (v) |
                      |  \            /                    |
                       \                                  /

              */
              elemat(vi*3    , ui*3    ) += fac_two_visc_afgdt_tauMp*conv_c_af_(ui)*viscs2_(0, 0, vi) ;
              elemat(vi*3    , ui*3 + 1) += fac_two_visc_afgdt_tauMp*conv_c_af_(ui)*viscs2_(0, 1, vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_two_visc_afgdt_tauMp*conv_c_af_(ui)*viscs2_(0, 1, vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_two_visc_afgdt_tauMp*conv_c_af_(ui)*viscs2_(1, 1, vi) ;

              /* viscous stabilisation --- diffusion  */

              /* factor: -(+)4*nu*nu*alphaF*gamma*dt*tauMp

                    /                                   \
                   |               /    \                |
                   |  nabla o eps | Dacc | , div eps (v) |
                   |               \    /                |
                    \                                   /
              */

              elemat(vi*3    , ui*3    ) -= fac_four_visceff_visc_afgdt_tauMp*
                                            (viscs2_(0, 0, ui)*viscs2_(0, 0, vi)
                                             +
                                             viscs2_(0, 1, ui)*viscs2_(0, 1, vi)) ;
              elemat(vi*3    , ui*3 + 1) -= fac_four_visceff_visc_afgdt_tauMp*
                                            (viscs2_(0, 0, vi)*viscs2_(0, 1, ui)
                                             +
                                             viscs2_(0, 1, vi)*viscs2_(1, 1, ui)) ;
              elemat(vi*3 + 1, ui*3    ) -= fac_four_visceff_visc_afgdt_tauMp*
                                            (viscs2_(0, 0, ui)*viscs2_(0, 1, vi)
                                             +
                                             viscs2_(0, 1, ui)*viscs2_(1, 1, vi)) ;
              elemat(vi*3 + 1, ui*3 + 1) -= fac_four_visceff_visc_afgdt_tauMp*
                                            (viscs2_(0, 1, ui)*viscs2_(0, 1, vi)
                                             +
                                             viscs2_(1, 1, ui)*viscs2_(1, 1, vi)) ;


              /* viscous stabilisation --- pressure   */

              /* factor: +(-)tauMp*2*nu

                    /                        \
                   |                          |
                   |  nabla Dp , div eps (v)  |
                   |                          |
                    \                        /
              */
              elemat(vi*3    , ui*3 + 2) += fac_two_visc_tauMp*
                                            (derxy_(0,ui)*viscs2_(0,0,vi)
                                             +
                                             derxy_(1,ui)*viscs2_(0,1,vi)) ;
              elemat(vi*3 + 1, ui*3 + 2) += fac_two_visc_tauMp*
                                            (derxy_(0,ui)*viscs2_(0,1,vi)
                                             +
                                             derxy_(1,ui)*viscs2_(1,1,vi)) ;

            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)
          if (ele->is_ale_)
          {
            for (int ui=0; ui<iel_; ++ui)
            {
              for (int vi=0; vi<iel_; ++vi)
              {

                /*  reduced convection through grid motion


                       /                                   \
                      |  /          \                       |
                  -/+ | | u  o nabla | Du ,  nabla o eps (v)|
                      |  \ G        /                       |
                       \                                   /

                */

                elemat(vi*3    , ui*3    ) += fac_two_visc_afgdt_tauMp*conv_g_af_(ui)*viscs2_(0, 0, vi) ;
                elemat(vi*3    , ui*3 + 1) += fac_two_visc_afgdt_tauMp*conv_g_af_(ui)*viscs2_(0, 1, vi) ;
                elemat(vi*3 + 1, ui*3    ) += fac_two_visc_afgdt_tauMp*conv_g_af_(ui)*viscs2_(0, 1, vi) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_two_visc_afgdt_tauMp*conv_g_af_(ui)*viscs2_(1, 1, vi) ;
              }//vi
            }// ui
          } // end if is_ale

          if (newton)
          {
            for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
            {
              for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
              {
                /* viscous stabilisation --- convection */

                /*  factor: +(-)2*nu*alphaF*gamma*dt*tauMp

                     /                                     \
                    |   /            \   n+af               |
                    |  | Dacc o nabla | u     , div eps (v) |
                    |   \            /                      |
                     \                                     /


                */

                elemat(vi*3    , ui*3    ) += fac_two_visc_afgdt_tauMp*
                                              (viscs2_(0, 0, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               viscs2_(0, 1, vi)*conv_r_af_(1, 0, ui)) ;
                elemat(vi*3    , ui*3 + 1) += fac_two_visc_afgdt_tauMp*
                                              (viscs2_(0, 0, vi)*conv_r_af_(0, 1, ui)
                                               +
                                               viscs2_(0, 1, vi)*conv_r_af_(1, 1, ui)) ;
                elemat(vi*3 + 1, ui*3     )+= fac_two_visc_afgdt_tauMp*
                                              (viscs2_(0, 1, vi)*conv_r_af_(0, 0, ui)
                                               +
                                               viscs2_(1, 1, vi)*conv_r_af_(1, 0, ui)) ;
                elemat(vi*3 + 1, ui*3 + 1) += fac_two_visc_afgdt_tauMp*
                                              (viscs2_(0, 1, vi)*conv_r_af_(0, 1, ui)
                                               +
                                               viscs2_(1, 1, vi)*conv_r_af_(1, 1, ui)) ;
              } // end loop rows (test functions for matrix)
            } // end loop columns (solution for matrix, test function for vector)
          }
        } // endif (a)gls

        if(cstab == Fluid2::continuity_stab_yes)
        {

          const double fac_gamma_dt_tauC = fac*gamma*dt*tauC;

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {
              /*  factor: +gamma*dt*tauC

                    /                          \
                   |                            |
                   | nabla o Dacc  , nabla o v  |
                   |                            |
                    \                          /
              */

              elemat(vi*3    , ui*3    ) += fac_gamma_dt_tauC*derxy_(0,ui)*derxy_(0,vi) ;
              elemat(vi*3    , ui*3 + 1) += fac_gamma_dt_tauC*derxy_(1,ui)*derxy_(0,vi) ;
              elemat(vi*3 + 1, ui*3    ) += fac_gamma_dt_tauC*derxy_(0,ui)*derxy_(1,vi) ;
              elemat(vi*3 + 1, ui*3 + 1) += fac_gamma_dt_tauC*derxy_(1,ui)*derxy_(1,vi) ;
            } // end loop rows vi (test functions for matrix)
          } // end loop columns ui (solution for matrix, test function for vector)
        } // end cstab

        if(cross == Fluid2::cross_stress_stab)
        {

          //---------------------------------------------------------------
          //
          //                     STABILISATION PART
          //       RESIDUAL BASED VMM STABILISATION --- CROSS STRESS
          //
          //---------------------------------------------------------------

          for (int ui=0; ui<iel_; ++ui) // loop columns (solution for matrix, test function for vector)
          {
            for (int vi=0; vi<iel_; ++vi)  // loop rows (test functions for matrix)
            {

              /*  factor:

               -alphaF*gamma*dt*tauM

                          /                          \
                         |  /            \            |
                         | | resM o nabla | Dacc , v  |
                         |  \            /            |
                          \                          /
              */
              elemat(vi*3    , ui*3    ) -= fac*afgdt*tauM*conv_resM_(ui)*funct_(vi) ;
              elemat(vi*3 + 1, ui*3 + 1) -= fac*afgdt*tauM*conv_resM_(ui)*funct_(vi) ;
            } // end loop rows (test functions for matrix)
          } // end loop columns (solution for matrix, test function for vector)

        } // end cross
      } // end if compute_elemat

      //---------------------------------------------------------------
      //---------------------------------------------------------------
      //
      //                       RIGHT HAND SIDE
      //
      //---------------------------------------------------------------
      //---------------------------------------------------------------


      {

        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /* inertia terms */

          /*  factor: +1

               /             \
              |     n+am      |
              |  acc     , v  |
              |               |
               \             /
          */

          elevec[ui*3    ] -= fac*funct_(ui)*accintam_(0) ;
          elevec[ui*3 + 1] -= fac*funct_(ui)*accintam_(1) ;

          /* convection */

          /*  factor: +1

               /                             \
              |  / n+af       \    n+af       |
              | | u    o nabla |  u      , v  |
              |  \            /               |
               \                             /
          */

          elevec[ui*3    ] -= fac*(velintaf_(0)*conv_r_af_(0,0,ui)
                                   +
                                   velintaf_(1)*conv_r_af_(0,1,ui)) ;
          elevec[ui*3 + 1] -= fac*(velintaf_(0)*conv_r_af_(1,0,ui)
                                   +
                                   velintaf_(1)*conv_r_af_(1,1,ui)) ;

          /* pressure */

          /*  factor: -1

               /                  \
              |   n+1              |
              |  p    , nabla o v  |
              |                    |
               \                  /
          */

          elevec[ui*3    ] += fac*prenp_*derxy_(0,ui) ;
          elevec[ui*3 + 1] += fac*prenp_*derxy_(1,ui) ;

          /* viscous term */

          /*  factor: +2*nu

               /                            \
              |       / n+af \         / \   |
              |  eps | u      | , eps | v |  |
              |       \      /         \ /   |
               \                            /
          */

          elevec[ui*3    ] -= visceff*fac*
                              (derxy_(0,ui)*vderxyaf_(0,0)*2.0
                               +
                               derxy_(1,ui)*vderxyaf_(0,1)
                               +
                               derxy_(1,ui)*vderxyaf_(1,0)) ;
          elevec[ui*3 + 1] -= visceff*fac*
                              (derxy_(0,ui)*vderxyaf_(0,1)
                               +
                               derxy_(0,ui)*vderxyaf_(1,0)
                               +
                               derxy_(1,ui)*vderxyaf_(1,1)*2.0) ;

          /* body force (dead load...) */

          /*  factor: -1

               /           \
              |   n+af      |
              |  f     , v  |
              |             |
               \           /
          */

          elevec[ui*3    ] += fac*funct_(ui)*edeadaf_(0);
          elevec[ui*3 + 1] += fac*funct_(ui)*edeadaf_(1);

          /* continuity equation */

        /*  factor: +1

               /                \
              |          n+1     |
              | nabla o u   , q  |
              |                  |
               \                /
        */

          elevec[ui*3 + 2] -= fac*funct_(ui)*divunp;

        } // end loop rows


        if (ele->is_ale_)
        {
          for (int ui=0; ui<iel_; ++ui)
          {
            elevec(ui*3)     += fac*(ugrid_af_(0)*conv_r_af_(0, 0, ui)
                                     +
                                     ugrid_af_(1)*conv_r_af_(0, 1, ui)) ;
            elevec(ui*3 + 1) += fac*(ugrid_af_(0)*conv_r_af_(1, 0, ui)
                                     +
                                     ugrid_af_(1)*conv_r_af_(1, 1, ui)) ;
          } // ui
        }

      }

      if(pspg == Fluid2::pstab_use_pspg)
      {

        const double fac_tauMp = fac*tauMp;

        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /*
            factor: +tauMp

            pressure stabilisation --- inertia


                  /                  \
                 |     n+am           |
                 |  acc    , nabla q  |
                 |                    |
                  \                  /

            pressure stabilisation --- convection


                  /                                   \
                 |  / n+af       \    n+af             |
                 | | u    o nabla |  u      , nabla q  |
                 |  \            /                     |
                  \                                   /


            pressure stabilisation --- diffusion

                  /                                  \
                 |               / n+af \             |
                 |  nabla o eps | u      | , nabla q  |
                 |               \      /             |
                  \                                  /

            pressure stabilisation --- pressure

                  /                      \
                 |         n+1            |
                 |  nabla p    , nabla q  |
                 |                        |
                  \                      /


            pressure stabilisation --- bodyforce
                  /                 \
                 |    n+af           |
                 |  f     , nabla q  |
                 |                   |
                  \                 /
          */
          elevec[ui*3 + 2] -= fac_tauMp*
                              (derxy_(0,ui)*resM_(0)
                               +
                               derxy_(1,ui)*resM_(1));
        } // end loop rows
      }

      if(supg == Fluid2::convective_stab_supg)
      {

        const double fac_tauM = fac*tauM;

        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /*
            factor: +tauM

            SUPG stabilisation --- inertia

                  /                              \
                 |     n+am   / n+af        \     |
                 |  acc    , | u     o nabla | v  |
                 |            \             /     |
                  \                              /

           SUPG stabilisation --- convection


                  /                                                \
                 |    / n+af        \   n+af    / n+af        \     |
                 |   | u     o nabla | u     , | u     o nabla | v  |
                 |    \             /           \             /     |
                  \                                                /


           SUPG stabilisation --- diffusion

                  /                                               \
                 |               / n+af \      / n+af        \     |
                 |  nabla o eps | u      |  , | u     o nabla | v  |
                 |               \      /      \             /     |
                  \                                               /

           SUPG stabilisation --- pressure

                  /                                  \
                 |         n+1    / n+af        \     |
                 |  nabla p    , | u     o nabla | v  |
                 |                \             /     |
                  \                                  /

           SUPG stabilisation --- bodyforce


                  /                             \
                 |   n+af    / n+af        \     |
                 |  f     , | u     o nabla | v  |
                 |           \             /     |
                  \                             /
          */

          elevec[ui*3    ] -= fac_tauM*conv_c_af_(ui)*resM_(0) ;
          elevec[ui*3 + 1] -= fac_tauM*conv_c_af_(ui)*resM_(1) ;

        } // end loop rows
      }

      if(vstab != Fluid2::viscous_stab_none)
      {

        const double fac_two_visc_tauMp = vstabfac*fac*2.0*visc*tauMp;

        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /*
            factor: -(+)tauMp*2*nu


            viscous stabilisation --- inertia


                 /                         \
                |      n+am                 |
                |  Dacc      , div eps (v)  |
                |                           |
                 \                         /

            viscous stabilisation --- convection

            /                                     \
           |  / n+af       \    n+af               |
           | | u    o nabla |  u     , div eps (v) |
           |  \            /                       |
            \                                     /

            viscous stabilisation --- diffusion

               /                                      \
              |               /  n+af \                |
              |  nabla o eps |  u      | , div eps (v) |
              |               \       /                |
               \                                      /

            viscous stabilisation --- pressure

                 /                           \
                |                             |
                |  nabla p , nabla o eps (v)  |
                |                             |
                 \                           /

           viscous stabilisation --- bodyforce

                  /                         \
                 |    n+af                   |
                 |  f     ,  nabla o eps (v) |
                 |                           |
                  \                         /
          */
          elevec[ui*3    ] -= fac_two_visc_tauMp*
                              (resM_(0)*viscs2_(0, 0, ui)
                               +
                               resM_(1)*viscs2_(0, 1, ui)) ;
          elevec[ui*3 + 1] -= fac_two_visc_tauMp*
                              (resM_(0)*viscs2_(0, 1, ui)
                               +
                               resM_(1)*viscs2_(1, 1, ui)) ;

        } // end loop rows ui
      } // endif (a)gls

      if(cstab == Fluid2::continuity_stab_yes)
      {

        const double fac_tauC = fac*tauC;
        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /* factor: +tauC

                  /                          \
                 |           n+1              |
                 |  nabla o u    , nabla o v  |
                 |                            |
                  \                          /
          */

          elevec[ui*3    ] -= fac_tauC*divunp*derxy_(0,ui) ;
          elevec[ui*3 + 1] -= fac_tauC*divunp*derxy_(1,ui) ;
        } // end loop rows

      } // end cstab

      if(cross == Fluid2::cross_stress_stab_only_rhs || cross == Fluid2::cross_stress_stab)
      {
        const double fac_tauM = fac*tauM;
        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /* factor: +tauM

                  /                            \
                 |                    n+af      |
                 |  ( resM o nabla ) u    ,  v  |
                 |                    (i)       |
                  \                            /
          */

          elevec[ui*3    ] += fac_tauM*(resM_(0)*vderxyaf_(0,0)
                                        +
                                        resM_(1)*vderxyaf_(0,1))*funct_(ui);
          elevec[ui*3 + 1] += fac_tauM*(resM_(0)*vderxyaf_(1,0)
                                        +
                                        resM_(1)*vderxyaf_(1,1))*funct_(ui);

        } // end loop rows

      }

      if(reynolds == Fluid2::reynolds_stress_stab_only_rhs)
      {

        const double fac_tauM_tauM = fac*tauM*tauM;
        for (int ui=0; ui<iel_; ++ui) // loop rows  (test functions)
        {
          /* factor: -tauM*tauM

                  /                             \
                 |                               |
                 |  resM   , ( resM o nabla ) v  |
                 |                               |
                  \                             /
          */
          elevec[ui*3    ] += fac_tauM_tauM*conv_resM_(ui)*resM_(0);
          elevec[ui*3 + 1] += fac_tauM_tauM*conv_resM_(ui)*resM_(1);
        } // end loop rows

      }
    }
  } // end loop iquad
  return;
}

// this is just for comparison of dynamic/quasistatic subscales --- NOT for
// the comparison with physical turbulence models (Smagorinsky etc.)

void DRT::ELEMENTS::Fluid2GenalphaResVMM::CalcRes(
  Fluid2*                                               ele,
  const blitz::Array<double,2>&                         evelnp,
  const blitz::Array<double,1>&                         eprenp,
  const blitz::Array<double,2>&                         eaccam,
  const blitz::Array<double,2>&                         evelaf,
  const struct _MATERIAL*                               material,
  const double                                          alphaM,
  const double                                          alphaF,
  const double                                          gamma,
  const double                                          dt,
  const double                                          time,
  const enum Fluid2::StabilisationAction                tds,
  blitz::Array<double,1>&                               mean_res,
  blitz::Array<double,1>&                               mean_sacc,
  blitz::Array<double,1>&                               mean_res_sq,
  blitz::Array<double,1>&                               mean_sacc_sq
  )
{
  cout << "Fluid2GenalphaResVMM:CalcRes is empty right now\n";

  return;
}



void DRT::ELEMENTS::Fluid2GenalphaResVMM::CalVisc(
  const struct _MATERIAL*                 material,
  double&                                 visc)
{

  blitz::firstIndex i;    // Placeholder for the first index
  blitz::secondIndex j;   // Placeholder for the second index

  // compute shear rate
  double rateofshear = 0.0;
  blitz::Array<double,2> epsilon(2,2,blitz::ColumnMajorArray<2>());   // strain rate tensor
  epsilon = 0.5 * ( vderxyaf_(i,j) + vderxyaf_(j,i) );

  for(int rr=0;rr<2;rr++)
    for(int mm=0;mm<2;mm++)
      rateofshear += epsilon(rr,mm)*epsilon(rr,mm);

  rateofshear = sqrt(2.0*rateofshear);

  if(material->mattyp == m_carreauyasuda)
  {
    double nu_0     = material->m.carreauyasuda->nu_0;      // parameter for zero-shear viscosity
    double nu_inf   = material->m.carreauyasuda->nu_inf;    // parameter for infinite-shear viscosity
    double lambda   = material->m.carreauyasuda->lambda;    // parameter for characteristic time
    double a        = material->m.carreauyasuda->a_param;   // constant parameter
    double b        = material->m.carreauyasuda->b_param;   // constant parameter

    // compute viscosity according to the Carreau-Yasuda model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    const double tmp = pow(lambda*rateofshear,b);
    visc = nu_inf + ((nu_0 - nu_inf)/pow((1 + tmp),a));
  }
  else if(material->mattyp == m_modpowerlaw)
  {
    // get material parameters
    double m      = material->m.modpowerlaw->m_cons;    // consistency constant
    double delta  = material->m.modpowerlaw->delta;     // safety factor
    double a      = material->m.modpowerlaw->a_exp;     // exponent

    // compute viscosity according to a modified power law model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    visc = m * pow((delta + rateofshear), (-1)*a);
  }
  else
    dserror("material type not yet implemented");
}




/*----------------------------------------------------------------------*
 |  get the body force in the nodes of the element (private) gammi 02/08|
 |  the Neumann condition associated with the nodes is stored in the    |
 |  array edeadng only if all nodes have a VolumeNeumann condition      |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Fluid2GenalphaResVMM::GetNodalBodyForce(Fluid2* ele, const double time)
{
  vector<DRT::Condition*> myneumcond;
  DRT::Node** nodes = ele->Nodes();

  // check whether all nodes have a unique VolumeNeumann condition
  int nodecount = 0;
  for (int inode=0;inode<iel_;inode++)
  {
    nodes[inode]->GetCondition("VolumeNeumann",myneumcond);

    if (myneumcond.size()>1)
    {
      dserror("more than one VolumeNeumann cond on one node");
    }
    if (myneumcond.size()==1)
    {
      nodecount++;
    }
  }

  if (nodecount == iel_)
  {
    // find out whether we will use a time curve
    const vector<int>* curve  = myneumcond[0]->Get<vector<int> >("curve");
    int curvenum = -1;

    if (curve) curvenum = (*curve)[0];

    // initialisation
    double curvefac    = 0.0;

    if (curvenum >= 0) // yes, we have a timecurve
    {
      // time factor for the intermediate step
      if(time >= 0.0)
      {
        curvefac = DRT::UTILS::TimeCurveManager::Instance().Curve(curvenum).f(time);
      }
      else
      {
	// do not compute an "alternative" curvefac here since a negative time value
	// indicates an error.
        dserror("Negative time value in body force calculation: time = %f",time);
        //curvefac = DRT::UTILS::TimeCurveManager::Instance().Curve(curvenum).f(0.0);
      }
    }
    else // we do not have a timecurve --- timefactors are constant equal 1
    {
      curvefac = 1.0;
    }

    // set this condition to the edeadng array
    for (int jnode=0; jnode<iel_; jnode++)
    {
      nodes[jnode]->GetCondition("VolumeNeumann",myneumcond);

      // get values and switches from the condition
      const vector<int>*    onoff = myneumcond[0]->Get<vector<int> >   ("onoff");
      const vector<double>* val   = myneumcond[0]->Get<vector<double> >("val"  );

      for(int isd=0;isd<2;isd++)
      {
        edeadaf_(isd,jnode) = (*onoff)[isd]*(*val)[isd]*curvefac;
      }
    }
  }
  else
  {
    // we have no dead load
    edeadaf_ = 0.;
  }
  return;
}

#endif
#endif
