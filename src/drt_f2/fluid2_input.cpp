/*!----------------------------------------------------------------------
\file fluid2_input.cpp
\brief

<pre>
Maintainer: Peter Gamnitzer
            gamnitzer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15235
</pre>

*----------------------------------------------------------------------*/
#ifdef D_FLUID2
#ifdef CCADISCRET

// This is just here to get the c++ mpi header, otherwise it would
// use the c version included inside standardtypes.h
#ifdef PARALLEL
#include "mpi.h"
#endif

extern "C"
{
#include "../headers/standardtypes.h"
}
#include "fluid2.H"
#include "../drt_lib/drt_utils.H"

using namespace DRT::UTILS;

/*----------------------------------------------------------------------*
 |  read element input (public)                              gammi 04/07|
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Fluid2::ReadElement()
{
  // what kind of element is this
  DiscretizationType distype = dis_none;

  // read element's nodes
  int   ierr = 0;
  int   nnode = 0;
  int   nodes[27];
  char  buffer[50];

  frchk("QUAD4",&ierr);
  if (ierr==1)
  {
    distype = quad4;
    nnode=4;
    frint_n("QUAD4",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of ELEMENT Topology failed\n");
  }

  frchk("QUAD8",&ierr);
  if (ierr==1)
  {
    distype = quad8;
    nnode=8;
    frint_n("QUAD8",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of ELEMENT Topology failed\n");
  }

  frchk("QUAD9",&ierr);
  if (ierr==1)
  {
    distype = quad9;
    nnode=9;
    frint_n("QUAD9",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of ELEMENT Topology failed\n");
  }

  frchk("TRI3",&ierr);
  if (ierr==1)
  {
    distype = tri3;
    nnode=3;
    frint_n("TRI3",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of ELEMENT Topology failed\n");
  }

  frchk("TRI6",&ierr); /* rearrangement??????? */
  if (ierr==1)
  {
    distype = tri6;
    nnode=6;
    frint_n("TRI6",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of ELEMENT Topology failed\n");
  }

  frchk("NURBS4",&ierr);
  if (ierr==1)
  {
    distype = nurbs4;
    nnode=4;
    frint_n("NURBS4",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of nurbs ELEMENT Topology failed\n");
  }

  frchk("NURBS9",&ierr);
  if (ierr==1)
  {
    distype = nurbs9;
    nnode=9;
    frint_n("NURBS9",nodes,nnode,&ierr);
    if (ierr!=1) dserror("Reading of nurbs ELEMENT Topology failed\n");
  }
  
  // reduce node numbers by one
  for (int i=0; i<nnode; ++i) nodes[i]--;

  SetNodeIds(nnode,nodes);

  // read number of material model
  int material = 0;
  frint("MAT",&material,&ierr);
  if (ierr!=1) dserror("Reading Material for FLUID2 element failed");
  if (material==0) dserror("No material defined for FLUID2 element");
  SetMaterial(material);

  // read gaussian points

//  if (nnode==4 || nnode==8 || nnode==9)
//  {
//    frint_n("GP",ngp_,2,&ierr);
//    if (ierr!=1) dserror("Reading of FLUID2 element failed: GP\n");
//  }
//
//  // read number of gaussian points for triangle elements */
//  if (nnode==3 || nnode==6)
//  {
//    frint("GP_TRI",&ngp_[0],&ierr);
//    if (ierr!=1) dserror("Reading of FLUID2 element failed: GP_TRI\n");
//
//    frchar("GP_ALT",buffer,&ierr);
//    if (ierr!=1) dserror("Reading of FLUID2 element failed: GP_ALT\n");
//    /*
//     * integration for TRI-elements is distinguished into different cases.
//     * This is necessary to get the right integration parameters from
//     * FLUID_DATA.
//     * The flag for the integration case is saved in nGP[1].
//     * For detailed informations see /fluid3/f3_intg.c.
//     */
//
//    switch(ngp_[0])
//    {
//      case 1:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=0;
//        else
//          dserror("Reading of FLUID2 element failed: GP_ALT: gauss-radau not possible!\n");
//        break;
//      case 3:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=1;
//        else if (strncmp(buffer,"gaussrad",8)==0)
//          ngp_[1]=2;
//        else
//          dserror("Reading of FLUID2 element failed: GP_ALT\n");
//        break;
//      case 4:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=3;
//        else
//          dserror("Reading of FLUID2 element failed: gauss-radau not possible!\n");
//        break;
//      case 6:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=4;
//        else if (strncmp(buffer,"gaussrad",8)==0)
//          ngp_[1]=5;
//        else
//          dserror("Reading of FLUID2 element failed: GP_ALT\n");
//        break;
//      case 7:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=6;
//        else if (strncmp(buffer,"gaussrad",8)==0)
//          ngp_[1]=7;
//        else
//        dserror("Reading of FLUID2 element failed: GP_ALT\n");
//      case 9:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=8;
//        else
//          dserror("Reading of FLUID2 element failed: gauss-radau not possible!\n");
//        break;
//      case 12:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=9;
//        else
//          dserror("Reading of FLUID2 element failed: gauss-radau not possible!\n");
//        break;
//      case 13:
//        if (strncmp(buffer,"standard",8)==0)
//          ngp_[1]=10;
//        else
//          dserror("Reading of FLUID2 element failed: gauss-radau not possible!\n");
//        break;
//
//      default:
//        dserror("Reading of FLUID2 element failed: integration points\n");
//    }
//  } // end reading gaussian points for tetrahedral elements

  // read gaussian points and set gaussrule
  int myngp[2];
  switch (distype)
  {
  case quad4: case quad8: case quad9: case nurbs4: case nurbs9:
  {
    frint_n("GP",myngp,2,&ierr);
    dsassert(ierr==1, "Reading of FLUID2 element failed: GP");
    switch (myngp[0])
    {
    case 1:
      gaussrule_ = intrule_quad_1point;
      break;
    case 2:
      gaussrule_ = intrule_quad_4point;
      break;
    case 3:
      gaussrule_ = intrule_quad_9point;
      break;
    default:
      dserror("Reading of FLUID2 element failed: Gaussrule for quad not supported!");
    }
    break;
  }
  case tri3: case tri6: 
    gaussrule_ = intrule2D_undefined; // no input of Gauss rule for triangles
    break;
  default:
    dserror("Reading of FLUID2 element failed: GP and set gaussrule");
  } // end switch distype



  // read net algo
  frchar("NA",buffer,&ierr);
  if (ierr==1)
  {
    if (strncmp(buffer,"ale",3)==0 ||
        strncmp(buffer,"ALE",3)==0 ||
        strncmp(buffer,"Ale",3)==0 )
    {
      is_ale_=true;
    }

    else if (strncmp(buffer,"euler",5)==0 ||
        strncmp(buffer,"EULER",5)==0 ||
        strncmp(buffer,"Euler",5)==0 )
        is_ale_=false;
    else
      dserror("Reading of FLUID2 element failed: Euler/Ale\n");
  }
   else
    dserror("Reading of FLUID2 element net algorithm failed: NA\n");



  // input of ale and free surface related stuff is not supported
  // at the moment. TO DO!


  return true;

} // Fluid2::ReadElement()


#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_FLUID2
