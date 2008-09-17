/*!----------------------------------------------------------------------
\file so_hex8_service.cpp
\brief

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET
#ifdef D_SOLID3
#include "so_hex8.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
using namespace std; // cout etc.


/*----------------------------------------------------------------------*
 |  return Center Coords in Reference System                   maf 11/07|
 *----------------------------------------------------------------------*/
const vector<double> DRT::ELEMENTS::So_hex8::soh8_ElementCenterRefeCoords()
{
  // update element geometry
  DRT::Node** nodes = Nodes();
  LINALG::FixedSizeSerialDenseMatrix<NUMNOD_SOH8,NUMDIM_SOH8> xrefe;  // material coord. of element
  for (int i=0; i<NUMNOD_SOH8; ++i){
    const double* x = nodes[i]->X();
    xrefe(i,0) = x[0];
    xrefe(i,1) = x[1];
    xrefe(i,2) = x[2];
  }
  const DRT::Element::DiscretizationType distype = Shape();
  LINALG::FixedSizeSerialDenseMatrix<NUMNOD_SOH8,1> funct;
  // Element midpoint at r=s=t=0.0
  DRT::UTILS::shape_function_3D(funct, 0.0, 0.0, 0.0, distype);
  LINALG::FixedSizeSerialDenseMatrix<1,NUMDIM_SOH8> midpoint;
  //midpoint.Multiply('T','N',1.0,funct,xrefe,0.0);
  midpoint.MultiplyTN(funct, xrefe);
  vector<double> centercoords(3);
  centercoords[0] = midpoint(0,0);
  centercoords[1] = midpoint(0,1);
  centercoords[2] = midpoint(0,2);
  return centercoords;
}

#endif
#endif
