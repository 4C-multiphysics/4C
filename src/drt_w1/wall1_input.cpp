/*!----------------------------------------------------------------------
\file wall1_input.cpp
\brief

<pre>
Maintainer: Markus Gitterle
            gitterle@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15251
</pre>

*----------------------------------------------------------------------*/
#ifdef D_WALL1
#ifdef CCADISCRET

#include "wall1.H"
#include "../drt_lib/drt_linedefinition.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Wall1::ReadElement(const std::string& eletype,
                                       const std::string& distype,
                                       DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  linedef->ExtractDouble("THICK",thickness_);

  std::vector<int> ngp;
  linedef->ExtractIntVector("GP",ngp);

  if ((NumNode()==4) and ((ngp[0]<2) or (ngp[1]<2)))
  {
    dserror("Insufficient number of Gauss points");
  }
  else if ((NumNode()==8) and ((ngp[0]<3) or (ngp[1]<3)))
  {
    dserror("Insufficient number of Gauss points");
  }
  else if ((NumNode()==9) and ((ngp[0]<3) or (ngp[1]<3)))
  {
    dserror("Insufficient number of Gauss points");
  }
  else if ((NumNode()==6) and (ngp[0]<3))
  {
    dserror("Insufficient number of Gauss points");
  }

  gaussrule_ = getGaussrule(&ngp[0]);

  std::string buffer;
  linedef->ExtractString("STRESS_STRAIN",buffer);
  if      (buffer=="PLANE_STRESS") wtype_ = plane_stress;
  else if (buffer=="PLANE_STRAIN") wtype_ = plane_strain;
  else dserror("Illegal strain/stress type '%s'",buffer.c_str());

  // kinematics type
  linedef->ExtractString("KINEM",buffer);

  // geometrically linear
  if      (buffer=="W_GeoLin")    kintype_ = w1_geolin;
  // geometrically non-linear with Total Lagrangean approach
  else if (buffer=="W_TotalLagr")    kintype_ = w1_totlag;
  // geometrically non-linear with Updated Lagrangean approach
  else if (buffer=="W_UpdLagr")
  {
    kintype_ = w1_updlag;
    dserror("Updated Lagrange for wall element NOT implemented!");
  }
  else dserror("Reading of WALL element failed");

  // EAS type
  linedef->ExtractString("EAS",buffer);
  if (buffer=="Displ_Model")
  {
    iseas_ = false;
  }
  else if (buffer=="EAS_Model")
  {
    iseas_ = true;

    if (NumNode()==9)
    {
      dserror("eas-technology not necessary with 9 nodes");
    }
    else if (NumNode()==8)
    {
      dserror("eas-technology not necessary with 8 nodes");
    }
    else if (NumNode()==3)
    {
      dserror("eas-technology not implemented for tri3 elements");
    }
    else if (NumNode()==6)
    {
      dserror("eas-technology not implemented for tri6 elements");
    }
    else
    {
      // EAS enhanced deformation gradient parameters
      Epetra_SerialDenseMatrix alpha(Wall1::neas_,1);  // if you change '4' here, then do it for alphao as well
      Epetra_SerialDenseMatrix alphao(Wall1::neas_,1);

      // EAS portion of internal forces, also called enhacement vector s or Rtilde
      Epetra_SerialDenseVector feas(Wall1::neas_);
      // EAS matrix K_{alpha alpha}, also called Dtilde
      Epetra_SerialDenseMatrix invKaa(Wall1::neas_,Wall1::neas_);
      // EAS matrix K_{d alpha}
      Epetra_SerialDenseMatrix Kda(2*NumNode(),Wall1::neas_);
      // EAS matrix K_{alpha d} // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
      Epetra_SerialDenseMatrix Kad(Wall1::neas_,2*NumNode());

      // save EAS data into element container easdata_
      data_.Add("alpha",alpha);
      data_.Add("alphao",alphao);
      data_.Add("feas",feas);
      data_.Add("invKaa",invKaa);
      data_.Add("Kda",Kda);
      data_.Add("Kad",Kad); // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
    }
  }
  else
  {
    dserror("Illegal EAS model");
  }

  // EAS type
  if (iseas_)
  {
    eastype_ = eas_q1e4;
  }
  else
  {
    eastype_ = eas_vague;
  }

  stresstype_ = w1_xy;
//   linedef->ExtractString("STRESSES",buffer);
//   if      (buffer=="XY")       stresstype_ = w1_xy;
//   else if (buffer=="RS")       stresstype_ = w1_rs;
//   else dserror("Reading of WALL1 element failed");

  // check for invalid combinations
  if (kintype_==w1_geolin && iseas_==true)
    dserror("ERROR: No EAS for geometrically linear WALL element");

  return true;
}


#if 0
/*----------------------------------------------------------------------*
 |  read element input (public)                              mgit 03/07|
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Wall1::ReadElement()
{
  // read element's nodes
  int ierr=0;
  int nnode=0;
  int nodes[9];
  frchk("QUAD4",&ierr);
  if (ierr==1)
  {
    nnode = 4;
    frint_n("QUAD4",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("QUAD8",&ierr);
  if (ierr==1)
  {
    nnode = 8;
    frint_n("QUAD8",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("QUAD9",&ierr);
  if (ierr==1)
  {
    nnode = 9;
    frint_n("QUAD9",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("TRI3",&ierr);
  if (ierr==1)
  {
    nnode = 3;
    frint_n("TRI3",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("TRI6",&ierr);
  if (ierr==1)
  {
    nnode = 6;
    frint_n("TRI6",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("NURBS4",&ierr);
  if (ierr==1)
  {
    nnode = 4;
    frint_n("NURBS4",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology of NURBS4 failed");
  }
  frchk("NURBS9",&ierr);
  if (ierr==1)
  {
    nnode = 9;
    frint_n("NURBS9",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology of NURBS9 failed");
  }

  // reduce node numbers by one
  for (int i=0; i<nnode; ++i) nodes[i]--;

  SetNodeIds(nnode,nodes);

  // read number of material model
  material_ = 0;
  frint("MAT",&material_,&ierr);
  if (ierr!=1) dserror("Reading of WALL1 element failed");
  SetMaterial(material_);

  // read wall thickness
  thickness_ = 1.0;
  frdouble("THICK",&thickness_,&ierr);
  if (ierr!=1) dserror("Reading of WALL1 element failed");

  // read gaussian points
  int ngp[2];
  frint_n("GP",ngp,2,&ierr);
  if (ierr!=1) dserror("Reading of WALL1 element failed");

  if ((nnode==4) and ((ngp[0]<2) or (ngp[1]<2)))
  {
      dserror("Insufficient number of Gauss points");
  }
  else if ((nnode==8) and ((ngp[0]<3) or (ngp[1]<3)))
  {
  	dserror("Insufficient number of Gauss points");
  }
  else if ((nnode==9) and ((ngp[0]<3) or (ngp[1]<3)))
  {
     dserror("Insufficient number of Gauss points");
  }

    gaussrule_ = getGaussrule(ngp);

  //read 2D problem type
  frchk("PLANE_STRESS",&ierr);
  if (ierr==1) wtype_ = plane_stress;
  frchk("PLANE_STRAIN",&ierr);
  if (ierr==1) wtype_ = plane_strain;

  //read model (EAS or not)
  frchk("EAS_Model",&ierr);
  if (ierr==1)
  {
    iseas_ = true;

    if (nnode==9)
    {
      dserror("eas-technology not necessary with 9 nodes");
    }
    else if (nnode==8)
    {
      dserror("eas-technology not necessary with 8 nodes");
    }
    else if (nnode==3)
    {
      dserror("eas-technology not implemented for tri3 elements");
    }
    else if (nnode==6)
    {
      dserror("eas-technology not implemented for tri6 elements");
    }
    else
    {
      // EAS enhanced deformation gradient parameters
      Epetra_SerialDenseMatrix alpha(Wall1::neas_,1);  // if you change '4' here, then do it for alphao as well
      Epetra_SerialDenseMatrix alphao(Wall1::neas_,1);

      // EAS portion of internal forces, also called enhacement vector s or Rtilde
      Epetra_SerialDenseVector feas(Wall1::neas_);
      // EAS matrix K_{alpha alpha}, also called Dtilde
      Epetra_SerialDenseMatrix invKaa(Wall1::neas_,Wall1::neas_);
      // EAS matrix K_{d alpha}
      Epetra_SerialDenseMatrix Kda(2*NumNode(),Wall1::neas_);
      // EAS matrix K_{alpha d} // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
      Epetra_SerialDenseMatrix Kad(Wall1::neas_,2*NumNode());

      // save EAS data into element container easdata_
      data_.Add("alpha",alpha);
      data_.Add("alphao",alphao);
      data_.Add("feas",feas);
      data_.Add("invKaa",invKaa);
      data_.Add("Kda",Kda);
      data_.Add("Kad",Kad); // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
    }
  }

  // EAS type
  if (iseas_)
  {
    eastype_ = eas_q1e4;
  }
  else
  {
    eastype_ = eas_vague;
  }

  //read lokal or global stresses
  char buffer [50];
  frchar("STRESSES",buffer,&ierr);
  if (ierr)
  {
     if      (strncmp(buffer,"XY",2)==0)       stresstype_ = w1_xy;
     else if (strncmp(buffer,"RS",2)==0)       stresstype_ = w1_rs;
     else dserror ("Reading of WALL1 element failed");
  }

  return true;
} // Wall1::ReadElement()
#endif

/*----------------------------------------------------------------------*
 |  Get gaussrule on dependance of gausspoints                     mgit |
 *----------------------------------------------------------------------*/
DRT::UTILS::GaussRule2D DRT::ELEMENTS::Wall1::getGaussrule(int* ngp)
{
  DRT::UTILS::GaussRule2D rule = DRT::UTILS::intrule2D_undefined;

  switch (Shape())
  {
    case DRT::Element::quad4:
    case DRT::Element::quad8:
    case DRT::Element::quad9:
    case DRT::Element::nurbs4:
    case DRT::Element::nurbs9:
    {
       if ( (ngp[0]==2) && (ngp[1]==2) )
       {
         rule = DRT::UTILS::intrule_quad_4point;
       }
       else if ( (ngp[0]==3) && (ngp[1]==3) )
       {
         rule = DRT::UTILS::intrule_quad_9point;
       }
       else
         dserror("Unknown number of Gauss points for quad element");
       break;
    }
    case DRT::Element::tri3:
    case DRT::Element::tri6:
    {
      if ( (ngp[0]==1) && (ngp[1]==0) )
      {
        rule = DRT::UTILS::intrule_tri_1point;
      }
      else if ( (ngp[0]==3) && (ngp[1]==0) )
      {
        rule = DRT::UTILS::intrule_tri_3point;
      }
      else if ( (ngp[0]==6) && (ngp[1]==0) )
      {
        rule = DRT::UTILS::intrule_tri_6point;
      }
      else
        dserror("Unknown number of Gauss points for tri element");
      break;
    }
    default:
       dserror("Unknown distype");
  }
  return rule;
}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_WALL1
