/*!----------------------------------------------------------------------
\file so_disp_input.cpp
\brief

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET


#include "so_disp.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_mat/elasthyper.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::SoDisp::ReadElement(const std::string& eletype,
                                        const std::string& distype,
                                        DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  // special element-dependent input of material parameters
  if (Material()->MaterialType() == INPAR::MAT::m_elasthyper){
    MAT::ElastHyper* elahy = static_cast <MAT::ElastHyper*>(Material().get());
    elahy->Setup(linedef);
  }

  DiscretizationType shape = StringToDistype(distype);

  switch (shape)
  {
  case hex8: case hex20: case hex27:
  {
    std::vector<int> ngp;
    linedef->ExtractIntVector("GP",ngp);

    switch (ngp[0])
    {
    case 1:
      gaussrule_ = DRT::UTILS::intrule_hex_1point;
      break;
    case 2:
      gaussrule_ = DRT::UTILS::intrule_hex_8point;
      break;
    case 3:
      gaussrule_ = DRT::UTILS::intrule_hex_27point;
      break;
    default:
      dserror("Reading of SOLID3 element failed: Gaussrule for hexaeder not supported!");
    }
    break;
  }
  case pyramid5:
  {
    int ngp;
    linedef->ExtractInt("GP_PYRAMID",ngp);
    switch (ngp)
    {
    case 1:
      gaussrule_ = DRT::UTILS::intrule_pyramid_1point;
      break;
    case 8:
      gaussrule_ = DRT::UTILS::intrule_pyramid_8point;
      break;
    default:
      dserror("Reading of SOLID3 element failed: Gaussrule for pyramid not supported!\n");
    }
    break;
  }
  case tet4: case tet10:
  {
    int ngp;
    linedef->ExtractInt("GP_TET",ngp);

    std::string buffer;
    linedef->ExtractString("GP_ALT",buffer);

    switch(ngp)
    {
    case 1:
      if (buffer=="standard")
        gaussrule_ = DRT::UTILS::intrule_tet_1point;
      else
        dserror("Reading of SOLID3 element failed: GP_ALT: gauss-radau not possible!\n");
      break;
    case 4:
      if (buffer=="standard")
        gaussrule_ = DRT::UTILS::intrule_tet_4point;
      else if (buffer=="gaussrad")
        gaussrule_ = DRT::UTILS::intrule_tet_4point_gauss_radau;
      else
        dserror("Reading of SOLID3 element failed: GP_ALT\n");
      break;
    case 10:
      if (buffer=="standard")
        gaussrule_ = DRT::UTILS::intrule_tet_5point;
      else
        dserror("Reading of SOLID3 element failed: GP_ALT: gauss-radau not possible!\n");
      break;
    default:
      dserror("Reading of SOLID3 element failed: Gaussrule for tetraeder not supported!\n");
    }
    break;
  } // end reading gaussian points for tetrahedral elements
  default:
    dserror("Reading of SOLID3 element failed: integration points\n");
  } // end switch distype

  std::string buffer;
  linedef->ExtractString("KINEM",buffer);

  // geometrically linear
  if      (buffer=="Geolin")    kintype_ = sodisp_geolin;
  // geometrically non-linear with Total Lagrangean approach
  else if (buffer=="Totlag")    kintype_ = sodisp_totlag;
  // geometrically non-linear with Updated Lagrangean approach
  else if (buffer=="Updlag")
  {
    kintype_ = sodisp_updlag;
    dserror("Updated Lagrange for SOLID3 is not implemented!");
  }
  else dserror("Reading of SOLID3 element failed");

  numnod_disp_ = NumNode();      // number of nodes
  numdof_disp_ = NumNode() * NODDOF_DISP;     // total dofs per element
  const DRT::UTILS::IntegrationPoints3D  intpoints(gaussrule_);
  numgpt_disp_ = intpoints.nquad;      // total gauss points per element

  return true;
}

#if 0
/*----------------------------------------------------------------------*
 |  read element input (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::SoDisp::ReadElement()
{
    typedef map<string, DiscretizationType> Gid2DisType;
    Gid2DisType gid2distype;
    gid2distype["HEX8"]  = hex8;
    gid2distype["HEX20"] = hex20;
    gid2distype["HEX27"] = hex27;
    gid2distype["TET4"]  = tet4;
    gid2distype["TET10"] = tet10;
    gid2distype["PYRAMID5"] = pyramid5;

    // read element's nodes
    int   ierr = 0;
    int   nnode = 0;
    int   nodes[27];
    DiscretizationType distype = dis_none;

    Gid2DisType::iterator iter;
    for( iter = gid2distype.begin(); iter != gid2distype.end(); iter++ )
    {
        const string eletext = iter->first;
        frchk(eletext.c_str(), &ierr);
        if (ierr == 1)
        {
            distype = gid2distype[eletext];
            nnode = DRT::UTILS::getNumberOfElementNodes(distype);
            frint_n(eletext.c_str(), nodes, nnode, &ierr);
            dsassert(ierr==1, "Reading of ELEMENT Topology failed\n");
            break;
        }
    }

    //cout << "reading " << DRT::DistypeToString(distype) << endl;

//    const bool allowed_element = (distype == hex27) || (distype == hex20) || (distype == tet10) || (distype == wedge15);
    // The intention of this element is to help debugging the xfem intersection routines.
    // The element is purely displacement based and has not been tested for correct computation
    // It serves solely as higher order geometry input data to XFEM problems
//    if (not allowed_element)
//    {
//        dserror("Only quadratic order for DISP (displacement based) element. LOCKING!\n for linear elements, use EAS enabled elements");
//    }

    // reduce node numbers by one
    for (int i=0; i<nnode; ++i) nodes[i]--;

    SetNodeIds(nnode,nodes);

    // read number of material model
    int material = 0;
    frint("MAT",&material,&ierr);
    dsassert(ierr==1, "Reading of material for SOLID3 element failed\n");
    dsassert(material!=0, "No material defined for SOLID3 element\n");
    SetMaterial(material);


    // read gaussian points and set gaussrule
    char  buffer[50];
    int ngp[3];
    switch (distype)
    {
    case hex8: case hex20: case hex27:
    {
        frint_n("GP",ngp,3,&ierr);
        dsassert(ierr==1, "Reading of SOLID3 element failed: GP\n");
        switch (ngp[0])
        {
        case 1:
            gaussrule_ = DRT::UTILS::intrule_hex_1point;
            break;
        case 2:
            gaussrule_ = DRT::UTILS::intrule_hex_8point;
            break;
        case 3:
            gaussrule_ = DRT::UTILS::intrule_hex_27point;
            break;
        default:
            dserror("Reading of SOLID3 element failed: Gaussrule for hexaeder not supported!\n");
        }
        break;
    }
    case pyramid5:
    {
        frint_n("GP_PYRAMID",ngp,1,&ierr);
        dsassert(ierr==1, "Reading of SOLID3 element failed: GP_PYRAMID\n");
        switch (ngp[0])
        {
        case 1:
            gaussrule_ = DRT::UTILS::intrule_pyramid_1point;
            break;
        case 8:
            gaussrule_ = DRT::UTILS::intrule_pyramid_8point;
            break;
        default:
            dserror("Reading of SOLID3 element failed: Gaussrule for pyramid not supported!\n");
        }
        break;
    }
    case tet4: case tet10:
    {
//        gaussrule_ = DRT::UTILS::intrule_tet_4point;
        frint("GP_TET",&ngp[0],&ierr);
        dsassert(ierr==1, "Reading of SOLID3 element failed: GP_TET\n");

        frchar("GP_ALT",buffer,&ierr);
        dsassert(ierr==1, "Reading of SOLID3 element failed: GP_ALT\n");

        switch(ngp[0])
        {
        case 1:
            if (strncmp(buffer,"standard",8)==0)
                gaussrule_ = DRT::UTILS::intrule_tet_1point;
            else
                dserror("Reading of SOLID3 element failed: GP_ALT: gauss-radau not possible!\n");
            break;
        case 4:
            if (strncmp(buffer,"standard",8)==0)
                gaussrule_ = DRT::UTILS::intrule_tet_4point;
            else if (strncmp(buffer,"gaussrad",8)==0)
                gaussrule_ = DRT::UTILS::intrule_tet_4point_gauss_radau;
            else
                dserror("Reading of SOLID3 element failed: GP_ALT\n");
            break;
        case 10:
            if (strncmp(buffer,"standard",8)==0)
                gaussrule_ = DRT::UTILS::intrule_tet_5point;
            else
                dserror("Reading of SOLID3 element failed: GP_ALT: gauss-radau not possible!\n");
            break;
        default:
            dserror("Reading of SOLID3 element failed: Gaussrule for tetraeder not supported!\n");
        }
        break;
    } // end reading gaussian points for tetrahedral elements
    default:
        dserror("Reading of SOLID3 element failed: integration points\n");
    } // end switch distype

    // we expect kintype to be total lagrangian
    kintype_ = sodisp_totlag;

    // eventually read kinematic type
    frchar("KINEM",buffer,&ierr);
    if (ierr)
    {
     // geometrically linear
     if      (strncmp(buffer,"Geolin",6)==0)    kintype_ = sodisp_geolin;
     // geometrically non-linear with Total Lagrangean approach
     else if (strncmp(buffer,"Totlag",6)==0)    kintype_ = sodisp_totlag;
     // geometrically non-linear with Updated Lagrangean approach
     else if (strncmp(buffer,"Updlag",6)==0)
     {
         kintype_ = sodisp_updlag;
         dserror("Updated Lagrange for SOLID3 is not implemented!");
     }
     else dserror("Reading of SOLID3 element failed");
    }

    numnod_disp_ = NumNode();      // number of nodes
    numdof_disp_ = NumNode() * NODDOF_DISP;     // total dofs per element
    const DRT::UTILS::IntegrationPoints3D  intpoints(gaussrule_);
    numgpt_disp_ = intpoints.nquad;      // total gauss points per element

  return true;

} // SoDisp::ReadElement()
#endif

#endif  // #ifdef CCADISCRET
