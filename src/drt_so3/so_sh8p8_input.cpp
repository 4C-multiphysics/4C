/*----------------------------------------------------------------------*/
/*!
\file so_sh8p8_input.cpp
\brief

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */

#include "so_sh8p8.H"
#include "../drt_mat/artwallremod.H"
#include "../drt_mat/anisotropic_balzani.H"
#include "../drt_mat/viscoanisotropic.H"
#include "../drt_mat/visconeohooke.H"
#include "../drt_mat/viscogenmax.H"
#include "../drt_mat/elasthyper.H"
#include "../drt_mat/aaaraghavanvorp_damage.H"
#include "../drt_lib/drt_linedefinition.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::So_sh8p8::ReadElement(const std::string& eletype,
                                          const std::string& distype,
                                          DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  // special element-dependent input of material parameters
  switch (Material()->MaterialType())
  {
  case INPAR::MAT::m_artwallremod:
  {
    MAT::ArtWallRemod* remo = static_cast <MAT::ArtWallRemod*>(Material().get());
    remo->Setup(NUMGPT_SOH8, this->Id(), linedef);
    break;
  }
  case INPAR::MAT::m_anisotropic_balzani:
  {
    MAT::AnisotropicBalzani* balz = static_cast <MAT::AnisotropicBalzani*>(Material().get());
    balz->Setup(linedef);
    break;
  }
  case INPAR::MAT::m_viscoanisotropic:
  {
    MAT::ViscoAnisotropic* visco = static_cast <MAT::ViscoAnisotropic*>(Material().get());
    visco->Setup(NUMGPT_SOH8, linedef);
    break;
  }
  case INPAR::MAT::m_visconeohooke:
  {
    MAT::ViscoNeoHooke* visco = static_cast <MAT::ViscoNeoHooke*>(Material().get());
    visco->Setup(NUMGPT_SOH8);
    break;
  }
  case INPAR::MAT::m_viscogenmax:
  {
    MAT::ViscoGenMax* viscogenmax = static_cast <MAT::ViscoGenMax*>(Material().get());
    viscogenmax->Setup(NUMGPT_SOH8,linedef);
    break;
  }
  case INPAR::MAT::m_elasthyper:
  {
    MAT::ElastHyper* elahy = static_cast <MAT::ElastHyper*>(Material().get());
    elahy->Setup(linedef);
    break;
  }
  case INPAR::MAT::m_aaaraghavanvorp_damage:
  {
    double strength = 0.0; // section for extracting the element strength
    linedef->ExtractDouble("STRENGTH",strength);
    MAT::AAAraghavanvorp_damage* aaadamage = static_cast <MAT::AAAraghavanvorp_damage*>(Material().get());
    aaadamage->Setup(NUMGPT_SOH8,strength);
    //aaadamage->Setup(NUMGPT_SOH8);
  }
  default:
    // Do nothing. Simple material.
    break;
  }

  // a temprorary variable for read-in
  std::string buffer;
  // read kinematic flag
  linedef->ExtractString("KINEM",buffer);
  if (buffer=="linear")
  {
    kintype_ = soh8_linear;
    dserror ("Only nonlinear kinematics for SO_SH8P8 implemented!");
  }
  else if (buffer=="nonlinear")
  {
    kintype_ = soh8_nonlinear;
  }
  else dserror ("Reading SO_SH8P8 element failed unknown KINEM Type");


  // we expect kintype to be total lagrangian
  kintype_ = soh8_nonlinear;
  

  // read EAS technology flag
  linedef->ExtractString("EAS",buffer);

  if (buffer=="sosh8")
  {
    eastype_ = soh8_eassosh8;
    neas_ = NUMEAS_SOSH8_;
  }
  else if (buffer=="atype")
  {
    eastype_ = soh8_easa;
    neas_ = NUMEAS_A_;
  }
  else if (buffer=="None")
  {
    eastype_ = soh8_easnone;
    neas_ = 0;
  }
  else if (buffer=="none")
  {
    eastype_ = soh8_easnone;
    neas_ = 0;
  }
  else
    dserror("Reading of SO_SH8P8 EAS type failed");

  if (eastype_ != soh8_easnone)
  {
    EasInit();
  }

  linedef->ExtractString("THICKDIR",buffer);
  nodes_rearranged_ = false;

  // global X
  if      (buffer=="xdir")    thickdir_ = globx;
  // global Y
  else if (buffer=="ydir")    thickdir_ = globy;
  // global Z
  else if (buffer=="zdir")    thickdir_ = globz;
  // find automatically through Jacobian of Xrefe
  else if (buffer=="auto")    thickdir_ = autoj;
  // local r
  else if (buffer=="rdir")    thickdir_ = enfor;
  // local s
  else if (buffer=="sdir")    thickdir_ = enfos;
  // local t
  else if (buffer=="tdir")    thickdir_ = enfot;
  // no noderearrangement
  else if (buffer=="none")
  {
    thickdir_ = none;
    nodes_rearranged_ = true;
  }
  else dserror("Reading of SO_SH8P8 thickness direction failed");

  linedef->ExtractString("STAB",buffer);
  if (buffer=="Aff")
    stab_ = stab_affine;
  else if (buffer=="NonAff")
    stab_ = stab_nonaffine;
  else if (buffer=="SpatAff")
    stab_ = stab_spatialaffine;
  else if (buffer=="Spat")
    stab_ = stab_spatial;
  else if (buffer=="PureDisp")
    stab_ = stab_puredisp;
  else
    dserror("Reading of SO_SH8P8 stabilisation failed");

  linedef->ExtractString("ANS",buffer);
  if (buffer=="Later")
    ans_ = ans_lateral;
  else if (buffer=="OnSpot")
    ans_ = ans_onspot;
  else if (buffer=="None")
    ans_ = ans_none;
  else
    dserror("Reading of SO_SH8P8 ANS type failed");

  // Linearization
  linedef->ExtractString("LIN",buffer);
  if (buffer=="One")
    lin_ = lin_one;
  else if (buffer=="Half")
    lin_ = lin_half;
  else if (buffer=="Sixth")
    lin_ = lin_sixth;
  else
    dserror("Reading of SO_SH8P8 LIN type failed");

  // Isochoric way
  linedef->ExtractString("ISO",buffer);
  if (buffer=="Mat")
    iso_ = iso_material;
  else if (buffer=="Enf")
    iso_ = iso_enforced;
  else
    dserror("Reading of SO_SH8P8 ISO type failed");

  return true;
}


#if 0
/*----------------------------------------------------------------------*
 |  read element input (public)                                         |
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::So_sh8p8::ReadElement()
{
  // read element's nodes
  int ierr = 0;
  const int nnode = NUMNOD_;
  int nodes[NUMNOD_];
  frchk("SOLIDSH8P8",&ierr);
  if (ierr==1)
  {
    frint_n("HEX8",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  else
  {
    dserror ("Reading of SOLIDSH8P8 failed");
  }
  // reduce node numbers by one
  for (int i=0; i<nnode; ++i){
    nodes[i]--;
  }

  SetNodeIds(nnode,nodes);


  // read number of material model
  int material = 0;
  frint("MAT",&material,&ierr);
  if (ierr!=1) dserror("Reading of SO_SH8P8 element material failed");
  SetMaterial(material);

  // special element-dependent input of material parameters
  if (Material()->MaterialType() == INPAR::MAT::m_artwallremod){
    MAT::ArtWallRemod* remo = static_cast <MAT::ArtWallRemod*>(Material().get());
    remo->Setup(NUMGPT_SOH8, this->Id());
  } else if (Material()->MaterialType() == INPAR::MAT::m_anisotropic_balzani){
    MAT::AnisotropicBalzani* balz = static_cast <MAT::AnisotropicBalzani*>(Material().get());
    balz->Setup();
  } else if (Material()->MaterialType() == INPAR::MAT::m_viscoanisotropic){
    MAT::ViscoAnisotropic* visco = static_cast <MAT::ViscoAnisotropic*>(Material().get());
    visco->Setup(NUMGPT_SOH8);
  } else if (Material()->MaterialType() == INPAR::MAT::m_visconeohooke){
    MAT::ViscoNeoHooke* visco = static_cast <MAT::ViscoNeoHooke*>(Material().get());
    visco->Setup(NUMGPT_SOH8);
  } else if (Material()->MaterialType() == INPAR::MAT::m_viscogenmax){
    MAT::ViscoGenMax* viscogenmax = static_cast <MAT::ViscoGenMax*>(Material().get());
    viscogenmax->Setup(NUMGPT_SOH8,linedef);
  } else if (Material()->MaterialType() == INPAR::MAT::m_elasthyper){
    MAT::ElastHyper* elahy = static_cast <MAT::ElastHyper*>(Material().get());
    elahy->Setup();
  }
  else if (Material()->MaterialType() == INPAR::MAT::m_aaaraghavanvorp_damage){
    double strength = 0.0; // section for extracting the element strength
    frdouble("STRENGTH",&strength,&ierr);
    if (ierr!=1) dserror("Reading of SO_SH8 element strength failed");
    MAT::AAAraghavanvorp_damage* aaadamage = static_cast <MAT::AAAraghavanvorp_damage*>(Material().get());
    aaadamage->Setup(NUMGPT_SOH8,strength);
//    aaadamage->Setup(NUMGPT_SOH8);
  }

  // read possible gaussian points, obsolete for computation
  {
    int ngp[3];
    frint_n("GP",ngp,3,&ierr);
    if (ierr==1) for (int i=0; i<3; ++i) if (ngp[i]!=2) dserror("Only 2 GP for So_SH8P8");
  }

  // we expect kintype to be total lagrangian
  kintype_ = soh8_totlag;

  // read kinematic type
  char buffer[50];
  frchar("KINEM",buffer,&ierr);
  if (ierr)
  {
    // geometrically linear
    if      (strncmp(buffer,"Geolin",6)==0)    kintype_ = soh8_geolin;
    // geometrically non-linear with Total Lagrangean approach
    else if (strncmp(buffer,"Totlag",6)==0)    kintype_ = soh8_totlag;
    // geometrically non-linear with Updated Lagrangean approach
    else if (strncmp(buffer,"Updlag",6)==0)
    {
      kintype_ = soh8_updlag;
      dserror("Updated Lagrange for SO_SH8P8 is not implemented!");
    }
    else dserror("Reading of SO_SH8P8 element failed");
  }

  // set EAS technology flag
  eastype_ = soh8_easnone;
  neas_ = 0;

  // read global coordinate of shell-thickness direction
  thickdir_ = autoj;           // default: auto by Jacobian
  nodes_rearranged_ = false;
  frchar("THICKDIR",buffer,&ierr);
  if (ierr)
  {
    // global X
    if      (strncmp(buffer,"xdir",4)==0)    thickdir_ = globx;
    // global Y
    else if (strncmp(buffer,"ydir",4)==0)    thickdir_ = globy;
    // global Z
    else if (strncmp(buffer,"zdir",4)==0)    thickdir_ = globz;
    // find automatically through Jacobian of Xrefe
    else if (strncmp(buffer,"auto",4)==0)    thickdir_ = autoj;
    // local r
    else if (strncmp(buffer,"rdir",4)==0)    thickdir_ = enfor;
    // local s
    else if (strncmp(buffer,"sdir",4)==0)    thickdir_ = enfos;
    // local t
    else if (strncmp(buffer,"tdir",4)==0)    thickdir_ = enfot;
    // no noderearrangement
    else if (strncmp(buffer,"none",4)==0){
      thickdir_ = none;
      nodes_rearranged_ = true;
    }
    else dserror("Reading of SO_SH8P8 thickness direction failed");
  }

  // stabilisation
  stab_ = stab_affine;
  frchar("STAB",buffer,&ierr);
  if (ierr)
  {
    if (strncmp(buffer,"Aff",3)==0)
      stab_ = stab_affine;
    else if (strncmp(buffer,"NonAff",6)==0)
      stab_ = stab_nonaffine;
    else if (strncmp(buffer,"SpatAff",7)==0)
      stab_ = stab_spatialaffine;
    else if (strncmp(buffer,"Spat",4)==0)
      stab_ = stab_spatial;
    else if (strncmp(buffer,"PureDisp",8)==0)
      stab_ = stab_puredisp;
    else
      dserror("Reading of SO_SH8P8 stabilisation failed");
  }

  // ANS
  ans_ = ans_lateral;
  frchar("ANS",buffer,&ierr);
  if (ierr)
  {
    if (strncmp(buffer,"Later",5)==0)
      ans_ = ans_lateral;
    else if (strncmp(buffer,"OnSpot",6)==0)
      ans_ = ans_onspot;
    else if (strncmp(buffer,"None",4)==0)
      ans_ = ans_none;
    else
      dserror("Reading of SO_SH8P8 ANS type failed");
  }

  // EAS
  eastype_ = soh8_easnone;
  neas_ = 0;
  frchar("EAS",buffer,&ierr);
  if (ierr)
  {
    if (strncmp(buffer,"sosh8",5)==0) {
      eastype_ = soh8_eassosh8;
      neas_ = NUMEAS_SOSH8_;
    }
    else if (strncmp(buffer,"atype",5)==0) {
      eastype_ = soh8_easa;
      neas_ = NUMEAS_A_;
    }
    else if (strncmp(buffer,"None",4)==0)
      eastype_ = soh8_easnone;
    else if (strncmp(buffer,"none",4)==0)
      eastype_ = soh8_easnone;
    else
      dserror("Reading of SO_SH8P8 EAS type failed");
  }
  if (eastype_ != soh8_easnone)
  {
    EasInit();
  }

  // Linearization
  lin_ = lin_one;
  frchar("LIN",buffer,&ierr);
  if (ierr)
  {
    if (strncmp(buffer,"One",3)==0)
      lin_ = lin_one;
    else if (strncmp(buffer,"Half",4)==0)
      lin_ = lin_half;
    else if (strncmp(buffer,"Sixth",5)==0)
      lin_ = lin_sixth;
    else
      dserror("Reading of SO_SH8P8 LIN type failed");
  }

  // Isochoric way
  iso_ = iso_material;
  frchar("ISO",buffer,&ierr);
  if (ierr)
  {
    if (strncmp(buffer,"Mat",3)==0)
      iso_ = iso_material;
    else if (strncmp(buffer,"Enf",3)==0)
      iso_ = iso_enforced;
    else
      dserror("Reading of SO_SH8P8 ISO type failed");
  }

  return true;
} // So_sh8p8::ReadElement()
#endif

#endif  // #ifdef CCADISCRET
