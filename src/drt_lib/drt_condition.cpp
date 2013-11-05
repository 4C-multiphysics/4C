/*!----------------------------------------------------------------------
\file drt_condition.cpp

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

\brief A condition of any kind

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/

#include "drt_condition.H"
#include "drt_element.H"


DRT::ConditionObjectType DRT::ConditionObjectType::instance_;


DRT::ParObject* DRT::ConditionObjectType::Create( const std::vector<char> & data )
{
  DRT::Condition* object = new DRT::Condition();
  object->Unpack(data);
  return object;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::Condition::Condition(const int id, const ConditionType type,
                          const bool buildgeometry, const GeometryType gtype) :
Container(),
id_(id),
buildgeometry_(buildgeometry),
type_(type),
gtype_(gtype),
comm_(Teuchos::null)
{
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::Condition::Condition() :
Container(),
id_(-1),
buildgeometry_(false),
type_(none),
gtype_(NoGeom),
comm_(Teuchos::null)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::Condition::Condition(const DRT::Condition& old) :
Container(old),
id_(old.id_),
buildgeometry_(old.buildgeometry_),
type_(old.type_),
gtype_(old.gtype_),
comm_(old.comm_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::Condition::~Condition()
{
  return;
}


/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 11/06|
 *----------------------------------------------------------------------*/
std::ostream& operator << (std::ostream& os, const DRT::Condition& cond)
{
  cond.Print(os);
  return os;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Condition::Print(std::ostream& os) const
{
  os << "Condition " << Id() << " ";
  if      (Type()==PointDirichlet)                os << "Point Dirichlet boundary condition: ";
  else if (Type()==LineDirichlet)                 os << "Line Dirichlet boundary condition: ";
  else if (Type()==SurfaceDirichlet)              os << "Surface Dirichlet boundary condition: ";
  else if (Type()==VolumeDirichlet)               os << "Volume Dirichlet boundary condition: ";
  else if (Type()==PointNeumann)                  os << "Point Neumann boundary condition: ";
  else if (Type()==LineNeumann)                   os << "Line Neumann boundary condition: ";
  else if (Type()==SurfaceNeumann)                os << "Surface Neumann boundary condition: ";
  else if (Type()==VolumeNeumann)                 os << "Volume Neumann boundary condition: ";
  else if (Type()==PointInitfield)                os << "Point Initfield boundary condition: ";
  else if (Type()==LineInitfield)                 os << "Line Initfield boundary condition: ";
  else if (Type()==SurfaceInitfield)              os << "Surface Initfield boundary condition: ";
  else if (Type()==VolumeInitfield)               os << "Volume Initfield boundary condition: ";
  else if (Type()==Mortar)                        os << "Mortar coupling boundary condition: ";
  else if (Type()==AleWear)                       os << "ALE Wear boundary condition: ";
  else if (Type()==PointLocsys)                   os << "Point local coordinate system condition: ";
  else if (Type()==LineLocsys)                    os << "Line local coordinate system condition: ";
  else if (Type()==SurfaceLocsys)                 os << "Surface local coordinate system condition: ";
  else if (Type()==VolumeLocsys)                  os << "Volume local coordinate system condition: ";
  else if (Type()==FSICoupling)                   os << "FSI Coupling condition: ";
  else if (Type()==XFEMCoupling)                  os << "XFEM Coupling condition: ";
  else if (Type()==FluidFluidCoupling)            os << "Fluid Fluid Coupling condition: ";
  else if (Type()==ALEFluidCoupling)              os << "ALE Fluid Coupling condition: ";
  else if (Type()==MovingFluid)                   os << "Moving Fluid Vol condition: ";
  else if (Type()==LineLIFTDRAG)                  os << "Line LIFTDRAG condition: ";
  else if (Type()==SurfLIFTDRAG)                  os << "Surf LIFTDRAG condition: ";
  else if (Type()==SurfaceTension)                os << "Surface tension condition: ";
  else if (Type()==Surfactant)                    os << "Surfactant condition: ";
  else if (Type()==MicroBoundary)                 os << "Microscale boundary condition: ";
  else if (Type()==VolumeConstraint_3D)           os << "Volume constraint surface boundary condition: ";
  else if (Type()==AreaConstraint_3D)             os << "Area constraint surface boundary condition: ";
  else if (Type()==AreaConstraint_2D)             os << "Area constraint surface boundary condition: ";
  else if (Type()==VolumeMonitor_3D)              os << "Volume monitor condition: ";
  else if (Type()==AreaMonitor_3D)                os << "Area monitor condition: ";
  else if (Type()==AreaMonitor_2D)                os << "Area monitor condition: ";
  else if (Type()==ImpedanceCond)                 os << "Impedance boundary condition: ";
  else if (Type()==Impedance_Calb_Cond)           os << "Impedance calibration boundary condition: ";
  else if (Type()==MPC_NodeOnPlane_3D)            os << "Multipoint constraint on a plane: ";
  else if (Type()==MPC_NodeOnLine_3D)             os << "Multipoint constraint on a line: ";
  else if (Type()==MPC_NodeOnLine_2D)             os << "Multipoint constraint on a line: ";
  else if (Type()==LJ_Potential_Volume)           os << "Lennard-Jones potential in a volume: ";
  else if (Type()==LJ_Potential_Surface)          os << "Lennard-Jones potential on a surface: ";
  else if (Type()==LJ_Potential_Line)             os << "Lennard-Jones potential on a line: ";
  else if (Type()==VanDerWaals_Potential_Volume)  os << "Van der Waals potential in a volume: ";
  else if (Type()==VanDerWaals_Potential_Surface) os << "Van der Waals potential on a surface: ";
  else if (Type()==VanDerWaals_Potential_Line)    os << "Van der Waals potential on a line: ";
  else if (Type()==ElectroRepulsion_Potential_Surface) os << "Electro repulsion potential on a surface: ";
  else if (Type()==ElectroRepulsion_Potential_Line)    os << "Electro repulsion potential on a line: ";
  else if (Type()==LineFlowDepPressure)             os << "line flow-dependent pressure condition: ";
  else if (Type()==SurfaceFlowDepPressure)          os << "surface flow-dependent pressure condition: ";
  else if (Type()==LineWeakDirichlet)             os << "line weak Dirichlet condition: ";
  else if (Type()==SurfaceWeakDirichlet)          os << "surface weak Dirichlet condition: ";
  else if (Type()==LinePeriodic)                  os << "line periodic boundary condition: ";
  else if (Type()==SurfacePeriodic)               os << "surface periodic boundary condition: ";
  else if (Type()==TransferTurbulentInflow)       os << "transfer turbulent inflow: ";
  else if (Type()==TurbulentInflowSection)        os << "turbulent inflow section: ";
  else if (Type()==BlendMaterial)                 os << "blend materials: ";
  else if (Type()==Brownian_Motion)               os << "stochastical surface condition (Brownian Motion): ";
  else if (Type()==FilamentNumber)                os << "line condition for polymer networks: ";
  else if (Type()==ForceSensor)                   os << "marking points in a system where force sensors are applied: ";
  else if (Type()==FlowRateThroughLine_2D)        os << "Monitor flow rate through an line interface: ";
  else if (Type()==FlowRateThroughSurface_3D)     os << "Monitor flow rate through a surface interface: ";
  else if (Type()==ImpulsRateThroughSurface_3D)   os << "Monitor impuls rate through a interface: ";
  else if (Type()==FluidNeumannInflow)            os << "Fluid Neumann inflow: ";
  else if (Type()==ElectrodeKinetics)             os << "ElectrodeKinetics boundary condition: ";
  else if (Type()==ArtJunctionCond)               os << "Artery junction boundary condition";
  else if (Type()==ArtWriteGnuplotCond)           os << "Artery write gnuplot format condition";
  else if (Type()==ArtPrescribedCond)             os << "Artery prescribed boundary condition";
  else if (Type()==ArtRfCond)                     os << "Artery reflective boundary condition";
  else if (Type()==ArtWkCond)                     os << "Artery windkessel boundary condition";
  else if (Type()==StructAleCoupling)             os << "Structure - ALE coupling condition";
  else if (Type()==StructFluidSurfCoupling)       os << "Structure - Fluid surface coupling condition";
  else if (Type()==StructFluidVolCoupling)        os << "Structure - Fluid volume coupling condition";
  else if (Type()==BioGrCoupling)                 os << "Biofilm growth coupling condition: ";
  else if (Type()==ArtInOutletCond)               os << "Artery terminal in_outlet condition";
  else if (Type()==ArtRedTo3DCouplingCond)        os << "Artery reduced D 3D coupling condition";
  else if (Type()==Art3DToRedCouplingCond)        os << "Artery 3D reduced D coupling condition";
  else if (Type()==RedAirwayPrescribedCond)       os << "Reduced d airway prescribed boundary condition";
  else if (Type()==RedAirwayPrescribedExternalPressure) os << "Reduced d airway prescribed external pressure boundary condition";
  else if (Type()==PatientSpecificData)           os << "Various Geometric Patient Specific Data";
  else if (Type()==VolumetricSurfaceFlowCond)     os << "Volumetric Surface Flow Profile";
  else if (Type()==VolumetricFlowBorderNodes)     os << "Border Nodes of the volumetric flow Surface";
  else if (Type()==VolSTCLayer)                   os << "Number of current STC layer";
  else if (Type()==ThermoConvections)             os << "ThermoConvections boundary condition: ";
  else if (Type()==FSICouplingCenterDisp)         os << "Sliding ALE Center Disp condition";
  else if (Type()==FSICouplingNoSlide)            os << "Do not consider these nodes for sliding ALE";
  else if (Type()==EmbeddingTissue)               os << "Embedding Tissue Condition";
  else if (Type()==TotalTractionCorrectionCond)   os << "Total traction correct condition";
  else if (Type()==NoPenetration)                 os << "No Penetration Condition";
  else if (Type()==TotalTractionCorrectionBorderNodes)  os << "Total traction correction border nodes condition";
  else if (Type()==RedAirwayVentilatorCond)       os << "Reduced d airway prescribed ventilator condition";
  else if (Type()==RedAirwayTissue)               os << "tissue RedAirway coupling surface condition";
  else if (Type()==RedAirwayNodeTissue)           os << "tissue RedAirway coupling node condition: ";
  else if (Type()==PoroCoupling)                  os << "porous media coupling condition: ";
  else if (Type()==PoroPartInt)                   os << "porous media partial integration condition: ";
  else if (Type()==PoroPresInt)                   os << "porous media pressure integration condition: ";
  else if (Type()==ScaTraCoupling)                os << "scatra coupling condition";
  else if (Type()==RedAirwayPrescribedScatraCond) os << "Reduced d airway prescribed scatra boundary condition";
  else if (Type()==ArtPrescribedScatraCond)       os << "one-D Arterial prescribed scatra boundary condition";
  else if (Type()==RedAirwayInitialScatraCond)    os << "Reduced d airway initial scatra boundary condition";
  else if (Type()==RedAirwayScatraExchangeCond)   os << "Reduced d airway scatra exchange condition";
  else if (Type()==RedAirwayScatraHemoglobinCond) os << "Reduced d airway scatra hemoglobin condition";
  else if (Type()==RedAirwayScatraAirCond)        os << "Reduced d airway scatra air condition";
  else if (Type()==RedAirwayScatraCapillaryCond)  os << "Reduced d airway scatra capillary condition";
  else if (Type()==ParticleInflow)                os << "particle inflow condition";
  else if (Type()==ParticleInitRadius)            os << "particle initial radius condition";
  else if (Type()==ParticleWall)                  os << "particle wall condition";
  else if (Type()==CrackMastersurface)            os << "Master crack surface";
  else if (Type()==CrackSlavesurface)             os << "Slave crack surface";
  else if (Type()==SurfaceModeKrylovProjection)   os << "Surface mode for Krylov space projection";
  else if (Type()==VolumeModeKrylovProjection)    os << "Volume mode for Krylov space projection";
  else if (Type()==HomoScaTraCoupling)    		  os << "Homogeneous ScaTra Coulping";
  else dserror("no output std::string for condition defined in DRT::Condition::Print");

  Container::Print(os);
  if ((int)geometry_.size())
  {
    os << std::endl;
    os << "Elements of this condition:\n";
    std::map<int,Teuchos::RCP<DRT::Element> >::const_iterator curr;
    for (curr=geometry_.begin(); curr!=geometry_.end(); ++curr)
      os << "      " << *(curr->second) << std::endl;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::Condition::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class container
  Container::Pack(data);
  // id_
  AddtoPack(data,id_);
  // buildgeometry_
  AddtoPack(data,buildgeometry_);
  // type_
  AddtoPack(data,type_);
  // gtype_
  AddtoPack(data,gtype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::Condition::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Container
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Container::Unpack(basedata);
  // id_
  ExtractfromPack(position,data,id_);
  // buildgeometry_
  buildgeometry_ = ExtractInt(position,data);
  // type_
  type_ = static_cast<ConditionType>( ExtractInt(position,data) );
  // gtype_
  gtype_ = static_cast<GeometryType>( ExtractInt(position,data) );

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |  Adjust Ids of elements in order to obtain unique Ids within one     |
 |  condition type                                                      |
 |                                                             lw 12/07 |
 *----------------------------------------------------------------------*/
void DRT::Condition::AdjustId(const int shift)
{
  std::map<int,Teuchos::RCP<DRT::Element> > geometry;
  std::map<int,Teuchos::RCP<DRT::Element> >::iterator iter;

  for (iter=geometry_.begin();iter!=geometry_.end();++iter)
  {
    iter->second->SetId(iter->first+shift);
    geometry[iter->first+shift]=geometry_[iter->first];
  }

  swap(geometry_, geometry);

  return;
}


