/*!-----------------------------------------------------------------------*
 \file fluidporo.cpp

 <pre>
   Maintainer: Anh-Tu Vuong
               vuong@lnm.mw.tum.de
               http://www.lnm.mw.tum.de
               089 - 289-15264
 </pre>
 *-----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <vector>
#include "fluidporo.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::FluidPoro::FluidPoro(Teuchos::RCP<MAT::PAR::Material> matdata) :
  Parameter(matdata), viscosity_(matdata->GetDouble("VISCOSITY")), density_(
      matdata->GetDouble("DENSITY")), permeability_(matdata->GetDouble(
      "PERMEABILITY"))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<MAT::Material> MAT::PAR::FluidPoro::CreateMaterial()
{
  return Teuchos::rcp(new MAT::FluidPoro(this));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::FluidPoroType MAT::FluidPoroType::instance_;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

DRT::ParObject* MAT::FluidPoroType::Create(const std::vector<char> & data)
{
  MAT::FluidPoro* fluid_poro = new MAT::FluidPoro();
  fluid_poro->Unpack(data);
  return fluid_poro;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::FluidPoro::FluidPoro() :
  params_(NULL)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::FluidPoro::FluidPoro(MAT::PAR::FluidPoro* params) :
  params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::FluidPoro::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != NULL)
    matid = params_->Id(); // in case we are in post-process mode
  AddtoPack(data, matid);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::FluidPoro::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId())
    dserror("wrong instance type data");

    // matid
    int matid;
    ExtractfromPack(position,data,matid);
    params_ = NULL;
    if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
      params_ = static_cast<MAT::PAR::FluidPoro*>(mat);
      else
      dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }

    if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}

    /*----------------------------------------------------------------------*/
    /*----------------------------------------------------------------------*/
double MAT::FluidPoro::ComputeReactionCoeff() const
{
  // check for zero or negative viscosity
  if (Viscosity() < EPS15)
    dserror("zero or negative viscosity");

  // check for zero or negative permeability
  if (Permeability() < EPS15) dserror("zero or negative permeability");

  // viscosity divided by permeability
  double reacoeff = Viscosity()/Permeability();

  return reacoeff;
}

    /*----------------------------------------------------------------------*/
    /*----------------------------------------------------------------------*/
double MAT::FluidPoro::SetViscosity() const
{
  // set zero viscosity and only modify it for Darcy-Stokes problems
  double viscosity = 0.0;
  // if (Type() == "Darcy-Stokes") viscosity = Viscosity();

  return viscosity;
}
#endif
