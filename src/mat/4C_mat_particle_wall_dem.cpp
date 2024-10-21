#include "4C_mat_particle_wall_dem.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | define static class member                                 sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEMType Mat::ParticleWallMaterialDEMType::instance_;

/*---------------------------------------------------------------------------*
 | constructor                                                sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
Mat::PAR::ParticleWallMaterialDEM::ParticleWallMaterialDEM(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      frictionTang_(matdata.parameters.get<double>("FRICT_COEFF_TANG")),
      frictionRoll_(matdata.parameters.get<double>("FRICT_COEFF_ROLL")),
      adhesionSurfaceEnergy_(matdata.parameters.get<double>("ADHESION_SURFACE_ENERGY"))
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | create material instance of matching type with parameters  sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
Teuchos::RCP<Core::Mat::Material> Mat::PAR::ParticleWallMaterialDEM::create_material()
{
  return Teuchos::make_rcp<Mat::ParticleWallMaterialDEM>(this);
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ParticleWallMaterialDEMType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ParticleWallMaterialDEM* particlewallmatdem = new Mat::ParticleWallMaterialDEM();
  particlewallmatdem->unpack(buffer);
  return particlewallmatdem;
}

/*---------------------------------------------------------------------------*
 | constructor (empty material object)                        sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEM::ParticleWallMaterialDEM() : params_(nullptr)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | constructor (with given material parameters)               sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEM::ParticleWallMaterialDEM(Mat::PAR::ParticleWallMaterialDEM* params)
    : params_(params)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | pack                                                       sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
void Mat::ParticleWallMaterialDEM::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*---------------------------------------------------------------------------*
 | unpack                                                     sfuchs 08/2019 |
 *---------------------------------------------------------------------------*/
void Mat::ParticleWallMaterialDEM::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != Teuchos::null)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::ParticleWallMaterialDEM*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
}

FOUR_C_NAMESPACE_CLOSE
