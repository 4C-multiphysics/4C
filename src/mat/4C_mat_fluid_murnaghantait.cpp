/*----------------------------------------------------------------------*/
/*! \file
\brief Weakly compressible fluid according to Murnaghan-Tait

\level 1

*/
/*----------------------------------------------------------------------*/

#include "4C_mat_fluid_murnaghantait.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::MurnaghanTaitFluid::MurnaghanTaitFluid(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscosity_(matdata.parameters.get<double>("DYNVISCOSITY")),
      refdensity_(matdata.parameters.get<double>("REFDENSITY")),
      refpressure_(matdata.parameters.get<double>("REFPRESSURE")),
      refbulkmodulus_(matdata.parameters.get<double>("REFBULKMODULUS")),
      matparameter_(matdata.parameters.get<double>("MATPARAMETER")),
      gamma_(matdata.parameters.get<double>("GAMMA"))
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::Mat::Material> Mat::PAR::MurnaghanTaitFluid::create_material()
{
  return Teuchos::make_rcp<Mat::MurnaghanTaitFluid>(this);
}


Mat::MurnaghanTaitFluidType Mat::MurnaghanTaitFluidType::instance_;


Core::Communication::ParObject* Mat::MurnaghanTaitFluidType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MurnaghanTaitFluid* fluid = new Mat::MurnaghanTaitFluid();
  fluid->unpack(buffer);
  return fluid;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::MurnaghanTaitFluid::MurnaghanTaitFluid() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::MurnaghanTaitFluid::MurnaghanTaitFluid(Mat::PAR::MurnaghanTaitFluid* params) : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MurnaghanTaitFluid::pack(Core::Communication::PackBuffer& data) const
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


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MurnaghanTaitFluid::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::MurnaghanTaitFluid*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::MurnaghanTaitFluid::compute_density(const double press) const
{
  // $ \rho=\rho_0[\dfrac{n}{K_0}\left(P-P_0)+1\right]^{\dfrac{1}{n}} $
  const double density =
      ref_density() *
      std::pow((mat_parameter() / ref_bulk_modulus() * (press - ref_pressure()) + 1.0),
          (1.0 / mat_parameter()));

  return density;
}

FOUR_C_NAMESPACE_CLOSE
