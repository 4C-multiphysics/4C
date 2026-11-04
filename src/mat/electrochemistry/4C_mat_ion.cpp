// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_ion.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Ion::Ion(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      valence_(matdata.parameters.get<double>("VALENCE")),
      diffusivity_(matdata.parameters.get<double>("DIFFUSIVITY")),
      densification_(matdata.parameters.get<double>("DENSIFICATION")),
      elimvalence_(matdata.parameters.get<double>("ELIM_VALENCE")),
      elimdiffusivity_(matdata.parameters.get<double>("ELIM_DIFFUSIVITY"))
{
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::Ion::create_material()
{
  return std::make_shared<Mat::Ion>(this);
}

Mat::IonType Mat::IonType::instance_;


Core::Communication::ParObject* Mat::IonType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Ion* ion = new Mat::Ion();
  ion->unpack(buffer);
  return ion;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Ion::Ion() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Ion::Ion(Mat::PAR::Ion* params) : params_(params) {}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Ion::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  /*
  for (unsigned i=0;i<data().size();i++)
  std::cout<<"Pack ION: pb["<<i<<"] = "<<(data())[i]<<std::endl;
*/
  /*
  // extract type
  std::vector<char>::size_type posit = 0;
  std::vector<char> pbtest;
  int typio = 0;
  extract_from_pack(posit,data(),typio);
  std::cout<<"ION Pack: Type will be "<<typio<<std::endl;
*/
  // std::cout<<"Ion Pack: "<<data().size()<<std::endl;

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Ion::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::Ion*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }



  return;
}

FOUR_C_NAMESPACE_CLOSE
