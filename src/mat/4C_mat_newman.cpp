/*----------------------------------------------------------------------------*/
/*! \file
\brief material stores parameters for ion species in electrolyte solution. The newman material is
derived for a binary electrolyte using the electroneutrality condition to condense the non-reacting
species

\level 2


*/
/*----------------------------------------------------------------------------*/

#include "4C_mat_newman.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_function_of_time.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// TODO: math.H was included automatically

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Newman::Newman(const Core::Mat::PAR::Parameter::Data& matdata)
    : ElchSingleMat(matdata),
      valence_(matdata.parameters.get<double>("VALENCE")),
      transnrcurve_(matdata.parameters.get<int>("TRANSNR")),
      thermfaccurve_(matdata.parameters.get<int>("THERMFAC")),
      transnrparanum_(matdata.parameters.get<int>("TRANS_PARA_NUM")),
      transnrpara_(matdata.parameters.get<std::vector<double>>("TRANS_PARA")),
      thermfacparanum_(matdata.parameters.get<int>("THERM_PARA_NUM")),
      thermfacpara_(matdata.parameters.get<std::vector<double>>("THERM_PARA"))
{
  if (transnrparanum_ != (int)transnrpara_.size())
    FOUR_C_THROW("number of materials %d does not fit to size of material vector %d",
        transnrparanum_, transnrpara_.size());
  if (thermfacparanum_ != (int)thermfacpara_.size())
    FOUR_C_THROW("number of materials %d does not fit to size of material vector %d",
        thermfacparanum_, thermfacpara_.size());

  // check if number of provided parameter is valid for a the chosen predefined function
  check_provided_params(transnrcurve_, transnrpara_);
  check_provided_params(thermfaccurve_, thermfacpara_);
}


Teuchos::RCP<Core::Mat::Material> Mat::PAR::Newman::create_material()
{
  return Teuchos::make_rcp<Mat::Newman>(this);
}

Mat::NewmanType Mat::NewmanType::instance_;


Core::Communication::ParObject* Mat::NewmanType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Newman* newman = new Mat::Newman();
  newman->unpack(buffer);
  return newman;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Newman::Newman() : params_(nullptr) { return; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Newman::Newman(Mat::PAR::Newman* params) : params_(params) { return; }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Newman::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Newman::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Newman*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_transference_number(const double cint) const
{
  double trans = 0.0;

  if (trans_nr_curve() < 0)
    trans = eval_pre_defined_funct(trans_nr_curve(), cint, trans_nr_params());
  else if (trans_nr_curve() == 0)
    trans = eval_pre_defined_funct(-1, cint, trans_nr_params());
  else
    trans = Global::Problem::instance()
                ->function_by_id<Core::UTILS::FunctionOfTime>(trans_nr_curve() - 1)
                .evaluate(cint);

  return trans;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_first_deriv_trans(const double cint) const
{
  double firstderiv = 0.0;

  if (trans_nr_curve() < 0)
    firstderiv = eval_first_deriv_pre_defined_funct(trans_nr_curve(), cint, trans_nr_params());
  else if (trans_nr_curve() == 0)
    firstderiv = eval_first_deriv_pre_defined_funct(-1, cint, trans_nr_params());
  else
    firstderiv = Global::Problem::instance()
                     ->function_by_id<Core::UTILS::FunctionOfTime>(trans_nr_curve() - 1)
                     .evaluate_derivative(cint);

  return firstderiv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_therm_fac(const double cint) const
{
  double therm = 0.0;

  if (therm_fac_curve() < 0)
    therm = eval_pre_defined_funct(therm_fac_curve(), cint, therm_fac_params());
  else if (therm_fac_curve() == 0)
    // thermodynamic factor has to be one if not defined
    therm = 1.0;
  else
    therm = Global::Problem::instance()
                ->function_by_id<Core::UTILS::FunctionOfTime>(therm_fac_curve() - 1)
                .evaluate(cint);

  return therm;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_first_deriv_therm_fac(const double cint) const
{
  double firstderiv = 0.0;

  if (therm_fac_curve() < 0)
    firstderiv = eval_first_deriv_pre_defined_funct(therm_fac_curve(), cint, therm_fac_params());
  else if (therm_fac_curve() == 0)
    // thermodynamic factor has to be one if not defined
    // -> first derivative = 0.0
    firstderiv = 0.0;
  else
    firstderiv = Global::Problem::instance()
                     ->function_by_id<Core::UTILS::FunctionOfTime>(therm_fac_curve() - 1)
                     .evaluate_derivative(cint);

  return firstderiv;
}

FOUR_C_NAMESPACE_CLOSE
