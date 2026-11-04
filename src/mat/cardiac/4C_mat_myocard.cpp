// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_myocard_fitzhugh_nagumo.hpp"
#include "4C_mat_myocard_inada.hpp"
#include "4C_mat_myocard_minimal.hpp"
#include "4C_mat_myocard_san_garny.hpp"
#include "4C_mat_myocard_tentusscher.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ENull.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |                                                          cbert 09/12 |
 *----------------------------------------------------------------------*/
Mat::PAR::Myocard::Myocard(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      diff1(matdata.parameters.get<double>("DIFF1")),
      diff2(matdata.parameters.get<double>("DIFF2")),
      diff3(0.0),
      dt_deriv(matdata.parameters.get<double>("PERTURBATION_DERIV")),
      model(matdata.parameters.get<std::string>("MODEL")),
      tissue(matdata.parameters.get<std::string>("TISSUE")),
      time_scale(matdata.parameters.get<double>("TIME_SCALE")),
      num_gp(0)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Myocard::create_material()
{
  return std::make_shared<Mat::Myocard>(this);
}


Mat::MyocardType Mat::MyocardType::instance_;


Core::Communication::ParObject* Mat::MyocardType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Myocard* myocard = new Mat::Myocard();
  myocard->unpack(buffer);
  return myocard;
}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
Mat::Myocard::Myocard()
    : params_(nullptr),
      difftensor_(0),
      nb_state_variables_(0),
      myocard_mat_(nullptr),
      diff_at_ele_center_(false)
{
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)   cbert 08/13 |
 *----------------------------------------------------------------------*/
Mat::Myocard::Myocard(Mat::PAR::Myocard* params)
    : params_(params),
      difftensor_(0),
      nb_state_variables_(0),
      myocard_mat_(nullptr),
      diff_at_ele_center_(false)
{
}


/*----------------------------------------------------------------------*
 |  Pack                                           (public)  cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  add_to_pack(data, nb_state_variables_);
  add_to_pack(data, difftensor_);
  int num;
  if (diff_at_ele_center_)
    num = 1;
  else
    num = 0;
  add_to_pack(data, num);
  if (myocard_mat_ != nullptr)
    add_to_pack(data, myocard_mat_->get_number_of_gp());
  else
    add_to_pack(data, 0);

  // pack history data
  if (myocard_mat_ != nullptr)
    for (int k = -1; k < nb_state_variables_; ++k)  // Starting from -1 for mechanical activation
      for (int i = 0; i < myocard_mat_->get_number_of_gp(); ++i)  // loop over Gauss points
        add_to_pack(data, myocard_mat_->get_internal_state(k, i));

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack                                         (public)  cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  int unpack_nb_state_variables;
  int num_gp;
  int num;
  extract_from_pack(buffer, matid);
  extract_from_pack(buffer, unpack_nb_state_variables);
  extract_from_pack(buffer, difftensor_);
  extract_from_pack(buffer, num);
  if (num)
    diff_at_ele_center_ = true;
  else
    diff_at_ele_center_ = false;
  extract_from_pack(buffer, num_gp);

  params_ = nullptr;
  if (Global::Problem::instance()->materials() !=
      nullptr)  // it does not enter here in postprocessing
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = static_cast<Mat::PAR::Myocard*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());

      // Set number of Gauss points
      set_gp(num_gp);

      if (num_gp > 0)
      {
        // Initialize material
        initialize();

        // unpack history data
        double val;
        for (int k = -1; k < unpack_nb_state_variables;
            ++k)                                     // Starting from -1 for mechanical activation
          for (int i = 0; i < params_->num_gp; ++i)  // loop over Gauss points
          {
            extract_from_pack(buffer, val);
            myocard_mat_->set_internal_state(k, val, i);
          }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  unpack_material                                       hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::unpack_material(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  int num_gp;
  int num;
  extract_from_pack(buffer, matid);
  extract_from_pack(buffer, nb_state_variables_);
  extract_from_pack(buffer, difftensor_);
  extract_from_pack(buffer, num);
  if (num)
    diff_at_ele_center_ = true;
  else
    diff_at_ele_center_ = false;
  extract_from_pack(buffer, num_gp);

  params_ = nullptr;
  if (Global::Problem::instance()->materials() !=
      nullptr)  // it does not enter here in postprocessing
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = static_cast<Mat::PAR::Myocard*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());

      // Set number of Gauss points
      set_gp(myocard_mat_->get_number_of_gp());

      // unpack history data
      double val;
      for (int k = -1; k < nb_state_variables_; ++k)  // Starting from -1 for mechanical activation
        for (int i = 0; i < params_->num_gp; ++i)     // loop over Gauss points
        {
          extract_from_pack(buffer, val);
          myocard_mat_->set_internal_state(k, val, i);
        }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Setup conductivity tensor                                cbert 02/13 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::setup(const Core::LinAlg::Tensor<double, 3>& fiber1)
{
  setup_diffusion_tensor(fiber1);
}

void Mat::Myocard::setup(const Core::LinAlg::Tensor<double, 2>& fiber1)
{
  setup_diffusion_tensor(fiber1);
}

void Mat::Myocard::setup(const Core::IO::InputParameterContainer& container)
{
  if (const auto& fiber1 = container.get<std::optional<std::vector<double>>>("FIBER1");
      fiber1.has_value())
  {
    diff_at_ele_center_ = true;
    setup_diffusion_tensor(*fiber1);
  }
}


void Mat::Myocard::setup_diffusion_tensor(const std::vector<double>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1[0] * fiber1[0] + fiber1[1] * fiber1[1] + fiber1[2] * fiber1[2];

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1[0] * fiber1[0] / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1[0] * fiber1[1] / fiber1normS;
  difftensor(0, 2) = (diff1 - diff2) * fiber1[0] * fiber1[2] / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1[1] * fiber1[0] / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1[1] * fiber1[1] / fiber1normS;
  difftensor(1, 2) = (diff1 - diff2) * fiber1[1] * fiber1[2] / fiber1normS;
  // third row
  difftensor(2, 0) = (diff1 - diff2) * fiber1[2] * fiber1[0] / fiber1normS;
  difftensor(2, 1) = (diff1 - diff2) * fiber1[2] * fiber1[1] / fiber1normS;
  difftensor(2, 2) = diff2 + (diff1 - diff2) * fiber1[2] * fiber1[2] / fiber1normS;
  // done

  difftensor_.push_back(difftensor);
  return;
}

void Mat::Myocard::setup_diffusion_tensor(const Core::LinAlg::Tensor<double, 3>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1(0) * fiber1(0) + fiber1(1) * fiber1(1) + fiber1(2) * fiber1(2);

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1(0) * fiber1(0) / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1(0) * fiber1(1) / fiber1normS;
  difftensor(0, 2) = (diff1 - diff2) * fiber1(0) * fiber1(2) / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1(1) * fiber1(0) / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1(1) * fiber1(1) / fiber1normS;
  difftensor(1, 2) = (diff1 - diff2) * fiber1(1) * fiber1(2) / fiber1normS;
  // third row
  difftensor(2, 0) = (diff1 - diff2) * fiber1(2) * fiber1(0) / fiber1normS;
  difftensor(2, 1) = (diff1 - diff2) * fiber1(2) * fiber1(1) / fiber1normS;
  difftensor(2, 2) = diff2 + (diff1 - diff2) * fiber1(2) * fiber1(2) / fiber1normS;
  // done

  difftensor_.push_back(difftensor);

  return;
}

void Mat::Myocard::setup_diffusion_tensor(const Core::LinAlg::Tensor<double, 2>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1(0) * fiber1(0) + fiber1(1) * fiber1(1);

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1(0) * fiber1(0) / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1(0) * fiber1(1) / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1(1) * fiber1(0) / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1(1) * fiber1(1) / fiber1normS;
  // done

  difftensor_.push_back(difftensor);

  return;
}

void Mat::Myocard::diffusivity(Core::LinAlg::Matrix<1, 1>& diffus3, int gp) const
{
  diffus3(0, 0) = difftensor_[gp](0, 0);
  return;
}

void Mat::Myocard::diffusivity(Core::LinAlg::Matrix<2, 2>& diffus3, int gp) const
{
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      diffus3(i, j) = difftensor_[gp](i, j);
    }
  }

  return;
}

void Mat::Myocard::diffusivity(Core::LinAlg::Matrix<3, 3>& diffus3, int gp) const
{
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      diffus3(i, j) = difftensor_[gp](i, j);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |                                                           cbert 09/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::rea_coeff(const double phi, const double dt) const
{
  double reacoeff = params_->time_scale;
  reacoeff *= myocard_mat_->rea_coeff(phi, dt * params_->time_scale);

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::rea_coeff(const double phi, const double dt, int gp) const
{
  double reacoeff = params_->time_scale;
  if (gp == -1)
    reacoeff *= myocard_mat_->rea_coeff(phi, dt * params_->time_scale);
  else
    reacoeff *= myocard_mat_->rea_coeff(phi, dt * params_->time_scale, gp);

  return reacoeff;
}

/*----------------------------------------------------------------------*
 | reaction coefficient at time n                        hoermann 11/16 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::rea_coeff_n(const double phi, const double dt, int gp) const
{
  double reacoeff = params_->time_scale;
  if (gp == -1)
    reacoeff *= myocard_mat_->rea_coeff_n(phi, dt * params_->time_scale);
  else
    reacoeff *= myocard_mat_->rea_coeff_n(phi, dt * params_->time_scale, gp);

  return reacoeff;
}


/*----------------------------------------------------------------------*
 |                                                           cbert 09/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::rea_coeff_deriv(const double phi, const double dt) const
{
  double ReaCoeffDeriv = 0.0;
  if (params_->dt_deriv != 0.0)
  {
    double ReaCoeff_t2 = rea_coeff((phi + params_->dt_deriv), dt);
    double ReaCoeff_t1 = rea_coeff(phi, dt);
    ReaCoeffDeriv = (ReaCoeff_t2 - ReaCoeff_t1) / (params_->dt_deriv);
  }
  return ReaCoeffDeriv;
}


/*----------------------------------------------------------------------*
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::rea_coeff_deriv(const double phi, const double dt, int gp) const
{
  double ReaCoeffDeriv = 0.0;
  if (params_->dt_deriv != 0.0)
  {
    double ReaCoeff_t2 = rea_coeff((phi + params_->dt_deriv), dt, gp);
    double ReaCoeff_t1 = rea_coeff(phi, dt, gp);
    ReaCoeffDeriv = (ReaCoeff_t2 - ReaCoeff_t1) / (params_->dt_deriv);
  }
  return ReaCoeffDeriv;
}


/*----------------------------------------------------------------------*
 |  returns number of internal state variables              cbert 08/13 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::get_number_of_internal_state_variables() const
{
  int val = 0;
  val = myocard_mat_->get_number_of_internal_state_variables();
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::get_internal_state(const int k) const
{
  double val = 0.0;
  val = myocard_mat_->get_internal_state(k);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double Mat::Myocard::get_internal_state(const int k, int gp) const
{
  double val = 0.0;
  val = myocard_mat_->get_internal_state(k, gp);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::set_internal_state(const int k, const double val)
{
  myocard_mat_->set_internal_state(k, val);
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
void Mat::Myocard::set_internal_state(const int k, const double val, int gp)
{
  myocard_mat_->set_internal_state(k, val, gp);
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::get_number_of_ionic_currents() const
{
  int val = 0;
  val = myocard_mat_->get_number_of_ionic_currents();
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal currents                    hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double Mat::Myocard::get_ionic_currents(const int k, int gp) const
{
  double val = 0.0;
  val = myocard_mat_->get_ionic_currents(k, gp);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal currents                       cbert 08/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::get_ionic_currents(const int k) const
{
  double val = 0.0;
  val = myocard_mat_->get_ionic_currents(k);
  return val;
}


/*----------------------------------------------------------------------*
 |  initialize internal variables (called by constructors)   cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::initialize()
{
  if ((params_->model) == "MV")
    myocard_mat_ =
        std::make_shared<MyocardMinimal>(params_->dt_deriv, (params_->tissue), params_->num_gp);
  else if ((params_->model) == "FHN")
    myocard_mat_ = std::make_shared<MyocardFitzhughNagumo>(
        params_->dt_deriv, (params_->tissue), params_->num_gp);
  else if ((params_->model) == "INADA")
    myocard_mat_ = std::make_shared<MyocardInada>(params_->dt_deriv, (params_->tissue));
  else if ((params_->model) == "TNNP")
    myocard_mat_ = std::make_shared<MyocardTenTusscher>(params_->dt_deriv, (params_->tissue));
  else if ((params_->model) == "SAN")
    myocard_mat_ = std::make_shared<MyocardSanGarny>(params_->dt_deriv, (params_->tissue));
  else
    FOUR_C_THROW(
        "Myocard Material type is not supported! (for the moment only MV,FHN,INADA,TNNP and SAN)");

  nb_state_variables_ = myocard_mat_->get_number_of_internal_state_variables();

  return;
}

/*----------------------------------------------------------------------*
 |  resize internal state variables                      hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::resize_internal_state_variables()
{
  myocard_mat_->resize_internal_state_variables(params_->num_gp);
  return;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::update(const double phi, const double dt)
{
  myocard_mat_->update(phi, dt * (params_->time_scale));

  return;
}

/*----------------------------------------------------------------------*
 |  get number of Gauss points                           hoermann 12/16 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::get_number_of_gp() const { return myocard_mat_->get_number_of_gp(); };

FOUR_C_NAMESPACE_CLOSE
