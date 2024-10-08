/*----------------------------------------------------------------------*/
/*! \file
\brief
This file contains a framework, which allows for the linear interpolation between two different
strain energy functions, where the interpolation ratio come from 'outside'. For now this is supposed
to come from the idea, that there is a scalar quantity (e.g. foam cells) which induces growth. The
grown volume fraction thereby consists of another material (e.g. softer or stiffer material
parameters or totally different strain energy function).

\level 2

The input line should read
MAT 1 MAT_ScalarDepInterp IDMATZEROSC 2 IDMATUNITSC 3



*/

/*----------------------------------------------------------------------*/

#include "4C_mat_scalardepinterp.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScalarDepInterp::ScalarDepInterp(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      id_lambda_zero_(matdata.parameters.get<int>("IDMATZEROSC")),
      id_lambda_unit_(matdata.parameters.get<int>("IDMATUNITSC")){

      };

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::Mat::Material> Mat::PAR::ScalarDepInterp::create_material()
{
  return Teuchos::RCP(new Mat::ScalarDepInterp(this));
}


Mat::ScalarDepInterpType Mat::ScalarDepInterpType::instance_;


Core::Communication::ParObject* Mat::ScalarDepInterpType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScalarDepInterp* ScalarDepInterp = new Mat::ScalarDepInterp();
  ScalarDepInterp->unpack(buffer);
  return ScalarDepInterp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScalarDepInterp::ScalarDepInterp()
    : params_(nullptr),
      isinit_(false),
      lambda_zero_mat_(Teuchos::null),
      lambda_unit_mat_(Teuchos::null),
      lambda_(Teuchos::null)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScalarDepInterp::ScalarDepInterp(Mat::PAR::ScalarDepInterp* params)
    : params_(params),
      isinit_(false),
      lambda_zero_mat_(Teuchos::null),
      lambda_unit_mat_(Teuchos::null),
      lambda_(Teuchos::null)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::setup(int numgp, const Core::IO::InputParameterContainer& container)
{
  if (isinit_)
    FOUR_C_THROW("This function should just be called, if the material is jet not initialized.");

  // Setup of elastic material for zero concentration
  lambda_zero_mat_ =
      Teuchos::rcp_dynamic_cast<Mat::So3Material>(Mat::factory(params_->id_lambda_zero_));
  lambda_zero_mat_->setup(numgp, container);

  // Setup of elastic material for zero concentration
  lambda_unit_mat_ =
      Teuchos::rcp_dynamic_cast<Mat::So3Material>(Mat::factory(params_->id_lambda_unit_));
  lambda_unit_mat_->setup(numgp, container);

  // Some safety check
  const double density1 = lambda_zero_mat_->density();
  const double density2 = lambda_unit_mat_->density();
  if (abs(density1 - density2) > 1e-14)
    FOUR_C_THROW(
        "The densities of the materials specified in IDMATZEROSC and IDMATUNITSC must be equal!");

  // Read lambda from input file, if available
  auto lambda = container.get_or<double>("lambda", 1.0);

  lambda_ = std::vector<double>(numgp, lambda);

  // initialization done
  isinit_ = true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  // evaluate elastic material corresponding to zero concentration
  Core::LinAlg::Matrix<6, 1> stress_lambda_zero = *stress;
  Core::LinAlg::Matrix<6, 6> cmat_zero_conc = *cmat;

  lambda_zero_mat_->evaluate(
      defgrd, glstrain, params, &stress_lambda_zero, &cmat_zero_conc, gp, eleGID);

  // evaluate elastic material corresponding to infinite concentration
  Core::LinAlg::Matrix<6, 1> stress_lambda_unit = *stress;
  Core::LinAlg::Matrix<6, 6> cmat_infty_conc = *cmat;

  lambda_unit_mat_->evaluate(
      defgrd, glstrain, params, &stress_lambda_unit, &cmat_infty_conc, gp, eleGID);

  double lambda;
  // get the ratio of interpolation
  // NOTE: if no ratio is available, we use the IDMATZEROSC material!
  if (params.isParameter("lambda"))
  {
    lambda = params.get<double>("lambda");

    // NOTE: this would be nice, but since negative concentrations can occur,
    // we have to catch 'unnatural' cases different...
    //  if ( lambda < -1.0e-14 or lambda > (1.0+1.0e-14) )
    //      FOUR_C_THROW("The lambda must be in [0,1]!");

    // e.g. like that:
    if (lambda < -1.0e-14) lambda = 0.0;
    if (lambda > (1.0 + 1.0e-14)) lambda = 1.0;

    lambda_.at(gp) = lambda;
  }
  else
  {
    lambda = lambda_.at(gp);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // \mym Psi = (1-\lambda(C)) * \Psi_0(\mym C) + \lambda(C) * \Psi_1(\mym C) +  = ...
  // \mym S = 2 \frac{\partial}{\partial \mym C} \left( \Psi(\mym C) \right) = ...
  // \mym S = 2 \frac{\partial}{\partial \mym C} \left( (1-\lambda(C)) * \Psi_0(\mym C) + \lambda(C)
  // * \Psi_1(\mym C) \right) = ...
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // do the linear interpolation between stresses:
  // ... = (1-\lambda(C)) * 2* \frac{\partial}{\partial \mym C} \Psi_0) + \lambda(C) * 2 *
  // \frac{\partial}{\partial \mym C} \Psi_1
  stress->update(1.0 - lambda, stress_lambda_zero, lambda, stress_lambda_unit, 0.0);

  cmat->update(1.0 - lambda, cmat_zero_conc, lambda, cmat_infty_conc, 0.0);

  if (params.isParameter("dlambda_dC"))
  {
    // get derivative of interpolation ratio w.r.t. glstrain
    Teuchos::RCP<Core::LinAlg::Matrix<6, 1>> dlambda_dC =
        params.get<Teuchos::RCP<Core::LinAlg::Matrix<6, 1>>>("dlambda_dC");

    // evaluate strain energy functions
    double psi_lambda_zero = 0.0;
    lambda_zero_mat_->strain_energy(*glstrain, psi_lambda_zero, gp, eleGID);

    double psi_lambda_unit = 0.0;
    lambda_unit_mat_->strain_energy(*glstrain, psi_lambda_unit, gp, eleGID);

    // and add the stresses due to possible dependency of the ratio w.r.t. to C
    // ... - 2 * \Psi_0 * \frac{\partial}{\partial \mym C} \lambda(C) ) + * 2 * \Psi_1 *
    // \frac{\partial}{\partial \mym C} \lambda(C)
    stress->update(-2.0 * psi_lambda_zero, *dlambda_dC, +2.0 * psi_lambda_unit, *dlambda_dC, 1.0);

    // Note: for the linearization we do neglect the derivatives of the ratio w.r.t. glstrain
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  int numgp = 0;
  if (isinit_)
  {
    numgp = lambda_.size();
    ;  // size is number of gausspoints
  }
  add_to_pack(data, numgp);

  for (int gp = 0; gp < numgp; gp++)
  {
    add_to_pack(data, lambda_.at(gp));
  }

  // Pack data of both elastic materials
  if (lambda_zero_mat_ != Teuchos::null and lambda_unit_mat_ != Teuchos::null)
  {
    lambda_zero_mat_->pack(data);
    lambda_unit_mat_->pack(data);
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


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
        params_ = dynamic_cast<Mat::PAR::ScalarDepInterp*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }

  int numgp;
  extract_from_pack(buffer, numgp);
  if (not(numgp == 0))
  {
    lambda_ = std::vector<double>(numgp, 1.0);

    for (int gp = 0; gp < numgp; gp++)
    {
      double lambda = 1.0;
      extract_from_pack(buffer, lambda);
      lambda_.at(gp) = lambda;
    }
  }

  // Unpack data of elastic material (these lines are copied from element.cpp)
  std::vector<char> dataelastic;
  extract_from_pack(buffer, dataelastic);
  if (dataelastic.size() > 0)
  {
    Core::Communication::UnpackBuffer buffer_dataelastic(dataelastic);
    Core::Communication::ParObject* o =
        Core::Communication::factory(buffer_dataelastic);  // Unpack is done here
    Mat::So3Material* matel = dynamic_cast<Mat::So3Material*>(o);
    if (matel == nullptr) FOUR_C_THROW("failed to unpack elastic material");
    lambda_zero_mat_ = Teuchos::RCP(matel);
  }
  else
    lambda_zero_mat_ = Teuchos::null;

  // Unpack data of elastic material (these lines are copied from element.cpp)
  std::vector<char> dataelastic2;
  extract_from_pack(buffer, dataelastic2);
  if (dataelastic2.size() > 0)
  {
    Core::Communication::UnpackBuffer buffer_dataelastic(dataelastic2);
    Core::Communication::ParObject* o =
        Core::Communication::factory(buffer_dataelastic);  // Unpack is done here
    Mat::So3Material* matel = dynamic_cast<Mat::So3Material*>(o);
    if (matel == nullptr) FOUR_C_THROW("failed to unpack elastic material");
    lambda_unit_mat_ = Teuchos::RCP(matel);
  }
  else
    lambda_unit_mat_ = Teuchos::null;

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");

  return;
}

/*----------------------------------------------------------------------------*/
void Mat::ScalarDepInterp::update()
{
  lambda_zero_mat_->update();
  lambda_unit_mat_->update();
}

/*----------------------------------------------------------------------------*/
void Mat::ScalarDepInterp::reset_step()
{
  lambda_zero_mat_->reset_step();
  lambda_unit_mat_->reset_step();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::vis_names(std::map<std::string, int>& names) const
{
  std::string fiber = "lambda";
  names[fiber] = 1;  // 1-dim vector

  lambda_zero_mat_->vis_names(names);
  lambda_unit_mat_->vis_names(names);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Mat::ScalarDepInterp::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "lambda")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");

    double temp = 0.0;
    for (int gp = 0; gp < numgp; gp++)
    {
      temp += lambda_.at(gp);
    }

    data[0] = temp / ((double)numgp);
  }

  bool tmp1 = lambda_zero_mat_->vis_data(name, data, numgp, eleID);
  bool tmp2 = lambda_unit_mat_->vis_data(name, data, numgp, eleID);
  return (tmp1 and tmp2);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScalarDepInterp::strain_energy(
    const Core::LinAlg::Matrix<6, 1>& glstrain, double& psi, const int gp, const int eleGID) const
{
  // evaluate strain energy functions
  double psi_lambda_zero = 0.0;
  lambda_zero_mat_->strain_energy(glstrain, psi_lambda_zero, gp, eleGID);

  double psi_lambda_unit = 0.0;
  lambda_unit_mat_->strain_energy(glstrain, psi_lambda_unit, gp, eleGID);

  double lambda = 0.0;
  for (unsigned gp = 0; gp < lambda_.size(); gp++)
  {
    lambda += lambda_.at(gp);
  }
  lambda = lambda / (lambda_.size());

  psi += (1 - lambda) * psi_lambda_zero + lambda * psi_lambda_unit;
}

FOUR_C_NAMESPACE_CLOSE
