// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_membrane_active_strain.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_membrane_elasthyper.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                     brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
Mat::PAR::MembraneActiveStrain::MembraneActiveStrain(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      matid_passive_(matdata.parameters.get<int>("MATIDPASSIVE")),
      scalid_voltage_(matdata.parameters.get<int>("SCALIDVOLTAGE")),
      density_(matdata.parameters.get<double>("DENS")),
      beta1_(matdata.parameters.get<double>("BETA1")),
      beta2_(matdata.parameters.get<double>("BETA2")),
      voltage_threshold_(matdata.parameters.get<double>("VOLTHRESH")),
      alpha1_(matdata.parameters.get<double>("ALPHA1")),
      alpha2_(matdata.parameters.get<double>("ALPHA2"))
{
  return;
}  // Mat::PAR::MembraneActiveStrain::MembraneActiveStrain

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::MembraneActiveStrain::create_material()
{
  return std::make_shared<Mat::MembraneActiveStrain>(this);
}  // Mat::PAR::MembraneActiveStrain::create_material

Mat::MembraneActiveStrainType Mat::MembraneActiveStrainType::instance_;

Core::Communication::ParObject* Mat::MembraneActiveStrainType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MembraneActiveStrain* membrane_activestrain = new Mat::MembraneActiveStrain();
  membrane_activestrain->unpack(buffer);

  return membrane_activestrain;
}  // Mat::Membrane_ActiveStrainType::Create

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
Mat::MembraneActiveStrain::MembraneActiveStrain()
    : params_(nullptr),
      matpassive_(nullptr),
      voltage_(nullptr),
      activation_(nullptr),
      isinit_(false),
      fibervecs_(false)
{
  return;
}  // Mat::MembraneActiveStrain::MembraneActiveStrain()

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
Mat::MembraneActiveStrain::MembraneActiveStrain(Mat::PAR::MembraneActiveStrain* params)
    : params_(params),
      matpassive_(nullptr),
      voltage_(nullptr),
      activation_(nullptr),
      isinit_(false),
      fibervecs_(false)
{
  return;
}  // Mat::MembraneActiveStrain::MembraneActiveStrain()

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // fiber vectors: Fiber1, Fiber2, Normal
  add_to_pack(data, fibervecs_);

  // data of passive elastic material
  if (matpassive_ != nullptr)
  {
    add_to_pack(data, true);
    matpassive_->pack(data);
  }
  else
  {
    add_to_pack(data, false);
  }

  // pack internal variables
  int numgp;
  // if material is not initialized, i.e. start simulation, nothing to pack
  if (!isinit_)
  {
    numgp = 0;
  }
  else
  {
    // if material is initialized, i.e. restart of simulation, size equates number of gausspoints
    numgp = voltage_->size();
  }
  // Length of internal vector(s)
  add_to_pack(data, numgp);
  for (int gp = 0; gp < numgp; ++gp)
  {
    // insert internal vectors to add_to_pack
    add_to_pack(data, voltage_->at(gp));
    add_to_pack(data, activation_->at(gp));
  }

  return;
}  // Mat::MembraneActiveStrain::pack()

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid = -1;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::MembraneActiveStrain*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // fiber vectors: Fiber1, Fiber2, Normal
  extract_from_pack(buffer, fibervecs_);

  // unpack data of passive material
  bool matpassive_exists;
  extract_from_pack(buffer, matpassive_exists);
  if (matpassive_exists)
  {
    Core::Communication::ParObject* o = Core::Communication::factory(buffer);
    Mat::So3Material* matpassive = dynamic_cast<Mat::So3Material*>(o);
    if (matpassive == nullptr) FOUR_C_THROW("failed to unpack passive material");

    matpassive_ = std::shared_ptr<So3Material>(matpassive);
  }
  else
  {
    matpassive_ = nullptr;
  }

  int numgp;
  extract_from_pack(buffer, numgp);
  isinit_ = true;
  if (numgp == 0)  // no internal data to unpack
  {
    isinit_ = false;

    return;
  }

  // unpack internal variables
  voltage_ = std::make_shared<std::vector<double>>(numgp);
  activation_ = std::make_shared<std::vector<double>>(numgp);
  double voltage_gp;
  double activation_gp;
  for (int gp = 0; gp < numgp; ++gp)
  {
    extract_from_pack(buffer, voltage_gp);
    voltage_->at(gp) = voltage_gp;
    extract_from_pack(buffer, activation_gp);
    activation_->at(gp) = activation_gp;
  }
}  // Mat::MembraneActiveStrain::unpack()

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::setup(int numgp, const Core::IO::InputParameterContainer& container)
{
  // setup fibervectors
  setup_fiber_vectors(numgp, container);

  // setup of passive material
  matpassive_ = std::dynamic_pointer_cast<Mat::So3Material>(Mat::factory(params_->matid_passive_));
  matpassive_->setup(numgp, container);

  // setup internal variables
  voltage_ = std::make_shared<std::vector<double>>();
  voltage_->resize(numgp);

  activation_ = std::make_shared<std::vector<double>>();
  activation_->resize(numgp);

  for (int gp = 0; gp < numgp; ++gp)
  {
    voltage_->at(gp) = 0.0;
    activation_->at(gp) = 0.0;
  }

  isinit_ = true;
}  // Mat::MembraneActiveStrain::setup()

/*----------------------------------------------------------------------*
 | active strain and hyperelastic stress response plus elasticity tensor|
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::evaluate_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
    const Core::LinAlg::Matrix<3, 3>& cauchygreen, Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<3, 3>& Q_trafo, Core::LinAlg::Matrix<3, 1>& stress,
    Core::LinAlg::Matrix<3, 3>& cmat, const int gp, const int eleGID)
{
  // blank resulting quantities
  stress.clear();
  cmat.clear();

  // get pointer to vector containing the scalar states at the gauss points
  std::shared_ptr<std::vector<std::vector<double>>> gpscalar =
      params.get<std::shared_ptr<std::vector<std::vector<double>>>>("gp_scalar",
          std::make_shared<std::vector<std::vector<double>>>(4, std::vector<double>(4, 0.0)));

  const unsigned int scalarid_voltage = params_->scalid_voltage_;

  if (scalarid_voltage >= gpscalar->at(0).size())
    FOUR_C_THROW("Mismatch in requested scalar and number of supplied scalars.");

  // voltage at current gp
  double gpvoltage = gpscalar->at(gp).at(scalarid_voltage);

  // save voltage for visualization
  voltage_->at(gp) = gpvoltage;

  // structural tensor in local coordinates
  std::vector<Core::LinAlg::Matrix<3, 3>> structural_tensors_loc;

  // loop over all fiber vectors
  Core::LinAlg::Matrix<3, 1> fibervector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> structuraltensor(Core::LinAlg::Initialization::zero);
  for (unsigned int p = 0; p < 3; ++p)
  {
    fibervector.multiply_tn(1.0, Q_trafo, fibervecs_[p], 0.0);
    structuraltensor.multiply_nt(1.0, fibervector, fibervector, 0.0);
    structural_tensors_loc.push_back(structuraltensor);
  }

  //******************
  // ACTIVE deformation gradient in local coordinates
  //******************
  Core::LinAlg::Matrix<3, 3> defgrd_active_inv_loc(Core::LinAlg::Initialization::zero);

  // set defgrd_active to identity tensor
  for (int i = 0; i < 3; i++) defgrd_active_inv_loc(i, i) = 1.0;

  // create full active def-grd
  double voltage_threshold = params_->voltage_threshold_;
  double beta1 = params_->beta1_;
  double beta2 = params_->beta2_;

  double gamma = 0;
  if (gpvoltage > voltage_threshold)
  {
    gamma = (1 - std::exp(-beta1 * (gpvoltage - voltage_threshold))) *
            (1 - std::exp(-beta2 * (gpvoltage - voltage_threshold)));
  }

  activation_->at(gp) = gamma;

  double gamma1 = gamma * params_->alpha1_;
  double gamma2 = gamma * params_->alpha2_;
  double gammaNormal =
      (1 - (1 - gamma1) * (1 - gamma2)) /
      ((1 - gamma1) * (1 - gamma2));  // compute gamma_n such that active material is incompressible

  defgrd_active_inv_loc.update(gamma1 / (1.0 - gamma1), structural_tensors_loc.at(0), 1.0);
  defgrd_active_inv_loc.update(gamma2 / (1.0 - gamma2), structural_tensors_loc.at(1), 1.0);
  defgrd_active_inv_loc.update(
      -gammaNormal / (1.0 + gammaNormal), structural_tensors_loc.at(2), 1.0);

  //******************
  // PASSIVE cauchy green in local coordinates
  //******************
  Core::LinAlg::Matrix<3, 3> cauchygreen_passive_local(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> defgrd_passive_local(Core::LinAlg::Initialization::zero);
  defgrd_passive_local.multiply_nn(1.0, defgrd, defgrd_active_inv_loc, 0.0);
  cauchygreen_passive_local.multiply_tn(1.0, defgrd_passive_local, defgrd_passive_local, 0.0);

  // compute passive green lagrange strain
  Core::LinAlg::Matrix<3, 3> cmatpassive_loc(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> S_passive_loc_voigt(Core::LinAlg::Initialization::zero);
  std::dynamic_pointer_cast<Mat::MembraneElastHyper>(matpassive_)
      ->evaluate_membrane(defgrd_passive_local, cauchygreen_passive_local, params, Q_trafo,
          S_passive_loc_voigt, cmatpassive_loc, gp, eleGID);

  //******************
  // FULL PART
  //******************
  Core::LinAlg::Matrix<2, 2> S_tot(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2, 2> S_passive_loc(Core::LinAlg::Initialization::zero);
  S_passive_loc(0, 0) = S_passive_loc_voigt(0);
  S_passive_loc(1, 1) = S_passive_loc_voigt(1);
  S_passive_loc(1, 0) = S_passive_loc_voigt(2);
  S_passive_loc(0, 1) = S_passive_loc_voigt(2);

  Core::LinAlg::Matrix<2, 2> defgrd_active_inv_loc_red(Core::LinAlg::Initialization::zero);
  defgrd_active_inv_loc_red(0, 0) = defgrd_active_inv_loc(0, 0);
  defgrd_active_inv_loc_red(1, 0) = defgrd_active_inv_loc(1, 0);
  defgrd_active_inv_loc_red(0, 1) = defgrd_active_inv_loc(0, 1);
  defgrd_active_inv_loc_red(1, 1) = defgrd_active_inv_loc(1, 1);

  Core::LinAlg::Matrix<2, 2> temp2(Core::LinAlg::Initialization::zero);
  temp2.multiply_nt(1.0, S_passive_loc, defgrd_active_inv_loc_red, 0.0);
  S_tot.multiply_nn(1.0, defgrd_active_inv_loc_red, temp2, 0.0);

  stress(0) = S_tot(0, 0);
  stress(1) = S_tot(1, 1);
  stress(2) = 0.5 * (S_tot(1, 0) + S_tot(0, 1));

  // pullback of the linearization
  pullback4th_tensor_voigt(defgrd_active_inv_loc_red, cmatpassive_loc, cmat);

  return;
}  // Mat::MembraneActiveStrain::Evaluate

/*----------------------------------------------------------------------*
 | Update internal variables                       brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::update()
{
  matpassive_->update();
}  // Mat::MembraneActiveStrain::Update

/*----------------------------------------------------------------------*
 | Reset internal variables                        brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::reset_step()
{
  matpassive_->reset_step();
}  // Mat::MembraneActiveStrain::reset_step

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::vis_names(std::map<std::string, int>& names) const
{
  matpassive_->vis_names(names);
  names["voltage"] = 1;     // scalar
  names["activation"] = 1;  // scalar
}  // Mat::MembraneActiveStrain::vis_names

/*----------------------------------------------------------------------*
 |                                                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
bool Mat::MembraneActiveStrain::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "voltage")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");

    for (int gp = 0; gp < numgp; gp++) data[0] += voltage_->at(gp);

    data[0] = data[0] / numgp;
    return true;
  }
  else if (name == "activation")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");

    for (int gp = 0; gp < numgp; gp++) data[0] += activation_->at(gp);

    data[0] = data[0] / numgp;
    return true;
  }

  return matpassive_->vis_data(name, data, numgp, eleID);
}  // Mat::MembraneActiveStrain::vis_data

/*----------------------------------------------------------------------*
 | setup fiber vectors                                                  |
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::setup_fiber_vectors(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  Core::LinAlg::Matrix<3, 1> dir;

  // CIR-AXI-RAD nomenclature
  if (container.get<std::optional<std::vector<double>>>("RAD").has_value() and
      container.get<std::optional<std::vector<double>>>("AXI").has_value() and
      container.get<std::optional<std::vector<double>>>("CIR").has_value())
  {
    // Axial direction
    read_dir(container, "AXI", dir);
    fibervecs_.push_back(dir);

    // Circumferential direction
    read_dir(container, "CIR", dir);
    fibervecs_.push_back(dir);

    // Radial direction
    read_dir(container, "RAD", dir);
    fibervecs_.push_back(dir);
  }
  // FIBER nomenclature
  else if (container.get<std::optional<std::vector<double>>>("FIBER1").has_value() and
           container.get<std::optional<std::vector<double>>>("FIBER2").has_value())
  {
    for (int i = 1; i < 3; ++i)
    {
      std::ostringstream ss;
      ss << i;
      std::string fibername = "FIBER" + ss.str();  // FIBER Name
                                                   // FiberN direction
      read_dir(container, fibername, dir);
      fibervecs_.push_back(dir);
    }

    setup_normal_direction();
  }
  else
  {
    FOUR_C_THROW("Either use Fiber or CIR-AXI-RAD nomenclature to set fiber directions");
  }

  // Check orthonormal basis
  if (fibervecs_.size() != 3)
    FOUR_C_THROW(
        "Wrong number of fiber vectors. This material need three, it is {}", fibervecs_.size());

  double eps = 1e-12;
  if (std::abs(fibervecs_[0].dot(fibervecs_[1])) > eps or
      std::abs(fibervecs_[1].dot(fibervecs_[2])) > eps or
      std::abs(fibervecs_[0].dot(fibervecs_[2])) > eps)
  {
    std::cout << std::endl;
    std::cout << "\tWARNING: fiber vectors do NOT build orthonormal basis!" << std::endl;
    std::cout << std::endl;
    FOUR_C_THROW(
        "Fiber vectors are not orthonormal: while this is not necessary in general, for now we "
        "limit ourselves to the orthonomal case!\n"
        "In particular the calculation of the inverse active deformation gradient depends on this "
        "assumption!");
  }

}  // Mat::MembraneActiveStrain::setup_fiber_vectors

/*----------------------------------------------------------------------*
 * Function which reads in the fiber direction
 *----------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::read_dir(const Core::IO::InputParameterContainer& container,
    std::string specifier, Core::LinAlg::Matrix<3, 1>& dir)
{
  const auto& fiber_opt = container.get<std::optional<std::vector<double>>>(specifier);
  FOUR_C_ASSERT(fiber_opt.has_value(), "Internal error: fiber vector not found.");
  const auto& fiber = *fiber_opt;

  double fnorm = 0.;
  // normalization
  for (int i = 0; i < 3; ++i)
  {
    fnorm += fiber[i] * fiber[i];
  }
  fnorm = sqrt(fnorm);

  // fill final normalized vector
  for (int i = 0; i < 3; ++i) dir(i) = fiber[i] / fnorm;

  return;
}  // Mat::MembraneActiveStrain::read_dir

void Mat::MembraneActiveStrain::setup_normal_direction()
{
  if (fibervecs_.size() != 2)
  {
    FOUR_C_THROW("Wrong number of fiber vectors to calculate a normal direction.");
  }

  Core::LinAlg::Matrix<3, 1> dir1 = fibervecs_[0];
  Core::LinAlg::Matrix<3, 1> dir2 = fibervecs_[1];
  Core::LinAlg::Matrix<3, 1> normaldir;

  normaldir(0) = dir1(1) * dir2(2) - dir1(2) * dir2(1);
  normaldir(1) = dir1(2) * dir2(0) - dir1(0) * dir2(2);
  normaldir(2) = dir1(0) * dir2(1) - dir1(1) * dir2(0);

  // normalization
  double norm = normaldir.norm2();
  normaldir.scale(1 / norm);

  fibervecs_.push_back(normaldir);
}  // Mat::MembraneActiveStrain::setup_normal_direction

/*---------------------------------------------------------------------*
 | Pullback of the tangent from intermediate to reference configuration|
 *---------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::pullback4th_tensor_voigt(
    const Core::LinAlg::Matrix<2, 2>& defgrd_active_inv_red,
    const Core::LinAlg::Matrix<3, 3>& cmat_passive_intermediate,
    Core::LinAlg::Matrix<3, 3>& cmat_reference)
{
  int i;
  int j;
  int k;
  int l;
  for (int p = 0; p < 3; ++p)
  {
    for (int q = 0; q < 3; ++q)
    {
      int M;
      int N;
      tensor2x2_indices(p, &i, &j);
      tensor2x2_indices(q, &k, &l);

      for (int A = 0; A < 2; ++A)
      {
        for (int B = 0; B < 2; ++B)
        {
          for (int C = 0; C < 2; ++C)
          {
            for (int D = 0; D < 2; ++D)
            {
              voigt3_index(A, B, &M);
              voigt3_index(C, D, &N);

              cmat_reference(p, q) += defgrd_active_inv_red(i, A) * defgrd_active_inv_red(j, B) *
                                      defgrd_active_inv_red(k, C) * defgrd_active_inv_red(l, D) *
                                      cmat_passive_intermediate(M, N);
            }
          }
        }
      }
    }
  }

}  // Mat::MembraneActiveStrain::pullback4th_tensor_voigt

/*---------------------------------------------------------------------*
 | transform voigt to tensor notation                                  |
 *---------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::tensor2x2_indices(int p, int* i, int* j)
{
  switch (p)
  {
    case 0:
      *i = 0;
      *j = 0;
      break;
    case 1:
      *i = 1;
      *j = 1;
      break;
    case 2:
      *i = 0;
      *j = 1;
      break;
  }
}  // Mat::MembraneActiveStrain::voigt3_index

/*---------------------------------------------------------------------*
 | transform tensor to voigt notation (public)                         |
 *---------------------------------------------------------------------*/
void Mat::MembraneActiveStrain::voigt3_index(int i, int j, int* p)
{
  if (i == 0 && j == 0)
    *p = 0;
  else if (i == 1 && j == 1)
    *p = 1;
  else if ((i == 0 && j == 1) || (i == 1 && j == 0))
    *p = 2;
}  // Mat::MembraneActiveStrain::voigt3_index

FOUR_C_NAMESPACE_CLOSE
