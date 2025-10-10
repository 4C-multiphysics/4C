// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_crystal_plasticity.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ENull.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                      			|
 *----------------------------------------------------------------------*/
Mat::PAR::CrystalPlasticity::CrystalPlasticity(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      tol_(matdata.parameters.get<double>("TOL")),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      poisson_(matdata.parameters.get<double>("NUE")),
      density_(matdata.parameters.get<double>("DENS")),
      lattice_(matdata.parameters.get<std::string>("LAT")),
      ctoa_(matdata.parameters.get<double>("CTOA")),
      lattice_const_(matdata.parameters.get<double>("ABASE")),
      num_slip_sys_(matdata.parameters.get<int>("NUMSLIPSYS")),
      num_slip_sets_(matdata.parameters.get<int>("NUMSLIPSETS")),
      slip_set_mem_(matdata.parameters.get<std::vector<int>>("SLIPSETMEMBERS")),
      slip_rate_exp_(matdata.parameters.get<std::vector<int>>("SLIPRATEEXP")),
      slip_ref_shear_rate_(matdata.parameters.get<std::vector<double>>("GAMMADOTSLIPREF")),
      dis_dens_init_(matdata.parameters.get<std::vector<double>>("DISDENSINIT")),
      dis_gen_coeff_(matdata.parameters.get<std::vector<double>>("DISGENCOEFF")),
      dis_dyn_rec_coeff_(matdata.parameters.get<std::vector<double>>("DISDYNRECCOEFF")),
      slip_lat_resist_(matdata.parameters.get<std::vector<double>>("TAUY0")),
      slip_micro_bound_(matdata.parameters.get<std::vector<double>>("MFPSLIP")),
      slip_hp_coeff_(matdata.parameters.get<std::vector<double>>("SLIPHPCOEFF")),
      slip_by_twin_(matdata.parameters.get<std::vector<double>>("SLIPBYTWIN")),
      num_twin_sys_(matdata.parameters.get<int>("NUMTWINSYS")),
      num_twin_sets_(matdata.parameters.get<int>("NUMTWINSETS")),
      twin_set_mem_(matdata.parameters.get<std::vector<int>>("TWINSETMEMBERS")),
      twin_rate_exp_(matdata.parameters.get<std::vector<int>>("TWINRATEEXP")),
      twin_ref_shear_rate_(matdata.parameters.get<std::vector<double>>("GAMMADOTTWINREF")),
      twin_lat_resist_(matdata.parameters.get<std::vector<double>>("TAUT0")),
      twin_micro_bound_(matdata.parameters.get<std::vector<double>>("MFPTWIN")),
      twin_hp_coeff_(matdata.parameters.get<std::vector<double>>("TWINHPCOEFF")),
      twin_by_slip_(matdata.parameters.get<std::vector<double>>("TWINBYSLIP")),
      twin_by_twin_(matdata.parameters.get<std::vector<double>>("TWINBYTWIN"))
{
  // check validity of input parameters
  if (tol_ <= 0.)
    FOUR_C_THROW("Newton tolerance TOL is negative or zero!!! Check your input!");
  else if (youngs_ <= 0.)
    FOUR_C_THROW("Young's modulus YOUNG is negative or zero!!! Check your input!");
  else if (poisson_ >= 0.5 || poisson_ < -1.)
    FOUR_C_THROW("Poisson's ratio needs be between NUE=-1 and NUE=0.5 Check your input!");
  else if (density_ <= 0.)
    FOUR_C_THROW("Density is negative or zero!!! Check your input!");
  else if (!(lattice_ == "L10" || lattice_ == "D019" || lattice_ == "FCC" || lattice_ == "BCC" ||
               lattice_ == "TEST"))
    FOUR_C_THROW(
        "Lattice type not known. Please check the LAT parameter in the input file. Currently it "
        "has to be FCC, BCC, HCP, D019 or L10");
  else if (ctoa_ <= 0.)
    FOUR_C_THROW("C to a ratio CTOA is negative or zero!!! Check your input!");
  else if (lattice_const_ <= 0.)
    FOUR_C_THROW("Lattice constant ABASE is negative or zero!!! Check your input!");
  else if (num_slip_sys_ <= 0.)
    FOUR_C_THROW("Number of slip systems NUMSLIPSYS is negative or zero!!! Check your input!");
  else if (num_slip_sets_ > num_slip_sys_)
    FOUR_C_THROW(
        "Number of slip system sets NUMSLIPSETS is greater than number of slip systems "
        "NUMSLIPSYS!!! Check your input!");
  else if ((num_twin_sys_ != 0) && (num_twin_sets_ > num_twin_sys_))
    FOUR_C_THROW(
        "The number of twinning system sets NUMTWINSETS is greater than the number of twinning "
        "systems NUMTWINSYS!!! Check your input!");

  // check for slip/twinning set index out of range
  for (int i = 0; i < num_slip_sys_; i++)
  {
    if (slip_set_mem_[i] > num_slip_sets_)
      FOUR_C_THROW(
          "The index of one slip system in SLIPSETMEMBERS exceeds the number of slip sets given in "
          "NUMSLIPSETS!!! Check your input!");
    else if (slip_set_mem_[i] < 1)
      FOUR_C_THROW(
          "The index of one slip system in SLIPSETMEMBERS is negative or zero!!! Check your "
          "input!");
  }
  if (num_twin_sys_ != 0)
  {
    for (int i = 0; i < num_twin_sys_; i++)
    {
      if (twin_set_mem_[i] > num_twin_sets_)
        FOUR_C_THROW(
            "The index of one twinning system in TWINSETMEMBERS exceeds the number of twinning "
            "sets given in "
            "NUMTWINSETS!!! Check your input!");
      else if (twin_set_mem_[i] < 1)
        FOUR_C_THROW(
            "The index of one twinning system in TWINSETMEMBERS is negative or zero!!! Check your "
            "input!");
    }
  }
  // check the model parameters of the different slip/twinning sets for correctness
  for (int i = 0; i < num_slip_sets_; i++)
  {
    if (slip_rate_exp_[i] <= 0)
      FOUR_C_THROW("One of the entries in SLIPRATEEXP is negative or zero!!! Check your input!");
    else if (slip_ref_shear_rate_[i] <= 0)
      FOUR_C_THROW(
          "One of the entries in GAMMADOTSLIPREF is negative or zero!!! Check your input!");
    else if (dis_dens_init_[i] < 0)
      FOUR_C_THROW("One of the entries in DISDENSINIT is negative!!! Check your input!");
    else if (dis_gen_coeff_[i] < 0)
      FOUR_C_THROW("One of the entries in DISGENCOEFF is negative!!! Check your input!");
    else if (dis_dyn_rec_coeff_[i] < 0)
      FOUR_C_THROW("One of the entries in DISDYNECCOEFF is negative!!! Check your input!");
    else if (slip_lat_resist_[i] < 0)
      FOUR_C_THROW("One of the entries in TAUY0 is negative!!! Check your input!");
    else if (slip_micro_bound_[i] <= 0)
      FOUR_C_THROW("One of the entries in MFPSLIP is negative or zero!!! Check your input!");
    else if (slip_hp_coeff_[i] < 0)
      FOUR_C_THROW("One of the entries in SLIPHPCOEFF is negative!!! Check your input!");
    else if (slip_by_twin_[i] < 0)
      FOUR_C_THROW("One of the entries in SLIPBYTWIN is negative!!! Check your input!");
  }
  if (num_twin_sys_ != 0)
  {
    for (int i = 0; i < num_twin_sets_; i++)
    {
      if (twin_rate_exp_[i] <= 0)
        FOUR_C_THROW("One of the entries in TWINRATEEXP is negative or zero!!! Check your input!");
      else if (twin_ref_shear_rate_[i] <= 0)
        FOUR_C_THROW(
            "One of the entries in GAMMADOTTWINREF is negative or zero!!! Check your input!");
      else if (twin_lat_resist_[i] < 0)
        FOUR_C_THROW("One of the entries in TAUT0 is negative!!! Check your input!");
      else if (twin_micro_bound_[i] <= 0)
        FOUR_C_THROW("One of the entries in MFPTWIN is negative or zero!!! Check your input!");
      else if (twin_hp_coeff_[i] < 0)
        FOUR_C_THROW("One of the entries in TWINHPCOEFF is negative!!! Check your input!");
      else if (twin_by_slip_[i] < 0)
        FOUR_C_THROW("One of the entries in TWINBYSLIP is negative!!! Check your input!");
      else if (twin_by_twin_[i] < 0)
        FOUR_C_THROW("One of the entries in TWINBYTWIN is negative!!! Check your input!");
    }
  }
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       			|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::CrystalPlasticity::create_material()
{
  return std::make_shared<Mat::CrystalPlasticity>(this);
}

/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       			|
 *----------------------------------------------------------------------*/

Mat::CrystalPlasticityType Mat::CrystalPlasticityType::instance_;

/*----------------------------------------------------------------------*
 | is called in Material::Factory from read_materials()       			|
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::CrystalPlasticityType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::CrystalPlasticity* cp = new Mat::CrystalPlasticity();
  cp->unpack(buffer);
  return cp;
}

/*----------------------------------------------------------------------*
 | constructor (public)                                                 |
 *----------------------------------------------------------------------*/
Mat::CrystalPlasticity::CrystalPlasticity() : params_(nullptr) {}

/*----------------------------------------------------------------------*
 | copy-constructor (public)                                            |
 *----------------------------------------------------------------------*/
Mat::CrystalPlasticity::CrystalPlasticity(Mat::PAR::CrystalPlasticity* params) : params_(params) {}

/*----------------------------------------------------------------------*
 | pack (public)                                                        |
 *----------------------------------------------------------------------*/
void Mat::CrystalPlasticity::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // pack matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack history data
  int histsize;
  // if material is not yet initialised, i.e. at start of simulation, nothing to pack
  if (!(isinit_ and (deform_grad_current_ != nullptr)))
  {
    histsize = 0;
  }
  else
  {
    // if material is initialized already (restart): size equates number of Gauss points
    histsize = deform_grad_last_->size();
  }
  add_to_pack(data, histsize);  // length of history vector(s)

  for (int var = 0; var < histsize; ++var)
  {
    // insert history vectors to add_to_pack
    add_to_pack(data, (*deform_grad_last_)[var]);
    add_to_pack(data, (*plastic_deform_grad_last_)[var]);
    add_to_pack(data, (*gamma_last_)[var]);
    add_to_pack(data, (*defect_densities_last_)[var]);
  }
}  // pack()

/*----------------------------------------------------------------------*
 | unpack (public)                                                      |
 *----------------------------------------------------------------------*/
void Mat::CrystalPlasticity::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // recover matid and params_
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::CrystalPlasticity*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  // history data
  int histsize;
  extract_from_pack(buffer, histsize);

  // if system is not yet initialised, the history vectors have to be initialized
  if (histsize == 0) isinit_ = false;

  if (params_ != nullptr)
  {
    deform_grad_last_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
    plastic_deform_grad_last_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
    gamma_last_ = std::make_shared<std::vector<std::vector<double>>>();
    defect_densities_last_ = std::make_shared<std::vector<std::vector<double>>>();

    for (int var = 0; var < histsize; ++var)
    {
      Core::LinAlg::Matrix<3, 3> tmp_matrix(Core::LinAlg::Initialization::zero);
      std::vector<double> tmp_vect(def_system_count_);

      extract_from_pack(buffer, tmp_matrix);
      deform_grad_last_->push_back(tmp_matrix);

      extract_from_pack(buffer, tmp_matrix);
      plastic_deform_grad_last_->push_back(tmp_matrix);

      extract_from_pack(buffer, tmp_vect);
      gamma_last_->push_back(tmp_vect);

      extract_from_pack(buffer, tmp_vect);
      defect_densities_last_->push_back(tmp_vect);
    }

    // in the postprocessing mode, we do not unpack everything we have packed
    // -> position check cannot be done in this case
  }
}  // Unpack

/*---------------------------------------------------------------------*
 | initialize / allocate internal variables (public)                   |
 *---------------------------------------------------------------------*/
void Mat::CrystalPlasticity::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // import material / model parameters and calculate derived values

  // General Properties:
  //-------------------
  // tolerance for local Newton Raphson iteration
  newton_tolerance_ = params_->tol_;

  // Elastic Properties:
  //-------------------------
  // Young's Modulus
  youngs_mod_ = params_->youngs_;
  // Poisson's ratio
  poisson_ratio_ = params_->poisson_;
  // 1st Lame constant
  lambda_ =
      (poisson_ratio_ * youngs_mod_) / ((1.0 + poisson_ratio_) * (1.0 - 2.0 * poisson_ratio_));
  // Shear modulus / 2nd Lame constant
  mue_ = youngs_mod_ / (2.0 * (1.0 + poisson_ratio_));
  // Bulk modulus
  bulk_mod_ = youngs_mod_ / (3.0 - (6.0 * poisson_ratio_));

  // Crystal Properties:
  //-------------------
  // setup lattice vectors
  // set up slip plane normals and directions and initialize crystal lattice related model
  // parameters according to chosen lattice type
  this->setup_lattice_vectors();

  //  read Lattice orientation matrix from input file
  this->setup_lattice_orientation(fibers);

  // rotate lattice vectors according to lattice orientation
  for (int i = 0; i < slip_system_count_; i++)
  {
    Core::LinAlg::Matrix<3, 1> unrotated_slip_normal = slip_plane_normal_[i];
    Core::LinAlg::Matrix<3, 1> unrotated_slip_direction = slip_direction_[i];

    slip_plane_normal_[i].multiply_nn(lattice_orientation_, unrotated_slip_normal);
    slip_direction_[i].multiply_nn(lattice_orientation_, unrotated_slip_direction);
  }
  if (is_twinning_)
  {
    for (int i = 0; i < twin_system_count_; i++)
    {
      Core::LinAlg::Matrix<3, 1> unrotated_twin_normal = twin_plane_normal_[i];
      Core::LinAlg::Matrix<3, 1> unrotated_twin_direction = twin_direction_[i];

      twin_plane_normal_[i].multiply_nn(lattice_orientation_, unrotated_twin_normal);
      twin_direction_[i].multiply_nn(lattice_orientation_, unrotated_twin_direction);
    }
  }

  // TODO ADD TEST WHETHER INPUT ROTATION MATRIX IS ORTHOGONAL Q*Q^T=I TO AVOID UNWANTED SCALING OF
  // LATTICE VECTORS DUE TO BAD USER INPUT!

  // Index to which subset a slip system belongs
  slip_set_index_ = params_->slip_set_mem_;

  // Index to which subset a twinning system belongs
  if (is_twinning_) twin_set_index_ = params_->twin_set_mem_;

  // Viscoplastic Properties:
  //------------------------
  // reference shear rates
  gamma_dot_0_slip_ = params_->slip_ref_shear_rate_;
  if (is_twinning_) gamma_dot_0_twin_ = params_->twin_ref_shear_rate_;

  // strain rate sensitivity exponents
  n_slip_ = params_->slip_rate_exp_;
  if (is_twinning_) n_twin_ = params_->twin_rate_exp_;

  // Dislocation Generation/Recovery:
  //--------------------------------
  // initial dislocation densities
  initial_dislocation_density_ = params_->dis_dens_init_;

  // dislocation generation coefficients
  dislocation_generation_coeff_ = params_->dis_gen_coeff_;
  // dynamic dislocation removal coefficients
  dislocation_dyn_recovery_coeff_ = params_->dis_dyn_rec_coeff_;

  // Initial Slip System Strengths:
  //------------------------------
  // lattice resistances to slip and twinning
  tau_y_0_ = params_->slip_lat_resist_;
  if (is_twinning_) tau_t_0_ = params_->twin_lat_resist_;

  // microstructural parameters which are relevant for Hall-Petch strengthening, e.g., grain size
  micro_boundary_distance_slip_ = params_->slip_micro_bound_;
  if (is_twinning_) micro_boundary_distance_twin_ = params_->twin_micro_bound_;

  // Hall-Petch coefficients corresponding to above microstructural boundaries
  hall_petch_coeffs_slip_ = params_->slip_hp_coeff_;
  if (is_twinning_) hall_petch_coeffs_twin_ = params_->twin_hp_coeff_;

  // Work Hardening Interactions:
  //------------------------------
  if (is_twinning_)
  {
    // vector of hardening coefficients of slip systems due to non-coplanar twinning
    slip_by_twin_hard_ = params_->slip_by_twin_;
    // vector of hardening coefficients of twinning systems due to slip
    twin_by_slip_hard_ = params_->twin_by_slip_;
    // vector of hardening coefficients of twinning systems due to non-coplanar twinning
    twin_by_twin_hard_ = params_->twin_by_twin_;
  }

  // set up 3x3 identity matrix
  const Core::LinAlg::Matrix<3, 3> identity3 = Core::LinAlg::identity_matrix<3>();

  // initialize history variables
  deform_grad_last_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  deform_grad_current_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();

  plastic_deform_grad_last_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  plastic_deform_grad_current_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();

  gamma_last_ = std::make_shared<std::vector<std::vector<double>>>();
  gamma_current_ = std::make_shared<std::vector<std::vector<double>>>();

  defect_densities_last_ = std::make_shared<std::vector<std::vector<double>>>();
  defect_densities_current_ = std::make_shared<std::vector<std::vector<double>>>();

  // set size to number of  Gauss points so each Gauss Point has its own set of history data
  deform_grad_last_->resize(numgp);
  deform_grad_current_->resize(numgp);

  plastic_deform_grad_last_->resize(numgp);
  plastic_deform_grad_current_->resize(numgp);

  gamma_last_->resize(numgp);
  gamma_current_->resize(numgp);

  defect_densities_last_->resize(numgp);
  defect_densities_current_->resize(numgp);

  // set initial values
  std::vector<double> emptyvect(def_system_count_);
  for (int i = 0; i < def_system_count_; i++) emptyvect[i] = 0.0;

  for (int i = 0; i < numgp; i++)
  {
    (*deform_grad_last_)[i] = identity3;
    (*deform_grad_current_)[i] = identity3;

    (*plastic_deform_grad_last_)[i] = identity3;
    (*plastic_deform_grad_current_)[i] = identity3;

    (*gamma_last_)[i].resize(def_system_count_);
    (*gamma_current_)[i].resize(def_system_count_);

    (*gamma_last_)[i] = emptyvect;
    (*gamma_current_)[i] = emptyvect;

    (*defect_densities_last_)[i].resize(def_system_count_);
    (*defect_densities_current_)[i].resize(def_system_count_);

    // initial defect densities. first dislocation densities than twinned volume
    // fractions
    for (int j = 0; j < slip_system_count_; j++)
    {
      // index of subset of the respective deformation system
      int ind = slip_set_index_[j] - 1;
      // initial dislocation densities
      (*defect_densities_last_)[i][j] = initial_dislocation_density_[ind];
    }
    if (is_twinning_)
    {
      for (int j = slip_system_count_; j < def_system_count_; j++)
      {
        // initial twinned volume fractions (currently assumed to be 0 at the beginning of
        // deformation)
        (*defect_densities_last_)[i][j] = 0.0;
      }
    }

    (*defect_densities_current_)[i] = emptyvect;
  }

  isinit_ = true;
  return;
}  // setup()


/*----------------------------------------------------------------------*
 | update internal variables (public)                                   |
 *----------------------------------------------------------------------*/
void Mat::CrystalPlasticity::update()
{
  // update values of last step t_n to current values at time step t_n+1
  deform_grad_last_ = deform_grad_current_;
  plastic_deform_grad_last_ = plastic_deform_grad_current_;

  gamma_last_ = gamma_current_;
  defect_densities_last_ = defect_densities_current_;

  // empty vectors of current data
  deform_grad_current_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  plastic_deform_grad_current_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 3>>>();
  gamma_current_ = std::make_shared<std::vector<std::vector<double>>>();
  defect_densities_current_ = std::make_shared<std::vector<std::vector<double>>>();

  // get the size of the vector (use the last vector, because it includes latest results,
  // current is already empty)
  const int histsize = deform_grad_last_->size();
  deform_grad_current_->resize(histsize);
  plastic_deform_grad_current_->resize(histsize);
  gamma_current_->resize(histsize);
  defect_densities_current_->resize(histsize);

  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);
  std::vector<double> emptyvect(def_system_count_);
  for (int i = 0; i < def_system_count_; i++) emptyvect[i] = 0.0;

  for (int i = 0; i < histsize; i++)
  {
    (*deform_grad_current_)[i] = emptymat;
    (*plastic_deform_grad_current_)[i] = emptymat;
    (*gamma_current_)[i] = emptyvect;
    (*defect_densities_current_)[i] = emptyvect;
  }
  return;
}  // update()

/*-------------------------------------------------------------------------------*
 | calculate stress, material stiffness matrix and evolution internal variables  |
 *-------------------------------------------------------------------------------*/
void Mat::CrystalPlasticity::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // simulation parameters
  //---------------------------------------------------------------
  if (eleGID == -1) FOUR_C_THROW("no element provided in material");

  // extract time increment
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  dt_ = *context.time_step_size;

  // get current deformation gradient
  (*deform_grad_current_)[gp] = Core::LinAlg::make_matrix(*defgrad);

  // local Newton-Raphson to determine plastic shears
  //--------------------------------------------------------------------------

  // define result output from NR
  Core::LinAlg::Matrix<3, 3> second_pk_stress_result;

  // call Newton-Raphson method with current deformation gradient
  this->newton_raphson((*deform_grad_current_)[gp], (*gamma_current_)[gp],
      (*defect_densities_current_)[gp], second_pk_stress_result,
      (*plastic_deform_grad_current_)[gp]);

  // update stress results
  stress = Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor_view(second_pk_stress_result));
}  // evaluate()

/*---------------------------------------------------------------------------------*
 | Setup slip directions and slip plane normals                                    |
 *---------------------------------------------------------------------------------*/
void Mat::CrystalPlasticity::setup_lattice_vectors()
{
  // extract lattice type from user input
  lattice_type_ = params_->lattice_;

  // extract number of slip and twinning systems from user input
  slip_system_count_ = params_->num_slip_sys_;

  // check user input whether or not twinning is to be considered
  if (params_->num_twin_sys_ != 0.)
  {
    twin_system_count_ = params_->num_twin_sys_;
    is_twinning_ = true;
  }
  else
  {
    twin_system_count_ = 0;
    is_twinning_ = false;
  }

  // set expected total number of slip and twinning systems for the lattice that was chosen by
  // the user
  if ((lattice_type_ == "L10" || lattice_type_ == "D019") && !is_twinning_)
    def_system_count_ = 12;  // 12 slip systems no twinning
  else if (lattice_type_ == "L10" && is_twinning_)
    def_system_count_ = 16;  // 12 slip systems and 4 twinning systems
  else if (lattice_type_ == "TEST" && !is_twinning_)
    def_system_count_ = 1;  // 1 slip system and no twinning
  else if (lattice_type_ == "TEST" && is_twinning_)
    def_system_count_ = 2;  // 1 slip systems and 1 twinning system
  else if (lattice_type_ == "D019" && is_twinning_)
    FOUR_C_THROW(
        "Twinning does not occur in the chosen lattice type {} or is not yet implemented. Omit "
        "all optional parameters of the material input line to switch off twinning.",
        lattice_type_.c_str());

  // check whether the number of slip/twinning systems given by the user coincides with the
  // lattice type
  if ((slip_system_count_ + twin_system_count_) != def_system_count_)
  {
    if (!is_twinning_)
      FOUR_C_THROW(
          "The number of slip systems NUMSLIPSYS = {} given in the input does not match the "
          "expected total number of {} deformation systems that is expected for {} lattices in "
          "the "
          "absence of twinning!",
          params_->num_slip_sys_, params_->num_twin_sys_, def_system_count_, lattice_type_.c_str());

    else if (is_twinning_)
      FOUR_C_THROW(
          "The sum of the number of slip systems NUMSLIPSYS = {} and the number twinning "
          "systems "
          "NUMTWINSYS = {} given in the input does not match the expected total number of {} "
          "deformation systems that is expected for {} lattices!",
          params_->num_slip_sys_, params_->num_twin_sys_, def_system_count_, lattice_type_.c_str());
  }

  // resize lattice related vectors to number of deformation systems
  slip_plane_normal_.resize(slip_system_count_);
  slip_direction_.resize(slip_system_count_);
  if (is_twinning_)
  {
    twin_plane_normal_.resize(twin_system_count_);
    twin_direction_.resize(twin_system_count_);
  }

  def_system_id_.resize(def_system_count_);

  // import c to a ratio of unit cell as provided by the user
  c_to_a_ratio_ = params_->ctoa_;

  // set up corresponding set of slip /twinning planes depending on the chosen crystallographic
  // lattice L10 super lattice
  if (lattice_type_ == "L10")
  {
    // ordinary slip systems
    def_system_id_[0] = "ordinary_1";

    slip_plane_normal_[0](0, 0) = 1.0;
    slip_plane_normal_[0](1, 0) = 1.0;
    slip_plane_normal_[0](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[0](0, 0) = 1.0 / 2.0;
    slip_direction_[0](1, 0) = -1.0 / 2.0;
    slip_direction_[0](2, 0) = 0.0 * c_to_a_ratio_;

    def_system_id_[1] = "ordinary_2";

    slip_plane_normal_[1](0, 0) = -1.0;
    slip_plane_normal_[1](1, 0) = -1.0;
    slip_plane_normal_[1](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[1](0, 0) = 1.0 / 2.0;
    slip_direction_[1](1, 0) = -1.0 / 2.0;
    slip_direction_[1](2, 0) = 0.0 * c_to_a_ratio_;

    def_system_id_[2] = "ordinary_3";

    slip_plane_normal_[2](0, 0) = 1.0;
    slip_plane_normal_[2](1, 0) = -1.0;
    slip_plane_normal_[2](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[2](0, 0) = 1.0 / 2.0;
    slip_direction_[2](1, 0) = 1.0 / 2.0;
    slip_direction_[2](2, 0) = 0.0 * c_to_a_ratio_;

    def_system_id_[3] = "ordinary_4";

    slip_plane_normal_[3](0, 0) = -1.0;
    slip_plane_normal_[3](1, 0) = 1.0;
    slip_plane_normal_[3](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[3](0, 0) = 1.0 / 2.0;
    slip_direction_[3](1, 0) = 1.0 / 2.0;
    slip_direction_[3](2, 0) = 0.0 * c_to_a_ratio_;

    // super slip systems
    def_system_id_[4] = "super_1";

    slip_plane_normal_[4](0, 0) = 1.0;
    slip_plane_normal_[4](1, 0) = 1.0;
    slip_plane_normal_[4](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[4](0, 0) = 0.0;
    slip_direction_[4](1, 0) = 1.0;
    slip_direction_[4](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[5] = "super_2";

    slip_plane_normal_[5](0, 0) = 1.0;
    slip_plane_normal_[5](1, 0) = 1.0;
    slip_plane_normal_[5](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[5](0, 0) = 1.0;
    slip_direction_[5](1, 0) = 0.0;
    slip_direction_[5](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[6] = "super_3";

    slip_plane_normal_[6](0, 0) = 1.0;
    slip_plane_normal_[6](1, 0) = -1.0;
    slip_plane_normal_[6](2, 0) = -1.0 / c_to_a_ratio_;

    slip_direction_[6](0, 0) = 0.0;
    slip_direction_[6](1, 0) = 1.0;
    slip_direction_[6](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[7] = "super_4";

    slip_plane_normal_[7](0, 0) = 1.0;
    slip_plane_normal_[7](1, 0) = -1.0;
    slip_plane_normal_[7](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[7](0, 0) = 1.0;
    slip_direction_[7](1, 0) = 0.0;
    slip_direction_[7](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[8] = "super_5";

    slip_plane_normal_[8](0, 0) = 1.0;
    slip_plane_normal_[8](1, 0) = 1.0;
    slip_plane_normal_[8](2, 0) = -1.0 / c_to_a_ratio_;

    slip_direction_[8](0, 0) = 0.0;
    slip_direction_[8](1, 0) = -1.0;
    slip_direction_[8](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[9] = "super_6";

    slip_plane_normal_[9](0, 0) = 1.0;
    slip_plane_normal_[9](1, 0) = 1.0;
    slip_plane_normal_[9](2, 0) = -1.0 / c_to_a_ratio_;

    slip_direction_[9](0, 0) = -1.0;
    slip_direction_[9](1, 0) = 0.0;
    slip_direction_[9](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[10] = "super_7";

    slip_plane_normal_[10](0, 0) = 1.0;
    slip_plane_normal_[10](1, 0) = -1.0;
    slip_plane_normal_[10](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[10](0, 0) = 0.0;
    slip_direction_[10](1, 0) = -1.0;
    slip_direction_[10](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[11] = "super_8";

    slip_plane_normal_[11](0, 0) = 1.0;
    slip_plane_normal_[11](1, 0) = -1.0;
    slip_plane_normal_[11](2, 0) = -1.0 / c_to_a_ratio_;

    slip_direction_[11](0, 0) = -1.0;
    slip_direction_[11](1, 0) = 0.0;
    slip_direction_[11](2, 0) = -1.0 * c_to_a_ratio_;

    if (is_twinning_)
    {
      def_system_id_[12] = "twin_1";

      twin_plane_normal_[0](0, 0) = 1.0;
      twin_plane_normal_[0](1, 0) = 1.0;
      twin_plane_normal_[0](2, 0) = 1.0 / c_to_a_ratio_;

      twin_direction_[0](0, 0) = 1.0 / 6.0;
      twin_direction_[0](1, 0) = 1.0 / 6.0;
      twin_direction_[0](2, 0) = -2.0 / 6.0 * c_to_a_ratio_;

      def_system_id_[13] = "twin_2";

      twin_plane_normal_[1](0, 0) = -1.0;
      twin_plane_normal_[1](1, 0) = 1.0;
      twin_plane_normal_[1](2, 0) = 1.0 / c_to_a_ratio_;

      twin_direction_[1](0, 0) = -1.0 / 6.0;
      twin_direction_[1](1, 0) = 1.0 / 6.0;
      twin_direction_[1](2, 0) = -2.0 / 6.0 * c_to_a_ratio_;

      def_system_id_[14] = "twin_3";

      twin_plane_normal_[2](0, 0) = 1.0;
      twin_plane_normal_[2](1, 0) = -1.0;
      twin_plane_normal_[2](2, 0) = 1.0 / c_to_a_ratio_;

      twin_direction_[2](0, 0) = 1.0 / 6.0;
      twin_direction_[2](1, 0) = -1.0 / 6.0;
      twin_direction_[2](2, 0) = -2.0 / 6.0 * c_to_a_ratio_;

      def_system_id_[15] = "twin_4";

      twin_plane_normal_[3](0, 0) = 1.0;
      twin_plane_normal_[3](1, 0) = 1.0;
      twin_plane_normal_[3](2, 0) = -1.0 / c_to_a_ratio_;

      twin_direction_[3](0, 0) = 1.0 / 6.0;
      twin_direction_[3](1, 0) = 1.0 / 6.0;
      twin_direction_[3](2, 0) = 2.0 / 6.0 * c_to_a_ratio_;
    }
  }
  // D019 super lattice
  else if (lattice_type_ == "D019")
  {
    // initialize slip plane normals and directions for Miller-Bravais indices
    std::vector<Core::LinAlg::Matrix<4, 1>> slip_plane_normal_hex(slip_system_count_);
    std::vector<Core::LinAlg::Matrix<4, 1>> slip_direction_hex(slip_system_count_);

    // NOTE: the c to a ratio is incorporated during transformation from Miller-Bravais to
    // Miller

    // prismatic slip systems
    def_system_id_[0] = "prismatic_1";

    slip_plane_normal_hex[0](0, 0) = 1.0;
    slip_plane_normal_hex[0](1, 0) = -1.0;
    slip_plane_normal_hex[0](2, 0) = 0.0;
    slip_plane_normal_hex[0](3, 0) = 0.0;

    slip_direction_hex[0](0, 0) = 1.0 / 3.0;
    slip_direction_hex[0](1, 0) = 1.0 / 3.0;
    slip_direction_hex[0](2, 0) = -2.0 / 3.0;
    slip_direction_hex[0](3, 0) = 0.0;

    def_system_id_[1] = "prismatic_2";

    slip_plane_normal_hex[1](0, 0) = 0.0;
    slip_plane_normal_hex[1](1, 0) = 1.0;
    slip_plane_normal_hex[1](2, 0) = -1.0;
    slip_plane_normal_hex[1](3, 0) = 0.0;

    slip_direction_hex[1](0, 0) = -2.0 / 3.0;
    slip_direction_hex[1](1, 0) = 1.0 / 3.0;
    slip_direction_hex[1](2, 0) = 1.0 / 3.0;
    slip_direction_hex[1](3, 0) = 0.0;

    def_system_id_[2] = "prismatic_3";

    slip_plane_normal_hex[2](0, 0) = 1.0;
    slip_plane_normal_hex[2](1, 0) = 0.0;
    slip_plane_normal_hex[2](2, 0) = -1.0;
    slip_plane_normal_hex[2](3, 0) = 0.0;

    slip_direction_hex[2](0, 0) = 1.0 / 3.0;
    slip_direction_hex[2](1, 0) = -2.0 / 3.0;
    slip_direction_hex[2](2, 0) = 1.0 / 3.0;
    slip_direction_hex[2](3, 0) = 0.0;

    // basal slip systems

    def_system_id_[3] = "basal_1";

    slip_plane_normal_hex[3](0, 0) = 0.0;
    slip_plane_normal_hex[3](1, 0) = 0.0;
    slip_plane_normal_hex[3](2, 0) = 0.0;
    slip_plane_normal_hex[3](3, 0) = 1.0;

    slip_direction_hex[3](0, 0) = 1.0 / 3.0;
    slip_direction_hex[3](1, 0) = 1.0 / 3.0;
    slip_direction_hex[3](2, 0) = -2.0 / 3.0;
    slip_direction_hex[3](3, 0) = 0.0;

    def_system_id_[4] = "basal_2";

    slip_plane_normal_hex[4](0, 0) = 0.0;
    slip_plane_normal_hex[4](1, 0) = 0.0;
    slip_plane_normal_hex[4](2, 0) = 0.0;
    slip_plane_normal_hex[4](3, 0) = 1.0;

    slip_direction_hex[4](0, 0) = -2.0 / 3.0;
    slip_direction_hex[4](1, 0) = 1.0 / 3.0;
    slip_direction_hex[4](2, 0) = 1.0 / 3.0;
    slip_direction_hex[4](3, 0) = 0.0;

    def_system_id_[5] = "basal_3";

    slip_plane_normal_hex[5](0, 0) = 0.0;
    slip_plane_normal_hex[5](1, 0) = 0.0;
    slip_plane_normal_hex[5](2, 0) = 0.0;
    slip_plane_normal_hex[5](3, 0) = 1.0;

    slip_direction_hex[5](0, 0) = 1.0 / 3.0;
    slip_direction_hex[5](1, 0) = -2.0 / 3.0;
    slip_direction_hex[5](2, 0) = 1.0 / 3.0;
    slip_direction_hex[5](3, 0) = 0.0;

    // pyramidal slip systems

    def_system_id_[6] = "pyramidal_1";

    slip_plane_normal_hex[6](0, 0) = 1.0;
    slip_plane_normal_hex[6](1, 0) = 1.0;
    slip_plane_normal_hex[6](2, 0) = -2.0;
    slip_plane_normal_hex[6](3, 0) = 1.0;

    slip_direction_hex[6](0, 0) = -1.0 / 3.0;
    slip_direction_hex[6](1, 0) = -1.0 / 3.0;
    slip_direction_hex[6](2, 0) = 2.0 / 3.0;
    slip_direction_hex[6](3, 0) = 6.0 / 3.0;

    def_system_id_[7] = "pyramidal_2";

    slip_plane_normal_hex[7](0, 0) = 1.0;
    slip_plane_normal_hex[7](1, 0) = -2.0;
    slip_plane_normal_hex[7](2, 0) = 1.0;
    slip_plane_normal_hex[7](3, 0) = 1.0;

    slip_direction_hex[7](0, 0) = -1.0 / 3.0;
    slip_direction_hex[7](1, 0) = 2.0 / 3.0;
    slip_direction_hex[7](2, 0) = -1.0 / 3.0;
    slip_direction_hex[7](3, 0) = 6.0 / 3.0;

    def_system_id_[8] = "pyramidal_3";

    slip_plane_normal_hex[8](0, 0) = -2.0;
    slip_plane_normal_hex[8](1, 0) = 1.0;
    slip_plane_normal_hex[8](2, 0) = 1.0;
    slip_plane_normal_hex[8](3, 0) = 1.0;

    slip_direction_hex[8](0, 0) = 2.0 / 3.0;
    slip_direction_hex[8](1, 0) = -1.0 / 3.0;
    slip_direction_hex[8](2, 0) = -1.0 / 3.0;
    slip_direction_hex[8](3, 0) = 6.0 / 3.0;

    def_system_id_[9] = "pyramidal_4";

    slip_plane_normal_hex[9](0, 0) = -1.0;
    slip_plane_normal_hex[9](1, 0) = -1.0;
    slip_plane_normal_hex[9](2, 0) = 2.0;
    slip_plane_normal_hex[9](3, 0) = 1.0;

    slip_direction_hex[9](0, 0) = 1.0 / 3.0;
    slip_direction_hex[9](1, 0) = 1.0 / 3.0;
    slip_direction_hex[9](2, 0) = -2.0 / 3.0;
    slip_direction_hex[9](3, 0) = 6.0 / 3.0;

    def_system_id_[10] = "pyramidal_5";

    slip_plane_normal_hex[10](0, 0) = -1.0;
    slip_plane_normal_hex[10](1, 0) = 2.0;
    slip_plane_normal_hex[10](2, 0) = -1.0;
    slip_plane_normal_hex[10](3, 0) = 1.0;

    slip_direction_hex[10](0, 0) = 1.0 / 3.0;
    slip_direction_hex[10](1, 0) = -2.0 / 3.0;
    slip_direction_hex[10](2, 0) = 1.0 / 3.0;
    slip_direction_hex[10](3, 0) = 6.0 / 3.0;

    def_system_id_[11] = "pyramidal_6";

    slip_plane_normal_hex[11](0, 0) = 2.0;
    slip_plane_normal_hex[11](1, 0) = -1.0;
    slip_plane_normal_hex[11](2, 0) = -1.0;
    slip_plane_normal_hex[11](3, 0) = 1.0;

    slip_direction_hex[11](0, 0) = -2.0 / 3.0;
    slip_direction_hex[11](1, 0) = 1.0 / 3.0;
    slip_direction_hex[11](2, 0) = 1.0 / 3.0;
    slip_direction_hex[11](3, 0) = 6.0 / 3.0;

    // transform Miller-Bravais index notation of hexagonal lattices to Miller index notation of
    // cubic lattices
    Mat::CrystalPlasticity::miller_bravais_to_miller(
        slip_plane_normal_hex, slip_direction_hex, slip_plane_normal_, slip_direction_);
  }
  else if (lattice_type_ == "HCP" or lattice_type_ == "BCC" or lattice_type_ == "FCC")
    FOUR_C_THROW(
        "{} lattices are not yet implemented. If you want to use these lattice types you "
        "need to extend the SetupLatticeVectors() method accordingly.",
        lattice_type_.c_str());
  // academic test lattice containing only one slip system
  else if (lattice_type_ == "TEST")
  {
    def_system_id_[0] = "test_slip";

    slip_plane_normal_[0](0, 0) = 1.0;
    slip_plane_normal_[0](1, 0) = 0.0;
    slip_plane_normal_[0](2, 0) = 1.0 / c_to_a_ratio_;

    slip_direction_[0](0, 0) = 1.0;
    slip_direction_[0](1, 0) = 0.0;
    slip_direction_[0](2, 0) = -1.0 * c_to_a_ratio_;

    def_system_id_[1] = "test_twin";

    twin_plane_normal_[0](0, 0) = -1.0;
    twin_plane_normal_[0](1, 0) = 0.0;
    twin_plane_normal_[0](2, 0) = 1.0 / c_to_a_ratio_;

    twin_direction_[0](0, 0) = 1.0;
    twin_direction_[0](1, 0) = 0.0;
    twin_direction_[0](2, 0) = 1.0 * c_to_a_ratio_;
  }
  else
    FOUR_C_THROW(
        "Lattice type not known. Please check the LAT parameter in the input file. Currently "
        "it has to be FCC, BCC, HCP, D019 or L10");
  // test whether directions and normals are perpendicular (required!) and identify which slip
  // and twinning systems are non-coplanar for work hardening
  for (int i = 0; i < slip_system_count_; i++)
  {
    // check for each system whether or not its direction and normal vector are perpendicular!
    bool is_perpendicular_ =
        Mat::CrystalPlasticity::check_orthogonal(slip_direction_[i], slip_plane_normal_[i]);
    if (!is_perpendicular_)
      FOUR_C_THROW(
          "Warning, slip direction and slip plane normal of slip system "
          "{} are not perpendicular! Check implementation of lattice vectors.",
          def_system_id_[i].c_str());
  }
  if (is_twinning_)
  {
    is_non_coplanar_.resize(twin_system_count_);
    for (int i = 0; i < twin_system_count_; i++)
    {
      // check for each system whether or not its direction and normal vector are perpendicular!
      bool is_perpendicular_ =
          Mat::CrystalPlasticity::check_orthogonal(twin_direction_[i], twin_plane_normal_[i]);
      if (!is_perpendicular_)
        FOUR_C_THROW(
            "Warning, twinning direction and twinning plane normal of twinning system "
            "{} are not perpendicular! Check implementation of lattice vectors.",
            def_system_id_[i].c_str());

      is_non_coplanar_[i].resize(def_system_count_);

      // check which slip systems are non-coplanar to twinning system i
      for (int j = 0; j < slip_system_count_; j++)
      {
        bool is_coplanar_ =
            Mat::CrystalPlasticity::check_parallel(twin_plane_normal_[i], slip_plane_normal_[j]);
        if (is_coplanar_)
          is_non_coplanar_[i][j] = false;
        else
          is_non_coplanar_[i][j] = true;
      }
      // check which twinning systems are non-coplanar to twinning system i
      for (int j = slip_system_count_; j < def_system_count_; j++)
      {
        bool is_coplanar_ = Mat::CrystalPlasticity::check_parallel(
            twin_plane_normal_[i], twin_plane_normal_[j - slip_system_count_]);
        if (is_coplanar_)
          is_non_coplanar_[i][j] = false;
        else
          is_non_coplanar_[i][j] = true;
      }
    }
  }

  // determine magnitude of Burgers vectors
  slip_burgers_mag_.resize(slip_system_count_);
  if (is_twinning_) twin_burgers_mag_.resize(twin_system_count_);

  // import lattice constant from user input
  lattice_constant_ = params_->lattice_const_;

  Core::LinAlg::Matrix<3, 1> tmp_scale;

  for (int i = 0; i < slip_system_count_; i++)
  {
    tmp_scale = slip_direction_[i];
    tmp_scale.scale(lattice_constant_);
    slip_burgers_mag_[i] = tmp_scale.norm2();
  }
  if (is_twinning_)
  {
    for (int i = 0; i < twin_system_count_; i++)
    {
      tmp_scale = twin_direction_[i];
      tmp_scale.scale(lattice_constant_);
      twin_burgers_mag_[i] = tmp_scale.norm2();
    }
  }

  // normalize slip/twinning plane normals and directions
  for (int i = 0; i < slip_system_count_; i++)
  {
    slip_plane_normal_[i].scale(1.0 / slip_plane_normal_[i].norm2());
    slip_direction_[i].scale(1.0 / slip_direction_[i].norm2());
  }
  if (is_twinning_)
  {
    for (int i = 0; i < twin_system_count_; i++)
    {
      twin_plane_normal_[i].scale(1.0 / twin_plane_normal_[i].norm2());
      twin_direction_[i].scale(1.0 / twin_direction_[i].norm2());
    }
  }
  return;
}

/*---------------------------------------------------------------------------------*
 | Read Lattice orientation matrix from input file                                    |
 *---------------------------------------------------------------------------------*/

void Mat::CrystalPlasticity::setup_lattice_orientation(const Discret::Elements::Fibers& fibers)
{
  FOUR_C_ASSERT_ALWAYS(fibers.element_fibers.size() == 3,
      "No lattice orientation matrix provided! Please add 3 fibers as columns of the rotation "
      "matrix that relates the lattice orientation to the global coordinate system");


  // assemble rotation matrix
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      lattice_orientation_(j, i) = fibers.element_fibers[i](j);
    }
  }
}


/*---------------------------------------------------------------------------------*
 | Transform Miller Bravais to Miller index notation                               |
 *---------------------------------------------------------------------------------*/

void Mat::CrystalPlasticity::miller_bravais_to_miller(
    const std::vector<Core::LinAlg::Matrix<4, 1>>& plane_normal_hex,
    const std::vector<Core::LinAlg::Matrix<4, 1>>& direction_hex,
    std::vector<Core::LinAlg::Matrix<3, 1>>& plane_normal,
    std::vector<Core::LinAlg::Matrix<3, 1>>& direction)
{
  for (int unsigned i = 0; i < plane_normal_hex.size(); i++)
  {
    plane_normal[i](0, 0) = plane_normal_hex.at(i)(0, 0);
    plane_normal[i](1, 0) =
        (plane_normal_hex.at(i)(0, 0) + 2.0 * plane_normal_hex.at(i)(1, 0)) / std::sqrt(3.0);
    plane_normal[i](2, 0) = plane_normal_hex.at(i)(3, 0) / c_to_a_ratio_;

    direction[i](0, 0) = 1.5 * direction_hex.at(i)(0, 0);
    direction[i](1, 0) =
        std::sqrt(3.0) * (0.5 * direction_hex.at(i)(0, 0) + direction_hex.at(i)(1, 0));
    direction[i](2, 0) = direction_hex.at(i)(3, 0) * c_to_a_ratio_;
  }
  return;
}

/*---------------------------------------------------------------------------------*
 | Check if two vectors are parallel			                                   |
 *---------------------------------------------------------------------------------*/

bool Mat::CrystalPlasticity::check_parallel(
    const Core::LinAlg::Matrix<3, 1>& vector_1, const Core::LinAlg::Matrix<3, 1>& vector_2)
{
  Core::LinAlg::Matrix<1, 1> parallel_test;
  parallel_test.multiply_tn(vector_1, vector_2);
  if (parallel_test(0, 0) - 1.0 < 1.0e-8)
    return true;
  else
    return false;
}

/*---------------------------------------------------------------------------------*
 | Check if two vectors are orthogonal                                             |
 *---------------------------------------------------------------------------------*/

bool Mat::CrystalPlasticity::check_orthogonal(
    const Core::LinAlg::Matrix<3, 1>& vector_1, const Core::LinAlg::Matrix<3, 1>& vector_2)
{
  Core::LinAlg::Matrix<1, 1> ortho_test;
  ortho_test.multiply_tn(vector_1, vector_2);
  if (ortho_test(0, 0) < 1.0e-8)
    return true;
  else
    return false;
}

/*---------------------------------------------------------------------------------*
 | local Newton-Raphson iteration                                                  |
 *---------------------------------------------------------------------------------*/
void Mat::CrystalPlasticity::newton_raphson(Core::LinAlg::Matrix<3, 3>& deform_grad,
    std::vector<double>& gamma_result, std::vector<double>& defect_densites_result,
    Core::LinAlg::Matrix<3, 3>& second_pk_stress_result,
    Core::LinAlg::Matrix<3, 3>& plastic_deform_grad_result)
{
  // initialize iteration
  // ----------------------------------
  // max. number of iterations
  const int max_iterations = 20;
  // iteration counter
  int iteration_number = 0;
  // convergence indicator
  bool converged = false;

  // elastic predictor
  //--------------------------------------------------------------------------
  // assume load step is elastic, i.e., there is no change in plastic shear
  std::vector<double> gamma_trial = gamma_last_->at(gp_);

  // trial values of internal variables
  std::vector<double> defect_densities_trial(def_system_count_);

  Core::LinAlg::Matrix<3, 3> second_pk_stress_trial;
  Core::LinAlg::Matrix<3, 3> plastic_deform_grad_trial;

  // residuals
  std::vector<double> residuals_trial(def_system_count_);

  double total_residual;

  // empty matrix
  Core::LinAlg::Matrix<3, 3> emptymat(Core::LinAlg::Initialization::zero);

  // iteration
  //--------------------------------------------------------------------------
  while (!converged && (iteration_number++ < max_iterations))
  {
    // reset iteration relevant values
    total_residual = 0.0;
    second_pk_stress_trial = emptymat;

    // set up flow rule with given deformation gradient F and trial plastic shears gamma_trial
    this->setup_flow_rule(deform_grad, gamma_trial, plastic_deform_grad_trial,
        defect_densities_trial, second_pk_stress_trial, residuals_trial);

    // determine total residuum as norm of the vector of slip and twinning system residuals
    for (int i = 0; i < def_system_count_; i++)
    {
      total_residual += std::pow(residuals_trial.at(i), 2.0);
    }

    total_residual = std::sqrt(total_residual);

    // convergence check
    if (total_residual < newton_tolerance_)
    {
      converged = true;

      // collect results
      //--------------------------------------------------------------------------
      gamma_result = gamma_trial;
      defect_densites_result = defect_densities_trial;
      second_pk_stress_result = second_pk_stress_trial;
      plastic_deform_grad_result = plastic_deform_grad_trial;
    }
    else
    {
      // update rule: gamma_trial(n+1) = gamma_trial(n) + d_gamma_trial(n)
      // therefore the Jacobean ResidualStiffness(j,i)=dr(j)/d_gamma_trial(i) is set up, where r
      // is the residuum solving ResidualStiffness * d_gamma_trial(n) = - r

      // depending on total number of deformation systems the equation system has a different
      // size def_system_count_ = 1
      Core::LinAlg::Matrix<1, 1> residual_tangent_1x1;
      Core::LinAlg::Matrix<1, 1> d_gamma_trial_1(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<1, 1> residuals_trial_LIN_1;
      Core::LinAlg::FixedSizeSerialDenseSolver<1, 1, 1> newton_raphson_solver_1x1;
      // depending on total number of deformation systems the equation system has a different
      // size def_system_count_ = 2
      Core::LinAlg::Matrix<2, 2> residual_tangent_2x2;
      Core::LinAlg::Matrix<2, 1> d_gamma_trial_2(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<2, 1> residuals_trial_LIN_2;
      Core::LinAlg::FixedSizeSerialDenseSolver<2, 2, 1> newton_raphson_solver_2x2;
      // def_system_count_ = 12
      Core::LinAlg::Matrix<12, 12> residual_tangent_12x12;
      Core::LinAlg::Matrix<12, 1> d_gamma_trial_12(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<12, 1> residuals_trial_LIN_12;
      Core::LinAlg::FixedSizeSerialDenseSolver<12, 12, 1> newton_raphson_solver_12x12;
      // def_system_count_ = 16
      Core::LinAlg::Matrix<16, 16> residual_tangent_16x16;
      Core::LinAlg::Matrix<16, 1> d_gamma_trial_16(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<16, 1> residuals_trial_LIN_16;
      Core::LinAlg::FixedSizeSerialDenseSolver<16, 16, 1> newton_raphson_solver_16x16;

      // differentiate Residuum with respect to gamma_trial by perturbation DEFAULT:
      // gamma_epsilon = 1.0e-9
      double gamma_epsilon = 1.0e-9;

      // change of residuals.at(j) with perturbation of gamma_trial_.at(i)

      // perturb single plastic shears
      for (int i = 0; i < def_system_count_; i++)
      {
        // perturbed vector of plastic shears
        std::vector<double> gamma_perturbed(def_system_count_);

        // resultant vectors of residuals
        std::vector<double> residuals_perturbed(def_system_count_);

        // resultant vectors of defect densities
        std::vector<double> defect_densities_perturbed(def_system_count_);

        // resultant 2nd PK stress
        Core::LinAlg::Matrix<3, 3> second_pk_stress_perturbed(Core::LinAlg::Initialization::zero);

        // resultant plastic part of deformation gradient
        Core::LinAlg::Matrix<3, 3> plastic_deform_grad_perturbed;

        // initially unperturbed
        gamma_perturbed = gamma_trial;

        // perturbation of single slip system
        gamma_perturbed.at(i) += gamma_epsilon;

        // evaluate flow rule for perturbed gamma
        this->setup_flow_rule(deform_grad, gamma_perturbed, plastic_deform_grad_perturbed,
            defect_densities_perturbed, second_pk_stress_perturbed, residuals_perturbed);

        // finite difference
        for (int j = 0; j < def_system_count_; j++)
        {
          if (def_system_count_ == 1)
          {
            residual_tangent_1x1(j, i) =
                (residuals_perturbed.at(j) - residuals_trial.at(j)) / gamma_epsilon;
            residuals_trial_LIN_1(i) = -residuals_trial.at(i);
          }
          else if (def_system_count_ == 2)
          {
            residual_tangent_2x2(j, i) =
                (residuals_perturbed.at(j) - residuals_trial.at(j)) / gamma_epsilon;
            residuals_trial_LIN_2(i) = -residuals_trial.at(i);
          }
          else if (def_system_count_ == 12)
          {
            residual_tangent_12x12(j, i) =
                (residuals_perturbed.at(j) - residuals_trial.at(j)) / gamma_epsilon;
            residuals_trial_LIN_12(i) = -residuals_trial.at(i);
          }
          else if (def_system_count_ == 16)
          {
            residual_tangent_16x16(j, i) =
                (residuals_perturbed.at(j) - residuals_trial.at(j)) / gamma_epsilon;
            residuals_trial_LIN_16(i) = -residuals_trial.at(i);
          }
        }
      }


      // solve resultant system of equations
      if (def_system_count_ == 1)
      {
        newton_raphson_solver_1x1.set_matrix(residual_tangent_1x1);
        newton_raphson_solver_1x1.set_vectors(d_gamma_trial_1, residuals_trial_LIN_1);
        newton_raphson_solver_1x1.solve();

        // update vector of plastic shears to new trial value
        for (int i = 0; i < def_system_count_; i++)
        {
          gamma_trial.at(i) += d_gamma_trial_1(i);
        }
      }
      else if (def_system_count_ == 2)
      {
        newton_raphson_solver_2x2.set_matrix(residual_tangent_2x2);
        newton_raphson_solver_2x2.set_vectors(d_gamma_trial_2, residuals_trial_LIN_2);
        newton_raphson_solver_2x2.solve();

        // update vector of plastic shears to new trial value
        for (int i = 0; i < def_system_count_; i++)
        {
          gamma_trial.at(i) += d_gamma_trial_2(i);
        }
      }
      else if (def_system_count_ == 12)
      {
        newton_raphson_solver_12x12.set_matrix(residual_tangent_12x12);
        newton_raphson_solver_12x12.set_vectors(d_gamma_trial_12, residuals_trial_LIN_12);
        newton_raphson_solver_12x12.solve();

        // update vector of plastic shears to new trial value
        for (int i = 0; i < def_system_count_; i++)
        {
          gamma_trial.at(i) += d_gamma_trial_12(i);
        }
      }
      else if (def_system_count_ == 16)
      {
        newton_raphson_solver_16x16.set_matrix(residual_tangent_16x16);
        newton_raphson_solver_16x16.set_vectors(d_gamma_trial_16, residuals_trial_LIN_16);
        newton_raphson_solver_16x16.solve();

        // update vector of plastic shears to new trial value
        for (int i = 0; i < def_system_count_; i++)
        {
          gamma_trial.at(i) += d_gamma_trial_16(i);
        }
      }
    }
  }  // while

  // check whether or not the result converged during max_iterations iterations.
  // TODO: If not: reduce time step and restart time increment!
  if (!converged)
  {
    FOUR_C_THROW("Internal Newton Raphson did not converge within max iterations.");
  }

  return;
}  // NewtonRaphson


/*---------------------------------------------------------------------------------*
 | Evaluate flow rule for a  given increment of plastic shear and set up residuum  |
 *---------------------------------------------------------------------------------*/

void Mat::CrystalPlasticity::setup_flow_rule(Core::LinAlg::Matrix<3, 3> deform_grad,
    std::vector<double> gamma_trial, Core::LinAlg::Matrix<3, 3>& plastic_deform_grad_trial,
    std::vector<double>& defect_densities_trial, Core::LinAlg::Matrix<3, 3>& second_pk_stress_trial,
    std::vector<double>& residuals_trial)
{
  // set up trial increment of plastic shear for each slip and twinning system
  std::vector<double> delta_gamma_trial(def_system_count_);

  for (int i = 0; i < def_system_count_; i++)
  {
    delta_gamma_trial[i] = gamma_trial[i] - (*gamma_last_)[gp_][i];
  }

  // determine trial defect densities that would result from delta_gamma_trial
  //--------------------------------------------------------------------------
  // total dislocation density and twinned volume fraction based on last time steps results
  double total_dislocation_density_last = 0.0;

  for (int i = 0; i < slip_system_count_; i++)
    total_dislocation_density_last += (*defect_densities_last_)[gp_][i];

  // current total defect densities
  double total_twinned_volume_curr = 0.0;
  double total_dislocation_density_curr = 0.0;

  // update defect densities for delta_gamma_trial
  for (int i = 0; i < slip_system_count_; i++)
  {
    // index of subset of the respective deformation system
    int ind = slip_set_index_[i] - 1;

    // dislocation generation
    defect_densities_trial[i] =
        (*defect_densities_last_)[gp_][i] +
        (1.0 / (slip_burgers_mag_[i] * dislocation_generation_coeff_[ind])) *
            std::sqrt(total_dislocation_density_last) * std::abs(delta_gamma_trial[i]);

    // dynamic recovery
    defect_densities_trial[i] -= dislocation_dyn_recovery_coeff_[ind] *
                                 (*defect_densities_last_)[gp_][i] * std::abs(delta_gamma_trial[i]);
    // determine updated total dislocation density
    total_dislocation_density_curr += defect_densities_trial[i];
  }
  if (is_twinning_)
  {
    // update twinned volumes for delta_gamma_twin_trial
    for (int i = slip_system_count_; i < def_system_count_; i++)
    {
      // twinned volume fraction directly evolves with twinning rate multiplied by twinning
      // shear
      // TODO INCLUDE TWINNING SHEAR OF RESPECTIVE LATTICE HERE!!! BEST IN SETUP LAZTTICE
      // VECTORS!
      defect_densities_trial[i] =
          (*defect_densities_last_)[gp_][i] + std::abs(delta_gamma_trial[i]) * std::sqrt(2.0);

      // determine updated total twinned volume fraction
      total_twinned_volume_curr += defect_densities_trial[i];
    }
  }


  // kinematics and plastic velocity gradient
  //--------------------------------------------------------------------------

  // determine trial plastic velocity gradient L_p
  // set up L_p_trial
  Core::LinAlg::Matrix<3, 3> plastic_velocity_grad_trial(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp_mat;

  // slip system contributions
  for (int i = 0; i < slip_system_count_; i++)
  {
    temp_mat.multiply_nt(delta_gamma_trial[i], slip_direction_[i], slip_plane_normal_[i]);
    plastic_velocity_grad_trial.update(plastic_velocity_grad_trial, temp_mat);
  }
  if (is_twinning_)
  {
    // twinning system contributions
    for (int i = slip_system_count_; i < def_system_count_; i++)
    {
      temp_mat.multiply_nt(delta_gamma_trial[i], twin_direction_[i - slip_system_count_],
          twin_plane_normal_[i - slip_system_count_]);
      plastic_velocity_grad_trial.update(plastic_velocity_grad_trial, temp_mat);
    }
  }

  // take unimodular part of I + L_p to ensure plastic incompressibility
  Core::LinAlg::Matrix<3, 3> unimod_identity_plus_plastic_velocity_grad_trial(
      Core::LinAlg::Initialization::zero);

  unimod_identity_plus_plastic_velocity_grad_trial.update(
      Core::LinAlg::identity_matrix<3>(), plastic_velocity_grad_trial);
  unimod_identity_plus_plastic_velocity_grad_trial.scale(
      std::pow(unimod_identity_plus_plastic_velocity_grad_trial.determinant(), -1.0 / 3.0));

  // determine trial plastic deformation gradient
  plastic_deform_grad_trial.multiply_nn(
      unimod_identity_plus_plastic_velocity_grad_trial, (*plastic_deform_grad_last_)[gp_]);

  // determine trial stress
  //--------------------------------------------------------------------------
  // determine trial elastic deformation gradient
  // get the inverse FP^{-1}
  Core::LinAlg::Matrix<3, 3> inv_plastic_deform_grad_trial;
  inv_plastic_deform_grad_trial.invert(plastic_deform_grad_trial);
  Core::LinAlg::Matrix<3, 3> elastic_deform_grad_trial;
  elastic_deform_grad_trial.multiply_nn(deform_grad, inv_plastic_deform_grad_trial);

  // calculate the Jacobi-determinant J = det(FE_{n+1}) and the logarithm of it
  double jacobi_det_trial = elastic_deform_grad_trial.determinant();
  double ln_jacobi_det_trial = std::log(jacobi_det_trial);

  // set up elastic right cauchy green and its inverse
  Core::LinAlg::Matrix<3, 3> elastic_right_cauchy_green;
  elastic_right_cauchy_green.multiply_tn(elastic_deform_grad_trial, elastic_deform_grad_trial);
  Core::LinAlg::Matrix<3, 3> inv_elastic_right_cauchy_green;
  inv_elastic_right_cauchy_green.invert(elastic_right_cauchy_green);

  // 2nd Piola-Kirchhoff stress
  // S = lambda * ln_jacobi_det_trial * inv_elastic_right_cauchy_green + mu * (identity3
  // - inv_elastic_right_cauchy_green)

  second_pk_stress_trial.update(lambda_ * ln_jacobi_det_trial, inv_elastic_right_cauchy_green, 1.0);
  second_pk_stress_trial.update(mue_, Core::LinAlg::identity_matrix<3>(), 1.0);
  second_pk_stress_trial.update(-mue_, inv_elastic_right_cauchy_green, 1.0);

  // Mandel stress
  Core::LinAlg::Matrix<3, 3> mandel_stress_trial;
  mandel_stress_trial.multiply_nn(elastic_right_cauchy_green, second_pk_stress_trial);

  // determine current slip systems strengths
  //--------------------------------------------------------------------------
  for (int i = 0; i < slip_system_count_; i++)
  {
    // index of subset of the respective deformation system
    int ind = slip_set_index_[i] - 1;

    // lattice resistance
    double tau_y0 = tau_y_0_[ind];

    // Hall-Petch strengthening term
    double hall_petch_strengthening =
        hall_petch_coeffs_slip_[ind] * (1.0 / std::sqrt(micro_boundary_distance_slip_[ind]));

    // work hardening increment
    double delta_tau_y = 0.0;

    // work hardening hardening increment of slip systems due to accumulated dislocation density
    delta_tau_y = 0.5 * mue_ * slip_burgers_mag_[i] * std::sqrt(total_dislocation_density_curr);

    // additional hardening due to non-coplanar twins
    if (is_twinning_)
    {
      // volume fraction of non-coplanar twins
      double twinned_volume_ncp = 0.0;

      for (int j = 0; j < twin_system_count_; j++)
      {
        if (is_non_coplanar_[j][i])
          twinned_volume_ncp += defect_densities_trial[slip_system_count_ + j];
      }
      if (total_twinned_volume_curr < 1.0)
        delta_tau_y += slip_by_twin_hard_[ind] * (twinned_volume_ncp / (1.0 - twinned_volume_ncp));
      else
        delta_tau_y += 10000.0;
    }

    // slip system strength, i.e. tau_y = tau_y0 + hall_petch_strengthening + delta_tau_y
    double tau_y = tau_y0 + hall_petch_strengthening + delta_tau_y;

    // check for consistency
    if (tau_y < 0.0) FOUR_C_THROW("Negative slip systems strength! Please check your input");

    // setting up Residua with flow/twinning rule
    //--------------------------------------------------------------------------

    // resolved shear stress/Schmid stress
    Core::LinAlg::Matrix<1, 1> resolved_shear_stress(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<3, 1> TempVec(Core::LinAlg::Initialization::zero);

    TempVec.multiply_nn(mandel_stress_trial, slip_plane_normal_[i]);

    resolved_shear_stress.multiply_tn(slip_direction_[i], TempVec);
    // shear rate as determined from the flow rule
    double gamma_dot = 0.0;

    gamma_dot = std::abs(resolved_shear_stress(0, 0)) / tau_y;
    gamma_dot = std::pow(gamma_dot, n_slip_[ind]);
    gamma_dot = gamma_dot * gamma_dot_0_slip_[ind];
    gamma_dot = std::copysign(gamma_dot, resolved_shear_stress(0, 0));

    // set up corresponding residual
    residuals_trial[i] = (delta_gamma_trial[i] / dt_) - gamma_dot;
  }
  if (is_twinning_)
  {
    // determine current twinning systems strengths
    //--------------------------------------------------------------------------
    for (int i = slip_system_count_; i < def_system_count_; i++)
    {
      // index of subset of the respective deformation system
      int ind = twin_set_index_[i - slip_system_count_] - 1;

      // lattice resistance
      double tau_t0 = tau_t_0_[ind];

      // Hall-Petch strengthening term
      double hall_petch_strengthening =
          hall_petch_coeffs_twin_[ind] * (1.0 / std::sqrt(micro_boundary_distance_twin_[ind]));

      // work hardening increment
      double delta_tau_t = 0.0;

      // effective dislocation density for work hardening of twins
      double effective_dislocation_density = 0.0;
      for (int j = 0; j < slip_system_count_; j++)
        effective_dislocation_density += defect_densities_trial[j] * slip_burgers_mag_[j];

      // work hardening by slip
      delta_tau_t = twin_by_slip_hard_[ind] * mue_ * twin_burgers_mag_[i - slip_system_count_] *
                    effective_dislocation_density;

      // volume fraction of non-coplanar twins
      double twinned_volume_ncp = 0.0;

      for (int j = 0; j < twin_system_count_; j++)
      {
        if (is_non_coplanar_[j][i])
        {
          twinned_volume_ncp += defect_densities_trial[slip_system_count_ + j];
        }
      }
      if (total_twinned_volume_curr < 1.0)
        delta_tau_t += twin_by_twin_hard_[ind] * (twinned_volume_ncp / (1.0 - twinned_volume_ncp));
      else
        delta_tau_t += 10000.0;

      // twinning system strength, i.e. tau_y = tau_y0 + hall_petch_strengthening + delta_tau_y
      double tau_t = tau_t0 + hall_petch_strengthening + delta_tau_t;

      // check for consistency
      if (tau_t < 0.0) FOUR_C_THROW("Negative twinning systems strength! Please check your input");

      // setting up Residua with flow/twinning rule
      //--------------------------------------------------------------------------

      // resolved shear stress/Schmid stress
      Core::LinAlg::Matrix<1, 1> resolved_shear_stress(Core::LinAlg::Initialization::zero);

      Core::LinAlg::Matrix<3, 1> TempVec(Core::LinAlg::Initialization::zero);

      TempVec.multiply_nn(mandel_stress_trial, twin_plane_normal_[i - slip_system_count_]);

      resolved_shear_stress.multiply_tn(twin_direction_[i - slip_system_count_], TempVec);

      // shear rate as determined from the twinning rule
      double gamma_dot = 0.0;

      if ((resolved_shear_stress(0, 0) > 0.0) && (total_twinned_volume_curr < 1.0))
      {
        gamma_dot = resolved_shear_stress(0, 0) / tau_t;
        gamma_dot = std::pow(gamma_dot, n_twin_[ind]);
        gamma_dot = gamma_dot * gamma_dot_0_twin_[ind];
      }
      // set up corresponding residual
      residuals_trial[i] = (delta_gamma_trial[i] / dt_) - gamma_dot;
    }
  }
  return;
}  // FlowRuleSetup()

/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::CrystalPlasticity::vis_names(std::map<std::string, int>& names) const
{
  // temporary string variable for assembling the output names
  std::string ID;

  // plastic shears on single systems
  for (int sys = 0; sys < def_system_count_; sys++)
  {
    ID = "plastic_shear_";
    ID += def_system_id_[sys];
    names[ID] = 1;  // scalar
  }

  // accumulated plastic shears
  ID = "accumulated_plastic_shear";
  names[ID] = 1;  // scalar

  // dislocation densities on single systems
  for (int sys = 0; sys < slip_system_count_; sys++)
  {
    ID = "dislocation_density_";
    ID += def_system_id_[sys];
    names[ID] = 1;  // scalar
  }
  if (is_twinning_)
  {
    // twinned volume fractions on single systems
    for (int sys = slip_system_count_; sys < def_system_count_; sys++)
    {
      ID = "twinned_volume_";
      ID += def_system_id_[sys];
      names[ID] = 1;  // scalar
    }
  }

  // total dislocation density
  ID = "total_dislocation_density";
  names[ID] = 1;  // scalar

  if (is_twinning_)
  {
    // total dislocation density
    ID = "total_twinned_volume";
    names[ID] = 1;  // scalar
  }

}  // vis_names()

/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::CrystalPlasticity::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  // plastic shears gamma
  for (int sys = 0; sys < def_system_count_; sys++)
  {
    if (name == "plastic_shear_" + def_system_id_[sys])
    {
      if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
      for (int gp_index = 0; gp_index < numgp; gp_index++)
      {
        data[0] = (*gamma_last_)[gp_index][sys];
      }
    }
  }

  // accumulated plastic shear gamma_acc
  if (name == "accumulated_plastic_shear")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double gamma_acc;
    for (int gp_index = 0; gp_index < numgp; gp_index++)
    {
      gamma_acc = 0.0;
      for (int sys = 0; sys < def_system_count_; sys++)
      {
        gamma_acc += std::abs((*gamma_last_)[gp_index][sys]);
      }
      data[0] = gamma_acc;
    }
  }

  // dislocation densities
  for (int sys = 0; sys < slip_system_count_; sys++)
  {
    if (name == "dislocation_density_" + def_system_id_[sys])
    {
      if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
      for (int gp_index = 0; gp_index < numgp; gp_index++)
      {
        data[0] = (*defect_densities_last_)[gp_index][sys];
      }
    }
  }
  if (is_twinning_)
  {
    // twinned volume fractions
    for (int sys = slip_system_count_; sys < def_system_count_; sys++)
    {
      if (name == "twinned_volume_" + def_system_id_[sys])
      {
        if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
        for (int gp_index = 0; gp_index < numgp; gp_index++)
        {
          data[0] = (*defect_densities_last_)[gp_index][sys];
        }
      }
    }
  }

  // accumulated dislocation density
  if (name == "total_dislocation_density")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double dis_dens_acc;
    for (int gp_index = 0; gp_index < numgp; gp_index++)
    {
      dis_dens_acc = 0.0;
      for (int sys = 0; sys < slip_system_count_; sys++)
      {
        dis_dens_acc += (*defect_densities_last_)[gp_index][sys];
      }
      data[0] = dis_dens_acc;
    }
  }

  if (is_twinning_)
  {
    // accumulated twinned volume fraction
    if (name == "total_twinned_volume")
    {
      if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
      double twin_vol_acc;
      for (int gp_index = 0; gp_index < numgp; gp_index++)
      {
        twin_vol_acc = 0.0;
        for (int sys = slip_system_count_; sys < def_system_count_; sys++)
        {
          twin_vol_acc += (*defect_densities_last_)[gp_index][sys];
        }
        data[0] = twin_vol_acc;
      }
    }
  }
  return true;
}  // vis_data()

FOUR_C_NAMESPACE_CLOSE
