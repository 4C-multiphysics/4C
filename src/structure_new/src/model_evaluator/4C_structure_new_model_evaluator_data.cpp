// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_data.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_solver_nonlin_nox_statustest_normupdate.hpp"
#include "4C_solver_nonlin_nox_statustest_normwrms.hpp"
#include "4C_structure_new_discretization_runtime_output_params.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_nln_solver_nox.hpp"
#include "4C_structure_new_nln_solver_utils.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"
#include "4C_structure_new_timint_implicit.hpp"


FOUR_C_NAMESPACE_OPEN

namespace
{
  static void send_to_next_proc(const int p, Core::Communication::Exporter& exporter,
      const std::vector<int>& mysize, const std::vector<char>& mydata,
      std::vector<int>& receivedsize, std::vector<char>& receiveddata)
  {
    MPI_Comm comm = exporter.get_comm();

    const int numprocs = Core::Communication::num_mpi_ranks(comm);
    const int myrank = Core::Communication::my_mpi_rank(comm);
    int tag = myrank;

    int frompid = myrank;
    const int topid = (myrank + 1) % numprocs;

    // send size
    MPI_Request sizerequest;
    exporter.i_send(frompid, topid, mysize.data(), mysize.size(), tag, sizerequest);

    // send data
    MPI_Request datarequest;
    exporter.i_send(frompid, topid, mydata.data(), mydata.size(), tag * 10, datarequest);

    // make sure that you do not think you received something if
    // you didn't
    if (not receiveddata.empty() or not receivedsize.empty())
      FOUR_C_THROW("Received data objects are not empty!");

    // receive from predecessor
    frompid = (myrank + numprocs - 1) % numprocs;

    // receive size information
    int length = 0;
    exporter.receive_any(frompid, tag, receivedsize, length);
    if (length != static_cast<int>(mysize.size()) or tag != frompid)
      FOUR_C_THROW(
          "Size information got mixed up!\n"
          "Received length = {}, Expected length = {} \n"
          "Received tag    = {}, Expected tag    = {}",
          length, mysize.size(), tag, frompid);

    exporter.wait(sizerequest);

    // receive the gids
    exporter.receive_any(frompid, tag, receiveddata, length);
    if (length != receivedsize[0] or tag != frompid * 10)
      FOUR_C_THROW(
          "Data information got mixed up! \n"
          "Received length = {}, Expected length = {} \n"
          "Received tag    = {}, Expected tag    = {}",
          length, receivedsize[0], tag, frompid * 10);

    exporter.wait(datarequest);
  }

  template <typename T>
  static void unpack_received_block(const std::vector<char>& receiveddata, T& collected_data)
  {
    Core::Communication::UnpackBuffer buffer(receiveddata);
    while (!buffer.at_end())
    {
      // the set gets cleared at the beginning of the extract_from_pack routine!
      T rs;
      extract_from_pack(buffer, rs);
      collected_data.insert(rs.begin(), rs.end());
    }
  }

  template <typename T>
  static void round_robin_loop(
      MPI_Comm comm, Core::Communication::PackBuffer& pack_data, T& collected_data)
  {
    // collect the information over all procs
    std::vector<char> mydata;

    // swap into std::vector
    std::swap(mydata, pack_data());

    std::vector<int> mysize(1, mydata.size());

    std::vector<int> receivedsize;
    std::vector<char> receiveddata;

    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(comm);
    const int numprocs = Core::Communication::num_mpi_ranks(comm);

    for (int p = 0; p < numprocs; ++p)
    {
      switch (p)
      {
        case 0:
        {
          std::swap(receivedsize, mysize);
          std::swap(receiveddata, mydata);
          break;
        }
        default:
        {
          send_to_next_proc(p, exporter, mysize, mydata, receivedsize, receiveddata);

          break;
        }
      }

      // unpack received block
      unpack_received_block(receiveddata, collected_data);

      // the received data will be sent to the next proc
      std::swap(receivedsize, mysize);
      std::swap(receiveddata, mydata);

      // we need a new receive buffer
      receivedsize.clear();
      receiveddata.clear();
    }
  }

  template <typename T>
  static void collect_data(MPI_Comm comm, const T& my_data, T& collected_data)
  {
    Core::Communication::PackBuffer pack_data;

    add_to_pack(pack_data, my_data);

    round_robin_loop(comm, pack_data, collected_data);
  }


}  // namespace


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Data::Data()
    : isinit_(false),
      issetup_(false),
      isntmaps_filled_(false),
      ele_action_(Core::Elements::none),
      predict_type_(Inpar::Solid::pred_vague),
      ele_eval_error_flag_(Solid::Elements::ele_error_none),
      is_tolerate_errors_(false),
      total_time_(-1.0),
      delta_time_(-1.0),
      step_length_(-1.0),
      is_default_step_(true),
      timintfactor_disp_(-1.0),
      timintfactor_vel_(-1.0),
      stressdata_ptr_(nullptr),
      stressdata_postprocessed_nodal_ptr_(nullptr),
      stressdata_postprocessed_element_ptr_(nullptr),
      straindata_ptr_(nullptr),
      straindata_postprocessed_nodal_ptr_(nullptr),
      straindata_postprocessed_element_ptr_(nullptr),
      plastic_straindata_ptr_(nullptr),
      gauss_point_data_manager_ptr_(nullptr),
      sdyn_ptr_(nullptr),
      io_ptr_(nullptr),
      gstate_ptr_(nullptr),
      timint_ptr_(nullptr),
      comm_(MPI_COMM_NULL),
      beam_data_ptr_(nullptr),
      contact_data_ptr_(nullptr),
      model_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::init(
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr)
{
  sdyn_ptr_ = timint_ptr->get_data_sdyn_ptr();
  io_ptr_ = timint_ptr->get_data_io_ptr();
  gstate_ptr_ = timint_ptr->get_data_global_state_ptr();
  timint_ptr_ = timint_ptr;
  comm_ = timint_ptr->get_data_global_state().get_comm_ptr();
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::setup()
{
  check_init();

  const std::set<enum Inpar::Solid::ModelType>& mt = sdyn_ptr_->get_model_types();
  std::set<enum Inpar::Solid::ModelType>::const_iterator it;
  // setup model type specific data containers
  for (it = mt.begin(); it != mt.end(); ++it)
  {
    switch (*it)
    {
      case Inpar::Solid::model_contact:
      {
        contact_data_ptr_ = std::make_shared<ContactData>();
        contact_data_ptr_->init(Core::Utils::shared_ptr_from_ref(*this));
        contact_data_ptr_->setup();
        break;
      }
      case Inpar::Solid::model_browniandyn:
      {
        browniandyn_data_ptr_ = std::make_shared<BrownianDynData>();
        browniandyn_data_ptr_->init(Core::Utils::shared_ptr_from_ref(*this));
        browniandyn_data_ptr_->setup();
        break;
      }
      default:
      {
        // nothing to do
        break;
      }
    }
  }

  /* so far, we need the special parameter data container for beams only if
   * the applied beam elements have non-additive rotation vector DOFs */
  if (sdyn_ptr_->have_ele_tech(Inpar::Solid::EleTech::rotvec))
  {
    beam_data_ptr_ = std::make_shared<BeamData>();
    beam_data_ptr_->init();
    beam_data_ptr_->setup();
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::fill_norm_type_maps()
{
  // we have to do all this only once...
  if (isntmaps_filled_) return;

  std::set<enum NOX::Nln::StatusTest::QuantityType> qtypes;
  Solid::Nln::SOLVER::create_quantity_types(qtypes, *sdyn_ptr_);

  // --- check if the nox nln solver is active ---------------------------------
  bool isnox = false;
  std::shared_ptr<const Solid::Nln::SOLVER::Nox> nox_nln_ptr = nullptr;
  std::shared_ptr<const Solid::TimeInt::Implicit> timint_impl_ptr =
      std::dynamic_pointer_cast<const Solid::TimeInt::Implicit>(timint_ptr_);
  if (timint_impl_ptr)
  {
    nox_nln_ptr = std::dynamic_pointer_cast<const Solid::Nln::SOLVER::Nox>(
        timint_impl_ptr->get_nln_solver_ptr());
    if (nox_nln_ptr) isnox = true;
  }

  // --- get the normtypes for the different quantities -------------------------
  std::set<enum NOX::Nln::StatusTest::QuantityType>::const_iterator qiter;
  if (isnox)
  {
    const ::NOX::StatusTest::Generic& ostatus = nox_nln_ptr->get_outer_status_test();
    for (qiter = qtypes.begin(); qiter != qtypes.end(); ++qiter)
    {
      // fill the normtype_force map
      int inormtype = NOX::Nln::Aux::get_norm_type<NOX::Nln::StatusTest::NormF>(ostatus, *qiter);
      if (inormtype != -100)
        normtype_force_[*qiter] = static_cast<::NOX::Abstract::Vector::NormType>(inormtype);
      // fill the normtype_update map
      inormtype = NOX::Nln::Aux::get_norm_type<NOX::Nln::StatusTest::NormUpdate>(ostatus, *qiter);
      if (inormtype != -100)
        normtype_update_[*qiter] = static_cast<::NOX::Abstract::Vector::NormType>(inormtype);

      // check for the root mean square test (wrms)
      if (NOX::Nln::Aux::is_quantity<NOX::Nln::StatusTest::NormWRMS>(ostatus, *qiter))
      {
        /* get the absolute and relative tolerances, since we have to use them
         * during the summation. */
        double atol = NOX::Nln::Aux::get_norm_wrms_class_variable(ostatus, *qiter, "ATOL");
        if (atol < 0.0)
          FOUR_C_THROW("The absolute wrms tolerance of the quantity {} is missing.",
              NOX::Nln::StatusTest::quantity_type_to_string(*qiter));
        else
          atol_wrms_[*qiter] = atol;
        double rtol = NOX::Nln::Aux::get_norm_wrms_class_variable(ostatus, *qiter, "RTOL");
        if (rtol < 0.0)
          FOUR_C_THROW("The relative wrms tolerance of the quantity {} is missing.",
              NOX::Nln::StatusTest::quantity_type_to_string(*qiter));
        else
          rtol_wrms_[*qiter] = rtol;
      }
    }  // loop over all quantity types
  }  // if (isnox)
  else
  {
    for (qiter = qtypes.begin(); qiter != qtypes.end(); ++qiter)
    {
      normtype_force_[*qiter] = sdyn_ptr_->get_nox_norm_type();
      normtype_update_[*qiter] = sdyn_ptr_->get_nox_norm_type();
    }
  }

  // do it only once!
  isntmaps_filled_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::collect_norm_types_over_all_procs(
    const quantity_norm_type_map& normtypes) const
{
  check_init();

  const quantity_norm_type_map mynormtypes(normtypes);
  quantity_norm_type_map& gnormtypes = const_cast<quantity_norm_type_map&>(normtypes);
  collect_data(comm_, mynormtypes, gnormtypes);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Data::get_update_norm_type(
    const enum NOX::Nln::StatusTest::QuantityType& qtype,
    enum ::NOX::Abstract::Vector::NormType& normtype)
{
  fill_norm_type_maps();

  // check if there is a normtype for the corresponding quantity type
  std::map<enum NOX::Nln::StatusTest::QuantityType,
      enum ::NOX::Abstract::Vector::NormType>::const_iterator miter;
  miter = normtype_update_.find(qtype);
  if (miter == normtype_update_.end()) return false;

  // we found the corresponding type
  normtype = miter->second;

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Data::get_wrms_tolerances(
    const enum NOX::Nln::StatusTest::QuantityType& qtype, double& atol, double& rtol)
{
  fill_norm_type_maps();

  // check if there is a wrms test for the corresponding quantity type
  std::map<enum NOX::Nln::StatusTest::QuantityType, double>::const_iterator iter;
  iter = atol_wrms_.find(qtype);
  if (iter == atol_wrms_.end()) return false;

  // we found the corresponding type
  atol = iter->second;
  rtol = rtol_wrms_.at(qtype);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::sum_into_my_update_norm(
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const int& numentries,
    const double* my_update_values, const double* my_new_sol_values, const double& step_length,
    const int& owner)
{
  if (owner != Core::Communication::my_mpi_rank(comm_)) return;
  // --- standard update norms
  enum ::NOX::Abstract::Vector::NormType normtype = ::NOX::Abstract::Vector::TwoNorm;
  if (get_update_norm_type(qtype, normtype))
    sum_into_my_norm(numentries, my_update_values, normtype, step_length, my_update_norm_[qtype]);

  // --- weighted root mean square norms
  double atol = 0.0;
  double rtol = 0.0;
  if (get_wrms_tolerances(qtype, atol, rtol))
  {
    sum_into_my_relative_mean_square(atol, rtol, step_length, numentries, my_update_values,
        my_new_sol_values, my_rms_norm_[qtype]);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::sum_into_my_previous_sol_norm(
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const int& numentries,
    const double* my_old_sol_values, const int& owner)
{
  if (owner != Core::Communication::my_mpi_rank(comm_)) return;

  enum ::NOX::Abstract::Vector::NormType normtype = ::NOX::Abstract::Vector::TwoNorm;
  if (not get_update_norm_type(qtype, normtype)) return;

  sum_into_my_norm(numentries, my_old_sol_values, normtype, 1.0, my_prev_sol_norm_[qtype]);
  // update the dof counter
  my_dof_number_[qtype] += numentries;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::sum_into_my_relative_mean_square(const double& atol,
    const double& rtol, const double& step_length, const int& numentries,
    const double* my_update_values, const double* my_new_sol_values, double& my_rms) const
{
  for (int i = 0; i < numentries; ++i)
  {
    // calculate v_i = x_{i}^{k-1} = x_{i}^{k} - sl* \Delta x_{i}^{k}
    double dx_i = step_length * my_update_values[i];
    double v_i = my_new_sol_values[i] - dx_i;
    // calculate the relative mean square sum:
    // my_rms_norm = \sum_{i} [(x_i^{k}-x_{i}^{k-1}) / (RTOL * |x_{i}^{k-1}| + ATOL)]^{2}
    v_i = dx_i / (rtol * std::abs(v_i) + atol);
    my_rms += v_i * v_i;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::sum_into_my_norm(const int& numentries, const double* my_values,
    const enum ::NOX::Abstract::Vector::NormType& normtype, const double& step_length,
    double& my_norm) const
{
  switch (normtype)
  {
    case ::NOX::Abstract::Vector::OneNorm:
    {
      for (int i = 0; i < numentries; ++i) my_norm += std::abs(my_values[i] * step_length);
      break;
    }
    case ::NOX::Abstract::Vector::TwoNorm:
    {
      for (int i = 0; i < numentries; ++i)
        my_norm += (my_values[i] * my_values[i]) * (step_length * step_length);
      break;
    }
    case ::NOX::Abstract::Vector::MaxNorm:
    {
      for (int i = 0; i < numentries; ++i)
        my_norm = std::max(my_norm, std::abs(my_values[i] * step_length));
      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::reset_my_norms(const bool& isdefaultstep)
{
  check_init_setup();
  std::map<enum NOX::Nln::StatusTest::QuantityType, double>::iterator it;
  for (it = my_update_norm_.begin(); it != my_update_norm_.end(); ++it) it->second = 0.0;
  for (it = my_rms_norm_.begin(); it != my_rms_norm_.end(); ++it) it->second = 0.0;

  if (isdefaultstep)
  {
    // reset the map holding the previous solution norms of the last converged
    // Newton step
    for (it = my_prev_sol_norm_.begin(); it != my_prev_sol_norm_.end(); ++it) it->second = 0.0;
    // reset the dof number
    std::map<enum NOX::Nln::StatusTest::QuantityType, std::size_t>::iterator dit;
    for (dit = my_dof_number_.begin(); dit != my_dof_number_.end(); ++dit) dit->second = 0;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Data::is_ele_eval_error() const
{
  check_init_setup();
  return (ele_eval_error_flag_ != Solid::Elements::ele_error_none);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Data::is_predictor_state() const
{
  check_init_setup();

  const Solid::IMPLICIT::Generic* impl_ptr =
      dynamic_cast<const Solid::IMPLICIT::Generic*>(&tim_int().integrator());

  if (not impl_ptr) return false;
  return impl_ptr->is_predictor_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::DampKind Solid::ModelEvaluator::Data::get_damping_type() const
{
  check_init_setup();
  return sdyn_ptr_->get_damping_type();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>>& Solid::ModelEvaluator::Data::stress_data_ptr()
{
  check_init_setup();
  return stressdata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& Solid::ModelEvaluator::Data::stress_data() const
{
  FOUR_C_ASSERT(stressdata_ptr_, "Undefined reference to the stress data!");
  return *stressdata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>>& Solid::ModelEvaluator::Data::strain_data_ptr()
{
  check_init_setup();
  return straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& Solid::ModelEvaluator::Data::strain_data() const
{
  FOUR_C_ASSERT(straindata_ptr_, "Undefined reference to the strain data!");
  return *straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>>& Solid::ModelEvaluator::Data::plastic_strain_data_ptr()
{
  check_init_setup();
  return plastic_straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& Solid::ModelEvaluator::Data::plastic_strain_data() const
{
  FOUR_C_ASSERT(plastic_straindata_ptr_, "Undefined reference to the plastic strain data!");
  return *plastic_straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Solid::ModelEvaluator::Data::opt_quantity_data_ptr()
{
  check_init_setup();
  return opt_quantity_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& Solid::ModelEvaluator::Data::opt_quantity_data() const
{
  FOUR_C_ASSERT(opt_quantity_data_ptr_, "Undefined reference to the optional quantity data!");
  return *opt_quantity_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::StressType Solid::ModelEvaluator::Data::get_stress_output_type() const
{
  check_init_setup();
  return io_ptr_->get_stress_output_type();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::StrainType Solid::ModelEvaluator::Data::get_strain_output_type() const
{
  check_init_setup();
  return io_ptr_->get_strain_output_type();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::StrainType Solid::ModelEvaluator::Data::get_plastic_strain_output_type() const
{
  check_init_setup();
  return io_ptr_->get_plastic_strain_output_type();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::OptQuantityType Solid::ModelEvaluator::Data::get_opt_quantity_output_type() const
{
  check_init_setup();

  return io_ptr_->get_runtime_output_params()->get_structure_params()->output_optional_quantity();
}

std::shared_ptr<Solid::ModelEvaluator::GaussPointDataOutputManager>&
Solid::ModelEvaluator::Data::gauss_point_data_output_manager_ptr()
{
  check_init_setup();
  return gauss_point_data_manager_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::map<enum Solid::EnergyType, double> const& Solid::ModelEvaluator::Data::get_energy_data() const
{
  return energy_data_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::ModelEvaluator::Data::get_energy_data(enum Solid::EnergyType type) const
{
  auto check = get_energy_data().find(type);
  if (check == get_energy_data().cend())
    FOUR_C_THROW(
        "Couldn't find the energy contribution: \"{}\".", Solid::energy_type_to_string(type));

  return check->second;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::ModelEvaluator::Data::get_energy_data(const std::string type) const
{
  if (type == "total_energy")
  {
    double total_energy = 0.0;
    for (auto& energy_data : get_energy_data()) total_energy += energy_data.second;
    return total_energy;
  }
  else
    return get_energy_data(Solid::string_to_energy_type(type));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::insert_energy_type_to_be_considered(enum Solid::EnergyType type)
{
  energy_data_[type] = 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::set_value_for_energy_type(
    double value, enum Solid::EnergyType type)
{
  energy_data_[type] = value;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::clear_values_for_all_energy_types()
{
  for (auto& energy_data_iter : energy_data_)
  {
    energy_data_iter.second = 0.0;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Data::add_contribution_to_energy_type(
    const double value, const enum Solid::EnergyType type)
{
  energy_data_[type] += value;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Data::is_predictor() const { return global_state().is_predict(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Solid::ModelEvaluator::Data::get_nln_iter() const
{
  if (is_predictor()) return 0;

  bool isnox = false;
  std::shared_ptr<const Solid::Nln::SOLVER::Nox> nox_nln_ptr = nullptr;
  const Solid::TimeInt::Implicit* timint_impl_ptr =
      dynamic_cast<const Solid::TimeInt::Implicit*>(&tim_int());
  if (timint_impl_ptr != nullptr)
  {
    std::shared_ptr<const Solid::Nln::SOLVER::Generic> nlnsolver_ptr =
        timint_impl_ptr->get_nln_solver_ptr();
    /* If we are still in the setup process we return -1. This will happen
     * for the compute_mass_matrix_and_init_acc() call in dynamic simulations. */
    if (!nlnsolver_ptr) return -1;
    nox_nln_ptr = std::dynamic_pointer_cast<const Solid::Nln::SOLVER::Nox>(nlnsolver_ptr);
    if (nox_nln_ptr) isnox = true;
  }
  if (not isnox)
    FOUR_C_THROW(
        "The get_nln_iter() routine supports only the NOX::NLN "
        "framework at the moment.");

  return nox_nln_ptr->get_num_nln_iterations();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Solid::ModelEvaluator::Data::get_step_np() const { return global_state().get_step_np(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string Solid::ModelEvaluator::ContactData::get_output_file_path() const
{
  check_init();
  return in_output().get_output_ptr()->output()->file_name();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::ModelEvaluator::Data::get_restart_step() const
{
  return global_state().get_restart_step();
}

FOUR_C_NAMESPACE_CLOSE
