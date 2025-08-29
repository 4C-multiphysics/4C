// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_dyn_vreman.hpp"

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                     krank 09/13|
 *----------------------------------------------------------------------*/
FLD::Vreman::Vreman(
    std::shared_ptr<Core::FE::Discretization> actdis, Teuchos::ParameterList& params)
    :  // call constructor for "nontrivial" objects
      discret_(actdis),
      params_(params),
      physicaltype_(Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(params_, "Physical Type"))
{
  boxf_ = std::make_shared<FLD::Boxfilter>(discret_, params_);
  // Initialize Boxfilter
  boxf_->initialize_vreman();

  return;
}


/*----------------------------------------------------------------------*
 | add some scatra specific parameters                  rasthofer 08/12 |
 * ---------------------------------------------------------------------*/
void FLD::Vreman::add_scatra(std::shared_ptr<Core::FE::Discretization> scatradis)
{
  scatradiscret_ = scatradis;

  boxfsc_ = std::make_shared<FLD::Boxfilter>(scatradiscret_, params_);

  // Initialize Boxfilter
  boxfsc_->initialize_vreman_scatra(scatradiscret_);

  return;
}

/*----------------------------------------------------------------------*
 | Perform box filter operation, compare filtered quantities            |
 | to solution to get an estimate for Cv, using volume averaging        |
 |                                                             (public) |
 |                                                           krank 09/13|
 *----------------------------------------------------------------------*/
void FLD::Vreman::apply_filter_for_dynamic_computation_of_cv(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velocity,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalar, const double thermpress,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle)
{
  const Core::LinAlg::Map* nodecolmap = discret_->node_col_map();


  col_filtered_strainrate_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 9, true);
  col_filtered_alphaij_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 9, true);
  col_filtered_expression_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  col_filtered_alpha2_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);


  // perform filtering
  boxf_->apply_filter(velocity, scalar, thermpress, dirichtoggle);

  // get fitered fields
  boxf_->get_filtered_vreman_strainrate(*col_filtered_strainrate_);
  boxf_->get_filtered_vreman_alphaij(*col_filtered_alphaij_);
  boxf_->get_expression(*col_filtered_expression_);
  boxf_->get_alpha2(*col_filtered_alpha2_);

  // compute Cv
  Cv_ = dyn_vreman_compute_cv();

  return;
}

void FLD::Vreman::apply_filter_for_dynamic_computation_of_dt(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalar, const double thermpress,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle,
    Teuchos::ParameterList& extraparams, const int ndsvel)
{
  const Core::LinAlg::Map* nodecolmap = scatradiscret_->node_col_map();

  col_filtered_phi_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
  col_filtered_phi2_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  col_filtered_phiexpression_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  col_filtered_alphaijsc_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 9, true);


  // perform filtering
  boxfsc_->apply_filter_scatra(scalar, thermpress, dirichtoggle, ndsvel);
  boxfsc_->get_filtered_phi(*col_filtered_phi_);
  boxfsc_->get_filtered_phi2(*col_filtered_phi2_);
  boxfsc_->get_filtered_phiexpression(*col_filtered_phiexpression_);
  boxfsc_->get_filtered_vreman_alphaijsc(*col_filtered_alphaijsc_);
  dyn_vreman_compute_dt(extraparams);
  return;
}



/*----------------------------------------------------------------------*
 | compute Cv from filtered quantities.                       (private) |
 |                                                      krank     09/13 |
 *----------------------------------------------------------------------*/
double FLD::Vreman::dyn_vreman_compute_cv()
{
  double Cv = 0.0;
  double cv_numerator_volumeav = 0.0;
  double cv_denominator_volumeav = 0.0;
  int id = Global::Problem::instance()->materials()->first_id_by_type(Core::Materials::m_fluid);
  const Core::Mat::PAR::Parameter* mat =
      Global::Problem::instance()->materials()->parameter_by_id(id);
  const Mat::PAR::NewtonianFluid* actmat = static_cast<const Mat::PAR::NewtonianFluid*>(mat);
  double dens = actmat->density_;
  double visc = actmat->viscosity_ / dens;
  // action for elements
  // generate a parameterlist for communication and control
  Teuchos::ParameterList calc_vreman_params;
  calc_vreman_params.set<FLD::Action>("action", FLD::calc_vreman_const);

  // hand filtered global vectors down to the element
  calc_vreman_params.set("col_filtered_strainrate", col_filtered_strainrate_);
  calc_vreman_params.set("col_filtered_alphaij", col_filtered_alphaij_);
  calc_vreman_params.set("col_filtered_alpha2", col_filtered_alpha2_);
  calc_vreman_params.set("col_filtered_expression", col_filtered_expression_);


  // loop all elements on this proc (excluding ghosted ones)
  std::shared_ptr<Core::LinAlg::SerialDenseVector> Cv_num_denom =
      std::make_shared<Core::LinAlg::SerialDenseVector>(2);


  // call loop over elements (assemble nothing)
  discret_->evaluate_scalars(calc_vreman_params, *Cv_num_denom);
  discret_->clear_state();


  cv_numerator_volumeav = (*Cv_num_denom)[0];
  cv_denominator_volumeav = (*Cv_num_denom)[1];
  // multiply with viscosity
  if (sqrt(cv_denominator_volumeav * cv_denominator_volumeav) <
      1.0e-12)  // the denominator might also become negative
    Cv = 0.0;   // constant vreman
  else
    Cv = (-1.0) * visc / 2.0 * cv_numerator_volumeav / cv_denominator_volumeav;
  params_.set<double>("C_vreman_theoretical", Cv);
  if (Cv < 0.0)
  {
    Cv = 0.0;
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      std::cout << "!!   Vreman constant negative --> clipping: Cv=0.0   !!" << std::endl;
  }

  // std::cout <<"Vreman constant:   "<< Cv << std::endl;
  params_.set<double>("C_vreman", Cv);



  return Cv;
}  // end FLD::Vreman::dyn_vreman_compute_cv

void FLD::Vreman::dyn_vreman_compute_dt(Teuchos::ParameterList& extraparams)
{
  double Dt = 0.0;
  double dt_numerator_volumeav = 0.0;
  double dt_denominator_volumeav = 0.0;
  int idscatra =
      Global::Problem::instance()->materials()->first_id_by_type(Core::Materials::m_scatra);
  const Core::Mat::PAR::Parameter* matscatra =
      Global::Problem::instance()->materials()->parameter_by_id(idscatra);
  const Mat::PAR::ScatraMat* actmatscatra = static_cast<const Mat::PAR::ScatraMat*>(matscatra);
  double diffus = Mat::PAR::ScatraMat(*actmatscatra)
                      .get_parameter(actmatscatra->diff, -1);  // actmatscatra->diffusivity_;

  // generate a parameterlist for communication and control
  Teuchos::ParameterList calc_vreman_params_scatra;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_vreman_scatra, calc_vreman_params_scatra);
  calc_vreman_params_scatra.set("col_filtered_phi", col_filtered_phi_);
  calc_vreman_params_scatra.set("col_filtered_phi2", col_filtered_phi2_);
  calc_vreman_params_scatra.set("col_filtered_phiexpression", col_filtered_phiexpression_);
  calc_vreman_params_scatra.set("col_filtered_alphaijsc", col_filtered_alphaijsc_);
  // loop all elements on this proc (excluding ghosted ones)
  std::shared_ptr<Core::LinAlg::SerialDenseVector> Dt_num_denom =
      std::make_shared<Core::LinAlg::SerialDenseVector>(2);
  // call loop over elements (assemble nothing)
  scatradiscret_->evaluate_scalars(calc_vreman_params_scatra, *Dt_num_denom);
  scatradiscret_->clear_state();
  dt_numerator_volumeav = (*Dt_num_denom)[0];
  dt_denominator_volumeav = (*Dt_num_denom)[1];
  if (sqrt(dt_denominator_volumeav * dt_denominator_volumeav) <
      1.0e-12)  // the denominator might also become negative
    Dt = 0.0;   // constant vreman
  else
    Dt = 1.0 / diffus * dt_numerator_volumeav / dt_denominator_volumeav;

  // remark:
  // Dt does not contain Cv, since Cv cancels out during the calculation of the subgrid diffusivity

  Teuchos::ParameterList* modelparams_scatra = &(extraparams.sublist("TURBULENCE MODEL"));

  modelparams_scatra->set<double>("Dt_vreman", Dt);
  params_.set<double>("Dt_vreman", Dt);
  return;
}  // end FLD::Vreman::dyn_vreman_compute_dt()

FOUR_C_NAMESPACE_CLOSE
