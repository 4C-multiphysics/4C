// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_dyn_smag.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                     gammi 09/08|
 *----------------------------------------------------------------------*/
FLD::DynSmagFilter::DynSmagFilter(
    std::shared_ptr<Core::FE::Discretization> actdis, Teuchos::ParameterList& params)
    :  // call constructor for "nontrivial" objects
      discret_(actdis),
      params_(params),
      physicaltype_(Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(params_, "Physical Type"))
{
  // the default is do nothing
  apply_dynamic_smagorinsky_ = false;
  homdir_ = false;
  special_flow_homdir_ = "not_specified";
  calc_ci_ = false;

  // -------------------------------------------------------------------
  // initialise the turbulence model
  // -------------------------------------------------------------------
  Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));

  if (modelparams->get<std::string>("TURBULENCE_APPROACH", "DNS_OR_RESVMM_LES") == "CLASSICAL_LES")
  {
    if (modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") == "Dynamic_Smagorinsky")
    {
      apply_dynamic_smagorinsky_ = true;

      // check, if averaging is desired
      if (params_.sublist("SUBGRID VISCOSITY").get<bool>("C_SMAGORINSKY_AVERAGED"))
      {
        if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        {
          std::cout << "------->  Prepare averaging of Smagorinsky constant ..." << std::endl;
          std::cout << "------->  Caution: works only for cartesian meshes!" << std::endl;
        }
        // for homogeneous directions we can perform an averaging
        if (modelparams->get<std::string>("HOMDIR", "not_specified") != "not_specified")
        {
          homdir_ = true;
          special_flow_homdir_ = modelparams->get<std::string>("HOMDIR", "not_specified");
        }
        else
          FOUR_C_THROW("Expected homogeneous direction!");
        if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        {
          std::cout << "------->  Homogeneous direction(s): " << special_flow_homdir_ << std::endl;
        }
      }
      else
      {
        if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        {
          std::cout << "------->  No averaging of Smagorinsky constant ..." << std::endl;
          std::cout << "------->  Point-wise clipping!" << std::endl;
        }
      }

      // check whether we would like to include a model for the isotropic part
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        if (params_.sublist("SUBGRID VISCOSITY").get<bool>("C_INCLUDE_CI"))
        {
          if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
            std::cout << "------->  Ci is included for loma problem" << std::endl;
          if (params_.sublist("SUBGRID VISCOSITY").get<double>("C_YOSHIZAWA") < 0.0)
          {
            if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
              std::cout << "------->  Ci is determined dynamically" << std::endl;

            calc_ci_ = true;
          }
          else
          {
            if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
              std::cout << "------->  Ci is set to "
                        << params_.sublist("SUBGRID VISCOSITY").get<double>("C_YOSHIZAWA")
                        << std::endl;

            calc_ci_ = false;
          }
        }
        else
        {
          if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
            std::cout << "------->  Ci is not included for loma problem" << std::endl;

          calc_ci_ = false;

          if (params_.sublist("SUBGRID VISCOSITY").get<double>("C_YOSHIZAWA") > 0.0)
            FOUR_C_THROW(
                "Set C_YOSHIZAWA < 0.0 in combination with C_SMAGORINSKY_AVERAGED==true and "
                "C_INCLUDE_CI==false!");
        }
      }
    }
  }

  boxf_ = std::make_shared<FLD::Boxfilter>(discret_, params_);

  return;
}



/*----------------------------------------------------------------------*
 | add some scatra specific parameters                  rasthofer 08/12 |
 * ---------------------------------------------------------------------*/
void FLD::DynSmagFilter::add_scatra(std::shared_ptr<Core::FE::Discretization> scatradis)
{
  scatradiscret_ = scatradis;

  boxfsc_ = std::make_shared<FLD::Boxfilter>(scatradiscret_, params_);
  boxfsc_->add_scatra(scatradiscret_);

  return;
}


/*----------------------------------------------------------------------*
 | Perform box filter operation, compare filtered quantities            |
 | to solution to get an estimate for Cs, average over element layers   |
 | or do clipping                                              (public) |
 |                                                           gammi 09/08|
 *----------------------------------------------------------------------*/
void FLD::DynSmagFilter::apply_filter_for_dynamic_computation_of_cs(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velocity,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalar, const double thermpress,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle)
{
  const Core::LinAlg::Map* nodecolmap = discret_->node_col_map();


  // perform filtering
  boxf_->apply_filter(velocity, scalar, thermpress, dirichtoggle);

  if (apply_dynamic_smagorinsky_)
  {
    col_filtered_vel_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
    col_filtered_reynoldsstress_ =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 9, true);
    col_filtered_modeled_subgrid_stress_ =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 9, true);
    boxf_->get_filtered_velocity(*col_filtered_vel_);
    boxf_->get_filtered_reynolds_stress(*col_filtered_reynoldsstress_);
    boxf_->get_filtered_modeled_subgrid_stress(*col_filtered_modeled_subgrid_stress_);

    if (physicaltype_ == Inpar::FLUID::loma)
    {
      col_filtered_dens_vel_ =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
      col_filtered_dens_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
      col_filtered_dens_strainrate_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
      boxf_->get_filtered_dens_velocity(*col_filtered_dens_vel_);
      boxf_->get_density(*col_filtered_dens_);
      boxf_->get_density_strainrate(*col_filtered_dens_strainrate_);
    }
  }



  // compute Cs, use averaging or clipping
  dyn_smag_compute_cs();

  // output of mean dynamic Samgorinsky parameters
  // reset to zero
  // turbulent channel flow only
  Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
  if (modelparams->get<std::string>("CANONICAL_FLOW", "no") == "channel_flow_of_height_2" or
      modelparams->get<std::string>("CANONICAL_FLOW", "no") == "loma_channel_flow_of_height_2" or
      modelparams->get<std::string>("CANONICAL_FLOW", "no") == "scatra_channel_flow_of_height_2")
  {
    size_t nlayer =
        (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_sum")).size();
    for (size_t rr = 0; rr < nlayer; rr++)
    {
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_sum"))[rr] = 0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_delta_sq_sum"))[rr] = 0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_visceff_sum"))[rr] = 0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Ci_sum"))[rr] = 0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Ci_delta_sq_sum"))[rr] = 0.0;
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | Perform box filter operation, compare filtered quantities            |
 | to solution to get an estimate for Prt, average over element layers  |
 | or do clipping                                              (public) |
 |                                                       rasthofer 08/12|
 *----------------------------------------------------------------------*/
void FLD::DynSmagFilter::apply_filter_for_dynamic_computation_of_prt(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalar, const double thermpress,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle,
    Teuchos::ParameterList& extraparams, const int ndsvel)
{
  const Core::LinAlg::Map* nodecolmap = scatradiscret_->node_col_map();

  // perform filtering
  boxfsc_->apply_filter_scatra(scalar, thermpress, dirichtoggle, ndsvel);
  col_filtered_vel_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
  col_filtered_dens_vel_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
  col_filtered_dens_vel_temp_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
  col_filtered_dens_rateofstrain_temp_ =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, 3, true);
  col_filtered_temp_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  col_filtered_dens_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  col_filtered_dens_temp_ = std::make_shared<Core::LinAlg::Vector<double>>(*nodecolmap, true);
  boxfsc_->get_filtered_velocity(*col_filtered_vel_);
  boxfsc_->get_filtered_dens_velocity(*col_filtered_dens_vel_);
  boxfsc_->get_filtered_dens_velocity_temp(*col_filtered_dens_vel_temp_);
  boxfsc_->get_filtered_dens_rateofstrain_temp(*col_filtered_dens_rateofstrain_temp_);
  boxfsc_->get_temp(*col_filtered_temp_);
  boxfsc_->get_density(*col_filtered_dens_);
  boxfsc_->get_dens_temp(*col_filtered_dens_temp_);
  // number of elements per layer
  // required for calculation of mean Prt in turbulent channel flow
  int numele_layer = 0;
  // compute Cs, use averaging or clipping
  dyn_smag_compute_prt(extraparams, numele_layer);

  Teuchos::ParameterList* extramodelparams = &(extraparams.sublist("TURBULENCE MODEL"));
  Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));

  // add pointer to variables of statistics manager
  // output of mean dynamic Samgorinsky parameters
  // reset to zero first
  if (modelparams->get<std::string>("CANONICAL_FLOW", "no") == "channel_flow_of_height_2" or
      modelparams->get<std::string>("CANONICAL_FLOW", "no") == "loma_channel_flow_of_height_2" or
      modelparams->get<std::string>("CANONICAL_FLOW", "no") == "scatra_channel_flow_of_height_2")
  {
    size_t nlayer =
        (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Prt_sum")).size();
    for (size_t rr = 0; rr < nlayer; rr++)
    {
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Prt_sum"))[rr] = 0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_delta_sq_Prt_sum"))[rr] =
          0.0;
      (*modelparams->get<std::shared_ptr<std::vector<double>>>("local_diffeff_sum"))[rr] = 0.0;
    }
    extramodelparams->set<std::shared_ptr<std::vector<double>>>(
        "local_Prt_sum", modelparams->get<std::shared_ptr<std::vector<double>>>("local_Prt_sum"));
    extramodelparams->set<std::shared_ptr<std::vector<double>>>("local_Cs_delta_sq_Prt_sum",
        modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_delta_sq_Prt_sum"));
    extramodelparams->set<std::shared_ptr<std::vector<double>>>("local_diffeff_sum",
        modelparams->get<std::shared_ptr<std::vector<double>>>("local_diffeff_sum"));
    // add (Cs*h)^2 to calculate Prt
    // therefore, it is assumed that finally the scatra field is solved after the fluid fields
    // be careful since this vector has not yet been commuicated
    std::shared_ptr<std::vector<double>> local_Cs_delta_sq_sum =
        modelparams->get<std::shared_ptr<std::vector<double>>>("local_Cs_delta_sq_sum");
    std::shared_ptr<std::vector<double>> global_Cs_delta_sq_sum;
    global_Cs_delta_sq_sum = std::make_shared<std::vector<double>>(nlayer, 0.0);
    *global_Cs_delta_sq_sum =
        Core::Communication::sum_all(*local_Cs_delta_sq_sum, discret_->get_comm());
    extramodelparams->set<std::shared_ptr<std::vector<double>>>(
        "global_Cs_delta_sq_sum", global_Cs_delta_sq_sum);
    extramodelparams->set<int>("numele_layer", numele_layer);
  }

  return;
}


/*----------------------------------------------------------------------*
 | compute Cs from filtered quantities. If possible, use in plane       |
 | averaging                                                  (private) |
 |                                                      rasthofer 02/11 |
 *----------------------------------------------------------------------*/
void FLD::DynSmagFilter::dyn_smag_compute_cs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ComputeCs");

  // for special flows, LijMij and MijMij averaged in each
  // home. direction
  int numlayers = 0;

  std::shared_ptr<std::vector<double>> averaged_LijMij = std::make_shared<std::vector<double>>();
  std::shared_ptr<std::vector<double>> averaged_MijMij = std::make_shared<std::vector<double>>();

  // additional averaged quantities for extension to variable-density flow at low-Mach number
  // quantities to estimate CI
  std::shared_ptr<std::vector<double>> averaged_CI_numerator =
      std::make_shared<std::vector<double>>();
  std::shared_ptr<std::vector<double>> averaged_CI_denominator =
      std::make_shared<std::vector<double>>();

  std::vector<int> count_for_average;
  std::vector<int> local_count_for_average;

  std::vector<double> local_ele_sum_LijMij;
  std::vector<double> local_ele_sum_MijMij;
  std::vector<double> local_ele_sum_CI_numerator;
  std::vector<double> local_ele_sum_CI_denominator;

  // final constants (Cs*delta)^2 and (Ci*delta)^2 (loma only)
  const Core::LinAlg::Map* elerowmap = discret_->element_row_map();
  Core::LinAlg::Vector<double> Cs_delta_sq(*elerowmap, true);
  Core::LinAlg::Vector<double> Ci_delta_sq(*elerowmap, true);

  if (homdir_)
  {
    if (special_flow_homdir_ == "xyz")
    {
      numlayers = 1;
    }
    else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
             special_flow_homdir_ == "yz")
    {
      // get planecoordinates
      Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
      dir1coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("planecoords_", nullptr);

      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates of planes for in plane averaging");
      }
      else if ((*dir1coords_).size() < 2)
      {
        FOUR_C_THROW("no planes for averaging are available");
      }

      numlayers = (*dir1coords_).size() - 1;
    }
    else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
             special_flow_homdir_ == "z")
    {
      // get coordinates
      Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
      dir1coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("dir1coords_", nullptr);
      dir2coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("dir2coords_", nullptr);

      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates 1 for averaging");
      }
      else if ((*dir2coords_).size() < 2)
      {
        FOUR_C_THROW("no coordinates 1 for averaging are available");
      }
      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates 2 for averaging");
      }
      else if ((*dir2coords_).size() < 2)
      {
        FOUR_C_THROW("no coordinates 2 for averaging are available");
      }

      numlayers = ((*dir1coords_).size() - 1) * ((*dir2coords_).size() - 1);
    }
    else
      FOUR_C_THROW("Homogeneous directions not supported!");

    count_for_average.resize(numlayers);
    local_count_for_average.resize(numlayers);

    local_ele_sum_LijMij.resize(numlayers);
    local_ele_sum_MijMij.resize(numlayers);
    local_ele_sum_CI_numerator.resize(numlayers);
    local_ele_sum_CI_denominator.resize(numlayers);

    (*averaged_LijMij).resize(numlayers);
    (*averaged_MijMij).resize(numlayers);
    (*averaged_CI_numerator).resize(numlayers);
    (*averaged_CI_denominator).resize(numlayers);

    for (int rr = 0; rr < numlayers; ++rr)
    {
      (*averaged_LijMij)[rr] = 0.0;
      (*averaged_MijMij)[rr] = 0.0;
      (*averaged_CI_numerator)[rr] = 0.0;
      (*averaged_CI_denominator)[rr] = 0.0;
      local_ele_sum_LijMij[rr] = 0.0;
      local_ele_sum_MijMij[rr] = 0.0;
      local_ele_sum_CI_numerator[rr] = 0.0;
      local_ele_sum_CI_denominator[rr] = 0.0;
      count_for_average[rr] = 0;
      local_count_for_average[rr] = 0;
    }
  }

  // ----------------------------------------------------
  // compute Cs

  // generate a parameterlist for communication and control
  Teuchos::ParameterList calc_smag_const_params;
  // action for elements
  calc_smag_const_params.set<FLD::Action>("action", FLD::calc_smagorinsky_const);

  // hand filtered global vectors down to the element
  calc_smag_const_params.set("col_filtered_vel", col_filtered_vel_);
  calc_smag_const_params.set("col_filtered_reynoldsstress", col_filtered_reynoldsstress_);
  calc_smag_const_params.set(
      "col_filtered_modeled_subgrid_stress", col_filtered_modeled_subgrid_stress_);

  if (physicaltype_ == Inpar::FLUID::loma)
  {
    calc_smag_const_params.set("col_filtered_dens", col_filtered_dens_);
    calc_smag_const_params.set("col_filtered_dens_vel", col_filtered_dens_vel_);
    calc_smag_const_params.set("col_filtered_dens_strainrate", col_filtered_dens_strainrate_);
  }

  // dummy matrices and vectors for element call
  Core::LinAlg::SerialDenseMatrix dummym1;
  Core::LinAlg::SerialDenseMatrix dummym2;
  Core::LinAlg::SerialDenseVector dummyv1;
  Core::LinAlg::SerialDenseVector dummyv2;
  Core::LinAlg::SerialDenseVector dummyv3;

  // loop all elements on this proc (excluding ghosted ones)
  for (int nele = 0; nele < discret_->num_my_row_elements(); ++nele)
  {
    // get the element
    Core::Elements::Element* ele = discret_->l_row_element(nele);

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    ele->location_vector(*discret_, lm, lmowner, lmstride);

    // call the element evaluate method to integrate functions
    // against heaviside function element
    int err = ele->evaluate(
        calc_smag_const_params, *discret_, lm, dummym1, dummym2, dummyv1, dummyv2, dummyv3);
    if (err)
      FOUR_C_THROW("Proc {}: Element {} returned err={}",
          Core::Communication::my_mpi_rank(discret_->get_comm()), ele->id(), err);

    // get turbulent Cs and Ci of this element
    double ele_Cs_delta_sq = calc_smag_const_params.get<double>("ele_Cs_delta_sq");
    double ele_Ci_delta_sq = calc_smag_const_params.get<double>("ele_Ci_delta_sq");
    // and store it in vector
    const int id = ele->id();
    Cs_delta_sq.replace_global_values(1, &ele_Cs_delta_sq, &id);
    Ci_delta_sq.replace_global_values(1, &ele_Ci_delta_sq, &id);


    // local contributions to in plane averaging for channel flows
    if (homdir_)
    {
      // get the result from the element call
      double LijMij = calc_smag_const_params.get<double>("LijMij");
      double MijMij = calc_smag_const_params.get<double>("MijMij");
      double CI_numerator = 0.0;
      double CI_denominator = 0.0;
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        CI_numerator = calc_smag_const_params.get<double>("CI_numerator");
        CI_denominator = calc_smag_const_params.get<double>("CI_denominator");
      }

      // add result into result vector

      int nlayer = 0;
      if (special_flow_homdir_ == "xyz")
      {
        nlayer = 0;
      }
      else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
               special_flow_homdir_ == "yz")
      {
        // get center
        double center = 0.0;
        if (special_flow_homdir_ == "xy")
          center = calc_smag_const_params.get<double>("zcenter");
        else if (special_flow_homdir_ == "xz")
          center = calc_smag_const_params.get<double>("ycenter");
        else if (special_flow_homdir_ == "yz")
          center = calc_smag_const_params.get<double>("xcenter");

        // for this purpose, determine the layer (the plane for average)
        bool found = false;
        for (nlayer = 0; nlayer < (int)(*dir1coords_).size() - 1;)
        {
          if (center < (*dir1coords_)[nlayer + 1])
          {
            found = true;
            break;
          }
          nlayer++;
        }
        if (found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }
      }
      else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
               special_flow_homdir_ == "z")
      {
        // get center
        double dim1_center = 0.0;
        double dim2_center = 0.0;
        if (special_flow_homdir_ == "x")
        {
          dim1_center = calc_smag_const_params.get<double>("ycenter");
          dim2_center = calc_smag_const_params.get<double>("zcenter");
        }
        else if (special_flow_homdir_ == "y")
        {
          dim1_center = calc_smag_const_params.get<double>("xcenter");
          dim2_center = calc_smag_const_params.get<double>("zcenter");
        }
        else if (special_flow_homdir_ == "z")
        {
          dim1_center = calc_smag_const_params.get<double>("xcenter");
          dim2_center = calc_smag_const_params.get<double>("ycenter");
        }

        // for this purpose, determine the layer (the direction for average)
        int n1layer = 0;
        int n2layer = 0;
        bool dir1found = false;
        bool dir2found = false;
        for (n1layer = 0; n1layer < (int)(*dir1coords_).size() - 1;)
        {
          if (dim1_center < (*dir1coords_)[n1layer + 1])
          {
            dir1found = true;
            break;
          }
          n1layer++;
        }
        if (dir1found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }
        for (n2layer = 0; n2layer < (int)(*dir2coords_).size() - 1;)
        {
          if (dim2_center < (*dir2coords_)[n2layer + 1])
          {
            dir2found = true;
            break;
          }
          n2layer++;
        }
        if (dir2found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }

        const int numdir1layer = (int)(*dir1coords_).size() - 1;
        nlayer = numdir1layer * n2layer + n1layer;
      }
      else
        FOUR_C_THROW("Homogeneous directions not supported!");

      // add it up
      local_ele_sum_LijMij[nlayer] += LijMij;
      local_ele_sum_MijMij[nlayer] += MijMij;
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        local_ele_sum_CI_numerator[nlayer] += CI_numerator;
        local_ele_sum_CI_denominator[nlayer] += CI_denominator;
      }

      local_count_for_average[nlayer]++;
    }  // end add element contribution to layer averaging for channel flows

  }  // end loop over elements

  // export from row to column map
  const Core::LinAlg::Map* elecolmap = discret_->element_col_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> col_Cs_delta_sq =
      std::make_shared<Core::LinAlg::Vector<double>>(*elecolmap, true);
  col_Cs_delta_sq->put_scalar(0.0);

  Core::LinAlg::export_to(Cs_delta_sq, *col_Cs_delta_sq);
  std::shared_ptr<Core::LinAlg::Vector<double>> col_Ci_delta_sq =
      std::make_shared<Core::LinAlg::Vector<double>>(*elecolmap, true);
  col_Ci_delta_sq->put_scalar(0.0);

  Core::LinAlg::export_to(Ci_delta_sq, *col_Ci_delta_sq);

  // store in parameters
  Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
  modelparams->set<std::shared_ptr<Core::LinAlg::Vector<double>>>(
      "col_Cs_delta_sq", col_Cs_delta_sq);
  modelparams->set<std::shared_ptr<Core::LinAlg::Vector<double>>>(
      "col_Ci_delta_sq", col_Ci_delta_sq);

  // ----------------------------------------------------
  // global in plane averaging of quantities for
  // turbulent channel flow

  if (homdir_)
  {
    // now add all the stuff from the different processors

    for (int rr = 0; rr < numlayers; ++rr)
    {
      (count_for_average[rr]) =
          Core::Communication::sum_all((local_count_for_average[rr]), discret_->get_comm());
      ((*averaged_LijMij)[rr]) =
          Core::Communication::sum_all((local_ele_sum_LijMij[rr]), discret_->get_comm());
      ((*averaged_MijMij)[rr]) =
          Core::Communication::sum_all((local_ele_sum_MijMij[rr]), discret_->get_comm());
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        ((*averaged_CI_numerator)[rr]) =
            Core::Communication::sum_all((local_ele_sum_CI_numerator[rr]), discret_->get_comm());
        ((*averaged_CI_denominator)[rr]) =
            Core::Communication::sum_all((local_ele_sum_CI_denominator[rr]), discret_->get_comm());
      }
    }

    // do averaging
    for (int rr = 0; rr < numlayers; ++rr)
    {
      // perform some checks first
      if (count_for_average[rr] == 0 and
          ((*averaged_LijMij)[rr] != 0.0 or (*averaged_MijMij)[rr] != 0.0))
        FOUR_C_THROW("Expected 'averaged_LijMij' or 'averaged_MijMij' equal zero!");
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        if (count_for_average[rr] == 0 and
            ((*averaged_CI_numerator)[rr] != 0.0 or (*averaged_CI_denominator)[rr] != 0.0))
          FOUR_C_THROW("Expected 'averaged_CI_numerator' or 'averaged_CI_denominator' equal zero!");
      }

      // calculate averaged quantities
      // we have to exclude zero here, since, for backward-facing steps, the step is contained and
      // we will obtain a zero block there
      if (count_for_average[rr] != 0)
      {
        (*averaged_LijMij)[rr] /= count_for_average[rr];
        (*averaged_MijMij)[rr] /= count_for_average[rr];
        if (physicaltype_ == Inpar::FLUID::loma)
        {
          (*averaged_CI_numerator)[rr] /= count_for_average[rr];
          (*averaged_CI_denominator)[rr] /= count_for_average[rr];
        }
      }
    }
    // provide necessary information for the elements
    {
      modelparams->set<std::shared_ptr<std::vector<double>>>("averaged_LijMij_", averaged_LijMij);
      modelparams->set<std::shared_ptr<std::vector<double>>>("averaged_MijMij_", averaged_MijMij);
      if (physicaltype_ == Inpar::FLUID::loma)
      {
        modelparams->set<std::shared_ptr<std::vector<double>>>(
            "averaged_CI_numerator_", averaged_CI_numerator);
        modelparams->set<std::shared_ptr<std::vector<double>>>(
            "averaged_CI_denominator_", averaged_CI_denominator);
      }
      if (special_flow_homdir_ == "xyz")
      {
        // nothing to do
      }
      else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
               special_flow_homdir_ == "yz")
      {
        modelparams->set<std::shared_ptr<std::vector<double>>>("planecoords_", dir1coords_);
      }
      else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
               special_flow_homdir_ == "z")
      {
        modelparams->set<std::shared_ptr<std::vector<double>>>("dir1coords_", dir1coords_);
        modelparams->set<std::shared_ptr<std::vector<double>>>("dir2coords_", dir2coords_);
      }
      else
        FOUR_C_THROW("More than two homogeneous directions not supported!");
    }
  }  // end if home dir


  return;
}  // end FLD::DynSmagFilter::dyn_smag_compute_cs


/*----------------------------------------------------------------------*
 | compute Prt from filtered quantities. If possible, use in plane      |
 | averaging                                                  (private) |
 |                                                      rasthofer 09/12 |
 *----------------------------------------------------------------------*/
void FLD::DynSmagFilter::dyn_smag_compute_prt(
    Teuchos::ParameterList& extraparams, int& numele_layer)
{
  TEUCHOS_FUNC_TIME_MONITOR("ComputePrt");

  const Core::LinAlg::Map* elerowmap = scatradiscret_->element_row_map();
  Core::LinAlg::Vector<double> Prt(*elerowmap, true);

  // for special flows, LijMij and MijMij averaged in each
  // home. direction
  int numlayers = 0;

  std::shared_ptr<std::vector<double>> averaged_LkMk = std::make_shared<std::vector<double>>();
  std::shared_ptr<std::vector<double>> averaged_MkMk = std::make_shared<std::vector<double>>();

  std::vector<int> count_for_average;
  std::vector<int> local_count_for_average;

  std::vector<double> local_ele_sum_LkMk;
  std::vector<double> local_ele_sum_MkMk;

  if (homdir_)
  {
    if (special_flow_homdir_ == "xyz")
    {
      numlayers = 1;
    }
    else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
             special_flow_homdir_ == "yz")
    {
      // get planecoordinates
      Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
      dir1coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("planecoords_", nullptr);

      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates of planes for in plane averaging");
      }
      else if ((*dir1coords_).size() < 2)
      {
        FOUR_C_THROW("no planes for averaging are available");
      }

      numlayers = (*dir1coords_).size() - 1;
    }
    else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
             special_flow_homdir_ == "z")
    {
      // get coordinates
      Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
      dir1coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("dir1coords_", nullptr);
      dir2coords_ = modelparams->get<std::shared_ptr<std::vector<double>>>("dir2coords_", nullptr);

      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates 1 for averaging");
      }
      else if ((*dir2coords_).size() < 2)
      {
        FOUR_C_THROW("no coordinates 1 for averaging are available");
      }
      if (dir1coords_ == nullptr)
      {
        FOUR_C_THROW("need the coordinates 2 for averaging");
      }
      else if ((*dir2coords_).size() < 2)
      {
        FOUR_C_THROW("no coordinates 2 for averaging are available");
      }

      numlayers = ((*dir1coords_).size() - 1) * ((*dir2coords_).size() - 1);
    }
    else
      FOUR_C_THROW("Homogeneous directions not supported!");

    count_for_average.resize(numlayers);
    local_count_for_average.resize(numlayers);

    local_ele_sum_LkMk.resize(numlayers);
    local_ele_sum_MkMk.resize(numlayers);

    (*averaged_LkMk).resize(numlayers);
    (*averaged_MkMk).resize(numlayers);

    for (int rr = 0; rr < numlayers; ++rr)
    {
      (*averaged_LkMk)[rr] = 0.0;
      (*averaged_MkMk)[rr] = 0.0;
      local_ele_sum_LkMk[rr] = 0.0;
      local_ele_sum_MkMk[rr] = 0.0;
      count_for_average[rr] = 0;
      local_count_for_average[rr] = 0;
    }
  }

  // ----------------------------------------------------
  // compute Prt

  // generate a parameterlist for communication and control
  Teuchos::ParameterList calc_turb_prandtl_params;
  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_turbulent_prandtl_number, calc_turb_prandtl_params);

  // hand filtered global vectors down to the element
  calc_turb_prandtl_params.set("col_filtered_vel", col_filtered_vel_);
  calc_turb_prandtl_params.set("col_filtered_dens_vel", col_filtered_dens_vel_);
  calc_turb_prandtl_params.set("col_filtered_dens_vel_temp", col_filtered_dens_vel_temp_);
  calc_turb_prandtl_params.set(
      "col_filtered_dens_rateofstrain_temp", col_filtered_dens_rateofstrain_temp_);
  calc_turb_prandtl_params.set("col_filtered_temp", col_filtered_temp_);
  calc_turb_prandtl_params.set("col_filtered_dens", col_filtered_dens_);
  calc_turb_prandtl_params.set("col_filtered_dens_temp", col_filtered_dens_temp_);

  // dummy matrices and vectors for element call
  Core::LinAlg::SerialDenseMatrix dummym1;
  Core::LinAlg::SerialDenseMatrix dummym2;
  Core::LinAlg::SerialDenseVector dummyv1;
  Core::LinAlg::SerialDenseVector dummyv2;
  Core::LinAlg::SerialDenseVector dummyv3;

  // loop all elements on this proc (excluding ghosted ones)
  for (int nele = 0; nele < scatradiscret_->num_my_row_elements(); ++nele)
  {
    // get the element
    Core::Elements::Element* ele = scatradiscret_->l_row_element(nele);

    // get element location vector, dirichlet flags and ownerships
    Core::Elements::LocationArray la(scatradiscret_->num_dof_sets());
    ele->location_vector(*scatradiscret_, la);

    // call the element evaluate method to integrate functions
    // against heaviside function element
    int err = ele->evaluate(
        calc_turb_prandtl_params, *scatradiscret_, la, dummym1, dummym2, dummyv1, dummyv2, dummyv3);
    if (err)
      FOUR_C_THROW("Proc {}: Element {} returned err={}",
          Core::Communication::my_mpi_rank(scatradiscret_->get_comm()), ele->id(), err);

    // get turbulent Prandlt number of this element
    double ele_Prt = calc_turb_prandtl_params.get<double>("ele_Prt");
    // and store it in vector
    const int id = ele->id();
    Prt.replace_global_values(1, &ele_Prt, &id);

    // local contributions to in plane averaging for channel flows
    if (homdir_)
    {
      // get the result from the element call
      double LkMk = calc_turb_prandtl_params.get<double>("LkMk");
      double MkMk = calc_turb_prandtl_params.get<double>("MkMk");

      // add result into result vector

      int nlayer = 0;
      if (special_flow_homdir_ == "xyz")
      {
        nlayer = 0;
      }
      else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
               special_flow_homdir_ == "yz")
      {
        // get center
        double center = 0.0;
        if (special_flow_homdir_ == "xy")
          center = calc_turb_prandtl_params.get<double>("zcenter");
        else if (special_flow_homdir_ == "xz")
          center = calc_turb_prandtl_params.get<double>("ycenter");
        else if (special_flow_homdir_ == "yz")
          center = calc_turb_prandtl_params.get<double>("xcenter");

        // for this purpose, determine the layer (the plane for average)
        bool found = false;
        for (nlayer = 0; nlayer < (int)(*dir1coords_).size() - 1;)
        {
          if (center < (*dir1coords_)[nlayer + 1])
          {
            found = true;
            break;
          }
          nlayer++;
        }
        if (found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }
      }
      else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
               special_flow_homdir_ == "z")
      {
        // get center
        double dim1_center = 0.0;
        double dim2_center = 0.0;
        if (special_flow_homdir_ == "x")
        {
          dim1_center = calc_turb_prandtl_params.get<double>("ycenter");
          dim2_center = calc_turb_prandtl_params.get<double>("zcenter");
        }
        else if (special_flow_homdir_ == "y")
        {
          dim1_center = calc_turb_prandtl_params.get<double>("xcenter");
          dim2_center = calc_turb_prandtl_params.get<double>("zcenter");
        }
        else if (special_flow_homdir_ == "z")
        {
          dim1_center = calc_turb_prandtl_params.get<double>("xcenter");
          dim2_center = calc_turb_prandtl_params.get<double>("ycenter");
        }

        // for this purpose, determine the layer (the direction for average)
        int n1layer = 0;
        int n2layer = 0;
        bool dir1found = false;
        bool dir2found = false;
        for (n1layer = 0; n1layer < (int)(*dir1coords_).size() - 1;)
        {
          if (dim1_center < (*dir1coords_)[n1layer + 1])
          {
            dir1found = true;
            break;
          }
          n1layer++;
        }
        if (dir1found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }
        for (n2layer = 0; n2layer < (int)(*dir2coords_).size() - 1;)
        {
          if (dim2_center < (*dir2coords_)[n2layer + 1])
          {
            dir2found = true;
            break;
          }
          n2layer++;
        }
        if (dir2found == false)
        {
          FOUR_C_THROW("could not determine element layer");
        }

        const int numdir1layer = (int)(*dir1coords_).size() - 1;
        nlayer = numdir1layer * n2layer + n1layer;
      }
      else
        FOUR_C_THROW("Homogeneous directions not supported!");

      // add it up
      local_ele_sum_LkMk[nlayer] += LkMk;
      local_ele_sum_MkMk[nlayer] += MkMk;

      local_count_for_average[nlayer]++;
    }  // end add element contribution to layer averaging for channel flows

  }  // end loop over elements

  // export from row to column map
  const Core::LinAlg::Map* elecolmap = scatradiscret_->element_col_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> col_Prt =
      std::make_shared<Core::LinAlg::Vector<double>>(*elecolmap, true);
  col_Prt->put_scalar(0.0);
  Core::LinAlg::export_to(Prt, *col_Prt);
  // store in parameters
  Teuchos::ParameterList* modelparams = &(extraparams.sublist("TURBULENCE MODEL"));
  modelparams->set<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_ele_Prt", col_Prt);

  // ----------------------------------------------------
  // global in plane averaging of quantities for
  // turbulent channel flow

  if (homdir_)
  {
    // now add all the stuff from the different processors

    for (int rr = 0; rr < numlayers; ++rr)
    {
      (count_for_average[rr]) =
          Core::Communication::sum_all((local_count_for_average[rr]), scatradiscret_->get_comm());
      ((*averaged_LkMk)[rr]) =
          Core::Communication::sum_all((local_ele_sum_LkMk[rr]), scatradiscret_->get_comm());
      ((*averaged_MkMk)[rr]) =
          Core::Communication::sum_all((local_ele_sum_MkMk[rr]), scatradiscret_->get_comm());
    }

    // do averaging
    for (int rr = 0; rr < numlayers; ++rr)
    {
      // perform some checks first
      if (count_for_average[rr] == 0 and
          ((*averaged_LkMk)[rr] != 0.0 or (*averaged_MkMk)[rr] != 0.0))
        FOUR_C_THROW("Expected 'averaged_LkMk' and 'averaged_MkMk' equal zero!");

      // calculate averaged quantities
      // we have to exclude zero here, since, for backward-facing steps, the step is contained and
      // we will obtain a zero block there
      if (count_for_average[rr] != 0)
      {
        (*averaged_LkMk)[rr] /= count_for_average[rr];
        (*averaged_MkMk)[rr] /= count_for_average[rr];
      }
    }

    // provide necessary information for the elements
    {
      modelparams->set<std::shared_ptr<std::vector<double>>>("averaged_LkMk_", averaged_LkMk);
      modelparams->set<std::shared_ptr<std::vector<double>>>("averaged_MkMk_", averaged_MkMk);
      if (special_flow_homdir_ == "xyz")
      {
        // nothing to do
      }
      else if (special_flow_homdir_ == "xy" or special_flow_homdir_ == "xz" or
               special_flow_homdir_ == "yz")
      {
        modelparams->set<std::shared_ptr<std::vector<double>>>("planecoords_", dir1coords_);
        // channel flow only
        // return number of elements per layer
        // equal number of elements in each layer assumed
        numele_layer = count_for_average[0];
      }
      else if (special_flow_homdir_ == "x" or special_flow_homdir_ == "y" or
               special_flow_homdir_ == "z")
      {
        modelparams->set<std::shared_ptr<std::vector<double>>>("dir1coords_", dir1coords_);
        modelparams->set<std::shared_ptr<std::vector<double>>>("dir2coords_", dir2coords_);
      }
      else
        FOUR_C_THROW("More than two homogeneous directions not supported!");
    }
  }  // end if turbulent channel flow

  return;
}  // end FLD::DynSmagFilter::dyn_smag_compute_prt

FOUR_C_NAMESPACE_CLOSE
