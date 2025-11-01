// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------*
 | calculate filtered quantities for dynamic Smagorinsky model  rasthofer 08/12|
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::scatra_apply_box_filter(double& dens_hat,
    double& temp_hat, double& dens_temp_hat, double& phi2_hat, double& phiexpression_hat,
    std::vector<double>& vel_hat, std::vector<double>& densvel_hat,
    std::vector<double>& densveltemp_hat, std::vector<double>& densstraintemp_hat,
    std::vector<double>& phi_hat, std::vector<std::vector<double>>& alphaijsc_hat, double& volume,
    const Core::Elements::Element* ele, Teuchos::ParameterList& params)
{
  // do preparations first
  // ---------------------------------------------
  Core::LinAlg::Matrix<nsd_, nsd_> vderxy(Core::LinAlg::Initialization::zero);
  double alpha2 = 0.0;
  // use one-point Gauss rule to do calculations at the element center
  volume = eval_shape_func_and_derivs_at_ele_center();

  // get material
  std::shared_ptr<const Core::Mat::Material> material = ele->material();

  // get phi at integration point
  const double phinp = funct_.dot(ephinp_[0]);

  // get temperature at integration point
  double tempnp = phinp;
  if (material->material_type() == Core::Materials::m_matlist)
  {
    const Mat::MatList* actmat = static_cast<const Mat::MatList*>(material.get());
    tempnp = funct_.dot(ephinp_[actmat->num_mat() - 1]);
  }

  // density at time n+1
  const double densnp = get_density(ele, material, params, tempnp);

  // get velocities (n+alpha_F/1,i) at integration point
  Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::zero);
  convelint.multiply(evelnp_, funct_);

  // compute rate of strain
  double rateofstrain = -1.0e30;
  rateofstrain = get_strain_rate(evelnp_);

  // gradient of scalar value
  Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
  gradphi.multiply(derxy_, ephinp_[0]);

  // perform integrations, i.e., convolution
  // ---------------------------------------------
  for (unsigned rr = 0; rr < nsd_; ++rr)
  {
    double tmp = convelint(rr) * volume;

    // add contribution to integral over velocities
    (vel_hat)[rr] += tmp;

    // add contribution to integral over dens times velocity
    (densvel_hat)[rr] += densnp * tmp;

    // add contribution to integral over dens times temperature times velocity
    (densveltemp_hat)[rr] += densnp * phinp * tmp;
  }

  for (unsigned rr = 0; rr < nsd_; ++rr)
  {
    double tmp = gradphi(rr) * volume;
    // add contribution to integral over dens times rate of strain times phi gradient
    (densstraintemp_hat)[rr] += densnp * rateofstrain * tmp;
    (phi_hat)[rr] = tmp;
    phi2_hat += tmp * gradphi(rr);
  }

  // calculate vreman part
  if (turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
  {
    if (nsd_ == 3)
    {
      double beta00;
      double beta11;
      double beta22;
      double beta01;
      double beta02;
      double beta12;
      double bbeta;
      double hk2 = pow(volume, (2.0 / 3.0));


      for (unsigned nn = 0; nn < nsd_; ++nn)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          vderxy(nn, rr) = derxy_(rr, 0) * evelnp_(nn, 0);
          for (unsigned mm = 1; mm < nen_; ++mm)
          {
            vderxy(nn, rr) += derxy_(rr, mm) * evelnp_(nn, mm);
          }
          (alphaijsc_hat)[rr][nn] = vderxy(nn, rr);  // change indices to make compatible to paper
          alpha2 += vderxy(nn, rr) * vderxy(nn, rr);
        }
      }

      beta00 = hk2 * (alphaijsc_hat)[0][0] * (alphaijsc_hat)[0][0] +
               hk2 * (alphaijsc_hat)[1][0] * (alphaijsc_hat)[1][0] +
               hk2 * (alphaijsc_hat)[2][0] * (alphaijsc_hat)[2][0];
      beta11 = hk2 * (alphaijsc_hat)[0][1] * (alphaijsc_hat)[0][1] +
               hk2 * (alphaijsc_hat)[1][1] * (alphaijsc_hat)[1][1] +
               hk2 * (alphaijsc_hat)[2][1] * (alphaijsc_hat)[2][1];
      beta22 = hk2 * (alphaijsc_hat)[0][2] * (alphaijsc_hat)[0][2] +
               hk2 * (alphaijsc_hat)[1][2] * (alphaijsc_hat)[1][2] +
               hk2 * (alphaijsc_hat)[2][2] * (alphaijsc_hat)[2][2];
      beta01 = hk2 * (alphaijsc_hat)[0][0] * (alphaijsc_hat)[0][1] +
               hk2 * (alphaijsc_hat)[1][0] * (alphaijsc_hat)[1][1] +
               hk2 * (alphaijsc_hat)[2][0] * (alphaijsc_hat)[2][1];
      beta02 = hk2 * (alphaijsc_hat)[0][0] * (alphaijsc_hat)[0][2] +
               hk2 * (alphaijsc_hat)[1][0] * (alphaijsc_hat)[1][2] +
               hk2 * (alphaijsc_hat)[2][0] * (alphaijsc_hat)[2][2];
      beta12 = hk2 * (alphaijsc_hat)[0][1] * (alphaijsc_hat)[0][2] +
               hk2 * (alphaijsc_hat)[1][1] * (alphaijsc_hat)[1][2] +
               hk2 * (alphaijsc_hat)[2][1] * (alphaijsc_hat)[2][2];

      bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
              beta11 * beta22 - beta12 * beta12;
      if (alpha2 < 1.0e-12)
        (phiexpression_hat) = 0.0;
      else
        (phiexpression_hat) = (phi2_hat)*sqrt(bbeta / alpha2);

      for (unsigned nn = 0; nn < nsd_; ++nn)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          (alphaijsc_hat)[rr][nn] *= volume;
        }
      }
    }
    else
      FOUR_C_THROW("Vreman model only for nsd_==3");
  }
  // add additional scalar quantities
  // i.e., filtered density, filtered density times scalar (i.e., temperature) and scalar
  dens_hat = densnp * volume;
  dens_temp_hat = densnp * phinp * volume;
  temp_hat = phinp * volume;

  return;
}  // ScaTraEleCalc::scatra_apply_box_filter


/*-----------------------------------------------------------------------------*
 | get density at integration point                                 fang 02/15 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleCalc<distype, probdim>::get_density(
    const Core::Elements::Element* ele, std::shared_ptr<const Core::Mat::Material> material,
    Teuchos::ParameterList& params, const double tempnp)
{
  // initialization
  double density(0.);

  if (material->material_type() == Core::Materials::m_scatra)
  {
    // access fluid discretization
    std::shared_ptr<Core::FE::Discretization> fluiddis = nullptr;
    fluiddis = Global::Problem::instance()->get_dis("fluid");
    // get corresponding fluid element (it has the same global ID as the scatra element)
    Core::Elements::Element* fluidele = fluiddis->g_element(ele->id());
    if (fluidele == nullptr) FOUR_C_THROW("Fluid element {} not on local processor", ele->id());
    // get fluid material
    std::shared_ptr<Core::Mat::Material> fluidmat = fluidele->material();
    if (fluidmat->material_type() != Core::Materials::m_fluid)
      FOUR_C_THROW("Invalid fluid material for passive scalar transport in turbulent flow!");

    density = std::dynamic_pointer_cast<const Mat::NewtonianFluid>(fluidmat)->density();
    if (density != 1.0) FOUR_C_THROW("Check your diffusivity! Dynamic diffusivity required!");
  }

  else
    FOUR_C_THROW("Invalid material type!");

  return density;
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::GetDensity


/*----------------------------------------------------------------------------------*
 | calculate turbulent Prandtl number for dynamic Smagorinsky model  rasthofer 08/12|
 *----------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::scatra_calc_smag_const_lk_mk_and_mk_mk(
    Core::LinAlg::MultiVector<double>& col_filtered_vel,
    Core::LinAlg::MultiVector<double>& col_filtered_dens_vel,
    Core::LinAlg::MultiVector<double>& col_filtered_dens_vel_temp,
    Core::LinAlg::MultiVector<double>& col_filtered_dens_rateofstrain_temp,
    Core::LinAlg::Vector<double>& col_filtered_temp,
    Core::LinAlg::Vector<double>& col_filtered_dens,
    Core::LinAlg::Vector<double>& col_filtered_dens_temp, double& LkMk, double& MkMk,
    double& xcenter, double& ycenter, double& zcenter, const Core::Elements::Element* ele)
{
  Core::LinAlg::Matrix<nsd_, nen_> evel_hat;
  Core::LinAlg::Matrix<nsd_, nen_> edensvel_hat;
  Core::LinAlg::Matrix<nsd_, nen_> edensveltemp_hat;
  Core::LinAlg::Matrix<nsd_, nen_> edensstraintemp_hat;
  Core::LinAlg::Matrix<1, nen_> etemp_hat;
  Core::LinAlg::Matrix<1, nen_> edens_hat;
  Core::LinAlg::Matrix<1, nen_> edenstemp_hat;
  // extract required (node-based) filtered quantities
  Core::FE::extract_my_node_based_values(ele, evel_hat, col_filtered_vel, nsd_);
  Core::FE::extract_my_node_based_values(ele, edensvel_hat, col_filtered_dens_vel, nsd_);
  Core::FE::extract_my_node_based_values(ele, edensveltemp_hat, col_filtered_dens_vel_temp, nsd_);
  Core::FE::extract_my_node_based_values(
      ele, edensstraintemp_hat, col_filtered_dens_rateofstrain_temp, nsd_);
  Core::FE::extract_my_node_based_values(ele, etemp_hat, col_filtered_temp, 1);
  Core::FE::extract_my_node_based_values(ele, edens_hat, col_filtered_dens, 1);
  Core::FE::extract_my_node_based_values(ele, edenstemp_hat, col_filtered_dens_temp, 1);

  // get center coordinates of element
  xcenter = 0.0;
  ycenter = 0.0;
  zcenter = 0.0;
  for (unsigned inode = 0; inode < nen_; inode++)
  {
    // xyze_ has been initialized at the beginning of Impl()
    xcenter += xyze_(0, inode);
    ycenter += xyze_(1, inode);
    zcenter += xyze_(2, inode);
  }
  xcenter /= nen_;
  ycenter /= nen_;
  zcenter /= nen_;

  // use one-point Gauss rule to do calculations at the element center
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToStabGaussRule<distype>::rule);

  eval_shape_func_and_derivs_at_int_point(intpoints, 0);

  // get filtered dens * velocities (n+alpha_F/1,i) at integration point
  //
  //                     +-----
  //        ^ n+af/1     \                  ^ n+af/1
  //    rho * vel (x) =   +      N (x) * rho*vel
  //                     /        j             j
  //                     +-----
  //                    node j
  //
  Core::LinAlg::Matrix<nsd_, 1> densvelint_hat;
  densvelint_hat.multiply(edensvel_hat, funct_);

  // get filtered dens * temp * velocities (n+alpha_F/1,i) at integration point
  //
  //                           +-----
  //        ^ n+af/1            \                    ^ n+af/1
  //    rho * vel * temp (x) =   +      N (x) * rho*temp*vel
  //                            /        j                  j
  //                           +-----
  //                           node j
  //
  Core::LinAlg::Matrix<nsd_, 1> densveltempint_hat;
  densveltempint_hat.multiply(edensveltemp_hat, funct_);

  // get filtered dens * rate-of-strain * grad T (n+alpha_F/1,i) at integration point
  //
  //                                        +-----
  //                ^ n+af/1                 \                    ^ n+af/1
  //    rho * rate-of-strain * grad T (x) =   +      N (x) * rho * rate-of-strain * grad T (x)
  //                                         /        j                                       j
  //                                        +-----
  //                                         node j
  //
  Core::LinAlg::Matrix<nsd_, 1> densstraintempint_hat;
  densstraintempint_hat.multiply(edensstraintemp_hat, funct_);

  // get filtered density at integration point
  //
  //         +-----
  //    ^     \              ^
  //   rho =   +    N (x) * rho
  //          /      k         k
  //         +-----
  //         node k
  //
  //
  double densint_hat = 0.0;
  // get filtered density times temperature at integration point
  //
  //            +-----
  //     ^       \                 ^
  //  rho * T =   +    N (x) * rho * T
  //             /      k             k
  //            +-----
  //              node k
  //
  //
  double denstempint_hat = 0.0;
  //  double tempint_hat = 0.0;
  for (unsigned mm = 0; mm < nen_; ++mm)
  {
    densint_hat += funct_(mm, 0) * edens_hat(0, mm);
    denstempint_hat += funct_(mm, 0) * edenstemp_hat(0, mm);
  }

  // compute rate of strain
  double rateofstrain = -1.0e30;
  rateofstrain = get_strain_rate(evel_hat);

  // gradient of scalar value (i.e., of filtered temperature)
  Core::LinAlg::Matrix<nsd_, 1> gradtemp_hat;
  gradtemp_hat.multiply_nt(derxy_, etemp_hat);

  // this is sqrt(3)
  const double filterwidthratio = 1.73;

  // calculate L_k and M_k
  Core::LinAlg::Matrix<nsd_, 1> L_k;
  Core::LinAlg::Matrix<nsd_, 1> M_k;

  for (unsigned rr = 0; rr < nsd_; rr++)
  {
    L_k(rr, 0) = densveltempint_hat(rr, 0) - densvelint_hat(rr, 0) * denstempint_hat / densint_hat;
    M_k(rr, 0) = densstraintempint_hat(rr, 0) - filterwidthratio * filterwidthratio * densint_hat *
                                                    rateofstrain * gradtemp_hat(rr, 0);
  }

  // perform contraction via dot product
  LkMk = 0.0;
  MkMk = 0.0;
  for (unsigned rr = 0; rr < nsd_; rr++)
  {
    LkMk += L_k(rr, 0) * M_k(rr, 0);
    MkMk += M_k(rr, 0) * M_k(rr, 0);
  }

  return;
}  // ScaTraEleCalc::scatra_calc_smag_const_lk_mk_and_mk_mk


/*----------------------------------------------------------------------------------*
 | calculate vreman constant                                             krank 08/13|
 *----------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::scatra_calc_vreman_dt(
    Core::LinAlg::MultiVector<double>& col_filtered_phi,
    Core::LinAlg::Vector<double>& col_filtered_phi2,
    Core::LinAlg::Vector<double>& col_filtered_phiexpression,
    Core::LinAlg::MultiVector<double>& col_filtered_alphaijsc, double& dt_numerator,
    double& dt_denominator, const Core::Elements::Element* ele)
{
  double phi_hat2 = 0.0;
  Core::LinAlg::Matrix<9, nen_> ealphaijsc_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nen_> ephi_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, nen_> ephi2_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, nen_> ephiexpression_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, nsd_> alphaijsc_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd_, 1> phi_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> phi2_hat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> phiexpression_hat(Core::LinAlg::Initialization::zero);

  Core::FE::extract_my_node_based_values(ele, ephi_hat, col_filtered_phi, nsd_);
  Core::FE::extract_my_node_based_values(ele, ephi2_hat, col_filtered_phi2, 1);
  Core::FE::extract_my_node_based_values(ele, ephiexpression_hat, col_filtered_phiexpression, 1);
  Core::FE::extract_my_node_based_values(ele, ealphaijsc_hat, col_filtered_alphaijsc, nsd_ * nsd_);
  // use one-point Gauss rule to do calculations at the element center
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToStabGaussRule<distype>::rule);
  double volume = eval_shape_func_and_derivs_at_int_point(intpoints, 0);

  phi_hat.multiply(ephi_hat, funct_);
  phi2_hat.multiply(ephi2_hat, funct_);
  phiexpression_hat.multiply(ephiexpression_hat, funct_);

  FOUR_C_ASSERT_ALWAYS(nsd_ >= 3, "This function is only well defined for 3D problems!");
  for (unsigned nn = 0; nn < 3; ++nn)
  {
    for (unsigned rr = 0; rr < 3; ++rr)
    {
      int index = 3 * nn + rr;
      alphaijsc_hat(nn, rr) = funct_(0) * ealphaijsc_hat(index, 0);

      for (unsigned mm = 1; mm < nen_; ++mm)
      {
        alphaijsc_hat(nn, rr) += funct_(mm) * ealphaijsc_hat(index, mm);
      }
    }
  }

  for (unsigned rr = 0; rr < nsd_; ++rr)
  {
    phi_hat2 += phi_hat(rr, 0) * phi_hat(rr, 0);
  }

  dt_denominator = phi2_hat(0, 0) - phi_hat2;

  // calculate vreman part
  {
    double beta00;
    double beta11;
    double beta22;
    double beta01;
    double beta02;
    double beta12;
    double bbeta;
    double hk2 = 3 * pow(volume, (2.0 / 3.0));
    double alpha2 = 0.0;
    double phiexpressionf_hat = 0.0;

    for (unsigned nn = 0; nn < nsd_; ++nn)
    {
      for (unsigned rr = 0; rr < nsd_; ++rr)
      {
        alpha2 += alphaijsc_hat(nn, rr) * alphaijsc_hat(nn, rr);
      }
    }

    beta00 = hk2 * alphaijsc_hat(0, 0) * alphaijsc_hat(0, 0) +
             hk2 * alphaijsc_hat(1, 0) * alphaijsc_hat(1, 0) +
             hk2 * alphaijsc_hat(2, 0) * alphaijsc_hat(2, 0);
    beta11 = hk2 * alphaijsc_hat(0, 1) * alphaijsc_hat(0, 1) +
             hk2 * alphaijsc_hat(1, 1) * alphaijsc_hat(1, 1) +
             hk2 * alphaijsc_hat(2, 1) * alphaijsc_hat(2, 1);
    beta22 = hk2 * alphaijsc_hat(0, 2) * alphaijsc_hat(0, 2) +
             hk2 * alphaijsc_hat(1, 2) * alphaijsc_hat(1, 2) +
             hk2 * alphaijsc_hat(2, 2) * alphaijsc_hat(2, 2);
    beta01 = hk2 * alphaijsc_hat(0, 0) * alphaijsc_hat(0, 1) +
             hk2 * alphaijsc_hat(1, 0) * alphaijsc_hat(1, 1) +
             hk2 * alphaijsc_hat(2, 0) * alphaijsc_hat(2, 1);
    beta02 = hk2 * alphaijsc_hat(0, 0) * alphaijsc_hat(0, 2) +
             hk2 * alphaijsc_hat(1, 0) * alphaijsc_hat(1, 2) +
             hk2 * alphaijsc_hat(2, 0) * alphaijsc_hat(2, 2);
    beta12 = hk2 * alphaijsc_hat(0, 1) * alphaijsc_hat(0, 2) +
             hk2 * alphaijsc_hat(1, 1) * alphaijsc_hat(1, 2) +
             hk2 * alphaijsc_hat(2, 1) * alphaijsc_hat(2, 2);

    bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
            beta11 * beta22 - beta12 * beta12;
    if (alpha2 < 1.0e-12)
      phiexpressionf_hat = 0.0;
    else
      phiexpressionf_hat = (phi_hat2)*sqrt(bbeta / alpha2);

    dt_numerator = phiexpressionf_hat - phiexpression_hat(0, 0);
    //    std::cout << "scatra_ele_impl_turb_service.cpp  dt_numerator  " << dt_numerator <<
    //    std::endl; std::cout << "scatra_ele_impl_turb_service.cpp  dt_denominator  " <<
    //    dt_denominator << std::endl; std::cout << "scatra_ele_impl_turb_service.cpp  phi_hat2  "
    //    << phi_hat2 << std::endl; std::cout << "scatra_ele_impl_turb_service.cpp  phi2_hat  " <<
    //    phi2_hat << std::endl; std::cout << "scatra_ele_impl_turb_service.cpp  alpha2  " << alpha2
    //    << std::endl;
  }

  return;
}


/*----------------------------------------------------------------------------------*
 | calculate mean turbulent Prandtl number                           rasthofer 08/12|
 *----------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::get_mean_prt_of_homogeneous_direction(
    Teuchos::ParameterList& turbmodelparams, int& nlayer)
{
  // NOTE: we calculate the inverse of the turbulent Prandtl number here (i.e., (Cs*h)^2 / Pr_t)

  if (nsd_ != 3) FOUR_C_THROW("turbulence and 3D flow !");

  if (turbmodelparams.get<std::string>("HOMDIR", "not_specified") != "not_specified")
  {
    std::shared_ptr<std::vector<double>> averaged_LkMk =
        turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_LkMk_");
    std::shared_ptr<std::vector<double>> averaged_MkMk =
        turbmodelparams.get<std::shared_ptr<std::vector<double>>>("averaged_MkMk_");

    // get homogeneous direction
    std::string homdir = turbmodelparams.get<std::string>("HOMDIR", "not_specified");

    // here, the layer is determined via the element center in order to get the correct
    // averaged value from the vector of averaged (M/L)kMk
    double xcenter = 0.0;
    double ycenter = 0.0;
    double zcenter = 0.0;
    for (unsigned inode = 0; inode < nen_; inode++)
    {
      xcenter += xyze_(0, inode);
      ycenter += xyze_(1, inode);
      zcenter += xyze_(2, inode);
    }
    xcenter /= nen_;
    ycenter /= nen_;
    zcenter /= nen_;

    // determine the layer
    if (homdir == "xyz")
    {
      nlayer = 0;
    }
    else if (homdir == "xy" or homdir == "xz" or homdir == "yz")
    {
      std::shared_ptr<std::vector<double>> planecoords =
          turbmodelparams.get<std::shared_ptr<std::vector<double>>>("planecoords_");
      // get center
      double center = 0.0;
      if (homdir == "xy")
        center = zcenter;
      else if (homdir == "xz")
        center = ycenter;
      else if (homdir == "yz")
        center = xcenter;

      bool found = false;
      for (nlayer = 0; nlayer < static_cast<int>((*planecoords).size() - 1);)
      {
        if (center < (*planecoords)[nlayer + 1])
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
    else if (homdir == "x" or homdir == "y" or homdir == "z")
    {
      std::shared_ptr<std::vector<double>> dir1coords =
          turbmodelparams.get<std::shared_ptr<std::vector<double>>>("dir1coords_");
      std::shared_ptr<std::vector<double>> dir2coords =
          turbmodelparams.get<std::shared_ptr<std::vector<double>>>("dir2coords_");
      // get center
      double dim1_center = 0.0;
      double dim2_center = 0.0;
      if (homdir == "x")
      {
        dim1_center = ycenter;
        dim2_center = zcenter;
      }
      else if (homdir == "y")
      {
        dim1_center = xcenter;
        dim2_center = zcenter;
      }
      else if (homdir == "z")
      {
        dim1_center = xcenter;
        dim2_center = ycenter;
      }

      int n1layer = 0;
      int n2layer = 0;
      bool dir1found = false;
      bool dir2found = false;
      for (n1layer = 0; n1layer < (int)(*dir1coords).size() - 1;)
      {
        if (dim1_center < (*dir1coords)[n1layer + 1])
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
      for (n2layer = 0; n2layer < (int)(*dir2coords).size() - 1;)
      {
        if (dim2_center < (*dir2coords)[n2layer + 1])
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

      const int numdir1layer = (int)(*dir1coords).size() - 1;
      nlayer = numdir1layer * n2layer + n1layer;
    }
    else
      FOUR_C_THROW("Homogeneous directions not supported!");

    // (Cs*Delta)^2/Prt is set by the averaged quantities
    if ((*averaged_MkMk)[nlayer] < 1E-16)
    {
      //  std::cout << "warning: abs(averaged_MkMk) < 1E-16 -> set inverse of turbulent Prandtl
      //  number to zero!"  << std::endl;
      tpn_ = 0.0;
    }
    else
      tpn_ = (*averaged_LkMk)[nlayer] / (*averaged_MkMk)[nlayer];
    // clipping to get algorithm stable
    if (tpn_ < 0.0)
    {
      tpn_ = 0.0;
    }
  }

  return;
}  // ScaTraEleCalc::GetMeanOfHomogeneousDirection


/*----------------------------------------------------------------------*
  |  calculate all-scale art. subgrid diffusivity (private)     vg 10/09 |
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_subgr_diff(
    double& visc, const double vol, const int k, const double densnp)
{
  // get number of dimensions
  const double dim = (double)nsd_;

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  const double h = std::pow(vol, (1.0 / dim));

  // subgrid-scale diffusivity to be computed and added to diffus
  double sgdiff(0.0);

  // all-scale subgrid diffusivity due to Smagorinsky model divided by
  // turbulent Prandtl number
  if (turbparams_->turb_model() == Inpar::FLUID::smagorinsky)
  {
    //
    // SMAGORINSKY MODEL
    // -----------------
    //                                   +-                                 -+ 1
    //                               2   |          / h \           / h \    | -
    //    visc          = dens * lmix  * | 2 * eps | u   |   * eps | u   |   | 2
    //        turbulent           |      |          \   / ij        \   / ij |
    //                            |      +-                                 -+
    //                            |
    //                            |      |                                   |
    //                            |      +-----------------------------------+
    //                            |           'resolved' rate of strain
    //                    mixing length
    // -> either provided by dynamic modeling procedure and stored in Cs_delta_sq
    // -> or computed based on fixed Smagorinsky constant Cs:
    //             Cs = 0.17   (Lilly --- Determined from filter
    //                          analysis of Kolmogorov spectrum of
    //                          isotropic turbulence)
    //             0.1 < Cs < 0.24 (depending on the flow)
    //
    //                       dens * visc
    //                                  turbulent
    //    kappa           =  ---------------------
    //         turbulent         Pr
    //                             turbulent
    // -> Prt prescribed in input file or estimated dynamically
    //

    // compute (all-scale) rate of strain
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(econvelnp_);

    // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
    sgdiff = densnp * turbparams_->cs() * turbparams_->cs() * h * h * rateofstrain / tpn_;

    // add subgrid viscosity to physical viscosity for computation
    // of subgrid-scale velocity when turbulence model is applied
    if (scatrapara_->rb_sub_gr_vel()) visc += sgdiff * tpn_;
  }
  else if (turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
  {
    // compute (all-scale) rate of strain
    double rateofstrain = -1.0e30;
    rateofstrain = get_strain_rate(econvelnp_);

    // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
    // remark: for dynamic estimation, tpn corresponds to (Cs*h)^2 / Pr_t
    sgdiff = densnp * rateofstrain * tpn_;
  }
  else if (turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
  {
    if (nsd_ == 3)
    {
      double beta00;
      double beta11;
      double beta22;
      double beta01;
      double beta02;
      double beta12;
      double bbeta;
      double alphavreman;
      double hkxpow2;
      double hkypow2;
      double hkzpow2;
      double sgviscwocv = 0.0;

      Core::LinAlg::Matrix<nsd_, nsd_> velderxy(Core::LinAlg::Initialization::zero);

      velderxy.multiply_nt(econvelnp_, derxy_);

      //- cube root of element volume

      hkxpow2 = pow(vol, (2.0 / 3.0));
      hkypow2 = hkxpow2;
      hkzpow2 = hkxpow2;

      beta00 = hkxpow2 * velderxy(0, 0) * velderxy(0, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(0, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(0, 2);
      beta11 = hkxpow2 * velderxy(1, 0) * velderxy(1, 0) +
               hkypow2 * velderxy(1, 1) * velderxy(1, 1) +
               hkzpow2 * velderxy(1, 2) * velderxy(1, 2);
      beta22 = hkxpow2 * velderxy(2, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(2, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(2, 2) * velderxy(2, 2);
      beta01 = hkxpow2 * velderxy(0, 0) * velderxy(1, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(1, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(1, 2);
      beta02 = hkxpow2 * velderxy(0, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(0, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(0, 2) * velderxy(2, 2);
      beta12 = hkxpow2 * velderxy(1, 0) * velderxy(2, 0) +
               hkypow2 * velderxy(1, 1) * velderxy(2, 1) +
               hkzpow2 * velderxy(1, 2) * velderxy(2, 2);

      bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
              beta11 * beta22 - beta12 * beta12;

      alphavreman = velderxy(0, 0) * velderxy(0, 0) + velderxy(0, 1) * velderxy(0, 1) +
                    velderxy(0, 2) * velderxy(0, 2) + velderxy(1, 0) * velderxy(1, 0) +
                    velderxy(1, 1) * velderxy(1, 1) + velderxy(1, 2) * velderxy(1, 2) +
                    velderxy(2, 0) * velderxy(2, 0) + velderxy(2, 1) * velderxy(2, 1) +
                    velderxy(2, 2) * velderxy(2, 2);

      if (alphavreman < 1.0E-12)
        sgviscwocv = 0.0;
      else
        sgviscwocv = sqrt(bbeta / alphavreman);


      // remark: Cs corresponds to Dt, calculated in the vreman class
      //        The vreman constant Cv is not required here, since it cancels out with the
      //        vreman constant omitted during the calculation of D_t
      if (turbparams_->cs() <= 1.0E-12)
        sgdiff = 0.0;
      else
        sgdiff = densnp * sgviscwocv / turbparams_->cs();
    }
    else
      FOUR_C_THROW("Vreman model only for nsd_==3");
  }

  // update diffusivity
  diffmanager_->set_isotropic_sub_grid_diff(sgdiff, k);

  return;
}  // ScaTraEleCalc::calc_subgr_diff


/*----------------------------------------------------------------------*
  |  calculate fine-scale art. subgrid diffusivity (private)    vg 10/09 |
  *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_fine_scale_subgr_diff(double& sgdiff,
    Core::LinAlg::SerialDenseVector& subgrdiff, Core::Elements::Element* ele, const double vol,
    const int k, const double densnp, const double diffus,
    const Core::LinAlg::Matrix<nsd_, 1> convelint)
{
  // get number of dimensions
  const double dim = (double)nsd_;

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  const double h = std::pow(vol, (1.0 / dim));

  //----------------------------------------------------------------------
  // computation of fine-scale subgrid diffusivity for non-incremental
  // solver -> only artificial subgrid diffusivity
  // (values are stored in subgrid-diffusivity-scaling vector)
  //----------------------------------------------------------------------
  if (not scatraparatimint_->is_incremental())
  {
    // get element-type constant
    const double mk = ScaTra::mk<distype>();

    // velocity norm
    const double vel_norm = convelint.norm2();

    // parameter relating convective and diffusive forces + respective switch
    const double epe = mk * densnp * vel_norm * h / diffus;
    const double xi = std::max(epe, 1.0);

    // compute artificial subgrid diffusivity
    sgdiff = (((h) * (h)) * mk * ((vel_norm) * (vel_norm)) * ((densnp) * (densnp))) /
             (2.0 * diffus * xi);

    // compute entries of (fine-scale) subgrid-diffusivity-scaling vector
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      subgrdiff(vi) = sgdiff / ele->nodes()[vi]->num_element();
    }
  }
  //----------------------------------------------------------------------
  // computation of fine-scale subgrid diffusivity for incremental solver
  // -> only all-scale Smagorinsky model
  //----------------------------------------------------------------------
  else
  {
    if (turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_all)
    {
      //
      // ALL-SCALE SMAGORINSKY MODEL
      // ---------------------------
      //                                      +-                                 -+ 1
      //                                  2   |          / h \           / h \    | -
      //    visc          = dens * (C_S*h)  * | 2 * eps | u   |   * eps | u   |   | 2
      //        turbulent                     |          \   / ij        \   / ij |
      //                                      +-                                 -+
      //                                      |                                   |
      //                                      +-----------------------------------+
      //                                            'resolved' rate of strain
      //

      // compute (all-scale) rate of strain
      double rateofstrain = -1.0e30;
      rateofstrain = get_strain_rate(econvelnp_);

      // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
      sgdiff = densnp * turbparams_->cs() * turbparams_->cs() * h * h * rateofstrain / tpn_;
    }
    else if (turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_small)
    {
      //
      // FINE-SCALE SMAGORINSKY MODEL
      // ----------------------------
      //                                      +-                                 -+ 1
      //                                  2   |          /    \          /   \    | -
      //    visc          = dens * (C_S*h)  * | 2 * eps | fsu |   * eps | fsu |   | 2
      //        turbulent                     |          \   / ij        \   / ij |
      //                                      +-                                 -+
      //                                      |                                   |
      //                                      +-----------------------------------+
      //                                           'fine-scale' rate of strain
      //

      // fine-scale rate of strain
      double fsrateofstrain = -1.0e30;
      fsrateofstrain = get_strain_rate(efsvel_);

      // subgrid diffusivity = subgrid viscosity / turbulent Prandtl number
      sgdiff = densnp * turbparams_->cs() * turbparams_->cs() * h * h * fsrateofstrain / tpn_;
    }
  }

  return;
}  // ScaTraEleCalcCalc::FineScaleSubgrDiff


/*----------------------------------------------------------------------*
 | calculation of coefficients B and D for multifractal subgrid-scales  |
 |                                                      rasthofer 12/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_b_and_d_for_multifrac_subgrid_scales(
    Core::LinAlg::Matrix<nsd_, 1>& B_mfs,  ///< coefficient for fine-scale velocity (will be filled)
    double& D_mfs,                         ///< coefficient for fine-scale scalar (will be filled)
    const double vol,                      ///< volume of element
    const int k, const double densnp, const double diffus, const double visc,
    const Core::LinAlg::Matrix<nsd_, 1> convelint, const Core::LinAlg::Matrix<nsd_, 1> fsvelint)
{
  //----------------------------------------------------------------
  // calculation of B for fine-scale velocity
  //----------------------------------------------------------------

  // STEP1: determine N and Csgs

  // allocate vector for parameter N
  // N may depend on the direction -> currently unused
  std::vector<double> Nvel(3);
  // variable for final (corrected) Csgs_vel
  double Csgs_vel_nw = turbparams_->csgs_sg_vel();

  // potential calculation of Re to determine N
  double Re_ele = -1.0;
  // characteristic element length
  double hk = 1.0e+10;
  double strainnorm = 0.0;
  // ratio of viscous scale to element length
  double scale_ratio = 0.0;

  // get velocity at element center
  // convelint_.multiply(econvelnp_,funct_);
  // get norm of velocity
  const double vel_norm = convelint.norm2();
  // also for fine-scale velocity
  // fsvelint_.multiply(efsvel_,funct_);
  const double fsvel_norm = fsvelint.norm2();

  // do we have a fixed parameter N
  if (not turbparams_->calc_n())
  {
    // yes, store value
    for (unsigned rr = 1; rr < 3; rr++) Nvel[rr] = turbparams_->n_vel();
  }
  else  // no, so we calculate N from Re
  {
    // calculate characteristic element length
    hk = calc_ref_length(vol, convelint);

    // warning: k=0, this first scalar is taken!
    // multifractal subgrid-scale model is for passive and active
    // scalar transport
    // therefore, we need the density of the fluid here
    switch (turbparams_->ref_vel())
    {
      case FLUID::resolved:
      {
        Re_ele = vel_norm * hk * densnp / visc;
        break;
      }
      case FLUID::fine_scale:
      {
        Re_ele = fsvel_norm * hk * densnp / visc;
        break;
      }
      case FLUID::strainrate:
      {
        strainnorm = get_strain_rate(econvelnp_);
        strainnorm /= sqrt(2.0);  // cf. Burton & Dahm 2008
        Re_ele = strainnorm * hk * hk * densnp / visc;
        break;
      }
      default:
        FOUR_C_THROW("Unknown velocity!");
        break;
    }
    if (Re_ele < 0.0) FOUR_C_THROW("Something went wrong!");
    // clip Re to prevent negative N
    if (Re_ele < 1.0) Re_ele = 1.0;

    //
    //   Delta
    //  ---------  ~ Re^(3/4)
    //  lambda_nu
    //
    scale_ratio = turbparams_->c_nu() * pow(Re_ele, 0.75);
    // scale_ratio < 1.0 leads to N < 0
    // therefore, we clip once more
    if (scale_ratio < 1.0) scale_ratio = 1.0;

    //         |   Delta     |
    //  N =log | ----------- |
    //        2|  lambda_nu  |
    double N_re = log(scale_ratio) / log(2.0);
    if (N_re < 0.0) FOUR_C_THROW("Something went wrong when calculating N!");

    // store calculated N
    for (unsigned i = 0; i < nsd_; i++) Nvel[i] = N_re;
  }

  // calculate near-wall correction
  double Cai_phi = 0.0;
  if (turbparams_->nwl())
  {
    // not yet calculated, estimate norm of strain rate
    if ((not turbparams_->calc_n()) or (turbparams_->ref_vel() != FLUID::strainrate))
    {
      strainnorm = get_strain_rate(econvelnp_);
      strainnorm /= sqrt(2.0);  // cf. Burton & Dahm 2008
    }
    // and reference length
    if (not turbparams_->calc_n()) hk = calc_ref_length(vol, convelint);

    // get Re from strain rate
    double Re_ele_str = strainnorm * hk * hk * densnp / visc;
    if (Re_ele_str < 0.0) FOUR_C_THROW("Something went wrong!");
    // ensure positive values
    if (Re_ele_str < 1.0) Re_ele_str = 1.0;

    // calculate corrected Csgs
    //           -3/16
    //  *(1 - (Re)   )
    //
    Csgs_vel_nw *= (1.0 - pow(Re_ele_str, -3.0 / 16.0));

    // store Cai for application to scalar field
    Cai_phi = (1.0 - pow(Re_ele_str, -3.0 / 16.0));
  }

  // STEP 2: calculate B

  //       1                           1
  //       2   |       1              |2
  //  kappa  = | -------------------- |
  //           |  1 - alpha ^ (-4/3)  |
  //
  double kappa = 1.0 / (1.0 - pow(turbparams_->alpha(), -4.0 / 3.0));

  //                  1                                     1
  //                  2                  |                 |2
  //  B = Csgs * kappa  * 2 ^ (-2*N/3) * | 2 ^ (4*N/3) - 1 |
  //                                     |                 |
  //
  for (unsigned dim = 0; dim < nsd_; dim++)
  {
    B_mfs(dim, 0) = Csgs_vel_nw * sqrt(kappa) * pow(2.0, -2.0 * Nvel[dim] / 3.0) *
                    sqrt((pow(2.0, 4.0 * Nvel[dim] / 3.0) - 1.0));
    //    if (eid_ == 100)
    //     std::cout << "B  " << std::setprecision (10) << B_mfs(dim,0) << std::endl;
  }

  //----------------------------------------------------------------
  // calculation of D for fine-scale scalar
  //----------------------------------------------------------------

  // STEP 1: determine N

  // calculate Prandtl number or Schmidt number (passive scalar)
  const double Pr = visc / diffus;

  // since there are differences in the physical behavior between low and high
  // Prandtl/Schmidt number regime, we define a limit
  // to distinguish between the low and high Prandtl/Schmidt number regime
  // note: there is no clear definition of the ranges
  const double Pr_limit = 2.0;

  // allocate vector for parameter N
  double Nphi = 0.0;
  // ratio of diffusive scale to element length
  double scale_ratio_phi = 0.0;

  if (turbparams_->calc_n())
  {
    //
    //   Delta
    //  ---------  ~ Re^(3/4)*Pr^(p)
    //  lambda_diff
    //
    // Pr <= 1: p=3/4
    // Pr >> 1: p=1/2
    double p = 0.75;
    if (Pr > Pr_limit) p = 0.5;

    scale_ratio_phi = turbparams_->c_diff() * pow(Re_ele, 0.75) * pow(Pr, p);
    // scale_ratio < 1.0 leads to N < 0
    // therefore, we clip again
    if (scale_ratio_phi < 1.0) scale_ratio_phi = 1.0;

    //         |   Delta     |
    //  N =log | ----------- |
    //        2|  lambda_nu  |
    Nphi = log(scale_ratio_phi) / log(2.0);
    if (Nphi < 0.0) FOUR_C_THROW("Something went wrong when calculating N!");
  }
  else
    FOUR_C_THROW("Multifractal subgrid-scales for loma with calculation of N, only!");

  // STEP 2: calculate D

  // here, we have to distinguish three different cases:
  // Pr ~ 1 : fluid and scalar field have the nearly the same cutoff (usual case)
  //          k^(-5/3) scaling -> gamma = 4/3
  // Pr >> 1: (i)  cutoff in the inertial-convective range (Nvel>0, tricky!)
  //               k^(-5/3) scaling in the inertial-convective range
  //               k^(-1) scaling in the viscous-convective range
  //          (ii) cutoff in the viscous-convective range (fluid field fully resolved, easier)
  //               k^(-1) scaling -> gamma = 2
  // rare:
  // Pr << 1: scatra field could be fully resolved, not necessary
  //          k^(-5/3) scaling -> gamma = 4/3
  // Remark: case 2.(i) not implemented, yet

  // caution: compared to the mfs-loma paper, gamma denotes gamma+1 here
  double gamma = 0.0;
  // special option for case 2 (i)
  bool two_ranges = false;
  if (Pr < Pr_limit)
  {  // Pr <= 1, i.e., case 1 and 3
    gamma = 4.0 / 3.0;
  }
  else  // Pr >> 1
  {
    if (Nvel[0] < 1.0)
    {  // Pr >> 1 and fluid fully resolved, i.e., case 2 (ii)
      gamma = 2.0;
    }
    else  // Pr >> 1 and fluid not fully resolved, i.e., case 2 (i)
    {
      if (Nvel[0] > Nphi)
      {
        std::cout << "Nvel   " << Nvel[0] << std::endl;
        std::cout << "Nphi   " << Nphi << std::endl;
        FOUR_C_THROW("Nvel < Nphi expected!");
      }
      // here different options are possible
      // we assume k^(-4/3) for the complete range
      gamma = 4.0 / 3.0;
    }
  }

  //
  //   Phi    |       1                |
  //  kappa = | ---------------------- |
  //          |  1 - alpha ^ (-gamma)  |
  //
  double kappa_phi = 1.0 / (1.0 - pow(turbparams_->alpha(), -gamma));

  //                                                             1
  //       Phi    Phi                       |                   |2
  //  D = Csgs * kappa * 2 ^ (-gamma*N/2) * | 2 ^ (gamma*N) - 1 |
  //                                        |                   |
  //
  if (not two_ranges)  // usual case
    D_mfs = turbparams_->csgs_sg_phi() * sqrt(kappa_phi) * pow(2.0, -gamma * Nphi / 2.0) *
            sqrt((pow(2.0, gamma * Nphi) - 1.0));
  else
  {
    double gamma1 = 4.0 / 3.0;
    double gamma2 = 2.0;
    kappa_phi = 1.0 / (1.0 - pow(turbparams_->alpha(), -gamma1));
    D_mfs = turbparams_->csgs_sg_phi() * sqrt(kappa_phi) * pow(2.0, -gamma2 * Nphi / 2.0) *
            sqrt((pow(2.0, gamma1 * Nvel[0]) - 1.0) +
                 2.0 / 3.0 * pow((std::numbers::pi / hk), 2.0 / 3.0) *
                     (pow(2.0, gamma2 * Nphi) - pow(2.0, gamma2 * Nvel[0])));
  }

  // apply near-wall limit if required
  if (turbparams_->nwl_scatra() and turbparams_->nwl())
  {
    D_mfs *= Cai_phi;
  }

  //  if (eid_ == 100){
  ////    std::cout << "sqrt(kappa_phi)  " << std::setprecision(10) << sqrt(kappa_phi) << std::endl;
  ////    std::cout << "pow(2.0,-gamma*Nphi/2.0)  " << std::setprecision(10) <<
  /// pow(2.0,-gamma*Nphi/2.0) << std::endl; /    std::cout << "sqrt((pow(2.0,gamma*Nphi)-1))  " <<
  /// std::setprecision(10) << sqrt((pow(2.0,gamma*Nphi)-1)) << std::endl;
  //    std::cout << "D  " << std::setprecision(10) << D_mfs << std::endl;
  //    std::cout << "B  " << std::setprecision(10) << B_mfs(0,0) << "  " << B_mfs(1,0) << "  " <<
  //    B_mfs(2,0) << "  " << std::endl; if (nwl_scatra and nwl)
  //     std::cout << "CsgsD  " << std::setprecision(10) << Csgs_sgphi*Cai_phi << std::endl;
  //    else
  //     std::cout << "CsgsD  " << std::setprecision(10) << Csgs_sgphi << std::endl;
  //    std::cout << "CsgsB  " << std::setprecision(10) << Csgs_vel_nw << "  " << Csgs_sgvel << " "
  //    << Cai_phi << std::endl;
  //  }

  return;
}


/*----------------------------------------------------------------------*
 | calculate reference length for multifractal subgrid-scales           |
 |                                                      rasthofer 09/12 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_ref_length(
    const double vol, const Core::LinAlg::Matrix<nsd_, 1> convelint)
{
  // calculate characteristic element length
  double hk = 1.0e+10;
  // cf. stabilization parameters
  switch (turbparams_->ref_length())
  {
    case FLUID::streamlength:
    {
      // a) streamlength due to Tezduyar et al. (1992)
      // get norm of velocity
      const double vel_norm = convelint.norm2();
      // normed velocity vector
      Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
      if (vel_norm >= 1e-6)
        velino.update(1.0 / vel_norm, convelint);
      else
      {
        velino.clear();
        velino(0, 0) = 1.0;
      }
      Core::LinAlg::Matrix<nen_, 1> tmp;
      tmp.multiply_tn(derxy_, velino);
      const double val = tmp.norm1();
      hk = 2.0 / val;

      break;
    }
    case FLUID::sphere_diameter:
    {
      // b) volume-equivalent diameter
      hk = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);

      break;
    }
    case FLUID::cube_edge:
    {
      // c) cubic element length
      hk = std::pow(vol, (1.0 / static_cast<double>(nsd_)));

      break;
    }
    case FLUID::metric_tensor:
    {
      if (nsd_ != 3) FOUR_C_THROW("Turbulence is 3d!");
      /*          +-           -+   +-           -+   +-           -+
                |             |   |             |   |             |
                |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
          G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
           ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                |    i     j  |   |    i     j  |   |    i     j  |
                +-           -+   +-           -+   +-           -+
      */
      Core::LinAlg::Matrix<nsd_, nsd_> G(Core::LinAlg::Initialization::zero);

      for (unsigned nn = 0; nn < nsd_; ++nn)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          G(nn, rr) = xij_(nn, 0) * xij_(rr, 0);
          for (unsigned mm = 1; mm < nsd_; ++mm)
          {
            G(nn, rr) += xij_(nn, mm) * xij_(rr, mm);
          }
        }
      }

      /*          +----
                 \
        G : G =   +   G   * G
        -   -    /     ij    ij
        -   -   +----
                 i,j
     */
      double normG = 0.0;
      for (unsigned nn = 0; nn < nsd_; ++nn)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          normG += G(nn, rr) * G(nn, rr);
        }
      }
      hk = std::pow(normG, -0.25);

      break;
    }
    case FLUID::gradient_based:
    {
      Core::LinAlg::Matrix<nsd_, nsd_> convderxy;
      convderxy.multiply_nt(econvelnp_, derxy_);
      if (nsd_ != 3) FOUR_C_THROW("Turbulence is 3d!");
      Core::LinAlg::Matrix<nsd_, 1> normed_velgrad;

      for (unsigned rr = 0; rr < nsd_; ++rr)
      {
        double val = 0.0;
        for (unsigned idim = 0; idim < nsd_; idim++)
          val += convderxy(idim, rr) * convderxy(idim, rr);

        normed_velgrad(rr) = std::sqrt(val);

        // normed_velgrad(rr)=sqrt(convderxy(0,rr)*convderxy(0,rr)
        //                        +
        //                        convderxy(1,rr)*convderxy(1,rr)
        //                        +
        //                        convderxy(2,rr)*convderxy(2,rr));
      }
      double norm = normed_velgrad.norm2();

      // normed gradient
      if (norm > 1e-6)
      {
        for (unsigned rr = 0; rr < nsd_; ++rr)
        {
          normed_velgrad(rr) /= norm;
        }
      }
      else
      {
        normed_velgrad(0) = 1.;
        for (unsigned rr = 1; rr < nsd_; ++rr)
        {
          normed_velgrad(rr) = 0.0;
        }
      }

      // get length in this direction
      double val = 0.0;
      for (unsigned rr = 0; rr < nen_; ++rr) /* loop element nodes */
      {
        double loc = 0.0;
        for (unsigned idim = 0; idim < nsd_; idim++) loc += normed_velgrad(idim) * derxy_(idim, rr);

        val += fabs(loc);

        // val += fabs( normed_velgrad(0)*derxy_(0,rr)
        //            +normed_velgrad(1)*derxy_(1,rr)
        //            +normed_velgrad(2)*derxy_(2,rr));
      } /* end of loop over element nodes */

      hk = 2.0 / val;

      break;
    }
    default:
      FOUR_C_THROW("Unknown length");
      break;
  }  // switch reflength
  if (hk == 1.0e+10) FOUR_C_THROW("Something went wrong!");

  return hk;
}


/*----------------------------------------------------------------------*
 | output of model parameters                           rasthofer 09/12 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::store_model_parameters_for_output(
    const Core::Elements::Element* ele, const bool isowned, Teuchos::ParameterList& turbulencelist,
    const int nlayer)
{
  if (isowned)
  {
    if (turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
    {
      if (turbulencelist.get<std::string>("TURBULENCE_APPROACH", "none") == "CLASSICAL_LES")
      {
        if (turbulencelist.get<std::string>("CANONICAL_FLOW", "no") == "channel_flow_of_height_2" or
            turbulencelist.get<std::string>("CANONICAL_FLOW", "no") ==
                "scatra_channel_flow_of_height_2" or
            turbulencelist.get<std::string>("CANONICAL_FLOW", "no") ==
                "loma_channel_flow_of_height_2")
        {
          // calculate Prt from (Cs*h)^2/Prt
          if (tpn_ > 1.0E-16)
          {
            // get dynamically estimated Smagorinsky constant from fluid element, i.e., (Cs*h)^2
            double Cs_delta_sq = (*(turbulencelist.get<std::shared_ptr<std::vector<double>>>(
                "global_Cs_delta_sq_sum")))[nlayer];
            // since Cs_delta_sq contains the sum over all elements of this layer,
            // we have to divide by the number of elements of this layer
            int numele_layer = turbulencelist.get<int>("numele_layer");
            (*(turbulencelist.get<std::shared_ptr<std::vector<double>>>(
                "local_Prt_sum")))[nlayer] += (Cs_delta_sq / numele_layer) / tpn_;
          }
          else
            (*(turbulencelist.get<std::shared_ptr<std::vector<double>>>(
                "local_Prt_sum")))[nlayer] += 0.0;

          // set (Cs*h)^2/Prt and diffeff for output
          (*(turbulencelist.get<std::shared_ptr<std::vector<double>>>(
              "local_Cs_delta_sq_Prt_sum")))[nlayer] += tpn_;
          if (numscal_ > 1) FOUR_C_THROW("One scalar assumed for dynamic Smagorinsky model!");

          // calculation of effective diffusion coefficient
          const double vol = eval_shape_func_and_derivs_at_ele_center();

          // get material  at element center
          // density at t_(n)
          std::vector<double> densn(numscal_, 1.0);
          // density at t_(n+1) or t_(n+alpha_F)
          std::vector<double> densnp(numscal_, 1.0);
          // density at t_(n+alpha_M)
          std::vector<double> densam(numscal_, 1.0);

          // fluid viscosity
          double visc(0.0);

          set_internal_variables_for_mat_and_rhs();

          get_material_params(ele, densn, densnp, densam, visc);

          calc_subgr_diff(visc, vol, 0, densnp[0]);

          (*(turbulencelist.get<std::shared_ptr<std::vector<double>>>(
              "local_diffeff_sum")))[nlayer] += diffmanager_->get_isotropic_diff(0);
        }
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | additional output for turbulent channel flow         rasthofer 11/12 |
 | dissipation introduced by stabilization and turbulence models        |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_dissipation(
    Teuchos::ParameterList& params,            //!< parameter list
    Core::Elements::Element* ele,              //!< pointer to element
    Core::FE::Discretization& discretization,  //!< scatra discretization
    Core::Elements::LocationArray& la          //!< location array
)
{
  // do some checks first
  if (numscal_ != 1 or numdofpernode_ != 1)
    FOUR_C_THROW("calc_dissipation only for one scalar field!");

  //----------------------------------------------------------------------
  // preliminary set-up of parameters
  // ---------------------------------------------------------------------

  // as the scalar field is constant in the turbulent inflow section
  // we do not need any turbulence model
  if (turbparams_->turb_inflow())
    FOUR_C_THROW("calc_dissipation in combination with inflow generation not supported!");
  //    if (params.get<bool>("turbulent inflow",false))
  //    {
  //      if (ScaTra::inflow_element(ele))
  //        turbmodel_ = Inpar::FLUID::no_model;
  //    }

  // set time integration
  if (scatraparatimint_->is_stationary()) FOUR_C_THROW("Turbulence is instationary!");

  // set turbulent Prandt number to value given in parameterlist
  tpn_ = turbparams_->tpn();

  // if we have a dynamic model,we overwrite this value by a local element-based one here
  if (turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
  {
    Teuchos::ParameterList& turbulencelist = params.sublist("TURBULENCE MODEL");
    // remark: for dynamic estimation, this returns (Cs*h)^2 / Pr_t
    std::shared_ptr<Core::LinAlg::Vector<double>> ele_prt =
        turbulencelist.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("col_ele_Prt");
    const int id = ele->lid();
    tpn_ = ele_prt->local_values_as_span()[id];

    int dummy = 0;
    // when no averaging was done, we just keep the calculated (clipped) value
    if (turbparams_->cs_av())
      get_mean_prt_of_homogeneous_direction(params.sublist("TURBULENCE MODEL"), dummy);
  }

  //----------------------------------------------------------------------
  // get all nodal values
  // ---------------------------------------------------------------------

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatrapara_->nds_vel();

  // get velocity values at nodes
  const std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");

  // safety check
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (unsigned inode = 0; inode < nen_; ++inode)
    for (unsigned idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  // extract local values of convective velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvelnp_, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->rotate_my_values_if_necessary(econvelnp_);

  // set econvelnp_ equal to evelnp_ since ale is not supported
  evelnp_ = econvelnp_;

  // get data required for subgrid-scale velocity: acceleration and pressure
  if (scatrapara_->rb_sub_gr_vel())
  {
    // get acceleration values at nodes
    const std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
        discretization.get_state(ndsvel, "acceleration field");
    if (acc == nullptr) FOUR_C_THROW("Cannot get state vector acceleration field");

    // extract local values of acceleration field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*acc, eaccnp_, lmvel);

    // rotate the vector field in the case of rotationally symmetric boundary conditions
    rotsymmpbc_->rotate_my_values_if_necessary(eaccnp_);

    // construct location vector for pressure dofs
    std::vector<int> lmpre(nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      lmpre[inode] = la[ndsvel].lm_[inode * numveldofpernode + nsd_];

    // extract local values of pressure field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*convel, eprenp_, lmpre);
  }

  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (hist == nullptr || phinp == nullptr)
    FOUR_C_THROW("Cannot get state vector 'hist' and/or 'phinp'");
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*hist, ehist_, la[0].lm_);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp_, la[0].lm_);

  // reset to zero; used in get_material_params if not incremental -> used to calculate densn which
  // is not required here
  for (int k = 0; k < numdofpernode_; ++k) ephin_[k].clear();

  // get fine-scale values
  if (turbparams_->which_fssgd() == Inpar::ScaTra::fssugrdiff_smagorinsky_small or
      turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    // get fine scale scalar field
    std::shared_ptr<const Core::LinAlg::Vector<double>> gfsphinp =
        discretization.get_state("fsphinp");
    if (gfsphinp == nullptr) FOUR_C_THROW("Cannot get state vector 'fsphinp'");

    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*gfsphinp, fsphinp_, la[0].lm_);

    // get fine-scale velocity at nodes
    const std::shared_ptr<const Core::LinAlg::Vector<double>> fsvelocity =
        discretization.get_state(ndsvel, "fine-scale velocity field");
    if (fsvelocity == nullptr)
      FOUR_C_THROW("Cannot get fine-scale velocity field from scatra discretization!");

    // extract local values of fine-scale velocity field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*fsvelocity, efsvel_, lmvel);

    // rotate the vector field in the case of rotationally symmetric boundary conditions
    rotsymmpbc_->rotate_my_values_if_necessary(efsvel_);
  }

  //----------------------------------------------------------------------
  // prepare mean values
  // ---------------------------------------------------------------------

  // the coordinates of the element layers in the channel
  // planecoords are named nodeplanes in turbulence_statistics_channel!
  std::shared_ptr<std::vector<double>> planecoords =
      params.get<std::shared_ptr<std::vector<double>>>("planecoords_", nullptr);
  if (planecoords == nullptr)
    FOUR_C_THROW("planecoords is null, but need channel_flow_of_height_2\n");

  // this will be the y-coordinate of a point in the element interior
  double center = 0.0;
  // get node coordinates of element
  for (unsigned inode = 0; inode < nen_; inode++) center += xyze_(1, inode);

  center /= nen_;

  // working arrays for the quantities we want to compute
  double vol = 0.0;

  double averaged_tauS = 0.0;

  double mean_resS = 0.0;
  double mean_resS_sq = 0.0;

  double eps_visc = 0.0;
  double eps_conv = 0.0;
  double eps_smag = 0.0;
  double eps_avm3 = 0.0;
  double eps_mfs = 0.0;
  double eps_mfscross = 0.0;
  double eps_mfsrey = 0.0;
  double eps_supg = 0.0;
  double eps_cross = 0.0;
  double eps_rey = 0.0;

  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  vol = eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // get material parameters (evaluation at element center)
  //----------------------------------------------------------------------

  // density at t_(n)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // material parameter at the element center are also necessary
  // even if the stabilization parameter is evaluated at the element center
  if (not scatrapara_->mat_gp() or scatrapara_->tau_gp())
  {
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  //----------------------------------------------------------------------
  // calculation of subgrid diffusivity and stabilization parameter(s)
  // at element center
  //----------------------------------------------------------------------

  // the stabilization parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);
  // subgrid-scale diffusion coefficient
  double sgdiff(0.0);

  if (not scatrapara_->tau_gp())
  {
    // calculation of all-scale subgrid diffusivity (by, e.g.,
    // Smagorinsky model) at element center
    if (turbparams_->turb_model() == Inpar::FLUID::smagorinsky or
        turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky or
        turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
    {
      calc_subgr_diff(visc, vol, 0, densnp[0]);
    }

    // calculation of fine-scale artificial subgrid diffusivity at element center
    // we have to set a vector, which is however only required for one
    // special case (computation of fine-scale subgrid diffusivity for non-incremental
    // solver -> only artificial subgrid diffusivity) not considered here
    Core::LinAlg::SerialDenseVector elevec1_subgrdiff_dummy;
    if (turbparams_->fssgd())
      calc_fine_scale_subgr_diff(sgdiff, elevec1_subgrdiff_dummy, ele, vol, 0, densnp[0],
          diffmanager_->get_isotropic_diff(0), scatravarmanager_->con_vel(0));

    // calculation of stabilization parameter at element center
    calc_tau(tau[0], diffmanager_->get_isotropic_diff(0),
        reamanager_->get_stabilization_coeff(0, scatravarmanager_->phinp(0)), densnp[0],
        scatravarmanager_->con_vel(0), vol);
  }


  // prepare multifractal subgrid-scale modeling
  // calculation of model coefficients B (velocity) and D (scalar)
  // at element center
  // coefficient B of fine-scale velocity
  Core::LinAlg::Matrix<nsd_, 1> B_mfs(Core::LinAlg::Initialization::zero);
  // coefficient D of fine-scale scalar
  double D_mfs = 0.0;
  if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
  {
    if (not turbparams_->bd_gp())
    {
      // make sure to get material parameters at element center
      // hence, determine them if not yet available
      if (scatrapara_->mat_gp())
      {
        set_internal_variables_for_mat_and_rhs();

        get_material_params(ele, densn, densnp, densam, visc);
      }
      // provide necessary velocities and gradients at element center
      // get velocity at element center
      Core::LinAlg::Matrix<nsd_, 1> fsvelint(Core::LinAlg::Initialization::zero);
      fsvelint.multiply(efsvel_, funct_);

      // calculate model coefficients
      for (int k = 0; k < numscal_; ++k)  // loop of each transported scalar
        calc_b_and_d_for_multifrac_subgrid_scales(B_mfs, D_mfs, vol, k, densnp[0],
            diffmanager_->get_isotropic_diff(k), visc, scatravarmanager_->con_vel(k), fsvelint);
    }
  }

  // get body force
  body_force(ele);

  //----------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    //---------------------------------------------------------------
    // evaluate shape functions and derivatives at integration point
    //---------------------------------------------------------------
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------

    set_internal_variables_for_mat_and_rhs();

    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc);

    // get velocity at integration point
    const Core::LinAlg::Matrix<nsd_, 1>& convelint = scatravarmanager_->con_vel(0);

    // scalar at integration point at time step n+1
    const double& phinp = scatravarmanager_->phinp(0);

    // gradient of current scalar value at integration point
    Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
    gradphi.multiply(derxy_, ephinp_[0]);

    // reactive part of the form: (reaction coefficient)*phi
    const double rea_phi = densnp[0] * phinp * reamanager_->get_rea_coeff(0);

    // get fine-scale velocity and its derivatives at integration point
    Core::LinAlg::Matrix<nsd_, 1> fsvelint(Core::LinAlg::Initialization::zero);
    if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
      fsvelint.multiply(efsvel_, funct_);

    // compute gradient of fine-scale part of scalar value
    Core::LinAlg::Matrix<nsd_, 1> fsgradphi(Core::LinAlg::Initialization::zero);
    if (turbparams_->fssgd()) fsgradphi.multiply(derxy_, fsphinp_[0]);

    double rhsint(0.0);
    get_rhs_int(rhsint, densnp[0], 0);

    //--------------------------------------------------------------------
    // calculation of (fine-scale) subgrid diffusivity, subgrid-scale
    // velocity and stabilization parameter(s) at integration point
    //--------------------------------------------------------------------

    // subgrid-scale convective term
    Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::zero);
    // subgrid-scale velocity vector in gausspoint
    Core::LinAlg::Matrix<nsd_, 1> sgvelint(Core::LinAlg::Initialization::zero);

    if (scatrapara_->tau_gp())
    {
      // artificial diffusion / shock capturing: adaption of diffusion coefficient
      if (scatrapara_->assgd()) FOUR_C_THROW("Not supported");

      // calculation of all-scale subgrid diffusivity (by, e.g.,
      // Smagorinsky model) at element center
      if (turbparams_->turb_model() == Inpar::FLUID::smagorinsky or
          turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky or
          turbparams_->turb_model() == Inpar::FLUID::dynamic_vreman)
      {
        calc_subgr_diff(visc, vol, 0, densnp[0]);
      }

      // calculation of fine-scale artificial subgrid diffusivity at element center
      // we have to set a vector, which is however only required for one
      // special case (computation of fine-scale subgrid diffusivity for non-incremental
      // solver -> only artificial subgrid diffusivity) not considered here
      Core::LinAlg::SerialDenseVector elevec1_subgrdiff_dummy;
      if (turbparams_->fssgd())
        calc_fine_scale_subgr_diff(sgdiff, elevec1_subgrdiff_dummy, ele, vol, 0, densnp[0],
            diffmanager_->get_isotropic_diff(0), scatravarmanager_->con_vel(0));

      // calculation of subgrid-scale velocity at integration point if required
      if (scatrapara_->rb_sub_gr_vel())
      {
        // calculation of stabilization parameter related to fluid momentum
        // equation at integration point
        calc_tau(tau[0], visc, 0.0, densnp[0], convelint, vol);
        // calculation of residual-based subgrid-scale velocity
        calc_subgr_velocity(ele, sgvelint, densam[0], densnp[0], visc, convelint, tau[0]);

        // calculation of subgrid-scale convective part
        sgconv.multiply_tn(derxy_, sgvelint);
      }

      // calculation of stabilization parameter at integration point
      calc_tau(tau[0], diffmanager_->get_isotropic_diff(0),
          reamanager_->get_stabilization_coeff(0, scatravarmanager_->phinp(0)), densnp[0],
          convelint, vol);
    }

    // prepare multifractal subgrid-scale modeling
    // calculation of model coefficients B (velocity) and D (scalar)
    // at Gauss point as well as calculation
    // of multifractal subgrid-scale quantities
    Core::LinAlg::Matrix<nsd_, 1> mfsgvelint(Core::LinAlg::Initialization::zero);
    double mfsvdiv(0.0);
    double mfssgphi(0.0);
    Core::LinAlg::Matrix<nsd_, 1> mfsggradphi(Core::LinAlg::Initialization::zero);
    if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      if (turbparams_->bd_gp())
        // calculate model coefficients
        calc_b_and_d_for_multifrac_subgrid_scales(B_mfs, D_mfs, vol, 0, densnp[0],
            diffmanager_->get_isotropic_diff(0), visc, convelint, fsvelint);

      // calculate fine-scale velocity, its derivative and divergence for multifractal subgrid-scale
      // modeling
      for (unsigned idim = 0; idim < nsd_; idim++)
        mfsgvelint(idim, 0) = fsvelint(idim, 0) * B_mfs(idim, 0);
      // required for conservative formulation in the context of passive scalar transport
      if (turbparams_->mfs_conservative() or scatrapara_->is_conservative())
      {
        // get divergence of subgrid-scale velocity
        Core::LinAlg::Matrix<nsd_, nsd_> mfsvderxy;
        mfsvderxy.multiply_nt(efsvel_, derxy_);
        for (unsigned idim = 0; idim < nsd_; idim++)
          mfsvdiv += mfsvderxy(idim, idim) * B_mfs(idim, 0);
      }

      // calculate fine-scale scalar and its derivative for multifractal subgrid-scale modeling
      mfssgphi = D_mfs * funct_.dot(fsphinp_[0]);
      mfsggradphi.multiply(derxy_, fsphinp_[0]);
      mfsggradphi.scale(D_mfs);
    }

    // residual of convection-diffusion-reaction eq
    double scatrares(0.0);

    // compute residual of scalar transport equation and
    // subgrid-scale part of scalar
    calc_strong_residual(0, scatrares, densam[0], densnp[0], rea_phi, rhsint, tau[0]);

    //--------------------------------------------------------------------
    // calculation of subgrid-scale part of scalar
    //--------------------------------------------------------------------
    double sgphi = -tau[0] * scatrares;

    // not supported anymore
    // update material parameters based on inclusion of subgrid-scale
    // part of scalar (active only for mixture fraction,
    // Sutherland law and progress variable, for the time being)
    //    if (update_mat_)
    //    {
    //      if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    //        update_material_params(ele,mfssgphi_[0],0);
    //      else
    //        update_material_params(ele,sgphi_[0],0);
    //    }
    // this yields updated material parameters (dens, visc, diffus)
    // scatrares_ and sgphi_ are not updated, since they are used
    // for the contributions of the stabilizations, for which we
    // do not use updated material parameters

    //    if (ele->Id()==100)
    //    {
    //      std::cout << "densnp_[0] " << densnp_[0] << std::endl;
    //      std::cout << "sgphi_[0] " << sgphi_[0] << std::endl;
    //      std::cout << "convelint_ " << convelint_ << std::endl;
    //      std::cout << "gradphi_ " << gradphi_ << std::endl;
    //      std::cout << "phi_[0] " << phi_[0] << std::endl;
    //      std::cout << "sgphi_[0] " << sgphi_[0] << std::endl;
    //      std::cout << "sgvelint_ " << sgvelint_ << std::endl;
    //      std::cout << "mfssgphi_[0] " << mfssgphi_[0] << std::endl;
    //      std::cout << "mfsgvelint_ " << mfsgvelint_ << std::endl;
    //      std::cout << "sgdiff_[0] " << sgdiff_[0] << std::endl;
    //      std::cout << "fsgradphi_ " << fsgradphi_ << std::endl;
    //      std::cout << "diffus_[0] " << diffus_[0] << std::endl;
    //      std::cout << "tau_[0] " << tau_[0] << std::endl;
    //      std::cout << "scatrares_[0] " << scatrares_[0] << std::endl;
    //    }

    //---------------------------------------------------------------
    // element average dissipation and production rates
    //---------------------------------------------------------------

    //---------------------------------------------------------------
    // residual-based subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation by supg-stabilization
    if (scatrapara_->stab_type() == Inpar::ScaTra::stabtype_SUPG)
    {
      eps_supg -= densnp[0] * sgphi * convelint.dot(gradphi) * fac;  // sgphi_ is negative
    }
    else if (scatrapara_->stab_type() == Inpar::ScaTra::stabtype_no_stabilization)
    {
      // nothing to do
    }
    else
      FOUR_C_THROW("Stabtype not yet supported!");

    // dissipation by cross-stress-stabilization
    // dissipation by reynolds-stress-stabilization
    if (scatrapara_->rb_sub_gr_vel())
    {
      eps_cross -= densnp[0] * phinp * sgvelint.dot(gradphi) * fac;
      eps_rey -= densnp[0] * sgphi * sgvelint.dot(gradphi) * fac;
    }

    //---------------------------------------------------------------
    // multifractal subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation multifractal subgrid-scales
    if (turbparams_->turb_model() == Inpar::FLUID::multifractal_subgrid_scales)
    {
      eps_mfs -= densnp[0] *
                 (mfssgphi * convelint.dot(gradphi) + phinp * mfsgvelint.dot(gradphi) +
                     mfssgphi * mfsgvelint.dot(gradphi)) *
                 fac;
      eps_mfscross -=
          densnp[0] * (mfssgphi * convelint.dot(gradphi) + phinp * mfsgvelint.dot(gradphi)) * fac;
      eps_mfsrey -= densnp[0] * mfssgphi * mfsgvelint.dot(gradphi) * fac;
    }

    //---------------------------------------------------------------
    // small-scale subgrid-viscosity subgrid-scale modeling terms
    //---------------------------------------------------------------

    // dissipation AVM3
    if (turbparams_->fssgd())
    {
      eps_avm3 += sgdiff * fsgradphi.dot(fsgradphi) * fac;
    }

    //---------------------------------------------------------------
    // Smagorinsky model
    //---------------------------------------------------------------

    // dissipation (Smagorinsky)
    if (scatrapara_->assgd() or turbparams_->turb_model() == Inpar::FLUID::smagorinsky or
        turbparams_->turb_model() == Inpar::FLUID::dynamic_smagorinsky)
    {
      eps_smag += diffmanager_->get_sub_gr_diff(0) * gradphi.dot(gradphi) * fac;
    }

    //---------------------------------------------------------------
    // standard Galerkin terms
    //---------------------------------------------------------------

    // convective (Galerkin)
    eps_conv -= densnp[0] * phinp * convelint.dot(gradphi) * fac;

    // dissipation (Galerkin) (diffus is diffus+sgdiff here)
    eps_visc += (diffmanager_->get_isotropic_diff(0) - diffmanager_->get_sub_gr_diff(0)) *
                gradphi.dot(gradphi) * fac;

    //---------------------------------------------------------------
    // element averages of tau_S and residual
    //---------------------------------------------------------------
    averaged_tauS += tau[0] * fac;
    mean_resS += scatrares * fac;
    mean_resS_sq += scatrares * scatrares * fac;

  }  // end loop integration points

  mean_resS /= vol;
  mean_resS_sq /= vol;

  averaged_tauS /= vol;

  eps_visc /= vol;
  eps_conv /= vol;
  eps_smag /= vol;
  eps_avm3 /= vol;
  eps_mfs /= vol;
  eps_mfscross /= vol;
  eps_mfsrey /= vol;
  eps_supg /= vol;
  eps_cross /= vol;
  eps_rey /= vol;


  std::shared_ptr<std::vector<double>> incrvol =
      params.get<std::shared_ptr<std::vector<double>>>("incrvol");

  std::shared_ptr<std::vector<double>> incr_eps_visc =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_visc");
  std::shared_ptr<std::vector<double>> incr_eps_conv =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_conv");
  std::shared_ptr<std::vector<double>> incr_eps_smag =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_eddyvisc");
  std::shared_ptr<std::vector<double>> incr_eps_avm3 =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_avm3");
  std::shared_ptr<std::vector<double>> incr_eps_mfs =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_mfs");
  std::shared_ptr<std::vector<double>> incr_eps_mfscross =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_mfscross");
  std::shared_ptr<std::vector<double>> incr_eps_mfsrey =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_mfsrey");
  std::shared_ptr<std::vector<double>> incr_eps_supg =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_supg");
  std::shared_ptr<std::vector<double>> incr_eps_cross =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_cross");
  std::shared_ptr<std::vector<double>> incr_eps_rey =
      params.get<std::shared_ptr<std::vector<double>>>("incr_scatra_eps_rey");

  std::shared_ptr<std::vector<double>> incrresS =
      params.get<std::shared_ptr<std::vector<double>>>("incrresS");
  std::shared_ptr<std::vector<double>> incrresS_sq =
      params.get<std::shared_ptr<std::vector<double>>>("incrresS_sq");

  std::shared_ptr<std::vector<double>> incrtauS =
      params.get<std::shared_ptr<std::vector<double>>>("incrtauS");

  bool found = false;

  int nlayer = 0;
  for (nlayer = 0; nlayer < (int)(*planecoords).size() - 1;)
  {
    if (center < (*planecoords)[nlayer + 1])
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

  // collect layer volume
  (*incrvol)[nlayer] += vol;

  // averages of stabilization parameters
  (*incrtauS)[nlayer] += averaged_tauS;

  // averages residual
  (*incrresS)[nlayer] += mean_resS;
  (*incrresS_sq)[nlayer] += mean_resS_sq;

  // averages dissipation
  (*incr_eps_visc)[nlayer] += eps_visc;
  (*incr_eps_conv)[nlayer] += eps_conv;
  (*incr_eps_smag)[nlayer] += eps_smag;
  (*incr_eps_avm3)[nlayer] += eps_avm3;
  (*incr_eps_mfs)[nlayer] += eps_mfs;
  (*incr_eps_mfscross)[nlayer] += eps_mfscross;
  (*incr_eps_mfsrey)[nlayer] += eps_mfsrey;
  (*incr_eps_supg)[nlayer] += eps_supg;
  (*incr_eps_cross)[nlayer] += eps_cross;
  (*incr_eps_rey)[nlayer] += eps_rey;

  return;
}

FOUR_C_NAMESPACE_CLOSE

// template classes

#include "4C_scatra_ele_calc.inst.hpp"
