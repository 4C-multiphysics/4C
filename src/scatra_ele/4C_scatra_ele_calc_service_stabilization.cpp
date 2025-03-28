// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_calc_utils.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  calculate stabilization parameter  (private)              gjb 06/08 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double vol                                 //!< element volume
)
{
  //----------------------------------------------------------------------
  // computation of stabilization parameters depending on definition used
  //----------------------------------------------------------------------
  switch (scatrapara_->tau_def())
  {
    case Inpar::ScaTra::tau_taylor_hughes_zarins:
    case Inpar::ScaTra::tau_taylor_hughes_zarins_wo_dt:
    {
      calc_tau_taylor_hughes_zarins(tau, diffus, reacoeff, densnp, convelint);
      break;
    }
    case Inpar::ScaTra::tau_franca_valentin:
    case Inpar::ScaTra::tau_franca_valentin_wo_dt:
    {
      calc_tau_franca_valentin(tau, diffus, reacoeff, densnp, convelint, vol);
      break;
    }
    case Inpar::ScaTra::tau_shakib_hughes_codina:
    case Inpar::ScaTra::tau_shakib_hughes_codina_wo_dt:
    {
      calc_tau_franca_shakib_codina(tau, diffus, reacoeff, densnp, convelint, vol);
      break;
    }
    case Inpar::ScaTra::tau_codina:
    case Inpar::ScaTra::tau_codina_wo_dt:
    {
      calc_tau_codina(tau, diffus, reacoeff, densnp, convelint, vol);
      break;
    }
    case Inpar::ScaTra::tau_franca_madureira_valentin:
    case Inpar::ScaTra::tau_franca_madureira_valentin_wo_dt:
    {
      calc_tau_franca_madureira_valentin(tau, diffus, reacoeff, densnp, vol);
      break;
    }
    case Inpar::ScaTra::tau_exact_1d:
    {
      calc_tau_1d_exact(tau, diffus, reacoeff, densnp, convelint, vol);
      break;
    }
    case Inpar::ScaTra::tau_zero:
    {
      // set tau's to zero (-> no stabilization effect)
      tau = 0.0;
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown definition for stabilization parameter tau\n");
      break;
    }
  }

  return;
}  // ScaTraEleCalc::CalcTau

/*----------------------------------------------------------------------*
 |  calculation of tau according to Taylor, Hughes and Zarins  vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_taylor_hughes_zarins(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint  //!< convective velocity at integration point
)
{
  /*
  literature:
  1) C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
  of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
  (1998) 155-196.
  2) V. Gravemeier, W.A. Wall, An algebraic variational multiscale-
  multigrid method for large-eddy simulation of turbulent variable-
  density flow at low Mach number, J. Comput. Phys. 229 (2010)
  6047-6070.
  -> version for variable-density scalar transport equation as
  implemented here, which corresponds to constant-density
  version as given in the previous publication when density
  is constant

                                                                  1
            +-                                               -+ - -
            |        2                                        |   2
            | c_1*rho                                  2      |
  tau = C * | -------   +  c_2*rho*u*G*rho*u  +  c_3*mu *G:G  |
            |     2                                           |
            |   dt                                            |
            +-                                               -+

  with the constants and covariant metric tensor defined as follows:

  C   = 1.0 (not explicitly defined here),
  c_1 = 4.0,
  c_2 = 1.0 (not explicitly defined here),
  c_3 = 12.0/m_k (36.0 for linear and 144.0 for quadratic elements)

         +-           -+   +-           -+   +-           -+
         |             |   |             |   |             |
         |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
   G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
    ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
         |    i     j  |   |    i     j  |   |    i     j  |
         +-           -+   +-           -+   +-           -+

          +----
           \
  G : G =   +   G   * G
           /     ij    ij
          +----
          i,j
                     +----
                     \
  rho*u*G*rho*u  =   +   rho*u * G  *rho*u
                     /        i   ij      j
                     +----
                      i,j
  */

  // get element-type constant for tau
  const double mk = ScaTra::mk<distype>();

  // effective velocity at element center:
  // (weighted) convective velocity + individual migration velocity
  Core::LinAlg::Matrix<nsd_, 1> veleff(convelint);

  // total reaction coefficient sigma_tot: sum of "artificial" reaction
  // due to time factor and reaction coefficient (reaction coefficient
  // ensured to be zero in get_material_params for non-reactive material)
  double sigma_tot = reacoeff;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_taylor_hughes_zarins)
    sigma_tot += 1.0 / scatraparatimint_->dt();

  // computation of various values derived from covariant metric tensor
  double G;
  double normG(0.0);
  double Gnormu(0.0);
  const double dens_sqr = densnp * densnp;
  for (unsigned nn = 0; nn < nsd_; ++nn)
  {
    for (unsigned rr = 0; rr < nsd_; ++rr)
    {
      G = xij_(nn, 0) * xij_(rr, 0);
      for (unsigned tt = 1; tt < nsd_; tt++)
      {
        G += xij_(nn, tt) * xij_(rr, tt);
      }
      normG += G * G;
      Gnormu += dens_sqr * veleff(nn, 0) * G * veleff(rr, 0);
    }
  }

  // definition of constants as described above
  const double c1 = 4.0;
  const double c3 = 12.0 / mk;

  // compute diffusive part
  const double Gdiff = c3 * diffus * diffus * normG;

  // computation of stabilization parameter tau
  tau = 1.0 / (sqrt(c1 * dens_sqr * ((sigma_tot) * (sigma_tot)) + Gnormu + Gdiff));

  return;
}


/*----------------------------------------------------------------------*
 |  calculation of tau according to Franca and Valentin        vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_franca_valentin(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double vol                                 //!< element volume
)
{
  /*

  literature:
  L.P. Franca, F. Valentin, On an improved unusual stabilized
  finite element method for the advective-reactive-diffusive
  equation, Comput. Methods Appl. Mech. Engrg. 190 (2000) 1785-1800.


  xi1,xi2 ^
  |      /
  |     /
  |    /
  1 +---+
  |
  |
  |
  +--------------> re1,re2
  1

  */

  // get element-type constant for tau
  const double mk = ScaTra::mk<distype>();

  // get Euclidean norm of (weighted) velocity at element center
  double vel_norm;
  vel_norm = convelint.norm2();

  // total reaction coefficient sigma_tot: sum of "artificial" reaction
  // due to time factor and reaction coefficient (reaction coefficient
  // ensured to be zero in get_material_params for non-reactive material)
  double sigma_tot = reacoeff;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_franca_valentin)
    sigma_tot += 1.0 / scatraparatimint_->time_fac();

  // calculate characteristic element length
  const double h = calc_char_ele_length(vol, vel_norm, convelint);

  // various parameter computations:
  // relating convective to viscous part
  const double epe = mk * densnp * vel_norm * h;
  // relating viscous to reactive part
  double epe1 = 0.0;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_franca_valentin or reacoeff != 0.0)
    epe1 = 2.0 * diffus / (mk * densnp * sigma_tot * ((h) * (h)));

  // respective "switching" parameters
  const double xi = std::max(epe, 1.0 * diffus);
  const double xi1 = std::max(epe1, 1.0);

  tau = ((h) * (h)) / (((h) * (h)) * densnp * sigma_tot * xi1 + 2.0 * xi / mk);

  return;
}

/*----------------------------------------------------------------------*
 |  calculation of tau according to Franca, Shakib and Codina  vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_franca_shakib_codina(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double vol                                 //!< element volume
)
{
  /*
  literature:
  1) F. Shakib, Finite element analysis of the compressible Euler and
  Navier-Stokes equations, PhD thesis, Division of Applied Mechanics,
  Stanford University, Stanford, CA, USA, 1989.
  2) F. Shakib, T.J.R. Hughes, A new finite element formulation for
  computational fluid dynamics: IX. Fourier analysis of space-time
  Galerkin/least-squares algorithms, Comput. Methods Appl. Mech.
  Engrg. 87 (1991) 35-58.
  3) R. Codina, Stabilized finite element approximation of transient
  incompressible flows using orthogonal subscales, Comput. Methods
  Appl. Mech. Engrg. 191 (2002) 4295-4321.

  All those proposed definitions were for non-reactive incompressible
  flow; they are adapted to potentially reactive scalar transport
  equations with potential density variations here.

  constants defined as in Shakib (1989) / Shakib and Hughes (1991),
  merely slightly different with respect to c_3:

  c_1 = 4.0,
  c_2 = 4.0,
  c_3 = 4.0/(m_k*m_k) (36.0 for linear, 576.0 for quadratic ele.)

  Codina (2002) proposed present version without dt and explicit
  definition of constants
  (condition for constants as defined here: c_2 <= sqrt(c_3)).

  */

  // get element-type constant for tau
  const double mk = ScaTra::mk<distype>();

  // get Euclidean norm of velocity
  const double vel_norm = convelint.norm2();

  // total reaction coefficient sigma_tot: sum of "artificial" reaction
  // due to time factor and reaction coefficient (reaction coefficient
  // ensured to be zero in get_material_params for non-reactive material)
  double sigma_tot = reacoeff;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_shakib_hughes_codina)
    sigma_tot += 1.0 / scatraparatimint_->dt();

  // calculate characteristic element length
  const double h = calc_char_ele_length(vol, vel_norm, convelint);

  // definition of constants as described above
  const double c1 = 4.0;
  const double c2 = 4.0;
  const double c3 = 4.0 / (mk * mk);
  // alternative value as proposed in Shakib (1989): c3 = 16.0/(mk*mk);

  if (densnp == 0.0)
  {
    tau = 0.0;
  }
  else
  {
    tau = 1.0 / (sqrt(c1 * ((densnp) * (densnp)) * ((sigma_tot) * (sigma_tot)) +
                      c2 * ((densnp) * (densnp)) * ((vel_norm) * (vel_norm)) / ((h) * (h)) +
                      c3 * ((diffus) * (diffus)) / (((h) * (h)) * ((h) * (h)))));
  }
}


/*----------------------------------------------------------------------*
 |  calculation of tau according to Codina                     vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_codina(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double vol                                 //!< element volume
)
{
  /*
  literature:
  R. Codina, Comparison of some finite element methods for solving
  the diffusion-convection-reaction equation, Comput. Methods
  Appl. Mech. Engrg. 156 (1998) 185-210.

  constants:
  c_1 = 1.0,
  c_2 = 2.0,
  c_3 = 4.0/m_k (12.0 for linear, 48.0 for quadratic elements)

  Codina (1998) proposed present version without dt.

  */

  // get element-type constant for tau
  const double mk = ScaTra::mk<distype>();

  // get Euclidean norm of velocity
  const double vel_norm = convelint.norm2();

  // total reaction coefficient sigma_tot: sum of "artificial" reaction
  // due to time factor and reaction coefficient (reaction coefficient
  // ensured to be zero in get_material_params for non-reactive material)
  double sigma_tot = reacoeff;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_codina)
    sigma_tot += 1.0 / scatraparatimint_->dt();

  // calculate characteristic element length
  const double h = calc_char_ele_length(vol, vel_norm, convelint);

  // definition of constants as described above
  const double c1 = 1.0;
  const double c2 = 2.0;
  const double c3 = 4.0 / mk;

  tau = 1.0 / (c1 * densnp * sigma_tot + c2 * densnp * vel_norm / h + c3 * diffus / (h * h));

  return;
}

/*---------------------------------------------------------------------------*
 |  calculation of tau according to Franca, Madureira and Valentin  vg 01/11 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_franca_madureira_valentin(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const double vol        //!< element volume
)
{
  /*
  This stabilization parameter is only intended to be used for
  reactive-diffusive problems such as structure-based scalar
  transport problems in case of potentially dominating reaction.

  literature:
  L.P. Franca, A.L. Madureira, F. Valentin, Towards multiscale
  functions: enriching finite element spaces with local but not
  bubble-like functions, Comput. Methods Appl. Mech. Engrg. 194
  (2005) 3006-3021.

  */

  // get element-type constant for tau
  const double mk = ScaTra::mk<distype>();

  // total reaction coefficient sigma_tot: sum of "artificial" reaction
  // due to time factor and reaction coefficient (reaction coefficient
  // ensured to be zero in get_material_params for non-reactive material)
  double sigma_tot = reacoeff;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_franca_madureira_valentin)
    sigma_tot += 1.0 / scatraparatimint_->time_fac();

  // calculate characteristic element length
  // -> currently: cubic/square root of element volume/area or
  //    element length (3-/2-/1-D)
  // cast dimension to a double variable -> pow()
  const double dim = (double)nsd_;
  const double h = std::pow(vol, 1 / dim);

  // parameter relating reactive to diffusive part
  double epe = 0.0;
  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_franca_madureira_valentin or reacoeff != 0.0)
    epe = 2.0 * diffus / (mk * densnp * sigma_tot * ((h) * (h)));

  // respective "switching" parameter
  const double xi = std::max(epe, 1.0);

  // constant c_u as suggested in Badia and Codina (2010), method A
  // is set to be 1.0 here as in Franca et al. (2005)
  // alternative: 4.0 as suggested in Badia and Codina (2010) for
  // Darcy flow
  const double c_u = 1.0;

  if (scatrapara_->tau_def() == Inpar::ScaTra::tau_franca_madureira_valentin or reacoeff != 0.0)
    tau = ((h) * (h)) / (c_u * ((h) * (h)) * densnp * sigma_tot * xi + (2.0 * diffus / mk));

  return;
}


/*----------------------------------------------------------------------*
 |  exact calculation of tau for 1D                            vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_tau_1d_exact(
    double& tau,            //!< the stabilisation parameters (one per transported scalar)
    const double diffus,    //!< diffusivity or viscosity
    const double reacoeff,  //!< reaction coefficient
    const double densnp,    //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double vol                                 //!< element volume
)
{
  // get number of dimensions (convert from int to double)
  const double dim = (double)nsd_;

  // get characteristic element length
  double h = std::pow(vol, (1.0 / dim));  // equals streamlength in 1D

  // get Euclidean norm of (weighted) velocity at element center
  double vel_norm(0.0);
  vel_norm = convelint.norm2();

  if (diffus < 1e-14) FOUR_C_THROW("Invalid diffusion coefficient");
  double epe = 0.5 * densnp * vel_norm * h / diffus;

  const double pp = exp(epe);
  const double pm = exp(-epe);
  double xi = 0.0;
  if (epe >= 700.0)
    tau = 0.5 * h / vel_norm;
  else if (epe < 700.0 and epe > 1e-15)
  {
    xi = (((pp + pm) / (pp - pm)) - (1.0 / epe));  // xi = coth(epe) - 1/epe
    // compute optimal stabilization parameter
    tau = 0.5 * h * xi / vel_norm;
  }
  else
    tau = 0.0;

  return;
}


/*----------------------------------------------------------------------*
 |  calculation of characteristic element length               vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_char_ele_length(
    const double vol,                               //!< element volume
    const double vel_norm,                          //!< norm of velocity
    const Core::LinAlg::Matrix<nsd_, 1>& convelint  //!< convective velocity at integration point
)
{
  // define and initialize streamlength
  double h = 0.0;

  //---------------------------------------------------------------------
  // select from various definitions for characteristic element length
  //---------------------------------------------------------------------
  switch (scatrapara_->char_ele_length())
  {
    // a) streamlength due to Tezduyar et al. (1992) -> default
    // normed velocity vector
    case Inpar::ScaTra::streamlength:
    {
      Core::LinAlg::Matrix<nsd_, 1> velino(Core::LinAlg::Initialization::zero);
      if (vel_norm >= 1e-6)
        velino.update(1.0 / vel_norm, convelint);
      else
      {
        velino.clear();
        velino(0, 0) = 1.0;
      }

      // get streamlength using the normed velocity at element centre
      Core::LinAlg::Matrix<nen_, 1> tmp;
      tmp.multiply_tn(derxy_, velino);
      const double val = tmp.norm1();
      h = 2.0 / val;  // h=streamlength
    }
    break;

    // b) volume-equivalent diameter (warning: 3-D formula!)
    case Inpar::ScaTra::volume_equivalent_diameter:
    {
      h = std::pow((6. * vol / M_PI), (1.0 / 3.0)) / sqrt(3.0);
    }
    break;

    // c) cubic/square root of element volume/area or element length (3-/2-/1-D)
    case Inpar::ScaTra::root_of_volume:
    {
      // cast dimension to a double variable -> pow()
      const double dim = double(nsd_ele_);
      h = std::pow(vol, 1.0 / dim);
    }
    break;

    default:
      FOUR_C_THROW("unknown characteristic element length\n");
      break;
  }  // switch (charelelength_)

  return h;
}


/*----------------------------------------------------------------------*
 |  calculate artificial diffusivity                           vg 10/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_artificial_diff(
    const double vol,                                //!< element volume
    const int k,                                     //!< id of current scalar
    const double densnp,                             //!< density at t_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi,    //!< scalar gradient
    const double conv_phi,                           //!< convective contribution
    const double scatrares,  //!< residual of convection-diffusion-reaction eq
    const double tau         //!< the stabilisation parameter
)
{
  // get number of dimensions
  const double dim = (double)nsd_;

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  const double h = std::pow(vol, (1.0 / dim));
  //  const double h = calc_char_ele_length(vol,convelint.norm2(),convelint);
  //  //std::pow(vol,(1.0/dim));

  // artificial diffusivity
  double artdiff = 0.0;

  // classical linear artificial all-scale subgrid diffusivity
  if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_artificial)
  {
    // get element-type constant
    const double mk = ScaTra::mk<distype>();

    // velocity norm
    const double vel_norm = convelint.norm2();

    // parameter relating convective and diffusive forces + respective switch
    double epe = 0.0;
    double xi = 1.0;
    if (diffmanager_->get_isotropic_diff(k) > 1.0e-8)
    {
      epe = 0.5 * mk * densnp * vel_norm * h / diffmanager_->get_isotropic_diff(k);
      xi = std::min(epe, 1.0);
    }

    // compute subgrid diffusivity
    artdiff = xi * 0.5 * densnp * vel_norm * h;
  }
  else if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_lin_reinit)
  {
    artdiff = 0.005 * h;
  }
  else if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_codina)
  {
    double alpha =
        std::max(0.0, (0.7 - 2.0 * diffmanager_->get_isotropic_diff(k) / convelint.norm2() / h));

    // gradient norm
    const double grad_norm = gradphi.norm2();
    if (grad_norm > 1e-8) artdiff = 0.5 * alpha * h * std::abs(scatrares) / grad_norm;
  }
  else if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_yzbeta)
  {
    // phiref is the tuning parameter for this form of artificial diffusion
    const double phiref = 0.01;

    // gradient norm
    const double grad_norm = gradphi.norm2();

    if (phiref > 1e-12 and grad_norm > 1e-12)
    {
      // normalized gradient of phi
      Core::LinAlg::Matrix<nsd_, 1> normalized_gradphi(Core::LinAlg::Initialization::zero);
      normalized_gradphi.update(1.0, gradphi, 0.0);
      normalized_gradphi.scale(1.0 / grad_norm);

      // compute reference length
      double h_sum(0.0);
      for (unsigned inode = 0; inode < nen_; inode++)
      {
        double val(0.0);
        for (unsigned idim = 0; idim < nsd_; idim++)
          val += normalized_gradphi(idim, 0) * derxy_(idim, inode);

        h_sum += std::abs(val);
      }
      double h_dc(0.0);
      if (h_sum > 1e-12)
        h_dc = 2.0 / h_sum;
      else
        h_dc = h;

      // compute intermediate quantity
      double kappa_inter(0.0);
      for (unsigned idim = 0; idim < nsd_; idim++)
      {
        double val = gradphi(idim, 0) / phiref;
        kappa_inter += (val * val);
      }
      //      if (kappa_inter < 1e-10) FOUR_C_THROW("Too low value");

      // smoothness parameter beta: 1 (smoother layers) or 2 (sharper layers)
      // note for 1.0, this form is equivalent to the Codina form above, except
      // for a different definition of the reference length
      const double beta = 2.0;

      // finally compute artificial diffusion
      artdiff =
          std::abs(scatrares / phiref) * pow(kappa_inter, beta / 2.0 - 1.0) * pow(h_dc / 2.0, beta);
    }
    else
      artdiff = 0.0;
  }
  else
  {
    // gradient norm
    const double grad_norm = gradphi.norm2();

    if (grad_norm > 1e-10)
    {
      // for the present definitions, sigma and a specific term (either
      // residual or convective term) are different
      double sigma = 0.0;
      double specific_term = 0.0;
      switch (scatrapara_->assgd_type())
      {
        case Inpar::ScaTra::assgd_hughes:
        {
          if (eid_ == 0)
          {
            std::cout << "WARNING: Nonlinear isotropic artificial diffusion according to Hughes et "
                         "al. (1986)\n";
            std::cout
                << "         is implemented based on the exact tau for 1D stationary problems!"
                << std::endl;
          }
          // remark on this warning:
          // 1. Here, tau is calculated based on the exact formula for 1-D stationary problems. This
          // is inconsistent
          //    if the problem is not 1-D and/or other definitions than the ecaxt tau are chosen in
          //    the input file. Consistently, one has to use the same definition here.
          // 2. Instead of using sigma = tau_bhbar, Hughes et al. suggested to use sigma = tau_bhbar
          // - tau to not double
          //    the SUPG stabilization. This is another inconsistent aspect on this implementation.
          //    To have the right tau here (i.e, not the one the last gauss point or even last
          //    step), one has to calaculate tau first. Then, sigma and, hence, the addition
          //    diffusion is computed based on this tau. Next, tau is recomputed with the
          //    diffusivity replaced by the original (physical) diffusivity plus the estimated
          //    artificial diffusivity. This is probably not a good choice, because, first, tau is
          //    considered in the estimation of the artificial diffusion and then this artificial
          //    diffusion is incorporated into tau. This would reduce the effect. Perhaps, one
          //    should either consider tau in sigma or the artificial diffusion in tau. When
          //    changing this aspect, be aware that tau has to be computed after the subgrid-scale
          //    velocity has been calculated since, for this calculation tau is overwritten by its
          //    value in the fluid field. Note that similar considerations may also hold for the
          //    methods by do Carmo and Almeida.

          // get norm of velocity vector b_h^par
          const double vel_norm_bhpar = abs(conv_phi / grad_norm);

          // compute stabilization parameter based on b_h^par
          // (so far, only exact formula for stationary 1-D implemented)
          // element Peclet number relating convective and diffusive forces
          double epe = 0.5 * vel_norm_bhpar * h / diffmanager_->get_isotropic_diff(k);
          const double pp = exp(epe);
          const double pm = exp(-epe);
          double xi = 0.0;
          double tau_bhpar = 0.0;
          if (epe >= 700.0)
            tau_bhpar = 0.5 * h / vel_norm_bhpar;
          else if (epe < 700.0 and epe > 1e-15)
          {
            xi = (((pp + pm) / (pp - pm)) - (1.0 / epe));  // xi = coth(epe) - 1/epe
            // compute optimal stabilization parameter
            tau_bhpar = 0.5 * h * xi / vel_norm_bhpar;
          }

          // compute sigma
          sigma = std::max(0.0, tau_bhpar - tau);

          // set specific term to convective term
          specific_term = conv_phi;
        }
        break;
        case Inpar::ScaTra::assgd_tezduyar:
        case Inpar::ScaTra::assgd_tezduyar_wo_phizero:
        {
          // velocity norm
          const double vel_norm = convelint.norm2();

          // calculate stream length
          // according to John and Knobloch stream length in direction of b_h^par should be used
          // const double h_stream = calc_char_ele_length(vol,vel_norm);

          // get norm of velocity vector b_h^par
          const double vel_norm_bhpar = abs(conv_phi / grad_norm);

          // compute stabilization parameter based on b_h^par
          // (so far, only exact formula for stationary 1-D implemented)

          // compute sigma (version 1 according to John and Knobloch (2007))
          if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_tezduyar_wo_phizero)
          {
            if (vel_norm > 1e-10) sigma = (h / vel_norm) * (1.0 - (vel_norm_bhpar / vel_norm));
          }
          else
          {
            // compute sigma (version 2 according to John and Knobloch (2007))
            // setting scaling phi_0=1.0 as in John and Knobloch (2007)
            const double phi0 = 1.0;
            if (vel_norm > 1e-10)
              sigma = (h * h * grad_norm / (vel_norm * phi0)) * (1.0 - (vel_norm_bhpar / vel_norm));
          }

          // set specific term to convective term
          specific_term = conv_phi;
        }
        break;
        case Inpar::ScaTra::assgd_docarmo:
        case Inpar::ScaTra::assgd_almeida:
        {
          // velocity norm
          const double vel_norm = convelint.norm2();

          // get norm of velocity vector z_h
          const double vel_norm_zh = abs(scatrares / grad_norm);

          // parameter zeta differentiating approaches by doCarmo and Galeao (1991)
          // and Almeida and Silva (1997)
          double zeta = 0.0;
          if (scatrapara_->assgd_type() == Inpar::ScaTra::assgd_docarmo)
            zeta = 1.0;
          else
          {
            if (abs(scatrares) > 1e-10) zeta = std::max(1.0, (conv_phi / scatrares));
          }

          // compute sigma
          if (vel_norm_zh > 1e-10) sigma = tau * std::max(0.0, (vel_norm / vel_norm_zh) - zeta);

          // set specific term to residual
          specific_term = scatrares;
        }
        break;
        default:
          FOUR_C_THROW("unknown type of all-scale subgrid diffusivity\n");
          break;
      }  // switch (whichassgd)

      // computation of subgrid diffusivity
      artdiff = sigma * scatrares * specific_term / (grad_norm * grad_norm);
      if (artdiff < 0.0)
      {
        //        std::cout << "WARNING: isotropic artificial diffusion sgdiff < 0.0\n";
        //        std::cout << "         -> set sgdiff to abs(sgdiff)!" << std::endl;
        artdiff = abs(sigma * scatrares * specific_term / (grad_norm * grad_norm));
      }
    }
    else
      artdiff = 0.0;
  }

  //  if (artdiff>1e-8)
  //    std::cout<<__FILE__<<__LINE__<<"\t artdiff=\t"<<artdiff<<std::endl;
  diffmanager_->set_isotropic_sub_grid_diff(artdiff, k);

  return;
}  // ScaTraEleCalc::calc_subgr_diff


/*-------------------------------------------------------------------------------*
 |  calculation of strong residual for stabilization                             |
 | (depending on respective stationary or time-integration scheme)      vg 10/11 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_strong_residual(
    const int k,           //!< index of current scalar
    double& scatrares,     //!< residual of convection-diffusion-reaction eq
    const double densam,   //!< density at t_(n+am)
    const double densnp,   //!< density at t_(n+1)
    const double rea_phi,  //!< reactive contribution
    const double rhsint,   //!< rhs at gauss point
    const double tau       //!< the stabilisation parameter
)
{
  // scalar at t_(n+1)
  const double phinp = scatravarmanager_->phinp(k);
  // history of time integration
  const double hist = scatravarmanager_->hist(k);
  // convective contribution
  const double conv_phi = scatravarmanager_->conv_phi(k);

  // diffusive part used in stabilization terms
  double diff_phi(0.0);
  Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::zero);

  // diffusive term using current scalar value for higher-order elements
  // Note: has to be recomputed here every time, since the diffusion coefficient may have changed
  // since the last call
  if (use2ndderiv_)
  {
    // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
    get_laplacian_strong_form(diff);
    diff.scale(diffmanager_->get_isotropic_diff(k));
    diff_phi = diff.dot(ephinp_[k]);
  }

  if (scatraparatimint_->is_gen_alpha())
  {
    // time derivative stored on history variable
    scatrares = densam * hist + densnp * conv_phi - diff_phi + rea_phi - rhsint;
  }
  else
  {
    // stationary residual
    scatrares = densnp * conv_phi - diff_phi + rea_phi - rhsint;

    if (not scatraparatimint_->is_stationary())
    {
      scatrares *= scatraparatimint_->time_fac() / scatraparatimint_->dt();
      scatrares += densnp * (phinp - hist) / scatraparatimint_->dt();
    }
  }

  return;
}  // ScaTraEleCalc::calc_strong_residual


/*----------------------------------------------------------------------*
 |  calculate subgrid-scale velocity                           vg 10/09 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_subgr_velocity(
    const Core::Elements::Element* ele,              //!< the element we are dealing with
    Core::LinAlg::Matrix<nsd_, 1>& sgvelint,         //!< subgrid velocity at integration point
    const double densam,                             //!< density at t_(n+am)
    const double densnp,                             //!< density at t_(n+1)
    const double visc,                               //!< fluid viscosity
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity at integration point
    const double tau                                 //!< the stabilisation parameter
)
{
  // definitions
  Core::LinAlg::Matrix<nsd_, 1> acc;
  Core::LinAlg::Matrix<nsd_, nsd_> vderxy;
  Core::LinAlg::Matrix<nsd_, 1> conv;
  Core::LinAlg::Matrix<nsd_, 1> gradp;
  Core::LinAlg::Matrix<nsd_, 1> epsilonvel;
  Core::LinAlg::Matrix<nsd_, 1> bodyforce;
  Core::LinAlg::Matrix<nsd_, 1> pressuregrad;
  Core::LinAlg::Matrix<nsd_, nen_> nodebodyforce;
  Core::LinAlg::Matrix<nsd_, nen_> nodepressuregrad;

  // get acceleration or momentum history data
  acc.multiply(eaccnp_, funct_);

  // get velocity derivatives
  vderxy.multiply_nt(evelnp_, derxy_);

  // compute convective fluid term
  conv.multiply(vderxy, convelint);

  // get pressure gradient
  gradp.multiply(derxy_, eprenp_);

  //--------------------------------------------------------------------
  // get nodal values of fluid body force
  //--------------------------------------------------------------------
  std::vector<Core::Conditions::Condition*> myfluidneumcond;

  // check whether all nodes have a unique Fluid Neumann condition
  switch (nsd_)
  {
    case 3:
      Core::Conditions::find_element_conditions(ele, "FluidVolumeNeumann", myfluidneumcond);
      break;
    case 2:
      Core::Conditions::find_element_conditions(ele, "FluidSurfaceNeumann", myfluidneumcond);
      break;
    case 1:
      Core::Conditions::find_element_conditions(ele, "FluidLineNeumann", myfluidneumcond);
      break;
    default:
      FOUR_C_THROW("Illegal number of space dimensions: {}", nsd_);
      break;
  }

  if (myfluidneumcond.size() > 1) FOUR_C_THROW("more than one Fluid Neumann condition on one node");

  if (myfluidneumcond.size() == 1)
  {
    const std::string* condtype = &myfluidneumcond[0]->parameters().get<std::string>("TYPE");

    // get values and switches from the condition
    const auto onoff = myfluidneumcond[0]->parameters().get<std::vector<int>>("ONOFF");
    const auto val = myfluidneumcond[0]->parameters().get<std::vector<double>>("VAL");
    const auto funct =
        myfluidneumcond[0]->parameters().get<std::vector<std::optional<int>>>("FUNCT");

    // factor given by spatial function
    double functfac = 1.0;

    // set this condition to the body-force array
    for (unsigned isd = 0; isd < nsd_; isd++)
    {
      double num = onoff[isd] * val[isd];

      for (unsigned jnode = 0; jnode < nen_; ++jnode)
      {
        if (funct[isd].has_value() && funct[isd].value() > 0)
        {
          // time factor for the intermediate step
          // (negative time value indicates error)
          if (scatraparatimint_->time() >= 0.0)
          {
            // evaluate function at the position of the current node
            // ------------------------------------------------------
            // comment: this introduces an additional error compared to an
            // evaluation at the integration point. However, we need a node
            // based element bodyforce vector for prescribed pressure gradients
            // in some fancy turbulance stuff.
            functfac =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[isd].value())
                    .evaluate((ele->nodes()[jnode])->x().data(), scatraparatimint_->time(), isd);
          }
          else
            FOUR_C_THROW("Negative time value in body force calculation: time = {}",
                scatraparatimint_->time());
        }
        else
          functfac = 1.0;

        // compute body force
        if (*condtype == "Dead" or *condtype == "Live")
          nodebodyforce(isd, jnode) = num * functfac;
        else
          nodebodyforce.clear();

        // compute prescribed pressure gradient
        if (*condtype == "PressureGrad")
          nodepressuregrad(isd, jnode) = num * functfac;
        else
          nodepressuregrad.clear();
      }
    }
  }
  else
  {
    nodebodyforce.clear();
    nodepressuregrad.clear();
  }

  // get fluid body force
  bodyforce.multiply(nodebodyforce, funct_);
  // or prescribed pressure gradient
  pressuregrad.multiply(nodepressuregrad, funct_);

  // get viscous term
  if (use2ndderiv_) /*--- viscous term: div(epsilon(u)) --------------------------------*/
    /*   /                                                \
         |  2 N_x,xx + N_x,yy + N_y,xy + N_x,zz + N_z,xz  |
         1 |                                                |
         - |  N_y,xx + N_x,yx + 2 N_y,yy + N_z,yz + N_y,zz  |
         2 |                                                |
         |  N_z,xx + N_x,zx + N_y,zy + N_z,yy + 2 N_z,zz  |
         \                                                /

         with N_x .. x-line of N
         N_y .. y-line of N                                             */

    /*--- subtraction for low-Mach-number flow: div((1/3)*(div u)*I) */
    /*   /                            \
         |  N_x,xx + N_y,yx + N_z,zx  |
         1 |                            |
         -  - |  N_x,xy + N_y,yy + N_z,zy  |
         3 |                            |
         |  N_x,xz + N_y,yz + N_z,zz  |
         \                            /

         with N_x .. x-line of N
         N_y .. y-line of N                                             */
    calc_subgr_velocity_visc(epsilonvel);
  else
    epsilonvel.clear();

  //--------------------------------------------------------------------
  // calculation of subgrid-scale velocity based on momentum residual
  // and stabilization parameter
  // (different for generalized-alpha and other time-integration schemes)
  //--------------------------------------------------------------------
  if (scatraparatimint_->is_gen_alpha())
  {
    for (unsigned rr = 0; rr < nsd_; ++rr)
    {
      sgvelint(rr) =
          -tau * (densam * acc(rr) + densnp * conv(rr) + gradp(rr) - 2 * visc * epsilonvel(rr) -
                     densnp * bodyforce(rr) - pressuregrad(rr));
    }
  }
  else
  {
    for (unsigned rr = 0; rr < nsd_; ++rr)
    {
      sgvelint(rr) = -tau *
                     (densnp * convelint(rr) +
                         scatraparatimint_->time_fac() *
                             (densnp * conv(rr) + gradp(rr) - 2 * visc * epsilonvel(rr) -
                                 densnp * bodyforce(rr) - pressuregrad(rr)) -
                         densnp * acc(rr)) /
                     scatraparatimint_->dt();
    }
  }

  return;
}  // ScaTraEleCalc::calc_subgr_velocity


/*---------------------------------------------------------------*
 | calculate viscous part of subgrid-scale velocity   fang 02/15 |
 *---------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_subgr_velocity_visc(
    Core::LinAlg::Matrix<nsd_, 1>& epsilonvel)
{
  if (nsd_ == 3)
  {
    for (unsigned i = 0; i < nen_; ++i)
    {
      double sum = (derxy2_(0, i) + derxy2_(1, i) + derxy2_(2, i));

      epsilonvel(0) += (sum * evelnp_(0, i)) / 2.0;
      epsilonvel(1) += (sum * evelnp_(1, i)) / 2.0;
      epsilonvel(2) += (sum * evelnp_(2, i)) / 2.0;
    }
  }

  else if (nsd_ == 2)
  {
    for (unsigned i = 0; i < nen_; ++i)
    {
      double sum = (derxy2_(0, i) + derxy2_(1, i));

      epsilonvel(0) += (sum * evelnp_(0, i)) / 2.0;
      epsilonvel(1) += (sum * evelnp_(1, i)) / 2.0;
    }
  }

  else
    FOUR_C_THROW("Epsilon(u) is not implemented for the 1D case!");

  return;
}  // Discret::Elements::ScaTraEleCalc<distype,probdim>::calc_subgr_velocity_visc

FOUR_C_NAMESPACE_CLOSE

// template classes

#include "4C_scatra_ele_calc.inst.hpp"
