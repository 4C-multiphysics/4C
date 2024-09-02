/*----------------------------------------------------------------------*/
/*! \file
\brief Everything concerning EAS technology for so_hex8
\level 1

*----------------------------------------------------------------------*/

#include "4C_fem_discretization.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  initialize EAS data (private)                              maf 05/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::soh8_easinit()
{
  // EAS enhanced strain parameters at currently investigated load/time step
  Core::LinAlg::SerialDenseMatrix alpha(neas_, 1);
  // EAS enhanced strain parameters of last converged load/time step
  Core::LinAlg::SerialDenseMatrix alphao(neas_, 1);
  // EAS portion of internal forces, also called enhacement vector s or Rtilde
  Core::LinAlg::SerialDenseMatrix feas(neas_, 1);
  // EAS matrix K_{alpha alpha}, also called Dtilde
  Core::LinAlg::SerialDenseMatrix invKaa(neas_, neas_);
  // EAS matrix K_{alpha alpha} of last converged load/time step
  Core::LinAlg::SerialDenseMatrix invKaao(neas_, neas_);
  // EAS matrix K_{d alpha}
  Core::LinAlg::SerialDenseMatrix Kda(neas_, NUMDOF_SOH8);
  // EAS matrix K_{d alpha} of last converged load/time step
  Core::LinAlg::SerialDenseMatrix Kdao(neas_, NUMDOF_SOH8);
  // EAS increment over last Newton step
  Core::LinAlg::SerialDenseMatrix eas_inc(neas_, 1);

  // save EAS data into eas data
  easdata_.alpha = alpha;
  easdata_.alphao = alphao;
  easdata_.feas = feas;
  easdata_.invKaa = invKaa;
  easdata_.invKaao = invKaao;
  easdata_.Kda = Kda;
  easdata_.Kdao = Kdao;
  easdata_.eas_inc = eas_inc;
}

/*----------------------------------------------------------------------*
 |  re-initialize EAS data (private)                           maf 05/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::soh8_reiniteas(const Discret::ELEMENTS::SoHex8::EASType EASType)
{
  switch (EASType)
  {
    case Discret::ELEMENTS::SoHex8::soh8_easfull:
      neas_ = 21;
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easmild:
      neas_ = 9;
      break;
    case Discret::ELEMENTS::SoHex8::soh8_eassosh8:
      neas_ = 7;
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easnone:
      neas_ = 0;
      break;
  }
  eastype_ = EASType;
  if (eastype_ == Discret::ELEMENTS::SoHex8::soh8_easnone) return;
  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;    // EAS alphas
  Core::LinAlg::SerialDenseMatrix* alphao = nullptr;   // EAS alphas
  Core::LinAlg::SerialDenseMatrix* feas = nullptr;     // EAS history
  Core::LinAlg::SerialDenseMatrix* Kaainv = nullptr;   // EAS history
  Core::LinAlg::SerialDenseMatrix* Kaainvo = nullptr;  // EAS history
  Core::LinAlg::SerialDenseMatrix* Kda = nullptr;      // EAS history
  Core::LinAlg::SerialDenseMatrix* Kdao = nullptr;     // EAS history
  Core::LinAlg::SerialDenseMatrix* eas_inc = nullptr;  // EAS history
  alpha = &easdata_.alpha;                             // get alpha of previous iteration
  alphao = &easdata_.alphao;                           // get alpha of previous iteration
  feas = &easdata_.feas;
  Kaainv = &easdata_.invKaa;
  Kaainvo = &easdata_.invKaao;
  Kda = &easdata_.Kda;
  Kdao = &easdata_.Kdao;
  eas_inc = &easdata_.eas_inc;

  if (!alpha || !Kaainv || !Kda || !feas || !eas_inc) FOUR_C_THROW("Missing EAS history-data");

  alpha->reshape(neas_, 1);
  alphao->reshape(neas_, 1);
  feas->reshape(neas_, 1);
  Kaainv->reshape(neas_, neas_);
  Kaainvo->reshape(neas_, neas_);
  Kda->reshape(neas_, NUMDOF_SOH8);
  Kdao->reshape(neas_, NUMDOF_SOH8);
  eas_inc->reshape(neas_, 1);
}

/*----------------------------------------------------------------------*
 |  setup of constant EAS data (private)                       maf 05/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::soh8_eassetup(
    std::vector<Core::LinAlg::SerialDenseMatrix>** M_GP,  // M-matrix evaluated at GPs
    double& detJ0,                                        // det of Jacobian at origin
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>&
        T0invT,  // maps M(origin) local to global
    const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xrefe) const  // material element coords
{
  // vector of df(origin)
  static double df0_vector[NUMDIM_SOH8 * NUMNOD_SOH8] = {-0.125, -0.125, -0.125, +0.125, -0.125,
      -0.125, +0.125, +0.125, -0.125, -0.125, +0.125, -0.125, -0.125, -0.125, +0.125, +0.125,
      -0.125, +0.125, +0.125, +0.125, +0.125, -0.125, +0.125, +0.125};
  // shape function derivatives, evaluated at origin (r=s=t=0.0)
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> df0(df0_vector);  // copy

  // compute Jacobian, evaluated at element origin (r=s=t=0.0)
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> jac0;
  jac0.multiply(df0, xrefe);
  // compute determinant of Jacobian at origin
  detJ0 = jac0.determinant();

  // first, build T0^T transformation matrix which maps the M-matrix
  // between global (r,s,t)-coordinates and local (x,y,z)-coords
  // later, invert the transposed to map from local to global
  // see literature for details (e.g. Andelfinger)
  // it is based on the voigt notation for strains: xx,yy,zz,xy,yz,xz
  T0invT(0, 0) = jac0(0, 0) * jac0(0, 0);
  T0invT(1, 0) = jac0(1, 0) * jac0(1, 0);
  T0invT(2, 0) = jac0(2, 0) * jac0(2, 0);
  T0invT(3, 0) = 2 * jac0(0, 0) * jac0(1, 0);
  T0invT(4, 0) = 2 * jac0(1, 0) * jac0(2, 0);
  T0invT(5, 0) = 2 * jac0(0, 0) * jac0(2, 0);

  T0invT(0, 1) = jac0(0, 1) * jac0(0, 1);
  T0invT(1, 1) = jac0(1, 1) * jac0(1, 1);
  T0invT(2, 1) = jac0(2, 1) * jac0(2, 1);
  T0invT(3, 1) = 2 * jac0(0, 1) * jac0(1, 1);
  T0invT(4, 1) = 2 * jac0(1, 1) * jac0(2, 1);
  T0invT(5, 1) = 2 * jac0(0, 1) * jac0(2, 1);

  T0invT(0, 2) = jac0(0, 2) * jac0(0, 2);
  T0invT(1, 2) = jac0(1, 2) * jac0(1, 2);
  T0invT(2, 2) = jac0(2, 2) * jac0(2, 2);
  T0invT(3, 2) = 2 * jac0(0, 2) * jac0(1, 2);
  T0invT(4, 2) = 2 * jac0(1, 2) * jac0(2, 2);
  T0invT(5, 2) = 2 * jac0(0, 2) * jac0(2, 2);

  T0invT(0, 3) = jac0(0, 0) * jac0(0, 1);
  T0invT(1, 3) = jac0(1, 0) * jac0(1, 1);
  T0invT(2, 3) = jac0(2, 0) * jac0(2, 1);
  T0invT(3, 3) = jac0(0, 0) * jac0(1, 1) + jac0(1, 0) * jac0(0, 1);
  T0invT(4, 3) = jac0(1, 0) * jac0(2, 1) + jac0(2, 0) * jac0(1, 1);
  T0invT(5, 3) = jac0(0, 0) * jac0(2, 1) + jac0(2, 0) * jac0(0, 1);


  T0invT(0, 4) = jac0(0, 1) * jac0(0, 2);
  T0invT(1, 4) = jac0(1, 1) * jac0(1, 2);
  T0invT(2, 4) = jac0(2, 1) * jac0(2, 2);
  T0invT(3, 4) = jac0(0, 1) * jac0(1, 2) + jac0(1, 1) * jac0(0, 2);
  T0invT(4, 4) = jac0(1, 1) * jac0(2, 2) + jac0(2, 1) * jac0(1, 2);
  T0invT(5, 4) = jac0(0, 1) * jac0(2, 2) + jac0(2, 1) * jac0(0, 2);

  T0invT(0, 5) = jac0(0, 0) * jac0(0, 2);
  T0invT(1, 5) = jac0(1, 0) * jac0(1, 2);
  T0invT(2, 5) = jac0(2, 0) * jac0(2, 2);
  T0invT(3, 5) = jac0(0, 0) * jac0(1, 2) + jac0(1, 0) * jac0(0, 2);
  T0invT(4, 5) = jac0(1, 0) * jac0(2, 2) + jac0(2, 0) * jac0(1, 2);
  T0invT(5, 5) = jac0(0, 0) * jac0(2, 2) + jac0(2, 0) * jac0(0, 2);

  // now evaluate T0^{-T} with solver
  Core::LinAlg::FixedSizeSerialDenseSolver<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D, 1>
      solve_for_inverseT0;
  solve_for_inverseT0.set_matrix(T0invT);
  int err2 = solve_for_inverseT0.factor();
  int err = solve_for_inverseT0.invert();
  if ((err != 0) || (err2 != 0)) FOUR_C_THROW("Inversion of T0inv (Jacobian0) failed");

  // build EAS interpolation matrix M, evaluated at the 8 GPs of so_hex8

  // fill up M at each gp
  if (eastype_ == soh8_easmild)
  {
    // static Core::LinAlg::SerialDenseMatrix M_mild(Mat::NUM_STRESS_3D*NUMGPT_SOH8,neas_);
    static std::vector<Core::LinAlg::SerialDenseMatrix> M_mild(NUMGPT_SOH8);
    static bool M_mild_eval = false;
    /* easmild is the EAS interpolation of 9 modes, based on
    **            r 0 0   0 0 0 0 0 0
    **            0 s 0   0 0 0 0 0 0
    **    M =     0 0 t   0 0 0 0 0 0
    **            0 0 0   r s 0 0 0 0
    **            0 0 0   0 0 s t 0 0
    **            0 0 0   0 0 0 0 r t
    */
    if (!M_mild_eval)
    {  // if true M already evaluated
      // (r,s,t) gp-locations of fully integrated linear 8-node Hex
      const double* r = soh8_get_coordinate_of_gausspoints(0);
      const double* s = soh8_get_coordinate_of_gausspoints(1);
      const double* t = soh8_get_coordinate_of_gausspoints(2);

      // fill up M at each gp
      for (unsigned i = 0; i < NUMGPT_SOH8; ++i)
      {
        M_mild[i].shape(Mat::NUM_STRESS_3D, neas_);
        M_mild[i](0, 0) = r[i];
        M_mild[i](1, 1) = s[i];
        M_mild[i](2, 2) = t[i];

        M_mild[i](3, 3) = r[i];
        M_mild[i](3, 4) = s[i];
        M_mild[i](4, 5) = s[i];
        M_mild[i](4, 6) = t[i];
        M_mild[i](5, 7) = r[i];
        M_mild[i](5, 8) = t[i];
      }
      M_mild_eval = true;  // now the array is filled statically
    }

    // return adress of just evaluated matrix
    *M_GP = &M_mild;  // return adress of static object to target of pointer
  }
  else if (eastype_ == soh8_easfull)
  {
    static std::vector<Core::LinAlg::SerialDenseMatrix> M_full(NUMGPT_SOH8);
    static bool M_full_eval = false;
    /* easfull is the EAS interpolation of 21 modes, based on
    **            r 0 0   0 0 0 0 0 0   0  0  0  0  0  0   rs rt 0  0  0  0
    **            0 s 0   0 0 0 0 0 0   0  0  0  0  0  0   0  0  rs st 0  0
    **    M =     0 0 t   0 0 0 0 0 0   0  0  0  0  0  0   0  0  0  0  rt st
    **            0 0 0   r s 0 0 0 0   rt st 0  0  0  0   0  0  0  0  0  0
    **            0 0 0   0 0 s t 0 0   0  0  rs rt 0  0   0  0  0  0  0  0
    **            0 0 0   0 0 0 0 r t   0  0  0  0  rs st  0  0  0  0  0  0
    */
    if (!M_full_eval)
    {  // if true M already evaluated
      // (r,s,t) gp-locations of fully integrated linear 8-node Hex
      const double* r = soh8_get_coordinate_of_gausspoints(0);
      const double* s = soh8_get_coordinate_of_gausspoints(1);
      const double* t = soh8_get_coordinate_of_gausspoints(2);

      // fill up M at each gp
      for (unsigned i = 0; i < NUMGPT_SOH8; ++i)
      {
        M_full[i].shape(Mat::NUM_STRESS_3D, neas_);
        M_full[i](0, 0) = r[i];
        M_full[i](0, 15) = r[i] * s[i];
        M_full[i](0, 16) = r[i] * t[i];
        M_full[i](1, 1) = s[i];
        M_full[i](1, 17) = r[i] * s[i];
        M_full[i](1, 18) = s[i] * t[i];
        M_full[i](2, 2) = t[i];
        M_full[i](2, 19) = r[i] * t[i];
        M_full[i](2, 20) = s[i] * t[i];

        M_full[i](3, 3) = r[i];
        M_full[i](3, 4) = s[i];
        M_full[i](3, 9) = r[i] * t[i];
        M_full[i](3, 10) = s[i] * t[i];
        M_full[i](4, 5) = s[i];
        M_full[i](4, 6) = t[i];
        M_full[i](4, 11) = r[i] * s[i];
        M_full[i](4, 12) = r[i] * t[i];
        M_full[i](5, 7) = r[i];
        M_full[i](5, 8) = t[i];
        M_full[i](5, 13) = r[i] * s[i];
        M_full[i](5, 14) = s[i] * t[i];
      }
      M_full_eval = true;  // now the array is filled statically
    }
    // return adress of just evaluated matrix
    *M_GP = &M_full;  // return adress of static object to target of pointer
  }
  else if (eastype_ == soh8_eassosh8)
  {
    static std::vector<Core::LinAlg::SerialDenseMatrix> M_sosh8(NUMGPT_SOH8);
    static bool M_sosh8_eval = false;
    /* eassosh8 is the EAS interpolation for the Solid-Shell with t=thickness dir.
    ** consisting of 7 modes, based on
    **            r 0 0   0 0 0  0
    **            0 s 0   0 0 0  0
    **    M =     0 0 t   0 0 rt st
    **            0 0 0   r s 0  0
    **            0 0 0   0 0 0  0
    **            0 0 0   0 0 0  0
    */
    if (!M_sosh8_eval)
    {  // if true M already evaluated
      // (r,s,t) gp-locations of fully integrated linear 8-node Hex
      const double* r = soh8_get_coordinate_of_gausspoints(0);
      const double* s = soh8_get_coordinate_of_gausspoints(1);
      const double* t = soh8_get_coordinate_of_gausspoints(2);

      // fill up M at each gp
      for (unsigned i = 0; i < NUMGPT_SOH8; ++i)
      {
        M_sosh8[i].shape(Mat::NUM_STRESS_3D, neas_);
        M_sosh8[i](0, 0) = r[i];
        M_sosh8[i](1, 1) = s[i];
        M_sosh8[i](2, 2) = t[i];
        M_sosh8[i](2, 5) = r[i] * t[i];
        M_sosh8[i](2, 6) = s[i] * t[i];

        M_sosh8[i](3, 3) = r[i];
        M_sosh8[i](3, 4) = s[i];
      }
      M_sosh8_eval = true;  // now the array is filled statically
    }
    // return adress of just evaluated matrix
    *M_GP = &M_sosh8;  // return adress of static object to target of pointer
  }
  else
  {
    FOUR_C_THROW("eastype not implemented");
  }
}  // end of soh8_eassetup

/*----------------------------------------------------------------------*
 |  Update EAS parameters (private)                                     |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::soh8_easupdate()
{
  const auto* alpha = &easdata_.alpha;    // Alpha_{n+1}
  auto* alphao = &easdata_.alphao;        // Alpha_n
  const auto* Kaainv = &easdata_.invKaa;  // Kaa^{-1}_{n+1}
  auto* Kaainvo = &easdata_.invKaao;      // Kaa^{-1}_{n}
  const auto* Kda = &easdata_.Kda;        // Kda_{n+1}
  auto* Kdao = &easdata_.Kdao;            // Kda_{n}

  switch (eastype_)
  {
    case Discret::ELEMENTS::SoHex8::soh8_easfull:
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, 1>(
          alphao->values(), alpha->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, soh8_easfull>(
          Kaainvo->values(), Kaainv->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, NUMDOF_SOH8>(
          Kdao->values(), Kda->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easmild:
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, 1>(
          alphao->values(), alpha->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, soh8_easmild>(
          Kaainvo->values(), Kaainv->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, NUMDOF_SOH8>(
          Kdao->values(), Kda->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_eassosh8:
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, 1>(
          alphao->values(), alpha->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, soh8_eassosh8>(
          Kaainvo->values(), Kaainv->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, NUMDOF_SOH8>(
          Kdao->values(), Kda->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easnone:
      break;
    default:
      FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
      break;
  }
}

/*----------------------------------------------------------------------*
 |  Restore EAS parameters (private)                                     |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::soh8_easrestore()
{
  auto* alpha = &easdata_.alpha;            // Alpha_{n+1}
  const auto* alphao = &easdata_.alphao;    // Alpha_n
  auto* Kaainv = &easdata_.invKaa;          // Kaa^{-1}_{n+1}
  const auto* Kaainvo = &easdata_.invKaao;  // Kaa^{-1}_{n}
  auto* Kda = &easdata_.Kda;                // Kda_{n+1}
  const auto* Kdao = &easdata_.Kdao;        // Kda_{n}

  switch (eastype_)
  {
    case Discret::ELEMENTS::SoHex8::soh8_easfull:
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, 1>(
          alpha->values(), alphao->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, soh8_easfull>(
          Kaainv->values(), Kaainvo->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easfull, NUMDOF_SOH8>(
          Kda->values(), Kdao->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easmild:
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, 1>(
          alpha->values(), alphao->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, soh8_easmild>(
          Kaainv->values(), Kaainvo->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_easmild, NUMDOF_SOH8>(
          Kda->values(), Kdao->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_eassosh8:
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, 1>(
          alpha->values(), alphao->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, soh8_eassosh8>(
          Kaainv->values(), Kaainvo->values());
      Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, NUMDOF_SOH8>(
          Kda->values(), Kdao->values());
      break;
    case Discret::ELEMENTS::SoHex8::soh8_easnone:
      break;
    default:
      FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
      break;
  }
}

FOUR_C_NAMESPACE_CLOSE
