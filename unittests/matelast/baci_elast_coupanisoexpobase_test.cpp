/*----------------------------------------------------------------------*/
/*! \file

\brief Testcases for the CoupAnisoExpoBase summand

\level 2


*/
/*----------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_linalg_fixedsizematrix_generators.hpp"
#include "baci_linalg_fixedsizematrix_voigt_notation.hpp"
#include "baci_mat_anisotropy.hpp"
#include "baci_mat_service.hpp"
#include "baci_matelast_coupanisoexpo.hpp"

#include <Teuchos_RCPDecl.hpp>

namespace
{
  using namespace BACI;

  class CoupAnisoExpoBaseInterfaceFake : public MAT::ELASTIC::CoupAnisoExpoBaseInterface
  {
   public:
    CoupAnisoExpoBaseInterfaceFake()
    {
      /// initialize dummy fibers
      std::array<CORE::LINALG::Matrix<3, 1>, 2> fibersa;
      std::array<CORE::LINALG::Matrix<3, 1>, 2> fibersb;
      // gp 0
      fibersa[0](0) = 0.469809238649817;
      fibersa[0](1) = 0.872502871778232;
      fibersa[0](2) = 0.134231211042805;

      fibersb[0](0) = 0.071428571428571;
      fibersb[0](1) = 0.142857142857143;
      fibersb[0](2) = 0.214285714285714;

      // gp 1
      fibersa[1](0) = 0.245358246032859;
      fibersa[1](1) = 0.858753861115007;
      fibersa[1](2) = 0.449823451060242;

      fibersb[1](0) = 0.068965517241379;
      fibersb[1](1) = 0.103448275862069;
      fibersb[1](2) = 0.137931034482759;

      // compute norm of fibers
      for (std::size_t i = 0; i < fibersa.size(); ++i)
      {
        scalarProducts_[i] = fibersa[i].Dot(fibersb[i]);
        CORE::LINALG::Matrix<3, 3> tmp;
        tmp.MultiplyNT(fibersa[i], fibersb[i]);
        tensors_[i].Update(0.5, tmp);
        tensors_[i].UpdateT(0.5, tmp, 1.0);
        CORE::LINALG::VOIGT::Stresses::MatrixToVector(tensors_[i], tensors_stress_[i]);
      }
    }

    [[nodiscard]] double GetScalarProduct(int gp) const override { return scalarProducts_[gp]; }

    [[nodiscard]] const CORE::LINALG::Matrix<3, 3>& GetStructuralTensor(int gp) const override
    {
      return tensors_.at(gp);
    }

    [[nodiscard]] const CORE::LINALG::Matrix<6, 1>& GetStructuralTensor_stress(
        int gp) const override
    {
      return tensors_stress_.at(gp);
    }

   private:
    std::array<double, 2> scalarProducts_;
    std::array<CORE::LINALG::Matrix<3, 3>, 2> tensors_;
    std::array<CORE::LINALG::Matrix<6, 1>, 2> tensors_stress_;
  };

  class CoupAnisoExpoFake : public MAT::ELASTIC::CoupAnisoExpoBase
  {
   public:
    CoupAnisoExpoFake(MAT::ELASTIC::PAR::CoupAnisoExpoBase* params)
        : CoupAnisoExpoBase(params), anisotropyExtension_()
    {
    }

    [[nodiscard]] INPAR::MAT::MaterialType MaterialType() const override
    {
      dserror("This is only a Mock. Don't use this here!");
      std::abort();
    }

   protected:
    [[nodiscard]] const MAT::ELASTIC::CoupAnisoExpoBaseInterface& GetCoupAnisoExpoBaseInterface()
        const override
    {
      return anisotropyExtension_;
    }

   private:
    const CoupAnisoExpoBaseInterfaceFake anisotropyExtension_;
  };

  class CoupAnisoExpoBaseTest : public ::testing::Test
  {
   protected:
    CoupAnisoExpoBaseTest()
        : parameters_(std::invoke(
              []()
              {
                MAT::ELASTIC::PAR::CoupAnisoExpoBase parameters;
                parameters.k1_ = 1.2;
                parameters.k2_ = 1.3;
                parameters.k1comp_ = 1.4;
                parameters.k2comp_ = 1.5;
                return parameters;
              })),
          summand_(&parameters_)
    {
      const CORE::LINALG::Matrix<3, 3> Id = CORE::LINALG::IdentityMatrix<3>();

      C1_(0, 0) = 1.484;
      C1_(1, 1) = 2.3;
      C1_(2, 2) = 1.84;
      C1_(0, 1) = C1_(1, 0) = 0.22;
      C1_(1, 2) = C1_(2, 1) = 0.3214;
      C1_(0, 2) = C1_(2, 0) = 0.523;

      C2_(0, 0) = 0.516;
      C2_(1, 1) = 0.3;
      C2_(2, 2) = 0.16;
      C2_(0, 1) = C2_(1, 0) = 0.022;
      C2_(1, 2) = C2_(2, 1) = 0.03214;
      C2_(0, 2) = C2_(2, 0) = 0.0523;

      E1_.Update(0.5, C1_, -0.5, Id);
      E2_.Update(0.5, C2_, -0.5, Id);

      CORE::LINALG::VOIGT::Strains::MatrixToVector(C1_, C1_strain_);
      CORE::LINALG::VOIGT::Strains::MatrixToVector(C2_, C2_strain_);
      CORE::LINALG::VOIGT::Strains::MatrixToVector(E1_, E1_strain_);
      CORE::LINALG::VOIGT::Strains::MatrixToVector(E2_, E2_strain_);
    }


    MAT::ELASTIC::PAR::CoupAnisoExpoBase parameters_;
    CoupAnisoExpoFake summand_;

    CORE::LINALG::Matrix<3, 3> C1_;
    CORE::LINALG::Matrix<3, 3> C2_;
    CORE::LINALG::Matrix<3, 3> E1_;
    CORE::LINALG::Matrix<3, 3> E2_;

    CORE::LINALG::Matrix<6, 1> C1_strain_;
    CORE::LINALG::Matrix<6, 1> C2_strain_;
    CORE::LINALG::Matrix<6, 1> E1_strain_;
    CORE::LINALG::Matrix<6, 1> E2_strain_;
  };


  TEST_F(CoupAnisoExpoBaseTest, TestAddStrainEnergy)
  {
    CORE::LINALG::Matrix<3, 1> prinv(true);
    CORE::LINALG::Matrix<3, 1> modinv(true);

    double psi = 3.3;
    summand_.AddStrainEnergy(psi, prinv, modinv, E1_strain_, 0, 0);
    EXPECT_NEAR(psi, 3.3970087043259376, 1e-10);

    psi = 3.3;
    summand_.AddStrainEnergy(psi, prinv, modinv, E2_strain_, 0, 0);
    EXPECT_NEAR(psi, 3.308930242120174, 1e-10);

    psi = 3.3;
    summand_.AddStrainEnergy(psi, prinv, modinv, E1_strain_, 1, 0);
    EXPECT_NEAR(psi, 3.358837786011725, 1e-10);

    psi = 3.3;
    summand_.AddStrainEnergy(psi, prinv, modinv, E2_strain_, 1, 0);
    EXPECT_NEAR(psi, 3.3088538178019045, 1e-10);
  }

  TEST_F(CoupAnisoExpoBaseTest, TestEvaluateFunc)
  {
    CORE::LINALG::Matrix<3, 1> prinv(true);
    CORE::LINALG::Matrix<3, 1> modinv(true);

    double psi = 3.3;
    summand_.EvaluateFunc(psi, C1_, 0, 0);
    EXPECT_NEAR(psi, 3.3970087043259376, 1e-10);

    psi = 3.3;
    summand_.EvaluateFunc(psi, C2_, 0, 0);
    EXPECT_NEAR(psi, 3.308930242120174, 1e-10);

    psi = 3.3;
    summand_.EvaluateFunc(psi, C1_, 1, 0);
    EXPECT_NEAR(psi, 3.358837786011725, 1e-10);

    psi = 3.3;
    summand_.EvaluateFunc(psi, C2_, 1, 0);
    EXPECT_NEAR(psi, 3.3088538178019045, 1e-10);
  }

  TEST_F(CoupAnisoExpoBaseTest, EvaluateFistDerivativeAniso)
  {
    CORE::LINALG::Matrix<2, 1> dPIaniso(true);

    summand_.EvaluateFirstDerivativesAniso(dPIaniso, C1_, 0, 0);
    EXPECT_NEAR(dPIaniso(0), 0.6000375695574949, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    dPIaniso.Clear();
    summand_.EvaluateFirstDerivativesAniso(dPIaniso, C2_, 0, 0);
    EXPECT_NEAR(dPIaniso(0), -0.1603915791546372, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    dPIaniso.Clear();
    summand_.EvaluateFirstDerivativesAniso(dPIaniso, C1_, 1, 0);
    EXPECT_NEAR(dPIaniso(0), 0.4435645526048857, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    dPIaniso.Clear();
    summand_.EvaluateFirstDerivativesAniso(dPIaniso, C2_, 1, 0);
    EXPECT_NEAR(dPIaniso(0), -0.15968456768492822, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);
  }

  TEST_F(CoupAnisoExpoBaseTest, EvaluateSecondDerivativeAnsio)
  {
    CORE::LINALG::Matrix<3, 1> ddPIaniso(true);

    summand_.EvaluateSecondDerivativesAniso(ddPIaniso, C1_, 0, 0);
    EXPECT_NEAR(ddPIaniso(0), 2.329771574299716, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    ddPIaniso.Clear();
    summand_.EvaluateSecondDerivativesAniso(ddPIaniso, C2_, 0, 0);
    EXPECT_NEAR(ddPIaniso(0), 1.4808816133878135, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    ddPIaniso.Clear();
    summand_.EvaluateSecondDerivativesAniso(ddPIaniso, C1_, 1, 0);
    EXPECT_NEAR(ddPIaniso(0), 1.9509145858930543, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    ddPIaniso.Clear();
    summand_.EvaluateSecondDerivativesAniso(ddPIaniso, C2_, 1, 0);
    EXPECT_NEAR(ddPIaniso(0), 1.480185139428871, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);
  }


  TEST_F(CoupAnisoExpoBaseTest, GetDerivativesAniso)
  {
    CORE::LINALG::Matrix<2, 1> dPIaniso(true);
    CORE::LINALG::Matrix<3, 1> ddPIaniso(true);
    CORE::LINALG::Matrix<4, 1> dddPIaniso(true);

    summand_.GetDerivativesAniso(dPIaniso, ddPIaniso, dddPIaniso, C1_, 0, 0);
    EXPECT_NEAR(dPIaniso(0), 0.6000375695574949, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    EXPECT_NEAR(ddPIaniso(0), 2.329771574299716, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    EXPECT_NEAR(dddPIaniso(0), 6.080288490892458, 1e-10);
    EXPECT_NEAR(dddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(2), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(3), 0.0, 1e-10);

    dPIaniso.Clear();
    ddPIaniso.Clear();
    dddPIaniso.Clear();
    summand_.GetDerivativesAniso(dPIaniso, ddPIaniso, dddPIaniso, C2_, 0, 0);
    EXPECT_NEAR(dPIaniso(0), -0.1603915791546372, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    EXPECT_NEAR(ddPIaniso(0), 1.4808816133878135, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    EXPECT_NEAR(dddPIaniso(0), -1.4617659684416482, 1e-10);
    EXPECT_NEAR(dddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(2), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(3), 0.0, 1e-10);

    dPIaniso.Clear();
    ddPIaniso.Clear();
    dddPIaniso.Clear();
    summand_.GetDerivativesAniso(dPIaniso, ddPIaniso, dddPIaniso, C1_, 1, 0);
    EXPECT_NEAR(dPIaniso(0), 0.4435645526048857, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    EXPECT_NEAR(ddPIaniso(0), 1.9509145858930543, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    EXPECT_NEAR(dddPIaniso(0), 4.308103249341099, 1e-10);
    EXPECT_NEAR(dddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(2), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(3), 0.0, 1e-10);

    dPIaniso.Clear();
    ddPIaniso.Clear();
    dddPIaniso.Clear();
    summand_.GetDerivativesAniso(dPIaniso, ddPIaniso, dddPIaniso, C2_, 1, 0);
    EXPECT_NEAR(dPIaniso(0), -0.15968456768492822, 1e-10);
    EXPECT_NEAR(dPIaniso(1), 0.0, 1e-10);

    EXPECT_NEAR(ddPIaniso(0), 1.480185139428871, 1e-10);
    EXPECT_NEAR(ddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(ddPIaniso(2), 0.0, 1e-10);

    EXPECT_NEAR(dddPIaniso(0), -1.4551684829788594, 1e-10);
    EXPECT_NEAR(dddPIaniso(1), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(2), 0.0, 1e-10);
    EXPECT_NEAR(dddPIaniso(3), 0.0, 1e-10);
  }

  TEST_F(CoupAnisoExpoBaseTest, AddStressAnisoPrincipal)
  {
    CORE::LINALG::Matrix<6, 1> S_stress;
    CORE::LINALG::Matrix<6, 6> cmat;
    Teuchos::ParameterList dummyParams{};

    summand_.AddStressAnisoPrincipal(C1_strain_, cmat, S_stress, dummyParams, 0, 0);
    EXPECT_NEAR(S_stress(0), 0.04027188481644165, 1e-10);
    EXPECT_NEAR(S_stress(1), 0.14958128646107013, 1e-10);
    EXPECT_NEAR(S_stress(2), 0.03451875841409305, 1e-10);
    EXPECT_NEAR(S_stress(3), 0.0776672064317092, 1e-10);
    EXPECT_NEAR(S_stress(4), 0.12369221765050004, 1e-10);
    EXPECT_NEAR(S_stress(5), 0.06616095362701158, 1e-10);
    EXPECT_NEAR(cmat(0, 0), 0.01049446655089949, 1e-10);
    EXPECT_NEAR(cmat(0, 1), 0.03897944718905555, 1e-10);
    EXPECT_NEAR(cmat(0, 2), 0.008995257043628187, 1e-10);
    EXPECT_NEAR(cmat(0, 3), 0.020239328348163384, 1e-10);
    EXPECT_NEAR(cmat(0, 4), 0.03223300440633433, 1e-10);
    EXPECT_NEAR(cmat(0, 5), 0.017240909333620668, 1e-10);
    EXPECT_NEAR(cmat(1, 0), 0.03897944718905555, 1e-10);
    EXPECT_NEAR(cmat(1, 1), 0.1447808038450646, 1e-10);
    EXPECT_NEAR(cmat(1, 2), 0.03341095473347638, 1e-10);
    EXPECT_NEAR(cmat(1, 3), 0.07517464815032171, 1e-10);
    EXPECT_NEAR(cmat(1, 4), 0.11972258779495701, 1e-10);
    EXPECT_NEAR(cmat(1, 5), 0.06403766323916298, 1e-10);
    EXPECT_NEAR(cmat(2, 0), 0.008995257043628187, 1e-10);
    EXPECT_NEAR(cmat(2, 1), 0.03341095473347638, 1e-10);
    EXPECT_NEAR(cmat(2, 2), 0.007710220323109921, 1e-10);
    EXPECT_NEAR(cmat(2, 3), 0.01734799572699729, 1e-10);
    EXPECT_NEAR(cmat(2, 4), 0.027628289491143872, 1e-10);
    EXPECT_NEAR(cmat(2, 5), 0.01477792228596066, 1e-10);
    EXPECT_NEAR(cmat(3, 0), 0.020239328348163384, 1e-10);
    EXPECT_NEAR(cmat(3, 1), 0.07517464815032171, 1e-10);
    EXPECT_NEAR(cmat(3, 2), 0.01734799572699729, 1e-10);
    EXPECT_NEAR(cmat(3, 3), 0.03903299038574383, 1e-10);
    EXPECT_NEAR(cmat(3, 4), 0.0621636513550736, 1e-10);
    EXPECT_NEAR(cmat(3, 5), 0.033250325143411426, 1e-10);
    EXPECT_NEAR(cmat(4, 0), 0.03223300440633433, 1e-10);
    EXPECT_NEAR(cmat(4, 1), 0.11972258779495701, 1e-10);
    EXPECT_NEAR(cmat(4, 2), 0.027628289491143872, 1e-10);
    EXPECT_NEAR(cmat(4, 3), 0.0621636513550736, 1e-10);
    EXPECT_NEAR(cmat(4, 4), 0.09900137067659885, 1e-10);
    EXPECT_NEAR(cmat(4, 5), 0.052954221524692355, 1e-10);
    EXPECT_NEAR(cmat(5, 0), 0.017240909333620668, 1e-10);
    EXPECT_NEAR(cmat(5, 1), 0.06403766323916298, 1e-10);
    EXPECT_NEAR(cmat(5, 2), 0.01477792228596066, 1e-10);
    EXPECT_NEAR(cmat(5, 3), 0.033250325143411426, 1e-10);
    EXPECT_NEAR(cmat(5, 4), 0.052954221524692355, 1e-10);
    EXPECT_NEAR(cmat(5, 5), 0.028324351048091223, 1e-10);

    S_stress.Clear();
    cmat.Clear();
    summand_.AddStressAnisoPrincipal(C2_strain_, cmat, S_stress, dummyParams, 0, 0);
    EXPECT_NEAR(S_stress(0), -0.010764777955497356, 1e-10);
    EXPECT_NEAR(S_stress(1), -0.03998346097756192, 1e-10);
    EXPECT_NEAR(S_stress(2), -0.009226952533283502, 1e-10);
    EXPECT_NEAR(S_stress(3), -0.02076064319988784, 1e-10);
    EXPECT_NEAR(S_stress(4), -0.03306324657759921, 1e-10);
    EXPECT_NEAR(S_stress(5), -0.017684992355460023, 1e-10);
    EXPECT_NEAR(cmat(0, 0), 0.0066706378981432205, 1e-10);
    EXPECT_NEAR(cmat(0, 1), 0.02477665505024644, 1e-10);
    EXPECT_NEAR(cmat(0, 2), 0.005717689626979938, 1e-10);
    EXPECT_NEAR(cmat(0, 3), 0.012864801660704837, 1e-10);
    EXPECT_NEAR(cmat(0, 4), 0.02048838783001144, 1e-10);
    EXPECT_NEAR(cmat(0, 5), 0.0109589051183782, 1e-10);
    EXPECT_NEAR(cmat(1, 0), 0.02477665505024644, 1e-10);
    EXPECT_NEAR(cmat(1, 1), 0.09202757590091606, 1e-10);
    EXPECT_NEAR(cmat(1, 2), 0.021237132900211363, 1e-10);
    EXPECT_NEAR(cmat(1, 3), 0.047783549025475464, 1e-10);
    EXPECT_NEAR(cmat(1, 4), 0.07609972622575735, 1e-10);
    EXPECT_NEAR(cmat(1, 5), 0.04070450472540505, 1e-10);
    EXPECT_NEAR(cmat(2, 0), 0.005717689626979938, 1e-10);
    EXPECT_NEAR(cmat(2, 1), 0.021237132900211363, 1e-10);
    EXPECT_NEAR(cmat(2, 2), 0.00490087682312569, 1e-10);
    EXPECT_NEAR(cmat(2, 3), 0.011026972852032782, 1e-10);
    EXPECT_NEAR(cmat(2, 4), 0.01756147528286705, 1e-10);
    EXPECT_NEAR(cmat(2, 5), 0.009393347244324226, 1e-10);
    EXPECT_NEAR(cmat(3, 0), 0.012864801660704837, 1e-10);
    EXPECT_NEAR(cmat(3, 1), 0.047783549025475464, 1e-10);
    EXPECT_NEAR(cmat(3, 2), 0.011026972852032782, 1e-10);
    EXPECT_NEAR(cmat(3, 3), 0.024810688917073713, 1e-10);
    EXPECT_NEAR(cmat(3, 4), 0.03951331938645079, 1e-10);
    EXPECT_NEAR(cmat(3, 5), 0.02113503129972947, 1e-10);
    EXPECT_NEAR(cmat(4, 0), 0.02048838783001144, 1e-10);
    EXPECT_NEAR(cmat(4, 1), 0.07609972622575735, 1e-10);
    EXPECT_NEAR(cmat(4, 2), 0.01756147528286705, 1e-10);
    EXPECT_NEAR(cmat(4, 3), 0.03951331938645079, 1e-10);
    EXPECT_NEAR(cmat(4, 4), 0.0629286197636069, 1e-10);
    EXPECT_NEAR(cmat(4, 5), 0.0336594942921618, 1e-10);
    EXPECT_NEAR(cmat(5, 0), 0.0109589051183782, 1e-10);
    EXPECT_NEAR(cmat(5, 1), 0.04070450472540505, 1e-10);
    EXPECT_NEAR(cmat(5, 2), 0.009393347244324226, 1e-10);
    EXPECT_NEAR(cmat(5, 3), 0.02113503129972947, 1e-10);
    EXPECT_NEAR(cmat(5, 4), 0.0336594942921618, 1e-10);
    EXPECT_NEAR(cmat(5, 5), 0.018003915551621407, 1e-10);

    S_stress.Clear();
    cmat.Clear();
    summand_.AddStressAnisoPrincipal(C1_strain_, cmat, S_stress, dummyParams, 1, 0);
    EXPECT_NEAR(S_stress(0), 0.015011340776480562, 1e-10);
    EXPECT_NEAR(S_stress(1), 0.07880953907652338, 1e-10);
    EXPECT_NEAR(S_stress(2), 0.055041582847095864, 1e-10);
    EXPECT_NEAR(S_stress(3), 0.03752835194120148, 1e-10);
    EXPECT_NEAR(S_stress(4), 0.07318028628534329, 1e-10);
    EXPECT_NEAR(S_stress(5), 0.028771736488254537, 1e-10);
    EXPECT_NEAR(cmat(0, 0), 0.00223441356538121, 1e-10);
    EXPECT_NEAR(cmat(0, 1), 0.011730671218251415, 1e-10);
    EXPECT_NEAR(cmat(0, 2), 0.00819284973973117, 1e-10);
    EXPECT_NEAR(cmat(0, 3), 0.005586033913453035, 1e-10);
    EXPECT_NEAR(cmat(0, 4), 0.010892766131233479, 1e-10);
    EXPECT_NEAR(cmat(0, 5), 0.0042826260003140035, 1e-10);
    EXPECT_NEAR(cmat(1, 0), 0.011730671218251415, 1e-10);
    EXPECT_NEAR(cmat(1, 1), 0.06158602389582026, 1e-10);
    EXPECT_NEAR(cmat(1, 2), 0.043012461133588885, 1e-10);
    EXPECT_NEAR(cmat(1, 3), 0.02932667804562859, 1e-10);
    EXPECT_NEAR(cmat(1, 4), 0.057187022188976075, 1e-10);
    EXPECT_NEAR(cmat(1, 5), 0.02248378650164864, 1e-10);
    EXPECT_NEAR(cmat(2, 0), 0.00819284973973117, 1e-10);
    EXPECT_NEAR(cmat(2, 1), 0.043012461133588885, 1e-10);
    EXPECT_NEAR(cmat(2, 2), 0.030040449045681216, 1e-10);
    EXPECT_NEAR(cmat(2, 3), 0.02048212434932797, 1e-10);
    EXPECT_NEAR(cmat(2, 4), 0.03994014248118976, 1e-10);
    EXPECT_NEAR(cmat(2, 5), 0.01570296200115148, 1e-10);
    EXPECT_NEAR(cmat(3, 0), 0.005586033913453035, 1e-10);
    EXPECT_NEAR(cmat(3, 1), 0.02932667804562859, 1e-10);
    EXPECT_NEAR(cmat(3, 2), 0.02048212434932797, 1e-10);
    EXPECT_NEAR(cmat(3, 3), 0.013965084783632613, 1e-10);
    EXPECT_NEAR(cmat(3, 4), 0.027231915328083747, 1e-10);
    EXPECT_NEAR(cmat(3, 5), 0.01070656500078503, 1e-10);
    EXPECT_NEAR(cmat(4, 0), 0.010892766131233479, 1e-10);
    EXPECT_NEAR(cmat(4, 1), 0.057187022188976075, 1e-10);
    EXPECT_NEAR(cmat(4, 2), 0.03994014248118976, 1e-10);
    EXPECT_NEAR(cmat(4, 3), 0.027231915328083747, 1e-10);
    EXPECT_NEAR(cmat(4, 4), 0.0531022348897636, 1e-10);
    EXPECT_NEAR(cmat(4, 5), 0.020877801751530922, 1e-10);
    EXPECT_NEAR(cmat(5, 0), 0.0042826260003140035, 1e-10);
    EXPECT_NEAR(cmat(5, 1), 0.02248378650164864, 1e-10);
    EXPECT_NEAR(cmat(5, 2), 0.01570296200115148, 1e-10);
    EXPECT_NEAR(cmat(5, 3), 0.01070656500078503, 1e-10);
    EXPECT_NEAR(cmat(5, 4), 0.020877801751530922, 1e-10);
    EXPECT_NEAR(cmat(5, 5), 0.008208366500601876, 1e-10);

    S_stress.Clear();
    cmat.Clear();
    summand_.AddStressAnisoPrincipal(C2_strain_, cmat, S_stress, dummyParams, 1, 0);
    EXPECT_NEAR(S_stress(0), -0.005404127647681264, 1e-10);
    EXPECT_NEAR(S_stress(1), -0.028371670150326794, 1e-10);
    EXPECT_NEAR(S_stress(2), -0.019815134708164803, 1e-10);
    EXPECT_NEAR(S_stress(3), -0.013510319119203187, 1e-10);
    EXPECT_NEAR(S_stress(4), -0.02634512228244636, 1e-10);
    EXPECT_NEAR(S_stress(5), -0.010357911324722469, 1e-10);
    EXPECT_NEAR(cmat(0, 0), 0.001695279628708896, 1e-10);
    EXPECT_NEAR(cmat(0, 1), 0.008900218050721753, 1e-10);
    EXPECT_NEAR(cmat(0, 2), 0.0062160253052660035, 1e-10);
    EXPECT_NEAR(cmat(0, 3), 0.004238199071772248, 1e-10);
    EXPECT_NEAR(cmat(0, 4), 0.00826448818995593, 1e-10);
    EXPECT_NEAR(cmat(0, 5), 0.003249285955025398, 1e-10);
    EXPECT_NEAR(cmat(1, 0), 0.008900218050721753, 1e-10);
    EXPECT_NEAR(cmat(1, 1), 0.04672614476628945, 1e-10);
    EXPECT_NEAR(cmat(1, 2), 0.0326341328526467, 1e-10);
    EXPECT_NEAR(cmat(1, 3), 0.02225054512680442, 1e-10);
    EXPECT_NEAR(cmat(1, 4), 0.04338856299726886, 1e-10);
    EXPECT_NEAR(cmat(1, 5), 0.017058751263883433, 1e-10);
    EXPECT_NEAR(cmat(2, 0), 0.0062160253052660035, 1e-10);
    EXPECT_NEAR(cmat(2, 1), 0.0326341328526467, 1e-10);
    EXPECT_NEAR(cmat(2, 2), 0.02279209278597554, 1e-10);
    EXPECT_NEAR(cmat(2, 3), 0.015540063263165042, 1e-10);
    EXPECT_NEAR(cmat(2, 4), 0.030303123363171994, 1e-10);
    EXPECT_NEAR(cmat(2, 5), 0.011914048501759892, 1e-10);
    EXPECT_NEAR(cmat(3, 0), 0.004238199071772248, 1e-10);
    EXPECT_NEAR(cmat(3, 1), 0.02225054512680442, 1e-10);
    EXPECT_NEAR(cmat(3, 2), 0.015540063263165042, 1e-10);
    EXPECT_NEAR(cmat(3, 3), 0.010595497679430639, 1e-10);
    EXPECT_NEAR(cmat(3, 4), 0.02066122047488986, 1e-10);
    EXPECT_NEAR(cmat(3, 5), 0.008123214887563509, 1e-10);
    EXPECT_NEAR(cmat(4, 0), 0.00826448818995593, 1e-10);
    EXPECT_NEAR(cmat(4, 1), 0.04338856299726886, 1e-10);
    EXPECT_NEAR(cmat(4, 2), 0.030303123363171994, 1e-10);
    EXPECT_NEAR(cmat(4, 3), 0.02066122047488986, 1e-10);
    EXPECT_NEAR(cmat(4, 4), 0.04028937992603545, 1e-10);
    EXPECT_NEAR(cmat(4, 5), 0.015840269030748932, 1e-10);
    EXPECT_NEAR(cmat(5, 0), 0.003249285955025398, 1e-10);
    EXPECT_NEAR(cmat(5, 1), 0.017058751263883433, 1e-10);
    EXPECT_NEAR(cmat(5, 2), 0.011914048501759892, 1e-10);
    EXPECT_NEAR(cmat(5, 3), 0.008123214887563509, 1e-10);
    EXPECT_NEAR(cmat(5, 4), 0.015840269030748932, 1e-10);
    EXPECT_NEAR(cmat(5, 5), 0.006227798080465372, 1e-10);
  }
}  // namespace
