/*----------------------------------------------------------------------*/
/*! \file
\brief
St. Venant-Kirchhoff material

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_STVENANTKIRCHHOFF_HPP
#define FOUR_C_MAT_STVENANTKIRCHHOFF_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for St. Venant--Kirchhoff
    class StVenantKirchhoff : public CORE::MAT::PAR::Parameter
    {
     public:
      /// standard constructor
      StVenantKirchhoff(Teuchos::RCP<CORE::MAT::PAR::Material> matdata);

      /// @name material parameters
      //@{

      /// Young's modulus
      const double youngs_;
      /// Possion's ratio
      const double poissonratio_;
      /// mass density
      const double density_;

      //@}

      Teuchos::RCP<CORE::MAT::Material> create_material() override;

    };  // class StVenantKirchhoff
  }     // namespace PAR

  class StVenantKirchhoffType : public CORE::COMM::ParObjectType
  {
   public:
    [[nodiscard]] std::string Name() const override { return "StVenantKirchhoffType"; }

    static StVenantKirchhoffType& Instance() { return instance_; };

    CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

   private:
    static StVenantKirchhoffType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for St.-Venant-Kirchhoff material
  class StVenantKirchhoff : public So3Material
  {
   public:
    /// construct empty material object
    StVenantKirchhoff();

    /// construct the material object given material parameters
    explicit StVenantKirchhoff(MAT::PAR::StVenantKirchhoff* params);

    [[nodiscard]] int UniqueParObjectId() const override
    {
      return StVenantKirchhoffType::Instance().UniqueParObjectId();
    }

    void Pack(CORE::COMM::PackBuffer& data) const override;

    void Unpack(const std::vector<char>& data) override;

    //@}

    //! @name Access methods

    [[nodiscard]] CORE::Materials::MaterialType MaterialType() const override
    {
      return CORE::Materials::m_stvenant;
    }

    void ValidKinematics(INPAR::STR::KinemType kinem) override
    {
      if (kinem != INPAR::STR::KinemType::linear && kinem != INPAR::STR::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    [[nodiscard]] Teuchos::RCP<CORE::MAT::Material> Clone() const override
    {
      return Teuchos::rcp(new StVenantKirchhoff(*this));
    }

    /// Young's modulus
    [[nodiscard]] double Youngs() const { return params_->youngs_; }

    /// Poisson's ratio
    [[nodiscard]] double PoissonRatio() const { return params_->poissonratio_; }

    [[nodiscard]] double Density() const override { return params_->density_; }

    /// shear modulus
    [[nodiscard]] double shear_mod() const
    {
      return 0.5 * params_->youngs_ / (1.0 + params_->poissonratio_);
    }

    [[nodiscard]] CORE::MAT::PAR::Parameter* Parameter() const override { return params_; }

    //@}

    //! @name Evaluation methods

    /// evaluates material law
    void Evaluate(const CORE::LINALG::SerialDenseVector* glstrain_e,
        CORE::LINALG::SerialDenseMatrix* cmat_e, CORE::LINALG::SerialDenseVector* stress_e);

    void Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
        const CORE::LINALG::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        CORE::LINALG::Matrix<6, 1>* stress, CORE::LINALG::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    void StrainEnergy(
        const CORE::LINALG::Matrix<6, 1>& glstrain, double& psi, int gp, int eleGID) override;

    // computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(CORE::LINALG::Matrix<6, 6>& cmat);
    //@}

    //! general setup of constitutive tensor based on Young's and poisson's ratio
    static void FillCmat(CORE::LINALG::Matrix<6, 6>& cmat, double Emod, double nu);

   private:
    /// my material parameters
    MAT::PAR::StVenantKirchhoff* params_;
  };
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif