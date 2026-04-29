// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PLASTICDRUCKERPRAGER_HPP
#define FOUR_C_MAT_PLASTICDRUCKERPRAGER_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /**
     * \brief Elasto-plasitc Drucker-Prager Material Model
     *
     * This material model simulates the elasto-plastic behaviour of materials including soil
     * and concrete based on the yield surface criteria and the plastic hardening of the material.
     *
     * Following the approach provided by EA de Souza Neto, D Peric, DRJ Owen.
     * Computational Methods of Plasticity: Theory and Applications, Page 338-339
     */
    class PlasticDruckerPrager : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticDruckerPrager(const Core::Mat::PAR::Parameter::Data& matdata);
      //! @name material parameters
      //@{
      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! Density
      const double density_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! tolerance for local Newton iteration
      const double abstol_;
      //! initial cohesion
      const double cohesion_;
      //! friction angle variable
      const double eta_;
      //! hardening factor
      const double xi_;
      //! dilatancy angle variable
      const double etabar_;
      //! method to compute the material tangent
      const std::string tang_;
      //! maximum iterations for local system solve
      const int itermax_;
      //@}
      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR
  class PlasticDruckerPragerType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "PlasticDruckerPragerType"; }
    static PlasticDruckerPragerType& instance() { return instance_; };
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static PlasticDruckerPragerType instance_;
  };

  class PlasticDruckerPrager : public So3Material
  {
   public:
    PlasticDruckerPrager();

    explicit PlasticDruckerPrager(Mat::PAR::PlasticDruckerPrager* params);
    int unique_par_object_id() const override
    {
      return PlasticDruckerPragerType::instance().unique_par_object_id();
    }
    void pack(Core::Communication::PackBuffer& data) const override;
    void unpack(Core::Communication::UnpackBuffer& buffer) override;
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_pldruckprag;
    }
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (kinem != Inpar::Solid::KinemType::linear)
        FOUR_C_THROW(
            "The plastic Drucker Prager material model is only compatible with linear kinematics.");
    }
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<PlasticDruckerPrager>(*this);
    }
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;
    void update() override;
    /**
     * \brief Evaluate the stresses from the strains in the material
     *
     * \param defgrad :deformation gradient
     * \param linstrain :linear total strains
     * \param params :parameter list for communication
     * \param stress :2nd PK-stress
     * \param cmat :material stiffness matrix
     * \param gp :Gauss point
     * \param eleGID :element global identifier
     */
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void setup_cmat(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat) const;
    /**
     * \brief setup the elastoplasticity tensor for 3d return to cone
     *
     * \param cmat :elasto-plastic tangent modulus (out)
     * \param Dgamma :plastic multiplier
     * \param G :shear modulus
     * \param Kappa :Bulk modulus
     * \param devstrain :deviatoric strain tensor
     * \param xi :Mohr-Coulomb parameter
     * \param Hiso :isotropic hardening modulus
     * \param eta :Mohr-Coulomb parameter
     * \param etabar :Mohr-Coulomb parameter
     */
    void setup_cmat_elasto_plastic_cone(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
        double Dgamma, double G, double Kappa,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& devstrain, double xi, double Hiso,
        double eta, double etabar) const;
    /**
     * \brief setup the elastoplasticity tensor for 3d return to apex
     *
     * \param cmat :elasto-plastic tangent modulus (out)
     * \param Kappa :Bulk modulus
     * \param xi :Mohr-Coulomb parameter
     * \param Hiso :isotropic hardening modulus
     * \param eta :Mohr-Coulomb parameter
     * \param etabar :Mohr-Coulomb parameter
     */
    void setup_cmat_elasto_plastic_apex(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
        double Kappa, double xi, double Hiso, double eta, double etabar) const;

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }
    double density() const override { return params_->density_; }
    std::pair<double, double> return_to_cone_funct_and_deriv(
        double Dgamma, double G, double kappa, double Phi_trial);
    std::pair<double, double> return_to_apex_funct_and_deriv(
        double dstrainv, double p, double kappa, double strainbar_p);
    bool initialized() const { return (isinit_ and !strainplcurr_.empty()); }
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    Mat::PAR::PlasticDruckerPrager* params_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> strainpllast_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> strainplcurr_;
    std::vector<double> strainbarpllast_;
    std::vector<double> strainbarplcurr_;
    bool isinit_;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
