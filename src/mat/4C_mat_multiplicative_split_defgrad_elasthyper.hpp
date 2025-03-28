// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_HPP
#define FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_monolithic_solid_scalar_material.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


// forward declaration
namespace Mat
{
  class Anisotropy;
  class InelasticDefgradFactors;

  namespace Elastic
  {
    class Summand;
  }

  namespace PAR
  {
    enum class InelasticSource
    {
      none,
      concentration,
      temperature
    };

    class MultiplicativeSplitDefgradElastHyper : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit MultiplicativeSplitDefgradElastHyper(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// length of elastic material list
      const int nummat_elast_;

      /// the list of elastic material IDs
      const std::vector<int> matids_elast_;

      /// number of factors of inelastic deformation gradient F_{in} = F_{in,1} . F_{in,2}. ... .
      /// F_{in,n} (n factors)
      const int numfac_inel_;

      /// IDs of inelastic deformation gradient factors (i-th ID specifies calculation of F_{in,i})
      const std::vector<int> inel_defgradfacids_;

      /// material mass density
      const double density_;

    };  // class MultiplicativeSplitDefgrad_ElastHyper

  }  // namespace PAR

  class MultiplicativeSplitDefgradElastHyperType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "MultiplicativeSplitDefgrad_ElastHyperType"; }

    static MultiplicativeSplitDefgradElastHyperType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MultiplicativeSplitDefgradElastHyperType instance_;
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticFactorsHandler

     All factors (class InelasticDefgradFactors) contributing to inelastic deformation are stored
     within this class. They are initially classified according to the source of inelastic
     deformation (e.g. concentration). All contributions from one source can be returned.
  */
  class InelasticFactorsHandler
  {
   public:
    /*!
     * @brief Evaluate inelastic deformation gradient from all factors and assign to different
     * sources
     *
     * @param[in]  defgrad  Deformation gradient
     * @param[out] iFinM    inverse inelastic deformation gradient
     */
    void evaluate_inverse_inelastic_def_grad(
        const Core::LinAlg::Matrix<3, 3>* defgrad, Core::LinAlg::Matrix<3, 3>& iFinM);

    /// Returns all inelastic factors as a vector
    const std::vector<
        std::pair<PAR::InelasticSource, std::shared_ptr<Mat::InelasticDefgradFactors>>>&
    fac_def_grad_in() const
    {
      return facdefgradin_;
    }

    /// Return vector of all inelastic deformation gradients
    const std::vector<std::pair<PAR::InelasticSource, Core::LinAlg::Matrix<3, 3>>>& geti_finj()
        const
    {
      return i_finj_;
    }

    /// total number of inelastic contributions
    int num_inelastic_def_grad() const { return static_cast<int>(facdefgradin_.size()); }

    /// Assigns the different inelastic factors to different sources
    void assign_to_source(Mat::PAR::MultiplicativeSplitDefgradElastHyper* params);

    /// Update the history variables for the different inelastic factors
    void update();

    /// Set initial conditions for the history variables of all Gauss points
    void setup(const int numgp, const Core::IO::InputParameterContainer& container);

    /// Pack
    void pack_inelastic(Core::Communication::PackBuffer& data) const;

    /// Unpack
    void unpack_inelastic(Core::Communication::UnpackBuffer& buffer);

   private:
    /// vector that holds pairs of inelastic contribution and respective source
    std::vector<std::pair<PAR::InelasticSource, std::shared_ptr<Mat::InelasticDefgradFactors>>>
        facdefgradin_;

    /// vector that holds pairs of inelastic deformation gradients and respective source
    std::vector<std::pair<PAR::InelasticSource, Core::LinAlg::Matrix<3, 3>>> i_finj_;
  };

  /*----------------------------------------------------------------------*/
  /*! @class MultiplicativeSplitDefgrad_ElastHyper

    In this class the deformation gradient is modeled to be split multiplicatively
    in elastic and inelastic deformation gradients (F = F_{el} * F_{in}).
    The elastic contribution can be any elastic material law provided in elast_summand.
    Only elastic strains cause stresses!
    The inelastic deformation gradient itself can be a product of different inelastic
    deformation gradients, i.e. F_{in} = F_{in,1} * F_{in,2} * ... * F_{in,n}.
    The inverse inelastic deformation gradients and their derivative w.r.t. the primary variables
    that are needed to set up the system to be solved are evaluated in the derived classes
    of the interface class 'InelasticDefgradFactors'.
*/
  class MultiplicativeSplitDefgradElastHyper : public So3Material,
                                               public MonolithicSolidScalarMaterial
  {
   public:
    /// construct empty material object
    MultiplicativeSplitDefgradElastHyper();

    /// construct the material object given material parameters
    explicit MultiplicativeSplitDefgradElastHyper(
        Mat::PAR::MultiplicativeSplitDefgradElastHyper* params);

    /// struct containing various kinematic quantities for the model evaluation
    struct KinematicQuantities
    {
      // ----- variables of kinetic quantities ----- //

      /// inverse right Cauchy-Green tensor \f$ \mathbf{C}^{-1} \f$ stored as 6x1 vector
      Core::LinAlg::Matrix<6, 1> iCV{Core::LinAlg::Initialization::zero};
      /// inverse inelastic right Cauchy-Green tensor \f$\mathbf{C}_\text{in}^{-1}\f$ stored as 6x1
      /// vector
      Core::LinAlg::Matrix<6, 1> iCinV{Core::LinAlg::Initialization::zero};
      /// \f$ \mathbf{C}_\text{in}^{-1} \cdot \mathbf{C} \cdot \mathbf{C}_\text{in}^{-1} \f$ stored
      /// as 6x1 vector
      Core::LinAlg::Matrix<6, 1> iCinCiCinV{Core::LinAlg::Initialization::zero};
      /// \f$ \mathbf{C}_\text{in}^{-1} \cdot \mathbf{C} \f$
      Core::LinAlg::Matrix<3, 3> iCinCM{Core::LinAlg::Initialization::zero};
      ///\f$ \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el} \f$
      Core::LinAlg::Matrix<3, 3> iFinCeM{Core::LinAlg::Initialization::zero};
      /// \f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \f$ stored as 9x1 vector
      Core::LinAlg::Matrix<9, 1> CiFin9x1{Core::LinAlg::Initialization::zero};
      ///\f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el} \f$ stored as 9x1
      /// vector
      Core::LinAlg::Matrix<9, 1> CiFinCe9x1{Core::LinAlg::Initialization::zero};
      /// \f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el}^{-1} \f$ stored
      /// as 9x1 vector
      Core::LinAlg::Matrix<9, 1> CiFiniCe9x1{Core::LinAlg::Initialization::zero};
      /// principal invariants of the elastic right Cauchy-Green tensor
      Core::LinAlg::Matrix<3, 1> prinv{Core::LinAlg::Initialization::zero};
      /// partial derivative of the elastic right Cauchy-Green tensor w.r.t. right Cauchy-Green
      /// tensor \f$ \frac{\partial \boldsymbol{C}_\text{e}}{\partial \boldsymbol{C}} \f$ (Voigt
      /// stress-stress notation)
      Core::LinAlg::Matrix<6, 6> dCedC{Core::LinAlg::Initialization::zero};
      /// partial derivative of the elastic right Cauchy-Green tensor w.r.t. inelastic deformation
      /// gradient \f$ \frac{\partial \boldsymbol{C}_\text{e}}{\partial \boldsymbol{F} _\text{in} ^
      /// { -1 }} \f$(Voigt stress notation)
      Core::LinAlg::Matrix<6, 9> dCediFin{Core::LinAlg::Initialization::zero};
      /// inverse inelastic deformation gradient
      Core::LinAlg::Matrix<3, 3> iFinM{Core::LinAlg::Initialization::zero};
      /// determinant of the inelastic deformation gradient
      double detFin = 1.0;

      // ----- derivatives of principal invariants ----- //

      /// first derivatives of principle invariants
      Core::LinAlg::Matrix<3, 1> dPIe{Core::LinAlg::Initialization::zero};
      /// second derivatives of principle invariants
      Core::LinAlg::Matrix<6, 1> ddPIIe{Core::LinAlg::Initialization::zero};
    };

    /// struct containing free-energy related stress factors, as presented in Holzapfel-Nonlinear
    /// Solid Mechanics
    struct StressFactors
    {
      // ----- gamma and delta factors ----- //

      // 2nd Piola Kirchhoff stresses factors (according to Holzapfel-Nonlinear Solid Mechanics p.
      // 216)
      Core::LinAlg::Matrix<3, 1> gamma{Core::LinAlg::Initialization::zero};
      // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
      Core::LinAlg::Matrix<8, 1> delta{Core::LinAlg::Initialization::zero};
    };

    int unique_par_object_id() const override
    {
      return MultiplicativeSplitDefgradElastHyperType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (kinem != Inpar::Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_multiplicative_split_defgrad_elasthyper;
    }

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MultiplicativeSplitDefgradElastHyper>(*this);
    }

    double density() const override { return params_->density_; }

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    Core::LinAlg::Matrix<6, 1> evaluate_d_stress_d_scalar(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<6, 1>& glstrain, Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    void evaluate_cauchy_n_dir_and_derivatives(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Core::LinAlg::Matrix<3, 1>& n, const Core::LinAlg::Matrix<3, 1>& dir,
        double& cauchy_n_dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
        Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
        Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, int gp, int eleGID,
        const double* concentration, const double* temp, double* d_cauchyndir_dT,
        Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT) override;

    void evaluate_linearization_od(const Core::LinAlg::Matrix<3, 3>& defgrd, double concentration,
        Core::LinAlg::Matrix<9, 1>* d_F_dx) override;

    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    void update() override;

    /*!
     * @brief Evaluate off-diagonal stiffness matrix (required for monolithic algorithms)
     *
     * @param[in] source   Source of inelastic deformation
     * @param[in] defgrad  Deformation gradient
     * @param[in] dSdiFin  Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                     deformation gradient
     * @param[out] dstressdx  Derivative of 2nd Piola Kirchhoff stresses w.r.t. primary variable of
     *                        different field
     */
    void evaluate_od_stiff_mat(PAR::InelasticSource source,
        const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<6, 9>& dSdiFin,
        Core::LinAlg::Matrix<6, 1>& dstressdx);

    /*!
     * @brief Evaluate additional terms of the elasticity tensor
     *
     * \f[
     *   cmat_{add} = \frac{\partial\mathbf{S}}{\partial \mathbf{F}_\text{in}^{-1}}
     *             : \frac{\partial\mathbf{F}_\text{in}^{-1}}{\partial \mathbf{C}}
     * \f]
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] iCV      Inverse right Cauchy-Green tensor stored as 6x1 vectors
     * @param[in] dSdiFin  Derivative of 2nd Piola Kirchhoff stress w.r.t. the inverse inelastic
     *                     deformation gradient
     * @return Additional elasticity tensor \f$ cmat_{add} \f$
     */
    Core::LinAlg::Matrix<6, 6> evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFin);

    /*!
     * @brief  Evaluate derivative of 2nd Piola Kirchhoff stresses w.r.t. the inelastic deformation
     * gradient for the isotropic components
     *
     * @param[in] kinemat_quant struct containing kinematic quantities
     * @param[in] stress_fact struct containing \f$ \gamma_i \f$ and \f$ \delta_i \f$ stress
     * factors, as presented in Holzapfel - Nonlinear Solid Mechanics
     * @return derivative \f$ \frac{\partial \mathsymbol{S}}{\partial
     * \mathsymbol{F}^{-1}_{\text{in}}} \f$
     */
    Core::LinAlg::Matrix<6, 9> evaluated_sdi_fin(
        const KinematicQuantities& kinemat_quant, const StressFactors& stress_factors) const;

    /*!
     * @brief  Evaluate the stress and stiffness components of the transversely isotropic components
     * separately
     *
     * @param[in] kinemat_quant struct containing kinematic quantities
     * @param[in] CM Right CG deformation tensor
     * \param[in]  params additional parameters for comp. of material properties
     * \param[in]  gp       Gauss point
     * \param[in]  eleGID element GID
     * @param[out] stress   2nd Piola--Kirchhoff stress tensor
     * @param[out] cmatiso  part of the elasticity tensor as shown in evaluate_stress_cmat_iso
     * @param[out] dSdiFin derivative of 2nd Piola Kirchhoff stresses w.r.t. the inelastic
     * deformation gradient
     */
    void evaluate_transv_iso_quantities(const KinematicQuantities& kinemat_quant,
        const Core::LinAlg::Matrix<3, 3>& CM, Teuchos::ParameterList& params, const int gp,
        const int eleGID, Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmatiso,
        Core::LinAlg::Matrix<6, 9>& dSdiFin) const;

    /*!
     * @brief Evaluates stress and cmat
     *
     * \f[
     *   cmat = 2 \frac{\partial \mathbf{S}}{\partial \mathbf{C}}
     * \f]
     *
     *  cf. Holzapfel [1], p. 216 (6.32) and p. 248
     *  The formulation in Holzapfel is slightly changed as only the elastic strains cause stresses
     *  with \f$\mathbf{C}_\text{in} = \mathbf{F}_\text{in}^T \cdot \mathbf{F}_\text{in}\f$
     *  and \f$\mathbf{F}_\text{in}\f$ denoting the inelastic deformation gradient
     * \f[
     *   \mathbf{S} = 2 \det(\mathbf{F}_\text{in}) \mathbf{F}_\text{in}^{-1} \frac{\partial
     *   \Psi_\text{e}}{\partial \mathbf{C}_\text{e}} \mathbf{F}_\text{in}^{-T}
     * \f]
     * \f[
     * \mathbf{S} = \det(\mathbf{F}_\text{in}) \left(\gamma_1 \ \mathbf{C}_\text{in}^{-1} + \gamma_2
     * \
     * \mathbf{C}_\text{in}^{-1} \cdot \mathbf{C} \cdot \mathbf{C}_\text{in}^{-1} + \gamma_3 \
     * \mathbf{C}^{-1} \right)
     * \f] with \f$\gamma_i\f$ as defined in Mat::CalculateGammaDelta()
     *
     * \f[
     *   \mathbb{C} = \det(\mathbf{F}_\text{in}) \left( \delta_1 \left( \mathbf{C}_\text{in}^{-1}
     *                \otimes \mathbf{C}_\text{in}^{-1} \right)
     *              + \delta_2 \left( \mathbf{C}_\text{in}^{-1} \otimes \mathbf{C}_\text{in}^{-1}
     *                \mathbf{C} \mathbf{C}_\text{in}^{-1}  + \mathbf{C}_\text{in}^{-1} \mathbf{C}
     *                \mathbf{C}_\text{in}^{-1} \otimes \mathbf{C}_\text{in}^{-1} \right)
     *              + \delta_3 \left( \mathbf{C}_\text{in}^{-1} \otimes \mathbf{C}^{-1} +
     * \mathbf{C}^{-1} \otimes \mathbf{C}_\text{in}^{-1} \right)
     *              + \delta_4 \left( \mathbf{C}_\text{in}^{-1} \mathbf{C} \mathbf{C}_\text{in}^{-1}
     *                \otimes \mathbf{C}_\text{in}^{-1} \mathbf{C} \mathbf{C}_\text{in}^{-1} \right)
     *              + \delta_5 \left( \mathbf{C}_\text{in}^{-1} \mathbf{C} \mathbf{C}_\text{in}^{-1}
     *                \otimes \mathbf{C}^{-1} + \mathbf{C}^{-1} \otimes \mathbf{C}_\text{in}^{-1}
     * \mathbf{C} \mathbf{C}_\text{in}^{-1} \right)
     *              + \delta_6 \left( \mathbf{C}^{-1} \otimes \mathbf{C}^{-1} \right)
     *              + \delta_7 \left( \mathbf{C}^{-1} \odot \mathbf{C}^{-1} \right)
     *              + \delta_8 \left( \mathbf{C}_\text{in}^{-1} \odot \mathbf{C}_\text{in}^{-1}
     * \right) \right) \f]
     *
     * \f$ \odot \f$ is defined as follows:
     * \f[
     *   - (A^{-1} \odot A^{-1})_{abcd} = 1/2 (A^{-1}_{ac} A^{-1}_{bd} +
     *   A^{-1}_{ad} A^{-1}_{bc}); \text{ for } A_{ab} = A_{ba}
     * \f] meaning a symmetric 2-tensor \f$A\f$
     *
     * @param[in] kinemat_quant struct containing kinematic quantities
     * @param[in] stress_fact struct containing \f$ \gamma_i \f$ and \f$ \delta_i \f$ stress
     * factors, as presented in Holzapfel - Nonlinear Solid Mechanics
     * @param[out] stress   2nd Piola--Kirchhoff stress tensor
     * @param[out] cmatiso  part of the elasticity tensor as shown above
     */
    void evaluate_stress_cmat_iso(const KinematicQuantities& kinemat_quant,
        const StressFactors& stress_fact, Core::LinAlg::Matrix<6, 1>& stress,
        Core::LinAlg::Matrix<6, 6>& cmatiso) const;

    /*!
     * @brief Evaluates some kinematic quantities that are used in stress and elasticity tensor
     * calculation
     *
     * @param[in] defgrad  Deformation gradient
     * @param[out] kinemat_quant struct containing kinematic quantities
     */
    void evaluate_kin_quant_elast(
        const Core::LinAlg::Matrix<3, 3>* defgrad, KinematicQuantities& kinemat_quant) const;

    /*!
     * @brief calculates the derivatives of the hyper-elastic laws with respect to the invariants
     *
     * @param[in] prinv   Principal invariants of the elastic right Cauchy-Green tensor
     * @param[in] gp      current gauss point
     * @param[in] eleGID  Element ID
     * @param[out] dPI    First derivative w.r.t. principle invariants
     * @param[out] ddPII  Second derivative w.r.t. principle invariants
     */
    void evaluate_invariant_derivatives(const Core::LinAlg::Matrix<3, 1>& prinv, int gp, int eleGID,
        Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII) const;

    /*!
     * @brief pre-evaluation, intended to be used for stuff that have to be done only once
     *
     * @param[in] params  parameter list as handed in from the element
     * @param[in] gp      current gauss point
     * @param[in] eleGID  Element ID
     */
    void pre_evaluate(Teuchos::ParameterList& params, int gp, int eleGID) const;

    /*!
     * @brief set the gauss point concentration to the respective parameter class of the inelastic
     * material
     *
     * @param[in]  concentration concentration at gauss point
     */
    void set_concentration_gp(double concentration);

   private:
    /// Holder for anisotropy
    std::shared_ptr<Anisotropy> anisotropy_;

    /// Holds and classifies all inelastic factors
    std::shared_ptr<InelasticFactorsHandler> inelastic_;

    /// My material parameters
    Mat::PAR::MultiplicativeSplitDefgradElastHyper* params_;

    /// map to elastic materials/potential summands (only isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsumel_;

    /// map to elastic materials/potential summands (only transversely isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> potsumel_transviso_;

    KinematicQuantities evaluate_kinematic_quantities(
        const Mat::MultiplicativeSplitDefgradElastHyper& splitdefgrd,
        Mat::InelasticFactorsHandler& inelastic_factors_handler,
        const Core::LinAlg::Matrix<3, 3>& defgrad, const int gp, const int eleGID);
  };

}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
