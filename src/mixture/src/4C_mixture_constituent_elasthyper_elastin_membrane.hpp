// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPER_ELASTIN_MEMBRANE_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPER_ELASTIN_MEMBRANE_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension.hpp"
#include "4C_mixture_constituent_elasthyperbase.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class Anisotropy;
  namespace Elastic
  {
    class StructuralTensorStrategyBase;
    class IsoNeoHooke;
  }  // namespace Elastic
}  // namespace Mat

namespace Mixture
{
  class MixtureConstituentElastHyperElastinMembrane;

  /*!
   * \brief Anisotropy extension for elastin material.
   *
   * The anisotropy extension provides the structural tensor of the plane of the membrane, which is
   * orthogonal to the radial direction
   */
  class ElastinMembraneAnisotropyExtension : public Mat::FiberAnisotropyExtension<1>
  {
   public:
    /*!
     * \brief Constructor of the new elastin anisotropy extension
     *
     * \param structuralTensorStrategy Structural tensor strategy to compute the structural tensors
     */
    explicit ElastinMembraneAnisotropyExtension(
        const std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>&
            structuralTensorStrategy);

    /*!
     * \brief This method will be called when all global data is initialized. Here we need to create
     * the membrane plane structural tensor
     */
    void on_global_data_initialized() override;

    /*!
     * \brief Returns the structural tensor of the membrane plane at the Gauss point
     *
     * \param gp (in) : Gauss point
     * \return const Core::LinAlg::Matrix<3, 3>& Reference to the structural tensor of the membrane
     * plane
     */
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_orthogonal_structural_tensor(int gp);

   private:
    /// Holder of the internal structural tensors
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> orthogonal_structural_tensor_;
  };

  namespace PAR
  {
    class MixtureConstituentElastHyperElastinMembrane
        : public Mixture::PAR::MixtureConstituentElastHyperBase
    {
     public:
      /*!
       * \brief Construct a new elastin material with a membrane
       *
       * \param matdata Material parameters
       * \param ref_mass_fraction reference mass fraction
       */
      explicit MixtureConstituentElastHyperElastinMembrane(
          const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) override;

      /// @name material parameters
      /// @{
      const int damage_function_id_;

      /// number of summands
      const int nummat_membrane_;

      /// List of material ids of the summands
      const std::vector<int> matids_membrane_;
      /// @}
    };
  }  // namespace PAR

  /*!
   * \brief Constituent for any hyperelastic material
   *
   * This constituent represents any hyperelastic material from the elasthyper toolbox. It has to
   * be paired with the Mat::Mixture material and a Mixture::MixtureRule.
   */
  class MixtureConstituentElastHyperElastinMembrane
      : public Mixture::MixtureConstituentElastHyperBase,
        public ElastinMembraneEvaluation
  {
   public:
    /*!
     * \brief Constructor for the material given the material parameters
     *
     * \param params Material parameters
     */
    explicit MixtureConstituentElastHyperElastinMembrane(
        Mixture::PAR::MixtureConstituentElastHyperElastinMembrane* params, int id);

    /// Returns the material type enum
    [[nodiscard]] Core::Materials::MaterialType material_type() const override;

    /*!
     * \brief Pack data into a char vector from this class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by unique_par_object_id().
     *
     * @param data (in/put) : vector storing all data to be packed into this instance.
     */
    void pack_constituent(Core::Communication::PackBuffer& data) const override;

    /*!
     * \brief Unpack data from a char vector into this class to be called from a derived class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by unique_par_object_id().
     *
     * @param position (in/out) : current position to unpack data
     * @param data (in) : vector storing all data to be unpacked into this instance.
     */
    void unpack_constituent(Core::Communication::UnpackBuffer& buffer) override;

    /*!
     * \brief Register anisotropy extensions to the global anisotropy manager
     *
     * \param anisotropy Reference to the global anisotropy manager
     */
    void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

    /*!
     * Initialize the constituent with the parameters of the input line
     *
     * @param numgp (in) Number of Gauss-points
     * @param params (in/out) Parameter list for exchange of parameters
     */
    void read_element(int numgp, const Core::IO::InputParameterContainer& container) override;


    /*!
     * \brief Updates the material and all its summands
     *
     * This method is called once between each timestep after convergence.
     *
     * @param defgrd Deformation gradient
     * @param params Container for additional information
     * @param gp Gauss point
     * @param eleGID Global element identifier
     */
    void update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd,
        const Teuchos::ParameterList& params, int gp, int eleGID) override;

    /*!
     * \brief Method that will be called once before the evaluation process. The elastin material
     * evaluates the prestress here.
     *
     * \param mixtureRule Mixture rule
     * \param params Container for additional information
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void pre_evaluate(MixtureRule& mixtureRule, const Teuchos::ParameterList& params, int gp,
        int eleGID) override;


    [[nodiscard]] double get_growth_scalar(int gp) const override;

    /*!
     * \brief Standard evaluation of the material. This material does only support evaluation with
     * an elastic part.
     *
     * \param F Total deformation gradient
     * \param E_strain Green-Lagrange strain tensor
     * \param params Container for additional information
     * \param S_stress 2. Piola-Kirchhoff stress tensor in stress-like Voigt notation
     * \param cmat Constitutive tensor
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /*!
     * \brief Evaluation of the constituent with an inelastic, external part.
     *
     * \param F Total deformation gradient
     * \param iF_in inverse inelastic (external) stretch tensor
     * \param params Container for additional information
     * \param S_stress 2. Piola Kirchhoff stress tensor in stress-like Voigt notation
     * \param cmat Constitutive tensor
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void evaluate_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iFextin, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /*!
     * \brief Evaluation of the membrane stress only
     *
     * \param S 2. Piola Kirchhoff stress tensor of the membrane in stress-like Voigt notation
     * \param params Container for additional information
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void evaluate_membrane_stress(Core::LinAlg::SymmetricTensor<double, 3, 3>& S,
        const Teuchos::ParameterList& params, int gp, int eleGID) override;

   protected:
    /*!
     * \brief Evaluation of the stress and constitutive tensor of the membrane material
     *
     * \param F Total deformation gradient
     * \param iFin Inelastic deformation gradient
     * \param params Container for additional information
     * \param S_stress 2. Piola-Kirchhoff stress tensor in stress-like Voigt notation
     * \param cmat Constitutive tensor
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void evaluate_stress_c_mat_membrane(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iFin, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) const;

    /*!
     * \brief Evaluates the structural tensors of the radial and membrane plane direction in the
     * grown configuration
     *
     * \param Aradgr Structural tensor in radial direction in the grown configuration
     * \param Aorthgr Structural tensor in the membrane direction in the grown configuration
     * \param iFin Inverse of the inelastic deformation gradient
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void evaluate_structural_tensors_in_grown_configuration(Core::LinAlg::Matrix<3, 3>& Aradgr,
        Core::LinAlg::Matrix<3, 3>& Aorthgr, const Core::LinAlg::Matrix<3, 3>& iFin, int gp,
        int eleGID) const;

    /*!
     * \brief Evaluate the matrix Product \[
     *      A_{orth,gr} C_e A_{orth,gr} + A_{rad,gr}
     * \]
     *
     * \param AorthgrCeAorthgrArad (out) : desired product
     * \param Aradgr (in) : Structural tensor of the radial direction in the grown configuration
     * \param Aorthgr (in) : Structural tensor of the membrane plane direction in the grown
     * configuration
     *
     * \param Ce (in) : Elastic Cauchy-Green deformation tensor
     */
    static void evaluate_aorthgr_ce_aorthgr_arad(Core::LinAlg::Matrix<3, 3>& AorthgrCeAorthgrArad,
        const Core::LinAlg::Matrix<3, 3>& Aradgr, const Core::LinAlg::Matrix<3, 3>& Aorthgr,
        const Core::LinAlg::Matrix<3, 3>& Ce);

    /*!
     * \brief Evaluate the matrix product \[
     *  F_{in}^{-1} A_{orth,gr} F_{in}^{-T}
     * \]
     *
     * \param iFinAorthgriFinT (out) : Desired product
     * \param iFin (in) : Inelastic inverse deformation gradient
     * \param Aorthgr (in) : Structural tensor of the membrane plane direction in grown
     * configuration
     */
    static void evaluatei_fin_aorthgri_fin_t(Core::LinAlg::Matrix<3, 3>& iFinAorthgriFinT,
        const Core::LinAlg::Matrix<3, 3>& iFin, const Core::LinAlg::Matrix<3, 3>& Aorthgr);

    /*!
     * \brief Evaluates the prioduct \[
     *  F_{in}^{-T} A_{orth,gr}^T (A_{orth,gr} C_e A_{orth,gr} + A_{rad,gr})^{-1} A_{orth,gr}
     * F_{in}^{-1}
     * \]
     *
     * \param iFinTAorthgrTiXTAorthgriFin Desired product
     * \param AorthgrCeAorthgrArad Product \[A_{orth,gr} C_e A_{orth,gr} + A_{rad,gr}/]
     * \param iFin inverse of the inelastic part of the deformation
     * \param Aorthgr Structural tensor of the membrane plane direction in grown configuration
     */
    static void evaluatei_fin_t_aorthgr_ti_xt_aorthgri_fin(
        Core::LinAlg::Matrix<3, 3>& iFinTAorthgrTiXTAorthgriFin,
        const Core::LinAlg::Matrix<3, 3>& AorthgrCeAorthgrArad,
        const Core::LinAlg::Matrix<3, 3>& iFin, const Core::LinAlg::Matrix<3, 3>& Aorthgr);

   private:
    /// my material parameters
    Mixture::PAR::MixtureConstituentElastHyperElastinMembrane* params_;

    /// Current growth factor with respect to the reference configuration
    std::vector<double> current_reference_growth_;

    /// fraction of membrane elastin material to ensure equilibrium
    std::vector<double> mue_frac_;

    /// map to membrane materials/potential summands (only IsoNeoHooke is possible)
    std::vector<std::shared_ptr<Mat::Elastic::IsoNeoHooke>> potsum_membrane_;

    /// Anisotropy extension holding the structural tensor of the anisotropy
    ElastinMembraneAnisotropyExtension anisotropy_extension_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
