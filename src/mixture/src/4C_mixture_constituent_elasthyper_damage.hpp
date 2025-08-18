// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPER_DAMAGE_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPER_DAMAGE_HPP

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
  class MixtureConstituentElastHyperDamage;

  namespace PAR
  {
    class MixtureConstituentElastHyperDamage : public Mixture::PAR::MixtureConstituentElastHyperBase
    {
     public:
      /*!
       * \brief Construct a new elastin material with a membrane
       *
       * \param matdata Material parameters
       * \param ref_mass_fraction reference mass fraction
       */
      explicit MixtureConstituentElastHyperDamage(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) override;

      /// @name material parameters
      /// @{
      const int damage_function_id_;
      /// @}
    };
  }  // namespace PAR

  /*!
   * \brief Constituent for any hyperelastic material
   *
   * This constituent represents any hyperelastic material from the elasthyper toolbox. It has to
   * be paired with the Mat::Mixture material and a Mixture::MixtureRule.
   */
  class MixtureConstituentElastHyperDamage : public Mixture::MixtureConstituentElastHyperBase
  {
   public:
    /*!
     * \brief Constructor for the material given the material parameters
     *
     * \param params Material parameters
     */
    explicit MixtureConstituentElastHyperDamage(
        Mixture::PAR::MixtureConstituentElastHyperDamage* params, int id);

    /// Returns the material type enum
    Core::Materials::MaterialType material_type() const override;

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
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
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

   private:
    /// my material parameters
    Mixture::PAR::MixtureConstituentElastHyperDamage* params_;

    /// Current growth factor with respect to the reference configuration
    std::vector<double> current_reference_growth_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
