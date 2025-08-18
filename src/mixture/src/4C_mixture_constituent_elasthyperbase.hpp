// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPERBASE_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_ELASTHYPERBASE_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_cylinder_cosy.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_prestress_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class Anisotropy;
}

namespace Mixture
{
  class MixtureConstituentElastHyperBase;

  namespace PAR
  {
    class MixtureConstituentElastHyperBase : public Mixture::PAR::MixtureConstituent
    {
     public:
      explicit MixtureConstituentElastHyperBase(const Core::Mat::PAR::Parameter::Data& matdata);

      [[nodiscard]] int get_prestressing_mat_id() const { return matid_prestress_strategy_; }

      /// @name material parameters
      /// @{

      /// Material id of the prestress strategy
      const int matid_prestress_strategy_;

      /// number of summands
      const int nummat_;

      /// List of material ids of the summands
      const std::vector<int> matids_;
      /// @}
    };
  }  // namespace PAR

  /*!
   * \brief Constituent for any hyperelastic material
   *
   * This constituent represents any hyperelastic material from the elasthyper toolbox. It has to
   * be paired with the Mat::Mixture material and a Mixture::MixtureRule.
   */
  class MixtureConstituentElastHyperBase : public Mixture::MixtureConstituent
  {
   public:
    /// Constructor for the material given the material parameters
    explicit MixtureConstituentElastHyperBase(
        Mixture::PAR::MixtureConstituentElastHyperBase* params, int id);

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
     * \brief Register all anisotropy extensions also for the sub-summands
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
     * \brief Returns a reference to all summands
     *
     * \return const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& Reference to the summands
     */
    [[nodiscard]] const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& summands() const
    {
      return potsum_;
    }

    /*!
     * \brief Returns a reference to all summand properties
     *
     * \return const Mat::SummandProperties& Reference to the summand properties
     */
    const Mat::SummandProperties& summand_properties() { return summand_properties_; }

    /*!
     * \brief Method that is called to setup the constituent once before the start of the simulation
     *
     * \param params Container for additional information
     * \param eleGID Global element id
     */
    void setup(const Teuchos::ParameterList& params, int eleGID) override;

    /*!
     * \brief Method that is called once for each Gauss point before the first evaluate call
     *
     * \param mixtureRule Reference to the mixture rule
     * \param params Container for additional information
     * \param gp Gauss point
     * \param eleGID Global element id
     */
    void pre_evaluate(MixtureRule& mixtureRule, const Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   protected:
    /*!
     * \brief Returns a reference to the prestretch tensor at the Gauss point
     *
     * \param gp Gauss point
     * \return const Core::LinAlg::Matrix<3, 3>& Reference to the prestretch tensor
     */
    [[nodiscard]] const Core::LinAlg::SymmetricTensor<double, 3, 3>& prestretch_tensor(
        const int gp) const
    {
      return prestretch_[gp];
    }

    /*!
     * \brief Returns a reference to the cylinder coordinate system
     *
     * \return const Mat::CylinderCoordinateSystemProvider&
     */
    [[nodiscard]] const Mat::CylinderCoordinateSystemAnisotropyExtension&
    cylinder_coordinate_system_anisotropy_extension() const
    {
      return cosy_anisotropy_extension_;
    }

    std::shared_ptr<Mixture::PrestressStrategy> prestress_strategy() { return prestress_strategy_; }

   private:
    /// @name Flags to specify the elastic formulations (initialize with false)
    //@{
    Mat::SummandProperties summand_properties_;  ///< holder for formulation specification
    //@}

    /// my material parameters
    Mixture::PAR::MixtureConstituentElastHyperBase* params_;

    /// map to materials/potential summands
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsum_;

    /// Prestretch of the constituent
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> prestretch_;

    /// AnisotropyExtension that handles the management of cylinder coordinate systems
    Mat::CylinderCoordinateSystemAnisotropyExtension cosy_anisotropy_extension_;

    /// Strategy for prestressing the constituent
    std::shared_ptr<Mixture::PrestressStrategy> prestress_strategy_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
