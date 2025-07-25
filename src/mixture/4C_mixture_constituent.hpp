// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Mat
{
  class Anisotropy;
}

/*!
 * \brief The mixture namespace holds all mixture specific classes.
 *
 * The idea behind mixtures is that multiple materials share the same
 * deformation and the stress response is a mass fraction weighted
 * average of the stresses of each constituent.
 */
namespace Mixture
{
  // forward declaration
  class MixtureConstituent;
  class MixtureRule;

  namespace PAR
  {
    class MixtureConstituent : public Core::Mat::PAR::Parameter
    {
     public:
      explicit MixtureConstituent(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      virtual std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) = 0;

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() final;

      static Mixture::PAR::MixtureConstituent* factory(int matnum);
    };
  }  // namespace PAR

  /*!
   * \brief This is the base class of a constituent in a mixture defining the interface to the
   * holder class
   *
   * This abstract class defines the interface that a constituents needs to implement. It has to be
   * paired with Mat::Mixture and Mixture::MixtureRule.
   */
  class MixtureConstituent
  {
   public:
    MixtureConstituent(Mixture::PAR::MixtureConstituent* params, int id);

    virtual ~MixtureConstituent() = default;

    /// Returns the id of the constituent
    [[nodiscard]] int id() const { return id_; }

    /*!
     * \brief Pack data into a char vector from this class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by unique_par_object_id().
     *
     * @param data (in/put) : vector storing all data to be packed into this instance.
     */
    virtual void pack_constituent(Core::Communication::PackBuffer& data) const;

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
    virtual void unpack_constituent(Core::Communication::UnpackBuffer& buffer);

    /// material type
    [[nodiscard]] virtual Core::Materials::MaterialType material_type() const = 0;

    /*!
     * \brief Register anisotropy extensions of all sub-materials of the constituent
     *
     * \param anisotropy Reference to the global anisotropy manager
     */
    virtual void register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
    {
      // do nothing in the default case
    }

    /*!
     * Initialize the constituent with the parameters of the input line
     *
     * @param numgp (in) Number of Gauss-points
     * @param params (in/out) Parameter list for exchange of parameters
     */
    virtual void read_element(int numgp, const Core::IO::InputParameterContainer& container);

    /*!
     * Returns whether the constituent is already set up
     * @return true if the constituent is already set up, otherwise false
     */
    bool is_setup() { return is_setup_; }

    /*!
     * \brief Setup the constituent
     *
     * This method is called once for Gauss point at the beginning of the simulation.
     * The constituent should setup all internal variables and materials.
     *
     * @param params Container for additional information
     * @param eleGID Global element id
     */
    virtual void setup(const Teuchos::ParameterList& params, int eleGID);

    /*!
     * \brief Update of the internal variables
     *
     * This method is called once per Gauss point between each time step to update the internal
     * variables. (Not needed for simple constituents)
     *
     * @param defgrd Deformation gradient of the previous timestep
     * @param params Container for additional information
     * @param gp Gauss point
     * @param eleGID Global element identifier
     */
    virtual void update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd,
        const Teuchos::ParameterList& params, const int gp, const int eleGID)
    {
    }

    /*!
     * \brief Update of the internal variables
     *
     * This method is called once per element between each time step to update the internal
     * variables. Not needed for simple constituents.
     */
    virtual void update() {}

    /*!
     * @brief Update of the internal variables used for mixture rules evaluating the elastic part of
     * the deformation. This method must be explicitly called by the mixture rule and will not be
     * invoked automatically!
     *
     * @param F Deformation gradient of the previous timestep
     * @param iFext External deformation gradient of the previous timestep
     * @param params Container for additional information
     * @param dt Time increment from the previous timestep
     * @param gp Gauss-point
     * @param eleGID Global element id
     */
    virtual void update_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iFext, const Teuchos::ParameterList& params,
        const double dt, const int gp, const int eleGID)
    {
      // do nothing
    }

    /*!
     *
     * @brief Method that is executed before the first evaluate call, once for each Gauss point
     *
     * @param params (in) : Container for additional information
     * @param gp (in) : Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void pre_evaluate(
        MixtureRule& mixtureRule, const Teuchos::ParameterList& params, int gp, int eleGID)
    {
      // do nothing in the default case
    }

    /*!
     * \brief Returns the scalar indicating the growth scale from the reference configuration
     *
     * \return double
     */
    [[nodiscard]] virtual double get_growth_scalar(int gp) const { return 1.0; }


    /*!
     * @brief evaluates the derivative of the growth scalar w.r.t. Cauchy-Green deformation tensor
     *
     * @note This matrix is usually just a zero-matrix. It is non-zero if the growth scalar changes
     * with the deformation.
     *
     * @param gp (in) : Gauss point id
     * @param eleGID (in) : Global element id
     * @return Core::LinAlg::Matrix<1, 6> Derivative of the growth scalar w.r.t. Cauchy-Green
     * deformation tensor
     */
    [[nodiscard]] virtual Core::LinAlg::SymmetricTensor<double, 3, 3> get_d_growth_scalar_d_cg(
        int gp, int eleGID) const
    {
      return {};
    };

    /*!
     * @brief Evaluates the stress and material linearization of the constituents with an
     * inelastic part of the deformation
     *
     * The total deformation is #F, which is split into two parts:
     *
     * \f$\boldsymbol{F} = \boldsymbol{F}_e \cdot \boldsymbol{F}_in\f$
     *
     * Only elastic part \f$\boldsymbol{F}_e\f$ causes stresses. The inelastic part is only needed
     * for the linearization.
     *
     * @note S_stress and the linearization are specific quantities. They have to be multiplied with
     * the density of the constituent to obtain the real stress or linearization.
     *
     * @param F Total deformation gradient
     * @param iF_in inverse inelastic part of the deformation
     * @param params Container for additional information
     * @param S_stress 2nd specific Piola-Kirchhoff stress in stress-like Voigt notation
     * @param cmat specific linearization of the material tensor in Voigt notation
     * @param gp Gauss-point
     * @param eleGID Global element id
     */
    virtual void evaluate_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iF_in, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID);

    /*!
     * Evaluates the constituents. Needs to compute the stress contribution of the constituent out
     * of the displacements. Will be called for each Gauss point
     *
     * @note S_stress and the linearization are specific quantities. They have to be multiplied with
     * the density of the constituent to obtain the real stress or linearization.
     *
     * @param F (in) : Deformation gradient
     * @param E_strain (in) : Green-Lagrange strain in strain-like Voigt notation
     * @param params Container for additional information
     * @param S_stress (out) : 2nd specific Piola Kirchhoff stress tensor in stress like
     * Voigt-notation
     * @param cmat (out) : specific constitutive tensor in Voigt notation
     * @param gp (in) : Number of Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) = 0;

    /// Returns the reference mass density. Needs to be implemented by the deriving class.

    /*!
     * \brief Register names of the internal data that should be saved during runtime output
     *
     * \param name_and_size [out] : unordered map of names of the data with the respective vector
     * size
     */
    virtual void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const
    {
      // do nothing for simple constituents
    }

    /*!
     * \brief Evaluate internal data for every Gauss point saved for output during runtime output
     *
     * \param name [in] : Name of the data to export
     * \param data [out] : NUMGPxNUMDATA Matrix holding the data
     *
     * \return true if data is set by the material, otherwise false
     */
    virtual bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
    {
      return false;
    }



   protected:
    /*!
     * Get number of Gauss points used
     *
     * @return Number of Gauss points
     */
    int num_gp() const { return numgp_; }

   private:
    ///! Number of Gauss points
    int numgp_;

    ///! Indicator, whether the constituent has already read the element
    bool has_read_element_;

    ///! Indicator, whether the constituent is already set up
    bool is_setup_;

    ///! Id of the constituent
    const int id_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
