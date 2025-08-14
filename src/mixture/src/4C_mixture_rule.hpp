// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_RULE_HPP
#define FOUR_C_MIXTURE_RULE_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_ENull.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication
namespace LinAlg
{
  class SerialDenseMatrix;
}

namespace Mat
{
  class Anisotropy;
  class Material;
  namespace PAR
  {
    class Material;
  }
}  // namespace Mat

namespace Mixture
{
  // forward declaration
  class MixtureRule;
  class MixtureConstituent;

  namespace PAR
  {
    class MixtureRule : public Core::Mat::PAR::Parameter
    {
      friend class Mixture::MixtureRule;

     public:
      /// constructor
      explicit MixtureRule(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Override this method and throw error, as only the CreateRule() should be used.
      std::shared_ptr<Core::Mat::Material> create_material() final
      {
        FOUR_C_THROW("Cannot create mixture rule from this method. Use CreateRule() instead.");
        return nullptr;
      }

      /// create material instance of matching type with my parameters
      virtual std::unique_ptr<Mixture::MixtureRule> create_rule() = 0;

      /*!
       * \brief Factory of the mixture rule parameters
       *
       * This static method generates the specific class of the mixture rule defined in the datfile
       * at the corresponding material id
       *
       * @param matid Material id of the mixturerule
       * @return Parameters of the referenced mixture rule
       */
      static Mixture::PAR::MixtureRule* factory(int matid);
    };
  }  // namespace PAR

  /*!
   * \brief Mixture rule containing the physics behind the mixture
   *
   * This class should be used within the Mixture framework as a mixture rule. This class
   * contains the whole physics. This is the base class defining the simplest possible physics,
   * i.e. constituents all deforming with the same deformation gradient and a homogenized stress
   * response using the mass density of each constituent.
   */
  class MixtureRule
  {
   public:
    /// Constructor for the material given the material parameters
    explicit MixtureRule(Mixture::PAR::MixtureRule* params);

    virtual ~MixtureRule() = default;

    virtual void pack_mixture_rule(Core::Communication::PackBuffer& data) const;

    /*!
     * \brief Unpack data from a char vector into this class to be called from a derived class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by unique_par_object_id().
     *
     * @param data (in) : vector storing all data to be unpacked into this instance.
     * @param position (in/out) : current position to unpack data
     */
    virtual void unpack_mixture_rule(Core::Communication::UnpackBuffer& buffer);

    /*!
     * This method should be called after creation of the constituents
     *
     * @param constituents (in) List of constituents
     */
    void set_constituents(
        std::shared_ptr<std::vector<std::unique_ptr<Mixture::MixtureConstituent>>> constituents)
    {
      constituents_ = std::move(constituents);
    }

    virtual void register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
    {
      // do nothing in the default case
    }

    /*!
     * Initialize the mixturerule with the element parameters of the input line
     *
     * @param numgp (in) Number of Gauss-points
     * @param params (in/out) : Parameter list for exchange of parameters
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
     * This method is called once for each element. The constituent should setup all its internal
     * variables and materials.
     *
     * @param params (in/out) : Container for additional information
     * @param eleGID (in) : global element id
     */
    virtual void setup(const Teuchos::ParameterList& params, int eleGID);

    /*!
     * \brief Update of the material law
     *
     * This simple mixture rule does not need to update anything, so this method is kept empty
     *
     * @param F (in) : Deformation gradient
     * @param params (in/out) : Container for additional information
     * @param gp (in) : Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void update(Core::LinAlg::Tensor<double, 3, 3> const& F,
        const Teuchos::ParameterList& params, const int gp, const int eleGID)
    {
      // Nothing needs to be updated in this simple mixture rule
    }

    /*!
     * \brief Update of the internal variables
     *
     * This method is called once per element between each time step to update the internal
     * variables. Not needed for simple mixture rules.
     */
    virtual void update() {}

    /*!
     *
     * @brief Method that is executed before the first evaluate call, once for each Gauss point
     *
     * @param params (in) : Container for additional information
     * @param gp (in) : Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void pre_evaluate(const Teuchos::ParameterList& params, const int gp, const int eleGID)
    {
      // do nothing in the default case
    }

    /*!
     * Evaluates the constituents. Needs to compute the stress contribution of the constituent out
     * of the displacements. Will be called for each Gauss point
     *
     * @param F (in) : Deformation gradient
     * @param E_strain (in) : Green-Lagrange strain tensor
     * @param params (in/out) : Container for additional parameters
     * @param S_stress (out) : 2nd Piola Kirchhoff stress tensor
     * @param cmat (out) : Constitutive tensor
     * @param gp (in) : Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E, const Teuchos::ParameterList& params,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) = 0;

    /*!
     * @brief Returns the material mass density
     *
     * @return material mass density
     */
    [[nodiscard]] virtual double return_mass_density() const
    {
      FOUR_C_THROW("Rule does not provide the evaluation of a material mass density.");
      return 0;
    }

    /*!
     * \brief Register names of the internal data that should be saved during runtime output
     *
     * \param name_and_size [out] : unordered map of names of the data with the respective vector
     * size
     */
    virtual void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const
    {
      // do nothing for simple mixture rules
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
     * \brief Returns a reference to the constituents
     * @return
     */
    [[nodiscard]] std::vector<std::unique_ptr<Mixture::MixtureConstituent>>& constituents() const
    {
      return *constituents_;
    }

    /*!
     * Get number of Gauss points used
     *
     * @return Number of Gauss points
     */
    [[nodiscard]] int num_gp() const { return numgp_; }

   private:
    ///! list of the references to the constituents
    std::shared_ptr<std::vector<std::unique_ptr<Mixture::MixtureConstituent>>> constituents_;

    ///! Number of Gauss points
    int numgp_;

    ///! Indicator, whether the constituent has already read the element definition
    bool has_read_element_;

    ///! Indicator, whether the constituent is already set up
    bool is_setup_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
