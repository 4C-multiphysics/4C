// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MIXTURE_HPP
#define FOUR_C_MAT_MIXTURE_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_rule.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class Mixture;

  namespace PAR
  {
    class Mixture : public Core::Mat::PAR::Parameter
    {
      friend class Mat::Mixture;

     public:
      /// Standard constructor
      ///
      /// This constructor recursively calls the constructors of the
      /// parameter sets of the hyperelastic summands.
      explicit Mixture(const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name material parameters
      /// @{
      /// list of the references to the constituents
      std::vector<FourC::Mixture::PAR::MixtureConstituent*> constituents_;

      /// rule of the mixture (contains the physics)
      FourC::Mixture::PAR::MixtureRule* mixture_rule_;

      /// @}

      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR


  class MixtureType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "MixtureType"; }

    static MixtureType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MixtureType instance_;
  };

  /*!
   * \brief Material class holding a general mixture material
   *
   * This class has to be paired with a mixture rule (Mixture::MixtureRule) containing the physics
   * and constituents (Mixture::MixtureConstituent). This class manages the interplay between
   * the mixture rule and the constituents defining an clear interface if an extension of the
   * mixture framework is needed.
   */
  class Mixture : public So3Material
  {
   public:
    /// Constructor for an empty material object
    Mixture();

    /// Constructor for the material given the material parameters
    explicit Mixture(Mat::PAR::Mixture* params);


    /// @name Packing and Unpacking
    /// @{

    /// \brief Return unique ParObject id
    int unique_par_object_id() const override
    {
      return MixtureType::instance().unique_par_object_id();
    }

    /*!
     * \brief Pack this class so it can be communicated
     *
     * Resizes the vector data and stores all information of a class in it. The first information
     * to be stored in data has to be the unique parobject id delivered by unique_par_object_id()
     * which will then identify the exact class on the receiving processor.
     *
     * @param data (in/out): char vector to store class information
     */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
     * \brief Unpack data from a char vector into this class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of this file and delivered by UniqueParObjectId().
     *
     * @param data (in) : vector storing all data to be unpacked into this instance
     */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// @)

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
      {
        FOUR_C_THROW(
            "element and material kinematics are not compatible. Use Nonlinear total lagrangian"
            "kinematics (KINEM nonlinear) in your element definition.");
      }
    }

    /// Return material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_mixture;
    }

    /// Create a copy of this material
    /// \return copy of this material
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Mixture>(*this);
    }


    /*!
     * \brief Quick access to the material parameters
     *
     * @return Material parameters
     */
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /*!
     * \brief Setup of the material (Read the a container of data to create the element)
     *
     * This method will be called during reading of the elements. Here, we create all local tensors.
     *
     * @param numgp Number of Gauss-points
     * @param container Input parameter container of the element
     */
    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    /*!
     * \brief Post setup routine that will be called before the first Evaluate call
     *
     * @param params Container for additional information
     */
    void post_setup(const Teuchos::ParameterList& params, const int eleGID) override;

    /*!
     * \brief Update of the material
     *
     * This method will be called between each timestep.
     */
    void update() override;

    /*!
     * \brief Update of the material
     *
     * This method will be called between each timestep.
     *
     * @param defgrd Deformation gradient
     * @param gp Gauss point
     * @param params Container for additional information
     * @param eleGID Global element id
     */
    void update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int gp,
        const Teuchos::ParameterList& params, int eleGID) override;

    /// \brief This material law uses the extended update method
    bool uses_extended_update() override { return true; }

    /*!
     * \brief Evaluation of the material
     *
     * This method will compute the 2. Piola-Kirchhoff stress and the linearization of the material.
     *
     * @param defgrd (in) Deformation gradient
     * @param glstrain (in) Green-Lagrange strain in strain-like Voigt notation
     * @param params (in/out) Parameter list for additional information
     * @param stress (out) 2nd Piola-Kirchhoff stress tensor in stress-like Voigt notation
     * @param cmat (out) Linearization of the material law in Voigt notation
     * @param eleGID (in) Global element id
     */
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) final;

    /// \brief Return material mass density given by mixture rule
    [[nodiscard]] double density() const override { return mixture_rule_->return_mass_density(); };

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    /// Material parameters
    Mat::PAR::Mixture* params_;

    /// list of the references to the constituents
    std::shared_ptr<std::vector<std::unique_ptr<FourC::Mixture::MixtureConstituent>>> constituents_;

    /// Reference to the mixturerule
    std::shared_ptr<FourC::Mixture::MixtureRule> mixture_rule_;

    /// Flag whether constituents are already set up.
    bool setup_;

    /// Flag for each gauss point whether the mixture rule and constituents are pre-evaluated
    std::vector<bool> is_pre_evaluated_;

    /// Holder for anisotropic materials
    Anisotropy anisotropy_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
