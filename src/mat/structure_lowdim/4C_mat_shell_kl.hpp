// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SHELL_KL_HPP
#define FOUR_C_MAT_SHELL_KL_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /**
     * @brief Material parameters for Kirchhoff-Love shell
     */
    class KirchhoffLoveShell : public Core::Mat::PAR::Parameter
    {
     public:
      /**
       * @brief Standard constructor
       */
      KirchhoffLoveShell(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      //! Young's modulus
      const double young_modulus_;

      //! Poisson's ratio
      const double poisson_ratio_;

      //! Thickness of the shell
      const double thickness_;

      //@}

      /**
       * @brief Create material instance of matching type with my parameters
       */
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };

  }  // namespace PAR

  class KirchhoffLoveShellType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "KirchhoffLoveShellType"; }

    static KirchhoffLoveShellType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static KirchhoffLoveShellType instance_;
  };

  /**
   * @brief Wrapper for shell material
   */
  class KirchhoffLoveShell : public Core::Mat::Material
  {
   public:
    /**
     * @brief Construct empty material object
     */
    KirchhoffLoveShell() = default;

    /**
     * @brief Construct the material object given material parameters
     */
    explicit KirchhoffLoveShell(Mat::PAR::KirchhoffLoveShell* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return KirchhoffLoveShellType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      UniqueParObjectId().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /**
     * @brief Return the material type
     */
    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::MaterialType::m_shell_kirchhoff_love;
    }

    /**
     * @brief Return copy of this material object
     */
    [[nodiscard]] std::shared_ptr<Material> clone() const override
    {
      return std::make_shared<KirchhoffLoveShell>(*this);
    }

    /**
     * @brief Return the Young's modulus of this material
     */
    [[nodiscard]] double young_modulus() const { return params_->young_modulus_; }

    /**
     * @brief Return the Poisson's ratio of this material
     */
    [[nodiscard]] double poisson_ratio() const { return params_->poisson_ratio_; }

    /**
     * @brief Return the thickness of this material
     */
    [[nodiscard]] double thickness() const { return params_->thickness_; }

    /**
     * @brief Return quick accessible material parameter data
     */
    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /**
     * @brief My material parameters
     */
    Mat::PAR::KirchhoffLoveShell* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
