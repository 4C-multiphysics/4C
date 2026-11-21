// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUID_LINEAR_DENSITY_VISCOSITY_HPP
#define FOUR_C_MAT_FLUID_LINEAR_DENSITY_VISCOSITY_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for fluid with linear law (pressure-dependent) for
    /// the density and the viscosity
    ///
    /// This object exists only once for each read fluid.
    class LinearDensityViscosity : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      LinearDensityViscosity(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// reference density
      const double refdensity_;
      /// reference viscosity
      const double refviscosity_;
      /// reference pressure
      const double refpressure_;
      /// density-pressure coefficient
      const double coeffdensity_;
      /// viscosity-pressure coefficient
      const double coeffviscosity_;
      /// surface tension coefficient
      const double gamma_;

      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class LinearDensityViscosity

  }  // namespace PAR

  class LinearDensityViscosityType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "LinearDensityViscosityType"; }

    static LinearDensityViscosityType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static LinearDensityViscosityType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for fluid with linear law (pressure-dependent) for
  /// the density and the viscosity
  ///
  /// This object exists (several times) at every element
  class LinearDensityViscosity : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    LinearDensityViscosity();

    /// construct the material object given material parameters
    explicit LinearDensityViscosity(Mat::PAR::LinearDensityViscosity* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return LinearDensityViscosityType::instance().unique_par_object_id();
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
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluid_linear_density_viscosity;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<LinearDensityViscosity>(*this);
    }

    /// compute density
    double compute_density(const double press) const;

    /// compute viscosity
    double compute_viscosity(const double press) const;

    /// return material parameters for element calculation
    //@{

    /// return reference density
    double ref_density() const { return params_->refdensity_; }

    /// return reference viscosity
    double ref_viscosity() const { return params_->refviscosity_; }

    /// return reference pressure
    double ref_pressure() const { return params_->refpressure_; }

    /// return density-pressure coefficient
    double coeff_density() const { return params_->coeffdensity_; }

    /// return viscosity-pressure coefficient
    double coeff_viscosity() const { return params_->coeffviscosity_; }

    /// return surface tension coefficient
    double gamma() const { return params_->gamma_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::LinearDensityViscosity* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
