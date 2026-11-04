// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUID_WEAKLY_COMPRESSIBLE_HPP
#define FOUR_C_MAT_FLUID_WEAKLY_COMPRESSIBLE_HPP



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
    /// material parameters for weakly compressible fluid
    ///
    /// This object exists only once for each read fluid.
    class WeaklyCompressibleFluid : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      WeaklyCompressibleFluid(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// viscosity
      const double viscosity_;
      /// reference density
      const double refdensity_;
      /// reference pressure
      const double refpressure_;
      /// compressibility coefficient
      const double comprcoeff_;

      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class WeaklyCompressibleFluid

  }  // namespace PAR

  class WeaklyCompressibleFluidType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "WeaklyCompressibleFluidType"; }

    static WeaklyCompressibleFluidType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static WeaklyCompressibleFluidType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for weakly compressible fluid
  ///
  /// This object exists (several times) at every element
  class WeaklyCompressibleFluid : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    WeaklyCompressibleFluid();

    /// construct the material object given material parameters
    explicit WeaklyCompressibleFluid(Mat::PAR::WeaklyCompressibleFluid* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return WeaklyCompressibleFluidType::instance().unique_par_object_id();
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
      return Core::Materials::m_fluid_weakly_compressible;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<WeaklyCompressibleFluid>(*this);
    }

    /// compute density
    double compute_density(const double press) const;

    /// compute pressure
    double compute_pressure(const double dens) const;

    /// return material parameters for element calculation
    //@{

    /// return viscosity
    double viscosity() const { return params_->viscosity_; }

    /// return reference density
    double ref_density() const { return params_->refdensity_; }

    /// return reference pressure
    double ref_pressure() const { return params_->refpressure_; }

    /// return compressibility coefficient
    double compr_coeff() const { return params_->comprcoeff_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::WeaklyCompressibleFluid* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
