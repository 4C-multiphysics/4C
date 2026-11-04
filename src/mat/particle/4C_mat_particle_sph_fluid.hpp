// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PARTICLE_SPH_FLUID_HPP
#define FOUR_C_MAT_PARTICLE_SPH_FLUID_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_mat_particle_base.hpp"
#include "4C_mat_particle_thermo.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class definitions                                                         |
 *---------------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    class ParticleMaterialSPHFluid : public ParticleMaterialBase, public ParticleMaterialThermo
    {
     public:
      //! constructor
      ParticleMaterialSPHFluid(const Core::Mat::PAR::Parameter::Data& matdata);

      //! speed of sound
      double speed_of_sound() const { return std::sqrt(bulkModulus_ / initDensity_); };

      //! @name material parameters
      //@{

      //! reference density factor in equation of state
      const double refDensFac_;

      //! exponent in equation of state
      const double exponent_;

      //! background pressure
      const double backgroundPressure_;

      //! bulk modulus
      const double bulkModulus_;

      //! dynamic shear viscosity
      const double dynamicViscosity_;

      //! bulk viscosity
      const double bulkViscosity_;

      //! artificial viscosity
      const double artificialViscosity_;

      //@}

      //! create material instance of matching type with parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };

  }  // namespace PAR

  class ParticleMaterialSPHFluidType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ParticleMaterialSPHType"; };

    static ParticleMaterialSPHFluidType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ParticleMaterialSPHFluidType instance_;
  };

  class ParticleMaterialSPHFluid : public Core::Mat::Material
  {
   public:
    //! constructor (empty material object)
    ParticleMaterialSPHFluid();

    //! constructor (with given material parameters)
    explicit ParticleMaterialSPHFluid(Mat::PAR::ParticleMaterialSPHFluid* params);

    //! @name Packing and Unpacking

    //@{

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ParticleMaterialSPHFluidType::instance().unique_par_object_id();
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

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_particle_sph_fluid;
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ParticleMaterialSPHFluid>(*this);
    }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    //! my material parameters
    Mat::PAR::ParticleMaterialSPHFluid* params_;
  };

}  // namespace Mat

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
