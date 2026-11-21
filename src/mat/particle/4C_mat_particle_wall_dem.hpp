// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PARTICLE_WALL_DEM_HPP
#define FOUR_C_MAT_PARTICLE_WALL_DEM_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class definitions                                                         |
 *---------------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    class ParticleWallMaterialDEM : public Core::Mat::PAR::Parameter
    {
     public:
      //! constructor
      ParticleWallMaterialDEM(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! @name material parameters
      //@{

      //! tangential contact friction coefficient
      const double frictionTang_;

      //! rolling contact friction coefficient
      const double frictionRoll_;

      //! adhesion surface energy
      const double adhesionSurfaceEnergy_;

      //@}
    };

  }  // namespace PAR

  class ParticleWallMaterialDEMType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ParticleWallMaterialDEMType"; };

    static ParticleWallMaterialDEMType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ParticleWallMaterialDEMType instance_;
  };

  class ParticleWallMaterialDEM : public Core::Mat::Material
  {
   public:
    //! constructor (empty material object)
    ParticleWallMaterialDEM();

    //! constructor (with given material parameters)
    explicit ParticleWallMaterialDEM(Mat::PAR::ParticleWallMaterialDEM* params);

    //! @name Packing and Unpacking

    //@{

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ParticleWallMaterialDEMType::instance().unique_par_object_id();
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

    //! @name Access methods

    //@{

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_particle_wall_dem;
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ParticleWallMaterialDEM>(*this);
    }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return tangential contact friction coefficient
    double mu_tangential() const { return params_->frictionTang_; }

    //! return rolling contact friction coefficient
    double mu_rolling() const { return params_->frictionRoll_; }

    //! return adhesion surface energy
    double adhesion_surface_energy() const { return params_->adhesionSurfaceEnergy_; }

    //@}

   private:
    //! my material parameters
    Mat::PAR::ParticleWallMaterialDEM* params_;
  };

}  // namespace Mat

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
