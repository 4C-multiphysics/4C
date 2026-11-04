// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_HPP
#define FOUR_C_MAT_SCATRA_HPP



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
    /// parameters for scalar transport material
    class ScatraMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ScatraMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      enum Matparamnames
      {
        diff,
        reac,
        densific,
        /*!< True if the scalar reacts to the external force.
         * Example: the scalar is magnetic and reacts to the magnetic field. Another scalar could be
         * not magnetic and not react to the magnetic field. */
        reacts_to_external_force,
        first = diff,
        last = reacts_to_external_force
      };

      double get_parameter(int parametername, const int EleId)
      {
        // check if we have an element based value via size
        if (matparams_[parametername]->global_length() == 1)
        {
          // we have a global value hence we directly return the first entry
          return matparams_[parametername]->local_values_as_span()[0];
        }
        // If someone calls this functions without a valid EleID and we have element based values
        // throw error
        else if (EleId < 0 && matparams_[parametername]->global_length() > 1)
        {
          FOUR_C_THROW("Global mat parameter requested but we have elementwise mat params");
          return 0.0;
        }
        // otherwise just return the element specific value
        else
        {
          // calculate LID here, instead of before each call
          return matparams_[parametername]
              ->local_values_as_span()[matparams_[parametername]->get_map().lid(EleId)];
        }
      }

     private:
      std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> matparams_;

    };  // class Scatra

  }  // namespace PAR

  class ScatraMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraMatType"; }

    static ScatraMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material
  class ScatraMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ScatraMat();

    /// construct the material object given material parameters
    explicit ScatraMat(Mat::PAR::ScatraMat* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ScatraMatType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMat>(*this);
    }

    /// diffusivity
    double diffusivity(int eleid = -1) const
    {
      return params_->get_parameter(params_->diff, eleid);
    }

    /// reaction coefficient
    double rea_coeff(int eleid = -1) const { return params_->get_parameter(params_->reac, eleid); }

    /// densification coefficient
    double densification(int eleid = -1) const
    {
      return params_->get_parameter(params_->densific, eleid);
    }

    /// reacts to external force
    [[nodiscard]] double reacts_to_external_force() const
    {
      return params_->get_parameter(params_->reacts_to_external_force, -1);
    }


    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::ScatraMat* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
