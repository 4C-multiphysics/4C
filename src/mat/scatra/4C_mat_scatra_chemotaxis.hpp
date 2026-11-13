// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_CHEMOTAXIS_HPP
#define FOUR_C_MAT_SCATRA_CHEMOTAXIS_HPP

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
    class ScatraChemotaxisMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ScatraChemotaxisMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// number of scalars in this pair
      const int numscal_;

      /// the list of material IDs
      const std::vector<int> pair_;

      /// reaction coefficient
      const double chemocoeff_;

    };  // class Scatra

  }  // namespace PAR

  class ScatraChemotaxisMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraChemotaxisMatType"; }

    static ScatraChemotaxisMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraChemotaxisMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material
  class ScatraChemotaxisMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ScatraChemotaxisMat();

    /// construct the material object given material parameters
    explicit ScatraChemotaxisMat(Mat::PAR::ScatraChemotaxisMat* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ScatraChemotaxisMatType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra_chemotaxis;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraChemotaxisMat>(*this);
    }

    /// return number of scalars for this reaction
    int num_scal() const { return params_->numscal_; }

    /// return chemotaxis coefficient
    double chemo_coeff() const { return params_->chemocoeff_; }

    /// return pairing
    const std::vector<int>* pair() const { return &params_->pair_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::ScatraChemotaxisMat* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
