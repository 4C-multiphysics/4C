// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CROSSLINKERMAT_HPP
#define FOUR_C_MAT_CROSSLINKERMAT_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_crosslinking_submodel_evaluator.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for de St. Venant--Kirchhoff
    class CrosslinkerMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      CrosslinkerMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// number of material for underlying linker element
      int const link_element_matnum_;

      /// type of joint
      BeamInteraction::JointType jointtype_;

      /// distance between the two binding domains of a linker
      double const linkinglength_;

      /// tolerance for linker length in the sense: length +- tolerance
      double const linkinglengthtol_;

      /// preferred binding angle enclosed by two filaments' axes in radians
      double const linkingangle_;

      /// tolerance for preferred binding angle in radians in the sense of: angle +- tolerance
      double const linkingangletol_;

      /// chemical association-rate
      double const k_on_;

      /// chemical dissociation-rate
      double const k_off_;

      /// deltaD in Bell's equation for force dependent off rate
      double const deltabelleq_;

      /// distance to sphere elements in which binding is prohibited
      double const nobonddistsphere;

      /// type of crosslinker
      BeamInteraction::CrosslinkerType linkertype_;


      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class CrosslinkerMat

  }  // namespace PAR

  class CrosslinkerMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "CrosslinkerMatType"; }

    static CrosslinkerMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static CrosslinkerMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for Crosslinker Material
  class CrosslinkerMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    CrosslinkerMat();

    /// construct the material object given material parameters
    explicit CrosslinkerMat(Mat::PAR::CrosslinkerMat* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return CrosslinkerMatType::instance().unique_par_object_id();
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

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_crosslinkermat;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<CrosslinkerMat>(*this);
    }

    /// number of beam elasthyper material
    virtual double beam_elast_hyper_mat_num() const { return params_->link_element_matnum_; }

    /// force dependent off rate according to bells equation
    virtual BeamInteraction::JointType joint_type() const { return params_->jointtype_; };

    /// distance between the two binding domains of a linker
    virtual double linking_length() const { return params_->linkinglength_; }

    /// tolerance for linker length in the sense: length +- tolerance
    virtual double linking_length_tolerance() const { return params_->linkinglengthtol_; }

    /// preferred binding angle enclosed by two filaments' axes in radians
    virtual double linking_angle() const { return params_->linkingangle_; }

    /// tolerance for preferred binding angle in radians in the sense of: angle +- tolerance
    virtual double linking_angle_tolerance() const { return params_->linkingangletol_; }

    /// chemical association rate
    virtual double const& k_on() const { return params_->k_on_; }

    /// chemical dissociation rate
    virtual double k_off() const { return params_->k_off_; }

    /// force dependent off rate according to bells equation
    virtual double delta_bell_eq() const { return params_->deltabelleq_; };

    /// distance to sphere elements in which no double bonded beam to beam linker is allowed
    virtual double no_bond_dist_sphere() const { return params_->nobonddistsphere; };

    /// force dependent off rate according to bells equation
    virtual BeamInteraction::CrosslinkerType linker_type() const { return params_->linkertype_; };

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //@}

   private:
    /// my material parameters
    Mat::PAR::CrosslinkerMat* params_;
  };
}  // namespace Mat



FOUR_C_NAMESPACE_CLOSE

#endif
