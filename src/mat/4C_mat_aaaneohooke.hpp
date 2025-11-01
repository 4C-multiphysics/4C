// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_AAANEOHOOKE_HPP
#define FOUR_C_MAT_AAANEOHOOKE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for aneurysm wall material
    class AAAneohooke : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      AAAneohooke(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      // !brief enum for mapping between material parameter and entry in the matparams_ vector
      enum Matparamnames
      {
        young,
        nue,
        beta,
        density,
        first = young,
        last = density
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
    };  // class AAAneohooke

  }  // namespace PAR

  class AAAneohookeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "AAAneohookeType"; }

    static AAAneohookeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static AAAneohookeType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// aneurysm wall material according to Raghavan and Vorp [2000]
  class AAAneohooke : public So3Material
  {
   public:
    // empty constructor
    AAAneohooke();

    // constructor with given material parameters
    AAAneohooke(Mat::PAR::AAAneohooke* params);

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return AAAneohookeType::instance().unique_par_object_id();
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

    /// material mass density
    // virtual double Density() const { return params_->GetDensity(); }
    double density() const override { return params_->get_parameter(params_->density, -1); }

    /// shear modulus
    // double shear_mod() const { return 0.5*params_->GetYoungs(-1)/(1.0+params_->GetNue()); }
    double shear_mod() const
    {
      return 0.5 * params_->get_parameter(params_->young, -1) /
             (1.0 + params_->get_parameter(params_->nue, -1));
    }

    // material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_aaaneohooke;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<AAAneohooke>(*this);
    }

    // THE material routine
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// evaluate strain energy function
    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const EvaluationContext& context, int gp, int eleGID) const override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleGID) const override;

   private:
    /// my material parameters
    Mat::PAR::AAAneohooke* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
