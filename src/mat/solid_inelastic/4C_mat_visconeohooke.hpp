// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCONEOHOOKE_HPP
#define FOUR_C_MAT_VISCONEOHOOKE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters
    class ViscoNeoHooke : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ViscoNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      const double youngs_slow_;
      const double poisson_;
      const double density_;
      const double youngs_fast_;
      const double relax_;
      const double theta_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ViscoNeoHooke

  }  // namespace PAR

  class ViscoNeoHookeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ViscoNeoHookeType"; }

    static ViscoNeoHookeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ViscoNeoHookeType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for Visco-NeoHooke material
  class ViscoNeoHooke : public So3Material
  {
   public:
    /// construct empty material object
    ViscoNeoHooke();

    /// construct the material object given material parameters
    explicit ViscoNeoHooke(Mat::PAR::ViscoNeoHooke* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ViscoNeoHookeType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.
      This material contains history variables, which are packed for restart purposes.

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
      History data is unpacked in restart.

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_visconeohooke;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ViscoNeoHooke>(*this);
    }

    /// Initialize internal stress variables
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Update internal stress variables
    void update() override;

    /// Evaluate material
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// Return density
    double density() const override { return params_->density_; }

    /// Return shear modulus
    double shear_mod() const { return 0.5 * params_->youngs_slow_ / (1.0 + params_->poisson_); };

    /// Check if history variables are already initialized
    bool initialized() const { return isinit_ && (histstresscurr_ != nullptr); }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::ViscoNeoHooke* params_;

    /// visco history stresses
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        histstresscurr_;  ///< current stress
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        histstresslast_;  ///< stress of last converged state
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        artstresscurr_;  ///< current artificial stress
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        artstresslast_;  ///< artificial stress in last converged state

    bool isinit_;  ///< indicates if #Initialized routine has been called
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
