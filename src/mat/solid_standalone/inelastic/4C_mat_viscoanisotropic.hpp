// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOANISOTROPIC_HPP
#define FOUR_C_MAT_VISCOANISOTROPIC_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters
    class ViscoAnisotropic : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ViscoAnisotropic(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      const double kappa_;
      const double mue_;
      const double density_;
      const double k1_;
      const double k2_;
      const double gamma_;
      const int numstresstypes_;
      double beta_[2];
      double relax_[2];
      const double minstretch_;
      const int elethick_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ViscoAnisotropic

  }  // namespace PAR

  class ViscoAnisotropicType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ViscoAnisotropicType"; }

    static ViscoAnisotropicType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ViscoAnisotropicType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for Visco-NeoHooke material
  class ViscoAnisotropic : public So3Material
  {
   public:
    /// construct empty material object
    ViscoAnisotropic();

    /// construct the material object given material parameters
    explicit ViscoAnisotropic(Mat::PAR::ViscoAnisotropic* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ViscoAnisotropicType::instance().unique_par_object_id();
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
      return Core::Materials::m_viscoanisotropic;
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
      return std::make_shared<ViscoAnisotropic>(*this);
    }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Setup and Initialize internal stress variables and align fibers based on a given vector
    void setup(const int numgp,             ///< number of Gauss points
        const std::vector<double> thickvec  ///< direction fibers should be oriented in
    );

    /// Update internal stress variables
    void update() override;

    void update_fiber_dirs(const int numgp, Core::LinAlg::Matrix<3, 3>* defgrad);

    /// Evaluate material
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// Return density
    double density() const override { return params_->density_; };

    /// Return shear modulus
    double shear_mod() const { return params_->mue_; };


    /// Check if history variables are already initialized
    bool initialized() const { return isinit_ && (histstresscurr_ != nullptr); }

    /// return a1s
    std::shared_ptr<std::vector<std::vector<double>>> geta1() const { return ca1_; }

    /// return a2s
    std::shared_ptr<std::vector<std::vector<double>>> geta2() const { return ca2_; }


    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

   private:
    /// my material parameters
    Mat::PAR::ViscoAnisotropic* params_;

    // internal variables for fibers
    std::shared_ptr<std::vector<std::vector<double>>>
        a1_;  ///< first fiber vector per gp (reference)
    std::shared_ptr<std::vector<std::vector<double>>>
        a2_;  ///< second fiber vector per gp (reference)
    std::shared_ptr<std::vector<std::vector<double>>>
        ca1_;  ///< first fiber vector per gp (spatial config)
    std::shared_ptr<std::vector<std::vector<double>>>
        ca2_;  ///< second fiber vector per gp (spatial config)

    // visco history stresses for every gausspoint and every stress type
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        histstresscurr_;  ///< current stress
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        histstresslast_;  ///< stress of last converged state
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        artstresscurr_;  ///< current artificial stress
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        artstresslast_;  ///< artificial stress in last converged state

    bool isinit_;  ///< indicates if material is initialized
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
