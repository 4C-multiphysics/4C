// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MEMBRANE_ACTIVE_STRAIN_HPP
#define FOUR_C_MAT_MEMBRANE_ACTIVE_STRAIN_HPP

/*----------------------------------------------------------------------*
 | headers                                         brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_membrane_material_interfaces.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*
 | active strain membrane material                 brandstaeter 05/2018 |
 *----------------------------------------------------------------------*/
namespace Mat
{
  // forward declaration
  class MembraneActiveStrain;

  namespace PAR
  {
    class MembraneActiveStrain : public Core::Mat::PAR::Parameter
    {
      friend class Mat::MembraneActiveStrain;

     public:
      /// standard constructor
      MembraneActiveStrain(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Number of the material that describes the elastic behavior
      const int matid_passive_;

      /// Position of the transmembrane voltage of the SMC cell
      const int scalid_voltage_;

      /// density
      const double density_;

      /// beta (parameter dynamics of the VDCC)
      const double beta1_;

      /// beta (parameter dynamics of the Ca-ions)
      const double beta2_;

      /// voltage level for the activation of the Ca-influx
      const double voltage_threshold_;

      /// alpha (parameter for the intensity of the contraction in fiber 1 direction)
      const double alpha1_;

      /// alpha (parameter for the intensity of the contraction in fiber 2 direction)
      const double alpha2_;

      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
    // class MembraneActiveStrain

  }  // namespace PAR

  class MembraneActiveStrainType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "Membrane_ActiveStrainType"; }

    static MembraneActiveStrainType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MembraneActiveStrainType instance_;
  };
  // class Membrane_ActiveStrainType

  /*----------------------------------------------------------------------*/

  // forward declaration
  class Material;

  class MembraneActiveStrain : public So3Material, public Mat::MembraneMaterialLocalCoordinates
  {
   public:
    /// construct empty material object
    MembraneActiveStrain();

    /// construct the material object given material parameters
    explicit MembraneActiveStrain(Mat::PAR::MembraneActiveStrain* params);

    ///@name Packing and Unpacking
    //@{

    /// \brief Return unique ParObject id
    ///
    /// every class implementing ParObject needs a unique id defined at the
    /// top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return MembraneActiveStrainType::instance().unique_par_object_id();
    }

    /// \brief Pack this class so it can be communicated
    ///
    /// Resizes the vector data and stores all information of a class in it.
    /// The first information to be stored in data has to be the
    /// unique parobject id delivered by unique_par_object_id() which will then
    /// identify the exact class on the receiving processor.
    ///
    /// \param data (in/out): char vector to store class information
    void pack(Core::Communication::PackBuffer& data) const override;

    /// \brief Unpack data from a char vector into this class
    ///
    /// The vector data contains all information to rebuild the
    /// exact copy of an instance of a class on a different processor.
    /// The first entry in data has to be an integer which is the unique
    /// parobject id defined at the top of this file and delivered by
    /// unique_par_object_id().
    ///
    /// \param data (in) : vector storing all data to be unpacked into this
    ///                    instance.
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_membrane_activestrain;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MembraneActiveStrain>(*this);
    }

    /// material mass density
    double density() const override { return params_->density_; }

    /// setup
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Standard SO3 evaluate (not meant to be used)
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp,
        int eleGID) override  ///< Element ID
    {
      FOUR_C_THROW("This a membrane material. Calling So3 evaluate does not make sense.");
    };

    void update_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        const Core::LinAlg::Matrix<3, 3>& Q_trafo, int gp, int eleGID) override
    {
      // nothing to do
    }

    void evaluate_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Core::LinAlg::Matrix<3, 3>& cauchygreen, const Teuchos::ParameterList& params,
        const EvaluationContext<3>& context, const Core::LinAlg::Matrix<3, 3>& Q_trafo,
        Core::LinAlg::Matrix<3, 1>& stress, Core::LinAlg::Matrix<3, 3>& cmat, int gp,
        int eleGID) override;

    /// Update internal variables
    void update() override;

    /// Reset internal variables
    void reset_step() override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

   private:
    /// My material parameters
    Mat::PAR::MembraneActiveStrain* params_;

    /// passive material
    std::shared_ptr<Mat::So3Material> matpassive_;

    /// (tansmembrane) voltage at every gp
    std::shared_ptr<std::vector<double>> voltage_;

    /// activation parameter at every gp
    std::shared_ptr<std::vector<double>> activation_;

    /// indicates if material is initialized
    bool isinit_;

    // setup fiber vectors
    void setup_fiber_vectors(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system);

    // calculate normal direction from FIBER1 and FIBER2
    void setup_normal_direction();

    // pullback of the tangent from intermediate to reference configuration
    void pullback4th_tensor_voigt(const Core::LinAlg::Matrix<2, 2>& defgrd_active_inv_red,
        const Core::LinAlg::Matrix<3, 3>& cmat_passive_intermediate,
        Core::LinAlg::Matrix<3, 3>& cmat_reference);

    // transform voigt to tensor notation
    void tensor2x2_indices(int p, int* i, int* j);

    // transform tensor to voigt notation
    void voigt3_index(int i, int j, int* p);

   protected:
    /// vector of fiber vectors
    std::vector<Core::LinAlg::Matrix<3, 1>> fibervecs_;
  };  // class MembraneActiveStrain

}  // namespace Mat
// namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
