// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MYOCARD_HPP
#define FOUR_C_MAT_MYOCARD_HPP

/*----------------------------------------------------------------------*
 |  headers                                                  cbert 08/13 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_myocard_general.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           cbert 08/13 |
 *----------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material
    class Myocard : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Myocard(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Diffusivity
      const double diff1;
      const double diff2;
      const double diff3;

      /// Perturbation for calculation of derivative
      const double dt_deriv;

      /// Model type. Possibilities are: "MV", "FHN", "INADA", "TNNP", "SAN"
      const std::string model;

      /// Tissue type. Possibilities are: "M", "ENDO", "EPI" for "TNNP"
      ///                                 "AN", "N", "NH" for "INADA"
      const std::string tissue;

      /// Time factor to correct for different Model specific time units
      const double time_scale;

      /// Number of Gauss Points for evaluating the material, i.e. the nonlinear reaction term
      int num_gp;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class myocard
  }  // namespace PAR

  class MyocardType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "MyocardType"; }
    static MyocardType& instance() { return instance_; };
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MyocardType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Myocard material
  ///
  /// This is a reaction-diffusion law of anisotropic, instationary electric conductivity in cardiac
  /// muscle tissue
  ///
  ///


  class Myocard : public Core::Mat::Material

  {
   public:
    /// construct empty material object
    Myocard();

    /// constructor with given material parameters
    Myocard(Mat::PAR::Myocard* params);

    /// @name Packing and Unpacking
    //@{

    /// Return unique ParObject id
    ///
    ///  every class implementing ParObject needs a unique id defined at the
    ///  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return MyocardType::instance().unique_par_object_id();
    }

    /// Pack this class so it can be communicated
    ///
    /// Resizes the vector data and stores all information of a class in it.
    /// The first information to be stored in data has to be the
    /// unique parobject id delivered by unique_par_object_id() which will then
    /// identify the exact class on the receiving processor.
    void pack(Core::Communication::PackBuffer& data)
        const override;  ///< (in/out): char vector to store class information

    /// \brief Unpack data from a char vector into this class
    ///
    /// The vector data contains all information to rebuild the
    /// exact copy of an instance of a class on a different processor.
    /// The first entry in data has to be an integer which is the unique
    /// parobject id defined at the top of this file and delivered by
    /// unique_par_object_id().
    ///
    void unpack(Core::Communication::UnpackBuffer& buffer)
        override;  ///< vector storing all data to be unpacked into this

    //@}

    /// Unpack Material for adaptive methods
    virtual void unpack_material(Core::Communication::UnpackBuffer& buffer);

    /// init material
    void set_gp(int gp) { params_->num_gp = gp; };

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_myocard;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Myocard>(*this);
    }

    /// material call from Discret::Elements::Transport::read_element function
    /// to setup conductivity tensor for each element
    void setup(const Core::LinAlg::Tensor<double, 3>& fiber1);
    void setup(const Core::LinAlg::Tensor<double, 2>& fiber1);
    void setup(const Core::IO::InputParameterContainer& container);

    void setup_diffusion_tensor(const std::vector<double>& fiber1);
    void setup_diffusion_tensor(const Core::LinAlg::Tensor<double, 3>& fiber1);
    void setup_diffusion_tensor(const Core::LinAlg::Tensor<double, 2>& fiber1);

    /// diffusivity
    void diffusivity(Core::LinAlg::Matrix<1, 1>& diffus3) const
    {
      diffusivity(diffus3, 0);
      return;
    };
    void diffusivity(Core::LinAlg::Matrix<2, 2>& diffus3) const
    {
      diffusivity(diffus3, 0);
      return;
    };
    void diffusivity(Core::LinAlg::Matrix<3, 3>& diffus3) const
    {
      diffusivity(diffus3, 0);
      return;
    };

    /// diffusivity
    void diffusivity(Core::LinAlg::Matrix<1, 1>& diffus3, int gp) const;
    void diffusivity(Core::LinAlg::Matrix<2, 2>& diffus3, int gp) const;
    void diffusivity(Core::LinAlg::Matrix<3, 3>& diffus3, int gp) const;

    bool diffusion_at_ele_center() const { return diff_at_ele_center_; };

    void reset_diffusion_tensor()
    {
      difftensor_.clear();
      return;
    };

    /// compute reaction coefficient
    double rea_coeff(const double phi, const double dt) const;

    /// compute reaction coefficient for multiple points per element
    double rea_coeff(const double phi, const double dt, int gp) const;

    /// compute reaction coefficient for multiple points per element at timestep n
    double rea_coeff_n(const double phi, const double dt, int gp) const;

    /// compute reaction coefficient derivative
    double rea_coeff_deriv(const double phi, const double dt) const;

    /// compute reaction coefficient derivative for multiple points per element
    double rea_coeff_deriv(const double phi, const double dt, int gp) const;

    /// compute Heaviside step function
    double gating_function(const double Gate1, const double Gate2, const double p, const double var,
        const double thresh) const;

    /// compute gating variable 'y' from dy/dt = (y_inf-y)/y_tau
    double gating_var_calc(
        const double dt, double y_0, const double y_inf, const double y_tau) const;

    ///  returns number of internal state variables of the material
    int get_number_of_internal_state_variables() const;

    ///  returns current internal state of the material
    double get_internal_state(const int k) const override;

    ///  returns current internal state of the material for multiple points per element
    double get_internal_state(const int k, int gp) const;

    ///  set internal state of the material
    void set_internal_state(const int k, const double val);

    ///  set internal state of the material for multiple points per element
    void set_internal_state(const int k, const double val, int gp);

    ///  return number of ionic currents
    int get_number_of_ionic_currents() const;

    ///  return ionic currents
    double get_ionic_currents(const int k) const;

    ///  return ionic currents for multiple points per element
    double get_ionic_currents(const int k, int gp) const;

    /// initialize internal variables (called by constructors)
    void initialize();

    /// resize internal state variables
    void resize_internal_state_variables();

    /// time update for this material
    void update(const double phi, const double dt);

    /// get number of Gauss points
    int get_number_of_gp() const;

    bool myocard_mat() const { return myocard_mat_ != nullptr; };

    /// @name Access material constants
    //@{

    //@}

    /// Return quick accessible material parameter data
    Mat::PAR::Myocard* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::Myocard* params_;

    /// conductivity tensor
    std::vector<Core::LinAlg::Matrix<3, 3>> difftensor_;

    /// number of internal state variables
    int nb_state_variables_;

    // Type of material model
    std::shared_ptr<MyocardGeneral> myocard_mat_;

    /// diffusion at element center
    bool diff_at_ele_center_;

  };  // Myocard
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
