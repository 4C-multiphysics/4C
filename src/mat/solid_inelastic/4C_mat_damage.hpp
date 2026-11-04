// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_DAMAGE_HPP
#define FOUR_C_MAT_DAMAGE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for small strain elasto-plastic material with
    //! isotropic ductile damage
    class Damage : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      Damage(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! isotropic hardening curve
      const std::vector<double> sigma_y_;
      //! accumulated plastic strain
      const std::vector<double> strainbar_p_ref_;
      //! damage evolution law denominator r
      const double damden_;
      //! damage evolution law exponent s
      const double damexp_;
      //! damage threshold
      const double epsbarD_;
      //! stress-like kinematic hardening modulus
      const double kinhard_;
      //! scalar-valued kinematic hardening modulus
      const double kinhard_rec_;
      // saturation hardening at reference temperature
      const double sathardening_;
      //! hardening exponent
      const double hardexpo_;
      //! tolerance for local Newton iteration
      const double abstol_;

      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class Damage

  }  // namespace PAR


  class DamageType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "DamageType"; }

    static DamageType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static DamageType instance_;

  };  // DamageType


  /*----------------------------------------------------------------------*/
  //! wrapper for elasto-plastic material including damage
  class Damage : public So3Material
  {
   public:
    //! construct empty material object
    Damage();

    //! construct the material object given material parameters
    explicit Damage(Mat::PAR::Damage* params);

    //! @name Packing and Unpacking

    //!  \brief Return unique ParObject id
    //!
    //!  every class implementing ParObject needs a unique id defined at the
    //!  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return DamageType::instance().unique_par_object_id();
    }

    //!  \brief Pack this class so it can be communicated
    //!
    //!  Resizes the vector data and stores all information of a class in it.
    //!  The first information to be stored in data has to be the
    //!  unique parobject id delivered by unique_par_object_id() which will then
    //!  identify the exact class on the receiving processor.
    //!
    //!  \param data (in/out): char vector to store class information
    void pack(Core::Communication::PackBuffer&
            data  //!<  data (i/o): char vector to store class information
    ) const override;

    //!  \brief Unpack data from a char vector into this class
    //!
    //!  The vector data contains all information to rebuild the
    //!  exact copy of an instance of a class on a different processor.
    //!  The first entry in data has to be an integer which is the unique
    //!  parobject id defined at the top of this file and delivered by
    //!  unique_par_object_id().
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_elpldamage;
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Damage>(*this);
    }

    //! check if element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::linear))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! initialise internal stress variables
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    //! evaluate material using Lemaitre material model
    virtual void evaluate_full_lemaitre(
        const Core::LinAlg::Matrix<3, 3>* defgrd,                 //!< deformation gradient
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* linstrain,  //!< linear total strains
        const Teuchos::ParameterList& params,            //!< parameter list for communication
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,  //!< 2nd PK-stress
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  //!< material stiffness matrix
        int gp,                                                    //!< Gauss point
        int eleGID);                                               //!< element GID

    //! evaluate material using the simplified Lemaitre model
    virtual void evaluate_simplified_lemaitre(
        const Core::LinAlg::Matrix<3, 3>* defgrd,                 //!< deformation gradient
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* linstrain,  //!< linear total strains
        const Teuchos::ParameterList& params,            //!< parameter list for communication
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,  //!< 2nd PK-stress
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  //!< material stiffness matrix
        int gp,                                                    //!< Gauss point
        int eleGID);                                               //!< element GID

    //! return derivative of piecewise linear function for the
    //! yield stress, i.e. isotropic hardening modulus at current
    //! accumulated plastic strain
    double get_iso_hard_at_strainbarnp(Mat::PAR::Damage* matparameter, const double strainbar_p);


    //! compute current yield stress sigma_y(astrain^p)
    //! calculate yield stress from (sigma_y-astrain^p)-samples
    double get_sigma_y_at_strainbarnp(Mat::PAR::Damage* matparameter, const double strainbar_p);



    //! This is the residual and tangent calculation without consideration of damage
    std::pair<double, double> residuum_and_jacobian_no_damage(
        Mat::PAR::Damage* matparameter, double Dgamma, double accplstrain_last, double q_tilde);


    //! This is the residual and tangent calculation including damage
    std::pair<double, double> residuum_and_jacobian_with_damage(Mat::PAR::Damage* matparameter,
        double Dgamma, double isohardvarlast, double q_tilde, double p_tilde, double omegaold);


    //! computes stress
    void stress(const double p,                                   //!< volumetric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress            //!< 2nd PK-stress
    );

    //! calculate relative/over stress
    void rel_stress(
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& beta,       //!< back stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& eta               //!< relative stress
    );

    //! computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat  //!< elastic material tangent
    );

    //! computes isotropic elastoplastic tensor in matrix notion for 3d
    void setup_cmat_elasto_plastic(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                       cmat,  //!< elasto-plastic material tangent
        int eleID,
        double Dgamma,         //!< plastic multiplier
        double G,              //!< shear modulus
        double bulk_modulus,   //!< bulk modulus
        double p_tilde,        // undamaged pressure
        double q_tilde,        // undamaged trial von Mises equivalent stress
        double energyrelrate,  // damage energy release rate
        double Ytan,           // derivative of engergy release rate Ytan w.r.t. Dgamma
        double sigma_y, double Hiso, Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar,
        int gp,                 //!< current Gauss point
        bool damevolution,      //!< flag indicating if damage threshold is passed, i.e. if damage
                                //!< evolves
        bool active_plasticity  //!< flag indicating active plasticity
    );

    //! computes isotropic elastoplastic damage tensor in matrix nation for 3d
    void setup_cmat_elasto_plastic_full_lemaitre(
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            cmat,                                        //!< elasto-plastic tangent modulus (out)
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> N_tilde,  //!< flow vector
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  //!< total stress
        bool active_plasticity,                          //!< flag indicating active plasticity
        double Dgamma,                                   //!< plastic multiplier
        double s_N,  //!< s_N = 2 * G * (Dgamma / omega_{n+1})^2 * s / r * (Y_{n+1} / r)^(s-1) * (1
                     //!< + nu) / E * s^tilde_{n+1} : N^tilde_{n+1}
        double
            g,  //!< g = (2 * G * (Dgamma / omega_{n+1}) + a * Dgamma / (1 + b * Dgamma)) / q_tilde
        double h_alg,         //!< h_alg = lengthy expression defined in Newton Raphson section
        double G,             //!< shear modulus
        double dkappa_dR,     //!< derivative of the hardening curve
        double bulk_modulus,  //!< bulk modulus
        double Hkin,          //!< kinematic hardening variable 1 (kinematic hardening modulus)
        double Hkin_rec,      //!< kinematic hardening variable 2 (saturation effect)
        double Nbetaold,      //!< Nbetaold = (N_tilde : beta_n)
        int gp,               // current Gauss point
        double qbar_tilde,    // effective relative eta^{~}
        double y,             // (-Y / r)^s
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dy_dsigma_tilde,  // dy/dsigma_tilde
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>
            b_NbetaoldN  // beta_n - 2/3 . (N_tilde : beta_n) . N_tilde
    );

    //! return density
    [[nodiscard]] double density() const override { return params_->density_; }

    //! return accumulated strain at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    [[nodiscard]] double accumulated_strain(int gp) const { return strainbarpllast_.at(gp); }

    //! return damaged isotropic hardening variable R at Gauss points
    //! in undamaged case: corresponding to the accumulated strain strainbar_p
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    [[nodiscard]] double isotropic_hardening_variable(int gp) const
    {
      return isohardvarlast_.at(gp);
    }

    //! return scalar-valued isotropic damage variable at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    [[nodiscard]] double isotropic_damage(int gp) const { return damagelast_.at(gp); }

    //! return boolean flag for complete failure at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    [[nodiscard]] double failure_flag(int gp) const { return failedlast_.at(gp); }

    //! check if history variables are already initialised
    [[nodiscard]] bool initialized() const { return isinit_; }

    //! return quick accessible material parameter data
    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //! return names of visualization data available for direct VTK output
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    //! return visualization data for direct VTK output
    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    //! my material parameters
    Mat::PAR::Damage* params_;

    //! plastic history vector
    //! old plastic strain at t_n
    std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>
        strainpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current plastic strain at t_n+1
    std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>
        strainplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old back stress at t_n
    std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>> backstresslast_;  //!< \f${\beta}_{n}\f$
    //! current back stress at t_n+1
    std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>> backstresscurr_;  //!< \f${\beta}_{n+1}\f$
    //! old accumulated plastic strain at t_n
    std::vector<double> strainbarpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current accumulated plastic strain at t_n+1
    std::vector<double> strainbarplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old damaged accumulated plastic strain corresponding to isotropic hardening at t_n
    std::vector<double> isohardvarlast_;  //!< \f${R}_{n+1}\f$
    //! current damaged accumulated plastic strain corresponding to isotropic hardening at t_n+1
    std::vector<double> isohardvarcurr_;  //!< \f${R}_{n}\f$
    //! old damage internal state variable D at t_n
    std::vector<double> damagelast_;  //!< \f${D}_{n}\f$
    //! current damage internal state variable D at t_n+1
    std::vector<double> damagecurr_;  //!< \f${D}_{n+1}\f$
    //! flag whether the integration point has failed at t_n
    std::vector<bool> failedlast_;
    //! flag whether the integration point has failed at t_n
    std::vector<bool> failedcurr_;

    //! indicator if #Initialize routine has been called
    bool isinit_;
    //! indicator if material has started to be plastic
    bool plastic_step_;

  };  // class Damage : public Core::Mat::Material
}  // namespace Mat


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
