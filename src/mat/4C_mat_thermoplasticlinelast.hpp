// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_THERMOPLASTICLINELAST_HPP
#define FOUR_C_MAT_THERMOPLASTICLINELAST_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 08/11 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_thermomechanical.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for plastic lin elastic Material
    class ThermoPlasticLinElast : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      ThermoPlasticLinElast(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! linear coefficient of thermal expansion \f$ \alpha_T \f$
      const double thermexpans_;
      //! initial temperature (constant) \f$ \theta_0 \f$
      const double thetainit_;
      //! yield stress (constant for perfect plasticity)  \f$ \sigma_y \f$
      const double yield_;
      //! isotropic hardening modulus \f$ H_i \f$
      const double isohard_;
      //! kinematic hardening modulus \f$ H_k \f$
      const double kinhard_;
      //! isotropic hardening curve
      const std::vector<double> sigma_y_;
      //! accumulated plastic strain
      const std::vector<double> strainbar_p_ref_;
      //! tolerance for local Newton iteration
      const double abstol_;
      //! thermal material id, -1 if not used (old interface)
      const int thermomat_;

      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ThermoThermoPlasticLinElast

  }  // namespace PAR


  class ThermoPlasticLinElastType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ThermoPlasticLinElastType"; }

    static ThermoPlasticLinElastType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ThermoPlasticLinElastType instance_;

  };  // ThermoThermoPlasticLinElastType


  /*----------------------------------------------------------------------*/
  //! wrapper for linear thermo-elasto-plastic material
  class ThermoPlasticLinElast : public ThermoMechanicalMaterial
  {
   public:
    //! construct empty material object
    ThermoPlasticLinElast();

    //! construct the material object given material parameters
    explicit ThermoPlasticLinElast(Mat::PAR::ThermoPlasticLinElast* params);

    //! @name Packing and Unpacking

    //!  \brief return unique ParObject id
    //!
    //!  every class implementing ParObject needs a unique id defined at the
    //!  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return ThermoPlasticLinElastType::instance().unique_par_object_id();
    }

    //!  \brief Pack this class so it can be communicated
    //!
    //!  Resizes the vector data and stores all information of a class in it.
    //!  The first information to be stored in data has to be the
    //!  unique parobject id delivered by unique_par_object_id() which will then
    //!  identify the exact class on the receiving processor.
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
      return Core::Materials::m_thermopllinelast;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::linear))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ThermoPlasticLinElast>(*this);
    }

    //! initialise internal stress variables
    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    //! update internal stress variables
    void update() override;

    //! reset internal stress variables
    void reset();

    //! evaluate material
    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,
        Teuchos::ParameterList& params,                  //!< parameter list for communication
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,  //!< 2nd PK-stress
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  //!< material stiffness matrix
        int gp,                                                    ///< Gauss point
        int eleGID) override;

    // computes stress
    void stress(const double p,                                   //!< volumetric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress            //!< 2nd PK-stress
    ) const;

    //! calculate relative/over stress
    void rel_dev_stress(
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& beta,       //!< back stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& eta               //!< relative stress
    ) const;

    //! computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat  //!< elastic material tangent
    ) const;

    //! computes isotropic elastoplastic tensor in matrix notion for 3d
    void setup_cmat_elasto_plastic(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                       cmat,  //!< elasto-plastic material tangent
        double Dgamma,                        //!< plastic multiplier
        double G,                             //!< shear modulus
        double q,                             //!< relative effective stress
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> flowvector,  //!< flow vector
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Nbar,        //!< unit vector
        double heaviside,                                         //!< Heaviside function
        double Hiso,                                              //!< isotropic hardening modulus
        double Hkin                                               //!< kinematic hardening modulus
    ) const;

    //! computes isotropic elastoplastic tensor in matrix notation for 3d
    void setup_cmat_elasto_plastic2(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat,
        double Dgamma, double q, Core::LinAlg::Matrix<NUM_STRESS_3D, 1> unitflow) const;

    //! computes continuum elastoplastic tensor in matrix notation for 3d
    void setup_continuum_cmat_elasto_plastic(
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat, double Dgamma, double q,
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> unitflow) const;

    //! calculates the derivative of get_sigma_y_at_strainbarnp() w.r.t. astrain_{n+1}
    //! and returns the isotropic hardening modulus
    double get_iso_hard_at_strainbarnp(
        const double strainbar_p  //!< current accumulated strain
                                  //!< or if damage!=0: isotropic hardening internal variable
    ) const;

    //! calculates current yield stress from (sigma_y-astrain)-samples which
    //! describe a piecewise constant function
    double get_sigma_y_at_strainbarnp(
        const double strainbar_p  //!< current accumulated strain, in case of dependent hardening
                                  //!< or if damage!=0: isotropic hardening internal variable
    ) const;

    //! return density
    double density() const override { return params_->density_; }

    //! return scalar-valued accumulated strain at Gauss points
    //! use method to return values at END of time step
    double accumulated_strain(int gp  //!< current Gauss point
    ) const
    {
      //! use the old vector (last_) for postprocessing
      //! Output is called after(!!) Update, so that newest values are included in
      //! the old history vectors last_
      //! in contrast: the current history vectors curr_are reset to zero
      return (strainbarpllast_->at(gp));
    }

    // return current scalar-valued accumulated strain
    double accumulated_strain_curr(int gp  //!< current Gauss point
    ) const
    {
      return (strainbarplcurr_->at(gp));
    }

    //! check if history variables are already initialised
    bool initialized() const { return (isinit_ and (strainplcurr_ != nullptr)); }

    void reinit(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, double temperature, unsigned gp) override;

    void stress_temperature_modulus_and_deriv(
        Core::LinAlg::Matrix<6, 1>& stm, Core::LinAlg::Matrix<6, 1>& stm_dT, int gp) override;

    Core::LinAlg::Matrix<6, 1> evaluate_d_stress_d_scalar(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<6, 1>& glstrain, Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! @name temperature specific methods
    //@{

    //! main 3D material call to determine stress and constitutive tensor ctemp
    //  originally method of fourier with const!!!
    void evaluate(const Core::LinAlg::Matrix<1, 1>& Ntemp,  //!< temperature of element
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& ctemp,  //!< temperature dependent material tangent
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stresstemp  //!< temperature dependent stress term
    );

    //! computes temperature dependent isotropic elasticity tensor in matrix
    //! notion for 3d
    void setup_cthermo(
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& ctemp  //!< temperature dependent material tangent
    ) const;

    //! calculates stress-temperature modulus
    double st_modulus() const;

    //! initial temperature
    double init_temp() const { return params_->thetainit_; }

    //@}

    //! @name specific methods for TSI and plastic material
    //@{

    //! calculate elastic strain rate, using additive split of strains
    //! (o) save strain^e' in strainelrate_
    void strain_rate_split(int gp,  //!< (i): current Gauss point
        const double stepsize,      //!< (i): stepsize
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strainrate  //!< (i): total strain rate ( B d')
    );

    //! return current plastic strain vector
    //! \f${\boldsymbol \varepsilon}^p_{n+1}\f$
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> plastic_strain(int gp) const
    {
      return strainplcurr_->at(gp);
    }

    //! return current elastic strain rate vector
    //! \f$\dot{\boldsymbol \varepsilon}^e_{n+1}\f$
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> elastic_strain_rate(int gp) const
    {
      return strainelrate_->at(gp);
    }

    //! compute internal dissipation terms
    void dissipation(int gp,  // current Gauss point
        double sigma_yiso,    // isotropic work hardening von Mises stress
        double Dgamma,        // plastic multiplier/increment
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& N,      // flow vector
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress  // total mechanical stress
    );

    //! compute linearisation of internal dissipation for k_Td
    void dissipation_coupl_cond(const Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                    cmat,                 // elasto-plastic tangent modulus (out)
        int gp,                                           // current Gauss point
        double G,                                         // shear modulus
        double Hiso,                                      // isotropic hardening modulus
        double Hkin,                                      // kinematic hardening modulus
        double heaviside,                                 // Heaviside function
        double etanorm,                                   // norm of eta^{trial}_{n+1}
        double Dgamma,                                    // plastic multiplier
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& N,  // flow vector
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress  // flow vector
    );

    //! return mechanical dissipation term due to kinematic hardening
    //! \f$ D_{\rm mech,kin}\f$
    double mechanical_kinematic_dissipation(int gp) const { return dmech_->at(gp); }

    //! return linearisation of mechanical dissipation w.r.t. displacements
    //! \f$ k_Td += \dfrac{\partial D_{\rm mech}}{\partial d_{n+1}}\f$
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dissipation_linearised_for_coupl_cond(int gp) const
    {
      return dmech_d_->at(gp);
    }

    //@}

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //! @name thermo material interface

    void evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp, Core::LinAlg::Matrix<3, 3>& cmat,
        Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const override;

    void evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp, Core::LinAlg::Matrix<2, 2>& cmat,
        Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const override;

    void evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp, Core::LinAlg::Matrix<1, 1>& cmat,
        Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const override;

    void conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const override;

    void conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const override;

    void conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const override;

    double capacity() const override;

    double capacity_deriv_t() const override;

    void reinit(double temperature, unsigned gp) override;

    void reset_current_state() override;

    void commit_current_state() override;

    //@}

   private:
    //! my material parameters
    Mat::PAR::ThermoPlasticLinElast* params_;

    //! pointer to the internal thermal material
    std::shared_ptr<Mat::Trait::Thermo> thermo_;

    //! current temperature (set by Reinit())
    double current_temperature_{};

    //! plastic history vector
    //! old plastic strain at t_n
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        strainpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current plastic strain at t_n+1
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        strainplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old back stress at t_n
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        backstresslast_;  //!< \f${\beta}_{n}\f$
    //! current back stress at t_n+1
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        backstresscurr_;  //!< \f${\beta}_{n+1}\f$
    //! old accumulated plastic strain at t_n
    std::shared_ptr<std::vector<double>> strainbarpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current accumulated plastic strain at t_n+1
    std::shared_ptr<std::vector<double>> strainbarplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$

    //! @name specific methods for the combination TSI and plastic material
    //@{

    //! mechanical dissipation term due to kinematic and isotropic hardening at t_n+1
    std::shared_ptr<std::vector<double>> dmech_;
    //! linearisation of mechanical dissipation term w.r.t. to d_{n+1}
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>> dmech_d_;
    //! save plastic strain increment: Incstrainpl = strain^p_{n+1} - strain^p_n
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>> incstrainpl_;
    //! elastic strain rate required for thermoelastic heating term
    //! use additive split: strain^e' = strain' - strain^p'
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>> strainelrate_;

    //@}

    //! indicator if #Initialize routine has been called
    bool isinit_;
    //! indicator if material has started to be plastic
    bool plastic_step_;
    //! element ID, in which first plasticity occurs
    int plastic_ele_id_;

  };  // class ThermoThermoPlasticLinElast : public Core::Mat::Material
}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
