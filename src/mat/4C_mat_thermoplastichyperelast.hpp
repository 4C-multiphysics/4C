// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_THERMOPLASTICHYPERELAST_HPP
#define FOUR_C_MAT_THERMOPLASTICHYPERELAST_HPP

/*----------------------------------------------------------------------*
 | headers                                                   dano 03/13 |
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
    //! material parameters for neo-Hooke
    class ThermoPlasticHyperElast : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      ThermoPlasticHyperElast(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! coefficient of thermal expansion
      const double cte_;
      //! initial, reference temperature
      const double inittemp_;
      //! initial yield stress (constant) at reference temperature
      const double yield_;
      //! linear isotropic hardening modulus at reference temperature
      const double isohard_;
      // saturation hardening at reference temperature
      const double sathardening_;
      //! hardening exponent
      const double hardexpo_;
      //! yield stress softening
      const double yieldsoft_;
      //! hardening softening
      const double hardsoft_;
      //! tolerance for local Newton iteration
      const double abstol_;
      //! thermal material id, -1 if not used (old interface)
      const int thermomat_;

      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ThermoPlasticHyperElast

  }  // namespace PAR


  class ThermoPlasticHyperElastType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ThermoPlasticHyperElastType"; }

    static ThermoPlasticHyperElastType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ThermoPlasticHyperElastType instance_;

  };  // class ThermoPlasticHyperElastType

  /*----------------------------------------------------------------------*/
  //! wrapper for finite strain elasto-plastic material
  class ThermoPlasticHyperElast : public ThermoMechanicalMaterial
  {
   public:
    //! construct empty material object
    ThermoPlasticHyperElast();

    //! construct the material object given material parameters
    explicit ThermoPlasticHyperElast(Mat::PAR::ThermoPlasticHyperElast* params);

    //! @name Packing and Unpacking

    /*!
    \brief Return unique ParObject id

    every class implementing ParObject needs a unique id defined at the
    top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ThermoPlasticHyperElastType::instance().unique_par_object_id();
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

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_thermoplhyperelast;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ThermoPlasticHyperElast>(*this);
    }

    //! Young's modulus
    double youngs() const { return params_->youngs_; }

    //! Poisson's ratio
    double poisson_ratio() const { return params_->poissonratio_; }

    //! density
    double density() const override { return params_->density_; }

    //! shear modulus
    double shear_mod() const { return 0.5 * (params_->youngs_) / (1.0 + params_->poissonratio_); }

    //! yield stress
    virtual double yield() const { return params_->yield_; }

    //! isotropic hardening modulus
    virtual double iso_hard() const { return params_->isohard_; }

    //! flow stress softening
    virtual double flow_stress_soft() const { return params_->yieldsoft_; }

    //! hardening softening
    virtual double hard_soft() const { return params_->hardsoft_; }

    //! saturation hardening
    virtual double sat_hardening() const { return params_->sathardening_; }

    //! coefficient of thermal expansion
    virtual double cte() const { return params_->cte_; }

    //! coefficient of thermal expansion
    virtual double hard_expo() const { return params_->hardexpo_; }

    //! initial, reference temperature
    virtual double init_temp() const { return params_->inittemp_; }

    void reinit(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, double temperature, unsigned gp) override;

    void stress_temperature_modulus_and_deriv(
        Core::LinAlg::Matrix<6, 1>& stm, Core::LinAlg::Matrix<6, 1>& stm_dT, int gp) override;

    Core::LinAlg::Matrix<6, 1> evaluate_d_stress_d_scalar(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<6, 1>& glstrain, Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return accumulated strain at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    double accumulated_strain(int gp) const { return (accplstrainlast_->at(gp)); }

    //! mechanical dissipation
    double mech_diss(int gp) const { return (mechdiss_->at(gp)); }

    //! linearisation of Dmech w.r.t. temperatures T_{n+1}
    //! contribution to K_TT
    double mech_diss_k_tt(int gp) const { return (mechdiss_k_tt_->at(gp)); }

    //! linearisation of the mechanical dissipation w.r.t. displacements d_{n+1}
    //! contribution to K_Td
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> mech_diss_k_td(int gp) const
    {
      return (mechdiss_k_td_->at(gp));
    }

    //! thermoplastic heating
    double thermo_plast_heating(int gp) const { return (thrplheat_->at(gp)); }

    //! linearisation of thermoplastic heating w.r.t. temperatures T_{n+1}
    //! contribution to K_TT
    double thermo_plast_heating_k_tt(int gp) const { return (thrplheat_k_tt_->at(gp)); }

    //! linearisation of thermoplastic heating w.r.t. temperatures T_{n+1}
    //! contribution to K_Td
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> thermo_plast_heating_k_td(int gp) const
    {
      return (thrplheat_k_td_->at(gp));
    }

    //! linearisation of material tangent w.r.t. temperatures T_{n+1}
    //! contribution to K_dT
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> c_mat_kd_t(int gp) const { return (cmat_kd_t_->at(gp)); }

    //! check if history variables are already initialised
    bool initialized() const { return (isinit_ and (defgrdcurr_ != nullptr)); }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //@}

    //! @name Evaluation methods

    //! initialise internal stress variables
    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material law
    void evaluate(const Core::LinAlg::Matrix<3, 3>*
                      defgrd,  //!< input deformation gradient for multiplicative sp
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>*
            glstrain,                    //!< input Green-Lagrange strain (redundant with defo
                                         //   but used for neo-hooke evaluation; maybe remove
        Teuchos::ParameterList& params,  //!< input parameter list (e.g. Young's, ...)
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>*
            stress,  //!< output (mandatory) second Piola-Kirchhoff stress
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>*
            cmat,  //!< output (mandatory) material stiffness matrix
        int gp,    ///< Gauss point
        int eleGID) override;

    //! evaluate the elasto-plastic tangent
    void setup_cmat_elasto_plastic(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
                                       cmat,  //!< elasto-plastic tangent modulus (out)
        double Dgamma,                        //!< plastic multiplier
        double Hiso_temp,           //!< temperature-dependent linear isotropic hardening modulus
        double sigma_y0infty_temp,  //!< temperature-dependent saturation hardening stress
        double sigma_y0_temp,       //!< temperature-dependent flow/yield stress
        double mubar,               //!< deformation-dependent shear modulus
        double q_trial,             //!< trial von Mises equivalent stress
        const Core::LinAlg::Matrix<3, 3>& defgrd,  //!< F_{n+1}
        Core::LinAlg::Matrix<3, 3> invdefgrdcurr,  //!< inverse of F_{n+1}
        Core::LinAlg::Matrix<3, 3> n,              //!< spatial flow vector
        double bulk,                               //!< bulk modulus
        int gp                                     //!< current Gauss-point
    ) const;

    //! calculate updated value of bebar_{n+1}
    void calculate_current_bebar(const Core::LinAlg::Matrix<3, 3>& devtau,  //!< s_{n+1}
        double G,                                                           //!< shear modulus
        const Core::LinAlg::Matrix<3, 3>& id2,  //!< second-order identity
        int gp                                  //!< current Gauss-point
    );

    //! main 3D material call to determine stress and constitutive tensor ctemp
    //  originally method of fourier with const!!!
    void evaluate(const Core::LinAlg::Matrix<1, 1>& Ntemp,  //!< temperature of element
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& ctemp,  //!< temperature-dependent material tangent
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            cmat_T,                                          //!< temperature-dependent mechanical
                                                             //!< material tangent
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stresstemp,  //!< temperature-dependent stress term
        Teuchos::ParameterList& params                       //!< parameter list
    );

    //! computes temperature-dependent isotropic thermal elasticity tensor in
    //! matrix notion for 3d
    void setup_cthermo(
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& ctemp,  //!< temperature dependent material tangent
        const double J,                                 //!< determinant of deformation gradient
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            Cinv_vct  //!< inverse of Cauchy-Green tensor (vector)
    ) const;

    //! computes temperature-dependent isotropic mechanical elasticity tensor in
    //! matrix notion for 3d
    void setup_cmat_thermo(const double temperature, Core::LinAlg::Matrix<6, 6>& cmat_T,
        const Core::LinAlg::Matrix<3, 3>& defgrd) const;

    //! calculates stress-temperature modulus
    double st_modulus() const;

    //! finite difference check of material tangent
    void fd_check(Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  // updated stress sigma_n+1
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            cmat,  // material tangent calculated with FD of stresses
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            cmatFD,                           // material tangent calculated with FD of stresses
        const double temperature,             // scalar-valued current temperature
        const Teuchos::ParameterList& params  // parameter list including F,C^{-1},...
    ) const;

    /// Return whether the material requires the deformation gradient for its evaluation
    bool needs_defgrd() const override { return true; };

    //@}

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
    Mat::PAR::ThermoPlasticHyperElast* params_;

    //! pointer to the internal thermal material
    std::shared_ptr<Mat::Trait::Thermo> thermo_;

    //! current temperature (set by Reinit())
    double current_temperature_{};

    //! @name Internal / history variables

    //! plastic history variables
    //! old (i.e. at t_n)  deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> defgrdlast_;
    //! current (i.e. at t_n+1) deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> defgrdcurr_;

    //! old (i.e. at t_n) elastic, isochoric right Cauchy-Green tensor
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> bebarlast_;
    //! current (i.e. at t_n+1) elastic, isochoric right Cauchy-Green tensor
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> bebarcurr_;

    //! old (i.e. at t_n) accumulated plastic strain
    std::shared_ptr<std::vector<double>> accplstrainlast_;
    //! current (i.e. at t_n+1) accumulated plastic strain
    std::shared_ptr<std::vector<double>> accplstraincurr_;

    //@}

    //! @name Linearisation terms for thermal equation

    //! current (i.e. at t_n+1) mechanical dissipation
    std::shared_ptr<std::vector<double>> mechdiss_;
    //! current (i.e. at t_n+1) linearised mechanical dissipation w.r.t. T_{n+1}
    std::shared_ptr<std::vector<double>> mechdiss_k_tt_;
    //! current (i.e. at t_n+1) linearised mechanical dissipation w.r.t. d_{n+1}
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>> mechdiss_k_td_;
    //! current (i.e. at t_n+1) thermoplastic heating term
    std::shared_ptr<std::vector<double>> thrplheat_;
    //! current (i.e. at t_n+1) thermoplastic heating term
    std::shared_ptr<std::vector<double>> thrplheat_k_tt_;
    //! current (i.e. at t_n+1) thermoplastic heating term w.r.t. d_{n+1}
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>> thrplheat_k_td_;

    //@}

    //! @name Linearisation terms for structural equation

    //! current (i.e. at t_n+1) linearised material tangent w.r.t. T_{n+1}
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>> cmat_kd_t_;

    //@}

    //! indicator if #Initialize routine has been called
    bool isinit_;
    //! indicator if material has started to be plastic
    bool plastic_step_;
    //! element ID, in which first plasticity occurs
    int plastic_ele_id_;

  };  // class ThermoPlasticHyperElast
}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
