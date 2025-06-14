// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ROBINSON_HPP
#define FOUR_C_MAT_ROBINSON_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_thermomechanical.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for visco-plastic Robinson's material
    class Robinson : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      Robinson(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! kind of Robinson material (slight
      //! differences:vague,butler,arya,arya_narloyz,arya_crmosteel)
      const std::string kind_;
      //! Young's modulus (temperature dependent --> polynomial expression)
      // 'E' [N/m^2]
      const std::vector<double> youngs_;
      //! Possion's ratio \f$ \nu \f$ [-]
      const double poissonratio_;
      //! mass density \f$ \rho [kg/m^3] \f$
      const double density_;
      //! linear coefficient of thermal expansion \f$ \alpha_T \f$ [1/K]
      const double thermexpans_;
      /// initial temperature (constant) \f$ \theta_0  \f$ [K]
      const double inittemp_;
      //! hardening factor 'A' (needed for flow law) [1/s]
      const double hrdn_fact_;
      //! hardening power 'n'  (exponent of F in the flow law) [-]
      const double hrdn_expo_;
      //! Bingam-Prager shear stress threshold \f$ \kappa^2 \f$
      //! 'K^2=K^2(K_0)' [N^2 / m^4]
      const std::vector<double> shrthrshld_;
      //! recovery factor 'R_0' [N/(s . m^2)]
      const double rcvry_;
      //! activation energy 'Q_0' for Arya_NARloy-Z [1/s]
      const double actv_ergy_;
      //! activation temperature 'T_0' [K]
      const double actv_tmpr_;
      //! 'G_0' (temperature independent, minimum value attainable by G )  [-]
      const double g0_;
      //! 'm'  [-]
      //! temperature independent, exponent in evolution law for back stress
      const double m_;
      //! '\f$\beta\f$' [-] (temperature independent)
      //! Arya_NarloyZ: \f$\beta = 0.533e-6 T^2 + 0.8\f$
      const std::vector<double> beta_;
      //! H
      //! Arya_NarloyZ: \f$H = 1.67e4 . (6.895)^(beta - 1) / (3 . K_0^2)\f$ [N^3/m^6]
      //! Arya_CrMoSteel: [N/m^2]
      const double h_;
      //! thermal material id, -1 if not used (old interface)
      const int thermomat_;

      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class Robinson

  }  // namespace PAR


  class RobinsonType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "RobinsonType"; }

    static RobinsonType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static RobinsonType instance_;

  };  // RobinsonType


  /*----------------------------------------------------------------------*/
  //! wrapper for visco-plastic Robinson's material
  class Robinson : public ThermoMechanicalMaterial
  {
   public:
    //! construct empty material object
    Robinson();

    //! construct the material object given material parameters
    explicit Robinson(Mat::PAR::Robinson* params);

    //! @name Packing and Unpacking

    //!  \brief return unique ParObject id
    //!
    //!  every class implementing ParObject needs a unique id defined at the
    //!  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return RobinsonType::instance().unique_par_object_id();
    }

    //!  \brief Pack this class so it can be communicated
    //!
    //!  Resizes the vector data and stores all information of a class in it.
    //!  The first information to be stored in data has to be the
    //!  unique parobject id delivered by unique_par_object_id() which will then
    //!  identify the exact class on the receiving processor.
    //!
    void pack(
        Core::Communication::PackBuffer& data  //!< (i/o): char vector to store class information
    ) const override;

    //!  \brief Unpack data from a char vector into this class
    //!
    //!  The vector data contains all information to rebuild the
    //!  exact copy of an instance of a class on a different processor.
    //!  The first entry in data has to be an integer which is the unique
    //!  parobject id defined at the top of this file and delivered by
    //!  unique_par_object_id().
    //!
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_vp_robinson;
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
      return std::make_shared<Robinson>(*this);
    }

    //! initialise internal stress variables
    void setup(const int numgp,  //!< number of Gauss points
        const Core::IO::InputParameterContainer& container) override;

    //! update internal stress variables
    void update() override;

    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    //! computes Cauchy stress
    void stress(const double p,                                   //!< volumetric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress            //!< 2nd PK-stress
    ) const;

    //! computes relative stress eta = stress - back stress
    void rel_dev_stress(
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& devstress,  //!< deviatoric stress tensor
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&,            //!< back stress tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& eta               //!< relative stress
    ) const;

    //! computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(double temp,                                  //!< current temperature
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat  //!< material tangent
    ) const;

    //! \brief calculate visco-plastic strain rate governed by the evolution law
    void calc_be_viscous_strain_rate(const double dt,  //!< (i) time step size
        double tempnp,                                 //!< (i) current temperature
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_p,  //!< (i) viscous strain \f$\varepsilon^v_n\f$ at t_n
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_on,  //!< (i) viscous strain \f$\varepsilon^v_{n+1}\f$ at t_n at t_{n+1}^<i>
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            devstress,  //!< (i) stress deviator \f$s_n\f$ at t_{n+1}^<i>
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            eta,  //!< (i) over stress/relative stress \f$\eta_{n+1}\f$ at t_{n+1}^<i>
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_pres,  //!< (o) viscous strain residual \f$f_{res}^v\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kve,  //!< (o) \f$\dfrac{\partial f_{res}^v}{\partial \Delta\varepsilon}\f$
                  //!< tangent of viscous strain residual with respect to total strain inc eps
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kvv,  //!< (o) \f$\dfrac{\partial f_{res}^v}{\partial \Delta\varepsilon^v}\f$
                  //!<  tangent of viscous strain residual with respec to viscous strains iinc eps^v
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kva  //!< (o) \f$\dfrac{\partial f_{res}^v}{\partial \Delta\alpha}\f$
                 //!< tangent of viscous strain residual with respect to back stresses iinc al
    ) const;

    //! \brief residual of BE-discretised back stress according to the flow rule
    //!        at Gauss point
    void calc_be_back_stress_flow(const double dt,  //!< (i) time step size
        const double tempnp,                        //!< (i) current temperature at t_{n+1}
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_p,  //!< (i) viscous strain \f$\varepsilon_{n}\f$ at t_n^i
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_on,  //!< (i) viscous strain \f$\varepsilon_{n+1}\f$ at t_{n+1}^i
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            devstress,  //!< (i) deviatoric stress \f$s_{n+1}\f$ at t_{n+1}^i
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            backstress,  //!<  (i)back stress \f$\alpha_{n}\f$  at t_{n}^i,
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            backstress_n,  //!< (i) back stress \f$\alpha_{n+1}\f$ at t_{n+1}^i,
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& backstress_res,  //!< (o) back stress residual
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kae,  //!< (o) \f$\dfrac{\partial f_{res}^{al}}{\partial \Delta\varepsilon}\f$
                  //!< tangent of back stress residual with respect to total strain inc eps
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kav,  //!< (o) \f$\dfrac{\partial f_{res}^{al}}{\partial \Delta\varepsilon^v}\f$
                  //!< tangent of back stress residual with respect to viscous strains iinc eps^v
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kaa  //!< (o) \f$\dfrac{\partial f_{res}^{al}}{\partial \Delta\alpha}\f$
                 //!< tangent of back stress residual with respect to back stresses iinc al
    ) const;

    //! Reduce (statically condense) system in eps,eps^v,al to purely eps
    /*!
    The linearised stress and internal residuals are

          [ sig   ]         [ sig    ]^i
      Lin [ res^v ]       = [ res^v  ]
          [ res^al]_{n+1}   [ res^al ]_{n+1}

                               [ kee  kev  kea ]^i  [ iinc eps   ]^i
                            +  [ kve  kvv  kva ]    [ iinc eps^v ]
                               [ kae  kav  kaa ]    [ iinc al    ]_{n+1}

                            [ sig ]
                          = [  0  ]  on every element (e)
                            [  0  ]  and at each Gauss point gp

    with - total strain increment/residual strains  iinc eps   -->  straininc
         - viscous strain increment                 iinc eps^v -->  strain_on
         - back stress increment                    iinc al    -->  backstress
         - material tangent                         kee        -->  cmat

         - kee = dsigma / d eps = cmat,  kev = dsigma / d eps^v, kea = dsigma / d alpha
         - kve = dres^v / d eps, kvv = dres^v / d eps^v, kva = dres^v / d alpha,
         - kae = dres^al / d eps, kav = dres^al / d eps^v, kaa = dres^al / d alpha,

    Due to the fact that the internal residuals (the BE-discretised evolution
    laws of the viscous strain and the back stress) are C^{-1}-continuous
    across element boundaries. We can statically condense this system.
    The iterative increments inc eps^v and inc al are expressed in inc eps.
    We achieve

      [ iinc eps^v ]   [ kvv  kva ]^{-1} (   [ res^v  ]   [ kve ]                )
      [            ] = [          ]      ( - [        ] - [     ] . [ iinc eps ] )
      [ iinc al    ]   [ kav  kaa ]      (   [ res^al ]   [ kae ]                )

    thus

                                         [ kvv  kva ]^{-1} [ res^v  ]^i
      sig_red^i = sig^i - [ kev  kea ]^i [          ]      [        ]
                                         [ kav  kaa ]      [ res^al ]

    and
                                         [ kvv  kva ]^{-1} [ kve ]^i
      kee_red^i = kee^i - [ kev  kea ]^i [          ]      [     ]
                                         [ kav  kaa ]      [ kae ]

      ==> condensed system:

      Lin sig = kee_red^i . iinc eps + sig_red^i

    */
    void calculate_condensed_system(
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>& stress,  //!< (6x1) (io) stress vector \f$\sigma\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            cmat,  //!< cmat == kee (6x6) (io) material stiffness matrix, constitutive tensor
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kev,  //!< (6x6) (i) \f$\dfrac{\partial \sigma}{\partial \varepsilon^v}\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kea,  //!< (6x6) (i) \f$\dfrac{\partial \sigma}{\partial \alpha}\f$
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            strain_pres,  //!< (6x1) (i) viscous strain residual
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kve,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{v}}{\partial \varepsilon}\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kvv,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{v}}{\partial \varepsilon^v}\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kva,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{v}}{\partial \alpha}\f$
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>&
            backstress_res,  //!< (6x1) (i) backstress residual
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kae,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{\alpha}}{\partial \varepsilon}\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kav,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{\alpha}}{\partial \varepsilon^v}\f$
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>&
            kaa,  //!< (6x6) (i) \f$\dfrac{\partial f_{res}^{\alpha}}{\partial \alpha}\f$
        Core::LinAlg::Matrix<(2 * NUM_STRESS_3D), 1>&
            kvarva,  //!< (12x1) (o) condensed matrix of residual
        Core::LinAlg::Matrix<(2 * NUM_STRESS_3D), NUM_STRESS_3D>&
            kvakvae  //!< (12x6) (o) condensed matrix of tangent
    ) const;

    //! \brief iterative update of material internal variables
    //!
    //! material internal variables (viscous strain and back stress) are updated by
    //! their iterative increments.
    //! Their iterative increments are expressed in terms of the iterative increment
    //! of the total strain.
    //! Here the reduction matrices (kvarvam,kvakvae) stored at previous call of
    //! calculate_condensed_system() care used.
    //!
    //! strainplcurr_ = strainpllast_ + Delta strain_p (o)
    //! backstresscurr_ = backstresslast_ + Delta backstress (o)
    void iterative_update_of_internal_variables(const int numgp,  //!< total number of Gauss points
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> straininc    //!< (i) increment of total strain
    );

    //! return density
    double density() const override { return params_->density_; }

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

    //! calculate temperature dependent material parameter and return value
    double get_mat_parameter_at_tempnp(
        const std::vector<double>* paramvector,  //!< (i) given parameter is a vector
        const double& tempnp                     //!< (i) current temperature
    ) const;

    //! calculate temperature dependent material parameter
    double get_mat_parameter_at_tempnp(
        const double paramconst,  //!< (i) given parameter is a constant
        const double& tempnp      //!< (i) current temperature
    ) const;

    //! Initial temperature \f$ \theta_0 \f$
    double init_temp() const { return params_->inittemp_; }

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
    Mat::PAR::Robinson* params_;

    //! indicator if #Initialize routine has been called
    bool isinit_;

    //! pointer to the internal thermal material
    std::shared_ptr<Mat::Trait::Thermo> thermo_;

    //! current temperature (set by Reinit())
    double current_temperature_{};

    //! robinson's material requires the following internal variables:
    //! - visco-plastic strain vector (at t_n, t_n+1^i)
    //! - back stress vector (at t_n, t_n+1^i)
    //! - scaled residual --> for condensation of the system
    //! - scaled tangent --> for condensation of the system
    //!
    //! visco-plastic strain vector Ev^{gp} at t_{n} for every Gauss point gp
    //!    Ev^{gp,T} = [ E_11  E_22  E_33  2*E_12  2*E_23  2*E_31 ]^{gp} */
    //!< \f${\varepsilon}^p_{n}\f$
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>> strainpllast_;
    //! current visco-plastic strain vector Ev^{gp} at t_{n+1} for every Gauss point gp
    //!    Ev^{gp,T} = [ E_11  E_22  E_33  2*E_12  2*E_23  2*E_31 ]^{gp} */
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        strainplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old back stress vector Alpha^{gp} at t_n for every Gauss point gp
    //!    Alpha^{gp,T} = [ A_11  A_22  A_33  A_12  A_23  A_31 ]^{gp}
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        backstresslast_;  //!< \f${\alpha}_{n}\f$
    //! current back stress vector Alpha^{gp} at t_{n+1} for every Gauss point gp
    //!< \f${\alpha}_{n+1}\f$
    //!    Alpha^{gp,T} = [ A_11  A_22  A_33  A_12  A_23  A_31 ]^{gp} */
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        backstresscurr_;  //!< \f${\alpha}_{n+1}\f$
    //! update vector for MIV iterative increments
    //!          [ kvv  kva ]^{-1}   [ res^v  ]
    //! kvarva = [          ]      . [        ]
    //!          [ kav  kaa ]      . [ res^al ]
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<(2 * NUM_STRESS_3D), 1>>> kvarva_;
    //! update matrix for MIV iterative increments
    //!              [ kvv  kva ]^{-1}   [ kve ]
    //!    kvakvae = [          ]      . [     ]
    //!              [ kav  kaa ]      . [ kae ]
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<(2 * NUM_STRESS_3D), NUM_STRESS_3D>>> kvakvae_;
    //! strain at last evaluation
    std::vector<Core::LinAlg::Matrix<6, 1>> strain_last_;

  };  // class Robinson : public Core::Mat::Material
}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
