// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_THERMOSTVENANTKIRCHHOFF_HPP
#define FOUR_C_MAT_THERMOSTVENANTKIRCHHOFF_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 02/10 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_trait_thermo_solid.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for de St. Venant--Kirchhoff with temperature
    //! dependent term
    //!
    //! <h3>Input line</h3>
    //! MAT 1 MAT_Struct_ThermoStVenantK YOUNG 400 NUE 0.3 DENS 1 THEXPANS 1 INITTEMP 20
    class ThermoStVenantKirchhoff : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      explicit ThermoStVenantKirchhoff(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus (temperature dependent --> polynomial expression)
      const std::vector<double> youngs_;
      //! Possion's ratio \f$ \nu \f$
      const double poissonratio_;
      //! mass density \f$ \rho \f$
      const double density_;
      //! linear coefficient of thermal expansion  \f$ \alpha_T \f$
      const double thermexpans_;
      //! initial temperature (constant) \f$ \theta_0 \f$
      const double thetainit_;
      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ThermoStVenantKirchhoff

  }  // namespace PAR

  class ThermoStVenantKirchhoffType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ThermoStVenantKirchhoffType"; }

    static ThermoStVenantKirchhoffType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ThermoStVenantKirchhoffType instance_;
  };

  /*----------------------------------------------------------------------*/
  //! Wrapper for St.-Venant-Kirchhoff material with temperature term
  class ThermoStVenantKirchhoff : public Trait::ThermoSolid
  {
   public:
    //! construct empty material object
    ThermoStVenantKirchhoff();

    //! construct the material object given material parameters
    explicit ThermoStVenantKirchhoff(Mat::PAR::ThermoStVenantKirchhoff* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ThermoStVenantKirchhoffType::instance().unique_par_object_id();
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::linear && kinem != Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.
    */
    void pack(
        Core::Communication::PackBuffer& data  //!< (i/o): char vector to store class information
    ) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_thermostvenant;
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ThermoStVenantKirchhoff>(*this);
    }

    //! evaluates stresses for 3d
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// add strain energy
    [[nodiscard]] double strain_energy(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,  ///< Green-Lagrange strain
        const EvaluationContext<3>& context,                          ///< evaluation context
        int gp,                                                       ///< Gauss point
        int eleGID                                                    ///< element GID
    ) const override;

    //! return true if Young's modulus is temperature dependent
    bool youngs_is_temp_dependent() const { return this->params_->youngs_.size() > 1; }

    //! density \f$ \rho \f$
    double density() const override { return params_->density_; }

    //! Initial temperature \f$ \theta_0 \f$
    double init_temp() const { return params_->thetainit_; }

    //! Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void reinit(double temperature, unsigned gp) override;

    Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_d_stress_d_scalar(
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int gp,
        int eleGID) override;

    void stress_temperature_modulus_and_deriv(Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp) override;

    //! general thermal tangent of material law depending on stress-temperature modulus
    static void fill_cthermo(Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp, double m);

   private:
    //! computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat) const;

    //! computes temperature dependent isotropic elasticity tensor in matrix
    //! notion for 3d
    void setup_cthermo(Core::LinAlg::SymmetricTensor<double, 3, 3>& ctemp) const;

    //! calculates stress-temperature modulus
    double st_modulus() const;

    //! calculates stress-temperature modulus
    double get_st_modulus_t() const;

    //! calculates derivative of Cmat with respect to current temperatures
    //! only in case of temperature-dependent material parameters
    void get_cmat_at_tempnp_t(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& derivcmat) const;

    //! calculates derivative of Cmat with respect to current temperatures
    //! only in case of temperature-dependent material parameters
    void get_cthermo_at_tempnp_t(Core::LinAlg::SymmetricTensor<double, 3, 3>&
            derivctemp  //!< linearisation of ctemp w.r.t. T
    ) const;

    //! calculate temperature dependent material parameter and return value
    double get_mat_parameter_at_tempnp(
        const std::vector<double>* paramvector,  //!< (i) given parameter is a vector
        const double& tempnp                     // tmpr (i) current temperature
    ) const;

    //! calculate temperature dependent material parameter and return value
    double get_mat_parameter_at_tempnp_t(
        const std::vector<double>* paramvector,  //!< (i) given parameter is a vector
        const double& tempnp                     // tmpr (i) current temperature
    ) const;

    //! my material parameters
    Mat::PAR::ThermoStVenantKirchhoff* params_;

    //! current temperature (set by Reinit())
    double current_temperature_{};

  };  // ThermoStVenantKirchhoff
}  // namespace Mat


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
