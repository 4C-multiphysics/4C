// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_HPP
#define FOUR_C_MAT_FLUIDPORO_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace FLUIDPORO
  {
    class PoroAnisotropyStrategyBase;
  }
  namespace PAR
  {
    enum PoroFlowType
    {
      undefined,
      darcy,
      darcy_brinkman
    };

    enum PoroFlowPermeabilityFunction
    {
      pf_undefined,
      constant,
      kozeny_carman,
      const_material_transverse,
      const_material_orthotropic,
      const_material_nodal_orthotropic
    };

    /*----------------------------------------------------------------------*/
    //! material parameters for fluid poro
    //! This object exists only once for each read Newton fluid.
    class FluidPoro : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      FluidPoro(const Core::Mat::PAR::Parameter::Data& matdata);

      //! set initial porosity from structural material and calculate
      //! permeability_correction_factor_
      void set_initial_porosity(double initial_porosity);

      //! @name material parameters
      //!@{

      //! kinematic or dynamic viscosity
      const double viscosity_;
      //! density
      const double density_;
      //! permeability
      const double permeability_;
      //! axial permeability for material transverse isotropy
      const double axial_permeability_;
      //! vector of orthotropic permeabilities
      std::vector<double> orthotropic_permeabilities_;
      //! flow type: Darcy or Darcy-Brinkman
      PoroFlowType type_;
      //! flag indicating varying permeability
      const bool varying_permeability_;
      //! permeability function type
      PoroFlowPermeabilityFunction permeability_func_;
      //! a correction factor to ensure the permeability set in the input file
      double permeability_correction_factor_;
      //! initial porosity
      double initial_porosity_;

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };

  }  // namespace PAR

  class FluidPoroType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroType"; }

    static FluidPoroType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroType instance_;
  };

  /*----------------------------------------------------------------------*/
  //! Wrapper for poro fluid flow material
  //! This object exists (several times) at every element
  class FluidPoro : public Core::Mat::Material
  {
   public:
    //! construct empty material object
    FluidPoro();

    //! construct the material object given material parameters
    explicit FluidPoro(Mat::PAR::FluidPoro* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return FluidPoroType::instance().unique_par_object_id();
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

    //!@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo;
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoro>(*this);
    }

    //! compute reaction tensor - 2D
    void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions = {},
        const std::vector<double>& anisotropic_permeability_coeffs = {}) const;

    //! compute reaction tensor - 3D
    void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions = {},
        const std::vector<double>& anisotropic_permeability_coeffs = {}) const;

    //! compute reaction coefficient
    double compute_reaction_coeff() const;

    //! compute linearization of reaction tensor - 2D
    void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<2, 2>& linreac_dphi,
        Core::LinAlg::Matrix<2, 2>& linreac_dJ, const double& J, const double& porosity) const;

    //! compute linearization of reaction tensor - 3D
    void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<3, 3>& linreac_dphi,
        Core::LinAlg::Matrix<3, 3>& linreac_dJ, const double& J, const double& porosity) const;

    //! effective viscosity (zero for Darcy and greater than zero for Darcy-Brinkman)
    double effective_viscosity() const;

    //! return type
    PAR::PoroFlowType type() const { return params_->type_; }

    //! return viscosity
    double viscosity() const { return params_->viscosity_; }

    //! return density
    double density() const override { return params_->density_; }

    //! return permeability function
    PAR::PoroFlowPermeabilityFunction permeability_function() const
    {
      return params_->permeability_func_;
    }

    //! Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! flag indicating a varying permeability
    bool varying_permeability() const { return params_->varying_permeability_; }

    //! flag indicating nodal orthotropy
    bool is_nodal_orthotropic() const
    {
      return params_->permeability_func_ == Mat::PAR::const_material_nodal_orthotropic;
    }

   private:
    //! my material parameters
    Mat::PAR::FluidPoro* params_;

    //! anisotropy strategy (isotropy, transverse isotropy, orthotropy, or nodal orthotropy)
    std::shared_ptr<Mat::FLUIDPORO::PoroAnisotropyStrategyBase> anisotropy_strategy_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
