// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_MULTIPHASE_HPP
#define FOUR_C_MAT_FLUIDPORO_MULTIPHASE_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for list of single fluid phases
    class FluidPoroMultiPhase : public MatList
    {
     public:
      /// standard constructor
      FluidPoroMultiPhase(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize the material
      virtual void initialize();

      /// @name material parameters
      //@{
      /// permeability
      const double permeability_;

      /// number of fluid phases of the nummat
      const int numfluidphases_;

      /// number of volume fractions of the nummat
      int numvolfrac_;

      // closing relation for additional porespace
      Mat::PAR::PoroFluidPressureBased::ClosingRelation closing_relation_volfrac_;

      //@}

      //! transformation of degrees of freedom to true pressures
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> dof2pres_;

      //! number of constraint saturation phase
      int constraintphaseID_;

      //! initialize flag
      bool isinit_;

    };  // class FluidPoroMultiPhase

  }  // namespace PAR

  class FluidPoroMultiPhaseType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroMultiPhaseType"; }

    static FluidPoroMultiPhaseType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroMultiPhaseType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a list of porous flow phases
  class FluidPoroMultiPhase : public MatList
  {
   public:
    /// construct empty material object
    FluidPoroMultiPhase();

    /// construct the material object given material parameters
    explicit FluidPoroMultiPhase(Mat::PAR::FluidPoroMultiPhase* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return FluidPoroMultiPhaseType::instance().unique_par_object_id();
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

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_multiphase;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroMultiPhase>(*this);
    }

    /// return permeability
    double permeability() const { return paramsporo_->permeability_; }

    /// return number of fluid phases
    int num_fluid_phases() const { return paramsporo_->numfluidphases_; }

    /// return number of volume fractions
    int num_vol_frac() const { return paramsporo_->numvolfrac_; }

    /// Return quick accessible material parameter data
    Mat::PAR::FluidPoroMultiPhase* parameter() const override { return paramsporo_; }

    /// initialize the material
    virtual void initialize();

    /// return whether reaction terms need to be evaluated
    virtual bool is_reactive() const { return false; };

    /// evaluate the generalized(!) pressure and saturation of all phases
    void evaluate_gen_pressure_and_saturation(
        std::vector<double>& genpressure, const std::vector<double>& phinp) const;

    /// evaluate the generalized(!) pressure of all phases
    void evaluate_gen_pressure(
        std::vector<double>& genpressure, const std::vector<double>& phinp) const;

    /// evaluate saturation of all phases
    void evaluate_saturation(std::vector<double>& saturation, const std::vector<double>& phinp,
        const std::vector<double>& pressure) const;

    //! transform generalized pressures to true pressures
    void transform_gen_pres_to_true_pres(
        const std::vector<double>& phi, std::vector<double>& phi_transformed) const;

    //! evaluate derivative of degree of freedom with respect to pressure
    void evaluate_deriv_of_dof_wrt_pressure(
        Core::LinAlg::SerialDenseMatrix& derivs, const std::vector<double>& state) const;

    //! evaluate derivative of saturation with respect to pressure
    void evaluate_deriv_of_saturation_wrt_pressure(
        Core::LinAlg::SerialDenseMatrix& derivs, const std::vector<double>& pressure) const;

    //! evaluate second derivative of saturation with respect to pressure
    void evaluate_second_deriv_of_saturation_wrt_pressure(
        std::vector<Core::LinAlg::SerialDenseMatrix>& derivs,
        const std::vector<double>& pressure) const;

   private:
    /// clear everything
    void clear();

    /// my material parameters
    Mat::PAR::FluidPoroMultiPhase* paramsporo_;
  };

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
