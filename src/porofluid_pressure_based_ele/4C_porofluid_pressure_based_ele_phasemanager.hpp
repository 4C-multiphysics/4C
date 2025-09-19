// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_PHASEMANAGER_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_PHASEMANAGER_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  class Material;
  class StructPoro;
  class FluidPoroSinglePhase;
  class FluidPoroMultiPhase;
}  // namespace Mat

namespace Core::Elements
{
  class Element;
}

namespace Discret
{

  namespace Elements
  {
    class PoroFluidMultiPhaseEleParameter;

    namespace PoroFluidManager
    {
      class VariableManagerMinAccess;

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief interface to phase manager classes

      These classes manage all accesses to variables at gauss points for the different
      phases. The phase manager is responsible for the choice of primary variables.
      As we can choose pressures or saturation as primary variables this class
      will make sure everything is accessed correctly. Therefore it differentiates
      between the phases (as each phase can have a different physical primary variable)

      (in contrast to variable manager, which manages the 'generic' primary variables. Note that
      these two manager classes are closely related).

      The idea is, that there are the methods setup(...) and EvaluateGPState(..), which
      need to be called before evaluation.
      All other methods are (more or less) constant access methods.

      All implementations are derived from this class.

      Two factory methods (in fact, only one as one calls the other), provide
      the specialization depending on the evaluation action.
      */

      class PhaseManagerInterface
      {
       public:
        //! constructor
        PhaseManagerInterface() {};

        //! destructor
        virtual ~PhaseManagerInterface() = default;

        //! factory method
        static std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface>
        create_phase_manager(const Discret::Elements::PoroFluidMultiPhaseEleParameter& para,
            int nsd, Core::Materials::MaterialType mattype, const PoroPressureBased::Action& action,
            int totalnumdofpernode, int numfluidphases, int numvolfrac);

        //! factory method
        static std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface>
        wrap_phase_manager(const Discret::Elements::PoroFluidMultiPhaseEleParameter& para, int nsd,
            Core::Materials::MaterialType mattype, const PoroPressureBased::Action& action,
            std::shared_ptr<PhaseManagerInterface> corephasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        virtual void setup(const Core::Elements::Element* ele, const int matnum = 0) = 0;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) = 0;

        //! clear the states
        virtual void clear_gp_state() = 0;

        //! check for reactions
        virtual bool is_reactive(int phasenum) const = 0;

        //! get scalar to phase mapping
        virtual Mat::ScaTraMatMultiPoro::ScalarToPhaseMap scalar_to_phase(int iscal) const = 0;

        //! check if EvaluateGPState() was called
        virtual void check_is_evaluated() const = 0;

        //! check if setup() was called
        virtual void check_is_setup() const = 0;

        //! @name Access methods

        //! get the number of phases
        virtual int num_fluid_phases() const = 0;

        //! get the total number of dofs (number of phases + 2*number of volume fractions)
        virtual int total_num_dof() const = 0;

        //! get the total number of volume fractions
        virtual int num_vol_frac() const = 0;

        //! get solid pressure
        virtual double solid_pressure() const = 0;

        //! recalculate solid pressure
        virtual void recalculate_solid_pressure(const double porosity) = 0;

        //! get saturation of phase 'phasenum'
        virtual double saturation(int phasenum) const = 0;

        //! get pressure of phase 'phasenum'
        virtual double pressure(int phasenum) const = 0;

        //! get saturation of all phases
        virtual const std::vector<double>& saturation() const = 0;

        //! get volfracs of all phases
        virtual const std::vector<double>& vol_frac() const = 0;

        //! get volfrac of volfrac 'volfracnum'
        virtual double vol_frac(int volfracnum) const = 0;

        //! get volfrac pressures of all phases
        virtual const std::vector<double>& vol_frac_pressure() const = 0;

        //! get volfrac pressure of volfrac 'volfracnum'
        virtual double vol_frac_pressure(int volfracnum) const = 0;

        //! get sum of additional volume fractions
        virtual double sum_add_vol_frac() const = 0;

        //! get pressure of all phases
        virtual const std::vector<double>& pressure() const = 0;

        //! get bulk modulus of phase 'phasenum'
        virtual double inv_bulkmodulus(int phasenum) const = 0;

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        virtual bool incompressible_fluid_phase(int phasenum) const = 0;

        //! get inverse bulk modulus of solid phase
        virtual double inv_bulkmodulus_solid() const = 0;

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        virtual bool incompressible_solid() const = 0;

        //! get density of phase 'phasenum'
        virtual double density(int phasenum) const = 0;

        //! get density of phase 'phasenum'
        virtual double vol_frac_density(int volfracnum) const = 0;

        //! get the density of solid phase
        virtual double solid_density() const = 0;

        //! get the current element the manager was set up with
        virtual const Core::Elements::Element* element() const = 0;

        //! get porosity
        virtual double porosity() const = 0;

        //! get Jacobian of deformation gradient
        virtual double jacobian_def_grad() const = 0;

        //! get derivative of porosity wrt JacobianDefGrad
        virtual double porosity_deriv_wrt_jacobian_def_grad() const = 0;

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        virtual double porosity_deriv(int doftoderive) const = 0;

        //! check if porosity depends on fluid (pressure)
        virtual bool porosity_depends_on_fluid() const = 0;

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        virtual bool porosity_depends_on_struct() const = 0;

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double saturation_deriv(int phasenum, int doftoderive) const = 0;

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double saturation_deriv_deriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const = 0;

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double pressure_deriv(int phasenum, int doftoderive) const = 0;

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        virtual double solid_pressure_deriv(int doftoderive) const = 0;

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        virtual double solid_pressure_deriv_deriv(int doftoderive, int doftoderive2) const = 0;

        //! get the reaction term
        virtual double reac_term(int phasenum) const = 0;

        //! get the derivative of the reaction term
        virtual double reac_deriv(int phasenum, int doftoderive) const = 0;

        //! get total number of scalars in system
        virtual int num_scal() const = 0;

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        virtual double reac_deriv_scalar(int phasenum, int scaltoderive) const = 0;

        //! get the derivative of the reaction term w.r.t. porosity
        virtual double reac_deriv_porosity(int phasenum) const = 0;

        //! get the diffusion tensor
        virtual void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<3, 3>& permeabilitytensor) const = 0;
        //! get the diffusion tensor
        virtual void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<2, 2>& permeabilitytensor) const = 0;
        //! get the diffusion tensor
        virtual void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<1, 1>& permeabilitytensor) const = 0;

        //! check for constant relpermeability
        virtual bool has_constant_rel_permeability(int phasenum) const = 0;
        //! get relative diffusivity of phase
        virtual double rel_permeability(int phasenum) const = 0;
        //! get derivative of relative permeability of phase
        virtual double rel_permeability_deriv(int phasenum) const = 0;

        //! check for constant dynamic viscosity
        virtual bool has_constant_dyn_viscosity(int phasenum) const = 0;

        //! get dynamic viscosity of phase (matnum is the material number of the porofluid-material
        //! on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual double dyn_viscosity(int phasenum, double abspressgrad, int matnum = 0) const = 0;
        //! get dynamic viscosity of phase
        virtual double dyn_viscosity(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const = 0;
        //! get derivative of dynamic viscosity of phase
        virtual double dyn_viscosity_deriv(int phasenum, double abspressgrad) const = 0;
        //! get derivative dynamic viscosity of phase
        virtual double dyn_viscosity_deriv(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const = 0;

        //! check for constant dynamic viscosity of volume fraction pressure
        virtual bool has_constant_dyn_viscosity_vol_frac_pressure(int volfracpressnum) const = 0;

        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual double dyn_viscosity_vol_frac_pressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const = 0;
        //! get dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const = 0;
        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual double dyn_viscosity_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad, int matnum = 0) const = 0;
        //! get dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_vol_frac_pressure_blood_lung(
            const Core::Mat::Material& material, int volfracpressnum,
            double abspressgrad) const = 0;
        //! get derivative of dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_deriv_vol_frac_pressure(
            int volfracpressnum, double abspressgrad) const = 0;
        //! get derivative dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_deriv_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const = 0;
        //! get derivative of dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad) const = 0;
        //! get derivative dynamic viscosity of volume fraction pressure
        virtual double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
            const Core::Mat::Material& material, int volfracpressnum,
            double abspressgrad) const = 0;

        //! get the diffusion tensor
        virtual void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<3, 3>& difftensorvolfrac) const = 0;
        //! get the diffusion tensor
        virtual void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<2, 2>& difftensorvolfrac) const = 0;
        //! get the diffusion tensor
        virtual void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<1, 1>& difftensorvolfrac) const = 0;

        //! get the permeability tensor for volume fraction pressures
        virtual void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<3, 3>& permeabilitytensorvolfracpressure) const = 0;
        //! get the permeability tensor for volume fraction pressures
        virtual void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<2, 2>& permeabilitytensorvolfracpressure) const = 0;
        //! get the permeability tensor for volume fraction pressures
        virtual void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<1, 1>& permeabilitytensorvolfracpressure) const = 0;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        virtual bool has_add_scalar_dependent_flux(int volfracnum) const = 0;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        virtual bool has_add_scalar_dependent_flux(int volfracnum, int iscal) const = 0;

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        virtual bool has_receptor_kinetic_law(int volfracnum, int iscal) const = 0;

        //! return scalar diffusivity of of scalar 'iscal' of volume fraction 'volfracnum'
        virtual double scalar_diff(int volfracnum, int iscal) const = 0;

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        virtual double omega_half(int volfracnum, int iscal) const = 0;

        // initial volume fraction (usually at end-expiration)
        virtual double initial_volfrac() const = 0;

        // parameter determining the deformation dependence
        virtual double volfrac_blood_lung_parameter_deformation_dependence() const = 0;

        // parameter determining the pressure dependence
        virtual double volfrac_blood_lung_parameter_pressure_dependence() const = 0;


        //@}
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief standard phase manager, holding pressures and saturations

      This class is a minimal version of a phase manager. It basically provides
      access to the primary variables (saturation, pressure, etc.) and quantities
      that can be accessed via the material in almost all cases (bulkmodulus, ...).
      */
      class PhaseManagerCore : public PhaseManagerInterface
      {
       public:
        //! constructor
        explicit PhaseManagerCore(int totalnumdofpernode, int numfluidphases, int numvolfrac);

        //! copy constructor
        PhaseManagerCore(const PhaseManagerCore& old);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void setup(const Core::Elements::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! check for reactions
        bool is_reactive(int phasenum) const override { return false; };

        //! @name Access methods

        //! get the number of phases
        int num_fluid_phases() const override { return numfluidphases_; };

        //! get the total number of dofs (number of phases + 2*number of volume fractions)
        int total_num_dof() const override { return totalnumdofpernode_; };

        //! get the number of volume fractions
        int num_vol_frac() const override { return numvolfrac_; };

        //! get solid pressure
        double solid_pressure() const override;

        //! recalculate solid pressure
        void recalculate_solid_pressure(const double porosity) override;

        //! get saturation of phase 'phasenum'
        double saturation(int phasenum) const override;

        //! get pressure of phase 'phasenum'
        double pressure(int phasenum) const override;

        //! get saturation of all phases
        const std::vector<double>& saturation() const override;

        //! get volfracs of all phases
        const std::vector<double>& vol_frac() const override;

        //! get volfrac of volfrac 'volfracnum'
        double vol_frac(int volfracnum) const override;

        //! get volfrac pressures of all volfracs
        const std::vector<double>& vol_frac_pressure() const override;

        //! get volfrac pressure of volfrac 'volfracnum'
        double vol_frac_pressure(int volfracnum) const override;

        //! get sum of additional volume fractions
        double sum_add_vol_frac() const override;

        //! get pressure of all phases
        const std::vector<double>& pressure() const override;

        //! get bulk modulus of phase 'phasenum'
        double inv_bulkmodulus(int phasenum) const override;

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        bool incompressible_fluid_phase(int phasenum) const override;

        //! get inverse bulk modulus of solid phase
        double inv_bulkmodulus_solid() const override;

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        bool incompressible_solid() const override;

        //! get density of phase 'phasenum'
        double density(int phasenum) const override;

        //! get density of phase 'phasenum'
        double vol_frac_density(int volfracnum) const override;

        //! get the density of the solid phase
        double solid_density() const override;

        //! get the current element the manager was set up with
        const Core::Elements::Element* element() const override { return ele_; };

        //@}

        //! check if EvaluateGPState() was called
        void check_is_evaluated() const override
        {
          if (not isevaluated_) FOUR_C_THROW("Gauss point states have not been set!");
        }

        //! check if EvaluateGPState() was called
        void check_is_setup() const override
        {
          if (not issetup_) FOUR_C_THROW("setup() was not called!");
        }

        //! get porosity
        double porosity() const override
        {
          FOUR_C_THROW("Porosity not available for this phase manager!");
        };

        //! get porosity
        double jacobian_def_grad() const override
        {
          FOUR_C_THROW("JacobianDefGrad not available for this phase manager!");
        };

        //! get derivative of porosity wrt JacobianDefGrad
        double porosity_deriv_wrt_jacobian_def_grad() const override
        {
          FOUR_C_THROW(
              "Derivative of Porosity w.r.t. JacobianDefGrad not available for this phase "
              "manager!");
        };

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double porosity_deriv(int doftoderive) const override
        {
          FOUR_C_THROW("Derivative of porosity not available for this phase manager!");
        };

        //! check if porosity depends on fluid (pressure)
        bool porosity_depends_on_fluid() const override
        {
          FOUR_C_THROW("porosity_depends_on_fluid() not available for this phase manager!");
        };

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool porosity_depends_on_struct() const override
        {
          FOUR_C_THROW("porosity_depends_on_struct() not available for this phase manager!");
        };

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double saturation_deriv(int phasenum, int doftoderive) const override
        {
          FOUR_C_THROW("Derivative of saturation not available for this phase manager!");
        };

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double saturation_deriv_deriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override
        {
          FOUR_C_THROW("2nd Derivative of saturation not available for this phase manager!");
        };

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double pressure_deriv(int phasenum, int doftoderive) const override
        {
          FOUR_C_THROW("Derivative of pressure not available for this phase manager!");
        };

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double solid_pressure_deriv(int doftoderive) const override
        {
          FOUR_C_THROW("Derivative of solid pressure not available for this phase manager!");
        };

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double solid_pressure_deriv_deriv(int doftoderive, int doftoderive2) const override
        {
          FOUR_C_THROW("Second derivative of solid pressure not available for this phase manager!");
        };

        //! get the reaction term
        double reac_term(int phasenum) const override
        {
          FOUR_C_THROW("Reaction term not available for this phase manager!");
        };

        //! get total number of scalars in system
        int num_scal() const override
        {
          FOUR_C_THROW("Number of scalars not available for this phase manager");
          return 0;
        };

        //! get the derivative of the reaction term
        double reac_deriv(int phasenum, int doftoderive) const override
        {
          FOUR_C_THROW("Reaction term derivative not available for this phase manager!");
        };

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double reac_deriv_scalar(int phasenum, int scaltoderive) const override
        {
          FOUR_C_THROW("Reaction term derivative (scalar) not available for this phase manager!");
        };

        //! get the derivative of the reaction term w.r.t. porosity
        double reac_deriv_porosity(int phasenum) const override
        {
          FOUR_C_THROW("Reaction term derivative (porosity) not available for this phase manager!");
        };

        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<3, 3>& permeabilitytensor) const override
        {
          FOUR_C_THROW("Diffusion tensor (3D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<2, 2>& permeabilitytensor) const override
        {
          FOUR_C_THROW("Diffusion tensor (2D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<1, 1>& permeabilitytensor) const override
        {
          FOUR_C_THROW("Diffusion tensor (1D) not available for this phase manager!");
        };

        //! check for constant relpermeability
        bool has_constant_rel_permeability(int phasenum) const override
        {
          FOUR_C_THROW(
              "Check for Constant Relative Permeability not available for this phase manager!");
        };
        //! get relative diffusivity of phase
        double rel_permeability(int phasenum) const override
        {
          FOUR_C_THROW("Relative Diffusivity not available for this phase manager!");
        };
        //! get derivative of relative permeability of phase
        double rel_permeability_deriv(int phasenum) const override
        {
          FOUR_C_THROW("Derivative of relative permeability not available for this phase manager!");
        };

        //! check for constant dynamic viscosity
        bool has_constant_dyn_viscosity(int phasenum) const override
        {
          FOUR_C_THROW(
              "Check for Constant Dynamic Viscosity not available for this phase manager!");
        };
        //! get relative diffusivity of phase
        double dyn_viscosity(int phasenum, double abspressgrad, int matnum = 0) const override
        {
          FOUR_C_THROW("Dynamic Viscosity not available for this phase manager!");
        };
        //! get dynamic viscosity of phase
        double dyn_viscosity(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override
        {
          FOUR_C_THROW("Dynamic Viscosity not available for this phase manager!");
        };
        //! get derivative of dynamic viscosity of phase
        double dyn_viscosity_deriv(int phasenum, double abspressgrad) const override
        {
          FOUR_C_THROW("Derivative of dynamic Viscosity not available for this phase manager!");
        };
        //! get derivative dynamic viscosity of phase
        double dyn_viscosity_deriv(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override
        {
          FOUR_C_THROW("Derivative of dynamic Viscosity not available for this phase manager!");
        };

        //! check for constant dynamic viscosity of volume fraction pressure
        bool has_constant_dyn_viscosity_vol_frac_pressure(int volfracpressnum) const override
        {
          FOUR_C_THROW(
              "Check for Constant Dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
        };
        //! get relative diffusivity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          FOUR_C_THROW("Dynamic Viscosity (VolFracPressure) not available for this phase manager!");
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW("Dynamic Viscosity (VolFracPressure) not available for this phase manager!");
        };
        //! get relative diffusivity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          FOUR_C_THROW(
              "Dynamic Viscosity (VolFracPressure in the lungs) not available for this phase "
              "manager!");
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW(
              "Dynamic Viscosity (VolFracPressure in the lungs) not available for this phase "
              "manager!");
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          FOUR_C_THROW(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
        };

        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<3, 3>& difftensorvolfrac) const override
        {
          FOUR_C_THROW(
              "Diffusion tensor for volume fractions (3D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<2, 2>& difftensorvolfrac) const override
        {
          FOUR_C_THROW(
              "Diffusion tensor for volume fractions (2D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<1, 1>& difftensorvolfrac) const override
        {
          FOUR_C_THROW(
              "Diffusion tensor for volume fractions (1D) not available for this phase manager!");
        };

        //! get the permeabilty tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<3, 3>& permeabilitytensorvolfracpressure) const override
        {
          FOUR_C_THROW(
              "Permeability tensor for volume fraction pressures (3D) not available for this phase "
              "manager!");
        };
        //! get the permeabilty tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<2, 2>& permeabilitytensorvolfracpressure) const override
        {
          FOUR_C_THROW(
              "Permeability tensor for volume fraction pressures (2D) not available for this phase "
              "manager!");
        };
        //! get the permeabilty tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<1, 1>& permeabilitytensorvolfracpressure) const override
        {
          FOUR_C_THROW(
              "Permeability tensor for volume fraction pressures (1D) not available for this phase "
              "manager!");
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool has_add_scalar_dependent_flux(int volfracnum) const override
        {
          FOUR_C_THROW("has_add_scalar_dependent_flux not available for this phase manager!");
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool has_add_scalar_dependent_flux(int volfracnum, int iscal) const override
        {
          FOUR_C_THROW("has_add_scalar_dependent_flux not available for this phase manager!");
        };

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool has_receptor_kinetic_law(int volfracnum, int iscal) const override
        {
          FOUR_C_THROW("has_receptor_kinetic_law not available for this phase manager!");
        };

        //! return scalar diffusivities of scalar 'iscal' of volume fraction 'volfracnum'
        double scalar_diff(int volfracnum, int iscal) const override
        {
          FOUR_C_THROW("ScalarDiff not available for this phase manager!");
        };

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double omega_half(int volfracnum, int iscal) const override
        {
          FOUR_C_THROW("OmegaHalf not available for this phase manager!");
        };

        //! get scalar to phase mapping
        Mat::ScaTraMatMultiPoro::ScalarToPhaseMap scalar_to_phase(int iscal) const override
        {
          FOUR_C_THROW("ScalarToPhase not available for this phase manager!");
          Mat::ScaTraMatMultiPoro::ScalarToPhaseMap null_map;
          return null_map;
        };

        // initial volume fraction (usually at end-expiration)
        [[nodiscard]] double initial_volfrac() const override;

        // parameter determining the deformation dependence
        double volfrac_blood_lung_parameter_deformation_dependence() const override;


        // parameter determining the pressure dependence
        double volfrac_blood_lung_parameter_pressure_dependence() const override;


       private:
        //! total number of dofs per node (numfluidphases + numvolfrac)
        const int totalnumdofpernode_;
        //! number of fluid phases
        const int numfluidphases_;
        //! number of phases
        const int numvolfrac_;

        //! generalized pressure
        std::vector<double> genpressure_;
        //! additional volume fraction
        std::vector<double> volfrac_;
        //! additional volume fraction pressures
        std::vector<double> volfracpressure_;
        //! sum of additional volume fractions
        double sumaddvolfrac_;
        //! true pressure
        std::vector<double> pressure_;
        //! saturation
        std::vector<double> saturation_;
        //! densities
        std::vector<double> density_;
        //! densities of volume fractions
        std::vector<double> volfracdensity_;
        //! solid density
        double soliddensity_;
        //! solid pressure
        double solidpressure_;
        //! inverse bulk moduli of the fluid phases
        std::vector<double> invbulkmodulifluid_;
        //! inverse solid bulk modulus
        double invbulkmodulussolid_;

        //! the current element
        const Core::Elements::Element* ele_;

        //! flag indicating of gauss point state has been set and evaluated
        bool isevaluated_;

        //! flag of Setup was called
        bool issetup_;
        //! get initial volfrac value in case the material has closing relation for blood in lung
        //! active
        double initialvolfrac_ = 0.0;

        double volfrac_deformationbased_scaling_parameter_deformation_ = 0.0;

        double volfrac_deformationbased_scaling_parameter_pressure_ = 0.0;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, base class for extensions to phase manager

      The idea is to use a core phase manager (class PhaseManagerCore) and extend
      it when the evaluation to be performed demands it. For this a decorator
       pattern is used, i.e. this class wraps another phase manager, extending it if necessary.

      This is the base class for all decorators.
      */
      class PhaseManagerDecorator : public PhaseManagerInterface
      {
       public:
        //! constructor
        explicit PhaseManagerDecorator(
            std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
            : phasemanager_(phasemanager) {};

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void setup(const Core::Elements::Element* ele, const int matnum = 0) override
        {
          phasemanager_->setup(ele, matnum);
        };

        //! check if EvaluateGPState() was called
        void check_is_evaluated() const override { phasemanager_->check_is_evaluated(); };

        //! check if setup() was called
        void check_is_setup() const override { phasemanager_->check_is_setup(); };

        //! @name Access methods

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double porosity_deriv(int doftoderive) const override
        {
          return phasemanager_->porosity_deriv(doftoderive);
        };

        //! check if porosity depends on fluid (pressure)
        bool porosity_depends_on_fluid() const override
        {
          return phasemanager_->porosity_depends_on_fluid();
        };

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool porosity_depends_on_struct() const override
        {
          return phasemanager_->porosity_depends_on_struct();
        };

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double saturation_deriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->saturation_deriv(phasenum, doftoderive);
        };

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double saturation_deriv_deriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override
        {
          return phasemanager_->saturation_deriv_deriv(
              phasenum, firstdoftoderive, seconddoftoderive);
        };

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double pressure_deriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->pressure_deriv(phasenum, doftoderive);
        };

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double solid_pressure_deriv(int doftoderive) const override
        {
          return phasemanager_->solid_pressure_deriv(doftoderive);
        };

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double solid_pressure_deriv_deriv(int doftoderive, int doftoderive2) const override
        {
          return phasemanager_->solid_pressure_deriv_deriv(doftoderive, doftoderive2);
        };

        //! check if the current phase is involved in a reaction
        bool is_reactive(int phasenum) const override
        {
          return phasemanager_->is_reactive(phasenum);
        };

        //! get scalar to phase mapping
        Mat::ScaTraMatMultiPoro::ScalarToPhaseMap scalar_to_phase(int iscal) const override
        {
          return phasemanager_->scalar_to_phase(iscal);
        }

        //! get the number of phases
        int num_fluid_phases() const override { return phasemanager_->num_fluid_phases(); };

        //! get the number of dofs (number of phases + 2*number of volfracs)
        int total_num_dof() const override { return phasemanager_->total_num_dof(); };

        //! get the number of volume fractions
        int num_vol_frac() const override { return phasemanager_->num_vol_frac(); };

        //! get solid pressure
        double solid_pressure() const override { return phasemanager_->solid_pressure(); };

        //! recalculate solid pressure
        void recalculate_solid_pressure(const double porosity) override
        {
          return phasemanager_->recalculate_solid_pressure(porosity);
        };

        //! get saturation of phase 'phasenum'
        double saturation(int phasenum) const override
        {
          return phasemanager_->saturation(phasenum);
        };

        //! get pressure of phase 'phasenum'
        double pressure(int phasenum) const override { return phasemanager_->pressure(phasenum); };

        //! get saturation of all phases
        const std::vector<double>& saturation() const override
        {
          return phasemanager_->saturation();
        };

        //! get volfracs of all phases
        const std::vector<double>& vol_frac() const override { return phasemanager_->vol_frac(); };

        //! get volfrac of volfrac 'volfracnum'
        double vol_frac(int volfracnum) const override
        {
          return phasemanager_->vol_frac(volfracnum);
        };

        //! get volfrac pressures of all phases
        const std::vector<double>& vol_frac_pressure() const override
        {
          return phasemanager_->vol_frac_pressure();
        };

        //! get volfrac pressure of volfrac 'volfracnum'
        double vol_frac_pressure(int volfracnum) const override
        {
          return phasemanager_->vol_frac_pressure(volfracnum);
        };

        //! get solid pressure
        double sum_add_vol_frac() const override { return phasemanager_->sum_add_vol_frac(); };

        //! get pressure of all phases
        const std::vector<double>& pressure() const override { return phasemanager_->pressure(); };

        //! get bulk modulus of phase 'phasenum'
        double inv_bulkmodulus(int phasenum) const override
        {
          return phasemanager_->inv_bulkmodulus(phasenum);
        };

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        bool incompressible_fluid_phase(int phasenum) const override
        {
          return phasemanager_->incompressible_fluid_phase(phasenum);
        };

        //! get inverse bulk modulus of solid phase
        double inv_bulkmodulus_solid() const override
        {
          return phasemanager_->inv_bulkmodulus_solid();
        };

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        bool incompressible_solid() const override
        {
          return phasemanager_->incompressible_solid();
        };

        //! get porosity
        double porosity() const override { return phasemanager_->porosity(); };

        //! get JacobianDefGrad
        double jacobian_def_grad() const override { return phasemanager_->jacobian_def_grad(); };

        //! get derivative of porosity wrt JacobianDefGrad
        double porosity_deriv_wrt_jacobian_def_grad() const override
        {
          return phasemanager_->porosity_deriv_wrt_jacobian_def_grad();
        };

        //! get density of phase 'phasenum'
        double density(int phasenum) const override { return phasemanager_->density(phasenum); };

        //! get density of volume fraction 'volfracnum'
        double vol_frac_density(int volfracnum) const override
        {
          return phasemanager_->vol_frac_density(volfracnum);
        };

        //! get density of solid phase
        double solid_density() const override { return phasemanager_->solid_density(); };

        //! get the current element the manager was set up with
        const Core::Elements::Element* element() const override
        {
          return phasemanager_->element();
        };

        //! get the reaction term
        double reac_term(int phasenum) const override
        {
          return phasemanager_->reac_term(phasenum);
        };

        //! get total number of scalars in system
        int num_scal() const override { return phasemanager_->num_scal(); };

        //! get the derivative of the reaction term
        double reac_deriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->reac_deriv(phasenum, doftoderive);
        };

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double reac_deriv_scalar(int phasenum, int scaltoderive) const override
        {
          return phasemanager_->reac_deriv_scalar(phasenum, scaltoderive);
        };

        //! get the derivative of the reaction term w.r.t. porosity
        double reac_deriv_porosity(int phasenum) const override
        {
          return phasemanager_->reac_deriv_porosity(phasenum);
        };

        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<3, 3>& permeabilitytensor) const override
        {
          phasemanager_->permeability_tensor(phasenum, permeabilitytensor);
        };
        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<2, 2>& permeabilitytensor) const override
        {
          phasemanager_->permeability_tensor(phasenum, permeabilitytensor);
        };
        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<1, 1>& permeabilitytensor) const override
        {
          phasemanager_->permeability_tensor(phasenum, permeabilitytensor);
        };

        //! check for constant relpermeability
        bool has_constant_rel_permeability(int phasenum) const override
        {
          return phasemanager_->has_constant_rel_permeability(phasenum);
        };

        //! get relative diffusivity of phase
        double rel_permeability(int phasenum) const override
        {
          return phasemanager_->rel_permeability(phasenum);
        };

        //! get derivative of relative permeability of phase
        double rel_permeability_deriv(int phasenum) const override
        {
          return phasemanager_->rel_permeability_deriv(phasenum);
        };

        //! check for constant dynamic visosity
        bool has_constant_dyn_viscosity(int phasenum) const override
        {
          return phasemanager_->has_constant_dyn_viscosity(phasenum);
        };
        //! get dynamic viscosity of phase
        double dyn_viscosity(int phasenum, double abspressgrad, int matnum = 0) const override
        {
          return phasemanager_->dyn_viscosity(phasenum, abspressgrad, matnum = 0);
        };
        //! get dynamic viscosity of phase
        double dyn_viscosity(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity(material, phasenum, abspressgrad);
        };
        //! get derivative of dynamic viscosity of phase
        double dyn_viscosity_deriv(int phasenum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv(phasenum, abspressgrad);
        };
        //! get derivative dynamic viscosity of phase
        double dyn_viscosity_deriv(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv(material, phasenum, abspressgrad);
        };

        //! check for constant dynamic visosity of volume fraction pressure
        bool has_constant_dyn_viscosity_vol_frac_pressure(int volfracpressnum) const override
        {
          return phasemanager_->has_constant_dyn_viscosity_vol_frac_pressure(volfracpressnum);
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          return phasemanager_->dyn_viscosity_vol_frac_pressure(
              volfracpressnum, abspressgrad, matnum = 0);
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_vol_frac_pressure(
              material, volfracpressnum, abspressgrad);
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          return phasemanager_->dyn_viscosity_vol_frac_pressure_blood_lung(
              volfracpressnum, abspressgrad, matnum = 0);
        };
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_vol_frac_pressure_blood_lung(
              material, volfracpressnum, abspressgrad);
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv_vol_frac_pressure(
              volfracpressnum, abspressgrad);
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv_vol_frac_pressure(
              material, volfracpressnum, abspressgrad);
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
              volfracpressnum, abspressgrad);
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
              material, volfracpressnum, abspressgrad);
        };

        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<3, 3>& difftensorvolfrac) const override
        {
          phasemanager_->diff_tensor_vol_frac(volfracnum, difftensorvolfrac);
        };
        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<2, 2>& difftensorvolfrac) const override
        {
          phasemanager_->diff_tensor_vol_frac(volfracnum, difftensorvolfrac);
        };
        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<1, 1>& difftensorvolfrac) const override
        {
          phasemanager_->diff_tensor_vol_frac(volfracnum, difftensorvolfrac);
        };

        //! get the permeability tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<3, 3>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->permeability_tensor_vol_frac_pressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };
        //! get the permeability tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<2, 2>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->permeability_tensor_vol_frac_pressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };
        //! get the permeability tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<1, 1>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->permeability_tensor_vol_frac_pressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool has_add_scalar_dependent_flux(int volfracnum) const override
        {
          return phasemanager_->has_add_scalar_dependent_flux(volfracnum);
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool has_add_scalar_dependent_flux(int volfracnum, int iscal) const override
        {
          return phasemanager_->has_add_scalar_dependent_flux(volfracnum, iscal);
        };

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool has_receptor_kinetic_law(int volfracnum, int iscal) const override
        {
          return phasemanager_->has_receptor_kinetic_law(volfracnum, iscal);
        };

        //! return scalar diffusivities of scalar 'iscal' of volume fraction 'volfracnum'
        double scalar_diff(int volfracnum, int iscal) const override
        {
          return phasemanager_->scalar_diff(volfracnum, iscal);
        };

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double omega_half(int volfracnum, int iscal) const override
        {
          return phasemanager_->omega_half(volfracnum, iscal);
        };

        // initial volume fraction (usually at end-expiration)
        [[nodiscard]] double initial_volfrac() const override
        {
          return phasemanager_->initial_volfrac();
        }

        // parameter determining the deformation dependence
        [[nodiscard]] double volfrac_blood_lung_parameter_deformation_dependence() const override
        {
          return phasemanager_->volfrac_blood_lung_parameter_deformation_dependence();
        }

        // parameter determining the pressure dependence
        [[nodiscard]] double volfrac_blood_lung_parameter_pressure_dependence() const override
        {
          return phasemanager_->volfrac_blood_lung_parameter_pressure_dependence();
        }

        //@}

       protected:
        //! wrapped phase manager
        std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for derivatives

      This class is a decorator for a phase manager, including the derivatives
      of the pressures and saturations w.r.t. to the primary variables
      */
      class PhaseManagerDeriv : public PhaseManagerDecorator
      {
       public:
        //! constructor
        explicit PhaseManagerDeriv(
            std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager);

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! @name Access methods

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double saturation_deriv(int phasenum, int doftoderive) const override;

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        //! (dS/dphi_{ii})
        double saturation_deriv_deriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override;

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double pressure_deriv(int phasenum, int doftoderive) const override;

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double solid_pressure_deriv(int doftoderive) const override;

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double solid_pressure_deriv_deriv(int doftoderive, int doftoderive2) const override;

        //@}

       private:
        //! derivative of true pressure w.r.t. degrees of freedom
        // first index: pressure, second index: dof
        std::shared_ptr<Core::LinAlg::SerialDenseMatrix> pressurederiv_;
        //! derivative of saturations w.r.t. degrees of freedom
        // first index: saturation, second index: dof
        std::shared_ptr<Core::LinAlg::SerialDenseMatrix> saturationderiv_;
        //! second derivative of saturations w.r.t. degrees of freedom
        std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> saturationderivderiv_;

        //! derivative of solid pressure w.r.t. degrees of freedom
        std::shared_ptr<Core::LinAlg::SerialDenseVector> solidpressurederiv_;
        //! second derivative of solid pressure w.r.t. degrees of freedom;
        std::shared_ptr<Core::LinAlg::SerialDenseMatrix> solidpressurederivderiv_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for derivatives and porosity

      This class is a decorator for a phase manager, including evaluation of
      the pressures and saturations w.r.t. to the primary variables and the porosity.
      */
      class PhaseManagerDerivAndPorosity : public PhaseManagerDeriv
      {
       public:
        //! constructor
        explicit PhaseManagerDerivAndPorosity(
            std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager);

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! @name Access methods

        //! get porosity
        double porosity() const override;

        //! get porosity
        double jacobian_def_grad() const override;

        //! get derivative of porosity wrt JacobianDefGrad
        double porosity_deriv_wrt_jacobian_def_grad() const override;

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double porosity_deriv(int doftoderive) const override;

        //! check if porosity depends on fluid (pressure)
        bool porosity_depends_on_fluid() const override;

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool porosity_depends_on_struct() const override;

        //@}

       private:
        //! porosity
        double porosity_;

        //! Jacobian of def gradient
        double j_;

        //! derivative of porosity w.r.t. Jacobian of defgradient
        double dporosity_dj_;

        //! derivative of porosity w.r.t. solid pressure
        double dporosity_dp_;

        //! derivative of porosity w.r.t. degrees of freedom
        std::shared_ptr<Core::LinAlg::SerialDenseVector> porosityderiv_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for reaction/mass exchange terms

      This class is a decorator for a phase manager, including evaluation of
      reaction/mass exchange terms.
      */
      class PhaseManagerReaction : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerReaction(std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void setup(const Core::Elements::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! @name Access methods

        //! get the reaction term
        double reac_term(int phasenum) const override;

        //! get the derivative of the reaction term
        double reac_deriv(int phasenum, int doftoderive) const override;

        //! check if the current phase is involved in a reaction
        bool is_reactive(int phasenum) const override
        {
          check_is_setup();
          return isreactive_[phasenum];
        };

        //! get scalar to phase mapping
        Mat::ScaTraMatMultiPoro::ScalarToPhaseMap scalar_to_phase(int iscal) const override;

        //! get total number of scalars in system
        int num_scal() const override;

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double reac_deriv_scalar(int phasenum, int scaltoderive) const override;

        //! get the derivative of the reaction term w.r.t. porosity
        double reac_deriv_porosity(int phasenum) const override;

        //@}

       private:
        //! reaction terms
        std::vector<double> reacterms_;
        //! derivatives of reaction terms w.r.t. fluid primary dofs
        std::vector<std::vector<double>> reactermsderivs_;
        //! derivatives of reaction terms w.r.t. (true) pressures
        std::vector<std::vector<double>> reactermsderivspressure_;
        //! derivatives of reaction terms w.r.t. saturations
        std::vector<std::vector<double>> reactermsderivssaturation_;
        //! derivatives of reaction terms w.r.t. porosity
        std::vector<double> reactermsderivsporosity_;
        //! derivatives of reaction terms w.r.t. scalars --> needed for off-diagonal matrices
        std::vector<std::vector<double>> reactermsderivsscalar_;
        //! derivatives of reaction terms w.r.t. volume fraction
        std::vector<std::vector<double>> reactermsderivsvolfrac_;
        //! derivatives of reaction terms w.r.t. volume fraction pressures
        std::vector<std::vector<double>> reactermsderivsvolfracpressure_;

        //! flags indicating whether the phase is involved in a reaction
        std::vector<bool> isreactive_;
        //! scalar to phase map
        std::vector<Mat::ScaTraMatMultiPoro::ScalarToPhaseMap> scalartophasemap_;
        //! number of scalars
        int numscal_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for diffusion

      This class is a decorator for a phase manager, including evaluation of
      the diffusion tensor (inverse permeability). As the tensor is saved
      as a Core::LinAlg::Matrix, it is templated by the number of space dimensions
      */
      template <int nsd>
      class PhaseManagerDiffusion : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerDiffusion(
            std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void setup(const Core::Elements::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! @name Access methods

        //! get the diffusion tensor
        void permeability_tensor(
            int phasenum, Core::LinAlg::Matrix<nsd, nsd>& permeabilitytensor) const override;

        //! check for constant relpermeability
        bool has_constant_rel_permeability(int phasenum) const override;

        //! get relative diffusivity of phase
        double rel_permeability(int phasenum) const override;

        //! get derivative of relative permeability of phase
        double rel_permeability_deriv(int phasenum) const override;

        //! check for constant dynamic viscosity
        bool has_constant_dyn_viscosity(int phasenum) const override;
        //! get dynamic viscosity of phase
        double dyn_viscosity(int phasenum, double abspressgrad, int matnum = 0) const override;
        //! get dynamic viscosity of phase
        double dyn_viscosity(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override;
        //! get derivative of dynamic viscosity of phase
        double dyn_viscosity_deriv(int phasenum, double abspressgrad) const override;
        //! get derivative dynamic viscosity of phase
        double dyn_viscosity_deriv(
            const Core::Mat::Material& material, int phasenum, double abspressgrad) const override;

        //! get the permeability tensor for volume fraction pressures
        void permeability_tensor_vol_frac_pressure(int volfracpressnum,
            Core::LinAlg::Matrix<nsd, nsd>& permeabilitytensorvolfracpressure) const override;

        //! check for constant dynamic viscosity of volume fraction pressure
        bool has_constant_dyn_viscosity_vol_frac_pressure(int volfracpressnum) const override;

        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        double dyn_viscosity_vol_frac_pressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override;
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override;
        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        double dyn_viscosity_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override;
        //! get dynamic viscosity of volume fraction pressure
        double dyn_viscosity_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override;
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(
            int volfracpressnum, double abspressgrad) const override;
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override;
        //! get derivative of dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
            int volfracpressnum, double abspressgrad) const override;
        //! get derivative dynamic viscosity of volume fraction pressure
        double dyn_viscosity_deriv_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
            int volfracpressnum, double abspressgrad) const override;



        //@}

       private:
        //! diffusion tensor
        std::vector<Core::LinAlg::Matrix<nsd, nsd>> permeabilitytensors_;
        //! relative diffusivities
        std::vector<double> relpermeabilities_;
        //! derivative of relative permeabilities w.r.t. saturation
        std::vector<double> derrelpermeabilities_;
        //! check for constant relative permeabilities of phase
        std::vector<bool> constrelpermeability_;
        //! check for constant dynamic viscosities of phase
        std::vector<bool> constdynviscosity_;
        //! permeability tensors
        std::vector<Core::LinAlg::Matrix<nsd, nsd>> permeabilitytensorsvolfracpress_;
        //! check for constant dynamic viscosities of volume fraction pressure
        std::vector<bool> constdynviscosityvolfracpress_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for additional volume fractions

      This class is a decorator for a phase manager, including evaluation of
      the additional volume fractions, e.g. for endothelial cells
      */
      template <int nsd>
      class PhaseManagerVolFrac : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerVolFrac(std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void setup(const Core::Elements::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void evaluate_gp_state(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void clear_gp_state() override;

        //! @name Access methods

        //! get the diffusion tensor
        void diff_tensor_vol_frac(
            int volfracnum, Core::LinAlg::Matrix<nsd, nsd>& difftensorvolfrac) const override;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool has_add_scalar_dependent_flux(int volfracnum) const override;

        //! return scalar diffusivity of scalar 'iscal' of volume fraction 'volfracnum'
        double scalar_diff(int volfracnum, int iscal) const override;

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double omega_half(int volfracnum, int iscal) const override;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool has_add_scalar_dependent_flux(int volfracnum, int iscal) const override;

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool has_receptor_kinetic_law(int volfracnum, int iscal) const override;

        //@}

       private:
        //! diffusion tensors
        std::vector<Core::LinAlg::Matrix<nsd, nsd>> difftensorsvolfrac_;
        //! does the material have additional scalar dependent flux
        std::vector<bool> hasaddscalardpendentflux_;
        //! matrix for scalar diffusivities
        std::vector<std::vector<double>> scalardiffs_;
        //! matrix for omega half values of receptor-kinetic law
        std::vector<std::vector<double>> omega_half_;
      };


      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/
    }  // namespace PoroFluidManager
  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
