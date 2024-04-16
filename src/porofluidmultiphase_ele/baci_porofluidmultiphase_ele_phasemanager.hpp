/*----------------------------------------------------------------------*/
/*! \file
 \brief manager class for handling the phases and their dofs on element level

   \level 3

 *----------------------------------------------------------------------*/

#ifndef FOUR_C_POROFLUIDMULTIPHASE_ELE_PHASEMANAGER_HPP
#define FOUR_C_POROFLUIDMULTIPHASE_ELE_PHASEMANAGER_HPP

#include "baci_config.hpp"

#include "baci_inpar_material.hpp"
#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_mat_scatra_multiporo.hpp"
#include "baci_porofluidmultiphase_ele_action.hpp"
#include "baci_utils_exceptions.hpp"

#include <Teuchos_RCP.hpp>

#include <vector>

BACI_NAMESPACE_OPEN


// forward declarations
namespace LINALG
{
  class SerialDenseMatrix;
  class SerialDenseVector;
}  // namespace LINALG

namespace MAT
{
  class Material;
  class StructPoro;
  class FluidPoroSinglePhase;
  class FluidPoroMultiPhase;
}  // namespace MAT

namespace DRT
{
  class Element;

  namespace ELEMENTS
  {
    class PoroFluidMultiPhaseEleParameter;

    namespace POROFLUIDMANAGER
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

      The idea is, that there are the methods Setup(...) and EvaluateGPState(..), which
      need to be called before evaluation.
      All other methods are (more or less) constant access methods.

      All implementations are derived from this class.

      Two factory methods (in fact, only one as one calls the other), provide
      the specialization depending on the evaluation action.

      \author vuong
      */

      class PhaseManagerInterface
      {
       public:
        //! constructor
        PhaseManagerInterface(){};

        //! destructor
        virtual ~PhaseManagerInterface() = default;

        //! factory method
        static Teuchos::RCP<DRT::ELEMENTS::POROFLUIDMANAGER::PhaseManagerInterface>
        CreatePhaseManager(const DRT::ELEMENTS::PoroFluidMultiPhaseEleParameter& para, int nsd,
            INPAR::MAT::MaterialType mattype, const POROFLUIDMULTIPHASE::Action& action,
            int totalnumdofpernode, int numfluidphases);

        //! factory method
        static Teuchos::RCP<DRT::ELEMENTS::POROFLUIDMANAGER::PhaseManagerInterface>
        WrapPhaseManager(const DRT::ELEMENTS::PoroFluidMultiPhaseEleParameter& para, int nsd,
            INPAR::MAT::MaterialType mattype, const POROFLUIDMULTIPHASE::Action& action,
            Teuchos::RCP<PhaseManagerInterface> corephasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        virtual void Setup(const DRT::Element* ele, const int matnum = 0) = 0;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) = 0;

        //! clear the states
        virtual void ClearGPState() = 0;

        //! check for reactions
        virtual bool IsReactive(int phasenum) const = 0;

        //! get scalar to phase mapping
        virtual MAT::ScatraMatMultiPoro::ScalarToPhaseMap ScalarToPhase(int iscal) const = 0;

        //! check if EvaluateGPState() was called
        virtual void CheckIsEvaluated() const = 0;

        //! check if Setup() was called
        virtual void CheckIsSetup() const = 0;

        //! @name Access methods

        //! get the number of phases
        virtual int NumFluidPhases() const = 0;

        //! get the total number of dofs (number of phases + 2*number of volume fractions)
        virtual int TotalNumDof() const = 0;

        //! get the total number of volume fractions
        virtual int NumVolFrac() const = 0;

        //! get solid pressure
        virtual double SolidPressure() const = 0;

        //! recalculate solid pressure
        virtual void RecalculateSolidPressure(const double porosity) = 0;

        //! get saturation of phase 'phasenum'
        virtual double Saturation(int phasenum) const = 0;

        //! get pressure of phase 'phasenum'
        virtual double Pressure(int phasenum) const = 0;

        //! get saturation of all phases
        virtual const std::vector<double>& Saturation() const = 0;

        //! get volfracs of all phases
        virtual const std::vector<double>& VolFrac() const = 0;

        //! get volfrac of volfrac 'volfracnum'
        virtual double VolFrac(int volfracnum) const = 0;

        //! get volfrac pressures of all phases
        virtual const std::vector<double>& VolFracPressure() const = 0;

        //! get volfrac pressure of volfrac 'volfracnum'
        virtual double VolFracPressure(int volfracnum) const = 0;

        //! get sum of additional volume fractions
        virtual double SumAddVolFrac() const = 0;

        //! get pressure of all phases
        virtual const std::vector<double>& Pressure() const = 0;

        //! get bulk modulus of phase 'phasenum'
        virtual double InvBulkmodulus(int phasenum) const = 0;

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        virtual bool IncompressibleFluidPhase(int phasenum) const = 0;

        //! get inverse bulk modulus of solid phase
        virtual double InvBulkmodulusSolid() const = 0;

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        virtual bool IncompressibleSolid() const = 0;

        //! get density of phase 'phasenum'
        virtual double Density(int phasenum) const = 0;

        //! get density of phase 'phasenum'
        virtual double VolFracDensity(int volfracnum) const = 0;

        //! get the density of solid phase
        virtual double SolidDensity() const = 0;

        //! get the current element the manager was set up with
        virtual const DRT::Element* Element() const = 0;

        //! get porosity
        virtual double Porosity() const = 0;

        //! get Jacobian of deformation gradient
        virtual double JacobianDefGrad() const = 0;

        //! get derivative of porosity wrt JacobianDefGrad
        virtual double PorosityDerivWrtJacobianDefGrad() const = 0;

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        virtual double PorosityDeriv(int doftoderive) const = 0;

        //! check if porosity depends on fluid (pressure)
        virtual bool PorosityDependsOnFluid() const = 0;

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        virtual bool PorosityDependsOnStruct() const = 0;

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double SaturationDeriv(int phasenum, int doftoderive) const = 0;

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double SaturationDerivDeriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const = 0;

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        virtual double PressureDeriv(int phasenum, int doftoderive) const = 0;

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        virtual double SolidPressureDeriv(int doftoderive) const = 0;

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        virtual double SolidPressureDerivDeriv(int doftoderive, int doftoderive2) const = 0;

        //! get the reaction term
        virtual double ReacTerm(int phasenum) const = 0;

        //! get the derivative of the reaction term
        virtual double ReacDeriv(int phasenum, int doftoderive) const = 0;

        //! get total number of scalars in system
        virtual int NumScal() const = 0;

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        virtual double ReacDerivScalar(int phasenum, int scaltoderive) const = 0;

        //! get the derivative of the reaction term w.r.t. porosity
        virtual double ReacDerivPorosity(int phasenum) const = 0;

        //! get the diffusion tensor
        virtual void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<3, 3>& permeabilitytensor) const = 0;
        //! get the diffusion tensor
        virtual void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<2, 2>& permeabilitytensor) const = 0;
        //! get the diffusion tensor
        virtual void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<1, 1>& permeabilitytensor) const = 0;

        //! check for constant relpermeability
        virtual bool HasConstantRelPermeability(int phasenum) const = 0;
        //! get relative diffusivity of phase
        virtual double RelPermeability(int phasenum) const = 0;
        //! get derivative of relative permeability of phase
        virtual double RelPermeabilityDeriv(int phasenum) const = 0;

        //! check for constant dynamic viscosity
        virtual bool HasConstantDynViscosity(int phasenum) const = 0;

        //! get dynamic viscosity of phase (matnum is the material number of the porofluid-material
        //! on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual double DynViscosity(int phasenum, double abspressgrad, int matnum = 0) const = 0;
        //! get dynamic viscosity of phase
        virtual double DynViscosity(
            const MAT::Material& material, int phasenum, double abspressgrad) const = 0;
        //! get derivative of dynamic viscosity of phase
        virtual double DynViscosityDeriv(int phasenum, double abspressgrad) const = 0;
        //! get derivative dynamic viscosity of phase
        virtual double DynViscosityDeriv(
            const MAT::Material& material, int phasenum, double abspressgrad) const = 0;

        //! check for constant dynamic viscosity of volume fraction pressure
        virtual bool HasConstantDynViscosityVolFracPressure(int volfracpressnum) const = 0;

        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        virtual double DynViscosityVolFracPressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const = 0;
        //! get dynamic viscosity of volume fraction pressure
        virtual double DynViscosityVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const = 0;
        //! get derivative of dynamic viscosity of volume fraction pressure
        virtual double DynViscosityDerivVolFracPressure(
            int volfracpressnum, double abspressgrad) const = 0;
        //! get derivative dynamic viscosity of volume fraction pressure
        virtual double DynViscosityDerivVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const = 0;

        //! get the diffusion tensor
        virtual void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<3, 3>& difftensorvolfrac) const = 0;
        //! get the diffusion tensor
        virtual void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<2, 2>& difftensorvolfrac) const = 0;
        //! get the diffusion tensor
        virtual void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<1, 1>& difftensorvolfrac) const = 0;

        //! get the permeability tensor for volume fraction pressures
        virtual void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<3, 3>& permeabilitytensorvolfracpressure) const = 0;
        //! get the permeability tensor for volume fraction pressures
        virtual void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<2, 2>& permeabilitytensorvolfracpressure) const = 0;
        //! get the permeability tensor for volume fraction pressures
        virtual void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<1, 1>& permeabilitytensorvolfracpressure) const = 0;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        virtual bool HasAddScalarDependentFlux(int volfracnum) const = 0;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        virtual bool HasAddScalarDependentFlux(int volfracnum, int iscal) const = 0;

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        virtual bool HasReceptorKineticLaw(int volfracnum, int iscal) const = 0;

        //! return scalar diffusivity of of scalar 'iscal' of volume fraction 'volfracnum'
        virtual double ScalarDiff(int volfracnum, int iscal) const = 0;

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        virtual double OmegaHalf(int volfracnum, int iscal) const = 0;

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

      \author vuong
      */
      class PhaseManagerCore : public PhaseManagerInterface
      {
       public:
        //! constructor
        explicit PhaseManagerCore(int totalnumdofpernode, int numfluidphases);

        //! copy constructor
        PhaseManagerCore(const PhaseManagerCore& old);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void Setup(const DRT::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! check for reactions
        bool IsReactive(int phasenum) const override { return false; };

        //! @name Access methods

        //! get the number of phases
        int NumFluidPhases() const override { return numfluidphases_; };

        //! get the total number of dofs (number of phases + 2*number of volume fractions)
        int TotalNumDof() const override { return totalnumdofpernode_; };

        //! get the number of volume fractions
        int NumVolFrac() const override { return numvolfrac_; };

        //! get solid pressure
        double SolidPressure() const override;

        //! recalculate solid pressure
        void RecalculateSolidPressure(const double porosity) override;

        //! get saturation of phase 'phasenum'
        double Saturation(int phasenum) const override;

        //! get pressure of phase 'phasenum'
        double Pressure(int phasenum) const override;

        //! get saturation of all phases
        const std::vector<double>& Saturation() const override;

        //! get volfracs of all phases
        const std::vector<double>& VolFrac() const override;

        //! get volfrac of volfrac 'volfracnum'
        double VolFrac(int volfracnum) const override;

        //! get volfrac pressures of all volfracs
        const std::vector<double>& VolFracPressure() const override;

        //! get volfrac pressure of volfrac 'volfracnum'
        double VolFracPressure(int volfracnum) const override;

        //! get sum of additional volume fractions
        double SumAddVolFrac() const override;

        //! get pressure of all phases
        const std::vector<double>& Pressure() const override;

        //! get bulk modulus of phase 'phasenum'
        double InvBulkmodulus(int phasenum) const override;

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        bool IncompressibleFluidPhase(int phasenum) const override;

        //! get inverse bulk modulus of solid phase
        double InvBulkmodulusSolid() const override;

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        bool IncompressibleSolid() const override;

        //! get density of phase 'phasenum'
        double Density(int phasenum) const override;

        //! get density of phase 'phasenum'
        double VolFracDensity(int volfracnum) const override;

        //! get the density of the solid phase
        double SolidDensity() const override;

        //! get the current element the manager was set up with
        const DRT::Element* Element() const override { return ele_; };

        //@}

        //! check if EvaluateGPState() was called
        void CheckIsEvaluated() const override
        {
          if (not isevaluated_) dserror("Gauss point states have not been set!");
        }

        //! check if EvaluateGPState() was called
        void CheckIsSetup() const override
        {
          if (not issetup_) dserror("Setup() was not called!");
        }

        //! get porosity
        double Porosity() const override
        {
          dserror("Porosity not available for this phase manager!");
          return 0.0;
        };

        //! get porosity
        double JacobianDefGrad() const override
        {
          dserror("JacobianDefGrad not available for this phase manager!");
          return 0.0;
        };

        //! get derivative of porosity wrt JacobianDefGrad
        double PorosityDerivWrtJacobianDefGrad() const override
        {
          dserror(
              "Derivative of Porosity w.r.t. JacobianDefGrad not available for this phase "
              "manager!");
          return 0.0;
        };

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double PorosityDeriv(int doftoderive) const override
        {
          dserror("Derivative of porosity not available for this phase manager!");
          return 0.0;
        };

        //! check if porosity depends on fluid (pressure)
        bool PorosityDependsOnFluid() const override
        {
          dserror("PorosityDependsOnFluid() not available for this phase manager!");
          return false;
        };

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool PorosityDependsOnStruct() const override
        {
          dserror("PorosityDependsOnStruct() not available for this phase manager!");
          return false;
        };

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double SaturationDeriv(int phasenum, int doftoderive) const override
        {
          dserror("Derivative of saturation not available for this phase manager!");
          return 0.0;
        };

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double SaturationDerivDeriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override
        {
          dserror("2nd Derivative of saturation not available for this phase manager!");
          return 0.0;
        };

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double PressureDeriv(int phasenum, int doftoderive) const override
        {
          dserror("Derivative of pressure not available for this phase manager!");
          return 0.0;
        };

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double SolidPressureDeriv(int doftoderive) const override
        {
          dserror("Derivative of solid pressure not available for this phase manager!");
          return 0.0;
        };

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double SolidPressureDerivDeriv(int doftoderive, int doftoderive2) const override
        {
          dserror("Second derivative of solid pressure not available for this phase manager!");
          return 0.0;
        };

        //! get the reaction term
        double ReacTerm(int phasenum) const override
        {
          dserror("Reaction term not available for this phase manager!");
          return 0.0;
        };

        //! get total number of scalars in system
        int NumScal() const override
        {
          dserror("Number of scalars not available for this phase manager");
          return 0;
        };

        //! get the derivative of the reaction term
        double ReacDeriv(int phasenum, int doftoderive) const override
        {
          dserror("Reaction term derivative not available for this phase manager!");
          return 0.0;
        };

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double ReacDerivScalar(int phasenum, int scaltoderive) const override
        {
          dserror("Reaction term derivative (scalar) not available for this phase manager!");
          return 0.0;
        };

        //! get the derivative of the reaction term w.r.t. porosity
        double ReacDerivPorosity(int phasenum) const override
        {
          dserror("Reaction term derivative (porosity) not available for this phase manager!");
          return 0.0;
        };

        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<3, 3>& permeabilitytensor) const override
        {
          dserror("Diffusion tensor (3D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<2, 2>& permeabilitytensor) const override
        {
          dserror("Diffusion tensor (2D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<1, 1>& permeabilitytensor) const override
        {
          dserror("Diffusion tensor (1D) not available for this phase manager!");
        };

        //! check for constant relpermeability
        bool HasConstantRelPermeability(int phasenum) const override
        {
          dserror("Check for Constant Relative Permeability not available for this phase manager!");
          return false;
        };
        //! get relative diffusivity of phase
        double RelPermeability(int phasenum) const override
        {
          dserror("Relative Diffusivity not available for this phase manager!");
          return 0.0;
        };
        //! get derivative of relative permeability of phase
        double RelPermeabilityDeriv(int phasenum) const override
        {
          dserror("Derivativ of relativ permeability not available for this phase manager!");
          return 0.0;
        };

        //! check for constant dynamic viscosity
        bool HasConstantDynViscosity(int phasenum) const override
        {
          dserror("Check for Constant Dynamic Viscosity not available for this phase manager!");
          return false;
        };
        //! get relative diffusivity of phase
        double DynViscosity(int phasenum, double abspressgrad, int matnum = 0) const override
        {
          dserror("Dynamic Viscosity not available for this phase manager!");
          return 0.0;
        };
        //! get dynamic viscosity of phase
        double DynViscosity(
            const MAT::Material& material, int phasenum, double abspressgrad) const override
        {
          dserror("Dynamic Viscosity not available for this phase manager!");
          return 0.0;
        };
        //! get derivative of dynamic viscosity of phase
        double DynViscosityDeriv(int phasenum, double abspressgrad) const override
        {
          dserror("Derivative of dynamic Viscosity not available for this phase manager!");
          return 0.0;
        };
        //! get derivative dynamic viscosity of phase
        double DynViscosityDeriv(
            const MAT::Material& material, int phasenum, double abspressgrad) const override
        {
          dserror("Derivative of dynamic Viscosity not available for this phase manager!");
          return 0.0;
        };

        //! check for constant dynamic viscosity of volume fraction pressure
        bool HasConstantDynViscosityVolFracPressure(int volfracpressnum) const override
        {
          dserror(
              "Check for Constant Dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
          return false;
        };
        //! get relative diffusivity of volume fraction pressure
        double DynViscosityVolFracPressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          dserror("Dynamic Viscosity (VolFracPressure) not available for this phase manager!");
          return 0.0;
        };
        //! get dynamic viscosity of volume fraction pressure
        double DynViscosityVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override
        {
          dserror("Dynamic Viscosity (VolFracPressure) not available for this phase manager!");
          return 0.0;
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            int volfracpressnum, double abspressgrad) const override
        {
          dserror(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
          return 0.0;
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override
        {
          dserror(
              "Derivative of dynamic Viscosity (VolFracPressure) not available for this phase "
              "manager!");
          return 0.0;
        };

        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<3, 3>& difftensorvolfrac) const override
        {
          dserror(
              "Diffusion tensor for volume fractions (3D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<2, 2>& difftensorvolfrac) const override
        {
          dserror(
              "Diffusion tensor for volume fractions (2D) not available for this phase manager!");
        };
        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<1, 1>& difftensorvolfrac) const override
        {
          dserror(
              "Diffusion tensor for volume fractions (1D) not available for this phase manager!");
        };

        //! get the permeabilty tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<3, 3>& permeabilitytensorvolfracpressure) const override
        {
          dserror(
              "Permeability tensor for volume fraction pressures (3D) not available for this phase "
              "manager!");
        };
        //! get the permeabilty tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<2, 2>& permeabilitytensorvolfracpressure) const override
        {
          dserror(
              "Permeability tensor for volume fraction pressures (2D) not available for this phase "
              "manager!");
        };
        //! get the permeabilty tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<1, 1>& permeabilitytensorvolfracpressure) const override
        {
          dserror(
              "Permeability tensor for volume fraction pressures (1D) not available for this phase "
              "manager!");
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool HasAddScalarDependentFlux(int volfracnum) const override
        {
          dserror("HasAddScalarDependentFlux not available for this phase manager!");
          return false;
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool HasAddScalarDependentFlux(int volfracnum, int iscal) const override
        {
          dserror("HasAddScalarDependentFlux not available for this phase manager!");
          return false;
        };

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool HasReceptorKineticLaw(int volfracnum, int iscal) const override
        {
          dserror("HasReceptorKineticLaw not available for this phase manager!");
          return false;
        };

        //! return scalar diffusivities of scalar 'iscal' of volume fraction 'volfracnum'
        double ScalarDiff(int volfracnum, int iscal) const override
        {
          dserror("ScalarDiff not available for this phase manager!");
          return 0.0;
        };

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double OmegaHalf(int volfracnum, int iscal) const override
        {
          dserror("OmegaHalf not available for this phase manager!");
          return 0.0;
        };

        //! get scalar to phase mapping
        MAT::ScatraMatMultiPoro::ScalarToPhaseMap ScalarToPhase(int iscal) const override
        {
          dserror("ScalarToPhase not available for this phase manager!");
          MAT::ScatraMatMultiPoro::ScalarToPhaseMap null_map;
          return null_map;
        };

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
        const DRT::Element* ele_;

        //! flag indicating of gauss point state has been set and evaluated
        bool isevaluated_;

        //! flag of Setup was called
        bool issetup_;
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

      \author vuong
      */
      class PhaseManagerDecorator : public PhaseManagerInterface
      {
       public:
        //! constructor
        explicit PhaseManagerDecorator(
            Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager)
            : phasemanager_(phasemanager){};

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void Setup(const DRT::Element* ele, const int matnum = 0) override
        {
          phasemanager_->Setup(ele, matnum);
        };

        //! check if EvaluateGPState() was called
        void CheckIsEvaluated() const override { phasemanager_->CheckIsEvaluated(); };

        //! check if Setup() was called
        void CheckIsSetup() const override { phasemanager_->CheckIsSetup(); };

        //! @name Access methods

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double PorosityDeriv(int doftoderive) const override
        {
          return phasemanager_->PorosityDeriv(doftoderive);
        };

        //! check if porosity depends on fluid (pressure)
        bool PorosityDependsOnFluid() const override
        {
          return phasemanager_->PorosityDependsOnFluid();
        };

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool PorosityDependsOnStruct() const override
        {
          return phasemanager_->PorosityDependsOnStruct();
        };

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double SaturationDeriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->SaturationDeriv(phasenum, doftoderive);
        };

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double SaturationDerivDeriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override
        {
          return phasemanager_->SaturationDerivDeriv(phasenum, firstdoftoderive, seconddoftoderive);
        };

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double PressureDeriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->PressureDeriv(phasenum, doftoderive);
        };

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double SolidPressureDeriv(int doftoderive) const override
        {
          return phasemanager_->SolidPressureDeriv(doftoderive);
        };

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double SolidPressureDerivDeriv(int doftoderive, int doftoderive2) const override
        {
          return phasemanager_->SolidPressureDerivDeriv(doftoderive, doftoderive2);
        };

        //! check if the current phase is involved in a reaction
        bool IsReactive(int phasenum) const override
        {
          return phasemanager_->IsReactive(phasenum);
        };

        //! get scalar to phase mapping
        MAT::ScatraMatMultiPoro::ScalarToPhaseMap ScalarToPhase(int iscal) const override
        {
          return phasemanager_->ScalarToPhase(iscal);
        }

        //! get the number of phases
        int NumFluidPhases() const override { return phasemanager_->NumFluidPhases(); };

        //! get the number of dofs (number of phases + 2*number of volfracs)
        int TotalNumDof() const override { return phasemanager_->TotalNumDof(); };

        //! get the number of volume fractions
        int NumVolFrac() const override { return phasemanager_->NumVolFrac(); };

        //! get solid pressure
        double SolidPressure() const override { return phasemanager_->SolidPressure(); };

        //! recalculate solid pressure
        void RecalculateSolidPressure(const double porosity) override
        {
          return phasemanager_->RecalculateSolidPressure(porosity);
        };

        //! get saturation of phase 'phasenum'
        double Saturation(int phasenum) const override
        {
          return phasemanager_->Saturation(phasenum);
        };

        //! get pressure of phase 'phasenum'
        double Pressure(int phasenum) const override { return phasemanager_->Pressure(phasenum); };

        //! get saturation of all phases
        const std::vector<double>& Saturation() const override
        {
          return phasemanager_->Saturation();
        };

        //! get volfracs of all phases
        const std::vector<double>& VolFrac() const override { return phasemanager_->VolFrac(); };

        //! get volfrac of volfrac 'volfracnum'
        double VolFrac(int volfracnum) const override
        {
          return phasemanager_->VolFrac(volfracnum);
        };

        //! get volfrac pressures of all phases
        const std::vector<double>& VolFracPressure() const override
        {
          return phasemanager_->VolFracPressure();
        };

        //! get volfrac pressure of volfrac 'volfracnum'
        double VolFracPressure(int volfracnum) const override
        {
          return phasemanager_->VolFracPressure(volfracnum);
        };

        //! get solid pressure
        double SumAddVolFrac() const override { return phasemanager_->SumAddVolFrac(); };

        //! get pressure of all phases
        const std::vector<double>& Pressure() const override { return phasemanager_->Pressure(); };

        //! get bulk modulus of phase 'phasenum'
        double InvBulkmodulus(int phasenum) const override
        {
          return phasemanager_->InvBulkmodulus(phasenum);
        };

        //! check if fluid phase 'phasenum' is incompressible (very low compressibility < 1e-14)
        bool IncompressibleFluidPhase(int phasenum) const override
        {
          return phasemanager_->IncompressibleFluidPhase(phasenum);
        };

        //! get inverse bulk modulus of solid phase
        double InvBulkmodulusSolid() const override
        {
          return phasemanager_->InvBulkmodulusSolid();
        };

        //! check if solid is incompressible (either very low compressibility < 1e-14 or
        //! MAT_PoroLawConstant)
        bool IncompressibleSolid() const override { return phasemanager_->IncompressibleSolid(); };

        //! get porosity
        double Porosity() const override { return phasemanager_->Porosity(); };

        //! get JacobianDefGrad
        double JacobianDefGrad() const override { return phasemanager_->JacobianDefGrad(); };

        //! get derivative of porosity wrt JacobianDefGrad
        double PorosityDerivWrtJacobianDefGrad() const override
        {
          return phasemanager_->PorosityDerivWrtJacobianDefGrad();
        };

        //! get density of phase 'phasenum'
        double Density(int phasenum) const override { return phasemanager_->Density(phasenum); };

        //! get density of volume fraction 'volfracnum'
        double VolFracDensity(int volfracnum) const override
        {
          return phasemanager_->VolFracDensity(volfracnum);
        };

        //! get density of solid phase
        double SolidDensity() const override { return phasemanager_->SolidDensity(); };

        //! get the current element the manager was set up with
        const DRT::Element* Element() const override { return phasemanager_->Element(); };

        //! get the reaction term
        double ReacTerm(int phasenum) const override { return phasemanager_->ReacTerm(phasenum); };

        //! get total number of scalars in system
        int NumScal() const override { return phasemanager_->NumScal(); };

        //! get the derivative of the reaction term
        double ReacDeriv(int phasenum, int doftoderive) const override
        {
          return phasemanager_->ReacDeriv(phasenum, doftoderive);
        };

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double ReacDerivScalar(int phasenum, int scaltoderive) const override
        {
          return phasemanager_->ReacDerivScalar(phasenum, scaltoderive);
        };

        //! get the derivative of the reaction term w.r.t. porosity
        double ReacDerivPorosity(int phasenum) const override
        {
          return phasemanager_->ReacDerivPorosity(phasenum);
        };

        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<3, 3>& permeabilitytensor) const override
        {
          phasemanager_->PermeabilityTensor(phasenum, permeabilitytensor);
        };
        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<2, 2>& permeabilitytensor) const override
        {
          phasemanager_->PermeabilityTensor(phasenum, permeabilitytensor);
        };
        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<1, 1>& permeabilitytensor) const override
        {
          phasemanager_->PermeabilityTensor(phasenum, permeabilitytensor);
        };

        //! check for constant relpermeability
        bool HasConstantRelPermeability(int phasenum) const override
        {
          return phasemanager_->HasConstantRelPermeability(phasenum);
        };

        //! get relative diffusivity of phase
        double RelPermeability(int phasenum) const override
        {
          return phasemanager_->RelPermeability(phasenum);
        };

        //! get derivative of relative permeability of phase
        double RelPermeabilityDeriv(int phasenum) const override
        {
          return phasemanager_->RelPermeabilityDeriv(phasenum);
        };

        //! check for constant dynamic visosity
        bool HasConstantDynViscosity(int phasenum) const override
        {
          return phasemanager_->HasConstantDynViscosity(phasenum);
        };
        //! get dynamic viscosity of phase
        double DynViscosity(int phasenum, double abspressgrad, int matnum = 0) const override
        {
          return phasemanager_->DynViscosity(phasenum, abspressgrad, matnum = 0);
        };
        //! get dynamic viscosity of phase
        double DynViscosity(
            const MAT::Material& material, int phasenum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosity(material, phasenum, abspressgrad);
        };
        //! get derivative of dynamic viscosity of phase
        double DynViscosityDeriv(int phasenum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosityDeriv(phasenum, abspressgrad);
        };
        //! get derivative dynamic viscosity of phase
        double DynViscosityDeriv(
            const MAT::Material& material, int phasenum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosityDeriv(material, phasenum, abspressgrad);
        };

        //! check for constant dynamic visosity of volume fraction pressure
        bool HasConstantDynViscosityVolFracPressure(int volfracpressnum) const override
        {
          return phasemanager_->HasConstantDynViscosityVolFracPressure(volfracpressnum);
        };
        //! get dynamic viscosity of volume fraction pressure
        double DynViscosityVolFracPressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override
        {
          return phasemanager_->DynViscosityVolFracPressure(
              volfracpressnum, abspressgrad, matnum = 0);
        };
        //! get dynamic viscosity of volume fraction pressure
        double DynViscosityVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosityVolFracPressure(
              material, volfracpressnum, abspressgrad);
        };
        //! get derivative of dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosityDerivVolFracPressure(volfracpressnum, abspressgrad);
        };
        //! get derivative dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override
        {
          return phasemanager_->DynViscosityDerivVolFracPressure(
              material, volfracpressnum, abspressgrad);
        };

        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<3, 3>& difftensorvolfrac) const override
        {
          phasemanager_->DiffTensorVolFrac(volfracnum, difftensorvolfrac);
        };
        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<2, 2>& difftensorvolfrac) const override
        {
          phasemanager_->DiffTensorVolFrac(volfracnum, difftensorvolfrac);
        };
        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<1, 1>& difftensorvolfrac) const override
        {
          phasemanager_->DiffTensorVolFrac(volfracnum, difftensorvolfrac);
        };

        //! get the permeability tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<3, 3>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->PermeabilityTensorVolFracPressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };
        //! get the permeability tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<2, 2>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->PermeabilityTensorVolFracPressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };
        //! get the permeability tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<1, 1>& permeabilitytensorvolfracpressure) const override
        {
          phasemanager_->PermeabilityTensorVolFracPressure(
              volfracpressnum, permeabilitytensorvolfracpressure);
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool HasAddScalarDependentFlux(int volfracnum) const override
        {
          return phasemanager_->HasAddScalarDependentFlux(volfracnum);
        };

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool HasAddScalarDependentFlux(int volfracnum, int iscal) const override
        {
          return phasemanager_->HasAddScalarDependentFlux(volfracnum, iscal);
        };

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool HasReceptorKineticLaw(int volfracnum, int iscal) const override
        {
          return phasemanager_->HasReceptorKineticLaw(volfracnum, iscal);
        };

        //! return scalar diffusivities of scalar 'iscal' of volume fraction 'volfracnum'
        double ScalarDiff(int volfracnum, int iscal) const override
        {
          return phasemanager_->ScalarDiff(volfracnum, iscal);
        };

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double OmegaHalf(int volfracnum, int iscal) const override
        {
          return phasemanager_->OmegaHalf(volfracnum, iscal);
        };

        //@}

       protected:
        //! wrapped phase manager
        Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for derivatives

      This class is a decorator for a phase manager, including the derivatives
      of the pressures and saturations w.r.t. to the primary variables

      \author vuong
      */
      class PhaseManagerDeriv : public PhaseManagerDecorator
      {
       public:
        //! constructor
        explicit PhaseManagerDeriv(
            Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager);

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! @name Access methods

        //! get derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double SaturationDeriv(int phasenum, int doftoderive) const override;

        //! get 2nd derivative of saturation of phase 'phasenum' w.r.t. DOF 'doftoderive'
        //! (dS/dphi_{ii})
        double SaturationDerivDeriv(
            int phasenum, int firstdoftoderive, int seconddoftoderive) const override;

        //! get derivative of pressure of phase 'phasenum' w.r.t. DOF 'doftoderive'
        double PressureDeriv(int phasenum, int doftoderive) const override;

        //! get derivative of solid pressure  w.r.t. DOF 'doftoderive'
        double SolidPressureDeriv(int doftoderive) const override;

        //! get derivative of pressure of phase 'phasenum'
        //! w.r.t. DOF 'doftoderive' (first derivative)
        //! and w.r.t. DOF 'doftoderive2' (second derivative)
        double SolidPressureDerivDeriv(int doftoderive, int doftoderive2) const override;

        //@}

       private:
        //! derivative of true pressure w.r.t. degrees of freedom
        // first index: pressure, second index: dof
        Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> pressurederiv_;
        //! derivative of saturations w.r.t. degrees of freedom
        // first index: saturation, second index: dof
        Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> saturationderiv_;
        //! second derivative of saturations w.r.t. degrees of freedom
        Teuchos::RCP<std::vector<CORE::LINALG::SerialDenseMatrix>> saturationderivderiv_;

        //! derivative of solid pressure w.r.t. degrees of freedom
        Teuchos::RCP<CORE::LINALG::SerialDenseVector> solidpressurederiv_;
        //! second derivative of solid pressure w.r.t. degrees of freedom;
        Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> solidpressurederivderiv_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for derivatives and porosity

      This class is a decorator for a phase manager, including evaluation of
      the pressures and saturations w.r.t. to the primary variables and the porosity.

      \author vuong
      */
      class PhaseManagerDerivAndPorosity : public PhaseManagerDeriv
      {
       public:
        //! constructor
        explicit PhaseManagerDerivAndPorosity(
            Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager);

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! @name Access methods

        //! get porosity
        double Porosity() const override;

        //! get porosity
        double JacobianDefGrad() const override;

        //! get derivative of porosity wrt JacobianDefGrad
        double PorosityDerivWrtJacobianDefGrad() const override;

        //! get derivative of porosity w.r.t. DOF 'doftoderive'
        double PorosityDeriv(int doftoderive) const override;

        //! check if porosity depends on fluid (pressure)
        bool PorosityDependsOnFluid() const override;

        //! check if porosity depends on structure (basically Jacobian of def. gradient)
        bool PorosityDependsOnStruct() const override;

        //@}

       private:
        //! porosity
        double porosity_;

        //! Jacobian of def gradient
        double J_;

        //! derivative of porosity w.r.t. Jacobian of defgradient
        double dporosity_dJ_;

        //! derivative of porosity w.r.t. solid pressure
        double dporosity_dp_;

        //! derivative of porosity w.r.t. degrees of freedom
        Teuchos::RCP<CORE::LINALG::SerialDenseVector> porosityderiv_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief wrapper class, extensions for reaction/mass exchange terms

      This class is a decorator for a phase manager, including evaluation of
      reaction/mass exchange terms.

      \author vuong
      */
      class PhaseManagerReaction : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerReaction(Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void Setup(const DRT::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! @name Access methods

        //! get the reaction term
        double ReacTerm(int phasenum) const override;

        //! get the derivative of the reaction term
        double ReacDeriv(int phasenum, int doftoderive) const override;

        //! check if the current phase is involved in a reaction
        bool IsReactive(int phasenum) const override
        {
          CheckIsSetup();
          return isreactive_[phasenum];
        };

        //! get scalar to phase mapping
        MAT::ScatraMatMultiPoro::ScalarToPhaseMap ScalarToPhase(int iscal) const override;

        //! get total number of scalars in system
        int NumScal() const override;

        //! get the derivative of the reaction term w.r.t. scalar 'scaltoderive'
        double ReacDerivScalar(int phasenum, int scaltoderive) const override;

        //! get the derivative of the reaction term w.r.t. porosity
        double ReacDerivPorosity(int phasenum) const override;

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
        std::vector<MAT::ScatraMatMultiPoro::ScalarToPhaseMap> scalartophasemap_;
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
      as a CORE::LINALG::Matrix, it is templated by the number of space dimensions

      \author vuong
      */
      template <int nsd>
      class PhaseManagerDiffusion : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerDiffusion(Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void Setup(const DRT::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! @name Access methods

        //! get the diffusion tensor
        void PermeabilityTensor(
            int phasenum, CORE::LINALG::Matrix<nsd, nsd>& permeabilitytensor) const override;

        //! check for constant relpermeability
        bool HasConstantRelPermeability(int phasenum) const override;

        //! get relative diffusivity of phase
        double RelPermeability(int phasenum) const override;

        //! get derivative of relative permeability of phase
        double RelPermeabilityDeriv(int phasenum) const override;

        //! check for constant dynamic viscosity
        bool HasConstantDynViscosity(int phasenum) const override;
        //! get dynamic viscosity of phase
        double DynViscosity(int phasenum, double abspressgrad, int matnum = 0) const override;
        //! get dynamic viscosity of phase
        double DynViscosity(
            const MAT::Material& material, int phasenum, double abspressgrad) const override;
        //! get derivative of dynamic viscosity of phase
        double DynViscosityDeriv(int phasenum, double abspressgrad) const override;
        //! get derivative dynamic viscosity of phase
        double DynViscosityDeriv(
            const MAT::Material& material, int phasenum, double abspressgrad) const override;

        //! get the permeability tensor for volume fraction pressures
        void PermeabilityTensorVolFracPressure(int volfracpressnum,
            CORE::LINALG::Matrix<nsd, nsd>& permeabilitytensorvolfracpressure) const override;

        //! check for constant dynamic viscosity of volume fraction pressure
        bool HasConstantDynViscosityVolFracPressure(int volfracpressnum) const override;

        //! get dynamic viscosity of volume fraction (matnum is the material number of the
        //! porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        double DynViscosityVolFracPressure(
            int volfracpressnum, double abspressgrad, int matnum = 0) const override;
        //! get dynamic viscosity of volume fraction pressure
        double DynViscosityVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override;
        //! get derivative of dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            int volfracpressnum, double abspressgrad) const override;
        //! get derivative dynamic viscosity of volume fraction pressure
        double DynViscosityDerivVolFracPressure(
            const MAT::Material& material, int volfracpressnum, double abspressgrad) const override;

        //@}

       private:
        //! diffusion tensor
        std::vector<CORE::LINALG::Matrix<nsd, nsd>> permeabilitytensors_;
        //! relative diffusivities
        std::vector<double> relpermeabilities_;
        //! derivative of relative permeabilities w.r.t. saturation
        std::vector<double> derrelpermeabilities_;
        //! check for constant relative permeabilities of phase
        std::vector<bool> constrelpermeability_;
        //! check for constant dynamic viscosities of phase
        std::vector<bool> constdynviscosity_;
        //! permeability tensors
        std::vector<CORE::LINALG::Matrix<nsd, nsd>> permeabilitytensorsvolfracpress_;
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

      \author vuong
      */
      template <int nsd>
      class PhaseManagerVolFrac : public PhaseManagerDecorator
      {
       public:
        //! constructor
        PhaseManagerVolFrac(Teuchos::RCP<POROFLUIDMANAGER::PhaseManagerInterface> phasemanager);

        //! setup (matnum is the material number of the porofluid-material on the current element)
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void Setup(const DRT::Element* ele, const int matnum = 0) override;

        //! evaluate pressures, saturations and derivatives at GP (matnum is the material number of
        //! the porofluid-material on the current element) default is set to zero, if called from a
        //! porofluidmultiphase-element otherwise it has to be explicitly passed from the caller
        void EvaluateGPState(
            double J, const VariableManagerMinAccess& varmanager, const int matnum = 0) override;

        //! clear the states
        void ClearGPState() override;

        //! @name Access methods

        //! get the diffusion tensor
        void DiffTensorVolFrac(
            int volfracnum, CORE::LINALG::Matrix<nsd, nsd>& difftensorvolfrac) const override;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux
        bool HasAddScalarDependentFlux(int volfracnum) const override;

        //! return scalar diffusivity of scalar 'iscal' of volume fraction 'volfracnum'
        double ScalarDiff(int volfracnum, int iscal) const override;

        //! return omega half of scalar 'iscal' of volume fraction 'volfracnum' for receptor kinetic
        //! law see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        double OmegaHalf(int volfracnum, int iscal) const override;

        //! check if volume frac 'volfracnum' has additional scalar dependent flux of scalar 'iscal'
        bool HasAddScalarDependentFlux(int volfracnum, int iscal) const override;

        //! check if volume frac 'volfracnum' has receptor kinetic-law of scalar 'iscal'
        //! see: Anderson, A. R. A. & Chaplain, M. A. J.
        //       Continuous and discrete mathematical models of tumor-induced angiogenesis
        bool HasReceptorKineticLaw(int volfracnum, int iscal) const override;

        //@}

       private:
        //! diffusion tensors
        std::vector<CORE::LINALG::Matrix<nsd, nsd>> difftensorsvolfrac_;
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
    }  // namespace POROFLUIDMANAGER
  }    // namespace ELEMENTS
}  // namespace DRT


BACI_NAMESPACE_CLOSE

#endif
