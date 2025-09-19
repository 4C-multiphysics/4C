// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_VARIABLEMANAGER_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_VARIABLEMANAGER_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class Material;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class PoroFluidMultiPhaseEleParameter;

    namespace PoroFluidManager
    {
      template <int, int>
      class VariableManagerInterface;

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/
      /*!
      \brief general interface to variable manager for minimal access (non template)

      These classes manage all accesses to primary variables
       (in contrast to the phase manager, which manages values of secondary).

      \note The 'phi' vector denotes the 'generic' primary variable, which
            can be a pressure, saturation or a differential pressure. This
            does not (and needs not) to differentiate between the actual choice
            of primary variables (this is the responsibility of the phase manager).

      This class is a pure virtual non-template class, which provides access to
      the generic state phi and the scalar state (coming from a ScaTra coupling).
      */
      class VariableManagerMinAccess
      {
       public:
        //! constructor
        VariableManagerMinAccess() {};

        //! destructor
        virtual ~VariableManagerMinAccess() = default;

        virtual const std::vector<double>* phinp() const = 0;
        virtual const std::vector<double>* scalarnp() const = 0;
      };


      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief general interface to variable manager (template)

      The idea is, that there are the methods extract_element_and_node_values(...)
      and evaluate_gp_variables(..), which need to be called before evaluation.
      extract_element_and_node_values(...) reads the node values associated with the element
      from the global state vector and evaluate_gp_variables(..) performs the interpolation
      to the gauss points.
      All other methods are (more or less) constant access methods.

      As fixed sized Core::LinAlg::Matrix is used for saving the values, almost
      all variables managers are templated by the number of space dimensions 'nsd'
      and the number of element nodes 'nen'.

      This class is a pure virtual template interface class.
      It implements the factory method, which builds the concrete variable manager
      based on the element action.
      */
      template <int nsd, int nen>
      class VariableManagerInterface : public VariableManagerMinAccess
      {
       public:
        //! constructor
        VariableManagerInterface() {};

        //! factory method
        static std::shared_ptr<VariableManagerInterface<nsd, nen>> create_variable_manager(
            const Discret::Elements::PoroFluidMultiPhaseEleParameter& para,
            const PoroPressureBased::Action& action, std::shared_ptr<Core::Mat::Material> mat,
            const int numdofpernode, const int numfluidphases, const int numvolfrac);

        //! extract element and node values from the discretization
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        virtual void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) = 0;

        //! evaluate variables at gauss point
        virtual void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) = 0;

        //! check if evaluate_gp_variables was called
        virtual void check_is_evaluated() const = 0;
        //! check if extract_element_and_node_values was called
        virtual void check_is_extracted() const = 0;
        //! return number of DOFs pre node
        virtual int num_dof_per_node() const = 0;

        //! @name Access methods
        const std::vector<double>* phinp() const override = 0;
        virtual const Core::LinAlg::Matrix<nen, 1>* element_phinp(const int k) const = 0;
        virtual bool element_has_valid_vol_frac_pressure(const int ivolfrac) const = 0;
        virtual bool element_has_valid_vol_frac_species(const int ivolfrac) const = 0;
        virtual const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_phinp() const = 0;
        virtual const std::vector<double>* phidtnp() const = 0;
        virtual const std::vector<double>* hist() const = 0;
        virtual const Core::LinAlg::Matrix<nsd, 1>* con_velnp() const = 0;
        virtual double div_con_velnp() const = 0;
        virtual const Core::LinAlg::Matrix<nsd, nen>* e_con_velnp() const = 0;
        virtual const Core::LinAlg::Matrix<nsd, 1>* dispnp() const = 0;
        const std::vector<double>* scalarnp() const override = 0;
        virtual const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_scalarnp() const = 0;

        //@}
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  base class for all core variable managers

      This class is the base class for all variable managers. It basically
      implements some safety checks.
      */
      template <int nsd, int nen>
      class VariableManagerBase : public VariableManagerInterface<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerBase(int numdofpernode)
            : VariableManagerInterface<nsd, nen>(),
              numdofpernode_(numdofpernode),
              isextracted_(false),
              isevaluated_(false),
              volfrac_closing_relation_(
                  Mat::PAR::PoroFluidPressureBased::ClosingRelation::undefined) {};

        //! check if evaluate_gp_variables has been called
        void check_is_evaluated() const override
        {
          if (not isevaluated_)
            FOUR_C_THROW("evaluate_gp_variables has not been called on variable manager!");
        };

        //! check if extract_element_and_node_values has been called
        void check_is_extracted() const override
        {
          if (not isextracted_)
            FOUR_C_THROW(
                "extract_element_and_node_values has not been called on variable manager!");
        };

        //! return number of DOFs per node
        int num_dof_per_node() const override { return numdofpernode_; }

        //! @name Access methods (throw error by default)
        const std::vector<double>* phinp() const override
        {
          FOUR_C_THROW("Access method Phinp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const Core::LinAlg::Matrix<nen, 1>* element_phinp(const int k) const override
        {
          FOUR_C_THROW("Access method ElementPhinp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        bool element_has_valid_vol_frac_pressure(const int ivolfrac) const override
        {
          FOUR_C_THROW(
              "Access method element_has_valid_vol_frac_pressure() not implemented! Wrong "
              "VariableManager?");
          return 0.0;
        };
        bool element_has_valid_vol_frac_species(const int ivolfrac) const override
        {
          FOUR_C_THROW(
              "Access method element_has_valid_vol_frac_species() not implemented! Wrong "
              "VariableManager?");
          return 0.0;
        };
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_phinp() const override
        {
          FOUR_C_THROW("Access method GradPhinp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const std::vector<double>* phidtnp() const override
        {
          FOUR_C_THROW("Access method Phidtnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };

        const std::vector<double>* hist() const override
        {
          FOUR_C_THROW("Access method Hist() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const Core::LinAlg::Matrix<nsd, 1>* con_velnp() const override
        {
          FOUR_C_THROW("Access method ConVelnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        double div_con_velnp() const override
        {
          FOUR_C_THROW("Access method DivConVelnp() not implemented! Wrong VariableManager?");
          return 0.0;
        };
        const Core::LinAlg::Matrix<nsd, nen>* e_con_velnp() const override
        {
          FOUR_C_THROW("Access method EConVelnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const Core::LinAlg::Matrix<nsd, 1>* dispnp() const override
        {
          FOUR_C_THROW("Access method Dispnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const std::vector<double>* scalarnp() const override
        {
          FOUR_C_THROW("Access method Salarnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_scalarnp() const override
        {
          FOUR_C_THROW("Access method GradScalarnp() not implemented! Wrong VariableManager?");
          return nullptr;
        };

        //@}

       protected:
        //! number of DOFs per node
        const int numdofpernode_;
        //! flag if ExtractElementAndNodeValues was called
        bool isextracted_;
        //! flag if evaluate_gp_variables was called
        bool isevaluated_;
        //! type of volume fractioin closing relation
        Mat::PAR::PoroFluidPressureBased::ClosingRelation volfrac_closing_relation_{
            Mat::PAR::PoroFluidPressureBased::ClosingRelation::undefined};
      };


      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief   minimal variable manager only holding the generic state vector 'phi'

      This class is the minimal version of a concrete variable managers. It only
      holds the generic state vector 'phi' (this can be e.g. the pressure or
      the saturation).
      */
      template <int nsd, int nen>
      class VariableManagerPhi : public VariableManagerBase<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerPhi(int numdofpernode)
            : VariableManagerBase<nsd, nen>(numdofpernode),
              ephinp_(numdofpernode),
              phinp_(numdofpernode, 0.0) {};

        //! extract node values related to the state vector 'phinp'
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) override;

        //! evaluate state vector at gauss point
        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! access method
        const std::vector<double>* phinp() const override
        {
          this->check_is_evaluated();
          return &phinp_;
        }

       protected:
        const Core::LinAlg::Matrix<nen, 1>* element_phinp(const int k) const override
        {
          this->check_is_extracted();
          return &ephinp_[k];
        }

        //! state variables at t_(n+1) or t_(n+alpha_F)
        std::vector<Core::LinAlg::Matrix<nen, 1>> ephinp_;

        //! scalar at t_(n+1) or t_(n+alpha_F)
        std::vector<double> phinp_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  variable manager only holding the state vector 'phi' and its gradient

      This class only holds the generic state vector 'phi' (this can be e.g. the pressure or
      the saturation) and its gradient.
      */
      template <int nsd, int nen>
      class VariableManagerPhiGradPhi : public VariableManagerPhi<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerPhiGradPhi(int numdofpernode)
            : VariableManagerPhi<nsd, nen>(numdofpernode), gradphi_(numdofpernode) {};

        //! evaluate phi and its gradient at gauss point
        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! access method
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_phinp() const override
        {
          this->check_is_evaluated();
          return &gradphi_;
        }

       private:
        //! spatial gradient of current scalar value
        std::vector<Core::LinAlg::Matrix<nsd, 1>> gradphi_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/
      /*!
      \brief  base class for all decorator/wrapper classes, i.e. extensions to core variable
      managers

      The idea is to use a core variable manager (e.g. VariableManagerPhi) and extend
      it when the evaluation to be performed demands it. For this a decorator
       pattern is used, i.e. this class wraps another variable manager, extending it if necessary.

      This is the base class for all decorators.
      */
      template <int nsd, int nen>
      class VariableManagerDecorator : public VariableManagerInterface<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerDecorator(std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager)
            : VariableManagerInterface<nsd, nen>(), varmanager_(varmanager) {};

        //! @name Access methods
        const std::vector<double>* phinp() const override { return varmanager_->phinp(); };
        const Core::LinAlg::Matrix<nen, 1>* element_phinp(const int k) const override
        {
          return varmanager_->element_phinp(k);
        };
        double div_con_velnp() const override { return varmanager_->div_con_velnp(); };
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_phinp() const override
        {
          return varmanager_->grad_phinp();
        };
        const std::vector<double>* phidtnp() const override { return varmanager_->phidtnp(); };

        const std::vector<double>* hist() const override { return varmanager_->hist(); };
        const Core::LinAlg::Matrix<nsd, 1>* con_velnp() const override
        {
          return varmanager_->con_velnp();
        };
        bool element_has_valid_vol_frac_pressure(const int ivolfrac) const override
        {
          return varmanager_->element_has_valid_vol_frac_pressure(ivolfrac);
        };
        bool element_has_valid_vol_frac_species(const int ivolfrac) const override
        {
          return varmanager_->element_has_valid_vol_frac_species(ivolfrac);
        };
        const Core::LinAlg::Matrix<nsd, nen>* e_con_velnp() const override
        {
          return varmanager_->e_con_velnp();
        };
        const Core::LinAlg::Matrix<nsd, 1>* dispnp() const override
        {
          return varmanager_->dispnp();
        };
        const std::vector<double>* scalarnp() const override { return varmanager_->scalarnp(); };
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_scalarnp() const override
        {
          return varmanager_->grad_scalarnp();
        };

        //@}

        //! check if evaluate_gp_variables was called
        void check_is_evaluated() const override { varmanager_->check_is_evaluated(); }

        //! check if extract_element_and_node_values was called
        void check_is_extracted() const override { varmanager_->check_is_extracted(); }

        //! return number of DOFs per node
        int num_dof_per_node() const override { return varmanager_->num_dof_per_node(); }

       protected:
        //! wrapped variable manager
        std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager_;
      };


      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  decorator/wrapper class for variable manager, extensions to time derivatives

      This class extends a wrapped variable manager by the instationary variables
      (time derivatives, and 'history'-vector)
      */
      template <int nsd, int nen>
      class VariableManagerInstat : public VariableManagerDecorator<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerInstat(std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager)
            : VariableManagerDecorator<nsd, nen>(varmanager),
              ephidtnp_(varmanager->num_dof_per_node()),
              ehist_(varmanager->num_dof_per_node()),
              phidtnp_(varmanager->num_dof_per_node(), 0.0),
              hist_(varmanager->num_dof_per_node(), 0.0) {};

        //! extract node values related to time derivatives
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) override;

        //! evaluate variables at gauss point
        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! @name Access methods

        //! get time derivative of state phi
        const std::vector<double>* phidtnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &phidtnp_;
        }
        //! get history vector
        const std::vector<double>* hist() const override
        {
          this->varmanager_->check_is_evaluated();
          return &hist_;
        }
        //@}

       private:
        //! time derivatives of state variables at t_(n+1)
        std::vector<Core::LinAlg::Matrix<nen, 1>> ephidtnp_;
        //! history vector of transported scalars
        std::vector<Core::LinAlg::Matrix<nen, 1>> ehist_;

        //! time derivative of scalar at t_(n+1)
        std::vector<double> phidtnp_;

        //! history data (or acceleration)
        std::vector<double> hist_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  decorator/wrapper class for variable manager, extensions to moving domain variables

      This class extends a wrapped variable manager by the  variables needed for an ALE
      formulation, i.e. mesh displacements and convective velocity.
      */
      template <int nsd, int nen>
      class VariableManagerStruct : public VariableManagerDecorator<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerStruct(
            int ndsvel, int ndsdisp, std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager)
            : VariableManagerDecorator<nsd, nen>(varmanager),
              ndsvel_(ndsvel),
              ndsdisp_(ndsdisp),
              econvelnp_(Core::LinAlg::Initialization::zero),  // initialized to zero
              edispnp_(Core::LinAlg::Initialization::zero),    // initialized to zero
              divconvelint_(0.0),
              convelint_(Core::LinAlg::Initialization::zero),
              dispint_(Core::LinAlg::Initialization::zero) {};

        //! extract variables related to structure coupling
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) override;

        //! evaluate variables at gauss point
        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! @name Access methods
        const Core::LinAlg::Matrix<nsd, 1>* con_velnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &convelint_;
        }
        const Core::LinAlg::Matrix<nsd, 1>* dispnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &dispint_;
        }
        double div_con_velnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return divconvelint_;
        }
        const Core::LinAlg::Matrix<nsd, nen>* e_con_velnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &econvelnp_;
        }

        //@}

       private:
        // number of dofset associated with velocity related dofs
        const int ndsvel_;
        // number of dofset associated with displacement related dofs
        const int ndsdisp_;

        //! nodal velocity values at t_(n+1) or t_(n+alpha_F)
        Core::LinAlg::Matrix<nsd, nen> econvelnp_;
        //! nodal displacement values for ALE
        Core::LinAlg::Matrix<nsd, nen> edispnp_;

        // velocity divergence required for conservative form
        double divconvelint_;

        // structure velocity
        Core::LinAlg::Matrix<nsd, 1> convelint_;

        // gauss point displacements
        Core::LinAlg::Matrix<nsd, 1> dispint_;
      };


      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  decorator/wrapper class for variable manager, extensions to ScaTra coupling

      This class extends a wrapped variable manager by the  variables needed for a ScaTra
      coupling, i.e. the scalar values.
      */
      template <int nsd, int nen>
      class VariableManagerScalar : public VariableManagerDecorator<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerScalar(
            int ndsscalar, std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager)
            : VariableManagerDecorator<nsd, nen>(varmanager),
              ndsscalar_(ndsscalar),
              escalarnp_(),
              scalarnp_(),
              gradscalarnp_() {};

        //! extract node values related to ScaTra coupling
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) override;

        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! access method
        const std::vector<double>* scalarnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &scalarnp_;
        };

        //! access method
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>* grad_scalarnp() const override
        {
          this->varmanager_->check_is_evaluated();
          return &gradscalarnp_;
        }

       private:
        // number of dofset associated with scalar related dofs
        const int ndsscalar_;

        //! nodal scalar values for scatra coupling
        std::vector<Core::LinAlg::Matrix<nen, 1>> escalarnp_;

        //! scalar values
        std::vector<double> scalarnp_;

        //! spatial gradient of current scalar value
        std::vector<Core::LinAlg::Matrix<nsd, 1>> gradscalarnp_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      /*!
      \brief  decorator/wrapper class for variable manager, extension to maximum nodal volume
      fraction

      This class evaluates for each element the maximum value of a specific volume fraction at its
      nodes. This is necessary to decide if the calculation of volume fraction pressures inside an
      element makes sense returns a bool in element_has_valid_vol_frac_pressure to decide if volume
      fraction pressure has to be evaluated
      */
      template <int nsd, int nen>
      class VariableManagerMaximumNodalVolFracValue : public VariableManagerDecorator<nsd, nen>
      {
       public:
        //! constructor
        VariableManagerMaximumNodalVolFracValue(const int numvolfrac,
            std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager,
            std::shared_ptr<Core::Mat::Material> multiphasemat,
            Mat::PAR::PoroFluidPressureBased::ClosingRelation volfrac_closing_relation)
            : VariableManagerDecorator<nsd, nen>(varmanager),
              numvolfrac_(numvolfrac),
              ele_has_valid_volfrac_press_(numvolfrac_, true),
              ele_has_valid_volfrac_spec_(numvolfrac_, true),
              multiphasemat_(multiphasemat),
              volfrac_closing_relation_(volfrac_closing_relation) {};

        //! extract node values related to time derivatives
        //! dofsetnum is the number of the porofluid-dofset on the current element
        //! default is set to zero, if called from a porofluidmultiphase-element
        //! otherwise it has to be explicitly passed from the caller
        void extract_element_and_node_values(const Core::Elements::Element& ele,
            const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
            Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum = 0) override;

        //! evaluate variables at gauss point
        void evaluate_gp_variables(
            const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
            const Core::LinAlg::Matrix<nsd, nen>&
                derxy  //! array for shape function derivatives w.r.t x,y,z
            ) override;

        //! @name Access methods
        bool element_has_valid_vol_frac_pressure(const int ivolfrac) const override
        {
          this->varmanager_->check_is_extracted();
          if (ivolfrac >= numvolfrac_)
            FOUR_C_THROW(
                "{} is bigger than the number of volume fractions {} in the VariableManager",
                ivolfrac + 1, numvolfrac_);

          return ele_has_valid_volfrac_press_[ivolfrac];
        }

        bool element_has_valid_vol_frac_species(const int ivolfrac) const override
        {
          this->varmanager_->check_is_extracted();
          if (ivolfrac >= numvolfrac_)
            FOUR_C_THROW(
                "{} is bigger than the number of volume fractions {} in the VariableManager",
                ivolfrac + 1, numvolfrac_);

          return ele_has_valid_volfrac_spec_[ivolfrac];
        }

        //@}

       private:
        const int numvolfrac_;

        //! check if volume fraction pressure equation can be evaluated within this element
        std::vector<bool> ele_has_valid_volfrac_press_;

        //! check if volume fraction species equation can be evaluated within this element
        std::vector<bool> ele_has_valid_volfrac_spec_;

        std::shared_ptr<Core::Mat::Material> multiphasemat_;

        //! type of volume fractioin closing relation
        Mat::PAR::PoroFluidPressureBased::ClosingRelation volfrac_closing_relation_;
      };

    }  // namespace PoroFluidManager

  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
