// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_S2I_ELCH_HPP
#define FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_S2I_ELCH_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_scatra_timint_elch.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /*!
  \brief Scatra-scatra interface coupling strategy for electrochemistry problems

  To keep the scalar transport time integrator class and derived classes as plain as possible,
  several algorithmic parts have been encapsulated within separate meshtying strategy classes.
  These algorithmic parts include initializing the system matrix and other relevant objects,
  computing meshtying residual terms and their linearizations, and solving the resulting
  linear system of equations. By introducing a hierarchy of strategies for these algorithmic
  parts, a bunch of unhandy if-else selections within the time integrator classes themselves
  can be circumvented. This class contains the scatra-scatra interface coupling strategy for
  electrochemistry problems.

  */

  class MeshtyingStrategyS2IElch : public MeshtyingStrategyS2I
  {
   public:
    //! constructor
    explicit MeshtyingStrategyS2IElch(
        ScaTra::ScaTraTimIntElch* elchtimint,  //!< elch time integrator
        const Teuchos::ParameterList&
            parameters  //!< input parameters for scatra-scatra interface coupling
    );

    //! compute time step size
    void compute_time_step_size(double& dt) final;

    //! compute meshtying residual terms and their linearizations
    void evaluate_meshtying() override;

    void evaluate_point_coupling() override;

    //! update solution after convergence of the nonlinear Newton-Raphson iteration
    void update() const override;

   private:
    //! copy constructor
    MeshtyingStrategyS2IElch(const MeshtyingStrategyS2IElch& old);

    //! return pointer to elch time integrator after cast
    ScaTra::ScaTraTimIntElch* elch_tim_int() const
    {
      return dynamic_cast<ScaTra::ScaTraTimIntElch*>(scatratimint_);
    };

    //! instantiate strategy for Newton-Raphson convergence check
    void init_conv_check_strategy() override;

    //! minimum interfacial overpotential associated with scatra-scatra interface layer growth
    double etagrowthmin_;

    //! time step at the start of scatra-scatra interface layer growth
    int intlayergrowth_startstep_;

    //! flag indicating modified time step size for scatra-scatra interface layer growth
    bool intlayergrowth_timestep_active_;
  };  // class MeshtyingStrategyS2IElch

  class MeshtyingStrategyS2IElchSCL : public MeshtyingStrategyS2IElch
  {
   public:
    explicit MeshtyingStrategyS2IElchSCL(
        ScaTra::ScaTraTimIntElch* elchtimint, const Teuchos::ParameterList& parameters);

    void add_time_integration_specific_vectors() const override {};

    void evaluate_meshtying() override {};

    void setup_meshtying() override;

    void solve(const std::shared_ptr<Core::LinAlg::Solver>& solver,
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& increment,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& phinp, const int iteration,
        Core::LinAlg::SolverParams& solver_params) const override;
  };

  template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
  class MortarCellCalcElch : public MortarCellCalc<distype_s, distype_m>
  {
   public:
    //! singleton access method
    static MortarCellCalcElch<distype_s, distype_m>* instance(
        const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
        const std::string& disname        //!< name of mortar discretization
    );


   protected:
    //! protected constructor for singletons
    MortarCellCalcElch(const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    );

   private:
    //! abbreviation
    using my = MortarCellCalc<distype_s, distype_m>;

   protected:
    using my::nen_master_;
    using my::nen_slave_;
    using my::nsd_slave_;

   private:
    //! evaluate and assemble interface linearizations and residuals
    void evaluate_condition(const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                                         //!< mortar integration cell
        Mortar::Element& slaveelement,             //!< slave-side mortar element
        Mortar::Element& masterelement,            //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master,  //!< master-side location array
        const Teuchos::ParameterList& params,      //!< parameter list
        Core::LinAlg::SerialDenseMatrix&
            k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_sm,  //!< linearizations of slave-side residuals w.r.t. master-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_ms,  //!< linearizations of master-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_mm,  //!< linearizations of master-side residuals w.r.t. master-side dofs
        Core::LinAlg::SerialDenseVector& r_s,  //!< slave-side residual vector
        Core::LinAlg::SerialDenseVector& r_m   //!< master-side residual vector
        ) override;

    //! evaluate and assemble interface linearizations and residuals for node-to-segment coupling
    void evaluate_condition_nts(const Core::Conditions::Condition&
                                    condition,  //!< scatra-scatra interface coupling condition
        const Mortar::Node& slavenode,          //!< slave-side node
        const double&
            lumpedarea,  //!< lumped interface area fraction associated with slave-side node
        Mortar::Element& slaveelement,   //!< slave-side mortar element
        Mortar::Element& masterelement,  //!< master-side mortar element
        const std::vector<Core::LinAlg::Matrix<nen_slave_, 1>>&
            ephinp_slave,  //!< state variables at slave-side nodes
        const std::vector<Core::LinAlg::Matrix<nen_master_, 1>>&
            ephinp_master,  //!< state variables at master-side nodes
        Core::LinAlg::SerialDenseMatrix&
            k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_sm,  //!< linearizations of slave-side residuals w.r.t. master-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_ms,  //!< linearizations of master-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_mm,  //!< linearizations of master-side residuals w.r.t. master-side dofs
        Core::LinAlg::SerialDenseVector& r_s,  //!< slave-side residual vector
        Core::LinAlg::SerialDenseVector& r_m   //!< master-side residual vector
        ) override;

    //! evaluate factor F/RT
    virtual double get_frt() const;
  };  // class MortarCellCalcElch


  template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
  class MortarCellCalcElchSTIThermo : public MortarCellCalcElch<distype_s, distype_m>
  {
   public:
    //! singleton access method
    static MortarCellCalcElchSTIThermo<distype_s, distype_m>* instance(
        const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
        const std::string& disname        //!< name of mortar discretization
    );


    //! evaluate single mortar integration cell of particular slave-side and master-side
    //! discretization types
    void evaluate(const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                               //!< mortar integration cell
        Mortar::Element& slaveelement,                       //!< slave-side mortar element
        Mortar::Element& masterelement,                      //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,             //!< slave-side location array
        Core::Elements::LocationArray& la_master,            //!< master-side location array
        const Teuchos::ParameterList& params,                //!< parameter list
        Core::LinAlg::SerialDenseMatrix& cellmatrix1,        //!< cell matrix 1
        Core::LinAlg::SerialDenseMatrix& cellmatrix2,        //!< cell matrix 2
        Core::LinAlg::SerialDenseMatrix& cellmatrix3,        //!< cell matrix 3
        Core::LinAlg::SerialDenseMatrix& cellmatrix4,        //!< cell matrix 4
        Core::LinAlg::SerialDenseVector& cellvector1,        //!< cell vector 1
        Core::LinAlg::SerialDenseVector& cellvector2         //!< cell vector 2
        ) override;

   private:
    //! abbreviation
    using my = MortarCellCalc<distype_s, distype_m>;
    using myelch = MortarCellCalcElch<distype_s, distype_m>;
    using my::nen_master_;
    using my::nen_slave_;
    using my::nsd_slave_;

    //! private constructor for singletons
    MortarCellCalcElchSTIThermo(
        const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    );

    //! evaluate and assemble off-diagonal interface linearizations
    void evaluate_condition_od(
        const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                     //!< mortar integration cell
        Mortar::Element& slaveelement,             //!< slave-side mortar element
        Mortar::Element& masterelement,            //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master,  //!< master-side location array
        const Teuchos::ParameterList& params,      //!< parameter list
        Core::LinAlg::SerialDenseMatrix&
            k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_ms  //!< linearizations of master-side residuals w.r.t. slave-side dofs
    );

    //! extract nodal state variables associated with mortar integration cell
    void extract_node_values(
        const Core::FE::Discretization& idiscret,  //!< interface discretization
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master   //!< master-side location array
        ) override;

    //! evaluate factor F/RT
    double get_frt() const override;

    //! nodal, slave-side temperature variables associated with time t_{n+1} or t_{n+alpha_f}
    Core::LinAlg::Matrix<nen_slave_, 1> etempnp_slave_;
  };  // class MortarCellCalcElchSTIThermo


  template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
  class MortarCellCalcSTIElch : public MortarCellCalc<distype_s, distype_m>
  {
   public:
    //! singleton access method
    static MortarCellCalcSTIElch<distype_s, distype_m>* instance(
        const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master,  //!< number of master-side degrees of freedom per node
        const std::string& disname        //!< name of mortar discretization
    );

    //! evaluate single mortar integration cell of particular slave-side and master-side
    //! discretization types
    void evaluate(const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                               //!< mortar integration cell
        Mortar::Element& slaveelement,                       //!< slave-side mortar element
        Mortar::Element& masterelement,                      //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,             //!< slave-side location array
        Core::Elements::LocationArray& la_master,            //!< master-side location array
        const Teuchos::ParameterList& params,                //!< parameter list
        Core::LinAlg::SerialDenseMatrix& cellmatrix1,        //!< cell matrix 1
        Core::LinAlg::SerialDenseMatrix& cellmatrix2,        //!< cell matrix 2
        Core::LinAlg::SerialDenseMatrix& cellmatrix3,        //!< cell matrix 3
        Core::LinAlg::SerialDenseMatrix& cellmatrix4,        //!< cell matrix 4
        Core::LinAlg::SerialDenseVector& cellvector1,        //!< cell vector 1
        Core::LinAlg::SerialDenseVector& cellvector2         //!< cell vector 2
        ) override;

   private:
    //! abbreviation
    using my = MortarCellCalc<distype_s, distype_m>;
    using my::nen_master_;
    using my::nen_slave_;
    using my::nsd_slave_;

    //! private constructor for singletons
    MortarCellCalcSTIElch(const S2I::CouplingType& couplingtype,  //!< flag for meshtying method
        const S2I::InterfaceSides&
            lmside,  //!< flag for interface side underlying Lagrange multiplier definition
        const int& numdofpernode_slave,  //!< number of slave-side degrees of freedom per node
        const int& numdofpernode_master  //!< number of master-side degrees of freedom per node
    );

    //! evaluate and assemble interface linearizations and residuals
    void evaluate_condition(const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                                         //!< mortar integration cell
        Mortar::Element& slaveelement,             //!< slave-side mortar element
        Mortar::Element& masterelement,            //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master,  //!< master-side location array
        const Teuchos::ParameterList& params,      //!< parameter list
        Core::LinAlg::SerialDenseMatrix&
            k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseVector& r_s  //!< slave-side residual vector
    );

    //! evaluate and assemble off-diagonal interface linearizations
    void evaluate_condition_od(
        const Core::FE::Discretization& idiscret,  //!< interface discretization
        Mortar::IntCell& cell,                     //!< mortar integration cell
        Mortar::Element& slaveelement,             //!< slave-side mortar element
        Mortar::Element& masterelement,            //!< master-side mortar element
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master,  //!< master-side location array
        const Teuchos::ParameterList& params,      //!< parameter list
        Core::LinAlg::SerialDenseMatrix&
            k_ss,  //!< linearizations of slave-side residuals w.r.t. slave-side dofs
        Core::LinAlg::SerialDenseMatrix&
            k_sm  //!< linearizations of slave-side residuals w.r.t. master-side dofs
    );

    //! extract nodal state variables associated with mortar integration cell
    void extract_node_values(
        const Core::FE::Discretization& idiscret,  //!< interface discretization
        Core::Elements::LocationArray& la_slave,   //!< slave-side location array
        Core::Elements::LocationArray& la_master   //!< master-side location array
        ) override;

    //! nodal, slave-side electrochemistry variables associated with time t_{n+1} or t_{n+alpha_f}
    std::vector<Core::LinAlg::Matrix<nen_slave_, 1>> eelchnp_slave_;

    //! nodal, master-side electrochemistry variables associated with time t_{n+1} or t_{n+alpha_f}
    std::vector<Core::LinAlg::Matrix<nen_master_, 1>> eelchnp_master_;
  };  // class MortarCellCalcSTIElch
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
