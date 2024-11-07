// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_BOUNDARY_CALC_ELCH_ELECTRODE_HPP
#define FOUR_C_SCATRA_ELE_BOUNDARY_CALC_ELCH_ELECTRODE_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_boundary_calc_elch.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class Electrode;
}
namespace Discret
{
  namespace Elements
  {
    class ScaTraEleBoundaryCalcElchElectrodeUtils;
  }
}  // namespace Discret

namespace Discret
{
  namespace Elements
  {
    // class implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype> + 1>
    class ScaTraEleBoundaryCalcElchElectrode : public ScaTraEleBoundaryCalcElch<distype, probdim>
    {
      using my = Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>;
      using myelch = Discret::Elements::ScaTraEleBoundaryCalcElch<distype, probdim>;
      using myelectrodeutils = Discret::Elements::ScaTraEleBoundaryCalcElchElectrodeUtils;

     protected:
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

     public:
      //! singleton access method
      static ScaTraEleBoundaryCalcElchElectrode<distype, probdim>* instance(
          int numdofpernode, int numscal, const std::string& disname);


      /*!
       * \brief evaluate scatra-scatra interface coupling condition at integration point
       *
       * \remark This is a static method as it is also called from
       * `scatra_timint_meshtying_strategy_s2i_elch.cpp` for the mortar implementation.
       *
       * @param[in] matelectrode     electrode material
       * @param[in] eslavephinp      state variables at slave-side nodes
       * @param[in] emasterphinp     state variables at master-side nodes
       * @param[in] eslavetempnp     temperature at slave side of interface
       * @param[in] emastertempnp    temperature at master side of interface
       * @param[in] pseudo_contact_fac  factor, modeling pseudo contact by considering the
       *                                mechanical stress state at the interface (1.0 if under
       *                                compressive stresses, 0.0 if under tensile stresses)
       * @param[in] funct_slave      slave-side shape function values
       * @param[in] funct_master     master-side shape function values
       * @param[in] test_slave                 slave-side test function values
       * @param[in] test_master                master-side test function values
       * @param[in] scatra_parameter_boundary  interface parameter class
       * @param[in] timefacfac       time-integration factor times domain-integration factor
       * @param[in] timefacrhsfac    time-integration factor for right-hand side times
       *                             domain-integration factor
       * @param[in] detF             determinant of jacobian at current integration point
       * @param[in] frt              factor F/(RT)
       * @param[in] num_dof_per_node number of dofs per node
       * @param[out] k_ss            linearizations of slave-side residuals w.r.t. slave-side dofs
       * @param[out] k_sm            linearizations of slave-side residuals w.r.t. master-side dofs
       * @param[out] k_ms            linearizations of master-side residuals w.r.t. slave-side dofs
       * @param[out] k_mm            linearizations of master-side residuals w.r.t. master-side
       *                             dofs
       * @param[out] r_s             slave-side residual vector
       * @param[out] r_m             master-side residual vector
       *
       * \tparam distype_master  This method is templated on the master-side discretization type.
       */
      template <Core::FE::CellType distype_master>
      static void evaluate_s2_i_coupling_at_integration_point(
          const std::shared_ptr<const Mat::Electrode>& matelectrode,
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
          const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
              emasterphinp,
          const Core::LinAlg::Matrix<nen_, 1>& eslavetempnp,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& emastertempnp,
          double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
          const Core::LinAlg::Matrix<nen_, 1>& test_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
          const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
          double timefacfac, double timefacrhsfac, double detF, double frt, int num_dof_per_node,
          Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
          Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
          Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m);

      /*!
       * @brief evaluate capacitive part of the scatra-scatra interface coupling condition at
       * integration point
       *
       * @param[in] eslavephidtnp  time derivative of state variables at slave-side nodes at time
       *                           n+1
       * @param[in] emasterphidtnp time derivative of state variables at master-side nodes at time
       *                           n+1
       * @param[in] eslavephinp    state variables at slave-side nodes at time n+1
       * @param[in] emasterphinp   state variables at master-side nodes at time n+1
       * @param[in] pseudo_contact_fac  factor, modeling pseudo contact by considering the
       *                                mechanical stress state at the interface (1.0 if under
       *                                compressive stresses, 0.0 if under tensile stresses)
       * @param[in] funct_slave    slave-side shape function values
       * @param[in] funct_master   master-side shape function values
       * @param[in] test_slave                 slave-side test function values
       * @param[in] test_master                master-side test function values
       * @param[in] scatra_parameter_boundary  interface parameter class
       * @param[in] timederivfac   factor to account for time-derivative in linearization
       * @param[in] timefacfac     time-integration factor times domain-integration factor
       * @param[in] timefacrhsfac  time-integration factor for right-hand side times
       *                           domain-integration factor
       * @param[in] num_dof_per_node number of dofs per node
       * @param[out] k_ss  linearizations of slave-side residuals w.r.t. slave-side dofs
       * @param[out] k_ms  linearizations of master-side residuals w.r.t. slave-side dofs
       * @param[out] r_s   slave-side residual vector
       * @param[out] r_m   master-side residual vector
       *
       * @tparam distype_master This method is templated on the master-side discretization type.
       */
      template <Core::FE::CellType distype_master>
      static void evaluate_s2_i_coupling_capacitance_at_integration_point(
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephidtnp,
          const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
              emasterphidtnp,
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
          const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
              emasterphinp,
          double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
          const Core::LinAlg::Matrix<nen_, 1>& test_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
          const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
          double timederivfac, double timefacfac, double timefacrhsfac, int num_dof_per_node,
          Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_ms,
          Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m);

      /*!
       * \brief calculate out-parameters such as residual vectors and linearizations of residuals
       *
       * \remark This is a static method as it is called from
       * static method evaluate_s2_i_coupling_at_integration_point.
       *
       * @param[in] funct_slave      slave-side shape function values
       * @param[in] funct_master     master-side shape function values
       * @param[in] test_slave       slave-side test function values
       * @param[in] test_master      master-side test function values
       * @param[in] pseudo_contact_fac  factor, modeling pseudo contact by considering the
       *                                mechanical stress state at the interface (1.0 if under
       *                                compressive stresses, 0.0 if under tensile stresses)
       * @param[in] numelectrons     number of electrons involved in charge transfer at
       *                             electrode-electrolyte interface
       * @param[in] nen_master       number of nodes of master-side mortar element
       * @param[in] timefacfac       time and domain integration factor for linearization terms
       * @param[in] timefacrhsfac    time and domain integration factor for RHS terms
       * @param[in] dj_dc_slave      linearization of Butler-Volmer mass flux density w.r.t.
       *                             concentration on slave-side
       * @param[in] dj_dc_master     linearization of Butler-Volmer mass flux density w.r.t.
       *                             concentration on master-side
       * @param[in] dj_dpot_slave    linearization of Butler-Volmer mass flux density w.r.t.
       *                             electric potential on slave-side
       * @param[in] dj_dpot_master   linearization of Butler-Volmer mass flux density w.r.t.
       *                             electric potential on master-side
       * @param[in] j                Butler-Volmer mass flux density
       * @param[in] num_dof_per_node number of dofs per node
       * @param[out] k_ss            linearizations of slave-side residuals w.r.t. slave-side dofs
       * @param[out] k_sm            linearizations of slave-side residuals w.r.t. master-side dofs
       * @param[out] k_ms            linearizations of master-side residuals w.r.t. slave-side dofs
       * @param[out] k_mm            linearizations of master-side residuals w.r.t. master-side dofs
       * @param[out] r_s             slave-side residual vector
       * @param[out] r_m             master-side residual vector
       *
       * @tparam distype_master  This method is templated on the master-side discretization type.
       */
      template <Core::FE::CellType distype_master>
      static void calculate_rh_sand_global_system(const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
          const Core::LinAlg::Matrix<nen_, 1>& test_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
          double pseudo_contact_fac, double numelectrons, int nen_master, double timefacfac,
          double timefacrhsfac, double dj_dc_slave, double dj_dc_master, double dj_dpot_slave,
          double dj_dpot_master, double j, int num_dof_per_node,
          Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
          Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
          Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m);

      /*!
       * @brief calculate residual vectors and linearizations of residuals due to capacitive flux at
       * the scatra-scatra interface
       *
       * @param[in] funct_slave     slave-side shape function values
       * @param[in] test_slave      slave-side test function values
       * @param[in] test_master     master-side test function values
       * @param[in] pseudo_contact_fac  factor, modeling pseudo contact by considering the
       *                                mechanical stress state at the interface (1.0 if under
       *                                compressive stresses, 0.0 if under tensile stresses)
       * @param[in] numelectrons    number of electrons involved in charge transfer
       * @param[in] timefacfac      time-integration factor times domain-integration factor
       * @param[in] timefacrhsfac   time-integration factor for right-hand side times
       *                            domain-integration factor
       * @param[in] nen_master      number of nodes of master-side mortar element
       * @param[in] jC              capacitance mass flux density
       * @param[in] djC_dpot_slave  derivative of capacitance mass flux density w.r.t. electric
       *                            potential at the slave-side
       * @param[in] num_dof_per_node number of dofs per node
       * @param[out] k_ss  linearizations of slave-side residuals w.r.t. slave-side dofs
       * @param[out] k_ms  linearizations of master-side residuals w.r.t. slave-side dofs
       * @param[out] r_s   slave-side residual vector
       * @param[out] r_m   master-side residual vector
       *
       * @tparam distype_master  This method is templated on the master-side discretization type.
       */
      template <Core::FE::CellType distype_master>
      static void calculate_rh_sand_global_system_capacitive_flux(
          const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
          const Core::LinAlg::Matrix<nen_, 1>& test_slave,
          const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
          double pseudo_contact_fac, int numelectrons, double timefacfac, double timefacrhsfac,
          int nen_master, double jC, double djC_dpot_slave, int num_dof_per_node,
          Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_ms,
          Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m);

     protected:
      //! protected constructor for singletons
      ScaTraEleBoundaryCalcElchElectrode(
          int numdofpernode, int numscal, const std::string& disname);

      void evaluate_s2_i_coupling(const Core::Elements::FaceElement* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix,
          Core::LinAlg::SerialDenseMatrix& emastermatrix,
          Core::LinAlg::SerialDenseVector& eslaveresidual) override;

      void evaluate_s2_i_coupling_capacitance(const Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& eslavematrix,
          Core::LinAlg::SerialDenseMatrix& emastermatrix,
          Core::LinAlg::SerialDenseVector& eslaveresidual,
          Core::LinAlg::SerialDenseVector& emasterresidual) override;

      void evaluate_s2_i_coupling_od(const Core::Elements::FaceElement* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& eslavematrix) override;

      void evaluate_s2_i_coupling_capacitance_od(Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& eslavematrix,
          Core::LinAlg::SerialDenseMatrix& emastermatrix) override;

      double get_valence(
          const std::shared_ptr<const Core::Mat::Material>& material, int k) const override;

      void calc_s2_i_coupling_flux(const Core::Elements::FaceElement* ele,
          const Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseVector& scalars) override;

      //! evaluate factor F/RT
      virtual double get_frt() const;
    };  // class ScaTraEleBoundaryCalcElchElectrode
  }     // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
