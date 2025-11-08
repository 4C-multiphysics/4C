// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_INTERFACE_UTILS_HPP
#define FOUR_C_XFEM_INTERFACE_UTILS_HPP

// forward declaration of enums not possible
#include "4C_config.hpp"

#include "4C_cut_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fluid_ele_calc_xfem_coupling.hpp"
#include "4C_inpar_xfem.hpp"

#include <numbers>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Discret
{
  namespace Utils
  {
    class GaussIntegration;
  }
}  // namespace Discret


namespace Cut
{
  class BoundaryCell;
}


namespace XFEM
{
  class ConditionManager;
  namespace Utils
  {
    //! @name GetAverageWeights
    /*!
    \brief Get the std - average weights kappa_m and kappa_s for the Nitsche calculations
     */
    void get_std_average_weights(
        const Inpar::XFEM::AveragingStrategy averaging_strategy, double& kappa_m);

    //! @name nit_get_trace_estimate_constant
    /*!
    \brief get the constant which satisfies the trace inequality depending on the spatial
    dimension and polynomial order of the element
     */
    double nit_get_trace_estimate_constant(
        const Core::FE::CellType ele_distype, const bool is_pseudo_2D);


    //! @name nit_compute_visc_penalty_stabfac
    /*!
    \brief compute viscous part of Nitsche's penalty term scaling for Nitsche's method
     */
    void nit_compute_visc_penalty_stabfac(
        const Core::FE::CellType ele_distype,  ///< the discretization type of the element w.r.t
                                               ///< which the stabilization factor is computed
        const double&
            penscaling,  ///< material dependent penalty scaling (e.g. visceff) divided by h
        const double& NIT_stabscaling,  ///< basic nit penalty stab scaling
        const bool& is_pseudo_2D,       ///< is pseudo 2d
        const Inpar::XFEM::ViscStabTraceEstimate&
            visc_stab_trace_estimate,  ///< how to estimate the scaling from the trace inequality
        double& NIT_visc_stab_fac      ///< viscous part of Nitsche's penalty term
    );

    //! @name get_navier_slip_stabilization_parameters
    /*!
    \brief Get NavierSlip Stabilization Parameters for tangential direction
     */
    void get_navier_slip_stabilization_parameters(
        const double& NIT_visc_stab_fac,  ///< viscous Nitsche stab fac
        double& dynvisc,                  ///< average dynamic viscosity
        double& sliplength,               ///< sliplength
        double& stabnit,                  ///< stabilization factor NIT_Penalty
        double& stabadj                   ///< stabilization factor Adjoint
    );

    //! compute transformation factor for surface integration, normal, local and global gp
    //! coordinates
    void compute_surface_transformation(double& drs,  ///< surface transformation factor
        Core::LinAlg::Matrix<3, 1>& x_gp_lin,         ///< global coordinates of gaussian point
        Core::LinAlg::Matrix<3, 1>& normal,           ///< normal vector on boundary cell
        Cut::BoundaryCell* bc,                        ///< boundary cell
        const Core::LinAlg::Matrix<2, 1>&
            eta,                   ///< local coordinates of gaussian point w.r.t boundarycell
        bool referencepos = false  ///< use the bc reference position for transformation
    );

    //! pre-compute the measure of the element's intersecting surface
    double compute_meas_cut_surf(const std::map<int, std::vector<Core::FE::GaussIntegration>>&
                                     bintpoints,  ///< boundary cell integration points
        const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells  ///< boundary cells
    );

    //! compute the measure of the elements surface with given local id
    double compute_meas_face(Core::Elements::Element* ele,  ///< fluid element
        Core::LinAlg::SerialDenseMatrix& ele_xyze,          ///< element coordinates
        const int local_face_id,  ///< the local id of the face w.r.t the fluid element
        const int nsd             ///< number of space dimensions
    );

    //! compute volume-equivalent diameter
    inline double compute_vol_eq_diameter(double vol)
    {
      // get element length for tau_Mp/tau_C: volume-equival. diameter/sqrt(3)
      const double hk = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);

      return hk;
    }

    //! evaluate element volume
    template <Core::FE::CellType distype>
    double eval_element_volume(Core::LinAlg::Matrix<3, Core::FE::num_nodes(distype)> xyze,
        Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>* nurbs_weights = nullptr,
        std::vector<Core::LinAlg::SerialDenseVector>* nurbs_knots = nullptr);

    //! compute characteristic element length h_k
    template <Core::FE::CellType distype>
    double compute_char_ele_length(Core::Elements::Element* ele,  ///< fluid element
        Core::LinAlg::SerialDenseMatrix& ele_xyze,                ///< element coordinates
        XFEM::ConditionManager& cond_manager,                     ///< XFEM condition manager
        const Cut::plain_volumecell_set& vcSet,  ///< volumecell sets for volume integration
        const std::map<int, std::vector<Cut::BoundaryCell*>>&
            bcells,  ///< bcells for boundary cell integration
        const std::map<int, std::vector<Core::FE::GaussIntegration>>&
            bintpoints,  ///< integration points for boundary cell integration
        const Inpar::XFEM::ViscStabHk visc_stab_hk,  ///< h definition
        std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>> emb =
            nullptr,  ///< pointer to the embedded coupling implementation
        Core::Elements::Element* face = nullptr  ///< side element in 3D
    );

    //! compute full scaling of Nitsche's penalty term (xfluid-fluid)
    void nit_compute_full_penalty_stabfac(
        double& NIT_full_stab_fac,  ///< to be filled: full Nitsche's penalty term scaling
                                    ///< (viscous+convective part)
        const Core::LinAlg::Matrix<3, 1>& normal,    ///< interface-normal vector
        const double h_k,                            ///< characteristic element length
        const double kappa_m,                        ///< Weight parameter (parameter +/master side)
        const double kappa_s,                        ///< Weight parameter (parameter -/slave  side)
        const Core::LinAlg::Matrix<3, 1>& velint_m,  ///< Master side velocity at gauss-point
        const Core::LinAlg::Matrix<3, 1>& velint_s,  ///< Slave side velocity at gauss-point
        const double NIT_visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
        const double timefac,            ///< timefac
        const bool isstationary,         ///< isstationary
        const double densaf_master,      ///< master density
        const double densaf_slave,       ///< slave density
        Inpar::XFEM::MassConservationScaling
            mass_conservation_scaling,  ///< kind of mass conservation scaling
        Inpar::XFEM::MassConservationCombination
            mass_conservation_combination,  ///< kind of mass conservation combination
        const double NITStabScaling,        ///< scaling of nit stab fac
        Inpar::XFEM::ConvStabScaling
            ConvStabScaling,  ///< which convective stab. scaling of inflow stab
        Inpar::XFEM::XffConvStabScaling
            XFF_ConvStabScaling,            ///< which convective stab. scaling on XFF interface
        const bool IsConservative = false,  ///< conservative formulation of navier stokes
        bool error_calc = false  ///< when called in error calculation, don't add the inflow terms
    );

    double evaluate_full_traction(const double& pres_m, const Core::LinAlg::Matrix<3, 3>& vderxy_m,
        const double& visc_m, const double& penalty_fac, const Core::LinAlg::Matrix<3, 1>& vel_m,
        const Core::LinAlg::Matrix<3, 1>& vel_s, const Core::LinAlg::Matrix<3, 1>& elenormal,
        const Core::LinAlg::Matrix<3, 1>& normal, const Core::LinAlg::Matrix<3, 1>& velpf_s,
        double porosity = -1);

    double evaluate_full_traction(const Core::LinAlg::Matrix<3, 1>& intraction,
        const double& penalty_fac, const Core::LinAlg::Matrix<3, 1>& vel_m,
        const Core::LinAlg::Matrix<3, 1>& vel_s, const Core::LinAlg::Matrix<3, 1>& elenormal,
        const Core::LinAlg::Matrix<3, 1>& normal);

    double evaluate_full_traction(const double& intraction, const double& penalty_fac,
        const Core::LinAlg::Matrix<3, 1>& vel_m, const Core::LinAlg::Matrix<3, 1>& vel_s,
        const Core::LinAlg::Matrix<3, 1>& elenormal, const Core::LinAlg::Matrix<3, 1>& normal);

    void evaluate_stateat_gp(const Core::Elements::Element* sele,
        const Core::LinAlg::Matrix<3, 1>& selexsi, const Core::FE::Discretization& discret,
        const std::string& state, Core::LinAlg::Matrix<3, 1>& vel_s);

  }  // namespace Utils
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
