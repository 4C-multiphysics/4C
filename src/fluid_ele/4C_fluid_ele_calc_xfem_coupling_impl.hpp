// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_CALC_XFEM_COUPLING_IMPL_HPP
#define FOUR_C_FLUID_ELE_CALC_XFEM_COUPLING_IMPL_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fluid_ele_calc_xfem_coupling.hpp"
#include "4C_xfem_input.hpp"

//   qnuP - option SHOULD BE ON!
//     projects the given velocity into normal direction in case there
//     is a smoothed projection matrix given.
#define PROJECT_VEL_FOR_PRESSURE_ADJOINT

FOUR_C_NAMESPACE_OPEN

//  For comparison to Urquizas paper with their slip length implementation
//   Only working for Navier-Slip (i.e. itraction_jump_ = 0)!
//   Can be modified to work for this case as well.
// #define ENFORCE_URQUIZA_GNBC

namespace Discret
{
  namespace Elements
  {
    class FluidEleParameterXFEM;
    namespace XFLUID
    {
      //! class for concrete coupling slave element
      //! this can be an arbitrary 2D/3D element and can be associated with a structure (monolithic
      //! XFSI), fluid (XFF, XFFSI, partitioned XFSI, XWDBC) or a xfluid-element with another active
      //! dofset (two-phase flow)
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      class SlaveElementRepresentation : virtual public SlaveElementInterface<distype>
      {
       public:
        /// number of nodes per master (xfem-fluid) element
        static constexpr unsigned nen_ = Core::FE::num_nodes(distype);
        /// number of spatial dimensions of the master element (xfem-fluid)
        static constexpr unsigned nsd_ = Core::FE::dim<distype>;
        /// number of spatial dimensions of the slave side
        static constexpr unsigned slave_nsd_ = Core::FE::dim<slave_distype>;
        /// number of slave element's nodes
        static constexpr unsigned slave_nen_ = Core::FE::num_nodes(slave_distype);

        //! ctor
        SlaveElementRepresentation(Core::LinAlg::SerialDenseMatrix& slave_xyze)
            : slave_xyze_(slave_xyze.values(), true)
        {
          SlaveElementInterface<distype>::define_state_names(
              slave_distype, disp_statename_, vel_statename_, veln_statename_);
        };

        //! add coupling slave element's displacements and set current slave element node
        //! coordinates
        void add_slave_ele_disp(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm                 ///< local map
            ) override;
        void add_slave_ele_disp(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm,                ///< local map
            std::vector<double>& mymatrix              ///< slave element displacement vector
            ) override;

        //! set slave element's interface velocity & pressure for current time step
        void set_slave_state(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm                 ///< local map
            ) override;

        //! set slave element's interface velocity & pressure for previous time step
        void set_slave_staten(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm                 ///< local map
            ) override;

        //! @name Accessors
        //@{

        //! extract interface velocity at current time step
        void get_interface_velnp(
            Core::LinAlg::Matrix<nsd_, 1>& ivelint  ///< interface velocity at coupling slave side
        ) const override;

        //! get interface pressure at current time step
        void get_interface_presnp(double& ipres  ///< interface pressure at coupling slave side
        ) const override;

        //! get interface velocity gradient at current time step
        void get_interface_vel_gradnp(Core::LinAlg::Matrix<nsd_, nsd_>&
                velgradint  ///< interface velocity gradients at coupling slave side
        ) const override;

        //! extract interface velocity at previous time step
        void get_interface_veln(
            Core::LinAlg::Matrix<nsd_, 1>& ivelintn  ///< interface velocity at coupling slave side
        ) const override;

        //! get interface pressure at previous time step
        void get_interface_presn(double& ipresn  ///< interface pressure at coupling slave side
        ) const override;

        //! get interface velocity gradient at previous time step
        void get_interface_vel_gradn(Core::LinAlg::Matrix<nsd_, nsd_>&
                velgradintn  ///< interface velocity gradients at coupling slave side
        ) const override;

        //! get slave elements nodal shape functions
        void get_slave_funct(
            Core::LinAlg::Matrix<slave_nen_, 1>& slave_funct  ///< coupling slave shape functions
        ) const;

        //! set state for interface velocity jump
        void set_interface_jump_statenp(
            const Core::FE::Discretization& cutterdis,  ///< cutter discretization
            const std::string state,                    ///< state
            const std::vector<int>& lm                  ///< local map
            ) override;

        //! set state for interface velocity jump for previous time step
        void set_interface_jump_staten(
            const Core::FE::Discretization& cutterdis,  ///< cutter discretization
            const std::string state,                    ///< state
            const std::vector<int>& lm                  ///< local map
            ) override;

        //! get interface velocity jump at Gaussian point
        void get_interface_jump_velnp(
            Core::LinAlg::Matrix<nsd_, 1>& ivelint_jump  ///< cutter element interface velocity jump
                                                         ///< or prescribed DBC at Gaussian point
        ) const override;

        //! get interface velocity jump for previous time step at Gaussian point
        void get_interface_jump_veln(Core::LinAlg::Matrix<nsd_, 1>&
                ivelintn_jump  ///< cutter element interface velocity jump or
                               ///< prescribed DBC at Gaussian point
        ) const override;

        //@}

        //!  evaluate shape function, derivatives and transformation w.r.t coupling slave element at
        //!  gaussian point
        void evaluate(Core::LinAlg::Matrix<nsd_, 1>& xslave) override;

        //!  evaluate shape function, derivatives and transformation w.r.t coupling slave element at
        //!  gaussian point
        void evaluate(Core::LinAlg::Matrix<nsd_, 1>& xslave,
            Core::LinAlg::Matrix<nsd_, 1>& rst_slave) override;

        //! compute interface force
        void compute_interface_force(
            Core::LinAlg::SerialDenseVector& iforce,  ///< interface force vector
            Core::LinAlg::Matrix<nsd_, 1>& traction,  ///< traction vector at gaussian point
            const double& fac                         ///< integration factor
            ) override;

        //! project gaussian point from linearized interface in normal direction onto corresponding
        //! side
        void project_on_side(
            Core::LinAlg::Matrix<nsd_, 1>&
                x_gp_lin,  ///< global coordinates of gaussian point w.r.t linearized interface
            Core::LinAlg::Matrix<nsd_, 1>& x_side,  ///< projected gaussian point on side
            Core::LinAlg::Matrix<nsd_, 1>&
                xi_side  ///< local coordinates of projected gaussian point w.r.t side
            ) override;

        //! evaluate element volume
        double eval_element_volume() override;

       protected:
        //! default constructor
        SlaveElementRepresentation()
        {
          SlaveElementInterface<distype>::define_state_names(
              slave_distype, disp_statename_, vel_statename_, veln_statename_);
        };

        //! @name accessors for derived classes
        //@{

        //! get spatial derivatives of slave elements nodal shape functions
        void get_slave_funct_deriv(Core::LinAlg::Matrix<nsd_, slave_nen_>& slave_derxy) const;

        //@}

       private:
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_xyze_;  ///< coupling slave element's node coordinates
        Core::LinAlg::Matrix<slave_nen_, 1>
            slave_funct_;  ///< coupling slave element's shape functions
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_derxy_;  ///< coupling slave element's local shape function derivatives
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_deriv_;  ///< coupling slave element's global shape function derivatives
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_vel_;  ///< coupling slave element's velocity at current step
        Core::LinAlg::Matrix<slave_nen_, 1>
            slave_pres_;  ///< coupling slave element's pressure at current step
        Core::LinAlg::Matrix<nsd_, nsd_>
            slave_vderxy_;  ///< coupling slave element's velocity derivatives at current step
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_disp_;  ///< coupling slave element's displacements at current step
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            slave_veln_;  ///< coupling slave element's velocity at previous step
        Core::LinAlg::Matrix<slave_nen_, 1>
            slave_presn_;  ///< coupling slave element's pressure at previous step
        Core::LinAlg::Matrix<nsd_, nsd_>
            slave_vderxyn_;  ///< coupling slave element's velocity derivatives at previous step

        std::string disp_statename_;  ///< name of current displacement state (for access from
                                      ///< discretization)
        std::string
            vel_statename_;  ///< name of current velocity state (for access from discretization)
        std::string
            veln_statename_;  ///< name of previous velocity state (for access from discretization)

        // TODO: shift this vector to the same class as project_on_side which is based on the cutter
        // discretization
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            interface_velnp_jump_;  ///< cutter element's prescribed velocity jump height or
                                    ///< prescribed DBC values
        Core::LinAlg::Matrix<nsd_, slave_nen_>
            interface_veln_jump_;  ///< cutter element's prescribed velocity jump height or
                                   ///< prescribed DBC values

        Core::LinAlg::Matrix<slave_nen_, 1> proj_funct_;  ///< shape functions for project on side
        Core::LinAlg::Matrix<2, slave_nen_>
            proj_deriv_;  ///< derivatives dr, ds for project on side
        Core::LinAlg::Matrix<3, slave_nen_>
            proj_deriv2_;  ///< 2nd derivatives drdr, dsds, drds for project on side

        Core::LinAlg::Matrix<3, 1> proj_x_;       ///< global coordinates
        Core::LinAlg::Matrix<3, 2> proj_derxy_;   ///< global xyz derivatives
        Core::LinAlg::Matrix<3, 3> proj_derxy2_;  ///< global xyz 2nd derivatives

        Core::LinAlg::Matrix<3, 1> proj_residuum_;  ///<  residuum of the newton iteration
        Core::LinAlg::Matrix<3, 3> proj_sysmat_;    ///<  matrix for the newton system
        Core::LinAlg::Matrix<3, 1> proj_incr_;      ///<  increment of the newton system

        Core::LinAlg::Matrix<3, 1> proj_sol_;  ///< sol carries xi_1, xi_2, d (distance)

        // get vector products
        Core::LinAlg::Matrix<3, 1> proj_dx_drdr_times_dx_ds_;
        Core::LinAlg::Matrix<3, 1> proj_dx_dr_times_dx_drds_;
        Core::LinAlg::Matrix<3, 1> proj_dx_drds_times_dx_ds_;
        Core::LinAlg::Matrix<3, 1> proj_dx_dr_times_dx_dsds_;
        Core::LinAlg::Matrix<3, 1> proj_dx_dr_times_dx_ds_;
      };

      /*!
       * specialized interface class for XFluid weak Dirichlet problems with a interface given by a
       * level-set field (we then don't couple with a concrete slave element!)
       */
      //!
      template <Core::FE::CellType distype, unsigned int slave_numdof>
      class SlaveElementRepresentation<distype, Core::FE::CellType::dis_none, slave_numdof>
      {
       public:
        /// number of nodes per master (xfem-fluid) element
        static constexpr unsigned nen_ = Core::FE::num_nodes(distype);
        /// number of spatial dimensions of the master element (xfem-fluid)
        static constexpr unsigned nsd_ = Core::FE::dim<distype>;
        /// number of spatial dimensions of the slave side
        static constexpr unsigned slave_nsd_ = Core::FE::dim<distype>;
        /// just for compatibility...
        static constexpr unsigned slave_nen_ = nen_;

        //! get slave elements nodal shape functions - if the interface is given as a level-set
        //! field, the request is unfulfilled
        void get_slave_funct(
            Core::LinAlg::Matrix<slave_nen_, 1>& slave_funct  ///< coupling slave shape functions
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

       protected:
        //! default ctor
        SlaveElementRepresentation() {};

        //! ctor
        SlaveElementRepresentation(Core::LinAlg::SerialDenseMatrix& slave_xyze) {};

        //! get nodal shape function derivatives
        void get_slave_funct_deriv(Core::LinAlg::Matrix<nsd_, slave_nen_>& slave_derxy) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };
      };

      //! concrete evaluation class for interface coupling using Nitsche's method
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      class NitscheCoupling
          : public NitscheInterface<distype>,
            public SlaveElementRepresentation<distype, slave_distype, slave_numdof>
      {
       public:
        /// number of nodes per master (xfem-fluid) element
        static constexpr unsigned nen_ =
            SlaveElementRepresentation<distype, slave_distype, slave_numdof>::nen_;
        /// number of spatial dimensions
        static constexpr unsigned nsd_ = SlaveElementInterface<distype>::nsd_;
        /// number of nodal dof for master element (coupling master is always a fluid element!)
        static constexpr unsigned master_numdof_ = nsd_ + 1;
        /// number of slave element's nodes
        static constexpr unsigned slave_nen_ =
            SlaveElementRepresentation<distype, slave_distype, slave_numdof>::slave_nen_;

        //! ctor for one-sided (xfluid weak dirichlet) problems (interface defined by level-set
        //! field)
        NitscheCoupling(Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_um,              ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        NitscheCoupling(Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseVector& rhC_um,              ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! ctor for one-sided (xfluid weak dirichlet) problems (interface defined by mesh)
        NitscheCoupling(Core::LinAlg::SerialDenseMatrix&
                            slave_xyze,               ///< global node coordinates of slave element
            Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_um,  ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        NitscheCoupling(Core::LinAlg::SerialDenseMatrix&
                            slave_xyze,               ///< global node coordinates of slave element
            Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseVector& rhC_um,  ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! ctor for two-sided problems
        NitscheCoupling(Core::LinAlg::SerialDenseMatrix&
                            slave_xyze,               ///< global node coordinates of slave element
            Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_usus,  ///< C_usus coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_um,  ///< C_um coupling rhs
            Core::LinAlg::SerialDenseMatrix& rhC_us,  ///< C_us coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        NitscheCoupling(Core::LinAlg::SerialDenseMatrix&
                            slave_xyze,               ///< global node coordinates of slave element
            Core::LinAlg::SerialDenseMatrix& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_usus,  ///< C_usus coupling matrix
            Core::LinAlg::SerialDenseVector& rhC_um,  ///< C_um coupling rhs
            Core::LinAlg::SerialDenseMatrix& rhC_us,  ///< C_us coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! add contributions from convective stabilization
        //! this method is applied in a non-Nitsche context (e.g. MHVS) by
        //! employing shape functions and velocities from another slave element coupling object
        void apply_conv_stab_terms(const std::shared_ptr<SlaveElementInterface<distype>>&
                                       slave_ele,  ///< associated slave element coupling object
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,   ///< master shape functions
            const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< vector of slave shape functions
            const Core::LinAlg::Matrix<nsd_, 1>& normal,    ///< normal vector n^b
            const double& density_m,                        ///< fluid density (master)
            const double& NIT_stab_fac_conv,                ///< full Nitsche's penalty term scaling
                                                            ///< (viscous+convective part)
            const double& timefacfac,                       ///< theta*dt
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height
                               ///< for coupled problems
            const XFEM::EleCouplingCondType& cond_type  ///< condition type
            ) override;

        //! build coupling matrices and assemble terms for Nitsche's (NIT) method
        void nit_evaluate_coupling(
            const Core::LinAlg::Matrix<nsd_, 1>&
                normal,                     ///< outward pointing normal (defined by the coupling
                                            ///< partner, that determines the interface traction)
            const double& timefacfac,       ///< theta*dt*fac
            const double& pres_timefacfac,  ///< scaling for pressure part
            const double& visceff_m,        ///< viscosity in coupling master fluid
            const double& visceff_s,        ///< viscosity in coupling slave fluid
            const double& density_m,        ///< fluid density (master) USED IN XFF
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m,  ///< spatial derivatives of coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                vderxy_m,          ///< coupling master spatial velocity derivatives
            const double& pres_m,  ///< coupling master pressure
            const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< coupling master interface velocity
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height
                               ///< for coupled problems
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump,  ///< prescribed interface traction, jump
                                 ///< height for coupled problems
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                proj_tangential,  ///< tangential projection matrix
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                LB_proj_matrix,  ///< prescribed projection matrix for laplace-beltrami problems
            const std::vector<Core::LinAlg::SerialDenseMatrix>&
                solid_stress,  ///< structural cauchy stress and linearization
            std::map<XFEM::CoupTerm, std::pair<bool, double>>&
                configmap  ///< Interface Terms configuration map
            ) override;

        //! add rhs contributions from old time step in Nitsche's (NIT) method
        void nit_evaluate_coupling_old_state(
            const Core::LinAlg::Matrix<nsd_, 1>&
                normal,  ///< outward pointing normal (defined by the coupling partner, that
                         ///< determines the interface traction)
            const double& timefacfac,                      ///< dt*(1-theta)*fac
            bool isImplPressure,                           ///< flag for implicit pressure treatment
            const double& visceff_m,                       ///< viscosity in coupling master fluid
            const double& visceff_s,                       ///< viscosity in coupling slave fluid
            const double& density_m,                       ///< fluid density (master) USED IN XFF
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m,  ///< spatial derivatives of coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                vderxy_m,          ///< coupling master spatial velocity derivatives
            const double& pres_m,  ///< coupling master pressure
            const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< coupling master interface velocity
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height
                               ///< for coupled problems
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                proj_tangential,  ///< tangential projection matrix
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump,  ///< prescribed interface traction, jump
                                 ///< height for coupled problems
            std::map<XFEM::CoupTerm, std::pair<bool, double>>&
                configmap  ///< Interface Terms configuration map
            ) override;

       private:
        //! evaluate traction-consistency term for Nitsche's method
        void nit_traction_consistency_term(
            const Core::LinAlg::Matrix<nen_, 1>&
                funct_m_timefacfac_ks,  ///< funct * timefacfac *kappa_s
            const Core::LinAlg::Matrix<slave_nen_, 1>&
                funct_s_timefacfac_km,  ///< funct_s * timefacfac *kappa_m
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction, jump height for coupled problems
        );

        //! evaluate traction-consistency term for Nitsche's method (for integration by parts
        //! approach)
        void nit_projected_traction_consistency_term(
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m_timefacfac_ks,  ///< master shape function derivatives * timefacfac *
                                        ///< kappa_s
            const Core::LinAlg::Matrix<nsd_, slave_nen_>&
                derxy_s_timefacfac_km,  ///< slave shape function derivatives * timefacfac * kappa_m
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                itraction_jump_matrix  ///< prescribed projection matrix
        );

        //! evaluate pressure-consistency term for Nitsche's method (scaled with master side's
        //! weighting factor)
        void nit_p_consistency_master_terms(const double& pres_m,    ///< master pressure
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,            ///< funct
            const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
            const std::pair<bool, double>& m_row,                    ///< scaling for master row
            const std::pair<bool, double>& s_row,                    ///< scaling for slave row
            const std::pair<bool, double>& m_col,                    ///< scaling for master col
            bool only_rhs = false                                    ///< evaluat only rhs
        );

        //! evaluate pressure-consistency term for Nitsche's method (scaled with slave side's
        //! weighting factor)
        void nit_p_consistency_slave_terms(const double& pres_s,  ///< slave pressure
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,         ///< funct
            const Core::LinAlg::Matrix<nsd_, 1>&
                normal_timefacfac_ks,              ///< normal vector * timefacfac
            const std::pair<bool, double>& m_row,  ///< scaling for master row
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& s_col,  ///< scaling for slave col
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //! evaluate pressure-adjoint-consistency term for Nitsche's method (scaled with master
        //! side's weighting factor)
        void nit_p_adjoint_consistency_master_terms(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,            ///< funct
            const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
            const double&
                velint_diff_normal_timefacfac,     ///< (velint_m - velint_s) * normal * timefacfac
            const std::pair<bool, double>& m_row,  ///< scaling for master row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col,  ///< scaling for slave row
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //! evaluate pressure-adjoint-consistency term for Nitsche's method (scaled with slave
        //! side's weighting factor)
        void nit_p_adjoint_consistency_slave_terms(
            const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
            const double&
                velint_diff_normal_timefacfac,     ///< (velint_m - velint_s) * normal * timefacfac
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col,  ///< scaling for slave col
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //! evaluate viscous-consistency term for Nitsche's method (scaled with master side's
        //! weighting factor)
        void nit_visc_consistency_master_terms(
            const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,  ///< master deriv
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,     ///< funct_m
            const std::pair<bool, double>& m_row,             ///< scaling for master row
            const std::pair<bool, double>& s_row,             ///< scaling for slave row
            const std::pair<bool, double>& m_col,             ///< scaling for master col
            bool only_rhs = false                             ///< evaluat only rhs
        );

        //! evaluate viscous-consistency term for Nitsche's method (scaled with master side's
        //! weighting factor)
        void nit_visc_consistency_master_terms_projected(
            const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,      ///< master deriv
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,         ///< funct_m
            const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,  ///< projection matrix
            const double& km_viscm_fac,                           ///< scaling factor
            const std::pair<bool, double>& m_row,                 ///< scaling for master row
            const std::pair<bool, double>& s_row,                 ///< scaling for slave row
            const std::pair<bool, double>& m_col                  ///< scaling for master col
        );

        //! evaluate viscous-consistency term for Nitsche's method (scaled with slave side's
        //! weighting factor)
        void nit_visc_consistency_slave_terms(const Core::LinAlg::Matrix<nsd_, slave_nen_>&
                                                  derxy_s,  ///< slave shape function derivatives
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,   ///< funct_m
            const std::pair<bool, double>& m_row,           ///< scaling for master row
            const std::pair<bool, double>& s_row,           ///< scaling for slave row
            const std::pair<bool, double>& s_col,           ///< scaling for slave col
            bool only_rhs = false                           ///< evaluat only rhs
        );

        //! evaluate solid-consistency term for Nitsche's method (scaled with slave side's weighting
        //! factor)
        void nit_solid_consistency_slave_terms(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct_m
            const double& timefacfac,                      ///< theta*dt*fac
            const std::pair<bool, double>& m_row,          ///< scaling for master row
            const std::pair<bool, double>& s_row,          ///< scaling for slave row
            const std::pair<bool, double>& s_col,          ///< scaling for slave col
            bool only_rhs = false                          ///< evaluat only rhs
        );

        //! evaluate projected solid-consistency term for Nitsche's method (scaled with slave side's
        //! weighting factor)
        void nit_solid_consistency_slave_terms_projected(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,         ///< funct_m
            const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,  ///< projection matrix
            const double& timefacfac,                             ///< theta*dt*fac
            const std::pair<bool, double>& m_row,                 ///< scaling for master row
            const std::pair<bool, double>& s_row,                 ///< scaling for slave row
            const std::pair<bool, double>& s_col,                 ///< scaling for slave col
            bool only_rhs = false                                 ///< evaluat only rhs
        );

        //! evaluate solid-adjoint-consistency term for Nitsche's method (scaled with slave side's
        //! weighting factor)
        void nit_solid_adjoint_consistency_slave_terms(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,      ///< funct_m
            const double& timefacfac,                          ///< theta*dt*fac
            const Core::LinAlg::Matrix<nsd_, 1>& velint_diff,  ///< (velint_m - velint_s)
            const Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_>&
                dtraction_vel,  ///< derivative of solid traction w.r.t. velocities
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col,  ///< scaling for slave col
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //! evaluate projected solid-adjoint-consistency term for Nitsche's method (scaled with
        //! slave side's weighting factor)
        void nit_solid_adjoint_consistency_slave_terms_projected(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,           ///< funct_m
            const double& timefacfac,                               ///< theta*dt*fac
            const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,    ///< projection matrix
            const Core::LinAlg::Matrix<nsd_, 1>& proj_velint_diff,  ///< P^T*(velint_m - velint_s)
            const Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_>&
                dtraction_vel,  ///< derivative of solid traction w.r.t. velocities
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col,  ///< scaling for slave col
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //! evaluate viscous-adjoint-consistency term for Nitsche's method (scaled with master
        //! side's weighting factor)
        void nit_visc_adjoint_consistency_master_terms(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct * timefacfac
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m,  ///< spatial derivatives of coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal-vector
            const double& viscm_fac,                      ///< scaling factor
            const std::pair<bool, double>& m_row,         ///< scaling for master row
            const std::pair<bool, double>& m_col,         ///< scaling for master col
            const std::pair<bool, double>& s_col,         ///< scaling for slave col
            bool only_rhs = false                         ///< evaluat only rhs
        );

        //! evaluate viscous-adjoint-consistency term for Nitsche's method (scaled with master
        //! side's weighting factor)
        void nit_visc_adjoint_consistency_master_terms_projected(
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m_viscm_timefacfac_km,  ///< master shape function derivatives * timefacfac *
                                              ///< 2 * mu_m * kappa_m
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct_m
            const Core::LinAlg::Matrix<nsd_, 1>& normal,   ///< normal vector
            const std::pair<bool, double>& m_row,          ///< scaling for master row
            const std::pair<bool, double>& m_col,          ///< scaling for master col
            const std::pair<bool, double>& s_col           ///< scaling for slave col
        );

        //! evaluate traction-traction term
        void nit_visc_neumann_adjoint_consistency_master_terms_projected(
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m_viscm_timefacfac_km,  ///< master shape function derivatives * timefacfac *
                                              ///< 2 * mu_m * kappa_m
            const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,  ///< master deriv
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                vderxy_m,  ///< coupling master spatial velocity derivatives
            const Core::LinAlg::Matrix<nen_, 1>&
                funct_m,                                  ///< embedded element funct *mu*timefacfac
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const std::pair<bool, double>& m_row,         ///< scaling for master row
            const std::pair<bool, double>& mstr_col       ///< scaling for master col
        );

        //! evaluate viscous-adjoint-consistency term for Nitsche's method (scaled with slave side's
        //! weighting factor)
        void nit_visc_adjoint_consistency_slave_terms(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct_m
            const Core::LinAlg::Matrix<nsd_, slave_nen_>&
                derxy_s_viscs_timefacfac_ks,  ///< master shape function derivatives * timefacfac *
                                              ///< 2 * mu_m * kappa_m
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const std::pair<bool, double>& s_row,         ///< scaling for slave row
            const std::pair<bool, double>& m_col,         ///< scaling for master col
            const std::pair<bool, double>& s_col,         ///< scaling for slave col
            bool only_rhs = false                         ///< evaluat only rhs
        );

        //!  evaluate Nitsche's penalty term
        void nit_stab_penalty(const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
            const double& timefacfac,              ///< time integration factor
            const std::pair<bool, double>& m_row,  ///< scaling for master row
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col,  ///< scaling for slave col
            bool only_rhs = false                  ///< evaluat only rhs
        );

        //!  evaluate linearization of Nitsche's penalty term
        void nit_stab_penalty_lin(const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
            const double& timefacfac,              ///< time integration factor
            const std::pair<bool, double>& m_row,  ///< scaling for master row
            const std::pair<bool, double>&
                m_row_linm1,  ///< linearization of scaling for master row w.r.t. master comp. one
            const std::pair<bool, double>&
                m_row_linm2,  ///< linearization of scaling for master row w.r.t. master comp. two
            const std::pair<bool, double>&
                m_row_linm3,  ///< linearization of scaling for master row w.r.t. master comp. three
            bool only_rhs = false  ///< evaluat only rhs
        );

        //!  evaluate Nitsche's penalty term
        void nit_stab_penalty_projected(const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
            const Core::LinAlg::Matrix<nsd_, nsd_>& projection_matrix,  ///< projection_matrix
            const Core::LinAlg::Matrix<nsd_, 1>&
                velint_diff_proj_matrix,           ///< velocity difference projected
            const double& timefacfac,              ///< time integration factor
            const std::pair<bool, double>& m_row,  ///< scaling for master row
            const std::pair<bool, double>& s_row,  ///< scaling for slave row
            const std::pair<bool, double>& m_col,  ///< scaling for master col
            const std::pair<bool, double>& s_col   ///< scaling for slave col
        );

        //! add stabilizing terms due to cross-interface convective mass transport (fluid-fluid
        //! only)
        void nit_stab_inflow_averaged_term(const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
            const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< master velocity
            const Core::LinAlg::Matrix<nsd_, 1>& normal,    ///< normal vector n^m
            const double& density,                          ///< fluid density
            const double& timefacfac,                       ///< timefac * fac
            bool only_rhs = false                           ///< evaluat only rhs
        );

        //! Do Nitsche consistency and adjoint consistency terms with projection
        void do_nit_visc_adjoint_and_neumann_master_terms_projected(
            const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct * timefacfac
            const Core::LinAlg::Matrix<nsd_, nen_>&
                derxy_m,  ///< spatial derivatives of coupling master shape functions
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                vderxy_m,  ///< coupling master spatial velocity derivatives
            const Core::LinAlg::Matrix<nsd_, nsd_>& projection_matrix,  ///< projection_matrix
            const Core::LinAlg::Matrix<nsd_, 1>&
                velint_diff_proj_matrix,                  ///< velocity difference projected
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal-vector
            const double& km_viscm_fac,                   ///< scaling factor
            const std::pair<bool, double>& m_row,         ///< scaling for master row
            const std::pair<bool, double>& m_col,         ///< scaling for master col
            const std::pair<bool, double>& s_col,         ///< scaling for slave col
            const std::pair<bool, double>& mstr_col       ///< scaling for master stress col
        );


       private:
        void nit_create_standard_projection_matrices(
            const Core::LinAlg::Matrix<nsd_, 1>& normal  ///< normal vector
        );

        // Get the stabilization parameters for the specific problem.
        void get_stabilization_parameters(
            const double& NIT_full_stab_fac,  ///< full Nitsche stab fac
            const double& NIT_visc_stab_fac,  ///< viscous Nitsche stab fac
            double& stabnit,                  ///< stabilization factor NIT_Penalty
            double& stabepsnit,               ///< stabilization factor NIT_Penalty Neumann
            double& stabadj,                  ///< stabilization factor Adjoint
            double& stabepsadj,               ///< stabilization factor Adjoint Neumann
            const bool& sliplength_not_zero   ///< bool for now, add a inpar for more options?
        );

        //! @name useful constants for DOF-index numbering
        //@{
        static constexpr unsigned Velx = 0;
        static constexpr unsigned Vely = 1;
        static constexpr unsigned Velz = 2;
        static constexpr unsigned Pres = 3;
        //@}

        /// get global master row/col-index of element coupling matrix for a given node index and
        /// dof-index
        static unsigned m_index(unsigned inod, unsigned idof)
        {
          //          if (idof >= master_numdof_)
          //             FOUR_C_THROW("Coupling master element has only {} dof!", master_numdof_);
          //          if (inod >= nen_)
          //             FOUR_C_THROW("Coupling master element has only {} nodes!", nen_);
          return inod * master_numdof_ + idof;
        }

        /// get global slave row/col-index of element coupling matrix for a given node index and
        /// dof-index
        static unsigned s_index(unsigned inod, unsigned idof)
        {
          //          if (idof >= slave_numdof)
          //             FOUR_C_THROW("Coupling slave element has only {} dof!", slave_numdof);
          //          if (inod >= slave_nen_)
          //             FOUR_C_THROW("Coupling slave element has only {} nodes!", slave_nen_);
          return inod * slave_numdof + idof;
        }

        //! @name get global master row/col-index of element coupling matrix for a given node index
        //@{
        static unsigned m_velx(unsigned inod) { return inod * master_numdof_ + Velx; }
        static unsigned m_vely(unsigned inod) { return inod * master_numdof_ + Vely; }
        static unsigned m_velz(unsigned inod) { return inod * master_numdof_ + Velz; }
        static unsigned m_pres(unsigned inod) { return inod * master_numdof_ + Pres; }
        //@}

        /// @name get global slave row/col-index of element coupling matrix for a given node index
        //@{
        static unsigned s_velx(unsigned inod) { return inod * slave_numdof + Velx; }
        static unsigned s_vely(unsigned inod) { return inod * slave_numdof + Vely; }
        static unsigned s_velz(unsigned inod) { return inod * slave_numdof + Velz; }
        static unsigned s_pres(unsigned inod) { return inod * slave_numdof + Pres; }
        //@}

        /// specific XFEM based fluid parameters
        const Discret::Elements::FluidEleParameterXFEM& fldparaxfem_;

        Core::LinAlg::Matrix<master_numdof_ * nen_, master_numdof_ * nen_>
            c_umum_;  ///< coupling matrix C_umum
        Core::LinAlg::Matrix<slave_numdof * slave_nen_, master_numdof_ * nen_>
            c_usum_;  ///< coupling matrix C_usum
        Core::LinAlg::Matrix<master_numdof_ * nen_, slave_numdof * slave_nen_>
            c_umus_;  ///< coupling matrix C_umus
        Core::LinAlg::Matrix<slave_numdof * slave_nen_, slave_numdof * slave_nen_>
            c_usus_;                                                  ///< coupling matrix C_usus
        Core::LinAlg::Matrix<master_numdof_ * nen_, 1> rh_c_um_;      ///< coupling rhs rhC_um
        Core::LinAlg::Matrix<slave_numdof * slave_nen_, 1> rh_c_us_;  ///< coupling rhs rhC_us

        /// scaling of Nitsche's adjoint viscous term
        const double adj_visc_scale_;

        const bool eval_coupling_;  ///< do we have to evaluate coupling terms?

        Core::LinAlg::Matrix<nsd_, 1> velint_s_;  ///< velocity at integration point on slave side

        Core::LinAlg::Matrix<slave_nen_, 1> funct_s_;

        Core::LinAlg::Matrix<slave_nen_, 1> funct_s_timefacfac_km_;

        Core::LinAlg::Matrix<nen_, 1> funct_m_timefacfac_ks_;

        Core::LinAlg::Matrix<nen_, nen_> funct_m_m_dyad_;

        Core::LinAlg::Matrix<slave_nen_, nen_> funct_s_m_dyad_;

        Core::LinAlg::Matrix<slave_nen_, slave_nen_> funct_s_s_dyad_;

        Core::LinAlg::Matrix<nsd_, nen_>
            derxy_m_viscm_timefacfac_;  // dN^(nen)/dx_i * mu_m * kappa_m

        Core::LinAlg::Matrix<nsd_, 1> normal_pres_timefacfac_;

        Core::LinAlg::Matrix<nsd_, 1> normal_pres_timefacfac_km_;

        Core::LinAlg::Matrix<nsd_, 1> normal_pres_timefacfac_ks_;

        Core::LinAlg::Matrix<nsd_, 1> half_normal_;

        Core::LinAlg::Matrix<nsd_, 1> half_normal_viscm_timefacfac_km_;

        Core::LinAlg::Matrix<nsd_, 1> half_normal_viscs_timefacfac_ks_;

        Core::LinAlg::Matrix<nen_, 1> half_normal_deriv_m_viscm_timefacfac_km_;

        Core::LinAlg::Matrix<slave_nen_, 1> half_normal_deriv_s_viscs_timefacfac_ks_;

        Core::LinAlg::Matrix<nen_, 1>
            normal_deriv_m_viscm_km_;  // dN^(nen)/dx_i * n * mu_m * kappa_m

        Core::LinAlg::Matrix<slave_nen_, 1> normal_deriv_s_viscs_ks_;

        Core::LinAlg::Matrix<nsd_, 1> vderxy_m_normal_;

        Core::LinAlg::Matrix<nsd_, 1> vderxy_s_normal_;

        Core::LinAlg::Matrix<nsd_, 1> vderxy_m_normal_transposed_viscm_timefacfac_km_;

        Core::LinAlg::Matrix<nsd_, 1> vderxy_s_normal_transposed_viscs_timefacfac_ks_;

        // Velocity difference between slave and master
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_;
        // Velocity difference between slave and master for normal projection
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_normal_;
        // Velocity difference between slave and master for tangential projection
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_tangential_;

        // Only exists in NIT_Stab_Penalty_MasterTerms and
        //               nit_stab_penalty
        //   i.e. probably should define, normal and tangential components
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_timefacfac_stabfac_;

        // Needed members for GNBC
        //  Projection matrices:
        Core::LinAlg::Matrix<nsd_, nsd_> proj_tangential_;
        Core::LinAlg::Matrix<nsd_, nsd_> proj_normal_;
        Core::LinAlg::Matrix<nsd_, nsd_> proj_matrix_;

        // Needed members for solid sided fsi
        Core::LinAlg::Matrix<nsd_, 1> traction_;
        Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_> dtraction_vel_;
        Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_ * slave_nen_> d2traction_vel_[3];

        // Projected
        Core::LinAlg::Matrix<nsd_, 1> vderxy_x_normal_transposed_viscx_timefacfac_kx_pmatrix_;

        // Help variables:
        Core::LinAlg::Matrix<nsd_, nsd_> velint_proj_norm_diff_dyad_normal_,
            velint_proj_norm_diff_dyad_normal_symm_;
        Core::LinAlg::Matrix<nsd_, 1> vderxy_m_normal_tang_, vderxy_m_normal_transposed_;

        //  projected velocity components:
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_proj_normal_;
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_proj_tangential_;
        Core::LinAlg::Matrix<nsd_, 1> velint_diff_proj_matrix_;

        //  projected traction jump components:
        Core::LinAlg::Matrix<nsd_, 1> itraction_jump_proj_matrix_;

        // ConsistencyNeumann help variable
        Core::LinAlg::Matrix<nsd_, nen_>
            proj_matrix_derxy_m_;  // (beta) * 2.0 * mu_m * timefacefac * km * p_1(IX,j)

        // AdjointNeumann help variables
        Core::LinAlg::Matrix<nen_, 1> normal_deriv_m_;  // 2.0 * half_normal(k) * derxy_m(k,ix)
        Core::LinAlg::Matrix<nen_, nen_>
            derxy_m_p_derxy_m_;  // 2.0 * derxy_m(j,IC) P^t_{jk} *
                                 // derxy_m(k,IR) * mu_m * timefacfac * km

        double velint_diff_normal_pres_timefacfac_;
        double velint_diff_pres_timefacfac_;
      };

      /// concrete class for interface coupling using mixed/hybrid stress-based Lagrange multipliers
      /// method
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      class HybridLMCoupling
          : public HybridLMInterface<distype>,
            public SlaveElementRepresentation<distype, slave_distype, slave_numdof>
      {
       public:
        /// number of nodes per master element
        static constexpr unsigned nen_ =
            SlaveElementRepresentation<distype, slave_distype, slave_numdof>::nen_;
        /// number of spatial dimensions
        static constexpr unsigned nsd_ = SlaveElementInterface<distype>::nsd_;
        /// number of nodal dof for master element (always a fluid element)
        static constexpr unsigned master_numdof_ = nsd_ + 1;
        /// number of slave element's nodes
        static constexpr unsigned slave_nen_ =
            SlaveElementRepresentation<distype, slave_distype, slave_numdof>::slave_nen_;
        /// number of independent stress-dof
        static constexpr unsigned numstressdof_ = HybridLMInterface<distype>::numstressdof_;

        //! ctor for one-sided (xfluid weak dirichlet) problems (interface defined by level-set
        //! field)
        HybridLMCoupling(bool isViscAdjointSymmetric = true)
            : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(),
              adj_visc_scale_(isViscAdjointSymmetric ? 1.0 : -1.0)
        {
        }

        //! ctor for xfluid weak dirichlet problem
        HybridLMCoupling(Core::LinAlg::SerialDenseMatrix&
                             slave_xyze,  ///< global node coordinates of slave element
            bool isViscAdjointSymmetric =
                true  ///< symmetric or skew-symmetric formulation of Nitsche's adjoint viscous term
            )
            : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(slave_xyze),
              adj_visc_scale_(isViscAdjointSymmetric ? 1.0 : -1.0)
        {
        }

        //! ctor for fluid-fluid
        HybridLMCoupling(Core::LinAlg::SerialDenseMatrix&
                             slave_xyz,               ///< global node coordinates of slave element
            Core::LinAlg::SerialDenseMatrix& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_us,  ///< C_us coupling rhs
            Core::LinAlg::SerialDenseMatrix& G_s_us,  ///< \f$G_{u^s \sigma}\f$ coupling matrix
            Core::LinAlg::SerialDenseMatrix& G_us_s,  ///< \f$G_{\sigma u^s}\f$ coupling matrix
            bool isViscAdjointSymmetric = false  ///< symmetric or skew-symmetric formulation of
                                                 ///< Nitsche's adjoint viscous term
            )
            : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(slave_xyz),
              C_usum_(C_usum.values(), true),
              C_umus_(C_umus.values(), true),
              rhC_us_(rhC_us.values(), true),
              G_sus_(G_s_us.values(), true),
              G_uss_(G_us_s.values(), true),
              adj_visc_scale_(isViscAdjointSymmetric ? 1.0 : -1.0)
        {
        }

        //! evaluate interface coupling matrices for mixed/hybrid Cauchy stress-based (MHCS)
        //! coupling
        void mhcs_build_coupling_matrices(
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const double& fac,                            ///< integration factor
            const Core::LinAlg::Matrix<nen_, 1>& funct,   ///< shape function
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s,  ///< block rhs vector \f$ rhs_{\sigma} \f$
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity or interface jump height
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction or interface jump height
            ) override;

        //! evaluate interface matrices for mixed/hybrid viscous stress-based (MHVS) coupling
        void mhvs_build_coupling_matrices(
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const double& fac,                            ///< integration factor
            const Core::LinAlg::Matrix<nen_, 1>& funct,   ///< background element shape functions
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s,                                ///< block rhs vector \f$ rhs_{\sigma}\f$
            const double& press,                      ///< background element pressure
            Core::LinAlg::Matrix<nen_, 1>& rhs_pmus,  ///< part of block rhs vector \f$rhs_p\f$
                                                      ///< including interface velocity terms
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity or interface jump height
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction or interface jump height
            ) override;

        //! apply the standard consistency traction interface jump term
        void mh_traction_consistency_term(
            const Core::LinAlg::Matrix<slave_nen_, 1>&
                funct_s_timefacfac_km,  ///< funct_s * timefacfac *kappa_m
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction, jump height for coupled problems
        );

        //! build the final coupling matrices for mixed/hybrid Cauchy or viscous stress-based
        //! coupling (MHCS or MHVS)
        void hybrid_lm_build_final_coupling_matrices(
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                numstressdof_>& BinvK_ss,  ///< block inverse \f$ K^{-1}_{\sigma\sigma} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, master_numdof_,
                numstressdof_>&
                BKumsInvKss,  ///< block matrix \f$ K_{u\sigma} \cdot K^{-1}_{\sigma\sigma} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                master_numdof_>& BK_sum,  ///< block matrix \f$ K_{\sigma u} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s  ///< block rhs vector \f$ rhs_{\sigma}\f$
            ) override;

       protected:
        //! @name useful constants for DOF-index numbering
        //@{
        static constexpr unsigned Velx = 0;
        static constexpr unsigned Vely = 1;
        static constexpr unsigned Velz = 2;
        static constexpr unsigned Pres = 3;
        //@}

        /// get stress dof-index
        static unsigned stress_index(unsigned xi, unsigned xj)
        {
          if (xi > 2 || xj > 2)
            FOUR_C_THROW("Invalid index combination ({},{}) for stress tensor!", xi, xj);

          return (xi * xj > 0) ? xi + xj + 1 : xi + xj;
        }

        /// get global master row/col-index of element coupling matrix for a given node index and
        /// dof-index
        static unsigned m_index(unsigned inod, unsigned idof)
        {
          //          if (idof >= master_numdof_)
          //             FOUR_C_THROW("Coupling master element has only {} dof!", master_numdof_);
          //          if (inod >= nen_)
          //             FOUR_C_THROW("Coupling master element has only {} nodes!", nen_);
          return inod * master_numdof_ + idof;
        }

        /// get global slave row/col-index of element coupling matrix for a given node index and
        /// dof-index
        static unsigned s_index(unsigned inod, unsigned idof)
        {
          //          if (idof >= slave_numdof)
          //             FOUR_C_THROW("Coupling slave element has only {} dof!", slave_numdof);
          //          if (inod >= slave_nen_)
          //             FOUR_C_THROW("Coupling slave element has only {} nodes!", slave_nen_);
          return inod * slave_numdof + idof;
        }

        Core::LinAlg::Matrix<slave_numdof * slave_nen_, master_numdof_ * nen_>
            C_usum_;  ///< coupling matrix C_usum
        Core::LinAlg::Matrix<master_numdof_ * nen_, slave_numdof * slave_nen_>
            C_umus_;                                                 ///< coupling matrix C_umus
        Core::LinAlg::Matrix<slave_numdof * slave_nen_, 1> rhC_us_;  ///< coupling rhs rhC_us
        Core::LinAlg::Matrix<numstressdof_ * nen_, slave_numdof * slave_nen_>
            G_sus_;  ///< G_sus coupling matrix
        Core::LinAlg::Matrix<slave_numdof * slave_nen_, numstressdof_ * nen_>
            G_uss_;  ///< G_uss coupling matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, slave_nen_>, numstressdof_, nsd_>
            BG_sus_;  ///< block G_sus coupling matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<slave_nen_, nen_>, nsd_, numstressdof_>
            BG_uss_;  ///< block G_uss coupling matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, slave_nen_>, 1, nsd_>
            BG_pmus_;  ///< block G_pmus coupling matrix
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<slave_nen_, nen_>, nsd_, 1>
            BG_uspm_;  ///< block G_uspm coupling matrix

        /// scaling of Nitsche's adjoint viscous term
        const double adj_visc_scale_;
      };
    }  // namespace XFLUID
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
