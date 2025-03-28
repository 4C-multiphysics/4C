// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_CALC_XFEM_COUPLING_HPP
#define FOUR_C_FLUID_ELE_CALC_XFEM_COUPLING_HPP

/*
 * When we enforce coupling conditions or dirichlet conditions on embedded interfaces in the XFEM
 * weakly, we obtain contributions from coupling terms, that have to appear in the element matrix &
 * rhs of the XFEM-element (further denoted as 'master') and the element we want to couple with
 * (named 'slave', can be an XFEM-element itself!). We try to keep this interface as abstract as
 * possible, such that we can use the same routines for xfluid WDBC-problems, xfluid-fluid coupling,
 * XFSI and two-phase flow (and future applications). The classes in here provide an interface to
 * evaluate the slave element and handle the coupling terms, especially all those, that are added to
 * the slave element system. There are two coupling methods available, which are inherently similar:
 * Nitsche's method and a mixed/hybrid stress-based Lagrange multiplier ("HybridLM") approach.
 * kruse, 08/14
 */

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_linalg_fixedsizeblockmatrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class FluidEleParameterXFEM;

    namespace XFLUID
    {
      /**
       * Interface class for slave element (whole 3D element or just a 2D boundary element).
       * Handles evaluation of slave element's shape functions, provides access to
       * velocities, velocity gradients and pressure at the coupling interface, and more.
       * This class includes all functionalities, that are related to the element to be coupled,
       * not to the coupling method itself.
       * It can be used as a factory for pure coupling slave element representations.
       * These are useful, if you want to access the coupled element
       * without assembling any coupling terms (e.g. in two-sided xfluid-fluid coupling
       * for gauss-point projection or error calculation in case of a given analytical solution).
       */
      template <Core::FE::CellType distype>
      class SlaveElementInterface
      {
       public:
        /// number of spatial dimensions
        static constexpr unsigned nsd_ = Core::FE::dim<distype>;

        //! set names of displacement and velocity states (differ dependent on the slave element)
        static void define_state_names(
            Core::FE::CellType slave_distype,  ///< coupling slave discretization type
            std::string& disp_statename,       ///< name of displacement state at current step
            std::string& vel_statename,        ///< name of velocity state at current step
            std::string& veln_statename        ///< name of velocity state at previous step
        );

        //! create a pure coupling slave element representation
        static std::shared_ptr<SlaveElementInterface<distype>> create_slave_element_representation(
            Core::Elements::Element* slave_ele,  ///< coupling slave element
            Core::LinAlg::SerialDenseMatrix&
                slave_xyz  ///< global node coordinates of coupling slave element
        );

        //! dtor
        virtual ~SlaveElementInterface() = default;

        //! add slave element's displacements and set current element's nodal coordinates
        virtual void add_slave_ele_disp(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm,                ///< local map
            std::vector<double>& mymatrix              ///< slave element displacement vector
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        virtual void add_slave_ele_disp(
            const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
            const std::vector<int>& lm                 ///< local map
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! @name accessors
        //@{

        //! set slave element's nodal velocities
        virtual void set_slave_state(
            const Core::FE::Discretization& slavedis,  ///< embedded discretization
            const std::vector<int>& lm                 ///< local map
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! set slave element's nodal velocities
        virtual void set_slave_staten(
            const Core::FE::Discretization& slavedis,  ///< embedded discretization
            const std::vector<int>& lm                 ///< local map
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        /*!
         * get interface velocity - return 0 velocity (if we don't have a concrete slave
         * element (level-set applications), this is correct - otherwise this method is
         * overloaded!!)
         */
        virtual void get_interface_velnp(
            Core::LinAlg::Matrix<nsd_, 1>& ivelint  ///< interface velocity at coupling slave side
        ) const
        {
          ivelint = Core::LinAlg::Matrix<nsd_, 1>(Core::LinAlg::Initialization::zero);
        };

        /*!
         * get interface velocity - return 0 velocity (if we don't have a concrete slave
         * element (level-set applications), this is correct - otherwise this method is
         * overloaded!!)
         */
        virtual void get_interface_veln(
            Core::LinAlg::Matrix<nsd_, 1>& ivelint  ///< interface velocity at coupling slave side
        ) const
        {
          ivelint = Core::LinAlg::Matrix<nsd_, 1>(Core::LinAlg::Initialization::zero);
        };

        //! get interface pressure
        virtual void get_interface_presnp(double& ipres  ///< interface pressure
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! get interface pressure
        virtual void get_interface_presn(double& ipres  ///< interface pressure
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! get interface velocity gradient
        virtual void get_interface_vel_gradnp(Core::LinAlg::Matrix<nsd_, nsd_>&
                velgradint  ///< interface velocity gradients at coupling slave side
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! get interface velocity gradient
        virtual void get_interface_vel_gradn(Core::LinAlg::Matrix<nsd_, nsd_>&
                velgradint  ///< interface velocity gradients at coupling slave side
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! set state for interface velocity jump
        virtual void set_interface_jump_statenp(
            const Core::FE::Discretization& cutterdis,  ///< cutter discretization
            const std::string state,                    ///< state
            const std::vector<int>& lm                  ///< local map
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! set state for interface velocity jump for previous time step
        virtual void set_interface_jump_staten(
            const Core::FE::Discretization& cutterdis,  ///< cutter discretization
            const std::string state,                    ///< state
            const std::vector<int>& lm                  ///< local map
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! get interface velocity jump at Gaussian point
        virtual void get_interface_jump_velnp(
            Core::LinAlg::Matrix<nsd_, 1>& ivelint_jump  ///< cutter element interface velocity jump
                                                         ///< or prescribed DBC at Gaussian point
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //! get interface velocity jump at Gaussian point for previous time step
        virtual void get_interface_jump_veln(Core::LinAlg::Matrix<nsd_, 1>&
                ivelintn_jump  ///< cutter element interface velocity jump or
                               ///< prescribed DBC at Gaussian point
        ) const
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        };

        //@}

        //! evaluate shape function, derivatives and transformation w.r.t coupling slave element at
        //! gaussian point
        virtual void evaluate(Core::LinAlg::Matrix<nsd_, 1>& xside)
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        }

        //! evaluate shape function, derivatives and transformation w.r.t coupling slave element at
        //! gaussian point
        virtual void evaluate(
            Core::LinAlg::Matrix<nsd_, 1>& xside, Core::LinAlg::Matrix<nsd_, 1>& rst_slave)
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        }

        //! compute interface force for side nodes
        virtual void compute_interface_force(
            Core::LinAlg::SerialDenseVector& iforce,  ///< interface force vector
            Core::LinAlg::Matrix<nsd_, 1>& traction,  ///< traction vector at gaussian point
            const double& fac                         ///< integration factor
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        }

        //! project gaussian point from linearized interface in normal direction onto corresponding
        //! side (2D slave element)
        virtual void project_on_side(
            Core::LinAlg::Matrix<nsd_, 1>&
                x_gp_lin,  ///< global coordinates of gaussian point w.r.t linearized interface
            Core::LinAlg::Matrix<nsd_, 1>& x_side,  ///< projected gaussian point on side
            Core::LinAlg::Matrix<nsd_, 1>&
                xi_side  ///< local coordinates of projected gaussian point w.r.t side
        )
        {
          FOUR_C_THROW("There is no concrete slave element available.");
        }

        //! evaluate element volume
        virtual double eval_element_volume()
        {
          FOUR_C_THROW("There is no concrete slave element available.");
          return 0.0;
        }
      };

      //! factory class for coupling with Nitsche's method
      template <Core::FE::CellType distype>
      class NitscheInterface : virtual public SlaveElementInterface<distype>
      {
       public:
        /// number of nodes per master element
        static constexpr unsigned nen_ = Core::FE::num_nodes<distype>;
        /// number of spatial dimensions
        static constexpr unsigned nsd_ = SlaveElementInterface<distype>::nsd_;
        /// number of nodal DOF for master element (always a xfem-fluid element)
        static constexpr unsigned master_numdof_ = nsd_ + 1;

        static std::shared_ptr<NitscheInterface<distype>> create_nitsche_coupling_x_fluid_wdbc(
            Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! create a coupling interface for Nitsche's method for xfluid weak dirichlet problems
        static std::shared_ptr<NitscheInterface<distype>> create_nitsche_coupling_x_fluid_wdbc(
            Core::Elements::Element* bele,  ///< boundary element
            Core::LinAlg::SerialDenseMatrix::Base&
                bele_xyz,  ///< global node coordinates of boundary element
            Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! create a coupling interface for Nitsche's method for xfluid-sided coupling strategy
        static std::shared_ptr<NitscheInterface<distype>> create_nitsche_coupling_x_fluid_sided(
            Core::Elements::Element* bele,  ///< boundary element
            Core::LinAlg::SerialDenseMatrix::Base&
                bele_xyz,  ///< global node coordinates of boundary element
            Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_usus,  ///< C_usus coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
            Core::LinAlg::SerialDenseMatrix::Base& rhC_us,  ///< C_us coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! create a coupling interface for Nitsche's method for two-sided coupling strategy
        //! (weighted or fully embedded-sided)
        static std::shared_ptr<NitscheInterface<distype>> create_nitsche_coupling_two_sided(
            Core::Elements::Element* vele,  ///< volumetric element to couple with
            Core::LinAlg::SerialDenseMatrix::Base&
                vele_xyz,  ///< global node coordinates of volumetric element
            Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& C_usus,  ///< C_usus coupling matrix
            Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
            Core::LinAlg::SerialDenseMatrix::Base& rhC_us,  ///< C_us coupling rhs
            const Discret::Elements::FluidEleParameterXFEM&
                fldparaxfem  ///< specific XFEM based fluid parameters
        );

        //! add contributions from convective stabilization
        //! this method can also be applied in a non-Nitsche context (e.g. MHVS) by
        //! employing shape functions and velocities from another coupling object (HybridLMCoupling)
        //! slave_ele
        virtual void apply_conv_stab_terms(
            const std::shared_ptr<SlaveElementInterface<distype>>&
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
            const Inpar::XFEM::EleCouplingCondType& cond_type  ///< condition type
            ) = 0;

        //! build coupling matrices for Nitsche's method (NIT)
        virtual void nit_evaluate_coupling(
            const Core::LinAlg::Matrix<nsd_, 1>&
                normal,  ///< outward pointing normal (defined by the coupling partner, that
                         ///< determines the interface traction)
            const double& timefacfac,                      ///< theta*dt*fac
            const double& pres_timefacfac,                 ///< scaling for pressure part
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
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump,  ///< prescribed interface traction, jump
                                 ///< height for coupled problems
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                proj_tangential,  ///< tangential projection matrix
            const Core::LinAlg::Matrix<nsd_, nsd_>&
                LB_proj_matrix,  ///< prescribed projection matrix for laplace-beltrami problems
            const std::vector<Core::LinAlg::SerialDenseMatrix>&
                solid_stress,  ///< structural cauchy stress and linearization
            std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>>&
                configmap  ///< Interface Terms configuration map
            ) = 0;

        //! add rhs contributions from old time step in Nitsche's (NIT) method
        virtual void nit_evaluate_coupling_old_state(
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
            std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>>&
                configmap  ///< Interface Terms configuration map
            ) = 0;
      };

      //! factory class for coupling with mixed/hybrid stress-based Lagrange multipliers
      template <Core::FE::CellType distype>
      class HybridLMInterface : virtual public SlaveElementInterface<distype>
      {
       public:
        /// number of nodes per master (xfluid) element
        static constexpr unsigned nen_ = Core::FE::num_nodes<distype>;
        /// number of spatial dimensions of the master element (xfem-fluid)
        static constexpr unsigned nsd_ = SlaveElementInterface<distype>::nsd_;
        /// number of nodal dof for master element (coupling master is always a fluid element!)
        static constexpr unsigned master_numdof_ = nsd_ + 1;
        /// number of independent stress-dof
        static constexpr unsigned numstressdof_ = Core::FE::DisTypeToNumDeriv2<distype>::numderiv2;

        //! create a coupling interface for mixed/hybrid LM approach for xfluid weak dirichlet
        //! problems
        static std::shared_ptr<HybridLMInterface<distype>> create_hybrid_lm_coupling_x_fluid_wdbc(
            bool is_viscAdjointSymmetric  ///< flag that indicates equal signs of Nitsche's standard
                                          ///< & adjoint viscous term
        );

        //! create a coupling interface for mixed/hybrid LM approach for xfluid weak dirichlet
        //! problems
        static std::shared_ptr<HybridLMInterface<distype>> create_hybrid_lm_coupling_x_fluid_wdbc(
            Core::Elements::Element* bele,  ///< boundary element
            Core::LinAlg::SerialDenseMatrix&
                bele_xyz,                 ///< global node coordinates of boundary element
            bool is_viscAdjointSymmetric  ///< flag that indicates equal signs of Nitsche's standard
                                          ///< & adjoint viscous term
        );

        //! create a coupling interface for mixed/hybrid LM approach for xfluid-sided coupling
        //! strategy
        static std::shared_ptr<HybridLMInterface<distype>> create_hybrid_lm_coupling_x_fluid_sided(
            Core::Elements::Element* bele,  ///< boundary element
            Core::LinAlg::SerialDenseMatrix&
                bele_xyz,  ///< global node coordinates of boundary element
            Core::LinAlg::SerialDenseMatrix& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_us,  ///< C_us coupling rhs
            Core::LinAlg::SerialDenseMatrix& G_s_us,  ///< \f$G_{u^s \sigma}\f$ coupling matrix
            Core::LinAlg::SerialDenseMatrix& G_us_s,  ///< \f$G_{\sigma u^s}\f$ coupling matrix
            bool is_viscAdjointSymmetric  ///< flag that indicates equal signs of Nitsche's standard
                                          ///< & adjoint viscous term
        );

        //! create a coupling interface for mixed/hybrid LM approach for two-sided coupling strategy
        //! (weighted or fully embedded-sided)
        static std::shared_ptr<HybridLMInterface<distype>> create_hybrid_lm_coupling_two_sided(
            Core::Elements::Element* vele,  ///< volumetric element to couple with
            Core::LinAlg::SerialDenseMatrix&
                vele_xyz,  ///< global node coordinates of volumetric element
            Core::LinAlg::SerialDenseMatrix& C_usum,  ///< C_usum coupling matrix
            Core::LinAlg::SerialDenseMatrix& C_umus,  ///< C_umus coupling matrix
            Core::LinAlg::SerialDenseMatrix& rhC_us,  ///< C_us coupling rhs
            Core::LinAlg::SerialDenseMatrix& G_s_us,  ///< \f$G_{u^s \sigma}\f$ coupling matrix
            Core::LinAlg::SerialDenseMatrix& G_us_s,  ///< \f$G_{\sigma u^s}\f$ coupling matrix
            Core::Elements::Element* bele,            ///< boundary element
            Core::LinAlg::SerialDenseMatrix&
                bele_xyz,                 ///< global node coordinates of slave element
            bool is_viscAdjointSymmetric  ///< flag, that indicates equal signs of Nitsche's
                                          ///< standard & adjoint viscous term
        )
        {
          FOUR_C_THROW("Embedded-sided mixed/hybrid stress-based LM is not implemented yet!");
          return nullptr;
        }

        //! evaluate interface matrices for mixed/hybrid Cauchy stress-based (MHCS) coupling
        virtual void mhcs_build_coupling_matrices(
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const double& fac,                            ///< integration factor
            const Core::LinAlg::Matrix<nen_, 1>& funct,   ///< shape function
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s,  ///< block rhs vector \f$ rhs_{\sigma} \f$
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity or interface jump height
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction or interface jump height
            ) = 0;

        //! evaluate interface matrices for mixed/hybrid viscous stress-based (MHVS) coupling
        virtual void mhvs_build_coupling_matrices(
            const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
            const double& fac,                            ///< integration factor
            const Core::LinAlg::Matrix<nen_, 1>& funct,   ///< background element shape functions
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s,            ///< block rhs vector \f$ rhs_{\sigma}\f$
            const double& press,  ///< background element pressure
            Core::LinAlg::Matrix<nen_, 1>&
                rhs_pmus,  ///< contribution to block rhs vector \f$rhs_p\f$
                           ///< (includes interface slave velocity terms)
            const Core::LinAlg::Matrix<nsd_, 1>&
                ivelint_jump,  ///< prescribed interface velocity or interface jump height
            const Core::LinAlg::Matrix<nsd_, 1>&
                itraction_jump  ///< prescribed interface traction or interface jump height
            ) = 0;

        //! build the final coupling matrices for mixed/hybrid Cauchy or viscous stress-based
        //! coupling (MHCS or MHVS)
        virtual void hybrid_lm_build_final_coupling_matrices(
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                numstressdof_>& BinvK_ss,  ///< block inverse \f$ K^{-1}_{\sigma\sigma} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, master_numdof_,
                numstressdof_>&
                BKumsInvKss,  ///< block matrix \f$ K_{u\sigma} \cdot K^{-1}_{\sigma\sigma} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                master_numdof_>& BK_sum,  ///< block matrix \f$ K_{\sigma u} \f$
            Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                rhs_s  ///< block rhs vector \f$ rhs_{\sigma}\f$
            ) = 0;
      };
    }  // end namespace XFLUID
  }  // end namespace Elements
}  // end namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
