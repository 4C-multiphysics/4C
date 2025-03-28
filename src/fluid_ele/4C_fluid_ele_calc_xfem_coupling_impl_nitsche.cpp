// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_xfem_coupling.hpp"
#include "4C_fluid_ele_calc_xfem_coupling_impl.hpp"
#include "4C_fluid_ele_parameter_xfem.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    namespace XFLUID
    {
      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      NitscheCoupling<distype, slave_distype, slave_numdof>::NitscheCoupling(
          Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
          const Discret::Elements::FluidEleParameterXFEM&
              fldparaxfem  ///< specific XFEM based fluid parameters
          )
          : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(),
            fldparaxfem_(fldparaxfem),
            c_umum_(C_umum.values(), true),
            rh_c_um_(rhC_um.values(), true),
            adj_visc_scale_(fldparaxfem_.get_viscous_adjoint_scaling()),
            eval_coupling_(false)
      {
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      NitscheCoupling<distype, slave_distype, slave_numdof>::NitscheCoupling(
          Core::LinAlg::SerialDenseMatrix::Base&
              slave_xyze,  ///< global node coordinates of slave element
          Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
          const Discret::Elements::FluidEleParameterXFEM&
              fldparaxfem  ///< specific XFEM based fluid parameters
          )
          : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(slave_xyze),
            fldparaxfem_(fldparaxfem),
            c_umum_(C_umum.values(), true),
            rh_c_um_(rhC_um.values(), true),
            adj_visc_scale_(fldparaxfem_.get_viscous_adjoint_scaling()),
            eval_coupling_(false)
      {
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      NitscheCoupling<distype, slave_distype, slave_numdof>::NitscheCoupling(
          Core::LinAlg::SerialDenseMatrix::Base&
              slave_xyze,  ///< global node coordinates of slave element
          Core::LinAlg::SerialDenseMatrix::Base& C_umum,  ///< C_umum coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& C_usum,  ///< C_usum coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& C_umus,  ///< C_umus coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& C_usus,  ///< C_usus coupling matrix
          Core::LinAlg::SerialDenseMatrix::Base& rhC_um,  ///< C_um coupling rhs
          Core::LinAlg::SerialDenseMatrix::Base& rhC_us,  ///< C_us coupling rhs
          const Discret::Elements::FluidEleParameterXFEM&
              fldparaxfem  ///< specific XFEM based fluid parameters
          )
          : SlaveElementRepresentation<distype, slave_distype, slave_numdof>(slave_xyze),
            fldparaxfem_(fldparaxfem),
            c_umum_(C_umum.values(), true),
            c_usum_(C_usum.values(), true),
            c_umus_(C_umus.values(), true),
            c_usus_(C_usus.values(), true),
            rh_c_um_(rhC_um.values(), true),
            rh_c_us_(rhC_us.values(), true),
            adj_visc_scale_(fldparaxfem_.get_viscous_adjoint_scaling()),
            eval_coupling_(true)
      {
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::apply_conv_stab_terms(
          const std::shared_ptr<SlaveElementInterface<distype>>&
              slave_ele,  ///< associated slave element coupling object
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,   ///< master shape functions
          const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< vector of slave shape functions
          const Core::LinAlg::Matrix<nsd_, 1>& normal,    ///< normal vector n^b
          const double& density_m,                        ///< fluid density (master)
          const double&
              NIT_stab_fac_conv,  ///< full Nitsche's penalty term scaling (viscous+convective part)
          const double& timefacfac,  ///< theta*dt
          const Core::LinAlg::Matrix<nsd_, 1>&
              ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height for
                             ///< coupled problems
          const Inpar::XFEM::EleCouplingCondType& cond_type  ///< condition type
      )
      {
        if (cond_type == Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID &&
            fldparaxfem_.xff_conv_stab_scaling() == Inpar::XFEM::XFF_ConvStabScaling_none)
          FOUR_C_THROW("Cannot apply convective stabilization terms for XFF_ConvStabScaling_none!");

        // funct_m * timefac * fac * funct_m  * kappa_m (dyadic product)
        funct_m_m_dyad_.multiply_nt(funct_m, funct_m);

        // velint_s
        velint_s_.clear();

        if (eval_coupling_) slave_ele->get_interface_velnp(velint_s_);

        // add the prescribed interface velocity for weak Dirichlet boundary conditions or the jump
        // height for coupled problems
        velint_s_.update(1.0, ivelint_jump, 1.0);

        velint_diff_.update(1.0, velint_m, -1.0, velint_s_, 0.0);


        // REMARK:
        // the (additional) convective stabilization is included in NIT_full_stab_fac
        // (in case of mixed/hybrid LM approaches, we don't compute the penalty term explicitly -
        // it 'evolves'); in that case we therefore don't chose the maximum, but add the penalty
        // term scaled with conv_stab_fac to the viscous counterpart; this happens by calling
        // nit_stab_penalty

        switch (cond_type)
        {
          case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
          case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
          case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
          {
            nit_stab_penalty(
                funct_m, timefacfac, std::pair<bool, double>(true, NIT_stab_fac_conv),  // F_Pen_Row
                std::pair<bool, double>(false, 0.0),                                    // X_Pen_Row
                std::pair<bool, double>(true, 1.0),                                     // F_Pen_Col
                std::pair<bool, double>(false, 0.0)                                     // X_Pen_Col
            );
            break;
          }
          case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
          {
            // funct_s
            std::shared_ptr<SlaveElementRepresentation<distype, slave_distype, slave_numdof>> set =
                std::dynamic_pointer_cast<
                    SlaveElementRepresentation<distype, slave_distype, slave_numdof>>(slave_ele);
            if (set == nullptr)
              FOUR_C_THROW("Failed to cast slave_ele to SlaveElementRepresentation!");
            Core::LinAlg::Matrix<slave_nen_, 1> funct_s;
            set->get_slave_funct(funct_s);

            // funct_s * timefac * fac * funct_s * kappa_s (dyadic product)
            funct_s_s_dyad_.multiply_nt(funct_s_, funct_s);

            funct_s_m_dyad_.multiply_nt(funct_s_, funct_m);

            if (fldparaxfem_.xff_conv_stab_scaling() == Inpar::XFEM::XFF_ConvStabScaling_upwinding)
            {
              nit_stab_penalty(funct_m, timefacfac,
                  std::pair<bool, double>(true, NIT_stab_fac_conv),  // F_Pen_Row
                  std::pair<bool, double>(true, NIT_stab_fac_conv),  // X_Pen_Row
                  std::pair<bool, double>(true, 1.0),                // F_Pen_Col
                  std::pair<bool, double>(true, 1.0)                 // X_Pen_Col
              );
            }

            // prevent instabilities due to convective mass transport across the fluid-fluid
            // interface
            if (fldparaxfem_.xff_conv_stab_scaling() ==
                    Inpar::XFEM::XFF_ConvStabScaling_upwinding ||
                fldparaxfem_.xff_conv_stab_scaling() ==
                    Inpar::XFEM::XFF_ConvStabScaling_only_averaged)
            {
              nit_stab_inflow_averaged_term(funct_m, velint_m, normal, density_m, timefacfac);
            }
            break;
          }
          case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
          {
            FOUR_C_THROW("Convective stabilization in monolithic XFSI is not yet available!");
            break;
          }
          default:
            FOUR_C_THROW(
                "Unsupported coupling condition type. Cannot apply convective stabilization "
                "terms.");
            break;
        }
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_evaluate_coupling(
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
              ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height for
                             ///< coupled problems
          const Core::LinAlg::Matrix<nsd_, 1>&
              itraction_jump,  ///< prescribed interface traction, jump height for coupled problems
          const Core::LinAlg::Matrix<nsd_, nsd_>&
              proj_tangential,  ///< tangential projection matrix
          const Core::LinAlg::Matrix<nsd_, nsd_>&
              LB_proj_matrix,  ///< prescribed projection matrix for laplace-beltrami problems
          const std::vector<Core::LinAlg::SerialDenseMatrix>&
              solid_stress,  ///< structural cauchy stress and linearization
          std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>>&
              configmap  ///< Interface Terms configuration map
      )
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::nit_evaluate_coupling");

        //--------------------------------------------

        // define the coupling between two not matching grids
        // for fluidfluidcoupling
        // domain Omega^m := Coupling master (XFluid)
        // domain Omega^s := Alefluid( or monolithic: structure) ( not available for non-coupling
        // (Dirichlet) )

        // [| v |] := vm - vs
        //  { v }  := kappa_m * vm + kappas * vs = kappam * vm (for Dirichlet coupling km=1.0, ks =
        //  0.0) < v >  := kappa_s * vm + kappam * vs = kappas * vm (for Dirichlet coupling km=1.0,
        //  ks = 0.0)
        //--------------------------------------------

        // Create projection matrices
        //--------------------------------------------
        proj_tangential_ = proj_tangential;
        proj_normal_.put_scalar(0.0);
        for (unsigned i = 0; i < nsd_; i++) proj_normal_(i, i) = 1.0;
        proj_normal_.update(-1.0, proj_tangential_, 1.0);

        half_normal_.update(0.5, normal, 0.0);
        normal_pres_timefacfac_.update(pres_timefacfac, normal, 0.0);

        // get velocity at integration point
        // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
        // interface velocity vector in gausspoint
        velint_s_.clear();

        if (configmap.at(Inpar::XFEM::X_Adj_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_Col).first ||
            configmap.at(Inpar::XFEM::X_Adj_n_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_n_Col).first ||
            configmap.at(Inpar::XFEM::X_Adj_t_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_t_Col).first)
          this->get_interface_velnp(velint_s_);

        // Calc full veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_Row).first)
        {
          velint_diff_.update(configmap.at(Inpar::XFEM::F_Adj_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Adj_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_.update(-1.0, ivelint_jump, 1.0);

#ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT
          Core::LinAlg::Matrix<nsd_, 1> tmp_pval;
          tmp_pval.multiply(proj_normal_, normal_pres_timefacfac_);
          // Project the velocity jump [|u|] in the pressure term with the projection matrix.
          //  Useful if smoothed normals are used (performs better for rotating cylinder case).
          velint_diff_pres_timefacfac_ = velint_diff_.dot(tmp_pval);
#else
          velint_diff_pres_timefacfac_ = velint_diff_.dot(normal_pres_timefacfac_);
#endif
        }

        // Calc normal-veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_n_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_n_Row).first)
        {
          // velint_diff_proj_normal_ = (u^m_k - u^s_k - u^{jump}_k) P^n_{kj}
          // (([|u|]-u_0)*P^n) Apply from right for consistency
          velint_diff_normal_.update(configmap.at(Inpar::XFEM::F_Pen_n_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Pen_n_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_normal_.update(-1.0, ivelint_jump, 1.0);
          velint_diff_proj_normal_.multiply_tn(proj_normal_, velint_diff_normal_);

#ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT
          Core::LinAlg::Matrix<nsd_, 1> tmp_pval;
          tmp_pval.multiply(proj_normal_, normal_pres_timefacfac_);
          // Project the velocity jump [|u|] in the pressure term with the projection matrix.
          //  Useful if smoothed normals are used (performs better for rotating cylinder case).
          velint_diff_normal_pres_timefacfac_ = velint_diff_normal_.dot(tmp_pval);
#else
          velint_diff_normal_pres_timefacfac_ = velint_diff_normal_.dot(normal_pres_timefacfac_);
#endif
        }

        // Calc tangential-veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_t_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_t_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_t_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_t_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_t_Row).first)
        {
          // velint_diff_proj_tangential_ = (u^m_k - u^s_k - u^{jump}_k) P^t_{kj}
          // (([|u|]-u_0)*P^t) Apply from right for consistency
          velint_diff_tangential_.update(configmap.at(Inpar::XFEM::F_Pen_t_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Pen_t_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_tangential_.update(-1.0, ivelint_jump, 1.0);
          velint_diff_proj_tangential_.multiply_tn(proj_tangential_, velint_diff_tangential_);
        }

        // funct_s * timefac * fac
        funct_s_.clear();
        if (slave_distype != Core::FE::CellType::dis_none) this->get_slave_funct(funct_s_);

        // funct_m * timefac * fac * funct_m  * kappa_m (dyadic product)
        funct_m_m_dyad_.multiply_nt(funct_m, funct_m);

        // funct_s * timefac * fac * funct_s * kappa_s (dyadic product)
        funct_s_s_dyad_.multiply_nt(funct_s_, funct_s_);

        // funct_s * timefac * fac * funct_m (dyadic product)
        funct_s_m_dyad_.multiply_nt(funct_s_, funct_m);

        //-----------------------------------------------------------------
        // viscous stability term
        // REMARK: this term includes also inflow coercivity in case of XFSI
        // with modified stabfac (see NIT_ComputeStabfac)

        if (configmap.at(Inpar::XFEM::F_Pen_n_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_n_Row).first)
        {
          // Normal Terms!
          nit_stab_penalty_projected(funct_m, proj_normal_, velint_diff_proj_normal_, timefacfac,
              configmap.at(Inpar::XFEM::F_Pen_n_Row), configmap.at(Inpar::XFEM::X_Pen_n_Row),
              configmap.at(Inpar::XFEM::F_Pen_n_Col), configmap.at(Inpar::XFEM::X_Pen_n_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Pen_t_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_t_Row).first)
        {
          // Tangential Terms!
          nit_stab_penalty_projected(funct_m, proj_tangential_, velint_diff_proj_tangential_,
              timefacfac, configmap.at(Inpar::XFEM::F_Pen_t_Row),
              configmap.at(Inpar::XFEM::X_Pen_t_Row), configmap.at(Inpar::XFEM::F_Pen_t_Col),
              configmap.at(Inpar::XFEM::X_Pen_t_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Pen_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_Row).first)
        {
          nit_stab_penalty(funct_m, timefacfac, configmap.at(Inpar::XFEM::F_Pen_Row),
              configmap.at(Inpar::XFEM::X_Pen_Row), configmap.at(Inpar::XFEM::F_Pen_Col),
              configmap.at(Inpar::XFEM::X_Pen_Col));

          if (configmap.at(Inpar::XFEM::F_Pen_Row_linF1).first)
          {
            if (!configmap.at(Inpar::XFEM::F_Pen_Row_linF1).first ||
                !configmap.at(Inpar::XFEM::F_Pen_Row_linF2).first ||
                !configmap.at(Inpar::XFEM::F_Pen_Row_linF3).first)
              FOUR_C_THROW("Linearization for Penalty Term not set for all Components!");

            nit_stab_penalty_lin(funct_m, timefacfac, configmap.at(Inpar::XFEM::F_Pen_Row),
                configmap.at(Inpar::XFEM::F_Pen_Row_linF1),
                configmap.at(Inpar::XFEM::F_Pen_Row_linF2),
                configmap.at(Inpar::XFEM::F_Pen_Row_linF3));
          }
        }

        // add averaged term (TODO: For XFF? How does this work for non-master coupled? @Benedikt?)
        // Todo: is not handled by configmap yet as it has the shape of a penalty term and therefore
        // will be evaluated there at the end!
        if (fldparaxfem_.xff_conv_stab_scaling() == Inpar::XFEM::XFF_ConvStabScaling_upwinding ||
            fldparaxfem_.xff_conv_stab_scaling() == Inpar::XFEM::XFF_ConvStabScaling_only_averaged)
        {
          nit_stab_inflow_averaged_term(funct_m, velint_m, normal, density_m, timefacfac);
        }
        //-------------------------------------------- Nitsche-Stab penalty added
        //-----------------------------------------------------------------

        // evaluate the terms, that contribute to the background fluid
        // system - standard Dirichlet case/pure xfluid-sided case
        // AND
        // system - two-sided or xfluid-sided:

        // 2 * mu_m * timefac * fac
        const double km_viscm_fac = 2.0 * timefacfac * visceff_m;
        half_normal_viscm_timefacfac_km_.update(km_viscm_fac, half_normal_, 0.0);

        half_normal_deriv_m_viscm_timefacfac_km_.multiply_tn(
            derxy_m, half_normal_);  // 0.5*normal(k)*derxy_m(k,ic)
        half_normal_deriv_m_viscm_timefacfac_km_.scale(km_viscm_fac);

        // 0.5* (\nabla u + (\nabla u)^T) * normal
        vderxy_m_normal_.multiply(vderxy_m, half_normal_);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.multiply_tn(vderxy_m, half_normal_);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.update(1.0, vderxy_m_normal_, 1.0);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.scale(km_viscm_fac);

        //-----------------------------------------------------------------
        // pressure consistency term
        if (configmap.at(Inpar::XFEM::F_Con_Col).first)
        {
          nit_p_consistency_master_terms(pres_m, funct_m, normal_pres_timefacfac_,
              configmap.at(Inpar::XFEM::F_Con_Row), configmap.at(Inpar::XFEM::X_Con_Row),
              configmap.at(Inpar::XFEM::F_Con_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Con_n_Col)
                .first)  //(COMMENT: evaluating this separately seems to be more efficient for our
                         // cases)
        {
          nit_p_consistency_master_terms(pres_m, funct_m, normal_pres_timefacfac_,
              configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
              configmap.at(Inpar::XFEM::F_Con_n_Col));
        }

        //-----------------------------------------------------------------
        // viscous consistency term
        if (configmap.at(Inpar::XFEM::F_Con_Col).first)
        {
#ifndef ENFORCE_URQUIZA_GNBC
          // Comment: Here vderxy_m_normal_transposed_viscm_timefacfac_km_ is used!
          nit_visc_consistency_master_terms(derxy_m, funct_m, configmap.at(Inpar::XFEM::F_Con_Row),
              configmap.at(Inpar::XFEM::X_Con_Row), configmap.at(Inpar::XFEM::F_Con_Col));

#else  // Todo: @Magnus, what to do with this?
          nit_visc_consistency_master_terms_projected(derxy_m, funct_m, proj_normal_, km_viscm_fac,
              1.0,   // not to chang anything in ENFORCE_URQUIZA_GNBC case
              1.0,   // not to chang anything in ENFORCE_URQUIZA_GNBC case
              1.0);  // not to chang anything in ENFORCE_URQUIZA_GNBC case
#endif
        }

        if (configmap.at(Inpar::XFEM::F_Con_n_Col).first)
        {
          nit_visc_consistency_master_terms_projected(derxy_m, funct_m, proj_normal_, km_viscm_fac,
              configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
              configmap.at(Inpar::XFEM::F_Con_n_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Con_t_Col).first)
        {
          nit_visc_consistency_master_terms_projected(derxy_m, funct_m, proj_tangential_,
              km_viscm_fac, configmap.at(Inpar::XFEM::F_Con_t_Row),
              configmap.at(Inpar::XFEM::X_Con_t_Row), configmap.at(Inpar::XFEM::F_Con_t_Col));
        }

        //-----------------------------------------------------------------
        // pressure adjoint consistency term
        if (configmap.at(Inpar::XFEM::F_Adj_Row).first)
        {
          //-----------------------------------------------------------------
          // +++ qnuP option added! +++
          nit_p_adjoint_consistency_master_terms(funct_m, normal_pres_timefacfac_,
              velint_diff_pres_timefacfac_, configmap.at(Inpar::XFEM::F_Adj_Row),
              configmap.at(Inpar::XFEM::F_Adj_Col), configmap.at(Inpar::XFEM::X_Adj_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Adj_n_Row)
                .first)  //(COMMENT: evaluating this separately seems to be more efficient for our
                         // cases)
        {
          //-----------------------------------------------------------------
          // +++ qnuP option added! +++
          nit_p_adjoint_consistency_master_terms(funct_m, normal_pres_timefacfac_,
              velint_diff_normal_pres_timefacfac_, configmap.at(Inpar::XFEM::F_Adj_n_Row),
              configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col));
        }

        //-----------------------------------------------------------------
        // viscous adjoint consistency term (and for NavierSlip Penalty Term ([v],{sigma}))
        // Normal Terms!

        if (configmap.at(Inpar::XFEM::F_Adj_n_Row).first)
        {
          do_nit_visc_adjoint_and_neumann_master_terms_projected(funct_m,  ///< funct * timefacfac
              derxy_m,                   ///< spatial derivatives of coupling master shape functions
              vderxy_m,                  ///< coupling master spatial velocity derivatives
              proj_normal_,              ///< projection_matrix
              velint_diff_proj_normal_,  ///< velocity difference projected
              normal,                    ///< normal-vector
              km_viscm_fac, configmap.at(Inpar::XFEM::F_Adj_n_Row),
              configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col),
              configmap.at(Inpar::XFEM::FStr_Adj_n_Col));
        }
        if (configmap.at(Inpar::XFEM::FStr_Adj_n_Col).first)
          FOUR_C_THROW("(NOT SUPPORTED FOR NORMAL DIR! Check Coercivity!)");

        // Tangential Terms!
        if (configmap.at(Inpar::XFEM::F_Adj_t_Row).first)
        {
          do_nit_visc_adjoint_and_neumann_master_terms_projected(funct_m,  ///< funct * timefacfac
              derxy_m,           ///< spatial derivatives of coupling master shape functions
              vderxy_m,          ///< coupling master spatial velocity derivatives
              proj_tangential_,  ///< projection_matrix
              velint_diff_proj_tangential_,  ///< velocity difference projected
              normal,                        ///< normal-vector
              km_viscm_fac, configmap.at(Inpar::XFEM::F_Adj_t_Row),
              configmap.at(Inpar::XFEM::F_Adj_t_Col), configmap.at(Inpar::XFEM::X_Adj_t_Col),
              configmap.at(Inpar::XFEM::FStr_Adj_t_Col));
        }

        if (configmap.at(Inpar::XFEM::F_Adj_Row).first)
        {
          nit_visc_adjoint_consistency_master_terms(funct_m,  ///< funct * timefacfac
              derxy_m,       ///< spatial derivatives of coupling master shape functions
              normal,        ///< normal-vector
              km_viscm_fac,  ///< scaling factor
              configmap.at(Inpar::XFEM::F_Adj_Row), configmap.at(Inpar::XFEM::F_Adj_Col),
              configmap.at(Inpar::XFEM::X_Adj_Col));

          if (configmap.at(Inpar::XFEM::FStr_Adj_Col).first)
            FOUR_C_THROW(
                "visc Adjoint Stress Term without projection not implemented - feel free!");
        }

        if ((configmap.at(Inpar::XFEM::XF_Con_Col).first ||
                configmap.at(Inpar::XFEM::XF_Con_n_Col).first ||
                configmap.at(Inpar::XFEM::XF_Con_t_Col).first ||
                configmap.at(Inpar::XFEM::XF_Adj_Row).first ||
                configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
                configmap.at(Inpar::XFEM::XF_Adj_t_Row).first))
        {
          // TODO: @Christoph:
          //--------------------------------------------------------------------------------
          // This part needs to be adapted if a Robin-condition needs to be applied not only
          //  xfluid_sided (i.e. kappa^m =/= 1.0).
          // Should be more or less analogue to the above implementation.
          //--------------------------------------------------------------------------------

          //-----------------------------------------------------------------
          // the following quantities are only required for two-sided coupling
          // kappa_s > 0.0

          //-----------------------------------------------------------------
          // pressure consistency term

          double pres_s = 0.0;
          // must use this-pointer because of two-stage lookup!
          this->get_interface_presnp(pres_s);

          if (configmap.at(Inpar::XFEM::XF_Con_Col).first)
          {
            nit_p_consistency_slave_terms(pres_s, funct_m, normal_pres_timefacfac_,
                configmap.at(Inpar::XFEM::F_Con_Row), configmap.at(Inpar::XFEM::X_Con_Row),
                configmap.at(Inpar::XFEM::XF_Con_Col));
          }

          if (configmap.at(Inpar::XFEM::XF_Con_n_Col)
                  .first)  //(COMMENT: evaluating this separately seems to be more efficient for our
                           // cases)
          {
            nit_p_consistency_slave_terms(pres_s, funct_m, normal_pres_timefacfac_,
                configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
                configmap.at(Inpar::XFEM::XF_Con_n_Col));
          }

          //-----------------------------------------------------------------
          // pressure adjoint consistency term
          // HAS PROJECTION FOR VELOCITY IMPLEMENTED!!!
          if (configmap.at(Inpar::XFEM::XF_Adj_Row).first)
          {
            nit_p_adjoint_consistency_slave_terms(normal_pres_timefacfac_,
                velint_diff_pres_timefacfac_, configmap.at(Inpar::XFEM::XF_Adj_Row),
                configmap.at(Inpar::XFEM::F_Adj_Col), configmap.at(Inpar::XFEM::X_Adj_Col));
          }
          if (configmap.at(Inpar::XFEM::XF_Adj_n_Row)
                  .first)  //(COMMENT: evaluating this separately seems to be more efficient for our
                           // cases)
          {
            nit_p_adjoint_consistency_slave_terms(normal_pres_timefacfac_,
                velint_diff_normal_pres_timefacfac_, configmap.at(Inpar::XFEM::XF_Adj_n_Row),
                configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col));
          }

          //-----------------------------------------------------------------
          // viscous consistency term

          // Shape function derivatives for slave side
          Core::LinAlg::Matrix<nsd_, slave_nen_> derxy_s;
          this->get_slave_funct_deriv(derxy_s);

          // Spatial velocity gradient for slave side
          Core::LinAlg::Matrix<nsd_, nsd_> vderxy_s;
          this->get_interface_vel_gradnp(vderxy_s);

          // 2 * mu_s * kappa_s * timefac * fac
          const double ks_viscs_fac = 2.0 * visceff_s * timefacfac;
          half_normal_viscs_timefacfac_ks_.update(ks_viscs_fac, half_normal_, 0.0);
          half_normal_deriv_s_viscs_timefacfac_ks_.multiply_tn(
              derxy_s, half_normal_);  // half_normal(k)*derxy_s(k,ic);
          half_normal_deriv_s_viscs_timefacfac_ks_.scale(ks_viscs_fac);
          vderxy_s_normal_.multiply(vderxy_s, half_normal_);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.multiply_tn(vderxy_s, half_normal_);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.update(1.0, vderxy_s_normal_, 1.0);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.scale(ks_viscs_fac);

          if (configmap.at(Inpar::XFEM::XF_Con_Col).first)
          {
            nit_visc_consistency_slave_terms(derxy_s, funct_m, configmap.at(Inpar::XFEM::F_Con_Row),
                configmap.at(Inpar::XFEM::X_Con_Row), configmap.at(Inpar::XFEM::XF_Con_Col));
          }
          if (configmap.at(Inpar::XFEM::XF_Con_n_Col).first ||
              configmap.at(Inpar::XFEM::XF_Con_t_Col).first)
            FOUR_C_THROW("Want to implement projected slave consistency?");

          //-----------------------------------------------------------------
          // viscous adjoint consistency term

          Core::LinAlg::Matrix<nsd_, slave_nen_> derxy_s_viscs_timefacfac_ks(derxy_s);
          derxy_s_viscs_timefacfac_ks.scale(adj_visc_scale_ * ks_viscs_fac);

          // TODO: Needs added Projection. (If deemed necessary!)
          if (configmap.at(Inpar::XFEM::XF_Adj_Row).first)
          {
            nit_visc_adjoint_consistency_slave_terms(funct_m, derxy_s_viscs_timefacfac_ks, normal,
                configmap.at(Inpar::XFEM::XF_Adj_Row), configmap.at(Inpar::XFEM::F_Adj_Col),
                configmap.at(Inpar::XFEM::X_Adj_Col));
          }
          if (configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
              configmap.at(Inpar::XFEM::XF_Adj_t_Row).first)
            FOUR_C_THROW("Want to  implement projected slave adjoint consistency?");

          //-----------------------------------------------------------------
          // standard consistency traction jump term
          // Only needed for XTPF
          if (configmap.at(Inpar::XFEM::F_TJ_Rhs).first ||
              configmap.at(Inpar::XFEM::X_TJ_Rhs).first)
          {
            // funct_s * timefac * fac * kappa_m
            funct_s_timefacfac_km_.update(
                configmap.at(Inpar::XFEM::X_TJ_Rhs).second * timefacfac, funct_s_, 0.0);

            // funct_m * timefac * fac * kappa_s
            funct_m_timefacfac_ks_.update(
                configmap.at(Inpar::XFEM::F_TJ_Rhs).second * timefacfac, funct_m, 0.0);

            nit_traction_consistency_term(
                funct_m_timefacfac_ks_, funct_s_timefacfac_km_, itraction_jump);
          }

          //-----------------------------------------------------------------
          // projection matrix approach (Laplace-Beltrami)
          if (configmap.at(Inpar::XFEM::F_LB_Rhs).first ||
              configmap.at(Inpar::XFEM::X_LB_Rhs).first)
          {
            Core::LinAlg::Matrix<nsd_, slave_nen_> derxy_s_timefacfac_km(derxy_s);
            derxy_s_timefacfac_km.scale(configmap.at(Inpar::XFEM::X_LB_Rhs).second * timefacfac);

            Core::LinAlg::Matrix<nsd_, nen_> derxy_m_timefacfac_ks(derxy_m);
            derxy_m_timefacfac_ks.scale(configmap.at(Inpar::XFEM::F_LB_Rhs).second * timefacfac);

            nit_projected_traction_consistency_term(
                derxy_m_timefacfac_ks, derxy_s_timefacfac_km, LB_proj_matrix);
          }
          //-------------------------------------------- Traction-Jump added (XTPF)
        }

        // Structural Stress Terms (e.g. non xfluid sided FSI)
        if (configmap.at(Inpar::XFEM::XS_Con_Col).first ||
            configmap.at(Inpar::XFEM::XS_Con_n_Col).first ||
            configmap.at(Inpar::XFEM::XS_Con_t_Col).first ||
            configmap.at(Inpar::XFEM::XS_Adj_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_t_Row).first)
        {
          traction_ = Core::LinAlg::Matrix<nsd_, 1>(solid_stress[0].values(), true);
          dtraction_vel_ =
              Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_>(solid_stress[1].values(), true);

          d2traction_vel_[0] = Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_ * slave_nen_>(
              solid_stress[2].values(), true);
          d2traction_vel_[1] = Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_ * slave_nen_>(
              solid_stress[3].values(), true);
          d2traction_vel_[2] = Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_ * slave_nen_>(
              solid_stress[4].values(), true);

          if (configmap.at(Inpar::XFEM::XS_Con_Col).first)
          {
            nit_solid_consistency_slave_terms(funct_m, timefacfac,
                configmap.at(Inpar::XFEM::F_Con_Row), configmap.at(Inpar::XFEM::X_Con_Row),
                configmap.at(Inpar::XFEM::XS_Con_Col));
          }

          if (configmap.at(Inpar::XFEM::XS_Con_n_Col).first)
          {
            nit_solid_consistency_slave_terms_projected(funct_m, proj_normal_, timefacfac,
                configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
                configmap.at(Inpar::XFEM::XS_Con_n_Col));
          }

          if (configmap.at(Inpar::XFEM::XS_Con_t_Col).first)
          {
            nit_solid_consistency_slave_terms_projected(funct_m, proj_tangential_, timefacfac,
                configmap.at(Inpar::XFEM::F_Con_t_Row), configmap.at(Inpar::XFEM::X_Con_t_Row),
                configmap.at(Inpar::XFEM::XS_Con_t_Col));
          }

          if (configmap.at(Inpar::XFEM::XS_Adj_Row).first)
          {
            nit_solid_adjoint_consistency_slave_terms(funct_m, timefacfac, velint_diff_,
                dtraction_vel_, configmap.at(Inpar::XFEM::XS_Adj_Row),
                configmap.at(Inpar::XFEM::F_Adj_Col), configmap.at(Inpar::XFEM::X_Adj_Col));
          }

          if (configmap.at(Inpar::XFEM::XS_Adj_n_Row).first)
          {
            nit_solid_adjoint_consistency_slave_terms_projected(funct_m, timefacfac, proj_normal_,
                velint_diff_proj_normal_, dtraction_vel_, configmap.at(Inpar::XFEM::XS_Adj_n_Row),
                configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col));
          }

          if (configmap.at(Inpar::XFEM::XS_Adj_t_Row).first)
          {
            nit_solid_adjoint_consistency_slave_terms_projected(funct_m, timefacfac,
                proj_tangential_, velint_diff_proj_tangential_, dtraction_vel_,
                configmap.at(Inpar::XFEM::XS_Adj_t_Row), configmap.at(Inpar::XFEM::F_Adj_t_Col),
                configmap.at(Inpar::XFEM::X_Adj_t_Col));
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_solid_consistency_slave_terms(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct_m
          const double& timefacfac,                      ///< theta*dt*fac
          const std::pair<bool, double>& m_row,          ///< scaling for master row
          const std::pair<bool, double>& s_row,          ///< scaling for slave row
          const std::pair<bool, double>& s_col,          ///< scaling for slave col
          bool only_rhs)
      {
        const double facms = m_row.second * s_col.second;
        const double facss = s_row.second * s_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir) * facms * timefacfac;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_um_(m_index(ir, ivel), 0) += tmp_val * traction_(ivel);
          }
        }

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double tmp_val = funct_s_(ir) * facss * timefacfac;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_us_(s_index(ir, ivel), 0) -= tmp_val * traction_(ivel);
          }
        }

        if (only_rhs) return;

        for (unsigned ic = 0; ic < slave_nen_; ++ic)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned col = s_index(ic, jvel);
              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                //-----------------------------------------------
                //    - (vm,
                //-----------------------------------------------
                c_umus_(m_index(ir, ivel), col) -=
                    funct_m(ir) * dtraction_vel_(col, ivel) * facms * timefacfac;
              }

              for (unsigned ir = 0; ir < slave_nen_; ++ir)
              {
                //-----------------------------------------------
                //    + (vs,
                //-----------------------------------------------
                // diagonal block
                c_usus_(s_index(ir, ivel), col) +=
                    funct_s_(ir) * dtraction_vel_(col, ivel) * facss * timefacfac;
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_solid_consistency_slave_terms_projected(const Core::LinAlg::Matrix<nen_,
                                                          1>& funct_m,  ///< funct_m
              const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,      ///< projection matrix
              const double& timefacfac,                                 ///< theta*dt*fac
              const std::pair<bool, double>& m_row,                     ///< scaling for master row
              const std::pair<bool, double>& s_row,                     ///< scaling for slave row
              const std::pair<bool, double>& s_col,                     ///< scaling for slave col
              bool only_rhs)
      {
        static Core::LinAlg::Matrix<nsd_, 1> proj_traction;
        proj_traction.multiply_tn(proj_matrix, traction_);

        const double facms = m_row.second * s_col.second;
        const double facss = s_row.second * s_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir) * facms * timefacfac;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_um_(m_index(ir, ivel), 0) += tmp_val * proj_traction(ivel);
          }
        }

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double tmp_val = funct_s_(ir) * facss * timefacfac;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_us_(s_index(ir, ivel), 0) -= tmp_val * proj_traction(ivel);
          }
        }

        if (only_rhs) return;

        static Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_> proj_dtraction_vel(
            Core::LinAlg::Initialization::zero);
        proj_dtraction_vel.clear();
        for (unsigned col = 0; col < nsd_ * slave_nen_; ++col)
        {
          for (unsigned j = 0; j < nsd_; ++j)
          {
            for (unsigned i = 0; i < nsd_; ++i)
              proj_dtraction_vel(col, j) += dtraction_vel_(col, i) * proj_matrix(i, j);
          }
        }

        for (unsigned ic = 0; ic < slave_nen_; ++ic)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned col = s_index(ic, jvel);
              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                //-----------------------------------------------
                //    - (vm,
                //-----------------------------------------------
                c_umus_(m_index(ir, ivel), col) -=
                    funct_m(ir) * proj_dtraction_vel(col, ivel) * facms * timefacfac;
              }

              for (unsigned ir = 0; ir < slave_nen_; ++ir)
              {
                //-----------------------------------------------
                //    + (vs,
                //-----------------------------------------------
                // diagonal block
                c_usus_(s_index(ir, ivel), col) +=
                    funct_s_(ir) * proj_dtraction_vel(col, ivel) * facss * timefacfac;
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype,
          slave_numdof>::nit_solid_adjoint_consistency_slave_terms(const Core::LinAlg::Matrix<nen_,
                                                                       1>& funct_m,  ///< funct_m
          const double& timefacfac,                          ///< theta*dt*fac
          const Core::LinAlg::Matrix<nsd_, 1>& velint_diff,  ///< (velint_m - velint_s)
          const Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_>&
              dtraction_vel,                     ///< derivative of solid traction w.r.t. velocities
          const std::pair<bool, double>& s_row,  ///< scaling for slave row
          const std::pair<bool, double>& m_col,  ///< scaling for master col
          const std::pair<bool, double>& s_col,  ///< scaling for slave col
          bool only_rhs)
      {
        //
        // RHS: dv<d(sigma)/dv|u*n,uF-uS>
        // Lin: dv<d(sigma)/dv|u*n>duF-dv<d(sigma)/dv|u*n>duS+dv<d(sigma)/dv|du/dus*n,uF-uS>duS

        const double facs = s_row.second * timefacfac * adj_visc_scale_;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = s_index(ir, ivel);
              // rhs
              rh_c_us_(row, 0) += dtraction_vel(row, jvel) * velint_diff(jvel, 0) * facs;
            }
          }
        }

        if (only_rhs) return;

        const double facsm = s_row.second * m_col.second * timefacfac * adj_visc_scale_;
        const double facss = s_row.second * s_col.second * timefacfac * adj_visc_scale_;

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = s_index(ir, ivel);
              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                const unsigned col = m_index(ic, jvel);
                c_usum_(row, col) -= funct_m(ic) * dtraction_vel(row, jvel) * facsm;
              }

              for (unsigned ic = 0; ic < slave_nen_; ++ic)
              {
                const unsigned col = s_index(ic, jvel);
                c_usus_(row, col) += funct_s_(ic) * dtraction_vel(row, jvel) * facss;
                for (unsigned k = 0; k < nsd_; ++k)
                {
                  c_usus_(row, col) -= d2traction_vel_[k](row, col) * velint_diff(k, 0) * facs;
                }
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_solid_adjoint_consistency_slave_terms_projected(
              const Core::LinAlg::Matrix<nen_, 1>& funct_m,           ///< funct_m
              const double& timefacfac,                               ///< theta*dt*fac
              const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,    ///< projection matrix
              const Core::LinAlg::Matrix<nsd_, 1>& proj_velint_diff,  ///< P^T*(velint_m - velint_s)
              const Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_>&
                  dtraction_vel,  ///< derivative of solid traction w.r.t. velocities
              const std::pair<bool, double>& s_row,  ///< scaling for slave row
              const std::pair<bool, double>& m_col,  ///< scaling for master col
              const std::pair<bool, double>& s_col,  ///< scaling for slave col
              bool only_rhs)
      {
        static Core::LinAlg::Matrix<nsd_ * slave_nen_, nsd_> proj_dtraction_vel(
            Core::LinAlg::Initialization::zero);
        if (!only_rhs)
        {
          proj_dtraction_vel.clear();
          for (unsigned col = 0; col < nsd_ * slave_nen_; ++col)
          {
            for (unsigned j = 0; j < nsd_; ++j)
            {
              for (unsigned i = 0; i < nsd_; ++i)
                proj_dtraction_vel(col, j) += dtraction_vel(col, i) * proj_matrix(i, j);
            }
          }
        }

        // Call Std Version with the projected quantities!
        nit_solid_adjoint_consistency_slave_terms(funct_m, timefacfac, proj_velint_diff,
            proj_dtraction_vel, s_row, m_col, s_col, only_rhs);

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_evaluate_coupling_old_state(
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
              ivelint_jump,  ///< prescribed interface velocity, Dirichlet values or jump height for
                             ///< coupled problems
          const Core::LinAlg::Matrix<nsd_, nsd_>&
              proj_tangential,  ///< tangential projection matrix
          const Core::LinAlg::Matrix<nsd_, 1>&
              itraction_jump,  ///< prescribed interface traction, jump height for coupled problems
          std::map<Inpar::XFEM::CoupTerm, std::pair<bool, double>>&
              configmap  ///< Interface Terms configuration map
      )
      {
        //--------------------------------------------

        // define the coupling between two not matching grids
        // for fluidfluidcoupling
        // domain Omega^m := Coupling master (XFluid)
        // domain Omega^s := Alefluid( or monolithic: structure) ( not available for non-coupling
        // (Dirichlet) )

        // [| v |] := vm - vs
        //  { v }  := kappa_m * vm + kappas * vs = kappam * vm (for Dirichlet coupling km=1.0, ks =
        //  0.0) < v >  := kappa_s * vm + kappam * vs = kappas * vm (for Dirichlet coupling km=1.0,
        //  ks = 0.0)

        //--------------------------------------------

        // TODO: @XFEM-Team Add possibility to use new One Step Theta with Robin Boundary Condition.

        // Create projection matrices
        //--------------------------------------------
        proj_tangential_ = proj_tangential;
        proj_normal_.put_scalar(0.0);
        for (unsigned i = 0; i < nsd_; i++) proj_normal_(i, i) = 1.0;
        proj_normal_.update(-1.0, proj_tangential_, 1.0);

        half_normal_.update(0.5, normal, 0.0);
        normal_pres_timefacfac_.update(timefacfac, normal, 0.0);

        // get velocity at integration point
        // (values at n)
        // interface velocity vector in gausspoint
        velint_s_.clear();

        if (configmap.at(Inpar::XFEM::X_Adj_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_Col).first ||
            configmap.at(Inpar::XFEM::X_Adj_n_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_n_Col).first ||
            configmap.at(Inpar::XFEM::X_Adj_t_Col).first ||
            configmap.at(Inpar::XFEM::X_Pen_t_Col).first)
          this->get_interface_veln(velint_s_);

        // Calc full veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_Row).first)
        {
          velint_diff_.update(configmap.at(Inpar::XFEM::F_Adj_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Adj_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_.update(-1.0, ivelint_jump, 1.0);

          //    #ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT //Todo: commented this not to change results
          //    with this commit
          //      Core::LinAlg::Matrix<nsd_,1> tmp_pval;
          //      tmp_pval.multiply(proj_normal_,normal_pres_timefacfac_);
          //      //Project the velocity jump [|u|] in the pressure term with the projection matrix.
          //      //  Useful if smoothed normals are used (performs better for rotating cylinder
          //      case). velint_diff_pres_timefacfac_ = velint_diff_.Dot(tmp_pval);
          //    #else
          velint_diff_pres_timefacfac_ = velint_diff_.dot(normal_pres_timefacfac_);
          //    #endif
        }

        // Calc normal-veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::XS_Adj_n_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_n_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_n_Row).first)
        {
          // velint_diff_proj_normal_ = (u^m_k - u^s_k - u^{jump}_k) P^n_{kj}
          // (([|u|]-u_0)*P^n) Apply from right for consistency
          velint_diff_normal_.update(configmap.at(Inpar::XFEM::F_Adj_n_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Adj_n_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_normal_.update(-1.0, ivelint_jump, 1.0);
          velint_diff_proj_normal_.multiply_tn(proj_normal_, velint_diff_normal_);

          //    #ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT //Todo: commented this not to change results
          //    with this commit
          //      Core::LinAlg::Matrix<nsd_,1> tmp_pval;
          //      tmp_pval.multiply(proj_normal_,normal_pres_timefacfac_);
          //      //Project the velocity jump [|u|] in the pressure term with the projection matrix.
          //      //  Useful if smoothed normals are used (performs better for rotating cylinder
          //      case). velint_diff_normal_pres_timefacfac_ = velint_diff_normal_.Dot(tmp_pval);
          //    #else
          velint_diff_normal_pres_timefacfac_ = velint_diff_normal_.dot(normal_pres_timefacfac_);
          //    #endif
        }

        // Calc tangential-veldiff
        if (configmap.at(Inpar::XFEM::F_Adj_t_Row).first ||
            configmap.at(Inpar::XFEM::XF_Adj_t_Row).first ||
            configmap.at(Inpar::XFEM::F_Pen_t_Row).first ||
            configmap.at(Inpar::XFEM::X_Pen_t_Row).first)
        {
          // velint_diff_proj_tangential_ = (u^m_k - u^s_k - u^{jump}_k) P^t_{kj}
          // (([|u|]-u_0)*P^t) Apply from right for consistency
          velint_diff_tangential_.update(configmap.at(Inpar::XFEM::F_Adj_t_Col).second, velint_m,
              -configmap.at(Inpar::XFEM::X_Adj_t_Col).second, velint_s_, 0.0);
          // add the prescribed interface velocity for weak Dirichlet boundary conditions or the
          // jump height for coupled problems
          velint_diff_tangential_.update(-1.0, ivelint_jump, 1.0);
          velint_diff_proj_tangential_.multiply_tn(proj_tangential_, velint_diff_tangential_);
        }

        // funct_s * timefac * fac
        funct_s_.clear();
        if (slave_distype != Core::FE::CellType::dis_none) this->get_slave_funct(funct_s_);

        // funct_m * funct_m (dyadic product)
        funct_m_m_dyad_.multiply_nt(funct_m, funct_m);

        // funct_s * funct_s (dyadic product)
        funct_s_s_dyad_.multiply_nt(funct_s_, funct_s_);

        // funct_s * funct_m (dyadic product)
        funct_s_m_dyad_.multiply_nt(funct_s_, funct_m);

        // penalty term
        if (fldparaxfem_.interface_terms_previous_state() == Inpar::XFEM::PreviousState_full)
        {
          if (configmap.at(Inpar::XFEM::F_Pen_Row).first ||
              configmap.at(Inpar::XFEM::X_Pen_Row).first)
          {
            nit_stab_penalty(funct_m, timefacfac, configmap.at(Inpar::XFEM::F_Pen_Row),
                configmap.at(Inpar::XFEM::X_Pen_Row), configmap.at(Inpar::XFEM::F_Pen_Col),
                configmap.at(Inpar::XFEM::X_Pen_Col), true);
          }

          // add averaged term
          if (fldparaxfem_.xff_conv_stab_scaling() == Inpar::XFEM::XFF_ConvStabScaling_upwinding ||
              fldparaxfem_.xff_conv_stab_scaling() ==
                  Inpar::XFEM::XFF_ConvStabScaling_only_averaged)
          {
            nit_stab_inflow_averaged_term(funct_m, velint_m, normal, density_m, timefacfac, true);
          }
        }

        //-----------------------------------------------------------------
        // evaluate the terms, that contribute to the background fluid
        // system - standard Dirichlet case/pure xfluid-sided case
        // AND
        // system - two-sided or xfluid-sided:

        // 2 * mu_m * kappa_m * timefac * fac
        const double km_viscm_fac = 2.0 * timefacfac * visceff_m;
        half_normal_viscm_timefacfac_km_.update(km_viscm_fac, half_normal_, 0.0);

        // 0.5* (\nabla u + (\nabla u)^T) * normal
        vderxy_m_normal_.multiply(vderxy_m, half_normal_);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.multiply_tn(vderxy_m, half_normal_);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.update(1.0, vderxy_m_normal_, 1.0);
        vderxy_m_normal_transposed_viscm_timefacfac_km_.scale(km_viscm_fac);

        // evaluate the terms, that contribute to the background fluid
        // system - two-sided or xfluid-sided:
        //-----------------------------------------------------------------
        // pressure consistency term
        if (not isImplPressure)
        {
          //-----------------------------------------------------------------
          // pressure consistency term
          if (configmap.at(Inpar::XFEM::F_Con_Col).first)
          {
            nit_p_consistency_master_terms(pres_m, funct_m, normal_pres_timefacfac_,
                configmap.at(Inpar::XFEM::F_Con_Row), configmap.at(Inpar::XFEM::X_Con_Row),
                configmap.at(Inpar::XFEM::F_Con_Col), true);
          }

          if (configmap.at(Inpar::XFEM::F_Con_n_Col)
                  .first)  //(COMMENT: evaluating this separately seems to be more efficient for our
                           // cases)
          {
            nit_p_consistency_master_terms(pres_m, funct_m, normal_pres_timefacfac_,
                configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
                configmap.at(Inpar::XFEM::F_Con_n_Col), true);
          }
        }

        //-----------------------------------------------------------------
        // viscous consistency term
        if (configmap.at(Inpar::XFEM::F_Con_Col).first)
        {
#ifndef ENFORCE_URQUIZA_GNBC
          const Core::LinAlg::Matrix<nsd_, nen_>
              dummy;  // as for the evaluation of the rhs this parameter is not used!
          // Comment: Here vderxy_m_normal_transposed_viscm_timefacfac_km_ is used!
          nit_visc_consistency_master_terms(dummy, funct_m, configmap.at(Inpar::XFEM::F_Con_Row),
              configmap.at(Inpar::XFEM::X_Con_Row), configmap.at(Inpar::XFEM::F_Con_Col), true);
#else
          FOUR_C_THROW(
              "ENFORCE_URQUIZA_GNBC for NIT_visc_Consistency_MasterRHS?");  //@Magnus: What
                                                                            // should we do here?
#endif
        }
        if (configmap.at(Inpar::XFEM::F_Con_n_Col).first)
          FOUR_C_THROW("F_Con_n_Col will come soon");
        if (configmap.at(Inpar::XFEM::F_Con_t_Col).first)
          FOUR_C_THROW("F_Con_t_Col will come soon");

        if (fldparaxfem_.interface_terms_previous_state() == Inpar::XFEM::PreviousState_full)
        {
          if (not isImplPressure)
          {
            //-----------------------------------------------------------------
            // pressure adjoint consistency term
            if (configmap.at(Inpar::XFEM::F_Adj_Row).first)
            {
              //-----------------------------------------------------------------
              // +++ qnuP option added! +++
              nit_p_adjoint_consistency_master_terms(funct_m, normal_pres_timefacfac_,
                  velint_diff_pres_timefacfac_, configmap.at(Inpar::XFEM::F_Adj_Row),
                  configmap.at(Inpar::XFEM::F_Adj_Col), configmap.at(Inpar::XFEM::X_Adj_Col), true);
            }

            if (configmap.at(Inpar::XFEM::F_Adj_n_Row)
                    .first)  //(COMMENT: evaluating this separately seems to be more efficient for
                             // our cases)
            {
              //-----------------------------------------------------------------
              // +++ qnuP option added! +++
              nit_p_adjoint_consistency_master_terms(funct_m, normal_pres_timefacfac_,
                  velint_diff_normal_pres_timefacfac_, configmap.at(Inpar::XFEM::F_Adj_n_Row),
                  configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col),
                  true);
            }
          }

          // Normal Terms!
          if (configmap.at(Inpar::XFEM::F_Adj_n_Row).first)
            FOUR_C_THROW("Implement normal Adjoint Consistency term RHS for NEW OST !");
          if (configmap.at(Inpar::XFEM::FStr_Adj_n_Col).first)
            FOUR_C_THROW("(NOT SUPPORTED FOR NORMAL DIR! Check Coercivity!)");
          if (configmap.at(Inpar::XFEM::F_Adj_t_Row).first)
            FOUR_C_THROW("Implement tangential Adjoint Consistency term RHS for NEW OST !");

          //-----------------------------------------------------------------
          // viscous adjoint consistency term
          if (configmap.at(Inpar::XFEM::F_Adj_Row).first)
          {
            const Core::LinAlg::Matrix<nsd_, nen_>
                dummy;  // as for the evaluation of the rhs this parameter is not used!
            nit_visc_adjoint_consistency_master_terms(funct_m,  ///< funct * timefacfac
                dummy,         ///< spatial derivatives of coupling master shape functions
                normal,        ///< normal-vector
                km_viscm_fac,  ///< scaling factor
                configmap.at(Inpar::XFEM::F_Adj_Row), configmap.at(Inpar::XFEM::F_Adj_Col),
                configmap.at(Inpar::XFEM::X_Adj_Col));
          }
        }

        //-----------------------------------------------------------------
        // the following quantities are only required for two-sided coupling
        // kappa_s > 0.0
        if ((configmap.at(Inpar::XFEM::XF_Con_Col).first ||
                configmap.at(Inpar::XFEM::XF_Con_n_Col).first ||
                configmap.at(Inpar::XFEM::XF_Con_t_Col).first ||
                configmap.at(Inpar::XFEM::XF_Adj_Row).first ||
                configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
                configmap.at(Inpar::XFEM::XF_Adj_t_Row).first))
        {
          //-----------------------------------------------------------------
          // pressure consistency term
          if (configmap.at(Inpar::XFEM::XF_Con_Col).first ||
              configmap.at(Inpar::XFEM::XF_Con_n_Col).first)
          {
            if (not isImplPressure)
            {
              double presn_s = 0.0;
              // must use this-pointer because of two-stage lookup!
              this->get_interface_presn(presn_s);

              if (configmap.at(Inpar::XFEM::XF_Con_Col).first)
              {
                nit_p_consistency_slave_terms(presn_s, funct_m, normal_pres_timefacfac_,
                    configmap.at(Inpar::XFEM::F_Con_Row), configmap.at(Inpar::XFEM::X_Con_Row),
                    configmap.at(Inpar::XFEM::XF_Con_Col), true);
              }

              if (configmap.at(Inpar::XFEM::XF_Con_n_Col)
                      .first)  //(COMMENT: evaluating this separately seems to be more efficient for
                               // our cases)
              {
                nit_p_consistency_slave_terms(presn_s, funct_m, normal_pres_timefacfac_,
                    configmap.at(Inpar::XFEM::F_Con_n_Row), configmap.at(Inpar::XFEM::X_Con_n_Row),
                    configmap.at(Inpar::XFEM::XF_Con_n_Col), true);
              }
            }
          }

          //-----------------------------------------------------------------
          // viscous consistency term

          // Spatial velocity gradient for slave side
          Core::LinAlg::Matrix<nsd_, nsd_> vderxyn_s;
          this->get_interface_vel_gradn(vderxyn_s);

          // 2 * mu_s * kappa_s * timefac * fac
          double ks_viscs_fac = 2.0 * visceff_s * timefacfac;

          vderxy_s_normal_.multiply(vderxyn_s, half_normal_);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.multiply_tn(vderxyn_s, half_normal_);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.update(1.0, vderxy_s_normal_, 1.0);
          vderxy_s_normal_transposed_viscs_timefacfac_ks_.scale(ks_viscs_fac);

          if (configmap.at(Inpar::XFEM::XF_Con_Col).first)
          {
            const Core::LinAlg::Matrix<nsd_, slave_nen_>
                dummy;  // as for the evaluation of the rhs this parameter is not used!
            nit_visc_consistency_slave_terms(dummy, funct_m, configmap.at(Inpar::XFEM::F_Con_Row),
                configmap.at(Inpar::XFEM::X_Con_Row), configmap.at(Inpar::XFEM::XF_Con_Col), true);
          }
          if (configmap.at(Inpar::XFEM::XF_Con_n_Col).first ||
              configmap.at(Inpar::XFEM::XF_Con_t_Col).first)
            FOUR_C_THROW("Want to implement projected slave consistency?");

          // consistency terms evaluated
          if (fldparaxfem_.interface_terms_previous_state() == Inpar::XFEM::PreviousState_full)
          {
            if (not isImplPressure)
            {
              //-----------------------------------------------------------------
              // pressure adjoint consistency term
              // HAS PROJECTION FOR VELOCITY IMPLEMENTED!!!
              if (configmap.at(Inpar::XFEM::XF_Adj_Row).first)
              {
                nit_p_adjoint_consistency_slave_terms(normal_pres_timefacfac_,
                    velint_diff_pres_timefacfac_, configmap.at(Inpar::XFEM::XF_Adj_Row),
                    configmap.at(Inpar::XFEM::F_Adj_Col), configmap.at(Inpar::XFEM::X_Adj_Col),
                    true);
              }
              if (configmap.at(Inpar::XFEM::XF_Adj_n_Row)
                      .first)  //(COMMENT: evaluating this separately seems to be more efficient for
                               // our cases)
              {
                nit_p_adjoint_consistency_slave_terms(normal_pres_timefacfac_,
                    velint_diff_normal_pres_timefacfac_, configmap.at(Inpar::XFEM::XF_Adj_n_Row),
                    configmap.at(Inpar::XFEM::F_Adj_n_Col), configmap.at(Inpar::XFEM::X_Adj_n_Col),
                    true);
              }
            }

            //-----------------------------------------------------------------
            // viscous adjoint consistency term
            // Shape function derivatives for slave side
            Core::LinAlg::Matrix<nsd_, slave_nen_> derxy_s;
            this->get_slave_funct_deriv(derxy_s);

            Core::LinAlg::Matrix<nsd_, slave_nen_> derxy_s_viscs_timefacfac_ks(derxy_s);
            derxy_s_viscs_timefacfac_ks.scale(adj_visc_scale_ * ks_viscs_fac);

            // TODO: Needs added Projection. (If deemed necessary!)
            if (configmap.at(Inpar::XFEM::XF_Adj_Row).first)
            {
              nit_visc_adjoint_consistency_slave_terms(funct_m, derxy_s_viscs_timefacfac_ks, normal,
                  configmap.at(Inpar::XFEM::XF_Adj_Row), configmap.at(Inpar::XFEM::F_Adj_Col),
                  configmap.at(Inpar::XFEM::X_Adj_Col), true);
            }
            if (configmap.at(Inpar::XFEM::XF_Adj_n_Row).first ||
                configmap.at(Inpar::XFEM::XF_Adj_t_Row).first)
              FOUR_C_THROW("Want to  implement projected slave adjoint consistency?");
          }
        }

        //-----------------------------------------------------------------
        // standard consistency traction jump term
        // Only needed for XTPF
        if (configmap.at(Inpar::XFEM::F_TJ_Rhs).first || configmap.at(Inpar::XFEM::X_TJ_Rhs).first)
        {
          // funct_s * timefac * fac * kappa_m
          funct_s_timefacfac_km_.update(
              configmap.at(Inpar::XFEM::F_TJ_Rhs).second * timefacfac, funct_s_, 0.0);

          // funct_m * timefac * fac * kappa_s
          funct_m_timefacfac_ks_.update(
              configmap.at(Inpar::XFEM::X_TJ_Rhs).second * timefacfac, funct_m, 0.0);

          nit_traction_consistency_term(
              funct_m_timefacfac_ks_, funct_s_timefacfac_km_, itraction_jump);
        }

        //-----------------------------------------------------------------
        // projection matrix approach (Laplace-Beltrami)
        if (configmap.at(Inpar::XFEM::F_LB_Rhs).first || configmap.at(Inpar::XFEM::X_LB_Rhs).first)
        {
          FOUR_C_THROW(
              "Check if we need the (Laplace-Beltrami) for the old timestep, "
              "then you should not forget to add the LB_proj_matrix as member to this function?");
          //    Core::LinAlg::Matrix<nsd_,slave_nen_> derxy_s_timefacfac_km(derxy_s);
          //    derxy_s_timefacfac_km.scale(configmap.at(Inpar::XFEM::F_LB_Rhs).second*timefacfac);
          //
          //    Core::LinAlg::Matrix<nsd_,nen_> derxy_m_timefacfac_ks(derxy_m);
          //    derxy_m_timefacfac_ks.scale(configmap.at(Inpar::XFEM::X_LB_Rhs).second*timefacfac);
          //
          //    nit_projected_traction_consistency_term(
          //    derxy_m_timefacfac_ks,
          //    derxy_s_timefacfac_km,
          //    LB_proj_matrix);
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_traction_consistency_term(
          const Core::LinAlg::Matrix<nen_, 1>&
              funct_m_timefacfac_ks,  ///< funct * timefacfac *kappa_s
          const Core::LinAlg::Matrix<slave_nen_, 1>&
              funct_s_timefacfac_km,  ///< funct_s * timefacfac *kappa_m
          const Core::LinAlg::Matrix<nsd_, 1>&
              itraction_jump  ///< prescribed interface traction, jump height for coupled problems
      )
      {
        /*            \
     - |  < v >,   t   |   with t = [sigma * n]
        \             /     */

        // All else:            [| sigma*n |] = 0


        // loop over velocity components
        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          //-----------------------------------------------
          //    - (vm, ks * t)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const double funct_m_ks_timefacfac_traction =
                funct_m_timefacfac_ks(ir) * itraction_jump(ivel);

            const unsigned row = m_index(ir, ivel);
            rh_c_um_(row, 0) += funct_m_ks_timefacfac_traction;
          }

          //-----------------------------------------------
          //    + (vs, km * t)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double funct_s_km_timefacfac_traction =
                funct_s_timefacfac_km(ir) * itraction_jump(ivel);

            const unsigned row = s_index(ir, ivel);
            rh_c_us_(row, 0) += funct_s_km_timefacfac_traction;
          }
        }  // end loop over velocity components
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_projected_traction_consistency_term(
              const Core::LinAlg::Matrix<nsd_, nen_>&
                  derxy_m_timefacfac_ks,  ///< master shape function derivatives * timefacfac *
                                          ///< kappa_s
              const Core::LinAlg::Matrix<nsd_, slave_nen_>&
                  derxy_s_timefacfac_km,  ///< slave shape function derivatives * timefacfac *
                                          ///< kappa_m
              const Core::LinAlg::Matrix<nsd_, nsd_>&
                  itraction_jump_matrix  ///< prescribed projection matrix
          )
      {
        /*                   \
     - |  < \nabla v > : P    |   with P = (I - n (x) n )
        \                    /     */

        // Two-Phase Flow:
        //
        //     t_{n+1}          ( < \nabla v > : P )
        //                                          P can be calculated in different ways.
        //                                          P_smooth*P_cut is best approach so far.
        //
        //      t_{n}           [| sigma*n |] = [|  -pI + \mu*[\nabla u + (\nabla u)^T]  |] * n


        // Two-Phase Flow, Laplace Beltrami approach:
        // loop over velocity components
        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          //-----------------------------------------------
          //    - (\nabla vm, ks * P)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            double derxy_m_ks_timefacfac_sum = 0.0;
            // Sum over space dimensions
            for (unsigned idum = 0; idum < nsd_; ++idum)
            {
              derxy_m_ks_timefacfac_sum +=
                  derxy_m_timefacfac_ks(idum, ir) * itraction_jump_matrix(idum, ivel);
            }

            const unsigned row = m_index(ir, ivel);
            rh_c_um_(row, 0) -= derxy_m_ks_timefacfac_sum;
          }

          //-----------------------------------------------
          //    + (\nabla vs, km * P)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            double derxy_s_km_timefacfac_sum = 0.0;
            // Sum over space dimensions
            for (unsigned idum = 0; idum < nsd_; ++idum)
            {
              derxy_s_km_timefacfac_sum +=
                  derxy_s_timefacfac_km(idum, ir) * itraction_jump_matrix(idum, ivel);
            }

            const unsigned row = s_index(ir, ivel);
            rh_c_us_(row, 0) -= derxy_s_km_timefacfac_sum;
          }
        }  // end loop over velocity components
      }



      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_p_consistency_master_terms(
          const double& pres_m,                                    ///< master pressure
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,            ///< funct
          const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
          const std::pair<bool, double>& m_row,                    ///< scaling for master row
          const std::pair<bool, double>& s_row,                    ///< scaling for slave row
          const std::pair<bool, double>& m_col,                    ///< scaling for master col
          bool only_rhs)
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::nit_p_consistency_master_terms");


        /*                   \       /          i      \
      + |  [ v ],   {Dp}*n    | = - | [ v ], { p }* n   |
        \                    /       \                */

        //-----------------------------------------------
        //    + (vm, km *(Dpm)*n)
        //-----------------------------------------------
        const double facmm = m_row.second * m_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double funct_m_pres = funct_m(ir) * pres_m * facmm;

          // loop over velocity components
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // -(v,p*n)
            rh_c_um_(m_index(ir, ivel), 0) -= funct_m_pres * normal_timefacfac(ivel);
          }
        }

        double facsm = 0.0;
        if (s_row.first)
        {
          facsm = s_row.second * m_col.second;
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double funct_s_pres = funct_s_(ir) * pres_m * facsm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // -(v,p*n)
              rh_c_us_(s_index(ir, ivel), 0) += funct_s_pres * normal_timefacfac(ivel);
            }
          }  // end loop over velocity components
        }

        if (only_rhs) return;

        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          const unsigned col = m_pres(ic);

          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const double tmp = funct_m_m_dyad_(ir, ic) * facmm;

            // loop over velocity components
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // (v,Dp*n)
              c_umum_(m_index(ir, ivel), col) += tmp * normal_timefacfac(ivel);
            }
          }
        }

        if (s_row.first)
        {
          for (unsigned ic = 0; ic < nen_; ++ic)
          {
            const unsigned col = m_pres(ic);

            for (unsigned ir = 0; ir < slave_nen_; ++ir)
            {
              const double tmp = funct_s_m_dyad_(ir, ic) * facsm;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                //-----------------------------------------------
                //    - (vs, km *(Dpm)*n)
                //-----------------------------------------------

                // (v,Dp*n)
                c_usum_(s_index(ir, ivel), col) -= tmp * normal_timefacfac(ivel);
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_p_consistency_slave_terms(
          const double& pres_s,                                    ///< slave pressure
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,            ///< funct
          const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
          const std::pair<bool, double>& m_row,                    ///< scaling for master row
          const std::pair<bool, double>& s_row,                    ///< scaling for slave row
          const std::pair<bool, double>& s_col,                    ///< scaling for slave col
          bool only_rhs)
      {
        const double facms = m_row.second * s_col.second;
        const double facss = s_row.second * s_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp = funct_m(ir) * pres_s * facms;
          // loop over velocity components
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // -(vm, ks * ps*n)
            rh_c_um_(m_index(ir, ivel), 0) -= tmp * normal_timefacfac(ivel);
          }
        }

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double tmp = funct_s_(ir) * pres_s * facss;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // +(vs,ks * ps*n)
            rh_c_us_(s_index(ir, ivel), 0) += tmp * normal_timefacfac(ivel);
          }
        }  // end loop over velocity components

        if (only_rhs) return;

        for (unsigned ic = 0; ic < slave_nen_; ++ic)
        {
          //-----------------------------------------------
          //    + (vm, ks *(Dps)*n)
          //-----------------------------------------------
          unsigned col = s_pres(ic);

          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const double tmp = funct_s_m_dyad_(ic, ir) * facms;
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // (vm, ks * Dps*n)
              c_umus_(m_index(ir, ivel), col) += tmp * normal_timefacfac(ivel);
            }
          }

          //-----------------------------------------------
          //    - (vs, ks *(Dps)*n)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double tmp = funct_s_s_dyad_(ir, ic) * facss;
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              c_usus_(s_index(ir, ivel), col) -= tmp * normal_timefacfac(ivel);
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      NitscheCoupling<distype, slave_distype, slave_numdof>::nit_p_adjoint_consistency_master_terms(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,            ///< funct
          const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
          const double&
              velint_diff_normal_timefacfac,     ///< (velint_m - velint_s) * normal * timefacfac
          const std::pair<bool, double>& m_row,  ///< scaling for master row
          const std::pair<bool, double>& m_col,  ///< scaling for master col
          const std::pair<bool, double>& s_col,  ///< scaling for slave col
          bool only_rhs)
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::nit_p_adjoint_consistency_master_terms");

        // 1) No-split no qunP option:
        /*                   \     /              i   \
      - |  { q }*n ,[ Du ]     | = |  { q }*n  ,[ u ]   |
        \                    /     \                 */

        // 2) qunP option:
        /*                       \     /              i      \
      - |  { q }*n ,[ Du ] P^n    | = |  { q }*n  ,[ u ] P^n  |
        \                        /     \                    */

        // REMARK:
        // the sign of the pressure adjoint consistency term is opposite to the sign of the pressure
        // consistency term (interface), as a non-symmetric pressure formulation is chosen in the
        // standard fluid the sign of the standard volumetric pressure consistency term is opposite
        // to the (chosen) sign of the pressure-weighted continuity residual; think about the
        // Schur-complement for the Stokes problem: S_pp = A_pp - A_pu A_uu^-1 A_up
        // (--> A_pu == -A_up^T; sgn(A_pp) == sgn(- A_pu A_uu^-1 Aup), where A_pp comes from
        // pressure-stabilizing terms) a symmetric adjoint pressure consistency term would also
        // affect the sign of the pressure stabilizing terms for Stokes' problem, this sign choice
        // leads to a symmetric, positive definite Schur-complement matrix S (v, p*n)--> A_up;
        // -(q,u*n)--> -A_up^T; S_pp = A_pp + A_up^T A_uu A_up


        const double velint_diff_normal_timefacfac_km =
            velint_diff_normal_timefacfac * m_row.second;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          // (qm*n, km * um)
          // -(qm*n,km * u_DBC) for weak DBC or
          // -(qm*n,km * us)
          rh_c_um_(m_pres(ir), 0) += funct_m(ir) * velint_diff_normal_timefacfac_km;
          //    rhC_um_(m_pres(ir),0) -= funct_m(ir)*velint_diff_normal_timefacfac; //TESTING!
        }

        if (only_rhs) return;

#ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT
        Core::LinAlg::Matrix<nsd_, 1> proj_norm_timefacfac;
        proj_norm_timefacfac.multiply(proj_normal_, normal_timefacfac);
#endif
        //-----------------------------------------------
        //    - (qm*n, km *(Dumb))
        //-----------------------------------------------
        const double facmm = m_row.second * m_col.second;
        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const unsigned row = m_pres(ir);

            const double tmp = funct_m_m_dyad_(ir, ic) * facmm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // - (qm*n, km *(Dumb))
#ifndef PROJECT_VEL_FOR_PRESSURE_ADJOINT
              C_umum_(row, m_index(ic, ivel)) -= tmp * normal_timefacfac(ivel);
              //        C_umum_(row, m_index(ic,ivel)) += tmp*normal_timefacfac_km(ivel); //TESTING!
#else
              c_umum_(row, m_index(ic, ivel)) -= tmp * proj_norm_timefacfac(ivel);
#endif
            }
          }
        }

        if (s_col.first)
        {
          const double facms = m_row.second * s_col.second;
          //-----------------------------------------------
          //    + (qm*n, km *(Dus))
          //-----------------------------------------------
          for (unsigned ic = 0; ic < slave_nen_; ++ic)
          {
            for (unsigned ir = 0; ir < nen_; ++ir)
            {
              const unsigned row = m_pres(ir);
              const double tmp = funct_s_m_dyad_(ic, ir) * facms;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                // -(qm*n, km * Dus)
#ifndef PROJECT_VEL_FOR_PRESSURE_ADJOINT
                C_umus_(row, s_index(ic, ivel)) += tmp * normal_timefacfac(ivel);
#else
                c_umus_(row, s_index(ic, ivel)) += tmp * proj_norm_timefacfac(ivel);
#endif
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      NitscheCoupling<distype, slave_distype, slave_numdof>::nit_p_adjoint_consistency_slave_terms(
          const Core::LinAlg::Matrix<nsd_, 1>& normal_timefacfac,  ///< normal vector * timefacfac
          const double& velint_diff_normal_timefacfac,  ///< (velint_m - velint_s) * n * timefacfac
          const std::pair<bool, double>& s_row,         ///< scaling for slave row
          const std::pair<bool, double>& m_col,         ///< scaling for master col
          const std::pair<bool, double>& s_col,         ///< scaling for slave col
          bool only_rhs)
      {
        // 1) No-split no qunP option:
        /*                   \     /              i   \
      - |  { q }*n ,[ Du ]     | = |  { q }*n  ,[ u ]   |
        \                    /     \                 */

        // 2) qunP option:
        /*                       \     /              i      \
      - |  { q }*n ,[ Du ] P^n    | = |  { q }*n  ,[ u ] P^n  |
        \                        /     \                    */

#ifdef PROJECT_VEL_FOR_PRESSURE_ADJOINT
        Core::LinAlg::Matrix<nsd_, 1> proj_norm_timefacfac;
        proj_norm_timefacfac.multiply(proj_normal_, normal_timefacfac);
#endif

        const double velint_diff_normal_timefacfac_ks =
            velint_diff_normal_timefacfac * s_row.second;
        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          // (qs*n,ks* um)
          rh_c_us_(s_pres(ir), 0) += funct_s_(ir) * velint_diff_normal_timefacfac_ks;
        }

        if (only_rhs) return;

        //-----------------------------------------------
        //    - (qs*n, ks *(Dumb))
        //-----------------------------------------------
        const double facsm = s_row.second * m_col.second;
        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const unsigned row = s_pres(ir);

            const double tmp = funct_s_m_dyad_(ir, ic) * facsm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // -(qs*n, ks* Dumb)
#ifndef PROJECT_VEL_FOR_PRESSURE_ADJOINT
              C_usum_(row, m_index(ic, ivel)) -= tmp * normal_timefacfac(ivel);
              //        C_usum_(row,m_index(ic,ivel)) += tmp*normal_timefacfac_ks(ivel); //TESTING!
#else
              c_usum_(row, m_index(ic, ivel)) -= tmp * proj_norm_timefacfac(ivel);
#endif
            }
          }
        }

        //-----------------------------------------------
        //    + (qs*n, ks *(Dus))
        //-----------------------------------------------
        const double facss = s_row.second * s_col.second;
        for (unsigned ic = 0; ic < slave_nen_; ++ic)
        {
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const unsigned row = s_pres(ir);

            const double tmp = funct_s_s_dyad_(ir, ic) * facss;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // +(qs*n, ks* Dus)
#ifndef PROJECT_VEL_FOR_PRESSURE_ADJOINT
              C_usus_(row, s_index(ic, ivel)) += tmp * normal_timefacfac(ivel);
#else
              c_usus_(row, s_index(ic, ivel)) += tmp * proj_norm_timefacfac(ivel);
#endif
            }
          }
        }
        return;
      }

      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_visc_consistency_master_terms(
          const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,  ///< master deriv
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,     ///< funct_m
          const std::pair<bool, double>& m_row,             ///< scaling for master row
          const std::pair<bool, double>& s_row,             ///< scaling for slave row
          const std::pair<bool, double>& m_col,             ///< scaling for master col
          bool only_rhs)
      {
        // viscous consistency term

        /*                           \       /                   i      \
      - |  [ v ],  { 2mu eps(u) }*n    | = + | [ v ],  { 2mu eps(u ) }*n  |
        \                            /       \                         */


        // here we use a non-optimal order to assemble the values into C_umum
        // however for this term we have to save operations
        const double facmm = m_row.second * m_col.second;
        const double facsm = s_row.second * m_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir) * facmm;

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            //-----------------------------------------------
            //    - (vm, (2*km*mum) *eps(Dumb)*n)
            //-----------------------------------------------
            rh_c_um_(m_index(ir, ivel), 0) +=
                tmp_val * vderxy_m_normal_transposed_viscm_timefacfac_km_(ivel);
          }
        }


        if (s_row.first)
        {
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double tmp_val = funct_s_(ir) * facsm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              //-----------------------------------------------
              //    + (vs, (2*km*mum) *eps(Dumb)*n)
              //-----------------------------------------------

              // rhs
              rh_c_us_(s_index(ir, ivel), 0) -=
                  tmp_val * vderxy_m_normal_transposed_viscm_timefacfac_km_(ivel);
            }
          }
        }

        if (only_rhs) return;

        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          const double normal_deriv_tmp = half_normal_deriv_m_viscm_timefacfac_km_(ic);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            const double tmp_derxy_m = derxy_m(ivel, ic);
            for (unsigned jvel = 0; jvel < nsd_; ++jvel)
            {
              const unsigned col = m_index(ic, jvel);

              double tmp = half_normal_viscm_timefacfac_km_(jvel) * tmp_derxy_m;
              if (ivel == jvel) tmp += normal_deriv_tmp;

              const double tmpm = tmp * facmm;
              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                c_umum_(m_index(ir, ivel), col) -= funct_m(ir) * tmpm;
              }

              if (s_row.first)
              {
                const double tmps = tmp * facsm;
                for (unsigned ir = 0; ir < slave_nen_; ++ir)
                {
                  c_usum_(s_index(ir, ivel), col) += funct_s_(ir) * tmps;
                }
              }
            }
          }
        }

        return;
      }


      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_visc_consistency_master_terms_projected(
              const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,      ///< master deriv
              const Core::LinAlg::Matrix<nen_, 1>& funct_m,         ///< funct_m
              const Core::LinAlg::Matrix<nsd_, nsd_>& proj_matrix,  ///< projection matrix
              const double& km_viscm_fac,                           ///< scaling factor
              const std::pair<bool, double>& m_row,                 ///< scaling for master row
              const std::pair<bool, double>& s_row,                 ///< scaling for slave row
              const std::pair<bool, double>& m_col                  ///< scaling for master col
          )
      {
        // 1) No-split WDBC option:
        //----------------------------------------------------------------------------------------
        /*                            \       /                   i      \
      - |  [ v ],  { 2mu eps(u) }*n    | = + | [ v ],  { 2mu eps(u ) }*n  |
        \                             /       \                         */
        //----------------------------------------------------------------------------------------

        // 2) (Normal - Tangential split):
        //----------------------------------------------------------------------------------------
        /*                                         \
      - |   { 2mu*eps(v) }*n  ,  [Du] P_n           |  =
        \                                         */

        /*                               i                   \
      + |  alpha* { 2mu*eps(v) }*n  , [ u ]  P_n              |
        \                                                   */

        //----------------------------------------------------------------------------------------

        // 2.0 * timefacfac * visceff_m * k_m * [\nabla N^(IX)]_k P^t_{kj}
        proj_matrix_derxy_m_.multiply_tn(proj_matrix, derxy_m);  // Apply from right for consistency
        proj_matrix_derxy_m_.scale(km_viscm_fac);

        // P_norm * {2.0 * timefacfac * visceff_m * 0.5 * (\nabla u + (\nabla u)^T)}
        vderxy_x_normal_transposed_viscx_timefacfac_kx_pmatrix_.multiply_tn(
            proj_matrix, vderxy_m_normal_transposed_viscm_timefacfac_km_);

        // here we use a non-optimal order to assemble the values into C_umum
        // however for this term we have to save operations
        const double facmm = m_row.second * m_col.second;
        const double facsm = s_row.second * m_col.second;
        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          // half_normal_deriv_m_viscm_timefacfac_km_ = 2.0 * timefacfac *
          // visceff_m*(0.5*normal(k)*derxy_m(k,ic))
          const double normal_deriv_tmp = half_normal_deriv_m_viscm_timefacfac_km_(ic);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            for (unsigned jvel = 0; jvel < nsd_; ++jvel)
            {
              const unsigned col = m_index(ic, jvel);

              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                // C_umum_(m_index(ir,ivel), col) -= funct_m(ir) * tmp;
                c_umum_(m_index(ir, ivel), col) -=
                    funct_m(ir) * facmm *
                    (proj_matrix(jvel, ivel) * normal_deriv_tmp +
                        proj_matrix_derxy_m_(ivel, ic) * half_normal_(jvel));
              }


              if (!s_row.first) continue;


              for (unsigned ir = 0; ir < slave_nen_; ++ir)
              {
                c_usum_(s_index(ir, ivel), col) +=
                    funct_s_(ir) * facsm *
                    (proj_matrix(jvel, ivel) * normal_deriv_tmp +
                        proj_matrix_derxy_m_(ivel, ic) * half_normal_(jvel));
              }
            }
          }
        }


        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = facmm * funct_m(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            //-----------------------------------------------
            //    - (vm, (2*km*mum) *eps(Dumb)*n)
            //-----------------------------------------------
            rh_c_um_(m_index(ir, ivel), 0) +=
                tmp_val * vderxy_x_normal_transposed_viscx_timefacfac_kx_pmatrix_(ivel);
          }
        }


        if (!s_row.first) return;


        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double tmp_val = facsm * funct_s_(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            //-----------------------------------------------
            //    + (vs, (2*km*mum) *eps(Dumb)*n)
            //-----------------------------------------------

            // rhs
            rh_c_us_(s_index(ir, ivel), 0) -=
                tmp_val * vderxy_x_normal_transposed_viscx_timefacfac_kx_pmatrix_(ivel);
          }
        }
        return;
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_visc_consistency_slave_terms(
          const Core::LinAlg::Matrix<nsd_, slave_nen_>&
              derxy_s,                                   ///< slave shape function derivatives
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct_m
          const std::pair<bool, double>& m_row,          ///< scaling for master row
          const std::pair<bool, double>& s_row,          ///< scaling for slave row
          const std::pair<bool, double>& s_col,          ///< scaling for slave col
          bool only_rhs)
      {
        // diagonal block (i,i): +/-2*ks*mus * ...
        /*       nsd_
         *       *---*
         *       \    dN                     dN
         *  N  *  *   --  * 0.5 * n_j + N  * --  * n_i * 0.5
         *       /    dxj                    dxi
         *       *---*
         *       j = 1
         */

        // off-diagonal block (i,j) : +/-2*ks*mus * ...
        /*       dN
         *  N  * -- * n_j * 0.5
         *       dxi
         */

        const double facms = m_row.second * s_col.second;
        const double facss = s_row.second * s_col.second;

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir) * facms;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_um_(m_index(ir, ivel), 0) +=
                tmp_val * vderxy_s_normal_transposed_viscs_timefacfac_ks_(ivel);
          }
        }

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double tmp_val = funct_s_(ir) * facss;
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // rhs
            rh_c_us_(s_index(ir, ivel), 0) -=
                tmp_val * vderxy_s_normal_transposed_viscs_timefacfac_ks_(ivel);
          }
        }

        if (only_rhs) return;

        for (unsigned ic = 0; ic < slave_nen_; ++ic)
        {
          const double normal_deriv_tmp = half_normal_deriv_s_viscs_timefacfac_ks_(ic);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            const double tmp_derxy_s = derxy_s(ivel, ic);

            for (unsigned jvel = 0; jvel < nsd_; ++jvel)
            {
              const unsigned col = s_index(ic, jvel);

              double tmp = half_normal_viscs_timefacfac_ks_(jvel) * tmp_derxy_s;

              if (ivel == jvel) tmp += normal_deriv_tmp;

              const double tmpm = tmp * facms;
              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                //-----------------------------------------------
                //    - (vm, (2*ks*mus) *eps(Dus)*n)
                //-----------------------------------------------
                c_umus_(m_index(ir, ivel), col) -= funct_m(ir) * tmpm;
              }

              const double tmps = tmp * facss;
              for (unsigned ir = 0; ir < slave_nen_; ++ir)
              {
                //-----------------------------------------------
                //    + (vs, (2*ks*mus) *eps(Dus)*n)
                //-----------------------------------------------
                // diagonal block
                c_usus_(s_index(ir, ivel), col) += funct_s_(ir) * tmps;
              }
            }
          }
        }
        return;
      }

      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_visc_adjoint_consistency_master_terms(
              const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct * timefacfac
              const Core::LinAlg::Matrix<nsd_, nen_>&
                  derxy_m,  ///< spatial derivatives of coupling master shape functions
              const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal-vector
              const double& viscm_fac,                      ///< scaling factor
              const std::pair<bool, double>& m_row,         ///< scaling for master row
              const std::pair<bool, double>& m_col,         ///< scaling for master col
              const std::pair<bool, double>& s_col,         ///< scaling for slave col
              bool only_rhs)
      {
        /*                                \       /                             i   \
      - |  alpha* { 2mu*eps(v) }*n , [ Du ] |  =  |  alpha* { 2mu eps(v) }*n ,[ u ]   |
        \                                 /       \                                */
        // (see Burman, Fernandez 2009)
        // +1.0 symmetric
        // -1.0 antisymmetric

        //-----------------------------------------------------------------
        // viscous adjoint consistency term

        double tmp_fac = adj_visc_scale_ * viscm_fac;
        derxy_m_viscm_timefacfac_.update(tmp_fac, derxy_m);  // 2 * mu_m * timefacfac *
                                                             // derxy_m(k,ic)

        static Core::LinAlg::Matrix<nsd_, nsd_> velint_diff_dyad_normal,
            velint_diff_dyad_normal_symm;
        velint_diff_dyad_normal.multiply_nt(velint_diff_, normal);

        for (unsigned jvel = 0; jvel < nsd_; ++jvel)
        {
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            velint_diff_dyad_normal_symm(ivel, jvel) =
                velint_diff_dyad_normal(ivel, jvel) + velint_diff_dyad_normal(jvel, ivel);
          }
        }

        const double facm = m_row.second * 0.5;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double derxy_m_viscm_timefacfac_km_half_tmp =
                derxy_m_viscm_timefacfac_(jvel, ir) * facm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // rhs
              rh_c_um_(m_index(ir, ivel), 0) +=
                  derxy_m_viscm_timefacfac_km_half_tmp * velint_diff_dyad_normal_symm(ivel, jvel);
            }
          }
        }

        if (only_rhs) return;

        const double facmm = m_row.second * m_col.second;
        const double facms = m_row.second * s_col.second;

        normal_deriv_m_viscm_km_.multiply_tn(
            derxy_m_viscm_timefacfac_, half_normal_);  // half_normal(k)*derxy_m(k,ic)*viscm*km

        // here we use a non-optimal order to assemble the values into C_umum
        // however for this term we have to save operations
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double normal_deriv_tmp = normal_deriv_m_viscm_km_(ir);

          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double tmp_derxy_m = derxy_m_viscm_timefacfac_(jvel, ir);
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = m_index(ir, ivel);

              double tmp = half_normal_(ivel) * tmp_derxy_m;
              if (ivel == jvel) tmp += normal_deriv_tmp;

              const double tmpm = tmp * facmm;
              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                c_umum_(row, m_index(ic, jvel)) -= funct_m(ic) * tmpm;
              }


              if (s_col.first)
              {
                const double tmps = tmp * facms;
                for (unsigned ic = 0; ic < slave_nen_; ++ic)
                {
                  c_umus_(row, s_index(ic, jvel)) += funct_s_(ic) * tmps;
                }
              }
            }
          }
        }
        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_visc_adjoint_consistency_master_terms_projected(
              const Core::LinAlg::Matrix<nsd_, nen_>&
                  derxy_m_viscm_timefacfac_km,  ///< master shape function derivatives * timefacfac
                                                ///< * 2 * mu_m * kappa_m
              const Core::LinAlg::Matrix<nen_, 1>&
                  funct_m,  ///< embedded element funct *mu*timefacfac
              const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
              const std::pair<bool, double>& m_row,         ///< scaling for master row
              const std::pair<bool, double>& m_col,         ///< scaling for master col
              const std::pair<bool, double>& s_col          ///< scaling for slave col
          )
      {
        // 1) No-split WDBC option:
        //----------------------------------------------------------------------------------------
        /*                                 \       /                             i   \
      - |  alpha* { 2mu*eps(v) }*n , [ Du ] |  =  |  alpha* { 2mu eps(v) }*n ,[ u ]   |
        \                                  /       \                                */
        //----------------------------------------------------------------------------------------

        // 2) (Normal - Tangential split):
        //----------------------------------------------------------------------------------------
        /*                                                       \
      - |  alpha* { 2mu*eps(v) }*n  ,  [Du] *  stab_fac *  P      |  =
        \                                                       */

        /*                               i                       \
      + |  alpha* { 2mu*eps(v) }*n  , [ u ]  *  stab_fac *  P     |
        \                                                       */

        //----------------------------------------------------------------------------------------

        // (see Burman, Fernandez 2009)
        // alpha =  +1.0 symmetric
        //          -1.0 antisymmetric

        // timefacfac                   = theta*dt*fac
        // derxy_m_viscm_timefacfac_km  = alpha * 2 * mu_m * timefacfac * derxy_m(k,IX)

        // normal_deriv_m_viscm_km_    = alpha * half_normal(k)* 2 * mu_m * timefacfac *
        // derxy_m(k,IX)
        //                             = alpha * mu_m * timefacfac * c(IX)

        // proj_matrix_derxy_m_        = alpha * 2 * mu_m * timefacfac * derxy_m_(k,ir) * P_{jk}
        //                             = alpha * 2 * mu_m * timefacfac * p^t_1(ir,j)

        //  // proj_tang_derxy_m = 2.0 * P^t_{jk} * derxy_m(k,IX) * mu_m * timefacfac * km
        //  //                   = 2.0 * mu_m * timefacefac * km * p_1(IX,j)

        // here we use a non-optimal order to assemble the values into C_umum
        // however for this term we have to save operations
        const double facmm = m_row.second * m_col.second;
        const double facms = m_row.second * s_col.second;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          // alpha * mu_m * timefacfac * \sum_k dN^(ir)/dx_k * n_k
          const double normal_deriv_tmp = normal_deriv_m_viscm_km_(ir);

          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = m_index(ir, ivel);

              const double tmpm = facmm * (proj_matrix_(jvel, ivel) * normal_deriv_tmp +
                                              proj_matrix_derxy_m_(jvel, ir) * half_normal_(ivel));
              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                c_umum_(row, m_index(ic, jvel)) -= funct_m(ic) * tmpm;
              }

              if (s_col.first)
              {
                const double tmps =
                    facms * (proj_matrix_(jvel, ivel) * normal_deriv_tmp +
                                proj_matrix_derxy_m_(jvel, ir) * half_normal_(ivel));
                for (unsigned ic = 0; ic < slave_nen_; ++ic)
                {
                  c_umus_(row, s_index(ic, jvel)) += funct_s_(ic) * tmps;
                }
              }
            }
          }
        }

        // Can this be made more effective?
        // static Core::LinAlg::Matrix<nsd_,nsd_> velint_proj_norm_diff_dyad_normal,
        // velint_proj_norm_diff_dyad_normal_symm;
        // velint_diff_proj_normal_ = (u^m_k - u^s_k) P^n_{kj} * n
        velint_proj_norm_diff_dyad_normal_.multiply_nt(velint_diff_proj_matrix_, normal);

        for (unsigned jvel = 0; jvel < nsd_; ++jvel)
        {
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            velint_proj_norm_diff_dyad_normal_symm_(ivel, jvel) =
                velint_proj_norm_diff_dyad_normal_(ivel, jvel) +
                velint_proj_norm_diff_dyad_normal_(jvel, ivel);
          }
        }

        const double facm = m_row.second * 0.5;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double derxy_m_viscm_timefacfac_km_half_tmp =
                derxy_m_viscm_timefacfac_km(jvel, ir) * facm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // rhs
              rh_c_um_(m_index(ir, ivel), 0) += derxy_m_viscm_timefacfac_km_half_tmp *
                                                velint_proj_norm_diff_dyad_normal_symm_(ivel, jvel);
            }
          }
        }

        return;
      }



      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_visc_adjoint_consistency_slave_terms(
              const Core::LinAlg::Matrix<nen_, 1>&
                  funct_m,  ///< embedded element funct *mu*timefacfac
              const Core::LinAlg::Matrix<nsd_, slave_nen_>&
                  derxy_s_viscs_timefacfac_ks,  ///< master shape function derivatives * timefacfac
                                                ///< * 2
                                                ///< * mu_m * kappa_m
              const Core::LinAlg::Matrix<nsd_, 1>& normal,
              const std::pair<bool, double>& s_row,  ///< scaling for slave row
              const std::pair<bool, double>& m_col,  ///< scaling for master col
              const std::pair<bool, double>& s_col,  ///< scaling for slave col
              bool only_rhs)
      {
        /*                                \       /                             i   \
      - |  alpha* { 2mu*eps(v) }*n , [ Du ] |  =  |  alpha* { 2mu eps(v) }*n ,[ u ]   |
        \                                 /       \                                */
        // (see Burman, Fernandez 2009)
        // +1.0 symmetric
        // -1.0 antisymmetric

        // diagonal block (i,i): +/-2*km*mum * alpha * ...
        /*
         *       nsd_
         *       *---*
         *       \    dN                    dN
         *        *   --  * 0.5 * n_j *N +  --  * n_i * 0.5 *N
         *       /    dxj                   dxi
         *       *---*
         *       j = 1
         */

        // off-diagonal block (i,j) : +/-2*km*mum * alpha * ...
        /*   dN
         *   -- * n_i * 0.5 * N
         *   dxj
         */

        static Core::LinAlg::Matrix<nsd_, nsd_> velint_diff_dyad_normal,
            velint_diff_dyad_normal_symm;
        velint_diff_dyad_normal.multiply_nt(velint_diff_, normal);

        for (unsigned jvel = 0; jvel < nsd_; ++jvel)
        {
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            velint_diff_dyad_normal_symm(ivel, jvel) =
                velint_diff_dyad_normal(ivel, jvel) + velint_diff_dyad_normal(jvel, ivel);
          }
        }

        const double facs = s_row.second * 0.5;
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double derxy_s_viscs_timefacfac_ks_half_tmp =
                derxy_s_viscs_timefacfac_ks(jvel, ir) * facs;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // rhs
              rh_c_us_(s_index(ir, ivel), 0) +=
                  derxy_s_viscs_timefacfac_ks_half_tmp * velint_diff_dyad_normal_symm(ivel, jvel);
            }
          }
        }

        if (only_rhs) return;

        const double facsm = s_row.second * m_col.second;
        const double facss = s_row.second * s_col.second;
        normal_deriv_s_viscs_ks_.multiply_tn(
            derxy_s_viscs_timefacfac_ks, half_normal_);  // half_normal(k)*derxy_m(k,ic)*viscm*km

        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          const double normal_deriv_tmp = normal_deriv_s_viscs_ks_(ir);

          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double tmp_derxy_s = derxy_s_viscs_timefacfac_ks(jvel, ir);
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = s_index(ir, ivel);

              double tmp = half_normal_(ivel) * tmp_derxy_s;
              if (ivel == jvel) tmp += normal_deriv_tmp;

              const double tmpm = tmp * facsm;
              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                c_usum_(row, m_index(ic, jvel)) -= funct_m(ic) * tmpm;
              }

              if (s_col.first)
              {
                const double tmps = tmp * facss;
                for (unsigned ic = 0; ic < slave_nen_; ++ic)
                {
                  c_usus_(row, s_index(ic, jvel)) += funct_s_(ic) * tmps;
                }
              }
            }
          }
        }
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_visc_neumann_adjoint_consistency_master_terms_projected(
              const Core::LinAlg::Matrix<nsd_, nen_>&
                  derxy_m_viscm_timefacfac_km,  ///< master shape function derivatives * timefacfac
                                                ///< * 2 * mu_m * kappa_m
              const Core::LinAlg::Matrix<nsd_, nen_>& derxy_m,  ///< master deriv
              const Core::LinAlg::Matrix<nsd_, nsd_>&
                  vderxy_m,  ///< coupling master spatial velocity derivatives
              const Core::LinAlg::Matrix<nen_, 1>&
                  funct_m,  ///< embedded element funct *mu*timefacfac
              const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
              const std::pair<bool, double>& m_row,         ///< scaling for master row
              const std::pair<bool, double>& mstr_col       ///< scaling for master col
          )
      {
        // 1) No-split WDBC option:
        /*           \
        |  v  ,  0   |
        \           */

        // 2) (Normal - Tangential split):
        /*                                                                                                      \
      - |  alpha* (\epsilon * \gamma * h_E)/(epsilon + \gamma *h_E)  { 2*eps(v) }*n ,{ 2mu eps(Du)
      }*n P_t       | =
        \ */

        /*                                                                               i \
      + |  alpha* (\epsilon * \gamma * h_E)/(epsilon + \gamma *h_E)  { 2*eps(v) }*n ,{ 2mu eps(u)
      }*n P_t       |
        \ */

        /*                                                                                \
      - |  alpha* { 2mu*eps(v) }*n  ,   g  ( epsilon*gamma*h_E/(gamma*h_E+epsilon) * P_t)  |
        \                                                                                */

        // (see Burman, Fernandez 2009)
        // +1.0 symmetric
        // -1.0 antisymmetric

        const double facmm = m_row.second * mstr_col.second;

        //  //normal_deriv_m_viscm_km_ = 2.0 * alpha * half_normal(k) * mu_m * timefacfac * km *
        //  derxy_m(k,ic)
        //  //                         = mu_m * alpha * timefacfac *km * c(ix)
        //  normal_deriv_m_viscm_km_.multiply_tn(derxy_m_viscm_timefacfac_km, half_normal_);

        //  Core::LinAlg::Matrix<nen_,1> normal_deriv_m;
        normal_deriv_m_.multiply_tn(derxy_m, half_normal_);
        normal_deriv_m_.scale(2.0);  // 2.0 * half_normal(k) * derxy_m(k,ix) =c(ix)

        //  Core::LinAlg::Matrix<nsd_,nen_> proj_tang_derxy_m_1;
        //  // proj_matrix_derxy_m_ = alpha * 2.0 * P^t_{jk} * derxy_m(k,IX) * mu_m * timefacfac *
        //  km
        //  //                      = 2.0 * mu_m * timefacefac * km * p_1(IX,j)
        //  // IF P^t_{jk} IS SYMMETRIC: p_1(IX,j) = p_2(IX,j)
        //  proj_tang_derxy_m_1.multiply(proj_matrix_,derxy_m_viscm_timefacfac_km);

        //  Core::LinAlg::Matrix<nsd_,nen_> proj_tang_derxy_m_2;
        //  // proj_tang_derxy_m = 2.0 * P^t_{jk} * derxy_m(k,IX) * mu_m * timefacfac * km
        //  //                   = 2.0 * mu_m * timefacefac * km * p_2(IX,j)
        //  // IF P^t_{jk} IS SYMMETRIC: p_1(IX,j) = p_2(IX,j)
        //  proj_tang_derxy_m_2.multiply_tn(proj_matrix_,derxy_m_viscm_timefacfac_km);

        //  Core::LinAlg::Matrix<nen_,nen_> derxy_m_P_derxy_m;
        // derxy_m_P_derxy_m = 2.0 * derxy_m(j,IC) P^t_{jk} * derxy_m(k,IR) * mu_m * timefacfac * km
        //                   = 2.0 * C(IC,IR) * mu_m * timefacfac * km
        //  derxy_m_P_derxy_m.multiply_tn(derxy_m,proj_tang_derxy_m_1);
        derxy_m_p_derxy_m_.multiply_tn(derxy_m, proj_matrix_derxy_m_);

        // here we use a non-optimal order to assemble the values into C_umum
        // however for this term we have to save operations
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double normal_deriv_tmp = normal_deriv_m_viscm_km_(
              ir);  // alpha * mu_m * timefacfac *km * \sum_k dN^(ir)/dx_k * n_k

          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned row = m_index(ir, ivel);

              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                c_umum_(row, m_index(ic, jvel)) -=
                    facmm *
                    (normal_deriv_m_(ic) *
                            (proj_matrix_(jvel, ivel) * normal_deriv_tmp +
                                proj_matrix_derxy_m_(jvel, ir) * half_normal_(ivel)) +
                        normal(ivel) * half_normal_(jvel) * derxy_m_p_derxy_m_(ic, ir) +
                        normal_deriv_m_(ir) * proj_matrix_derxy_m_(ivel, ic) * half_normal_(jvel));
              }
            }
          }
        }

        // 2.0 * timefacfac * visceff_m * 0.5* (\nabla u + (\nabla u)^T) * normal
        // vderxy_m_normal_transposed_viscm_timefacfac_km_

        vderxy_m_normal_tang_.multiply(vderxy_m, normal);
        vderxy_m_normal_transposed_.multiply_tn(vderxy_m, normal);
        vderxy_m_normal_transposed_.update(
            1.0, vderxy_m_normal_tang_, 1.0);  // (\nabla u + (\nabla u)^T) * normal

        // (\nabla u + (\nabla u)^T) * normal * P^t
        vderxy_m_normal_tang_.multiply_tn(proj_matrix_, vderxy_m_normal_transposed_);

        // 2.0 * derxy_m(k,IX) * mu_m * timefacfac * km ( (\nabla u + (\nabla u)^T) * normal * P^t
        // )_k
        static Core::LinAlg::Matrix<nen_, 1> tmp_rhs;
        tmp_rhs.multiply_tn(derxy_m_viscm_timefacfac_km, vderxy_m_normal_tang_);

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          // alpha * mu_m * timefacfac *km * \sum_k dN^(ir)/dx_k * n_k
          const double normal_deriv_tmp = normal_deriv_m_viscm_km_(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            rh_c_um_(m_index(ir, ivel), 0) +=
                facmm *
                (normal_deriv_tmp * vderxy_m_normal_tang_(ivel) + tmp_rhs(ir) * half_normal_(ivel));
          }
        }
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_stab_penalty(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
          const double& timefacfac,                      ///< time integration factor
          const std::pair<bool, double>& m_row,          ///< scaling for master row
          const std::pair<bool, double>& s_row,          ///< scaling for slave row
          const std::pair<bool, double>& m_col,          ///< scaling for master col
          const std::pair<bool, double>& s_col,          ///< scaling for slave col
          bool only_rhs)
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::nit_stab_penalty");

        // viscous stability term

        // combined viscous and inflow stabilization for one-sided problems (XFSI)
        // gamma_combined = max(alpha*mu/hk, |u*n| )
        /*                      _        \        /                     i   _   \
       |  gamma_combined *  v , u - u     | =  - |   gamma/h_K *  v , (u  - u)   |
        \                                /        \                            */

        // just viscous stabilization for two-sided problems (XFF, XFFSI)
        /*                                  \        /                           i   \
       |  gamma*mu/h_K *  [ v ] , [ Du ]     | =  - |   gamma*mu/h_K * [ v ], [ u ]   |
        \                                   /        \                              */

        // + gamma*mu/h_K (vm, um))

        const double stabfac_timefacfac_m = timefacfac * m_row.second;
        velint_diff_timefacfac_stabfac_.update(stabfac_timefacfac_m, velint_diff_, 0.0);

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // +(stab * vm, u_DBC) (weak dirichlet case) or from
            // +(stab * vm, u_s)
            rh_c_um_(m_index(ir, ivel), 0) -= tmp_val * velint_diff_timefacfac_stabfac_(ivel);
          }
        }

        if (s_row.first)
        {
          const double stabfac_timefacfac_s = timefacfac * s_row.second;
          velint_diff_timefacfac_stabfac_.update(stabfac_timefacfac_s, velint_diff_, 0.0);

          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double tmp_val = funct_s_(ir);

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // +(stab * vs, um)
              // -(stab * vs, us)
              rh_c_us_(s_index(ir, ivel), 0) += tmp_val * velint_diff_timefacfac_stabfac_(ivel);
            }
          }
        }

        if (only_rhs) return;

        const double stabfac_timefacfac_mm = timefacfac * m_row.second * m_col.second;

        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const double tmp_val = funct_m_m_dyad_(ir, ic) * stabfac_timefacfac_mm;

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              c_umum_(m_index(ir, ivel), m_index(ic, ivel)) += tmp_val;
            }
          }
        }

        if (s_col.first)
        {
          // - gamma*mu/h_K (vm, us))
          // - gamma*mu/h_K (vs, um))

          const double stabfac_timefacfac_ms = timefacfac * m_row.second * s_col.second;

          for (unsigned ic = 0; ic < slave_nen_; ++ic)
          {
            for (unsigned ir = 0; ir < nen_; ++ir)
            {
              const double tmp_val = funct_s_m_dyad_(ic, ir) * stabfac_timefacfac_ms;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                c_umus_(m_index(ir, ivel), s_index(ic, ivel)) -= tmp_val;
              }
            }
          }
        }

        if (s_row.first && s_col.first)
        {
          const double stabfac_timefacfac_ss = timefacfac * s_row.second * s_col.second;

          for (unsigned ic = 0; ic < slave_nen_; ++ic)
          {
            // + gamma*mu/h_K (vs, us))
            for (unsigned ir = 0; ir < slave_nen_; ++ir)
            {
              const double tmp_val = funct_s_s_dyad_(ir, ic) * stabfac_timefacfac_ss;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                c_usus_(s_index(ir, ivel), s_index(ic, ivel)) += tmp_val;
              }
            }
          }
        }

        if (s_row.first)
        {
          const double stabfac_timefacfac_sm = timefacfac * s_row.second * m_col.second;

          for (unsigned ic = 0; ic < nen_; ++ic)
          {
            for (unsigned ir = 0; ir < slave_nen_; ++ir)
            {
              const double tmp_val = funct_s_m_dyad_(ir, ic) * stabfac_timefacfac_sm;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                c_usum_(s_index(ir, ivel), m_index(ic, ivel)) -= tmp_val;
              }
            }
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_stab_penalty_lin(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,  ///< funct
          const double& timefacfac,                      ///< time integration factor
          const std::pair<bool, double>& m_row,          ///< scaling for master row
          const std::pair<bool, double>&
              m_row_linm1,  ///< linearization of scaling for master row w.r.t. master comp. one
          const std::pair<bool, double>&
              m_row_linm2,  ///< linearization of scaling for master row w.r.t. master comp. two
          const std::pair<bool, double>&
              m_row_linm3,  ///< linearization of scaling for master row w.r.t. master comp. three
          bool only_rhs)
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::NIT_Stab_Penalty_linearization");

        if (only_rhs) return;

        const double stabfac_timefacfac_m = timefacfac;
        velint_diff_timefacfac_stabfac_.update(stabfac_timefacfac_m, velint_diff_, 0.0);

        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            for (unsigned ic = 0; ic < nen_; ++ic)
            {
              c_umum_(m_index(ir, ivel), m_index(ic, 0)) += tmp_val *
                                                            velint_diff_timefacfac_stabfac_(ivel) *
                                                            funct_m(ic) * m_row_linm1.second;
              c_umum_(m_index(ir, ivel), m_index(ic, 1)) += tmp_val *
                                                            velint_diff_timefacfac_stabfac_(ivel) *
                                                            funct_m(ic) * m_row_linm2.second;
              c_umum_(m_index(ir, ivel), m_index(ic, 2)) += tmp_val *
                                                            velint_diff_timefacfac_stabfac_(ivel) *
                                                            funct_m(ic) * m_row_linm3.second;
            }
          }
        }
        return;
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_stab_penalty_projected(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,               ///< funct
          const Core::LinAlg::Matrix<nsd_, nsd_>& projection_matrix,  ///< projection_matrix
          const Core::LinAlg::Matrix<nsd_, 1>&
              velint_diff_proj_matrix,           ///< velocity difference projected
          const double& timefacfac,              ///< time integration factor
          const std::pair<bool, double>& m_row,  ///< scaling for master row
          const std::pair<bool, double>& s_row,  ///< scaling for slave row
          const std::pair<bool, double>& m_col,  ///< scaling for master col
          const std::pair<bool, double>& s_col   ///< scaling for slave col
      )
      {
        TEUCHOS_FUNC_TIME_MONITOR("FLD::nit_stab_penalty");

        // viscous stability term

        // combined viscous and inflow stabilization for one-sided problems (XFSI)
        // gamma_combined = max(alpha*mu/hk, |u*n| )
        /*                      _        \        /                     i   _   \
       |  gamma_combined *  v , u - u     | =  - |   gamma/h_K *  v , (u  - u)   |
        \                                /        \                            */

        // just viscous stabilization for two-sided problems (XFF, XFFSI)
        /*                                  \        /                           i   \
       |  gamma*mu/h_K *  [ v ] , [ Du ]     | =  - |   gamma*mu/h_K * [ v ], [ u ]   |
        \                                   /        \                              */

        // + gamma*mu/h_K (vm, um))


        // 2) (Normal - Tangential split):
        /*                                                                                       \
      + |   [v]  ,  [ Du ] ( gamma_comb_n P_n + {mu} * gamma*h_E/(gamma*h_E+epsilon) * P_t )     | =
        \                                                                                       */

        /*             i                                                                         \
      - |   [v]  ,  [ u ] ( gamma_comb_n P_n + {mu} * gamma*h_E/(gamma*h_E+epsilon) * P_t )      | =
        \                                                                                       */

        velint_diff_proj_matrix_ = velint_diff_proj_matrix;
        proj_matrix_ = projection_matrix;

        const double stabfac_timefacfac_mm = timefacfac * m_row.second * m_col.second;

        for (unsigned ic = 0; ic < nen_; ++ic)
        {
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const double stab_funct_m_m_dyad_iric = funct_m_m_dyad_(ir, ic) * stabfac_timefacfac_mm;
            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              const unsigned col = m_index(ic, ivel);
              //        C_umum_(m_index(ir,ivel), m_index(ic,ivel)) += tmp_val;
              for (unsigned jvel = 0; jvel < nsd_; ++jvel)
              {
                c_umum_(m_index(ir, jvel), col) +=
                    stab_funct_m_m_dyad_iric * proj_matrix_(ivel, jvel);
              }
            }
          }
        }

        const double stabfac_timefacfac_m = timefacfac * m_row.second;
        velint_diff_timefacfac_stabfac_.update(stabfac_timefacfac_m, velint_diff_proj_matrix_, 0.0);
        for (unsigned ir = 0; ir < nen_; ++ir)
        {
          const double tmp_val = funct_m(ir);

          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            // +(stab * vm, u_DBC) (weak dirichlet case) or from
            // +(stab * vm, u_s)
            rh_c_um_(m_index(ir, ivel), 0) -= tmp_val * (velint_diff_timefacfac_stabfac_(ivel));
          }
        }


        if (s_col.first)
        {
          // - gamma*mu/h_K (vm, us))
          // - gamma*mu/h_K (vs, um))

          const double stabfac_timefacfac_ms = timefacfac * m_row.second * s_col.second;

          for (unsigned ic = 0; ic < slave_nen_; ++ic)
          {
            for (unsigned ir = 0; ir < nen_; ++ir)
            {
              const double tmp_val = funct_s_m_dyad_(ic, ir) * stabfac_timefacfac_ms;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                for (unsigned jvel = 0; jvel < nsd_; ++jvel)
                {
                  c_umus_(m_index(ir, jvel), s_index(ic, ivel)) -=
                      tmp_val * proj_matrix_(ivel, jvel);
                }
              }
            }
          }
        }

        if (s_row.first && s_col.first)
        {
          const double stabfac_timefacfac_ss = timefacfac * s_row.second * s_col.second;

          for (unsigned ic = 0; ic < slave_nen_; ++ic)
          {
            // + gamma*mu/h_K (vs, us))
            for (unsigned ir = 0; ir < slave_nen_; ++ir)
            {
              const double tmp_val = funct_s_s_dyad_(ir, ic) * stabfac_timefacfac_ss;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                for (unsigned jvel = 0; jvel < nsd_; ++jvel)
                {
                  c_usus_(s_index(ir, jvel), s_index(ic, ivel)) +=
                      tmp_val * proj_matrix_(ivel, jvel);
                }
              }
            }
          }
        }

        if (s_row.first)
        {
          const double stabfac_timefacfac_sm = timefacfac * s_row.second * m_col.second;

          for (unsigned ic = 0; ic < nen_; ++ic)
          {
            for (unsigned ir = 0; ir < slave_nen_; ++ir)
            {
              const double tmp_val = funct_s_m_dyad_(ir, ic) * stabfac_timefacfac_sm;

              for (unsigned ivel = 0; ivel < nsd_; ++ivel)
              {
                for (unsigned jvel = 0; jvel < nsd_; ++jvel)
                {
                  c_usum_(s_index(ir, jvel), m_index(ic, ivel)) -=
                      tmp_val * proj_matrix_(ivel, jvel);
                }
              }
            }
          }

          const double stabfac_timefacfac_s = timefacfac * s_row.second;
          velint_diff_timefacfac_stabfac_.update(
              stabfac_timefacfac_s, velint_diff_proj_matrix_, 0.0);

          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double tmp_val = funct_s_(ir);

            for (unsigned ivel = 0; ivel < nsd_; ++ivel)
            {
              // +(stab * vs, um)
              // -(stab * vs, us)
              rh_c_us_(s_index(ir, ivel), 0) += tmp_val * velint_diff_timefacfac_stabfac_(ivel);
            }
          }
        }

        return;
      }



      /*----------------------------------------------------------------------*
       * add averaged term to balance instabilities due to convective
       * mass transport across the fluid-fluid interface
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::nit_stab_inflow_averaged_term(
          const Core::LinAlg::Matrix<nen_, 1>& funct_m,   ///< funct
          const Core::LinAlg::Matrix<nsd_, 1>& velint_m,  ///< master velocity
          const Core::LinAlg::Matrix<nsd_, 1>& normal,    ///< normal vector n^m
          const double& density,                          ///< fluid density
          const double& timefacfac,                       ///< timefac * fac
          bool only_rhs)
      {
        //
        /*                                        \
       -|  [rho * (beta * n)] *  { v }_m , [   u ] |
        \ ----stab_avg-----                       / */

        // { v }_m = 0.5* (v^b + v^e) leads to the scaling with 0.5;
        // beta: convective velocity, currently beta=u^b_Gamma;
        // n:= n^b
        const double stabfac_avg_scaled = 0.5 * velint_m.dot(normal) * density * timefacfac;

        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const unsigned mrow = m_index(ir, ivel);

            const double tmp = funct_m(ir) * stabfac_avg_scaled;
            rh_c_um_(mrow, 0) += tmp * velint_diff_(ivel);
          }

          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const unsigned srow = s_index(ir, ivel);

            const double tmp = funct_s_(ir) * stabfac_avg_scaled;
            rh_c_us_(srow, 0) += tmp * velint_diff_(ivel);
          }
        }

        if (only_rhs) return;

        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          //  [rho * (beta * n^b)] (0.5*vb,ub)
          for (unsigned ir = 0; ir < nen_; ++ir)
          {
            const unsigned mrow = m_index(ir, ivel);

            for (unsigned ic = 0; ic < nen_; ++ic)
            {
              c_umum_(mrow, m_index(ic, ivel)) -= funct_m_m_dyad_(ir, ic) * stabfac_avg_scaled;
            }

            //  -[rho * (beta * n^b)] (0.5*vb,ue)
            for (unsigned ic = 0; ic < slave_nen_; ++ic)
            {
              c_umus_(mrow, s_index(ic, ivel)) += funct_s_m_dyad_(ic, ir) * stabfac_avg_scaled;
            }
          }

          //  [rho * (beta * n^b)] (0.5*ve,ub)
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const unsigned srow = s_index(ir, ivel);

            for (unsigned ic = 0; ic < nen_; ++ic)
            {
              c_usum_(srow, m_index(ic, ivel)) -= funct_s_m_dyad_(ir, ic) * stabfac_avg_scaled;
            }

            //-[rho * (beta * n^b)] (0.5*ve,ue)
            for (unsigned ic = 0; ic < slave_nen_; ++ic)
            {
              c_usus_(srow, s_index(ic, ivel)) += funct_s_s_dyad_(ir, ic) * stabfac_avg_scaled;
            }
          }
        }
        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          nit_create_standard_projection_matrices(
              const Core::LinAlg::Matrix<nsd_, 1>& normal  ///< normal vector n^b
          )
      {
        // Create the identity matrix (Probably not the fastest way...) Might make it global?
        proj_tangential_.put_scalar(0.0);
        for (unsigned int i = 0; i < nsd_; ++i) proj_tangential_(i, i) = 1;

        //   Non-smoothed projection matrix
        for (unsigned int i = 0; i < nsd_; ++i)
        {
          for (unsigned int j = 0; j < nsd_; ++j)
          {
            proj_tangential_(i, j) -= normal(i, 0) * normal(j, 0);
          }
        }

        proj_normal_.put_scalar(0.0);
        for (unsigned i = 0; i < nsd_; ++i) proj_normal_(i, i) = 1.0;
        proj_normal_.update(-1.0, proj_tangential_, 1.0);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void NitscheCoupling<distype, slave_distype, slave_numdof>::
          do_nit_visc_adjoint_and_neumann_master_terms_projected(
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
          )
      {
        velint_diff_proj_matrix_ = velint_diff_proj_matrix;
        proj_matrix_ = projection_matrix;

        // 2.0 * timefacfac * visceff_m * k_m * [\nabla N^(IX)]_k P^t_{kj}
        proj_matrix_derxy_m_.multiply_tn(
            proj_matrix_, derxy_m);  // Apply from right for consistency
        proj_matrix_derxy_m_.scale(km_viscm_fac);

        //-----------------------------------------------------------------
        // viscous adjoint consistency term

        double tmp_fac = adj_visc_scale_ * km_viscm_fac;
        derxy_m_viscm_timefacfac_.update(tmp_fac, derxy_m);  // 2 * mu_m * timefacfac *
                                                             // derxy_m(k,ic)

        // Scale with adjoint viscous scaling {-1,+1}
        proj_matrix_derxy_m_.scale(adj_visc_scale_);

        // Same as half_normal_deriv_m_viscm_timefacfac_km_? Might be unnecessary?
        // normal_deriv_m_viscm_km_ = alpha * half_normal(k)* 2 * km * mu_m * timefacfac *
        // derxy_m(k,IX)
        //                          = alpha * mu_m * viscfac_km * c(IX)
        normal_deriv_m_viscm_km_.multiply_tn(derxy_m_viscm_timefacfac_, half_normal_);

        nit_visc_adjoint_consistency_master_terms_projected(
            derxy_m_viscm_timefacfac_, funct_m, normal, m_row, m_col, s_col);

#ifndef ENFORCE_URQUIZA_GNBC
        //-----------------------------------------------------------------
        // Terms needed for Neumann consistency terms

        if (mstr_col.first)
        {
          nit_visc_neumann_adjoint_consistency_master_terms_projected(
              derxy_m_viscm_timefacfac_, derxy_m, vderxy_m, funct_m, normal, m_row, mstr_col);
        }
        //-----------------------------------------------------------------
#endif

        return;
      }


    }  // namespace XFLUID
  }  // namespace Elements
}  // namespace Discret



// pairs with numdof=3
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 3>;

template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::dis_none, 3>;

// volume coupled with numdof = 3, FSI Slavesided, FPI
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex8, 3>;

// pairs with numdof=4
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 4>;
//

// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tet4, 4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::wedge15,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex27, 4>;
// template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::wedge15,4>;


template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::dis_none, 4>;
template class Discret::Elements::XFLUID::NitscheCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::dis_none, 4>;

FOUR_C_NAMESPACE_CLOSE
