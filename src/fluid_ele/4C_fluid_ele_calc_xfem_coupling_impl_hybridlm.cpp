// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_xfem_coupling.hpp"
#include "4C_fluid_ele_calc_xfem_coupling_impl.hpp"

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
      void HybridLMCoupling<distype, slave_distype, slave_numdof>::mhcs_build_coupling_matrices(
          const Core::LinAlg::Matrix<nsd_, 1>& normal,  ///< normal vector
          const double& fac,                            ///< integration factor
          const Core::LinAlg::Matrix<nen_, 1>& funct,   ///< shape function
          Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
              rhs_s,  ///< block rhs vector \f$ rhs_{\sigma} \f$
          const Core::LinAlg::Matrix<nsd_, 1>&
              ivelint_jump,  ///< prescribed interface velocity or interface jump height
          const Core::LinAlg::Matrix<nsd_, 1>&
              itraction_jump  ///< prescribed interface traction or interface jump height
      )
      {
        Core::LinAlg::Matrix<nen_, slave_nen_> bK_ms;
        Core::LinAlg::Matrix<slave_nen_, nen_> bK_sm;

        // interface velocity at gauss-point (current gauss-point in calling method)
        Core::LinAlg::Matrix<nsd_, 1> velint_s(Core::LinAlg::Initialization::zero);
        this->get_interface_velnp(velint_s);

        // add the prescribed interface velocity for weak Dirichlet boundary conditions or the jump
        // height for coupled problems
        velint_s.update(1.0, ivelint_jump, 1.0);

        // get nodal shape function vector
        Core::LinAlg::Matrix<slave_nen_, 1> slave_funct(Core::LinAlg::Initialization::zero);
        this->get_slave_funct(slave_funct);

        bK_ms.multiply_nt(funct, slave_funct);
        bK_sm.update_t(bK_ms);

        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            const double tmp = fac * normal(jvel);

            const unsigned sigma = stress_index(ivel, jvel);
            // G_sus
            BG_sus_(sigma, ivel)->update(tmp, bK_ms, 1.0);
            rhs_s(sigma, 0)->update(-tmp * velint_s(ivel), funct, 1.0);

            // G_uss
            BG_uss_(ivel, sigma)->update(tmp, bK_sm, 1.0);
          }
        }

        const double km = 1.0;  // only master-sided weighting
        Core::LinAlg::Matrix<slave_nen_, 1>
            funct_s_timefacfac_km;  ///< funct_s * timefacfac *kappa_m
        funct_s_timefacfac_km.update(km, slave_funct, 0.0);

        // Traction Standard Consistency term
        mh_traction_consistency_term(funct_s_timefacfac_km, itraction_jump);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void HybridLMCoupling<distype, slave_distype, slave_numdof>::mhvs_build_coupling_matrices(
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
      )
      {
        // interface velocity at gauss-point (current gauss-point in calling method)
        Core::LinAlg::Matrix<nsd_, 1> velint_s(Core::LinAlg::Initialization::zero);
        this->get_interface_velnp(velint_s);

        // add the prescribed interface velocity for weak Dirichlet boundary conditions or the jump
        // height for coupled problems
        velint_s.update(1.0, ivelint_jump, 1.0);

        // block submatrices for interface coupling; s: slave side, m: master side (always
        // background here)
        Core::LinAlg::Matrix<nen_, slave_nen_> bG_ms(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<slave_nen_, nen_> bG_sm(Core::LinAlg::Initialization::zero);

        Core::LinAlg::Matrix<slave_nen_, 1> slave_funct;
        this->get_slave_funct(slave_funct);

        bG_ms.multiply_nt(funct, slave_funct);
        bG_sm.multiply_nt(slave_funct, funct);

        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          for (unsigned jvel = 0; jvel < nsd_; ++jvel)
          {
            /*
             * G_sus
             *
             * from:
             *   /              \
             *  | \tau^m n, u^s  |
             *   \              /
             */
            BG_sus_(stress_index(ivel, jvel), ivel)->update(fac * normal(jvel), bG_ms, 1.0);
            rhs_s(stress_index(ivel, jvel), 0)
                ->update(-fac * normal(jvel) * velint_s(ivel), funct, 1.0);

            /*
             *  G_uss
             *
             *  from:
             *   /                 \
             *  | v^s, \sigma^m n   |
             *   \                 /
             *
             */
            BG_uss_(ivel, stress_index(ivel, jvel))->update(fac * normal(jvel), bG_sm, 1.0);
          }

          // Build cross-interface pressure-velocity coupling matrices G_uip, G_pui!

          // G_pmus - from adjoint pressure consistency term
          /*
           *  /            \
           *  | q^m, u^s n  |
           *  \            /
           *
           */

          BG_pmus_(0, ivel)->update(fac * normal(ivel), bG_ms, 1.0);


          // G_uspm - from pressure consistency term
          /*
           *  /           \
           *  | -v^s, p n  |
           *  \           /
           *
           */

          BG_uspm_(ivel, 0)->update(-fac * normal(ivel), bG_sm, 1.0);
        }

        // add normal interface velocity to rhs vector (pressure row)
        const double svelnormal = velint_s.dot(normal);

        // (Viscous) stress/Velocities

        // rhs_pm_us, residual from:
        /*
         *  /            \
         *  | q^m, u^s n  |
         *  \            /
         *
         */
        rhs_pmus.update(-fac * svelnormal, funct, 1.0);

        // rhs_us_pm, residual from:
        /*
         *  /            \
         *  | -v^s, p^m n |
         *  \            /
         *
         */
        // belongs to the side and therefore contributes to rhC_ui_!
        for (unsigned ir = 0; ir < slave_nen_; ++ir)
        {
          for (unsigned ivel = 0; ivel < nsd_; ++ivel)
          {
            rhC_us_(s_index(ir, ivel), 0) += press * fac * normal(ivel) * slave_funct(ir);
          }

          rhC_us_(s_index(ir, Pres), 0) = 0.0;
        }

        const double km = 1.0;  // only master-sided weighting
        Core::LinAlg::Matrix<slave_nen_, 1>
            funct_s_timefacfac_km;  ///< funct_s * timefacfac *kappa_m
        funct_s_timefacfac_km.update(km, slave_funct, 0.0);

        // Traction Standard Consistency term
        mh_traction_consistency_term(funct_s_timefacfac_km, itraction_jump);


        return;
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void HybridLMCoupling<distype, slave_distype, slave_numdof>::mh_traction_consistency_term(
          const Core::LinAlg::Matrix<slave_nen_, 1>&
              funct_s_timefacfac_km,  ///< funct_s * timefacfac *kappa_m
          const Core::LinAlg::Matrix<nsd_, 1>&
              itraction_jump  ///< prescribed interface traction, jump height for coupled problems
      )
      {
        /*            \
     - |  < v >,   t   |   with t = [sigma * n]
        \             /     */

        // loop over velocity components
        for (unsigned ivel = 0; ivel < nsd_; ++ivel)
        {
          /*
          //-----------------------------------------------
          //    - (vm, ks * t) = 0 as ks=0
          //-----------------------------------------------
          */

          //-----------------------------------------------
          //    + (vs, km * t)
          //-----------------------------------------------
          for (unsigned ir = 0; ir < slave_nen_; ++ir)
          {
            const double funct_s_km_timefacfac_traction =
                funct_s_timefacfac_km(ir) * itraction_jump(ivel);

            const unsigned row = s_index(ir, ivel);
            rhC_us_(row, 0) += funct_s_km_timefacfac_traction;
          }
        }  // end loop over velocity components
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void HybridLMCoupling<distype, slave_distype, slave_numdof>::
          hybrid_lm_build_final_coupling_matrices(
              Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                  numstressdof_>& BinvK_ss,  ///< block inverse \f$ K^{-1}_{\sigma\sigma} \f$
              Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, master_numdof_,
                  numstressdof_>&
                  BKumsInvKss,  ///< block matrix \f$ K_{u\sigma} \cdot K^{-1}_{\sigma\sigma} \f$
              Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, nen_>, numstressdof_,
                  master_numdof_>& BK_sum,  ///< block matrix \f$ K_{\sigma u} \f$
              Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, 1>, numstressdof_, 1>&
                  rhs_s  ///< block rhs vector \f$ rhs_{\sigma}\f$
          )
      {
        // final coupling matrices in block form
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<nen_, slave_nen_>, master_numdof_, nsd_>
            BCumus;
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<slave_nen_, nen_>, nsd_, master_numdof_>
            BCusum;

        // auxiliary matrices for intermediate calculation results of the matrix product
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<slave_nen_, nen_>, nsd_, numstressdof_>
            BGussInvKss;
        Core::LinAlg::BlockMatrix<Core::LinAlg::Matrix<slave_nen_, 1>, nsd_, 1> BGussinvKssrhs_s;

        // BKusInvKss:
        // (K_ums + G_ums) * K_ss^-1 (MHVS) or
        // G_ums * K_ss^-1 (MHCS)

        // (K_ums + G_ums) * K_ss^-1 G_sus (MHVS) or
        // G_ums * K_ss^-1 G_sus (MHCS)
        BCumus.multiply(BKumsInvKss, BG_sus_);

        // G_uss * K_ss^-1
        BGussInvKss.multiply(BG_uss_, BinvK_ss);

        // G_uss * K_ss^-1 * (K_sum + G_sum) (MHVS) or
        // G_uss * K_ss^-1 * (K_sum + G_sum + K_spm) (MHCS)
        BCusum.multiply(BGussInvKss, BK_sum);

        // G_uss K_ss^-1 rhs_s
        BGussinvKssrhs_s.multiply(BGussInvKss, rhs_s);

        // transfer the entries from Cumus,Cusum, rhCus (in case of MHCS, coupling term us-p is
        // included in Cusum!)

        // loop over slave velocity dof
        for (unsigned isvel = 0; isvel < nsd_; ++isvel)
        {
          // loop over background element dof (velocity & pressure)
          for (unsigned imdof = 0; imdof < master_numdof_; ++imdof)
          {
            // (um-us)
            if (BCumus.is_used(imdof, isvel))
            {
              const Core::LinAlg::Matrix<nen_, slave_nen_>& bCumus = *BCumus(imdof, isvel);
              // loop over slave element nodes
              for (unsigned isn = 0; isn < slave_nen_; ++isn)
              {
                // loop over background element nodes
                for (unsigned imn = 0; imn < nen_; ++imn)
                {
                  C_umus_(m_index(imn, imdof), s_index(isn, isvel)) -= bCumus(imn, isn);
                }
              }
            }  // (um-us)

            // (us-um), MHCS: (us-pm)
            if (BCusum.is_used(isvel, imdof))
            {
              const Core::LinAlg::Matrix<slave_nen_, nen_>& bCusum = *BCusum(isvel, imdof);
              // loop over slave element nodes
              for (unsigned isn = 0; isn < slave_nen_; ++isn)
              {
                // loop over background element nodes
                for (unsigned imn = 0; imn < nen_; ++imn)
                {
                  C_usum_(s_index(isn, isvel), m_index(imn, imdof)) -= bCusum(isn, imn);
                }
              }
            }  // (us-um), MHCS: (us-pm)
          }

          // add surface-based pressure coupling terms (only MHVS)

          // (us-pm)
          if (BG_uspm_.is_used(isvel, 0))
          {
            const Core::LinAlg::Matrix<slave_nen_, nen_>& bGuspm = *BG_uspm_(isvel, 0);
            // loop over slave element nodes
            for (unsigned isn = 0; isn < slave_nen_; ++isn)
            {
              // loop over background element nodes
              for (unsigned imn = 0; imn < nen_; ++imn)
              {
                C_usum_(s_index(isn, isvel), m_index(imn, Pres)) = bGuspm(isn, imn);
              }
            }
          }  // (us-pm)

          // (pm-us)
          if (BG_pmus_.is_used(0, isvel))
          {
            const Core::LinAlg::Matrix<nen_, slave_nen_>& bGpmus = *BG_pmus_(0, isvel);
            // loop over slave element nodes
            for (unsigned isn = 0; isn < slave_nen_; ++isn)
            {
              // loop over background element nodes
              for (unsigned imn = 0; imn < nen_; ++imn)
              {
                C_umus_(m_index(imn, Pres), s_index(isn, isvel)) = bGpmus(imn, isn);
              }
            }
          }  // (pm-us)

          // rhs - us
          if (BGussinvKssrhs_s.is_used(isvel, 0))
          {
            const Core::LinAlg::Matrix<slave_nen_, 1>& bGussinvKssrhs_s =
                *BGussinvKssrhs_s(isvel, 0);

            // loop over slave element nodes
            for (unsigned isn = 0; isn < slave_nen_; ++isn)
            {
              rhC_us_(s_index(isn, isvel), 0) -= bGussinvKssrhs_s(isn, 0);
            }
          }  // rhs - us
        }  // end loop over slave velocity dof

        // finally, build G_uss & G_sus for C_usus

        // G_sus
        // loop over block rows
        for (unsigned ibr = 0; ibr < numstressdof_; ++ibr)
        {
          // loop over block columns (interface velocity)
          for (unsigned ibc = 0; ibc < nsd_; ++ibc)
          {
            // extract the stress-velocity coupling submatrix
            if (BG_sus_.is_used(ibr, ibc))
            {
              Core::LinAlg::Matrix<nen_, slave_nen_>& bGsus = *BG_sus_(ibr, ibc);

              // transfer the entries
              for (unsigned ir = 0; ir < nen_; ++ir)
              {
                unsigned stressrow = ibr + ir * numstressdof_;

                for (unsigned ic = 0; ic < slave_nen_; ++ic)
                {
                  unsigned slavevelcol = ibc + ic * slave_numdof;

                  G_sus_(stressrow, slavevelcol) = bGsus(ir, ic);
                }
              }
            }
          }
        }  // G_sus filled

        // fill G_uss_ from BG_uss_

        // loop over block columns (sigmaxx, sigmaxy, ...)
        for (unsigned ibc = 0; ibc < numstressdof_; ++ibc)
        {
          // loop over block rows (interface velocity)
          for (unsigned ibr = 0; ibr < nsd_; ++ibr)
          {
            if (BG_uss_.is_used(ibr, ibc))
            {
              Core::LinAlg::Matrix<slave_nen_, nen_>& bGuss = *BG_uss_(ibr, ibc);

              // transfer the entries
              for (unsigned ic = 0; ic < nen_; ++ic)
              {
                unsigned stresscol = ibc + ic * numstressdof_;

                for (unsigned ir = 0; ir < slave_nen_; ++ir)
                {
                  unsigned slavevelrow = ibr + ir * slave_numdof;

                  G_uss_(slavevelrow, stresscol) = bGuss(ir, ic);
                }
              }
            }
          }
        }  // G_uss filled
      }

    }  // namespace XFLUID
  }  // namespace Elements
}  // namespace Discret


// pairs with numdof=3
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 3>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 3>;

template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::dis_none, 3>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::dis_none, 3>;


// pairs with numdof=4
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 4>;
// template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::HybridLMCoupling<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 4>;

FOUR_C_NAMESPACE_CLOSE
