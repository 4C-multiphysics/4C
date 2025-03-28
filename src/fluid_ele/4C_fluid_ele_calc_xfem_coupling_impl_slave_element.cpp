// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_position.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_ele_calc_xfem_coupling_impl.hpp"
#include "4C_xfem_interface_utils.hpp"

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
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::add_slave_ele_disp(
          const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
          const std::vector<int>& lm                 ///< local map
      )
      {
        std::vector<double> mymatrix(lm.size());
        add_slave_ele_disp(slavedis, lm, mymatrix);
        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::add_slave_ele_disp(
          const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
          const std::vector<int>& lm,                ///< local map
          std::vector<double>& mymatrix              ///< slave element displacement vector
      )
      {
        // leave, if displacements are not set
        if (!slavedis.has_state(disp_statename_)) return;
        // get state of the global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
            slavedis.get_state(disp_statename_);
        if (matrix_state == nullptr)
          FOUR_C_THROW("Cannot get state vector {}", disp_statename_.c_str());

        // extract local values of the global vector
        mymatrix = Core::FE::extract_values(*matrix_state, lm);

        for (unsigned inode = 0; inode < slave_nen_; ++inode)  // number of nodes
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)  // number of dimensions
          {
            (slave_disp_)(idim, inode) =
                mymatrix[idim + (inode * slave_numdof)];  // attention! for fluidfluid disp state
                                                          // vector has 3+1 dofs for displacement
                                                          // (the same as for (u,p))
          }
        }

        // add the displacement of the interface
        for (unsigned inode = 0; inode < slave_nen_; ++inode)
        {
          slave_xyze_(0, inode) += slave_disp_(0, inode);
          slave_xyze_(1, inode) += slave_disp_(1, inode);
          slave_xyze_(2, inode) += slave_disp_(2, inode);
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::set_slave_state(
          const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
          const std::vector<int>& lm                 ///< local map
      )
      {
        // get state of the global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
            slavedis.get_state(vel_statename_);
        if (matrix_state == nullptr)
          FOUR_C_THROW("Cannot get state vector {}", vel_statename_.c_str());

        // extract local values of the global vectors
        std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

        for (unsigned inode = 0; inode < slave_nen_; ++inode)  // number of nodes
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)  // number of dimensions
          {
            (slave_vel_)(idim, inode) =
                mymatrix[idim +
                         (inode * slave_numdof)];  // state vector includes velocity and pressure
          }
          if (slave_numdof == nsd_ + 1)
            (slave_pres_)(inode, 0) = mymatrix[nsd_ + (inode * slave_numdof)];
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::set_slave_staten(
          const Core::FE::Discretization& slavedis,  ///< coupling slave discretization
          const std::vector<int>& lm                 ///< local map
      )
      {
        // get state of the global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
            slavedis.get_state(veln_statename_);
        if (matrix_state == nullptr)
          FOUR_C_THROW("Cannot get state vector {}", veln_statename_.c_str());

        // extract local values of the global vectors
        std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

        for (unsigned inode = 0; inode < slave_nen_; ++inode)  // number of nodes
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)  // number of dimensions
          {
            (slave_veln_)(idim, inode) =
                mymatrix[idim +
                         (inode * slave_numdof)];  // state vector includes velocity and pressure
          }
          if (slave_numdof == nsd_ + 1)
            (slave_presn_)(inode, 0) = mymatrix[nsd_ + (inode * slave_numdof)];
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_velnp(
          Core::LinAlg::Matrix<nsd_, 1>& ivelint  ///< interface velocity at coupling slave side
      ) const
      {
        ivelint.multiply(slave_vel_, slave_funct_);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_veln(
          Core::LinAlg::Matrix<nsd_, 1>& ivelintn  ///< interface velocity at coupling slave side
      ) const
      {
        ivelintn.multiply(slave_veln_, slave_funct_);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_presnp(
          double& ipres  ///< interface pressure at coupling slave side
      ) const
      {
        // pressure at current gauss-point
        ipres = slave_funct_.dot(slave_pres_);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_presn(
          double& ipresn  ///< interface pressure at coupling slave side
      ) const
      {
        // pressure at current gauss-point
        ipresn = slave_funct_.dot(slave_presn_);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_vel_gradnp(
          Core::LinAlg::Matrix<nsd_, nsd_>&
              velgradint  ///< interface velocity gradients at coupling slave side
      ) const
      {
        velgradint = slave_vderxy_;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_vel_gradn(
          Core::LinAlg::Matrix<nsd_, nsd_>&
              velgradintn  ///< interface velocity gradients at coupling slave side
      ) const
      {
        velgradintn = slave_vderxyn_;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_slave_funct(
          Core::LinAlg::Matrix<slave_nen_, 1>& slave_funct  ///< coupling slave shape functions
      ) const
      {
        slave_funct = slave_funct_;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::set_interface_jump_statenp(
          const Core::FE::Discretization& cutterdis,  ///< cutter discretization
          const std::string state,                    ///< state
          const std::vector<int>& lm                  ///< local map
      )
      {
        // get state of the global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
            cutterdis.get_state(state);
        if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state.c_str());

        // extract local values of the global vectors
        std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

        for (unsigned inode = 0; inode < slave_nen_; ++inode)  // number of nodes
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)  // number of dimensions
          {
            (interface_velnp_jump_)(idim, inode) =
                mymatrix[idim +
                         (inode * slave_numdof)];  // state vector includes velocity and pressure
            // no pressure jump required
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::set_interface_jump_staten(
          const Core::FE::Discretization& cutterdis,  ///< cutter discretization
          const std::string state,                    ///< state
          const std::vector<int>& lm                  ///< local map
      )
      {
        // get state of the global vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
            cutterdis.get_state(state);
        if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state.c_str());

        // extract local values of the global vectors
        std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

        for (unsigned inode = 0; inode < slave_nen_; ++inode)  // number of nodes
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)  // number of dimensions
          {
            (interface_veln_jump_)(idim, inode) =
                mymatrix[idim +
                         (inode * slave_numdof)];  // state vector includes velocity and pressure
            // no pressure jump required
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_jump_velnp(
          Core::LinAlg::Matrix<nsd_, 1>& ivelint_jump  ///< cutter element interface velocity jump
                                                       ///< or prescribed DBC at Gaussian point
      ) const
      {
        ivelint_jump.multiply(interface_velnp_jump_, slave_funct_);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_interface_jump_veln(
          Core::LinAlg::Matrix<nsd_, 1>& ivelintn_jump  ///< cutter element interface velocity jump
                                                        ///< or prescribed DBC at Gaussian point
      ) const
      {
        ivelintn_jump.multiply(interface_veln_jump_, slave_funct_);
      }


      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::evaluate(
          Core::LinAlg::Matrix<nsd_, 1>& xslave)
      {
        Core::LinAlg::Matrix<3, 1> rst_slave(Core::LinAlg::Initialization::zero);
        evaluate(xslave, rst_slave);
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::evaluate(
          Core::LinAlg::Matrix<nsd_, 1>& xslave, Core::LinAlg::Matrix<nsd_, 1>& rst_slave)
      {
        // coupling with a 2D element
        if (slave_nsd_ == nsd_ - 1)
        {
          // evaluate shape function at solution
          Core::FE::shape_function_2d(slave_funct_, xslave(0), xslave(1), slave_distype);
          rst_slave(0) = xslave(0);
          rst_slave(1) = xslave(1);
          //    FOUR_C_THROW("You called 3D evaluation routine when coupling with a 2D element.");
          return;
        }

        // find element local position of gauss point
        std::shared_ptr<Cut::Position> pos =
            Cut::PositionFactory::build_position<nsd_, slave_distype>(slave_xyze_, xslave);
        pos->compute();

        if (slave_nsd_ == nsd_)
        {
          pos->local_coordinates(rst_slave);
          Core::FE::shape_function_3d(
              slave_funct_, rst_slave(0), rst_slave(1), rst_slave(2), slave_distype);
          Core::FE::shape_function_3d_deriv1(
              slave_deriv_, rst_slave(0), rst_slave(1), rst_slave(2), slave_distype);
        }
        else
          FOUR_C_THROW("Unsupported dimension clash!");


        Core::LinAlg::Matrix<nsd_, nsd_> slave_xjm(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nsd_, nsd_> slave_xji(Core::LinAlg::Initialization::zero);

        slave_xjm.multiply_nt(slave_deriv_, slave_xyze_);
        slave_xji.invert(slave_xjm);

        // compute global first derivates
        slave_derxy_.multiply(slave_xji, slave_deriv_);

        // get velocity derivatives at integration point
        // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
        slave_vderxy_.multiply_nt(slave_vel_, slave_derxy_);
        // previous time step
        slave_vderxyn_.multiply_nt(slave_veln_, slave_derxy_);

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void
      SlaveElementRepresentation<distype, slave_distype, slave_numdof>::compute_interface_force(
          Core::LinAlg::SerialDenseVector& iforce,  ///< interface force vector
          Core::LinAlg::Matrix<nsd_, 1>& traction,  ///< traction vector at gaussian point
          const double& fac                         ///< integration factor
      )
      {
        for (unsigned inode = 0; inode < slave_nen_; ++inode)
        {
          double tmp = slave_funct_(inode) * fac;

          for (unsigned idim = 0; idim < nsd_; ++idim)
          {
            // f^i = ( N^i, t ) = ( N^i, (-pI+2mu*eps(u))*n )
            iforce[idim + inode * nsd_] += tmp * traction(idim);
          }
        }

        return;
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::project_on_side(
          Core::LinAlg::Matrix<nsd_, 1>&
              x_gp_lin,  ///< global coordinates of gaussian point w.r.t linearized interface
          Core::LinAlg::Matrix<nsd_, 1>& x_side,  ///< projected gaussian point on side
          Core::LinAlg::Matrix<nsd_, 1>&
              xi_side  ///< local coordinates of projected gaussian point w.r.t side
      )
      {
        //  TEUCHOS_FUNC_TIME_MONITOR( "FLD::XFluid::XFluidState::project_on_side" );

        // check, if called on a 3D-element
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (slave_nsd_ == nsd_)
        {
          FOUR_C_THROW(
              "You can't project onto a 3D coupling slave element directly. You need an associated "
              "boundary element!");
        }
#endif

        TEUCHOS_FUNC_TIME_MONITOR("FLD::XFluid::XFluidState::project_on_side");


        if (slave_distype == Core::FE::CellType::tri3 or slave_distype == Core::FE::CellType::tri6)
        {
          proj_sol_(0) = 0.333333333333333;
          proj_sol_(1) = 0.333333333333333;
        }
        else if (slave_distype == Core::FE::CellType::quad4 or
                 slave_distype == Core::FE::CellType::quad8 or
                 slave_distype == Core::FE::CellType::quad9)
        {
          proj_sol_(0) = 0.0;
          proj_sol_(1) = 0.0;
        }
        else
        {
          FOUR_C_THROW("Define start side xi-coordinates for unsupported cell type");
        }
        proj_sol_(2) = 0.0;

        // we only use absolute tolerances, since we compute local coordinates
        const double absTolIncr = 1.0e-9;  // abs tolerance for the local coordinates increment
        const double absTolRes = 1.0e-9;   // abs tolerance for the whole residual
        const double absTOLdist = 1.0e-9;  // abs tolerance for distance

        unsigned iter = 0;
        const unsigned maxiter = 7;

        bool converged = false;

        while (iter < maxiter && !converged)
        {
          ++iter;

          // get current values
          Core::FE::shape_function_2d(proj_funct_, proj_sol_(0), proj_sol_(1), slave_distype);
          Core::FE::shape_function_2d_deriv1(
              proj_deriv_, proj_sol_(0), proj_sol_(1), slave_distype);
          Core::FE::shape_function_2d_deriv2(
              proj_deriv2_, proj_sol_(0), proj_sol_(1), slave_distype);

          proj_x_.multiply(slave_xyze_, proj_funct_);
          proj_derxy_.multiply_nt(slave_xyze_, proj_deriv_);
          proj_derxy2_.multiply_nt(slave_xyze_, proj_deriv2_);

          proj_dx_drdr_times_dx_ds_(0) =
              proj_derxy2_(1, 0) * proj_derxy_(2, 1) - proj_derxy_(1, 1) * proj_derxy2_(2, 0);
          proj_dx_drdr_times_dx_ds_(1) =
              proj_derxy2_(2, 0) * proj_derxy_(0, 1) - proj_derxy_(2, 1) * proj_derxy2_(0, 0);
          proj_dx_drdr_times_dx_ds_(2) =
              proj_derxy2_(0, 0) * proj_derxy_(1, 1) - proj_derxy_(0, 1) * proj_derxy2_(1, 0);

          proj_dx_dr_times_dx_drds_(0) =
              proj_derxy_(1, 0) * proj_derxy2_(2, 2) - proj_derxy2_(1, 2) * proj_derxy_(2, 0);
          proj_dx_dr_times_dx_drds_(1) =
              proj_derxy_(2, 0) * proj_derxy2_(0, 2) - proj_derxy2_(2, 2) * proj_derxy_(0, 0);
          proj_dx_dr_times_dx_drds_(2) =
              proj_derxy_(0, 0) * proj_derxy2_(1, 2) - proj_derxy2_(0, 2) * proj_derxy_(1, 0);

          proj_dx_drds_times_dx_ds_(0) =
              proj_derxy2_(1, 2) * proj_derxy_(2, 1) - proj_derxy_(1, 1) * proj_derxy2_(2, 2);
          proj_dx_drds_times_dx_ds_(1) =
              proj_derxy2_(2, 2) * proj_derxy_(0, 1) - proj_derxy_(2, 1) * proj_derxy2_(0, 2);
          proj_dx_drds_times_dx_ds_(2) =
              proj_derxy2_(0, 2) * proj_derxy_(1, 1) - proj_derxy_(0, 1) * proj_derxy2_(1, 2);

          proj_dx_dr_times_dx_dsds_(0) =
              proj_derxy_(1, 0) * proj_derxy2_(2, 1) - proj_derxy2_(1, 1) * proj_derxy_(2, 0);
          proj_dx_dr_times_dx_dsds_(1) =
              proj_derxy_(2, 0) * proj_derxy2_(0, 1) - proj_derxy2_(2, 1) * proj_derxy_(0, 0);
          proj_dx_dr_times_dx_dsds_(2) =
              proj_derxy_(0, 0) * proj_derxy2_(1, 1) - proj_derxy2_(0, 1) * proj_derxy_(1, 0);

          proj_dx_dr_times_dx_ds_(0) =
              proj_derxy_(1, 0) * proj_derxy_(2, 1) - proj_derxy_(1, 1) * proj_derxy_(2, 0);
          proj_dx_dr_times_dx_ds_(1) =
              proj_derxy_(2, 0) * proj_derxy_(0, 1) - proj_derxy_(2, 1) * proj_derxy_(0, 0);
          proj_dx_dr_times_dx_ds_(2) =
              proj_derxy_(0, 0) * proj_derxy_(1, 1) - proj_derxy_(0, 1) * proj_derxy_(1, 0);

          // define sysmat
          for (unsigned i = 0; i < 3; ++i)
          {
            // d/dr
            proj_sysmat_(i, 0) =
                proj_derxy_(i, 0) -
                proj_sol_(2) * (proj_dx_drdr_times_dx_ds_(i) + proj_dx_dr_times_dx_drds_(i));

            // d/ds
            proj_sysmat_(i, 1) =
                proj_derxy_(i, 1) -
                proj_sol_(2) * (proj_dx_drds_times_dx_ds_(i) + proj_dx_dr_times_dx_dsds_(i));

            // d/d(dist)
            proj_sysmat_(i, 2) = -proj_dx_dr_times_dx_ds_(i);


            // residual
            proj_residuum_(i) =
                proj_x_(i) - proj_sol_(2) * proj_dx_dr_times_dx_ds_(i) - x_gp_lin(i);
          }

          proj_sysmat_.invert();

          // solve Newton iteration
          proj_incr_.multiply(
              -1.0, proj_sysmat_, proj_residuum_);  // incr = -Systemmatrix^-1 * residuum

          // update solution
          proj_sol_.update(1.0, proj_incr_, 1.0);

          // check 1) absolute criterion for local coordinates (between [-1,1]^2)
          //       2) absolute criterion for distance (-> 0)
          //       3) absolute criterion for whole residuum
          if (proj_incr_(2) < absTOLdist &&
              sqrt(proj_incr_(0) * proj_incr_(0) + proj_incr_(1) * proj_incr_(1)) < absTolIncr &&
              proj_residuum_.norm2() < absTolRes)
          {
            converged = true;
          }
        }

        if (!converged)
        {
          std::cout.precision(15);

          std::cout << "increment criterion loc coord "
                    //<< sqrt(incr(0)*incr(0)+incr(1)*incr(1))/sqrt(sol(0)*sol(0)+sol(1)*sol(1))
                    << sqrt(proj_incr_(0) * proj_incr_(0) + proj_incr_(1) * proj_incr_(1))
                    << " \tabsTOL: " << absTolIncr << std::endl;
          std::cout << "absolute criterion for distance " << proj_incr_(2)
                    << " \tabsTOL: " << absTOLdist << std::endl;
          std::cout << "relative criterion whole residuum " << proj_residuum_.norm2()
                    << " \tabsTOL: " << absTolRes << std::endl;


          std::cout << "sysmat.invert" << proj_sysmat_ << std::endl;
          std::cout << "sol-norm " << proj_sol_.norm2() << std::endl;
          std::cout << "sol " << proj_sol_ << std::endl;
          std::cout << "x_gp_lin" << x_gp_lin << std::endl;
          std::cout << "side " << slave_xyze_ << std::endl;

          FOUR_C_THROW("Newton scheme in project_on_side not converged! ");
        }

        // evaluate shape function at solution
        Core::FE::shape_function_2d(slave_funct_, proj_sol_(0), proj_sol_(1), slave_distype);

        // get projected gauss point
        x_side.multiply(slave_xyze_, slave_funct_);

        // set local coordinates w.r.t side
        xi_side(0) = proj_sol_(0);
        xi_side(1) = proj_sol_(1);
        xi_side(2) = 0.0;  // actually 2D coordinates
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      double SlaveElementRepresentation<distype, slave_distype, slave_numdof>::eval_element_volume()
      {
        switch (Core::FE::dim<slave_distype>)
        {
          case 3:
          {
            return XFEM::Utils::eval_element_volume<slave_distype>(slave_xyze_);
            break;
          }
          default:
          {
            FOUR_C_THROW("Element volume for non 3D element type?");
            return 0.0;
          }
        }
      }

      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      template <Core::FE::CellType distype, Core::FE::CellType slave_distype,
          unsigned int slave_numdof>
      void SlaveElementRepresentation<distype, slave_distype, slave_numdof>::get_slave_funct_deriv(
          Core::LinAlg::Matrix<nsd_, slave_nen_>& slave_derxy) const
      {
        slave_derxy = slave_derxy_;
      }


    }  // namespace XFLUID
  }  // namespace Elements
}  // namespace Discret


// pairs with numdof=3
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 3>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,3>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 3>;

// volume coupled with numdof = 3, FSI Slavesided, FPI
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex8, 3>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex8, 3>;

// pairs with numdof=4
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::quad9, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri3,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tri6,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad4, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::quad9, 4>;

// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tet4, 4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex8,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex20,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::hex27,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet4,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::tet10,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge6,
    Core::FE::CellType::hex27, 4>;
// template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tet4,4>; template class
// Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
// Core::FE::CellType::tet10,4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex8, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex20, 4>;
template class Discret::Elements::XFLUID::SlaveElementRepresentation<Core::FE::CellType::wedge15,
    Core::FE::CellType::hex27, 4>;

FOUR_C_NAMESPACE_CLOSE
