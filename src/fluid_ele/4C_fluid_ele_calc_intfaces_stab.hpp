// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_CALC_INTFACES_STAB_HPP
#define FOUR_C_FLUID_ELE_CALC_INTFACES_STAB_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_parameter_intface.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    //
    //                        INTERFACE CLASS
    //
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------

    /*-----------------------------------------------------------------

    \brief an interface class for edge-oriented/continuous interior penalty stabilization
           for every surface element distype this class allocates one
           instance of the edge-oriented stabilization implementation


    -----------------------------------------------------------------*/
    class FluidIntFaceStab
    {
     public:
      //! Empty constructor
      FluidIntFaceStab() {}
      //! Empty destructor
      virtual ~FluidIntFaceStab() = default;
      /*!
        \brief Evaluate the surface elements edge-oriented stabilization and ghost penalty

        This class does not provide a definition for this function, it
        is defined in the implementation.


      */
      virtual int evaluate_edge_based_stabilization(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          Discret::Elements::FluidEleParameterTimInt&
              fldparatimint,  ///< time-integration parameter
          Discret::Elements::FluidEleParameterIntFace&
              fldintfacepara,                        ///< general parameter for internal face
          Teuchos::ParameterList& params,            ///< parameter list
          Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& patchlm,                 ///< patch local map
          std::vector<int>& lm_masterToPatch,        ///< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,         ///< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,          ///< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  ///< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   ///< element vector blocks
          ) = 0;


      /*!
        \brief Allocate one static instance of the internal
               implementation class for edge oriented stabilization and
               return pointer to it


        \param surfele (in):   fluid internal surface element

      */
      static FluidIntFaceStab* impl(Discret::Elements::FluidIntFace* surfele);
    };


    //-----------------------------------------------------------------
    //-----------------------------------------------------------------
    //
    //                        IMPLEMENTATION
    //
    //-----------------------------------------------------------------
    //-----------------------------------------------------------------

    /*!
    \brief Internal FluidInternalSurfaceStab EOS-implementation
           (surface and parent/neighbor element specific)
    */

    //-----------------------------------------------------------------
    ///
    //-----------------------------------------------------------------
    template <Core::FE::CellType distype, Core::FE::CellType pdistype, Core::FE::CellType ndistype>
    class FluidInternalSurfaceStab : public FluidIntFaceStab
    {
     public:
      /// Singleton access method
      static FluidInternalSurfaceStab<distype, pdistype, ndistype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      /// Constructor with number of nodes
      FluidInternalSurfaceStab();


      /// number of space dimensions of the FluidIntFace element
      static constexpr int facensd_ = Core::FE::dim<distype>;

      /// number of space dimensions of the parent element
      static constexpr int nsd_ = facensd_ + 1;

      /// number of dof's per node
      static constexpr int numdofpernode_ = nsd_ + 1;

      /// number of nodes
      static constexpr int iel = Core::FE::num_nodes<distype>;

      /// number of parentnodes
      static constexpr int piel = Core::FE::num_nodes<pdistype>;

      /// number of parentnodes of neighbor element
      static constexpr int niel = Core::FE::num_nodes<ndistype>;

      /// number of second order derivatives for master element
      static constexpr int numderiv2_p = Core::FE::DisTypeToNumDeriv2<pdistype>::numderiv2;

      /// number of second order derivatives for slave element
      static constexpr int numderiv2_n = Core::FE::DisTypeToNumDeriv2<ndistype>::numderiv2;



      /*!
        \brief Evaluate EOS stabilization

        This method calculates the contributions to rhs and matrix of
        EOS and ghost penalty stabilization for a generalized alpha
        system.

        Literature:
        Burman Fernandez and Hansbo (2006-2009)

      */
      int evaluate_edge_based_stabilization(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          Discret::Elements::FluidEleParameterTimInt&
              fldparatimint,  ///< time-integration parameter
          Discret::Elements::FluidEleParameterIntFace&
              fldintfacepara,                        ///< general parameter for internal face
          Teuchos::ParameterList& params,            ///< parameter list
          Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& patchlm,                 ///< patch local map
          std::vector<int>& lm_masterToPatch,        ///< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,         ///< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,          ///< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  ///< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   ///< element vector blocks
          ) override;

     private:
      int degree(const Core::FE::CellType parent_ele_distype);

      //! reassemble matrix block from master-slave pairs to patch-node block for field (row, col)
      void reassemble_mat_block(const int row_block,   ///< row block
          const int col_block,                         ///< column block
          Core::LinAlg::SerialDenseMatrix& mat_block,  ///< matrix block
          Core::LinAlg::Matrix<numdofpernode_ * piel, numdofpernode_ * piel>&
              elematrix_mm,  ///< element matrix master-master block
          Core::LinAlg::Matrix<numdofpernode_ * piel, numdofpernode_ * niel>&
              elematrix_ms,  ///< element matrix master-slave block
          Core::LinAlg::Matrix<numdofpernode_ * niel, numdofpernode_ * piel>&
              elematrix_sm,  ///< element matrix slave-master block
          Core::LinAlg::Matrix<numdofpernode_ * niel, numdofpernode_ * niel>&
              elematrix_ss,  ///< element matrix slave-slave block
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch  ///< local map between slave nodes and nodes in patch
      );

      //! reassemble rhs block from master/slave rhs to patch-node block for field (row)
      void reassemble_rhs_block(const int row_block,   ///< row block
          Core::LinAlg::SerialDenseVector& rhs_block,  ///< rhs block
          Core::LinAlg::Matrix<numdofpernode_ * piel, 1>&
              elevector_m,  ///< element vector master block
          Core::LinAlg::Matrix<numdofpernode_ * niel, 1>&
              elevector_s,  ///< element vector slave block
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch  ///< local map between slave nodes and nodes in patch
      );

      //! fill element vectors with extracted data
      void get_element_data(FluidIntFace* surfele,         ///< surface FluidIntFace element
          Fluid* master_ele,                               ///< master parent element
          Fluid* slave_ele,                                ///< slave  parent element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          std::vector<double>& mypvelaf,                   ///< master velaf
          std::vector<double>& mypvelnp,                   ///< master velnp
          std::vector<double>& mypedispnp,                 ///< master dispnp
          std::vector<double>& mypgridv,                   ///< master grid velocity (ALE)
          std::vector<double>& myedispnp,                  ///< surfele dispnp
          std::vector<double>& mynvelaf,                   ///< slave velaf
          std::vector<double>& mynvelnp,                   ///< slave velnp
          std::vector<double>& mynedispnp,                 ///< slave dispnp
          std::vector<double>& myngridv                    ///< slave grid velocity (ALE)
      );

      //! evaluate shape functions and derivatives at integr. point
      double eval_shape_func_and_derivs_at_int_point(const double wquad,  ///< Gaussian weight
          const Core::LinAlg::Matrix<facensd_, 1>&
              xi_gp,  ///< local coordinates of gaussian point w.r.t the master's face
          const Core::LinAlg::Matrix<nsd_, 1>&
              p_xi_gp,  ///< local coordinates of gaussian point w.r.t master element
          const Core::LinAlg::Matrix<nsd_, 1>&
              n_xi_gp,              ///< local coordinates of gaussian point w.r.t slave element
          int master_eid,           ///< master parent element
          int slave_eid,            ///< slave parent element
          bool use2ndderiv = false  ///< flag to use 2nd order derivatives
      );

      //! evaluate shape functions and derivatives at integr. point
      double eval_shape_func_and_derivs_at_int_point(
          Core::FE::GaussIntegration::iterator& iquad,  ///< actual integration point
          int master_eid,                               ///< master parent element
          int slave_eid,                                ///< slave parent element
          bool use2ndderiv = false                      ///< flag to use 2nd order derivatives
      );

      //! evaluate velocity and pressure and derivatives at integr. point
      void eval_vel_pres_and_derivs_at_int_point(
          bool use2ndderiv,  ///< flag to use 2nd order derivatives
          bool isAle         ///< flag, whether we are on an ALE-fluid
      );

      //! set the convective velocity
      void set_convective_velint(
          Discret::Elements::FluidEleParameterIntFace& fldintfacepara, const bool isale);

      //! Provide pressure and viscous u (EOS) ghost penalty stabilization for full! 2nd order
      //! derivatives
      void ghost_penalty2nd_full(
          const double& tau_timefacfac_u_2nd, const double& tau_timefacfac_p_2nd);

      //! Provide pressure and viscous u (EOS) ghost penalty stabilization for 2nd order normal
      //! derivatives
      void ghost_penalty2nd_normal(
          const double& tau_timefacfac_u_2nd, const double& tau_timefacfac_p_2nd);

      //! Provide pressure (EOS) stabilization assembly for fluid
      void pressure_eos(
          const double& tau_timefacfacpre,  ///< tau * (time factor pressure) x (integration factor)
          const double& tau_timefacfacrhs   ///< tau * (time factor rhs)      x (integration factor)
      );

      //! Provide divergence and streamline (EOS) stabilization assembly for fluid
      void div_streamline_eos(const Core::LinAlg::Matrix<nsd_, nsd_>&
              vderxyaf_diff_scaled  ///< difference of velocity gradients * tau x (time factor rhs)
                                    ///< x (integration factor)
      );

      //! Provide divergence (EOS) stabilization assembly for fluid
      void div_eos(
          const double& tau_timefacfac_div,  ///< tau_div * (time factor div) x (integration factor)
          const double& tau_timefacfac_rhs   ///< tau_div * (time factor rhs) x (integration factor)
      );

      //! Provide special condition to fix uncoupled pressure layers for pseudo 2D examples in
      //! combination with Krylov projection
      void pressure_krylov_2dz(
          const double& tau_timefacfacpre,  ///< tau * (time factor pressure) x (integration factor)
          const double& tau_timefacfacrhs   ///< tau * (time factor rhs)      x (integration factor)
      );

      //! compute h_k w.r.t master and slave element
      void compute_patch_hk(Fluid* master,           ///< master fluid element
          Fluid* slave,                              ///< slave fluid element
          Discret::Elements::FluidIntFace* intface,  ///< intface element
          const Inpar::FLUID::EosElementLength&
              eos_element_length  ///< which definition of element length?
      );

      //! compute h_k based on the largest diameter of the element's faces(3D), lines(2D) element
      double compute_patch_hk_surf_with_max_diameter(Fluid* master,  ///< master fluid element
          Fluid* slave,                                              ///< slave fluid element
          Discret::Elements::FluidIntFace* intface                   ///< intface element
      );

      //! compute h_e based on the diameter of the intfaace surface(3D) and the length of the
      //! intface line(2D)
      double compute_surf_diameter(Discret::Elements::FluidIntFace* intface  ///< intface element
      );

      //! compute h_k based on distance of quadrilateral element to opposite surface/edge - just for
      //! quadrilateral/hexahedral elements
      double compute_patch_hk_dist_to_opp_surf(Fluid* master,  ///< master fluid element
          Fluid* slave,                                        ///< slave fluid element
          Discret::Elements::FluidIntFace* intface             ///< intface element
      );

      //! compute h_k based on the largest diameter of the element's faces(3D), lines(2D) element,
      //! however do not take into account the face itself (and its opposite face/line) for hex/quad
      //! elements)
      double compute_patch_hk_diameter_to_opp_surf(Fluid* master,  ///< master fluid element
          Fluid* slave,                                            ///< slave fluid element
          Discret::Elements::FluidIntFace* intface                 ///< intface element
      );

      //! compute h_k based on the maximal diameter of the master and slave element
      double compute_patch_hk_ele_diameter(Fluid* master,  ///< master fluid element
          Fluid* slave                                     ///< slave fluid element
      );

      //! compute the minimum of viscous/convective scaling factors depending on polynomial degree
      inline void determine_poly_degree(double& r_min_visc, double& r_min_conv)
      {
        double r_visc_master = 0.0;
        double r_conv_master = 0.0;

        double r_visc_slave = 0.0;
        double r_conv_slave = 0.0;

        // get the polynomial scaling factors for the viscous and the convective regime for master
        // and slave parent element
        r_pow_alpha(pdistype, r_visc_master, r_conv_master);
        r_pow_alpha(ndistype, r_visc_slave, r_conv_slave);

        // get the minimal degree as the stabilization parameters scale with 1/(r^alpha), not to
        // loose stability
        r_min_visc = std::min(r_visc_master, r_visc_slave);
        r_min_conv = std::min(r_conv_master, r_conv_slave);
      }

      /*!
        \brief provide parameter r_alpha depending on polynomial degree and shape of element
        for each parent distype required for certain stabilization parameter definitions
       */
      inline void r_pow_alpha(Core::FE::CellType parent_distype, double& r_visc, double& r_conv)
      {
        // the scaling factor for face/edge-oriented stabilizations is r^alpha with
        // *    r_triangle      = highest polynomial degree on triangle/tet
        // *    r_quadrilateral = highest polynomial degree in each direction
        // and
        // *    alpha = 4   for viscous terms
        // *    alpha = 7/2 for convective terms
        // independent of dimension,
        // see Braack et al 2007 and hp-paper by Erik Burman 2006

        switch (parent_distype)
        {
          case Core::FE::CellType::hex8:
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::pyramid5:
          case Core::FE::CellType::wedge6:
          case Core::FE::CellType::quad4:
          case Core::FE::CellType::tri3:
          case Core::FE::CellType::line2:
          {
            r_visc = 1.0;  ///< 1.0^4
            r_conv = 1.0;  ///< 1.0^(7/2)
            break;
          }
          case Core::FE::CellType::hex20:
          case Core::FE::CellType::hex27:
          case Core::FE::CellType::tet10:
          case Core::FE::CellType::wedge15:
          case Core::FE::CellType::quad8:
          case Core::FE::CellType::quad9:
          case Core::FE::CellType::tri6:
          case Core::FE::CellType::line3:
          {
            r_visc = 16.0;                   ///< 2.0^4
            r_conv = 11.313708498984761164;  ///< 2.0^(7/2)
            break;
          }
          default:
            FOUR_C_THROW("Element shape not supported.");
            break;
        }
      };

      //! compute volumetric element diameter w.r.t to parent master element or parent slave element
      //! (flag master=true/false)
      template <int numnode>
      void diameter(bool master,  ///< master or slave parent element
          double& h_k             ///< diameter of parent element
      )
      {
        int numnodes_to_check = -1;

        if (master)
        {
          if (nsd_ == 3)
          {
            if (piel == 8 or piel == 20 or piel == 27)
              numnodes_to_check = 8;  // 3D hexahedral elements
            else if (piel == 4 or piel == 10)
              numnodes_to_check = 4;  // 3D tetrahedral elements
            else
              FOUR_C_THROW("unknown element type");
          }
          else if (nsd_ == 2)
          {
            if (piel == 4 or piel == 8 or piel == 9)
              numnodes_to_check = 4;  // 2D quadrilateral elements
            else if (piel == 3 or piel == 6)
              numnodes_to_check = 3;  // 2D triangle elements
            else
              FOUR_C_THROW("unknown element type");
          }
          else
            FOUR_C_THROW("invalid nsd");

          for (int i = 0; i < numnodes_to_check; i++)
          {
            for (int j = i + 1; j < numnodes_to_check; j++)
            {
              // squared length
              double length = 0.0;

              for (int k = 0; k < nsd_; k++)
              {
                length += (pxyze_(k, j) - pxyze_(k, i)) * (pxyze_(k, j) - pxyze_(k, i));
              }
              length = sqrt(length);
              h_k = std::max(length, h_k);
            }
          }
        }
        else
        {
          if (nsd_ == 3)
          {
            if (niel == 8 or niel == 20 or niel == 27)
              numnodes_to_check = 8;  // 3D hexahedral elements
            else if (niel == 4 or niel == 10)
              numnodes_to_check = 4;  // 3D tetrahedral elements
            else
              FOUR_C_THROW("unknown element type");
          }
          else if (nsd_ == 2)
          {
            if (niel == 4 or niel == 8 or niel == 9)
              numnodes_to_check = 4;  // 2D quadrilateral elements
            else if (niel == 3 or niel == 6)
              numnodes_to_check = 3;  // 2D triangle elements
            else
              FOUR_C_THROW("unknown element type");
          }
          else
            FOUR_C_THROW("invalid nsd");


          for (int i = 0; i < numnodes_to_check; i++)
          {
            for (int j = i + 1; j < numnodes_to_check; j++)
            {
              // squared length
              double length = 0.0;

              for (int k = 0; k < nsd_; k++)
              {
                length += (nxyze_(k, j) - nxyze_(k, i)) * (nxyze_(k, j) - nxyze_(k, i));
              }
              length = sqrt(length);
              h_k = std::max(length, h_k);
            }
          }
        }

        if (h_k <= 0.0) FOUR_C_THROW("negative or zero diameter for current element!");

        return;
      };

      //! compute surface diameter w.r.t to parent master element or parent slave element (flag
      //! master=true/false)
      template <int numnode>
      void diameter_2d(bool master,        ///< master or slave parent element
          std::vector<int>& connectivity,  ///< connectivity vector for parent element
          double& h_e                      ///< element length w.r.t parent element
      )
      {
        static Core::LinAlg::Matrix<nsd_, numnode> xyz_surf(true);
        xyz_surf.clear();

        if (connectivity.size() != numnode)
          FOUR_C_THROW("wrong number of nodes for parent's surface");

        if (master == true)  // is parent element the master element?
        {
          // extract node coords
          for (int i = 0; i < (int)numnode; ++i)
          {
            int col = connectivity[i];

            for (int isd = 0; isd < nsd_; isd++) xyz_surf(isd, i) = pxyze_(isd, col);
          }
        }
        else  // parent element is the slave element
        {
          // extract node coords
          for (int i = 0; i < (int)numnode; ++i)
          {
            int col = connectivity[i];

            for (int isd = 0; isd < nsd_; isd++) xyz_surf(isd, i) = nxyze_(isd, col);
          }
        }


        if (numnode == 4 or numnode == 8 or
            numnode == 9)  // quad4 surface and take the corner points also for quad8 surfaces
        {
          double diam1 = 0.0;
          double diam2 = 0.0;
          double line0 = 0.0;
          double line1 = 0.0;
          double line2 = 0.0;
          double line3 = 0.0;

          for (int i = 0; i < nsd_; i++)
          {
            // diagonals
            // line nodes (0,2)
            diam1 += (xyz_surf(i, 2) - xyz_surf(i, 0)) * (xyz_surf(i, 2) - xyz_surf(i, 0));

            // line nodes (1,3)
            diam2 += (xyz_surf(i, 3) - xyz_surf(i, 1)) * (xyz_surf(i, 3) - xyz_surf(i, 1));

            // lines
            line0 += (xyz_surf(i, 1) - xyz_surf(i, 0)) * (xyz_surf(i, 1) - xyz_surf(i, 0));
            line1 += (xyz_surf(i, 2) - xyz_surf(i, 1)) * (xyz_surf(i, 2) - xyz_surf(i, 1));
            line2 += (xyz_surf(i, 3) - xyz_surf(i, 2)) * (xyz_surf(i, 3) - xyz_surf(i, 2));
            line3 += (xyz_surf(i, 0) - xyz_surf(i, 3)) * (xyz_surf(i, 0) - xyz_surf(i, 3));
          }
          diam1 = sqrt(diam1);
          diam2 = sqrt(diam2);
          line0 = sqrt(line0);
          line1 = sqrt(line1);
          line2 = sqrt(line2);
          line3 = sqrt(line3);


          h_e = std::max(
              diam1, std::max(diam2, std::max(line0, std::max(line1, std::max(line2, line3)))));
        }
        else if (numnode == 3 or numnode == 6)  // tri3 or tri6 surface
        {
          double line0 = 0.0;
          double line1 = 0.0;
          double line2 = 0.0;

          for (int i = 0; i < nsd_; i++)
          {
            // no diagonals

            // lines
            line0 += (xyz_surf(i, 1) - xyz_surf(i, 0)) * (xyz_surf(i, 1) - xyz_surf(i, 0));
            line1 += (xyz_surf(i, 2) - xyz_surf(i, 1)) * (xyz_surf(i, 2) - xyz_surf(i, 1));
            line2 += (xyz_surf(i, 0) - xyz_surf(i, 2)) * (xyz_surf(i, 0) - xyz_surf(i, 2));
          }

          line0 = sqrt(line0);
          line1 = sqrt(line1);
          line2 = sqrt(line2);


          h_e = std::max(line0, std::max(line1, line2));
        }
        else
          FOUR_C_THROW("unknown number of nodes");

        if (h_e <= 0.0) FOUR_C_THROW("negative or zero diameter for current face!");

        return;
      }

      //! compute line diameter (length) w.r.t to parent master element or parent slave element
      //! (flag master=true/false)
      template <int numnode>
      void diameter_1d(bool master,        ///< master or slave parent element
          std::vector<int>& connectivity,  ///< connectivity vector for parent element
          double& h_e                      ///< element length w.r.t parent element
      )
      {
        static Core::LinAlg::Matrix<nsd_, numnode> xyz_surf(true);
        xyz_surf.clear();

        if (connectivity.size() != numnode)
          FOUR_C_THROW("wrong number of nodes for parent's surface");

        if (master == true)  // is parent element the master element?
        {
          // extract node coords
          for (int i = 0; i < (int)numnode; ++i)
          {
            int col = connectivity[i];

            for (int isd = 0; isd < nsd_; isd++) xyz_surf(isd, i) = pxyze_(isd, col);
          }
        }
        else  // parent element is the slave element
        {
          // extract node coords
          for (int i = 0; i < (int)numnode; ++i)
          {
            int col = connectivity[i];

            for (int isd = 0; isd < nsd_; isd++) xyz_surf(isd, i) = nxyze_(isd, col);
          }
        }


        if (numnode == 2 or numnode == 3)  // line 2 or line 3 face
        {
          double diam1 = 0.0;

          for (int i = 0; i < nsd_; i++)
          {
            // diagonals
            // line nodes (0,1) -> (0,1)
            diam1 += (xyz_surf(i, 1) - xyz_surf(i, 0)) * (xyz_surf(i, 1) - xyz_surf(i, 0));
          }
          diam1 = sqrt(diam1);
          h_e = diam1;
        }
        else
          FOUR_C_THROW("unknown number of nodes");

        if (h_e <= 0.0) FOUR_C_THROW("negative or zero diameter for current face!");

        return;
      }

      template <int numlines>
      void max_length_of_lines(bool master,  ///< master or slave parent element
          std::map<int, std::vector<int>>
              p_lines_nodes,  ///< for each line the vector of start and end nodes
          double& h_e         ///< element length w.r.t parent element
      )
      {
        // matrix that contains the coordinates of the start and end point of the connecting lines
        // between the two opposite surfaces
        static Core::LinAlg::Matrix<nsd_, numlines * 2> xyze_distance_lines;
        xyze_distance_lines.clear();

        if (master == true)  // is parent element the master element?
        {
          // find the distance
          int count = 0;
          for (std::map<int, std::vector<int>>::iterator iter = p_lines_nodes.begin();
              iter != p_lines_nodes.end(); iter++)
          {
            std::vector<int>& nodesofline = iter->second;
            for (int isd = 0; isd < nsd_; isd++)
            {
              // extract just start and end point of the line
              xyze_distance_lines(isd, count) = pxyze_(isd, nodesofline.at(0));
              xyze_distance_lines(isd, count + 1) = pxyze_(isd, nodesofline.at(1));
            }
            count = count + 2;
          }
        }
        else
        {
          // find the distance
          int count = 0;
          for (std::map<int, std::vector<int>>::iterator iter = p_lines_nodes.begin();
              iter != p_lines_nodes.end(); iter++)
          {
            std::vector<int>& nodesofline = iter->second;
            for (int isd = 0; isd < nsd_; isd++)
            {
              // extract just start and end point of the line
              xyze_distance_lines(isd, count) = nxyze_(isd, nodesofline.at(0));
              xyze_distance_lines(isd, count + 1) = nxyze_(isd, nodesofline.at(1));
            }
            count = count + 2;
          }
        }

        // just reasonable for quadrilateral intfaces
        if (numlines == 4)  // for hex8/20/27 element with quadrilateral intface element
        {
          double line0 = 0.0;
          double line1 = 0.0;
          double line2 = 0.0;
          double line3 = 0.0;

          for (int i = 0; i < nsd_; i++)
          {
            line0 += (xyze_distance_lines(i, 0) - xyze_distance_lines(i, 1)) *
                     (xyze_distance_lines(i, 0) - xyze_distance_lines(i, 1));
            line1 += (xyze_distance_lines(i, 2) - xyze_distance_lines(i, 3)) *
                     (xyze_distance_lines(i, 2) - xyze_distance_lines(i, 3));
            line2 += (xyze_distance_lines(i, 4) - xyze_distance_lines(i, 5)) *
                     (xyze_distance_lines(i, 4) - xyze_distance_lines(i, 5));
            line3 += (xyze_distance_lines(i, 6) - xyze_distance_lines(i, 7)) *
                     (xyze_distance_lines(i, 6) - xyze_distance_lines(i, 7));
          }

          line0 = sqrt(line0);
          line1 = sqrt(line1);
          line2 = sqrt(line2);
          line3 = sqrt(line3);

          h_e = std::max(line0, std::max(line1, std::max(line2, line3)));
        }
        else if (numlines == 2)  // quad4/8/9 parent element with line intface element
        {
          double line0 = 0.0;
          double line1 = 0.0;

          for (int i = 0; i < nsd_; i++)
          {
            line0 += (xyze_distance_lines(i, 0) - xyze_distance_lines(i, 1)) *
                     (xyze_distance_lines(i, 0) - xyze_distance_lines(i, 1));
            line1 += (xyze_distance_lines(i, 2) - xyze_distance_lines(i, 3)) *
                     (xyze_distance_lines(i, 2) - xyze_distance_lines(i, 3));
          }

          line0 = sqrt(line0);
          line1 = sqrt(line1);

          h_e = std::max(line0, line1);
        }
        else
          FOUR_C_THROW("call not reasonable for non-quadrilateral or line intface elements");
      }



      //! find the surface at the opposite side, return the local side id of opposite side
      //! return -1 if not unique!
      int find_opposite_surface(Core::FE::CellType ele_distype, const int surfacelocalid)
      {
        if (ele_distype == Core::FE::CellType::hex8 or ele_distype == Core::FE::CellType::hex20 or
            ele_distype == Core::FE::CellType::hex27)
        {
          switch (surfacelocalid)
          {
            case 0:
              return 5;
              break;
            case 1:
              return 3;
              break;
            case 2:
              return 4;
              break;
            case 3:
              return 1;
              break;
            case 4:
              return 2;
              break;
            case 5:
              return 0;
              break;
            default:
              FOUR_C_THROW("wrong number!!");
              break;
          }
        }
        else if (ele_distype == Core::FE::CellType::quad4 or
                 ele_distype == Core::FE::CellType::quad8 or
                 ele_distype == Core::FE::CellType::quad9)
        {
          switch (surfacelocalid)
          {
            case 0:
              return 2;
              break;
            case 1:
              return 3;
              break;
            case 2:
              return 0;
              break;
            case 3:
              return 1;
              break;
            default:
              FOUR_C_THROW("wrong number!!");
              break;
          }
        }
        if (ele_distype == Core::FE::CellType::tet4 or ele_distype == Core::FE::CellType::tet10 or
            ele_distype == Core::FE::CellType::tri3 or ele_distype == Core::FE::CellType::tri6 or
            ele_distype == Core::FE::CellType::pyramid5)
        {
          // no unique opposite side
          return -1;
        }
        else
          FOUR_C_THROW("opposite sides not implemented yet for wedge elements yet!");

        return -1;
      }


      //! find the lines which connect this surface to the surface at the other side
      void find_hex_connecting_lines_2d(const int numnode,
          std::vector<std::vector<int>> connectivity_line_surf, const int side_id_master,
          const int side_id_slave,
          std::set<int>& p_lines_m,  //< to be filled
          std::set<int>& p_lines_s,  //< to be filled
          const int opposite_side_id_master, const int opposite_side_id_slave)
      {
        if (numnode == 4 or numnode == 8 or numnode == 9)
        {
          for (size_t i_line = 0; i_line < connectivity_line_surf.size(); i_line++)
          {
            // find all adjacent surfaces to this line
            std::vector<int> line_surfs = connectivity_line_surf.at(i_line);

            int countsurf_m = 0;
            int countsurf_s = 0;
            for (size_t i_surf = 0; i_surf < line_surfs.size(); i_surf++)
            {
              if ((line_surfs.at(i_surf) != side_id_master) and
                  (line_surfs.at(i_surf) != opposite_side_id_master))
              {
                countsurf_m++;
                // insert the line, where the two surfaces does not belong to it
                if (countsurf_m == 2) p_lines_m.insert(i_line);
              }
              if ((line_surfs.at(i_surf) != side_id_slave) and
                  (line_surfs.at(i_surf) != opposite_side_id_slave))
              {
                countsurf_s++;
                // insert the line, where the two surfaces does not belong to it
                if (countsurf_s == 2) p_lines_s.insert(i_line);
              }
            }
          }
        }
        else
          FOUR_C_THROW(
              "The implementation of FindHEXConnectingLines is not available for non-hexahedral "
              "elements!");
      }

      //! find the lines which connect this line to the other opposite line
      void find_quad_connecting_lines_1d(const int numnode, const int masterlocalid,
          const int slavelocalid, std::set<int>& p_lines_m, std::set<int>& p_lines_s,
          const int opposite_side_id_master, const int opposite_side_id_slave)
      {
        if (numnode == 2 or numnode == 3)
        {
          if (((masterlocalid == 3) and (opposite_side_id_master == 1)) or
              ((masterlocalid == 1) and (opposite_side_id_master == 3)))
          {
            p_lines_m.insert(0);
            p_lines_m.insert(2);
          }
          else if (((masterlocalid == 0) and (opposite_side_id_master == 2)) or
                   ((masterlocalid == 2) and (opposite_side_id_master == 0)))
          {
            p_lines_m.insert(1);
            p_lines_m.insert(3);
          }

          if (((slavelocalid == 3) and (opposite_side_id_slave == 1)) or
              ((slavelocalid == 1) and (opposite_side_id_slave == 3)))
          {
            p_lines_s.insert(0);
            p_lines_s.insert(2);
          }
          else if (((slavelocalid == 0) and (opposite_side_id_slave == 2)) or
                   ((slavelocalid == 2) and (opposite_side_id_slave == 0)))
          {
            p_lines_s.insert(1);
            p_lines_s.insert(3);
          }
        }
        else
          FOUR_C_THROW(
              "The implementation of FindQUADConnectingLines is not available for non-line2/3 "
              "elements!");
      }

      /*!
       \brief computation of the EOS-stabilization parameter and ghost penalty parameter
      */
      void compute_stabilization_params(const bool is_ghost_penalty_reconstruct,
          const bool use2ndderiv, const Inpar::FLUID::EosTauType tautype,
          const bool EOS_conv_stream, const bool EOS_conv_cross, const bool EOS_div_vel_jump,
          const double max_vel_L2_norm, const double timefac, const double gamma_ghost_penalty_visc,
          const double gamma_ghost_penalty_trans, const double gamma_ghost_penalty_u_2nd,
          const double gamma_ghost_penalty_p_2nd);

     private:
      // element matrices in block structure master vs. slave
      Core::LinAlg::Matrix<numdofpernode_ * piel, numdofpernode_ * piel>
          elematrix_mm_;  ///< element matrix master-master block
      Core::LinAlg::Matrix<numdofpernode_ * piel, numdofpernode_ * niel>
          elematrix_ms_;  ///< element matrix master-slave block
      Core::LinAlg::Matrix<numdofpernode_ * niel, numdofpernode_ * piel>
          elematrix_sm_;  ///< element matrix slave-master block
      Core::LinAlg::Matrix<numdofpernode_ * niel, numdofpernode_ * niel>
          elematrix_ss_;  ///< element matrix slave-slave block

      Core::LinAlg::Matrix<numdofpernode_ * piel, 1> elevector_m_;  ///< element vector master block
      Core::LinAlg::Matrix<numdofpernode_ * niel, 1> elevector_s_;  ///< element vector slave block

      // nodal arrays
      // ------------
      //! node coordinates of parent (master) element
      Core::LinAlg::Matrix<nsd_, piel> pxyze_;
      //! array of nodal velocities, intermediate time level
      Core::LinAlg::Matrix<nsd_, piel> pevelaf_;
      //! array of nodal grid velocities, time n+1
      Core::LinAlg::Matrix<nsd_, piel> pegridv_;
      //! array of nodal velocities, new time level
      Core::LinAlg::Matrix<nsd_, piel> pevelnp_;
      //! array of nodal advective velocities
      Core::LinAlg::Matrix<nsd_, piel> peconvvelaf_;
      //! array of nodal pressure, new time level
      Core::LinAlg::Matrix<piel, 1> peprenp_;
      //! array of nodal grid displacements, parent element, new time level
      Core::LinAlg::Matrix<nsd_, piel> pedispnp_;
      //! array of nodal grid displacements, surface element, new time level
      Core::LinAlg::Matrix<nsd_, piel> edispnp_;

      //! node coordinates of neighbor (slave) element
      Core::LinAlg::Matrix<nsd_, niel> nxyze_;
      //! array of nodal velocities, intermediate time level
      Core::LinAlg::Matrix<nsd_, niel> nevelaf_;
      //! array of nodal grid velocities, time n+1
      Core::LinAlg::Matrix<nsd_, niel> negridv_;
      //! array of nodal velocities, new time level
      Core::LinAlg::Matrix<nsd_, niel> nevelnp_;
      //! array of nodal advective velocities
      Core::LinAlg::Matrix<nsd_, niel> neconvvelaf_;
      //! array of nodal pressure, new time level
      Core::LinAlg::Matrix<niel, 1> neprenp_;
      //! array of nodal grid displacements, neighbor element, new time level
      Core::LinAlg::Matrix<nsd_, niel> nedispnp_;

      //! node coordinates of boundary element
      Core::LinAlg::Matrix<nsd_, iel> xyze_;

      //! linearisation of convection, convective part for parent element
      Core::LinAlg::Matrix<piel, 1> p_conv_c_;
      //! linearisation of convection, convective part for neighbor element
      Core::LinAlg::Matrix<niel, 1> n_conv_c_;


      // shape functions and derivatives, mapping from reference element to actual geometry
      // ----------------------------------------------------------------------------------
      //! transpose of the jacobian matrix of the mapping (r,s,t)->(x,y,z)
      Core::LinAlg::Matrix<nsd_, nsd_> pxjm_;
      //! its inverse
      Core::LinAlg::Matrix<nsd_, nsd_> pxji_;
      //! vector of shape functions, parent element
      Core::LinAlg::Matrix<piel, 1> pfunct_;
      //! vector of shape function derivatives in reference coordinate system, parent element
      Core::LinAlg::Matrix<nsd_, piel> pderiv_;
      //! vector of shape function derivatives in global coordinate system
      Core::LinAlg::Matrix<nsd_, piel> pderxy_;
      //! vector of shape function (2nd) derivatives in reference coordinate system, parent element
      Core::LinAlg::Matrix<numderiv2_p, piel> pderiv2_;
      //! vector of shape function (2nd) derivatives in global coordinate system
      Core::LinAlg::Matrix<numderiv2_p, piel> pderxy2_;

      // ----------------------------------------------------------------------------------
      //! transpose of the jacobian matrix of the mapping (r,s,t)->(x,y,z)
      Core::LinAlg::Matrix<nsd_, nsd_> nxjm_;
      //! its inverse
      Core::LinAlg::Matrix<nsd_, nsd_> nxji_;
      //! vector of shape functions, parent element
      Core::LinAlg::Matrix<niel, 1> nfunct_;
      //! vector of shape function derivatives in reference coordinate system, parent element
      Core::LinAlg::Matrix<nsd_, niel> nderiv_;
      //! vector of shape function derivatives in global coordinate system
      Core::LinAlg::Matrix<nsd_, niel> nderxy_;
      //! vector of shape function (2nd) derivatives in reference coordinate system, parent element
      Core::LinAlg::Matrix<numderiv2_n, niel> nderiv2_;
      //! vector of shape function (2nd) derivatives in global coordinate system
      Core::LinAlg::Matrix<numderiv2_n, niel> nderxy2_;


      //! vector of shape functions, boundary element
      Core::LinAlg::Matrix<iel, 1> funct_;
      //! vector of shape function derivatives in reference coordinate system, boundary element
      Core::LinAlg::Matrix<facensd_, iel> deriv_;
      //! normal vector on surface element (outward pointing from parent element)
      Core::LinAlg::Matrix<nsd_, 1> n_;
      //! derivatives of surface in all reference directions
      Core::LinAlg::Matrix<facensd_, nsd_> dxyzdrs_;
      //! the metric tensor
      Core::LinAlg::Matrix<facensd_, facensd_> metrictensor_;
      //! the area of an infinitesimal surface element
      double drs_;



      // values of non geometrical quantities in gausspoints
      // ---------------------------------------------------

      //! local coordinates w.r.t face
      Core::LinAlg::Matrix<facensd_, 1> xsi_;
      //! velocity in gausspoint, time n+af
      Core::LinAlg::Matrix<nsd_, 1> velintaf_;
      //! velocity in gausspoint, time n+1
      Core::LinAlg::Matrix<nsd_, 1> velintnp_;
      //! grid velocity in gausspoint, time n+1 (master side, for ALE-fluid)
      Core::LinAlg::Matrix<nsd_, 1> gridvelint_;
      //! convective velocity in gausspoint, time n+1 (master side, for ALE-fluid)
      Core::LinAlg::Matrix<nsd_, 1> convvelint_;
      //! parent velocity derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<nsd_, nsd_> pvderxyaf_;
      //! neighbor velocity derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<nsd_, nsd_> nvderxyaf_;
      //! parent velocity derivatives in gausspoint, time n+1
      Core::LinAlg::Matrix<nsd_, nsd_> pvderxynp_;
      //! neighbor velocity derivatives in gausspoint, time n+1
      Core::LinAlg::Matrix<nsd_, nsd_> nvderxynp_;
      //! parent velocity (2nd) derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<nsd_, numderiv2_p> pvderxy2af_;
      //! neighbor velocity (2nd) derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<nsd_, numderiv2_n> nvderxy2af_;

      //! parent velocity (2nd) derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<1, numderiv2_p> ppderxy2af_;
      //! neighbor velocity (2nd) derivatives in gausspoint, time n+af
      Core::LinAlg::Matrix<1, numderiv2_n> npderxy2af_;

      //! velocity gradient difference, time n+af
      Core::LinAlg::Matrix<nsd_, nsd_> vderxyaf_diff_;
      //! velocity gradient difference, time n+1
      Core::LinAlg::Matrix<nsd_, nsd_> vderxynp_diff_;


      //! pressure in gausspoint, time n+1
      double prenp_;
      //! parent pressure derivatives in gausspoint, time n+1
      Core::LinAlg::Matrix<nsd_, 1> pprederxy_;
      //! neighbor velocity derivatives in gausspoint, time n+1
      Core::LinAlg::Matrix<nsd_, 1> nprederxy_;

      Core::LinAlg::Matrix<piel, piel> pderiv_dyad_pderiv_;
      Core::LinAlg::Matrix<piel, piel> pderiv_dyad_pderiv_tau_timefacfac_;
      Core::LinAlg::Matrix<piel, piel> pderiv_dyad_pderiv_tau_timefacfacpre_;

      Core::LinAlg::Matrix<piel, niel> pderiv_dyad_nderiv_;
      Core::LinAlg::Matrix<piel, niel> pderiv_dyad_nderiv_tau_timefacfac_;
      Core::LinAlg::Matrix<piel, niel> pderiv_dyad_nderiv_tau_timefacfacpre_;

      Core::LinAlg::Matrix<niel, niel> nderiv_dyad_nderiv_;
      Core::LinAlg::Matrix<niel, niel> nderiv_dyad_nderiv_tau_timefacfac_;
      Core::LinAlg::Matrix<niel, niel> nderiv_dyad_nderiv_tau_timefacfacpre_;

      //! vector of shape function derivatives in global coordinate system scaled with
      //! tau_timefacfac
      Core::LinAlg::Matrix<nsd_, piel> pderxy_tau_timefacfac_;
      //! vector of shape function derivatives in global coordinate system scaled with
      //! tau_timefacfac
      Core::LinAlg::Matrix<nsd_, niel> nderxy_tau_timefacfac_;


      //! face/edge integration points
      std::shared_ptr<Core::FE::GaussPoints> intpoints_;

      //! number of Gaussian points
      unsigned int numgp_;

      Core::LinAlg::SerialDenseMatrix p_xi_points_;
      Core::LinAlg::SerialDenseMatrix n_xi_points_;
      Core::LinAlg::SerialDenseMatrix face_xi_points_master_;
      Core::LinAlg::SerialDenseMatrix face_xi_points_slave_;


      Core::LinAlg::Matrix<facensd_, 1> face_xi_gp_;
      Core::LinAlg::Matrix<nsd_, 1> p_xi_gp_;
      Core::LinAlg::Matrix<nsd_, 1> n_xi_gp_;

      // element, side, line connectivity
      // ---------------------------------------------------
      //! numbering of master's surfaces/lines w.r.t parent element
      std::vector<std::vector<int>> m_connectivity_;

      // numbering of slave's surfaces/lines w.r.t parent element
      std::vector<std::vector<int>> s_connectivity_;

      //! numbering of master's lines w.r.t parent element
      std::vector<std::vector<int>> connectivity_line_surf_;

      //! numbering of master's lines and nodes
      std::vector<std::vector<int>> connectivity_line_nodes_;


      // material parameters
      // ---------------------------------------------------
      //! kinematic viscosity
      double kinvisc_;

      //! density
      double density_;

      //! reaction coefficient!
      double reacoeff_;

      //! longest edge in parent element
      double p_hk_;

      //! longest edge in parent element squared
      double p_hk_squared_;

      //! longest edge in parent element cubed
      double p_hk_cubed_;

      // effective stabilization parameters
      // ---------------------------------------------------
      double tau_u_;    ///< stabilization factor for component-wise velocity gradient jump
                        ///< stabilization terms
      double tau_div_;  ///< stabilization factor for coupled divergence jump stabilization term
      double tau_p_;    ///< stabilization factor for pressure gradient jump stabilization term

      double tau_u_gp_1_visc_reaction_;  ///< 1st order viscous and reactive stabilization factor
                                         ///< for ghost penalty stabilization factor for velocity
                                         ///< gradient jump ghost-penalty term
      double tau_u_gp_1_;  ///< 1st order stabilization factor for component-wise velocity gradient
                           ///< jump higher-order ghost penalty stabilization terms
      double tau_div_gp_1_;  ///< 1st order stabilization factor for coupled divergence jump
                             ///< higher-order ghost penalty stabilization term
      double tau_p_gp_1_;    ///< 1st order stabilization factor for pressure gradient jump
                             ///< higher-order ghost penalty stabilization term

      double tau_u_gp_2_visc_reaction_;  ///< 2nd order viscous and reactive stabilization factor
                                         ///< for ghost penalty stabilization factor for velocity
                                         ///< gradient jump ghost-penalty term
      double tau_u_gp_2_;  ///< 2nd order stabilization factor for component-wise velocity gradient
                           ///< jump higher-order ghost penalty stabilization terms
      double tau_div_gp_2_;  ///< 2nd order stabilization factor for coupled divergence jump
                             ///< higher-order ghost penalty stabilization term
      double tau_p_gp_2_;    ///< 2nd order stabilization factor for pressure gradient jump
                             ///< higher-order ghost penalty stabilization term

      double tau_vel_1st_final_;  ///< final scaling for 1st order velocity chips and gps
      double tau_pre_1st_final_;  ///< final scaling for 1st order pressure chips and gps
      double tau_div_1st_final_;  ///< final scaling for 1st order divergence chips and gps

      double tau_vel_2nd_final_;  ///< final scaling for 2nd order velocity gps
      double tau_pre_2nd_final_;  ///< final scaling for 2nd order pressure gps

      bool ishigherorder_;  ///< is the face a higher order face with higher order neighboring
                            ///< elements?
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
