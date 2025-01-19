// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_COUPLING_MESH_COUPLED_LEVELSET_HPP
#define FOUR_C_XFEM_COUPLING_MESH_COUPLED_LEVELSET_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_xfem_coupling_base.hpp"
#include "4C_xfem_coupling_levelset.hpp"
#include "4C_xfem_coupling_mesh.hpp"
#include "4C_xfem_interface_utils.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  /*!
  \brief
   */
  class MeshCouplingNavierSlipTwoPhase : public MeshCouplingNavierSlip
  {
   public:
    //! constructor
    explicit MeshCouplingNavierSlipTwoPhase(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,           ///< discretization from which cutter discretization can be derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step,         ///< time step
        bool marked_geometry = false  ///< is this a marked geometry mesh boundary
    );


    /*!
    Return prescribed velocities and traction vectors for a GNBC boundary condition.
    Also returns the projection matrix (to the plane of the surface) needed for the GNBC condition.
    */
    template <Core::FE::CellType distype, class V1, class V2, class X1, class T1, class M1,
        class M2, class M3>
    void evaluate_coupling_conditions(V1& ivel,   ///< prescribed velocity at interface
        V2& itraction,                            ///< prescribed traction at interface
        X1& x,                                    ///< coordinates of gauss point
        const Core::Conditions::Condition* cond,  ///< condition prescribed to this surface
        T1& proj_matrix,  ///< Laplace-Beltrami matrix for surface tension calculations
        int eid,          ///< element ID
        M1& funct,        ///< local shape function for Gauss Point (from fluid element)
        M2& derxy,   ///< local derivatives of shape function for Gauss Point (from fluid element)
        M3& normal,  ///< surface normal of cut element
        const bool& eval_dirich_at_gp,
        double& kappa_m,  ///< fluid sided weighting
        double& visc_m,   ///< fluid sided weighting
        double& visc_s    ///< slave sided dynamic viscosity
    )
    {
      setup_projection_matrix(proj_matrix, normal);

      // help variable
      int robin_id;

      if (eval_dirich_at_gp)
      {
        // evaluate interface velocity (given by weak Dirichlet condition)
        const auto maybe_id = cond->parameters().get<Core::IO::Noneable<int>>("ROBIN_DIRICHLET_ID");
        robin_id = maybe_id.value_or(-1) - 1;
        if (robin_id >= 0)
          evaluate_dirichlet_function(
              ivel, x, conditionsmap_robin_dirch_.find(robin_id)->second, time_);

        // Safety checks
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if ((conditionsmap_robin_dirch_.find(robin_id)) == conditionsmap_robin_dirch_.end())
        {
          FOUR_C_THROW(
              "Key was not found in this instance!! Fatal error! (conditionsmap_robin_dirch_)");
        }
#endif
      }

      // evaluate interface traction (given by Neumann condition)
      const auto maybe_id = cond->parameters().get<Core::IO::Noneable<int>>("ROBIN_NEUMANN_ID");
      robin_id = maybe_id.value_or(-1) - 1;

      if (robin_id >= 0)
      {
        // This is maybe not the most efficient implementation as we evaluate dynvisc as well as the
        // sliplenght twice (also done in update_configuration_map_gp ... as soon as this gets
        // relevant we should merge this functions)

        // evaluate interface traction (given by Neumann condition)
        // Add this to the veljump!

        double sliplength = 0.0;

        if (sliplength < 0.0) FOUR_C_THROW("The slip length can not be negative.");

        if (sliplength != 0.0)
        {
          evaluate_neumann_function(
              itraction, x, conditionsmap_robin_neumann_.find(robin_id)->second, time_);

          double sl_visc_fac = sliplength / (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);
          Core::LinAlg::Matrix<3, 1> tmp_itraction(true);
          tmp_itraction.multiply_tn(proj_matrix, itraction);
          // Project this into tangential direction!!!

          ivel.update(sl_visc_fac, tmp_itraction, 1.0);

          itraction.clear();
        }
      }

      if (force_tangvel_map_.find(cond->id())->second)
      {
        Core::LinAlg::Matrix<3, 1> tmp_ivel(true);
        tmp_ivel.multiply_tn(
            proj_matrix, ivel);  // apply Projection matrix from the right. (u_0 * P^t)
        ivel.update(1.0, tmp_ivel, 0.0);
      }

// Safety checks
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (robin_id >= 0)
      {
        if ((conditionsmap_robin_neumann_.find(robin_id)) == conditionsmap_robin_neumann_.end())
        {
          FOUR_C_THROW(
              "Key was not found in this instance!! Fatal error! (conditionsmap_robin_neumann_)");
        }
      }
      std::map<int, bool>::iterator it_bool;
      if ((it_bool = force_tangvel_map_.find(cond->id())) == force_tangvel_map_.end())
      {
        FOUR_C_THROW("Key was not found in this instance!! Fatal error! (force_tangvel_map_)");
      }
#endif
    };


    // template <Core::FE::CellType DISTYPE>//,class M1, class M2>
    //! Updates configurationmap for specific Gausspoint
    void update_configuration_map_gp(double& kappa_m,  //< fluid sided weighting
        double& visc_m,                                //< master sided dynamic viscosity
        double& visc_s,                                //< slave sided dynamic viscosity
        double& density_m,                             //< master sided density
        double& visc_stab_tang,                        //< viscous tangential NIT Penalty scaling
        double& full_stab,                             //< full NIT Penalty scaling
        const Core::LinAlg::Matrix<3, 1>& x,           //< Position x in global coordinates
        const Core::Conditions::Condition* cond,       //< Condition
        Core::Elements::Element* ele,                  //< Element
        Core::Elements::Element* bele,                 //< Boundary Element
        double* funct,  //< local shape function for Gauss Point (from fluid element)
        double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
        Core::LinAlg::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
        Core::LinAlg::Matrix<3, 1>& normal,     //< normal at gp
        Core::LinAlg::Matrix<3, 1>& vel_m,      //< master velocity at gp
        double* fulltraction  //< precomputed fsi traction (sigmaF n + gamma relvel)
        ) override
    {
      double dynvisc = (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);
      double sliplength = 0.0;

      if (ele->shape() == Core::FE::CellType::hex8)
      {
        const Core::FE::CellType shape = Core::FE::CellType::hex8;
        //
        const size_t nsd = Core::FE::dim<shape>;
        const size_t nen = Core::FE::num_nodes<shape>;
        Core::LinAlg::Matrix<nen, 1> funct_(funct, true);
        Core::LinAlg::Matrix<nen, nsd> derxy_(derxy, true);
      }
      else if (ele->shape() == Core::FE::CellType::hex27)
      {
        const Core::FE::CellType shape = Core::FE::CellType::hex27;
        //
        const size_t nsd = Core::FE::dim<shape>;
        const size_t nen = Core::FE::num_nodes<shape>;
        Core::LinAlg::Matrix<nen, 1> funct_(funct, true);
        Core::LinAlg::Matrix<nen, nsd> derxy_(derxy, true);
      }
      else if (ele->shape() == Core::FE::CellType::hex20)
      {
        const Core::FE::CellType shape = Core::FE::CellType::hex20;
        //
        const size_t nsd = Core::FE::dim<shape>;
        const size_t nen = Core::FE::num_nodes<shape>;
        Core::LinAlg::Matrix<nen, 1> funct_(funct, true);
        Core::LinAlg::Matrix<nen, nsd> derxy_(derxy, true);
      }
      else
        FOUR_C_THROW("Element not considered.");

      if (sliplength < 0.0) FOUR_C_THROW("The slip length can not be negative.");

      if (sliplength != 0.0)
      {
        double stabnit = 0.0;
        double stabadj = 0.0;
        XFEM::Utils::get_navier_slip_stabilization_parameters(
            visc_stab_tang, dynvisc, sliplength, stabnit, stabadj);
        configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = stabnit;
        configuration_map_[Inpar::XFEM::F_Con_t_Row] =
            std::pair<bool, double>(true, -stabnit);  //+sign for penalty!
        configuration_map_[Inpar::XFEM::F_Con_t_Col] =
            std::pair<bool, double>(true, sliplength / dynvisc);
        configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = stabadj;
        configuration_map_[Inpar::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(true, sliplength);
      }
      else
      {
        configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::F_Con_t_Row] = std::pair<bool, double>(false, 0.0);
        configuration_map_[Inpar::XFEM::F_Con_t_Col] = std::pair<bool, double>(false, 0.0);
        configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = 1.0;
        configuration_map_[Inpar::XFEM::FStr_Adj_t_Col] = std::pair<bool, double>(false, 0.0);
      }
    };

    void set_condition_specific_parameters() override;
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
