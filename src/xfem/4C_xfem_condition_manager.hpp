// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_CONDITION_MANAGER_HPP
#define FOUR_C_XFEM_CONDITION_MANAGER_HPP

// Might want to forward declare the levelset and mesh coupling and put these functions into
// the xfem_condition_manager.cpp file in a later stage
#include "4C_config.hpp"

#include "4C_xfem_coupling_fpi_mesh.hpp"
#include "4C_xfem_coupling_levelset.hpp"
#include "4C_xfem_coupling_mesh.hpp"
#include "4C_xfem_coupling_mesh_coupled_levelset.hpp"


FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class VolumeCell;
}


namespace Discret
{
  namespace Elements
  {
    // finally this parameter list should go and all interface relevant parameters should be
    // stored in the condition manager or coupling objects
    class FluidEleParameterXFEM;
  }  // namespace Elements
}  // namespace Discret

namespace XFEM
{
  template <typename Tp>
  inline const Tp& argmin(const Tp& a, const Tp& b, int& arg)
  {
    if (b < a)
    {
      arg = 2;
      return b;
    }
    arg = 1;
    return a;
  }

  template <typename Tp>
  inline const Tp& argmax(const Tp& a, const Tp& b, int& arg)
  {
    if (a < b)
    {
      arg = 2;
      return b;
    }
    arg = 1;
    return a;
  }

  /*!
  \brief Manages the conditions for the xfluid (i.e. levelset/mesh cut and what BC are applied at
  these)
   */
  class ConditionManager
  {
   public:
    //! constructor
    explicit ConditionManager(const std::map<std::string, int>& dofset_coupling_map,  ///< ???
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        std::vector<std::shared_ptr<Core::FE::Discretization>>&
            meshcoupl_dis,  ///< mesh coupling discretizations
        std::vector<std::shared_ptr<Core::FE::Discretization>>&
            levelsetcoupl_dis,  ///< levelset coupling discretizations
        const double time,      ///< time
        const int step          ///< time step
    );


    // TODO: we need set private and public !!! this however causes moving functions in the header
    // file!

    void set_time_and_step(const double time, const int step);

    void get_coupling_ids(const Core::FE::Discretization& cond_dis,
        const std::string& condition_name, std::set<int>& coupling_ids);

    void set_dof_set_coupling_map(const std::map<std::string, int>& dofset_coupling_map);

    void status();

    void increment_time_and_step(const double dt);

    void create_new_level_set_coupling(const std::string& cond_name,
        std::shared_ptr<Core::FE::Discretization>
            cond_dis,  ///< discretization from which the cutter discretization can be derived
        const int coupling_id);

    void create_couplings(std::vector<std::shared_ptr<Core::FE::Discretization>>&
                              coupl_dis,  ///< coupling discretizations
        const std::vector<std::string>&
            conditions_to_check,   ///< conditions for which coupling objects shall be created
        bool create_mesh_coupling  ///< create mesh coupling or level-set coupling object
    );

    void create_new_mesh_coupling(const std::string& cond_name,
        std::shared_ptr<Core::FE::Discretization>
            cond_dis,  ///< discretization from which the cutter discretization can be derived
        const int coupling_id);

    /// create a new mesh-coupling object based on the given coupling discretization
    void add_mesh_coupling(const std::string& cond_name,
        std::shared_ptr<Core::FE::Discretization> cond_dis, const int coupling_id)
    {
      if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_FSI_PART or
          cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_FSI_MONO)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFSI>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_FPI_MONO)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFPI>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, MeshCouplingFPI::ps_ps));
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFPI>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, MeshCouplingFPI::ps_pf));
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFPI>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, MeshCouplingFPI::pf_ps));
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFPI>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, MeshCouplingFPI::pf_pf));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingWeakDirichlet>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, (bg_dis_ == cond_dis)));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_NEUMANN)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingNeumann>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, (bg_dis_ == cond_dis)));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingNavierSlip>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, (bg_dis_ == cond_dis)));
      }
      else if (cond_type_string_to_enum(cond_name) ==
               Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingNavierSlipTwoPhase>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, (bg_dis_ == cond_dis)));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCouplingFluidFluid>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
      else if (cond_type_string_to_enum(cond_name) ==
               Inpar::XFEM::CouplingCond_EMBEDDEDMESH_SOLID_SURF)
      {
        mesh_coupl_.push_back(std::make_shared<MeshCoupling>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, "", false));
      }
      else if (cond_type_string_to_enum(cond_name) ==
               Inpar::XFEM::CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL)
      {
        // do nothing
      }
      else
      {
        mesh_coupl_.push_back(std::make_shared<MeshCoupling>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_, "", (bg_dis_ == cond_dis)));
      }
    }

    /// add a new level-set-coupling object based on the given coupling discretization
    void add_level_set_coupling(const std::string& cond_name,
        std::shared_ptr<Core::FE::Discretization>
            cond_dis,  ///< discretization from which the cutter discretization can be derived
        const int coupling_id)
    {
      if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET)
      {
        levelset_coupl_.push_back(std::make_shared<LevelSetCouplingWeakDirichlet>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
      else if (cond_type_string_to_enum(cond_name) == Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN)
      {
        levelset_coupl_.push_back(std::make_shared<LevelSetCouplingNeumann>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
      else if (cond_type_string_to_enum(cond_name) ==
               Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP)
      {
        levelset_coupl_.push_back(std::make_shared<LevelSetCouplingNavierSlip>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
      else
      {
        levelset_coupl_.push_back(std::make_shared<LevelSetCoupling>(
            bg_dis_, cond_name, cond_dis, coupling_id, time_, step_));
      }
    }



    /// Getters

    /// get cutter discretization the coupling side belongs to
    std::shared_ptr<Core::FE::Discretization> get_cutter_dis(
        const int coup_sid  ///< the global id of the coupling side
    )
    {
      if (is_level_set_coupling(coup_sid)) return nullptr;

      // get the mesh coupling object index
      int mc_idx = get_mesh_coupling_index(coup_sid);

      return mesh_coupl_[mc_idx]->get_cutter_dis();
    }

    /// get cutter discretization the coupling side belongs to
    std::shared_ptr<Core::FE::Discretization> get_coupling_dis(
        const int coup_sid  ///< the global id of the coupling side
    )
    {
      if (is_level_set_coupling(coup_sid)) return bg_dis_;

      // get the mesh coupling object index
      int mc_idx = get_mesh_coupling_index(coup_sid);

      return mesh_coupl_[mc_idx]->get_coupling_dis();
    }

    std::shared_ptr<MeshCoupling> get_mesh_coupling(const int mc_idx)
    {
      if (mc_idx < (int)mesh_coupl_.size() and mc_idx >= 0) return mesh_coupl_[mc_idx];

      return nullptr;
    }

    std::shared_ptr<LevelSetCoupling> get_level_set_coupling(const int ls_idx)
    {
      if (ls_idx < (int)levelset_coupl_.size() and ls_idx >= 0) return levelset_coupl_[ls_idx];

      return nullptr;
    }

    std::shared_ptr<CouplingBase> get_coupling(const std::string& name)
    {
      std::shared_ptr<CouplingBase> coupling = nullptr;

      coupling = get_mesh_coupling(name);

      if (coupling != nullptr) return coupling;

      coupling = get_level_set_coupling(name);

      if (coupling != nullptr) return coupling;

      return nullptr;
    }


    std::shared_ptr<MeshCoupling> get_mesh_coupling(const std::string& name)
    {
      for (int m = 0; m < num_mesh_coupling(); m++)
      {
        if (mesh_coupl_[m]->get_name() == name) return mesh_coupl_[m];
      }
      return nullptr;
    }

    int get_coupling_index(const std::string& name)
    {
      int coup_idx = -1;

      for (int m = 0; m < num_mesh_coupling(); m++)
      {
        if (mesh_coupl_[m]->get_name() == name) return m;
      }

      for (int l = 0; l < num_level_set_coupling(); l++)
      {
        if (levelset_coupl_[l]->get_name() == name) return (num_mesh_coupling() + l);
      }

      return coup_idx;
    }


    int get_mesh_coupling_index(const std::string& name)
    {
      for (int m = 0; m < num_mesh_coupling(); m++)
      {
        if (mesh_coupl_[m]->get_name() == name) return m;
      }
      return -1;
    }

    std::shared_ptr<LevelSetCoupling> get_level_set_coupling(const std::string& name)
    {
      for (int l = 0; l < num_level_set_coupling(); l++)
      {
        if (levelset_coupl_[l]->get_name() == name) return levelset_coupl_[l];
      }
      return nullptr;
    }


    Inpar::XFEM::AveragingStrategy get_averaging_strategy(
        const int coup_sid,  ///< the global id of the coupling side
        const int back_eid   ///< the global element id of the background mesh
    )
    {
      if (is_level_set_coupling(coup_sid))
      {
        // TODO: currently only one level-set field supported
        // get the level-set coupling object index for given background element
        const int lsc_idx = get_level_set_coupling_index(back_eid);

        return levelset_coupl_[lsc_idx]->get_averaging_strategy();
      }
      else if (is_mesh_coupling(coup_sid))
      {
        // get the mesh coupling object index
        const int mc_idx = get_mesh_coupling_index(coup_sid);

        return mesh_coupl_[mc_idx]->get_averaging_strategy();
      }
      else
        FOUR_C_THROW(
            "there is no valid mesh-/levelset-coupling condition object for side: {}", coup_sid);

      return Inpar::XFEM::invalid;
    }

    /// ...
    int get_mesh_coupling_index(const int coup_sid)
    {
      // safety checks
      if (coup_sid < 0)
      {
        //       FOUR_C_THROW("invalid negative coupling side id {}", coup_sid);
        return -1;
      }
      if (levelset_gid_ >= 0 and coup_sid > levelset_gid_)
      {
        //       FOUR_C_THROW("invalid coupling side id {}", coup_sid);
        return -1;
      }

      if (is_level_set_coupling(coup_sid))
      {
        //      FOUR_C_THROW("level-set side does not have a cutter discretization. Why do you
        //      call this?");
        return -1;
      }


      const int num_mesh_coupl = mesh_coupl_.size();
      if (num_mesh_coupl == 0)
      {
        //      FOUR_C_THROW("no mesh coupling objects available?!");
        return -1;
      }

      for (int idx = (int)mesh_coupl_.size() - 1; idx >= 0; idx--)  // inverse loop
      {
        if (coup_sid >= mesh_coupl_start_gid_[idx])
        {
          return idx;
        }
      }

      FOUR_C_THROW("no valid mesh coupling index found!");
      return -1;
    }


    /// ...
    int get_level_set_coupling_index(const int back_eid  ///< global element id
    )
    {
      // find out which level-set index is the active one for the given background element

      const Core::LinAlg::Map* elecolmap = bg_dis_->element_col_map();
      const int lid = elecolmap->lid(back_eid);

      const int lsc_idx = ele_lsc_coup_idx_col_->get_local_values()[lid];

      return lsc_idx;
    }

    int get_coupling_index(const int coup_sid, const int back_eid)
    {
      int coup_idx = -1;

      if (is_level_set_coupling(coup_sid))
        coup_idx = num_mesh_coupling() + get_level_set_coupling_index(back_eid);
      else
        coup_idx = get_mesh_coupling_index(coup_sid);

      return coup_idx;
    }

    // Get Boundary Cell Clone Information <clone_coup_idx, clone_coup_sid>
    std::vector<std::pair<int, int>> get_bc_clone_information(
        const int coup_sid, const int back_eid, int coup_idx = -1);

    int get_level_set_coupling_gid() { return levelset_gid_; }

    /// check if the given coupling side corresponds the unique level-set side
    bool is_level_set_coupling(const int coupl_sid) { return coupl_sid == levelset_gid_; }

    bool is_mesh_coupling(const int coup_sid) { return get_mesh_coupling_index(coup_sid) != -1; }

    bool has_level_set_coupling() { return levelset_coupl_.size() > 0; }

    bool has_mesh_coupling() { return mesh_coupl_.size() > 0; }

    int num_coupling() { return (num_mesh_coupling() + num_level_set_coupling()); }

    std::shared_ptr<CouplingBase> get_coupling_by_idx(const int coup_idx)
    {
      if (coup_idx >= num_mesh_coupling())
        return get_level_set_coupling(coup_idx - num_mesh_coupling());
      else if (coup_idx >= 0)
        return get_mesh_coupling(coup_idx);
      else
        return nullptr;

      return nullptr;
    }


    int num_mesh_coupling() { return mesh_coupl_.size(); }

    int num_level_set_coupling() { return levelset_coupl_.size(); }


    bool is_level_set_condition(const int coup_idx)
    {
      if (coup_idx >= num_mesh_coupling()) return true;

      return false;
    }


    bool is_mesh_condition(const int coup_idx)
    {
      if (coup_idx >= 0 and !is_level_set_condition(coup_idx)) return true;

      return false;
    }


    /// get the side element of the respective boundary discretization
    Core::Elements::Element* get_side(const int coup_sid  ///< the overall global coupling side id
    )
    {
      // get the mesh coupling object index
      const int mc_idx = get_mesh_coupling_index(coup_sid);

      // compute the side id w.r.t the cutter discretization the side belongs to
      const int cutterdis_sid = get_cutter_dis_ele_id(coup_sid, mc_idx);

      // get the boundary discretization, the side belongs to
      return mesh_coupl_[mc_idx]->get_side(cutterdis_sid);
    }

    /// get the coupling element (the side for xfluid-sided averaging) for a given global coupl.
    /// side id
    Core::Elements::Element* get_coupling_element(
        const int coup_sid,  ///< the overall global coupling side id
        Core::Elements::Element* ele);

    //! get the element from the conditioned dis for a local coupling side element id
    Core::Elements::Element* get_cond_element(
        const int coup_sid  ///< global side element id w.r.t cutter discretization
    )
    {
      if (!is_mesh_coupling(coup_sid))
        FOUR_C_THROW("No cond. element available for non-mesh coupling!");

      // get the mesh coupling object index
      const int mc_idx = get_mesh_coupling_index(coup_sid);

      // a map between cond. elements and side ids of the cutter dis is only available
      // for fluidfluid conditions; otherwise this is a bad request
      std::shared_ptr<MeshCouplingFluidFluid> mc_xff =
          std::dynamic_pointer_cast<MeshCouplingFluidFluid>(mesh_coupl_[mc_idx]);
      if (mc_xff == nullptr)
        FOUR_C_THROW("Can't access cond dis elements for a given side id in non-xff cases!");
      const int cutterdis_sid = get_cutter_dis_ele_id(coup_sid, mc_idx);
      return mc_xff->get_cond_element(cutterdis_sid);
    }

    // the cutwizard should add elements via the manager!!!

    // TODO: TransformID routines (localToGlobal, global_to_local

    // get the side id w.r.t. the cutter discretization
    int get_cutter_dis_ele_id(const int coup_sid, const int mc_idx)
    {
      return coup_sid - mesh_coupl_start_gid_[mc_idx];
    }

    // get the global coupling side id for a given mesh coupling and local side-id w.r.t. cutter
    // discretization
    int get_global_ele_id(const int cutterdis_sid, const int mc_idx)
    {
      return cutterdis_sid + mesh_coupl_start_gid_[mc_idx];
    }

    // get the global coupling side id for a given mesh coupling and local side-id w.r.t. cutter
    // discretization
    int get_mesh_coupling_start_gid(const int mc_idx) { return mesh_coupl_start_gid_[mc_idx]; }

    EleCoupCond get_coupling_condition(const int coup_sid,  ///< the global id of the coupling side
        const int back_eid  ///< the global element id of the background mesh
    )
    {
      if (is_level_set_coupling(coup_sid))
      {
        // TODO: currently only one level-set field supported
        // get the level-set coupling object index for given background element
        const int lsc_idx = get_level_set_coupling_index(back_eid);

        return levelset_coupl_[lsc_idx]->get_coupling_condition(back_eid);
      }
      else if (is_mesh_coupling(coup_sid))
      {
        // get the mesh coupling object index
        const int mc_idx = get_mesh_coupling_index(coup_sid);

        // compute the side id w.r.t the cutter discretization the side belongs to
        const int cutterdis_sid = get_cutter_dis_ele_id(coup_sid, mc_idx);

        return mesh_coupl_[mc_idx]->get_coupling_condition(cutterdis_sid);
      }
      else
        FOUR_C_THROW(
            "there is no valid mesh-/levelset-coupling condition object for side: {}", coup_sid);

      return EleCoupCond(Inpar::XFEM::CouplingCond_NONE, nullptr);
    }


    bool is_coupling(const int coup_sid,  ///< the global id of the coupling side
        const int back_eid                ///< the global element id of the background mesh
    )
    {
      const EleCoupCond& coup_cond = get_coupling_condition(coup_sid, back_eid);

      return is_coupling_condition(coup_cond.first);
    }

    /// have coupling matrices to be evaluated or not?
    bool is_coupling_condition(const std::string& cond_name)
    {
      return is_coupling_condition(cond_type_string_to_enum(cond_name));
    }

    /// have coupling matrices to be evaluated or not?
    bool is_coupling_condition(const Inpar::XFEM::EleCouplingCondType& cond_type)
    {
      switch (cond_type)
      {
        case Inpar::XFEM::CouplingCond_SURF_FSI_MONO:
        case Inpar::XFEM::CouplingCond_SURF_FPI_MONO:
        case Inpar::XFEM::CouplingCond_SURF_FLUIDFLUID:
        case Inpar::XFEM::CouplingCond_LEVELSET_TWOPHASE:
        case Inpar::XFEM::CouplingCond_LEVELSET_COMBUSTION:
        {
          return true;
          break;
        }
        case Inpar::XFEM::CouplingCond_SURF_FSI_PART:
        case Inpar::XFEM::CouplingCond_SURF_WEAK_DIRICHLET:
        case Inpar::XFEM::CouplingCond_SURF_NEUMANN:
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP:
        case Inpar::XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE:
        case Inpar::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
        case Inpar::XFEM::CouplingCond_LEVELSET_NEUMANN:
        case Inpar::XFEM::CouplingCond_LEVELSET_NAVIER_SLIP:
        {
          return false;
          break;
        }
        default:
          FOUR_C_THROW("coupling condition type not known {}", cond_type);
          break;
      }

      return false;
    }

    void set_level_set_field(const double time);

    void write_access_geometric_quantities(Core::LinAlg::Vector<double>& scalaraf,
        Core::LinAlg::MultiVector<double>& smoothed_gradphiaf,
        Core::LinAlg::Vector<double>& curvatureaf);

    void export_geometric_quantities();

    std::shared_ptr<Core::LinAlg::Vector<double>>& get_level_set_field()
    {
      if (!is_levelset_uptodate_)
        update_level_set_field();  // update the unique level-set field based on the background
                                   // discretization

      return bg_phinp_;
    }

    std::shared_ptr<const Core::LinAlg::Vector<double>> get_level_set_field_col();

    void clear_state();

    void set_state();

    void set_state_displacement();

    /// update interface field state vectors
    void update_state_vectors();

    void complete_state_vectors();

    void zero_state_vectors_fsi();

    void gmsh_output(const std::string& filename_base, const int step, const int gmsh_step_diff,
        const bool gmsh_debug_out_screen);

    void gmsh_output_discretization(std::ostream& gmshfilecontent);

    void output(const int step, const double time, const bool write_restart_data);

    /// compute lift and drag values by integrating the true residuals
    void lift_drag(const int step, const double time);

    void read_restart(const int step);

    void prepare_solve();

    bool has_moving_interface();

    bool has_averaging_strategy(Inpar::XFEM::AveragingStrategy strategy);

    void get_coupling_ele_location_vector(const int coup_sid, std::vector<int>& patchlm);

    /// Get the average weights from the coupling objects
    void get_average_weights(const int coup_sid,  ///< the overall global coupling side id
        Core::Elements::Element* xfele,           ///< xfluid ele
        double& kappa_m,                          ///< Weight parameter (parameter +/master side)
        double& kappa_s,                          ///< Weight parameter (parameter -/slave  side)
        bool& non_xfluid_coupling);

    /// compute viscous part of Nitsche's penalty term scaling for Nitsche's method
    void get_visc_penalty_stabfac(const int coup_sid,  ///< the overall global coupling side id
        Core::Elements::Element* xfele,                ///< xfluid ele
        const double& kappa_m,  ///< Weight parameter (parameter +/master side)
        const double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
        const double& inv_h_k,  ///< the inverse characteristic element length h_k
        const Discret::Elements::FluidEleParameterXFEM*
            params,                 ///< parameterlist which specifies interface configuration
        double& NIT_visc_stab_fac,  ///< viscous part of Nitsche's penalty term
        double&
            NIT_visc_stab_fac_tang  ///< viscous part of Nitsche's penalty term in tang direction
    );

    /// get the estimation of the penalty scaling in Nitsche's method from the trace inequality
    /// for a specific coupling side
    double get_trace_estimate_max_eigenvalue(
        const int coup_sid  ///< the overall global coupling side id
    );

    /// set material pointer for volume
    void get_volume_cell_material(Core::Elements::Element* actele,
        std::shared_ptr<Core::Mat::Material>& mat, const Cut::VolumeCell* vc);

    /// set material pointer for volume cell for (coupling) master side
    void get_interface_master_material(Core::Elements::Element* actele,
        std::shared_ptr<Core::Mat::Material>& mat, const Cut::VolumeCell* vc);

    /// set material pointer for coupling slave side
    void get_interface_slave_material(
        Core::Elements::Element* actele, std::shared_ptr<Core::Mat::Material>& mat, int coup_sid);

    /// Initialize Fluid intersection/Cut State
    bool initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
        std::shared_ptr<Core::FE::Discretization> fluiddis,
        std::shared_ptr<XFEM::ConditionManager> condition_manager,
        std::shared_ptr<Teuchos::ParameterList> fluidparams);

   public:
    //! initialized the coupling object
    void init();

    //! setup the coupling object
    void setup();

    /// get the indicator state
    inline const bool& is_init() const { return isinit_; };

    /// get the indicator state
    inline const bool& is_setup() const { return issetup_; };

    /// Check if init() and setup() have been called, yet.
    inline void check_init_setup() const
    {
      if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
    }

    /// Check if init() has been called
    inline void check_init() const
    {
      if (not is_init()) FOUR_C_THROW("Call init() first!");
    }

   private:
    // build the whole object which then can be used
    void create();

    ///
    void update_level_set_field();

    /// combine two levelset fields via boolean type set operations and set result into vec1
    void combine_level_set_field(Core::LinAlg::Vector<double>& vec1,
        Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
        Core::LinAlg::Vector<int>& node_lsc_coup_idx,
        XFEM::CouplingBase::LevelSetBooleanType ls_boolean_type);

    /// check if the vector maps are equal
    void check_for_equal_maps(
        Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2);

    /// combine two levelset fields via boolean type "union" set operation and put result into
    /// vec1
    void set_minimum(Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2,
        const int lsc_index_2, Core::LinAlg::Vector<int>& node_lsc_coup_idx);

    /// combine two levelset fields via boolean type "cut" set operation and put result into vec1
    void set_maximum(Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2,
        const int lsc_index_2, Core::LinAlg::Vector<int>& node_lsc_coup_idx);

    /// combine two levelset fields via boolean type "difference" set operation and put result
    /// into vec1
    void set_difference(Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2,
        const int lsc_index_2, Core::LinAlg::Vector<int>& node_lsc_coup_idx);

    /// combine two levelset fields via boolean type "sym_difference" set operation and put result
    /// into vec1
    void set_symmetric_difference(Core::LinAlg::Vector<double>& vec1,
        Core::LinAlg::Vector<double>& vec2, const int lsc_index_2,
        Core::LinAlg::Vector<int>& node_lsc_coup_idx);

    void build_complementary_level_set(Core::LinAlg::Vector<double>& vec1);

    ///<
    std::map<std::string, int> dofset_coupling_map_;

    ///< background discretiaztion w.r.t for which the couling manager is constructed
    std::shared_ptr<Core::FE::Discretization> bg_dis_;

    /// mesh coupling objects
    std::vector<std::shared_ptr<MeshCoupling>> mesh_coupl_;

    /// level-set coupling objects
    std::vector<std::shared_ptr<LevelSetCoupling>> levelset_coupl_;

    /// starting index for element side global id
    std::vector<int> mesh_coupl_start_gid_;

    /// index for the unique level-set-side global id
    int levelset_gid_;

    /// global number of mesh and level-set coupling sides over all processors
    int numglobal_coupling_sides_;

    /// time
    double time_;

    /// time step
    int step_;

    //! @name state vectors based on background discretization

    //! background-dis state vectors for levelset applications
    bool is_levelset_uptodate_;
    std::shared_ptr<Core::LinAlg::Vector<int>> ele_lsc_coup_idx_col_;
    std::shared_ptr<Core::LinAlg::Vector<double>> bg_phinp_;
    //@}

    bool isinit_;  //! is conditionmanager initialized

    bool issetup_;  //! is conditionmanager set up
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
