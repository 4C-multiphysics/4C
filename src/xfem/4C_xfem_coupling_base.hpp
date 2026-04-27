// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_COUPLING_BASE_HPP
#define FOUR_C_XFEM_COUPLING_BASE_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_material_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_xfem_input.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // finally this parameter list should go and all interface relevant parameters should be stored
    // in the condition manager or coupling objects
    class FluidEleParameterXFEM;
  }  // namespace Elements
}  // namespace Discret

namespace XFEM
{
  using EleCoupCond = std::pair<XFEM::EleCouplingCondType, const Core::Conditions::Condition*>;

  XFEM::EleCouplingCondType cond_type_string_to_enum(const std::string& condname);

  class CouplingBase
  {
   public:
    //! which boolean set operator used to combine current field with previous one
    enum LevelSetBooleanType
    {
      ls_none = 0,           // used for first Boundary condition level-setcoupling
      ls_cut = 1,            // latex: \cap:         Omega 1 \cap \Omega 2
      ls_union = 2,          // latex: \cup          Omega 1 \cup \Omega 2
      ls_difference = 3,     // latex: \backslash    Omega 1 - Omega 2
      ls_sym_difference = 4  // latex: \triangle     (Omega 1 - Omega 2) \cup (Omega 2 - \Omega 1)
    };

    //! constructor
    explicit CouplingBase(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
    );

    //! destructor
    virtual ~CouplingBase() = default;
    virtual void set_dof_set_coupling_map(const std::map<std::string, int>& dofset_coupling_map)
    {
      dofset_coupling_map_ = dofset_coupling_map;
    }

    virtual void set_coupling_dofsets() {};

    int get_coupling_dofset_nds(const std::string& name)
    {
      if (not(dofset_coupling_map_.count(name) == 1))
        FOUR_C_THROW("{} -dofset not set in dofset_coupling_map for fluid dis!", name);

      return dofset_coupling_map_[name];
    }


    //! initialized the coupling object
    virtual void init();

    //! setup the coupling object
    virtual void setup();

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

    //! cutter dis should be loaded into the cut?
    virtual bool cut_geometry() { return true; }

    void set_time_and_step(const double time, const int step)
    {
      time_ = time;
      step_ = step;
    }

    void increment_time_and_step(const double dt)
    {
      dt_ = dt;
      time_ += dt;
      step_ += 1;
    }

    void get_condition_by_coupling_id(const std::vector<const Core::Conditions::Condition*>& mycond,
        const int coupling_id, std::vector<const Core::Conditions::Condition*>& mynewcond);

    void status(const int coupling_idx, const int side_start_gid);


    std::string dis_name_to_string(std::shared_ptr<Core::FE::Discretization> dis)
    {
      if (dis == nullptr) return "---";

      return dis->name();
    }

    std::string type_to_string_for_print(const XFEM::EleCouplingCondType& type)
    {
      if (type == XFEM::CouplingCond_SURF_FSI_PART)
        return "XFSI Partitioned";
      else if (type == XFEM::CouplingCond_SURF_FSI_MONO)
        return "XFSI Monolithic";
      else if (type == XFEM::CouplingCond_SURF_FPI_MONO)
        return "XFPI Monolithic";
      else if (type == XFEM::CouplingCond_SURF_FLUIDFLUID)
        return "FLUID-FLUID Coupling";
      else if (type == XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET)
        return "WEAK DIRICHLET BC / LS";
      else if (type == XFEM::CouplingCond_LEVELSET_NEUMANN)
        return "NEUMANN BC        / LS";
      else if (type == XFEM::CouplingCond_LEVELSET_NAVIER_SLIP)
        return "NAVIER SLIP BC    / LS";
      else if (type == XFEM::CouplingCond_LEVELSET_TWOPHASE)
        return "TWO-PHASE Coupling";
      else if (type == XFEM::CouplingCond_LEVELSET_COMBUSTION)
        return "COMBUSTION Coupling";
      else if (type == XFEM::CouplingCond_SURF_WEAK_DIRICHLET)
        return "WEAK DIRICHLET BC / MESH";
      else if (type == XFEM::CouplingCond_SURF_NEUMANN)
        return "NEUMANN BC        / MESH";
      else if (type == XFEM::CouplingCond_SURF_NAVIER_SLIP)
        return "NAVIER SLIP BC    / MESH";
      else if (type == XFEM::CouplingCond_SURF_NAVIER_SLIP_TWOPHASE)
        return "NAVIER SLIP TWOPHASE BC    / MESH";
      else if (type == XFEM::CouplingCond_EMBEDDEDMESH_SOLID_SURF)
        return "EMBEDDEDMESH SOLID SURF COUPLING / MESH";
      else if (type == XFEM::CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL)
        return "XFEM Background Solid Volume";
      else
        FOUR_C_THROW("unsupported coupling condition type {}", type);

      return "UNKNOWN";
    }

    std::string averaging_to_string_for_print(const XFEM::AveragingStrategy& strategy)
    {
      if (strategy == XFEM::Xfluid_Sided)
        return "XFLUID-sided averaging";
      else if (strategy == XFEM::Embedded_Sided)
        return "EMBEDDED-sided averaging";
      else if (strategy == XFEM::Mean)
        return "MEAN averaging";
      else if (strategy == XFEM::Harmonic)
        return "HARMONIC averaging";
      else if (strategy == XFEM::invalid)
        return "INVALID";
      else
        FOUR_C_THROW("unsupported averaging strategy {}", strategy);

      return "UNKNOWN";
    }


    const EleCoupCond& get_coupling_condition(
        const int gid  ///< global element element id w.r.t cutter discretization (bgele->Id for
                       ///< LevelsetCoupling cut and side-Id for MeshCoupling)
    )
    {
      int lid = cutter_dis_->element_col_map()->lid(gid);
      return cutterele_conds_[lid];
    }

    //! get the coupling element (equal to the side for xfluid-sided, mesh-based coupling)
    virtual Core::Elements::Element* get_coupling_element(
        const int eid  ///< global side element id w.r.t coupling discretization (background element
                       ///< eid for levelset couplings)
    )
    {
      return (coupl_dis_ != nullptr) ? coupl_dis_->g_element(eid) : nullptr;
    }

    virtual const std::string& get_name() { return coupl_name_; }

    std::shared_ptr<Core::FE::Discretization> get_cutter_dis() { return cutter_dis_; }
    std::shared_ptr<Core::FE::Discretization> get_coupling_dis() { return coupl_dis_; }
    std::shared_ptr<Core::FE::Discretization> get_cond_dis() { return cond_dis_; }

    XFEM::AveragingStrategy get_averaging_strategy() { return averaging_strategy_; }

    virtual void prepare_solve() {};

    virtual bool has_moving_interface() = 0;

    virtual void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond)
    {
      FOUR_C_THROW("evaluate_coupling_conditions should be implemented by derived class");
    };

    virtual void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<6, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond)
    {
      FOUR_C_THROW("evaluate_coupling_conditions should be implemented by derived class");
    };

    virtual void evaluate_coupling_conditions_old_state(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond)
    {
      FOUR_C_THROW("evaluate_coupling_conditions_old_state should be implemented by derived class");
    };

    /// set material pointer for coupling slave side
    virtual void get_interface_slave_material(
        Core::Elements::Element* actele, std::shared_ptr<Core::Mat::Material>& mat)
    {
      mat = nullptr;
    }

    /// get the sliplength for the specific coupling condition
    virtual void get_slip_coefficient(double& slipcoeff, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond)
    {
      slipcoeff = 0.0;
    }

    std::map<XFEM::CoupTerm, std::pair<bool, double>>& get_configurationmap(
        double& kappa_m,                          //< fluid sided weighting
        double& visc_m,                           //< master sided dynamic viscosity
        double& visc_s,                           //< slave sided dynamic viscosity
        double& density_m,                        //< master sided density
        double& visc_stab_tang,                   //< viscous tangential NIT Penalty scaling
        double& full_stab,                        //< full NIT Penalty scaling
        const Core::LinAlg::Matrix<3, 1>& x,      //< Position x
        const Core::Conditions::Condition* cond,  //< Condition
        Core::Elements::Element* ele,             //< Element
        Core::Elements::Element* bele,            //< Boundary Element
        double* funct,  //< local shape function for Gauss Point (from fluid element)
        double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
        Core::LinAlg::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
        Core::LinAlg::Matrix<3, 1>& normal,     //< normal at gp
        Core::LinAlg::Matrix<3, 1>& vel_m,      //< master velocity at gp
        double* fulltraction  //< precomputed fsi traction (sigmaF n + gamma relvel)
    )
    {
      update_configuration_map_gp(kappa_m, visc_m, visc_s, density_m, visc_stab_tang, full_stab, x,
          cond, ele, bele, funct, derxy, rst_slave, normal, vel_m, fulltraction);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      // Do some safety checks, every combination which is not handled correct by the element level
      // should be caught here ... As this check is done for every Gausspoint, just do this in DEBUG
      // Version ... Feel free the add more checks ...

      // In Case we do just use Penalty or Adjoint we should still set the scaling on both, to
      // guarantee we the the correct constraint!
      if (((configuration_map_.at(XFEM::F_Adj_Col).first &&
               !configuration_map_.at(XFEM::F_Pen_Col).first) ||
              (!configuration_map_.at(XFEM::F_Adj_Col).first &&
                  configuration_map_.at(XFEM::F_Pen_Col).first)) &&
          fabs(configuration_map_.at(XFEM::F_Adj_Col).second -
               configuration_map_.at(XFEM::F_Pen_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());
      if (((configuration_map_.at(XFEM::X_Adj_Col).first &&
               !configuration_map_.at(XFEM::X_Pen_Col).first) ||
              (!configuration_map_.at(XFEM::X_Adj_Col).first &&
                  configuration_map_.at(XFEM::X_Pen_Col).first)) &&
          fabs(configuration_map_.at(XFEM::X_Adj_Col).second -
               configuration_map_.at(XFEM::X_Pen_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());
      if (((configuration_map_.at(XFEM::F_Adj_n_Col).first &&
               !configuration_map_.at(XFEM::F_Pen_n_Col).first) ||
              (!configuration_map_.at(XFEM::F_Adj_n_Col).first &&
                  configuration_map_.at(XFEM::F_Pen_n_Col).first)) &&
          fabs(configuration_map_.at(XFEM::F_Adj_n_Col).second -
               configuration_map_.at(XFEM::F_Pen_n_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());
      if (((configuration_map_.at(XFEM::X_Adj_n_Col).first &&
               !configuration_map_.at(XFEM::X_Pen_n_Col).first) ||
              (!configuration_map_.at(XFEM::X_Adj_n_Col).first &&
                  configuration_map_.at(XFEM::X_Pen_n_Col).first)) &&
          fabs(configuration_map_.at(XFEM::X_Adj_n_Col).second -
               configuration_map_.at(XFEM::X_Pen_n_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());
      if (((configuration_map_.at(XFEM::F_Adj_t_Col).first &&
               !configuration_map_.at(XFEM::F_Pen_t_Col).first) ||
              (!configuration_map_.at(XFEM::F_Adj_t_Col).first &&
                  configuration_map_.at(XFEM::F_Pen_t_Col).first)) &&
          fabs(configuration_map_.at(XFEM::F_Adj_t_Col).second -
               configuration_map_.at(XFEM::F_Pen_t_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());
      if (((configuration_map_.at(XFEM::X_Adj_t_Col).first &&
               !configuration_map_.at(XFEM::X_Pen_t_Col).first) ||
              (!configuration_map_.at(XFEM::X_Adj_t_Col).first &&
                  configuration_map_.at(XFEM::X_Pen_t_Col).first)) &&
          fabs(configuration_map_.at(XFEM::X_Adj_t_Col).second -
               configuration_map_.at(XFEM::X_Pen_t_Col).second) > 1e-16)
        FOUR_C_THROW(
            "{}: You should set Scalings for Adjoint and Penalty Column, even if just one is used, "
            "as we support at the moment just equal penalty and adjoint consistent constraints!",
            cond_name_.c_str());

      // At the moment you cannot use different consistent constraints between Adjoint and Penalty
      // terms
      // Check if we need a more general implementation (If constraints between Adjoint and Penalty
      // are not the same!)
      if (fabs(configuration_map_.at(XFEM::F_Adj_Col).second -
               configuration_map_.at(XFEM::F_Pen_Col).second) > 1e-16 ||
          fabs(configuration_map_.at(XFEM::F_Adj_n_Col).second -
               configuration_map_.at(XFEM::F_Pen_n_Col).second) > 1e-16 ||
          fabs(configuration_map_.at(XFEM::F_Adj_t_Col).second -
               configuration_map_.at(XFEM::F_Pen_t_Col).second) > 1e-16 ||
          fabs(configuration_map_.at(XFEM::X_Adj_Col).second -
               configuration_map_.at(XFEM::X_Pen_Col).second) > 1e-16 ||
          fabs(configuration_map_.at(XFEM::X_Adj_n_Col).second -
               configuration_map_.at(XFEM::X_Pen_n_Col).second) > 1e-16 ||
          fabs(configuration_map_.at(XFEM::X_Adj_t_Col).second -
               configuration_map_.at(XFEM::X_Pen_t_Col).second) > 1e-16)
      {
        std::cout << "F_Adj_Col/F_Pen_Col: " << configuration_map_.at(XFEM::F_Adj_Col).second << "/"
                  << configuration_map_.at(XFEM::F_Pen_Col).second << std::endl;
        std::cout << "F_Adj_n_Col/F_Pen_n_Col: " << configuration_map_.at(XFEM::F_Adj_n_Col).second
                  << "/" << configuration_map_.at(XFEM::F_Pen_n_Col).second << std::endl;
        std::cout << "F_Adj_t_Col/F_Pen_t_Col: " << configuration_map_.at(XFEM::F_Adj_t_Col).second
                  << "/" << configuration_map_.at(XFEM::F_Pen_t_Col).second << std::endl;
        std::cout << "X_Adj_Col/X_Pen_Col: " << configuration_map_.at(XFEM::X_Adj_Col).second << "/"
                  << configuration_map_.at(XFEM::X_Pen_Col).second << std::endl;
        std::cout << "X_Adj_n_Col/X_Pen_n_Col: " << configuration_map_.at(XFEM::X_Adj_n_Col).second
                  << "/" << configuration_map_.at(XFEM::X_Pen_n_Col).second << std::endl;
        std::cout << "X_Adj_t_Col/X_Pen_t_Col: " << configuration_map_.at(XFEM::X_Adj_t_Col).second
                  << "/" << configuration_map_.at(XFEM::X_Pen_t_Col).second << std::endl;
        FOUR_C_THROW(
            "{}: Your consistent constraint for Penalty and Adjoint term is not equal, go to "
            "element level and split up velint_diff_ for penalty and adjoint!",
            cond_name_.c_str());
      }

#endif
      return configuration_map_;
    }

    virtual void gmsh_output(const std::string& filename_base, const int step,
        const int gmsh_step_diff, const bool gmsh_debug_out_screen) {};

    /// get viscosity of the master fluid
    void get_viscosity_master(Core::Elements::Element* xfele,  ///< xfluid ele
        double& visc_m);                                       ///< viscosity mastersided

    /// get scaling of the master side for penalty (viscosity, E-modulus for solids)
    virtual void get_penalty_scaling_slave(Core::Elements::Element* coup_ele,  ///< xfluid ele
        double& penscaling_s)  ///< penalty scaling slavesided
    {
      FOUR_C_THROW("get_penalty_scaling_slave not implemented for this coupling object!");
    }

    /// get weighting parameters
    void get_average_weights(Core::Elements::Element* xfele,  ///< xfluid ele
        Core::Elements::Element* coup_ele,                    ///< coup_ele ele
        double& kappa_m,  ///< Weight parameter (parameter +/master side)
        double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
        bool& non_xfluid_coupling);

    /// get coupling specific weighting parameters (should be overload, whenever required)
    virtual void get_coupling_specific_average_weights(
        Core::Elements::Element* xfele,     ///< xfluid ele
        Core::Elements::Element* coup_ele,  ///< coup_ele ele
        double& kappa_m)                    ///< Weight parameter (parameter +/master side)
    {
      FOUR_C_THROW(
          "XFEM::CouplingBase: get_coupling_specific_average_weights not implemented for this "
          "coupling "
          "object!");
    }

    /// compute viscous part of Nitsche's penalty term scaling for Nitsche's method
    void get_visc_penalty_stabfac(Core::Elements::Element* xfele,  ///< xfluid ele
        Core::Elements::Element* coup_ele,                         ///< coup_ele ele
        const double& kappa_m,      ///< Weight parameter (parameter +/master side)
        const double& kappa_s,      ///< Weight parameter (parameter -/slave  side)
        const double& inv_h_k,      ///< the inverse characteristic element length h_k
        double& NIT_visc_stab_fac,  ///< viscous part of Nitsche's penalty term
        double&
            NIT_visc_stab_fac_tang,    ///< viscous part of Nitsche's penalty term in tang direction
        const double& NITStabScaling,  ///< prefactor of Nitsche's scaling in normal direction
        const double&
            NITStabScalingTang,  ///< prefactor of Nitsche's scaling in tangential direction
        const bool& IsPseudo2D,  ///< is this a pseudo 2d problem
        const XFEM::ViscStabTraceEstimate
            ViscStab_TraceEstimate  ///< trace estimate for visc stab fac
    );

    /// compute viscous part of Nitsche's penalty term scaling for Nitsche's method
    void get_visc_penalty_stabfac(Core::Elements::Element* xfele,  ///< xfluid ele
        Core::Elements::Element* coup_ele,                         ///< coup_ele ele
        const double& kappa_m,  ///< Weight parameter (parameter +/master side)
        const double& kappa_s,  ///< Weight parameter (parameter -/slave  side)
        const double& inv_h_k,  ///< the inverse characteristic element length h_k
        const Discret::Elements::FluidEleParameterXFEM*
            params,                 ///< parameterlist which specifies interface configuration
        double& NIT_visc_stab_fac,  ///< viscous part of Nitsche's penalty term
        double&
            NIT_visc_stab_fac_tang  ///< viscous part of Nitsche's penalty term in tang direction
    );


   protected:
    virtual void set_coupling_name()
    {
      coupl_name_ =
          cond_name_;  // the standard case are equal name of condition and coupling object
    }

    virtual void set_conditions_to_copy() {};

    virtual void set_cutter_discretization() {};

    virtual void set_condition_specific_parameters() {};

    virtual void set_element_conditions();

    void set_averaging_strategy();

    void set_coupling_discretization();

    virtual void prepare_cutter_output() {};

    virtual void do_condition_specific_setup() {};

    //! set the configuration map up for the specific coupling object
    virtual void setup_configuration_map() {};

    //! Updates configurationmap for specific Gausspoint
    virtual void update_configuration_map_gp(double& kappa_m,  //< fluid sided weighting
        double& visc_m,                                        //< master sided dynamic viscosity
        double& visc_s,                                        //< slave sided dynamic viscosity
        double& density_m,                                     //< master sided density
        double& visc_stab_tang,                   //< viscous tangential NIT Penalty scaling
        double& full_stab,                        //< full NIT Penalty scaling
        const Core::LinAlg::Matrix<3, 1>& x,      //< Position x in global coordinates
        const Core::Conditions::Condition* cond,  //< Condition
        Core::Elements::Element* ele,             //< Element
        Core::Elements::Element* bele,            //< Boundary Element
        double* funct,  //< local shape function for Gauss Point (from fluid element)
        double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
        Core::LinAlg::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
        Core::LinAlg::Matrix<3, 1>& normal,     //< normal at gp
        Core::LinAlg::Matrix<3, 1>& vel_m,      //< master velocity at gp
        double* fulltraction  //< precomputed fsi traction (sigmaF n + gamma relvel)
    ) {};

    virtual void init_state_vectors() {};

    virtual void perpare_cutter_output() {};

    void evaluate_dirichlet_function(Core::LinAlg::Matrix<3, 1>& ivel,
        const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time);

    void evaluate_neumann_function(Core::LinAlg::Matrix<3, 1>& itraction,
        const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time);

    void evaluate_neumann_function(Core::LinAlg::Matrix<6, 1>& itraction,
        const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond, double time);

    void evaluate_function(std::vector<double>& final_values, const double* x,
        const Core::Conditions::Condition* cond, const double time);

    void evaluate_scalar_function(double& final_value, const double* x, const double& val,
        const Core::Conditions::Condition* cond, const double time);

    //! @name Sets up a projection matrix
    /*!
    \brief Is utilized for separating Dirichlet and Neumann conditions
     */
    template <class M1, class M2>
    inline void setup_projection_matrix(M1& proj_matrix, const M2& normal)
    {
      double n_j = 0.0;
      for (unsigned int j = 0; j < nsd_; ++j)
      {
        n_j = normal(j, 0);
        for (unsigned int i = 0; i < nsd_; ++i)
        {
          proj_matrix(i, j) = ((i == j) ? 1.0 : 0.0) - normal(i, 0) * n_j;
        }
      }
      return;
    }

    ///< number of spatial dimensions
    size_t nsd_;

    ///< background discretization
    std::shared_ptr<Core::FE::Discretization> bg_dis_;

    ///------------------------
    // CUTTER-DISCRETIZATION specific member
    ///------------------------

    ///< name of the condition, by which the derived cutter discretization is identified
    std::string cond_name_;

    ///< discretization from which the cutter discretization is derived
    std::shared_ptr<Core::FE::Discretization> cond_dis_;

    ///< id of composite of coupling conditions
    const int coupling_id_;

    ///< discretization w.r.t which the interface is described and w.r.t which the state vectors
    ///< describing the interface position are defined (bgdis for LevelSetCoupling and boundary dis
    ///< for MeshCoupling)
    std::shared_ptr<Core::FE::Discretization> cutter_dis_;

    ///< pairs of condition type and pointer to Core::Conditions::Condition for all column elements
    ///< of the cutter discretization (bgdis for LevelSetCoupling and boundary dis for MeshCoupling)
    std::vector<EleCoupCond> cutterele_conds_;

    std::vector<std::string>
        conditions_to_copy_;  ///< list of conditions that will be copied to the new discretization
                              ///< and used to set for each cutter element

    //! Output specific
    std::shared_ptr<Core::IO::DiscretizationWriter> cutter_output_;


    ///------------------------
    // Coupling-DISCRETIZATION specific member
    ///------------------------

    ///< discretization with which the background discretization is coupled (structural dis, fluid
    ///< dis, poro dis, scatra dis, boundary dis), nullptr in case that no coupling terms but
    ///< only boundary terms are evaluated
    std::shared_ptr<Core::FE::Discretization> coupl_dis_;

    // TODO: be aware of the fact, that accessing the coupling object via Name is unsafe, it assumes
    // that only one coupling of that type is available < name of the mesh/levelset coupling object
    std::string coupl_name_;

    ///< averaging strategy, type of weighting
    XFEM::AveragingStrategy averaging_strategy_;

    int myrank_;

    double dt_;  ///< current time step size

    double time_;

    int step_;

    ///< map which configures element level (which terms are evaluated & scaled with which value)
    std::map<XFEM::CoupTerm, std::pair<bool, double>> configuration_map_;

    bool issetup_;

    bool isinit_;

    std::map<std::string, int> dofset_coupling_map_;

   private:
    //! Initializes configurationmap to zero (non-virtual)
    void init_configuration_map();
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
