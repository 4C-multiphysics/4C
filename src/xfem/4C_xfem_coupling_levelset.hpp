// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_COUPLING_LEVELSET_HPP
#define FOUR_C_XFEM_COUPLING_LEVELSET_HPP

#include "4C_config.hpp"

#include "4C_cut_point.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_xfem_coupling_base.hpp"
#include "4C_xfem_utils.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class CutWizard;
}

namespace XFEM
{
  /*!
  \brief
   */
  class LevelSetCoupling : public CouplingBase
  {
   public:
    //! constructor
    explicit LevelSetCoupling(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
    );

    void set_coupling_dofsets() override;

    bool have_matching_nodes(Core::FE::Discretization& dis_A, Core::FE::Discretization& dis_B);

    void map_cutter_to_bg_vector(Core::FE::Discretization& source_dis,
        Core::LinAlg::Vector<double>& source_vec_dofbased, const int source_nds,
        Core::FE::Discretization& target_dis, Core::LinAlg::Vector<double>& target_vec_dofbased,
        const int target_nds);

    // TODO: sort the functions...

    void set_cutter_discretization() override;

    void set_condition_specific_parameters() override {};

    void prepare_cutter_output() override;

    void do_condition_specific_setup() override;


    /// set levelset field by function, return if interface moved compared to last time step
    bool set_level_set_field(const double time);

    /// initialize level set based state vectors
    void init_state_vectors() override;

    virtual void init_state_vectors_bg();

    virtual void init_state_vectors_cutter();

    /// set level-boolean type
    virtual void set_level_set_boolean_type();

    virtual bool apply_complementary_operator();

    virtual void output(
        const int step, const double time, const bool write_restart_data, const int lsc_idx = 0);

    void gmsh_output(const std::string& filename_base, const int step, const int gmsh_step_diff,
        const bool gmsh_debug_out_screen) override;

    std::shared_ptr<Core::LinAlg::Vector<double>> get_level_set_field_as_node_row_vector();

    virtual void read_restart(const int step, const int lsc_idx = 0);

    bool has_moving_interface() override { return true; }

    void get_interface_slave_material(
        Core::Elements::Element* actele, std::shared_ptr<Core::Mat::Material>& mat) override
    {
      mat = nullptr;
    }

    XFEM::CouplingBase::LevelSetBooleanType get_boolean_combination() { return ls_boolean_type_; }

    //! export row vectors storing geometric quantities to col vectors
    virtual void export_geometric_quantities() {};


   private:
    void set_conditions_to_copy() override;

    /// set level-set field implemented in this routine
    double funct_implementation(const int func_no, const double* coords, const double t);

   protected:
    //! Output specific
    std::shared_ptr<Core::IO::DiscretizationWriter> bg_output_;

    //! @name fluid discretization related state vectors

    //! fluid-dis (bgdis) state vectors for levelset applications
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_;


    //@}

    //! @name scatra discretization related state vectors

    //! scatra-dis (cutterdis) state vectors for levelset applications, prepares nonmatching
    //! discretizations between scatra and fluid
    std::shared_ptr<Core::LinAlg::Vector<double>> cutter_phinp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> cutter_phinp_col_;

    //! The nodal curvature and smoothed gradient of the levelset field. (Stored w.r.t to the
    //! scatra-dis = cutter-dis)
    std::shared_ptr<Core::LinAlg::Vector<double>> curvaturenp_node_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> gradphinp_smoothed_node_;
    // std::shared_ptr<Core::LinAlg::MultiVector<double>>   gradphi2np_smoothed_node_;

    //! and column versions
    std::shared_ptr<Core::LinAlg::Vector<double>> curvaturenp_node_col_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> gradphinp_smoothed_node_col_;

    //! boolean operation type on level-set for current ls-field and previous combination of
    //! level-set fields
    XFEM::CouplingBase::LevelSetBooleanType ls_boolean_type_;


    // Specify way of creating the projection matrix
    Inpar::XFEM::ProjToSurface projtosurf_;

    //@}

    int bg_nds_phi_;      ///<
    int cutter_nds_phi_;  ///<

    double normal_orientation_;  ///< correction factor between normal of phi-gradient and normal in
                                 ///< xfluid

    bool have_nodematching_dis_;  ///< are bgdis and cutterdis node-matching?
  };

  class LevelSetCouplingBC : public LevelSetCoupling
  {
   public:
    //! constructor
    explicit LevelSetCouplingBC(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
    );


    void prepare_solve() override;

    bool has_moving_interface() override;

   protected:
    bool has_interface_moved_;  ///< did interface move compared to the last time step?
  };


  /*!
  \brief
   */
  class LevelSetCouplingWeakDirichlet : public LevelSetCouplingBC
  {
   public:
    //! constructor
    explicit LevelSetCouplingWeakDirichlet(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
        )
        : LevelSetCouplingBC(bg_dis, cond_name, cond_dis, coupling_id, time, step)
    {
    }

   public:
    void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

    void evaluate_coupling_conditions_old_state(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

   protected:
    //! Initializes configurationmap
    void setup_configuration_map() override;

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
        ) override;
  };


  /*!
  \brief
   */
  class LevelSetCouplingNeumann : public LevelSetCouplingBC
  {
   public:
    //! constructor
    explicit LevelSetCouplingNeumann(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
        )
        : LevelSetCouplingBC(bg_dis, cond_name, cond_dis, coupling_id, time, step),
          inflow_stab_(false)
    {
    }

   public:
    //! Evaluate Neumann traction 3 components
    void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

    //! Evaluate Neumann traction 6 components
    void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<6, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

    void evaluate_coupling_conditions_old_state(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

   protected:
    //! Do condition specific setup
    void do_condition_specific_setup() override;

    //! Initializes configurationmap
    void setup_configuration_map() override;

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
        ) override;

   private:
    //! Flag for inflow stabilization
    bool inflow_stab_;
  };

  /*!
  \brief
   */
  class LevelSetCouplingNavierSlip : public LevelSetCouplingBC
  {
   public:
    //! constructor
    explicit LevelSetCouplingNavierSlip(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,  ///< full discretization from which the cutter discretization is derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step          ///< time step
    );

   public:
    void evaluate_coupling_conditions(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

    void evaluate_coupling_conditions_old_state(Core::LinAlg::Matrix<3, 1>& ivel,
        Core::LinAlg::Matrix<3, 1>& itraction, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;

    void get_slip_coefficient(double& slipcoeff, const Core::LinAlg::Matrix<3, 1>& x,
        const Core::Conditions::Condition* cond) override;


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
        T1& projection_matrix,  ///< Laplace-Beltrami matrix for surface tension calculations
        int eid,                ///< element ID
        M1& funct,              ///< local shape function for Gauss Point (from fluid element)
        M2& derxy,   ///< local derivatives of shape function for Gauss Point (from fluid element)
        M3& normal,  ///< surface normal of cut element
        double& kappa_m,  ///< fluid sided weighting
        double& visc_m,   ///< fluid sided weighting
        double& visc_s    ///< slave sided dynamic viscosity
    )
    {
      eval_projection_matrix<distype>(projection_matrix, eid, funct, derxy, normal);
      evaluate_coupling_conditions(ivel, itraction, x, cond);

      if (robin_neumann_id_ >= 0)
      {
        // This is maybe not the most efficient implementation as we evaluate dynvisc as well as the
        // sliplenght twice evaluate interface traction (given by Neumann condition) Add this to the
        // veljump!
        double sliplength = 0.0;
        get_slip_coefficient(sliplength, x, cond);

        if (sliplength < 0.0) FOUR_C_THROW("The slip length can not be negative.");

        if (sliplength != 0.0)
        {
          double sl_visc_fac = sliplength / (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);
          V2 tmp_itraction(true);
          tmp_itraction.multiply_tn(projection_matrix, itraction);
          // Project this into tangential direction!!!
          ivel.update(sl_visc_fac, tmp_itraction, 1.0);
        }
        itraction.clear();
      }

      /*Here one could do a projection of ivel to only point in the tangential direction.
        This would enforce that no spurious velocities occur in the normal direction (i.e.
        no-penetration always enforced). However, these will occur in an XFSI. Thus a solution has
        to be found which can handle this the best.
      */

      if (forcetangvel_)
      {
        // We project in the normal direction
        Core::LinAlg::Matrix<3, 1> tmp_ivel(true);
        tmp_ivel.multiply_tn(
            projection_matrix, ivel);  // apply Projection matrix from the right. (u_0 * P^t)
        ivel.update(1.0, tmp_ivel, 0.0);
      }
    };

    /*!
     Return a smoothed/non-smoothed tangiential projection of the level set surface.
     */
    template <Core::FE::CellType distype, class T1, class M1, class M2, class M3>
    void eval_projection_matrix(T1& projection_matrix,  ///< Projection matrix
        int eid,                                        ///< element ID
        M1& funct,  ///< local shape function for Gauss Point (from fluid element)
        M2& derxy,  ///< local derivatives of shape function for Gauss Point (from fluid element)
        M3& normal  ///< surface normal of cut element
    )
    {
      // Properties of a projection matrix:
      //-------------------------------------------------------------------------
      // 1) P is singular (i.e. not of full rank, no inverse exists).
      // 2) P*P = P
      // 3) P^T = P
      // 4) a*P*a \geq 0 \forall a
      //-------------------------------------------------------------------------

      // number space dimensions for element
      const size_t nsd = Core::FE::dim<distype>;

      // number of nodes of element
      const size_t nen = Core::FE::num_nodes<distype>;

      // Should this be provided as well by the input?
      Core::Elements::Element* actele = cutter_dis_->g_element(eid);

      //   Non-smoothed projection matrix
      Core::LinAlg::Matrix<nsd, 1> gradphi;
      if (projtosurf_ == Inpar::XFEM::Proj_normal)
      {
        gradphi = normal;
      }
      else if (projtosurf_ == Inpar::XFEM::Proj_smoothed)
      {
        // smoothed normal at cutter element nodes, the Gaussian point lies in
        Core::LinAlg::SerialDenseMatrix esmoothedgradphi_test(nsd, nen);
        Core::LinAlg::Matrix<nsd, nen> esmoothedgradphi(esmoothedgradphi_test, View);
        XFEM::Utils::extract_quantity_at_element(esmoothedgradphi_test, actele,
            *gradphinp_smoothed_node_col_, *cutter_dis_, cutter_nds_phi_, nsd_);

        // Gradients @ GaussPoints
        gradphi.multiply(esmoothedgradphi, funct);
      }
      else if (projtosurf_ == Inpar::XFEM::Proj_normal_phi)
      {
        Core::LinAlg::SerialDenseMatrix ephi_test(nen, 1);
        Core::LinAlg::Matrix<nen, 1> ephi(ephi_test, View);
        XFEM::Utils::extract_quantity_at_element(
            ephi_test, actele, *cutter_phinp_col_, *cutter_dis_, cutter_nds_phi_, 1);

        // Gradients @ GaussPoints
        gradphi.multiply(derxy, ephi);
      }
      else if (projtosurf_ == Inpar::XFEM::Proj_normal_smoothed_comb)
      {
        // smoothed normal at cutter element nodes, the Gaussian point lies in
        Core::LinAlg::SerialDenseMatrix esmoothedgradphi_test(nsd, nen);
        Core::LinAlg::Matrix<nsd, nen> esmoothedgradphi(esmoothedgradphi_test, View);
        XFEM::Utils::extract_quantity_at_element(esmoothedgradphi_test, actele,
            *gradphinp_smoothed_node_col_, *cutter_dis_, cutter_nds_phi_, nsd_);

        // Gradients @ GaussPoints
        gradphi.multiply(esmoothedgradphi, funct);

        const double normgradphi = gradphi.norm2();
        if (normgradphi > 1e-9)  // 1e-9 is set to create a reasonable scaling.
          gradphi.scale(1.0 / normgradphi);
        else
          gradphi.put_scalar(0.0);  // This to catch the cases when gradphi \approx 0

        // normal_comb = alpha_n * normal + (1-alpha)*gradphi
        Core::LinAlg::Matrix<nsd, 1> normal_comb(true);
        double alpha_n = 0.3;
        normal_comb.update(alpha_n, normal, -(1.0 - alpha_n), gradphi);

        gradphi = normal_comb;
      }
      else
      {
        FOUR_C_THROW("This option for a projection matrix %d does not exist. \n", projtosurf_);
      }

      // Normalize the smoothed gradient
      const double normgradphi = gradphi.norm2();
      if (normgradphi > 1e-9)  // 1e-9 is set to create a reasonable scaling.
        gradphi.scale(1.0 / normgradphi);
      else
        gradphi.put_scalar(0.0);  // This to catch the cases when gradphi \approx 0

      setup_projection_matrix(projection_matrix, gradphi);
    }

   protected:
    void set_element_conditions() override;

    void set_element_specific_conditions(std::vector<Core::Conditions::Condition*>& cutterele_cond,
        const std::string& cond_name, const int& robin_id);

    void set_condition_specific_parameters() override;

    void get_condition_by_robin_id(const std::vector<Core::Conditions::Condition*>& mycond,
        const int coupling_id, std::vector<Core::Conditions::Condition*>& mynewcond);

    //! Initializes configurationmap
    void setup_configuration_map() override;

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
        ) override;

   private:
    bool forcetangvel_;
    bool is_constant_sliplength_;
    double sliplength_;

    // ID given the Robin Dirichlet/-Neumann conditions
    // (need to match the one given in the "MAIN condition")
    int robin_dirichlet_id_;
    int robin_neumann_id_;

    // Get the condition for the dirichlet and neumann condition associated with the Robin-condition
    std::vector<Core::Conditions::Condition*> cutterele_cond_robin_dirichlet_;
    std::vector<Core::Conditions::Condition*> cutterele_cond_robin_neumann_;


  };  // End LevelSetCouplingNavierSlip



  /// set levelset field by given vector
  void write_access_geometric_quantities(std::shared_ptr<Core::LinAlg::Vector<double>>& scalaraf,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>& smoothed_gradphiaf,
      std::shared_ptr<Core::LinAlg::Vector<double>>& curvatureaf);


  /// set material pointer for coupling slave side
  void get_interface_slave_material(
      Core::Elements::Element* actele, std::shared_ptr<Core::Mat::Material>& mat);

  template <Core::FE::CellType distype, class M1, class M2>
  void evaluate_curvature(double& icurvature,  ///< curvature to be computed
      int eid,                                 ///< element ID
      M1& funct,  ///< local shape function for Gauss Point (from fluid element)
      M2& derxy   ///< local derivatives of shape function for Gauss Point (from fluid element)
  )
  {
    // number space dimensions for element
    const size_t nsd = Core::FE::dim<distype>;

    // number of nodes of element
    const size_t nen = Core::FE::num_nodes<distype>;

    // smoothed normal at cutter element nodes, the Gaussian point lies in
    Core::LinAlg::SerialDenseMatrix esmoothedgradphi(nsd, nen);
    Core::LinAlg::SerialDenseMatrix esmoothedcurvature(nen, 1);

    Core::LinAlg::Matrix<nsd, nen> esmoothedgradphi_T(esmoothedgradphi, View);
    Core::LinAlg::Matrix<nen, 1> esmoothedcurvature_T(esmoothedcurvature, View);

    return;
  }

  template <Core::FE::CellType distype, class M1, class M2>
  void get_phi_at_gp(double& phi_gp,  ///< phi at gausspoint
      int eid,                        ///< element ID
      M1& funct,                      ///< local shape function for Gauss Point (from fluid element)
      M2& derxy  ///< local derivatives of shape function for Gauss Point (from fluid element)
  )
  {
    // number space dimensions for element
    //    const size_t nsd = Core::FE::dim<DISTYPE>;

    // number of nodes of element
    const size_t nen = Core::FE::num_nodes<distype>;

    // smoothed normal at cutter element nodes, the Gaussian point lies in
    Core::LinAlg::SerialDenseMatrix ephinp(nen, 1);
    Core::LinAlg::Matrix<nen, 1> ephinp_T(ephinp, View);

    phi_gp = funct.Dot(ephinp_T);
  }

  template <Core::FE::CellType distype, class M1, class M2, class M3, class M4>
  double interpolate_curvature(const M1& funct, const M2& derxy, const M3& esmoothedgradphi_T,
      const M4& esmoothedcurvature_T, const int numnode)
  {
    double icurvature = 0.0;
    return icurvature;
  }


  template <Core::FE::CellType distype, class T1, class M1, class M2>
  void eval_projection_matrix(
      T1& itraction_jump_matrix,  ///< Laplace-Beltrami matrix for surface tension calculations
      int eid,                    ///< element ID
      M1& funct,                  ///< local shape function for Gauss Point (from fluid element)
      M2& normal                  ///< surface normal of cut element
  )
  {
    // number space dimensions for element
    const size_t nsd = Core::FE::dim<distype>;

    Core::LinAlg::Matrix<nsd, nsd> p_matrix(false);
    Core::LinAlg::Matrix<nsd, nsd> p_smoothed_matrix(false);

    //  -----------------------------------------------------------
    //
    // HERE DEPENDING ON HOW THE PROJECTION SHOULD BE CALCULATED
    // different ways of the projection matrix should be tested.
    //
    // Might, only want to use matrix_mixed_smoothed option as this is the most promising!
    //   Check for stabilized Laplace-Beltrami option!
    //    Burman, Erik and Hansbo, Peter and Larson, Mats G
    //    A stabilized cut finite element method for partial differential equations on surfaces:
    //    The Laplace--Beltrami operator Computer Methods in Applied Mechanics and Engineering
    //       2015
    //
    //  -----------------------------------------------------------

    SetupConcatenatedProjectionMatrix(itraction_jump_matrix, p_matrix, p_smoothed_matrix);

    return;
  }


};  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
