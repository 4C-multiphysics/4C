// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_COUPLING_FPI_MESH_HPP
#define FOUR_C_XFEM_COUPLING_FPI_MESH_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_xfem_coupling_mesh.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class BoundaryCell;
}


namespace XFEM
{
  /*!
  specialized class for coupling of fluid with an porous media (Darcy Flow)
    \author ager
    \date 06/16
   */
  class MeshCouplingFPI : public MeshVolCoupling
  {
   public:
    enum CoupledField
    {
      ps_ps,
      ps_pf,
      pf_ps,
      pf_pf
    };

    //! constructor
    explicit MeshCouplingFPI(
        std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
        const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                       ///< discretization is identified
        std::shared_ptr<Core::FE::Discretization>&
            cond_dis,           ///< discretization from which cutter discretization can be derived
        const int coupling_id,  ///< id of composite of coupling conditions
        const double time,      ///< time
        const int step,         ///< time step
        MeshCouplingFPI::CoupledField field  ///< which field is coupled to the fluid
    );

    //! cutter dis should be loaded into the cut?
    bool cut_geometry() override { return (coupled_field_ == MeshCouplingFPI::ps_ps); }

    bool is_bj() const { return full_bj_; }

    void set_full_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> pres)
    {
      fulldispnp_ = dispnp;
      fullpres_ = pres;
    }

    void initialize_struct_pres_map(const Epetra_Map& pfmap, const Epetra_Map& psmap)
    {
      // We need to identify cutter dis dofs and pressure dofs on all processors for the whole
      // cutter_dis, as long as we don't have another ghosting strategy for the cutter_dis ...

      if (pfmap.NumMyElements() != psmap.NumMyElements())
        FOUR_C_THROW(
            "initialize_struct_pres_map: (pfmap->NumGlobalElements() != "
            "psmap->NumGlobalElements())!");

      std::shared_ptr<Epetra_Map> fullpfmap = Core::LinAlg::allreduce_e_map(pfmap);
      std::shared_ptr<Epetra_Map> fullpsmap = Core::LinAlg::allreduce_e_map(psmap);

      if (fullpfmap->NumMyElements() != fullpsmap->NumMyElements())
        FOUR_C_THROW(
            "initialize_struct_pres_map: (fullpfmap->NumGlobalElements() != "
            "fullpsmap->NumGlobalElements())!");

      for (int lid = 0; lid < fullpfmap->NumMyElements(); ++lid)
        lm_struct_x_lm_pres_[fullpsmap->GID(lid)] =
            fullpfmap->GID(lid) + 1;  // z-component of structure --> pressure
    }

    /*!
     Return prescribed velocities and traction vectors for a GNBC boundary condition.
     Also returns the projection matrix (to the plane of the surface) needed for the GNBC
     condition.
     */
    template <Core::FE::CellType distype, class T1, class M3>
    void evaluate_coupling_conditions(T1& projection_matrix,  ///< Projection matrix
        M3& normal                                            ///< surface normal of cut element
    )
    {
      eval_projection_matrix<distype>(projection_matrix, normal);
      return;
    };

    // finalize the interface true residual vector
    void complete_state_vectors() override;

    virtual void zero_state_vectors_fpi();

    void lift_drag(const int step, const double time) const override;

    void read_restart(const int step) override;

    // interface foces
    std::shared_ptr<Core::LinAlg::Vector<double>> i_true_residual() { return itrueresidual_; }

    // for assembly of fluid interface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> i_forcecol() { return iforcecol_; }

    //! Calculate the Porosity for this FaceElement Gausspoint
    double calc_porosity(
        Core::Elements::Element* ele, Core::LinAlg::Matrix<3, 1>& rst_slave, double& J);

    //! get distance when transition between FPSI and PSCI is started
    double get_fpi_pcontact_exchange_dist() { return fpsi_contact_hfraction_ * h_scaling_; }

    //! ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is starte
    double get_fpi_pcontact_fullfraction() { return fpsi_contact_fullpcfraction_; }

    /// Assign communicator to contact to mesh coupling object
    void assign_contact_comm(std::shared_ptr<XFEM::XFluidContactComm> xf_c_comm)
    {
      xf_c_comm_ = xf_c_comm;
    }

    /// Get communicator to contact
    std::shared_ptr<XFEM::XFluidContactComm> get_contact_comm()
    {
      if (xf_c_comm_ == nullptr)
        FOUR_C_THROW("Get_Contact_Comm: Xfluid_Contact_Communicator not assigned!");
      return xf_c_comm_;
    }

    /// Register this side on this proc
    void register_side_proc(int sid);

    /// Reconnect Parent Pointers
    void reconnect_parent_pointers();

    /// Initialize Fluid State
    bool initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
        std::shared_ptr<Core::FE::Discretization> fluiddis,
        std::shared_ptr<XFEM::ConditionManager> condition_manager,
        std::shared_ptr<Teuchos::ParameterList> fluidparams);

   private:
    void output(const int step, const double time, const bool write_restart_data) override;

   protected:
    //! Initializes configurationmap
    void setup_configuration_map() override;

    //! set the name of the coupling object based on the field coupling
    void set_coupling_name() override;

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

    void init_state_vectors() override;

    void do_condition_specific_setup() override;

    bool has_moving_interface() override { return true; }

    void set_condition_specific_parameters() override;

   private:
    //! Updates configurationmap for specific Gausspoint in the contact case
    virtual void update_configuration_map_gp_contact(double& kappa_m,  //< fluid sided weighting
        double& visc_m,                           //< master sided dynamic viscosity
        double& visc_s,                           //< slave sided dynamic viscosity
        double& density_m,                        //< master sided density
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
    );

    //! Calculate the Porosity for J,porosity pair on this FaceElement
    double calctr_permeability(Core::Elements::Element* ele, double& porosity, double& J);

    //! Compute Jacobian and extract PoroFluidPressure this FaceElement Gausspoint
    double compute_jacobianand_pressure(
        Core::Elements::Element* ele, Core::LinAlg::Matrix<3, 1>& rst_slave, double& pres);

    //------------------------------- vectors -----------------------------
    //! @name cutter-dis state vectors
    std::shared_ptr<Core::LinAlg::Vector<double>>
        itrueresidual_;  //! interface forces acting on the structural surface (= -iforcenp)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        iforcecol_;  //! interface forces acting on the fluid surface (column vector assembly)
    //@}

    //---------------------------------configuration--------------------------------
    //! @name type of poro field coupled to xfluid handled by this mesh coupling object!
    MeshCouplingFPI::CoupledField coupled_field_;

    //! Full BJ Variant or BJSaffmann?
    bool full_bj_;
    bool sub_tang_;

    std::shared_ptr<const Core::LinAlg::Vector<double>> fulldispnp_;
    std::shared_ptr<const Core::LinAlg::Vector<double>> fullpres_;

    //! map from structural x dof to pres dof of a node!
    std::map<int, int> lm_struct_x_lm_pres_;

    double bj_coeff_;

    //! flag for contact
    bool contact_;

    //! factor of element size, when transition between FPSI and PSCI is started! (if 1,
    //! interpolation within one element)
    double fpsi_contact_hfraction_;
    //! ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is started! (if 0, pure contact
    //! starts when gap is zero)
    double fpsi_contact_fullpcfraction_;

    //! Xfluid Contact Communicator
    std::shared_ptr<XFEM::XFluidContactComm> xf_c_comm_;

    //@}
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
