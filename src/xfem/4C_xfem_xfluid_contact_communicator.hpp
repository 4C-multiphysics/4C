// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_XFLUID_CONTACT_COMMUNICATOR_HPP
#define FOUR_C_XFEM_XFLUID_CONTACT_COMMUNICATOR_HPP


#include "4C_config.hpp"

#include "4C_inpar_xfem.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  class Element;
}
namespace CONTACT
{
  class Element;
  class NitscheStrategyFsi;
  class NitscheStrategyFpi;
  class NitscheStrategy;
}  // namespace CONTACT

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class StructuralSurface;
  }
}  // namespace Discret

namespace Core::Elements
{
  class Element;
}


namespace Cut
{
  class CutWizard;
  class SideHandle;
  class VolumeCell;
  class Facet;
  class Element;
  class ElementHandle;
  class Side;
}  // namespace Cut


namespace XFEM
{
  class ConditionManager;
  class MeshCouplingFSI;
  class MeshCoupling;
  class MeshCouplingFPI;

  class XFluidContactComm
  {
   public:
    //! constructor
    explicit XFluidContactComm(CONTACT::NitscheStrategy& contact_strategy)
        : fluid_init_(false),
          ele_ptrs_already_setup_(false),
          cutwizard_(nullptr),
          fluiddis_(nullptr),
          condition_manager_(nullptr),
          mc_(std::vector<std::shared_ptr<XFEM::MeshCoupling>>()),
          mcfpi_ps_pf_(nullptr),
          mcidx_(0),
          isporo_(false),
          visc_stab_trace_estimate_(Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk),
          visc_stab_hk_(Inpar::XFEM::ViscStab_hk_ele_vol_div_by_max_ele_surf),
          nit_stab_gamma_(-1),
          is_pseudo_2d_(false),
          mass_conservation_scaling_(Inpar::XFEM::MassConservationScaling_only_visc),
          mass_conservation_combination_(Inpar::XFEM::MassConservationCombination_sum),
          dt_(-1),
          theta_(-1),
          parallel_(false),
          min_surf_id_(-1),
          min_mortar_id_(-1),
          so_surf_id_to_mortar_ele_(std::vector<CONTACT::Element*>()),
          mortar_id_to_so_surf_ele_(std::vector<Discret::Elements::StructuralSurface*>()),
          mortar_id_to_somc_(std::vector<int>()),
          mortar_id_to_sosid_(std::vector<int>()),
          extrapolate_to_zero_(false),
          my_sele_ids_(std::set<int>()),
          contact_ele_rowmap_fluidownerbased_(nullptr),
          contact_strategy_(contact_strategy),
          contact_strategy_fsi_(nullptr),
          contact_strategy_fpi_(nullptr)
    {
    }

    //! destructor
    virtual ~XFluidContactComm() = default;
    /// Initialize overall Fluid State (includes the Cut intersection information)
    void initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
        std::shared_ptr<Core::FE::Discretization> fluiddis,
        std::shared_ptr<XFEM::ConditionManager> condition_manager,
        std::shared_ptr<Teuchos::ParameterList> fluidparams);

    /// Reset overall Fluid State
    void reset_fluid_state()
    {
      fluid_init_ = false;
      cutwizard_ = nullptr;
      fluiddis_ = nullptr;
    }

    /// Get the FSI traction called from contact gausspoint
    double get_fsi_traction(Mortar::Element* ele,        // Mortar Element
        const Core::LinAlg::Matrix<3, 1>& xsi_parent,    // local coord in the parent element
        const Core::LinAlg::Matrix<2, 1>& xsi_boundary,  // local coord on the boundary element
        const Core::LinAlg::Matrix<3, 1>& normal,        // normal for projection
        bool& FSI_integrated,
        bool& gp_on_this_proc,  // for serial run
        double* poropressure = nullptr);

    /// Get the FSI traction called from contact gausspoint
    double get_fsi_traction(const Mortar::Element* ele,
        const Core::LinAlg::Matrix<2, 1>& xsi_parent,
        const Core::LinAlg::Matrix<1, 1>& xsi_boundary, const Core::LinAlg::Matrix<2, 1>& normal,
        bool& FSI_integrated,
        bool& gp_on_this_proc,  // for serial run
        double* poropressure = nullptr)
    {
      FOUR_C_THROW("no 2D xfsi with contact");
      return -1.0;
    }

    /// Get_Contact_State at gausspoint called from XFSI: return true-->evaluate FSI, return false
    /// -->evaluate NIT-Contact
    bool get_contact_state(int sid,  // Solid Surface Element
        std::string mcname,
        const Core::LinAlg::Matrix<2, 1>& xsi,  // local coord on the ele element
        const double& full_fsi_traction,        // stressfluid + penalty ...
        double& gap);

    /// Is this Structural surface registered in the Xfluid Contact Communicator
    bool is_registered_surface(const int soSurfId)
    {
      return (soSurfId >= min_surf_id_ &&
              soSurfId < ((int)so_surf_id_to_mortar_ele_.size() + min_surf_id_));
    }

    /// Get the contact element for this solid surface id
    CONTACT::Element* get_contact_ele(const int soSurfId)
    {
      return so_surf_id_to_mortar_ele_.at(soSurfId - min_surf_id_);
    }
    /// Get the solid surface element for the contact element id
    Discret::Elements::StructuralSurface* get_surf_ele(const int mortarId)
    {
      return mortar_id_to_so_surf_ele_.at(mortarId - min_mortar_id_);
    }

    /// Get the mesh coupling id for the contact element id
    int get_surf_mc(const int mortarId) { return mortar_id_to_somc_.at(mortarId - min_mortar_id_); }

    /// Get the solid surface element if for the contact element id
    int get_surf_sid(const int mortarId)
    {
      return mortar_id_to_sosid_.at(mortarId - min_mortar_id_);
    }

    /// Setup Interface element connection vectors based on points
    void setup_surf_ele_ptrs(Core::FE::Discretization& contact_interface_dis);

    /// Get element size of background mesh
    double get_h();

    /// Register Evaluation Processor rank for specific solid surface (is the fluid proc)
    void register_side_proc(int sid);

    /// Get the CUT integration points for this contact element (id)
    void get_cut_side_integration_points(
        int sid, Core::LinAlg::SerialDenseMatrix& coords, std::vector<double>& weights, int& npg);

    /// Finalize Map of interface element owners
    void fill_complete_sele_map();

    /// Rowmap of contact elements based on the fluid element owner
    std::shared_ptr<Epetra_Map>& get_contact_ele_row_map_f_ownerbased()
    {
      return contact_ele_rowmap_fluidownerbased_;
    }

    /// prepare_time_step
    void prepare_time_step();

    /// prepare_iteration_step
    void prepare_iteration_step();

    /// Register contact element for to use CUT integration points
    void register_contact_elementfor_higher_integration(int cid)
    {
      higher_contact_elements_.insert(cid);
    }

    /// Does this contact element use CUT integration points?
    bool higher_integrationfor_contact_element(int cid)
    {
      return (higher_contact_elements_comm_.find(cid) != higher_contact_elements_comm_.end());
    }

    /// Print a summary of contact GPs
    void print_summary_contact_gps();

    /// Increment gausspoint counter
    void inc_gp(int state) { ++sum_gps_[state]; }

    //! get distance when transition between FPSI and PSCI is started
    double get_fpi_pcontact_exchange_dist();

    //! ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is starte
    double get_fpi_pcontact_fullfraction();

   private:
    //! The the contact state at local coord of Element cele and compare to the fsi_traction,
    //! return true if contact is evaluated, return false if FSI is evaluated
    bool check_nitsche_contact_state(CONTACT::Element* cele,
        const Core::LinAlg::Matrix<2, 1>& xsi,  // local coord on the ele element
        const double& full_fsi_traction,        // stressfluid + penalty
        double& gap                             // gap
    );

    /// Get the fluid states at specific selexi
    void get_states(const int fluidele_id, const std::vector<int>& fluid_nds,
        const Discret::Elements::StructuralSurface* sele, const Core::LinAlg::Matrix<2, 1>& selexsi,
        const Core::LinAlg::Matrix<3, 1>& x, Core::Elements::Element*& fluidele,
        Core::LinAlg::SerialDenseMatrix& ele_xyze, std::vector<double>& velpres,
        std::vector<double>& disp, std::vector<double>& ivel, double& pres_m,
        Core::LinAlg::Matrix<3, 1>& vel_m, Core::LinAlg::Matrix<3, 1>& vel_s,
        Core::LinAlg::Matrix<3, 3>& vderxy_m, Core::LinAlg::Matrix<3, 1>& velpf_s);

    /// Get the Nitsche penalty parameter
    void get_penalty_param(Core::Elements::Element* fluidele, Cut::VolumeCell* volumecell,
        Core::LinAlg::SerialDenseMatrix& ele_xyze, const Core::LinAlg::Matrix<3, 1>& elenormal,
        double& penalty_fac, const Core::LinAlg::Matrix<3, 1>& vel_m);

    /// Get the Nitsche penalty parameter
    void get_penalty_param(Discret::Elements::StructuralSurface* sele, double& penalty_fac);

    /// Get the volumecell for local coord xsi on sele
    bool get_volumecell(Discret::Elements::StructuralSurface*& sele,
        Core::LinAlg::Matrix<2, 1>& xsi, Cut::SideHandle*& sidehandle, std::vector<int>& nds,
        int& eleid, Cut::VolumeCell*& volumecell, Core::LinAlg::Matrix<3, 1>& elenormal,
        Core::LinAlg::Matrix<3, 1>& x, bool& FSI_integrated, double& distance);

    /// Evaluate the distance of x the boundary of a side
    double distanceto_side(
        Core::LinAlg::Matrix<3, 1>& x, Cut::Side* side, Core::LinAlg::Matrix<3, 1>& closest_x);

    /// Find the next physical interface side to x
    Cut::Side* findnext_physical_side(Core::LinAlg::Matrix<3, 1>& x, Cut::Side* initSide,
        Cut::SideHandle*& sidehandle, Core::LinAlg::Matrix<2, 1>& newxsi, double& distance);

    /// Get list of potentiall next physical sides
    void update_physical_sides(Cut::Side* side, std::set<Cut::Side*>& performed_sides,
        std::set<Cut::Side*>& physical_sides);

    /// Get neighboring sides
    std::vector<Cut::Side*> get_new_neighboring_sides(
        Cut::Side* side, std::set<Cut::Side*>& performed_sides);

    /// Get next element
    Cut::Element* get_next_element(
        Cut::Element* ele, std::set<Cut::Element*>& performed_elements, int& lastid);

    /// access to contact/meshtying bridge
    CONTACT::NitscheStrategy& get_contact_strategy() { return contact_strategy_; }

    /// fluid state members initialized
    bool fluid_init_;
    /// Surface element pointers setup
    bool ele_ptrs_already_setup_;
    /// The XFluid CutWizard
    std::shared_ptr<Cut::CutWizard> cutwizard_;
    /// The Background Fluid discretization
    std::shared_ptr<Core::FE::Discretization> fluiddis_;
    /// The XFEM Condition Manager
    std::shared_ptr<XFEM::ConditionManager> condition_manager_;
    /// A list of all mesh coupling objects
    std::vector<std::shared_ptr<XFEM::MeshCoupling>> mc_;
    /// In case of poro, the fluid mesh coupling object
    std::shared_ptr<XFEM::MeshCouplingFPI> mcfpi_ps_pf_;
    /// Mesh coupling index
    int mcidx_;
    /// Is a poro problem
    bool isporo_;

    /// Viscous trace estimate for FSI-Nit-Pen
    Inpar::XFEM::ViscStabTraceEstimate visc_stab_trace_estimate_;
    /// h-definition for FSI-Nit-Pen
    Inpar::XFEM::ViscStabHk visc_stab_hk_;
    /// reference penalty parameter for FSI-Nit-Pen
    double nit_stab_gamma_;
    /// pseudo 2D flag for 2D simulation with one element in z-direction
    bool is_pseudo_2d_;
    /// mass conservation scaline on FSI-Nit-Pen
    Inpar::XFEM::MassConservationScaling mass_conservation_scaling_;
    /// How to combine the contribution on FSI-Nit-Pen
    Inpar::XFEM::MassConservationCombination mass_conservation_combination_;
    /// timestep
    double dt_;
    /// theta factor of OST-scheme
    double theta_;

    /// parallel computation (NumProc > 1), there are some check we can only do in serial
    bool parallel_;

    /// Min Structural Surface Id
    int min_surf_id_;
    /// Min Mortar Element Id
    int min_mortar_id_;
    /// Vector for translation of Structural Surface Id to Contact Element
    std::vector<CONTACT::Element*> so_surf_id_to_mortar_ele_;
    /// Vector for translation of Mortar Element Id to Structural Surface
    std::vector<Discret::Elements::StructuralSurface*> mortar_id_to_so_surf_ele_;
    /// Vector for translation of Mortar Element Id to Mesh Coupling Object Id
    std::vector<int> mortar_id_to_somc_;
    /// Vector for translation of Mortar Element Id to Structural Surface Id
    std::vector<int> mortar_id_to_sosid_;

    /// Fluid traction in extrapolation zone goes to zero
    bool extrapolate_to_zero_;

    // all sele which have a row fluid-element on this proc
    std::set<int> my_sele_ids_;
    /// contact ele romap - based on the background fluid element owners
    std::shared_ptr<Epetra_Map> contact_ele_rowmap_fluidownerbased_;

    /// The Contact Strategy
    CONTACT::NitscheStrategy& contact_strategy_;

    /// The Contact Strategy casted to fsi
    CONTACT::NitscheStrategyFsi* contact_strategy_fsi_;

    /// The Contact Strategy casted to fpi
    CONTACT::NitscheStrategyFpi* contact_strategy_fpi_;

    /// Contact Elements with increased number of GPs
    std::set<int> higher_contact_elements_;
    /// Contact Elements with increased number of GPs synchronized
    std::set<int> higher_contact_elements_comm_;

    /// Summarized Contact gps
    /// 0 ... Contact, 1 ... Contact_NoContactNoFSI, 2 ... Contact_NoContactFSI, 3 ...
    /// FSI_NoContact, 4 ... FSI_Contact
    std::vector<int> sum_gps_;

    /// store the last evaluated set of physical sides for solid side with id key
    std::pair<int, std::set<Cut::Side*>> last_physical_sides_;

    /// last computed element h measure with key fluidele id
    std::pair<int, double> last_ele_h_;
  };  // class XFluidContactComm
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
