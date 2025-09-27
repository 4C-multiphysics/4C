// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfluid_contact_communicator.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_nitsche_strategy_fpi.hpp"
#include "4C_contact_nitsche_strategy_fsi.hpp"
#include "4C_cut_boundingbox.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_facet.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mortar_element.hpp"
#include "4C_solid_3D_ele_surface.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN

void XFEM::XFluidContactComm::initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
    std::shared_ptr<Core::FE::Discretization> fluiddis,
    std::shared_ptr<XFEM::ConditionManager> condition_manager,
    std::shared_ptr<Teuchos::ParameterList> fluidparams)
{
  if (fluiddis != nullptr && condition_manager != nullptr) fluid_init_ = true;
  cutwizard_ = cutwizard;
  fluiddis_ = fluiddis;
  condition_manager_ = condition_manager;

  parallel_ = (Core::Communication::num_mpi_ranks(fluiddis_->get_comm()) > 1);

  Teuchos::ParameterList& params_xf_stab = fluidparams->sublist("XFLUID DYNAMIC/STABILIZATION");

  visc_stab_trace_estimate_ = Teuchos::getIntegralValue<Inpar::XFEM::ViscStabTraceEstimate>(
      params_xf_stab, "VISC_STAB_TRACE_ESTIMATE");
  visc_stab_hk_ =
      Teuchos::getIntegralValue<Inpar::XFEM::ViscStabHk>(params_xf_stab, "VISC_STAB_HK");
  nit_stab_gamma_ = params_xf_stab.get<double>("NIT_STAB_FAC");
  is_pseudo_2d_ = params_xf_stab.get<bool>("IS_PSEUDO_2D");
  mass_conservation_scaling_ = Teuchos::getIntegralValue<Inpar::XFEM::MassConservationScaling>(
      params_xf_stab, "MASS_CONSERVATION_SCALING");
  mass_conservation_combination_ =
      Teuchos::getIntegralValue<Inpar::XFEM::MassConservationCombination>(
          params_xf_stab, "MASS_CONSERVATION_COMBO");


  Inpar::XFEM::ConvStabScaling ConvStabScaling =
      Teuchos::getIntegralValue<Inpar::XFEM::ConvStabScaling>(params_xf_stab, "CONV_STAB_SCALING");
  if (ConvStabScaling != Inpar::XFEM::ConvStabScaling_none)
    FOUR_C_THROW("ConvStabScaling not handled correctly!");

  extrapolate_to_zero_ = Global::Problem::instance()
                             ->x_fluid_dynamic_params()
                             .sublist("XFPSI MONOLITHIC")
                             .get<bool>("EXTRAPOLATE_TO_ZERO");

  if (extrapolate_to_zero_)
    std::cout << "==| The Fluid Stress Extrapolation is relaxed to zero! |==" << std::endl;

  dt_ = fluidparams->get<double>("time step size");
  theta_ = fluidparams->get<double>("theta");

  mc_.clear();
  std::shared_ptr<XFEM::MeshCouplingFSI> mcfsi = std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(
      condition_manager->get_mesh_coupling("XFEMSurfFSIMono"));
  if (mcfsi != nullptr) mc_.push_back(mcfsi);

  std::shared_ptr<XFEM::MeshCouplingFPI> mcfpi = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
      condition_manager->get_mesh_coupling("XFEMSurfFPIMono_ps_ps"));
  if (mcfpi != nullptr)
  {
    mc_.push_back(mcfpi);
    mcfpi_ps_pf_ = std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
        condition_manager->get_mesh_coupling("XFEMSurfFPIMono_ps_pf"));
    if (mcfpi_ps_pf_ == nullptr) FOUR_C_THROW("Couldn't find mcfpi_ps_pf_ object!");
  }
  else
  {
    mcfpi_ps_pf_ = nullptr;
  }

  if (mc_.size())
  {
    if (!Core::Communication::my_mpi_rank(fluiddis_->get_comm()))
      std::cout << "==| XFluidContactComm: Loaded " << mc_.size()
                << " Mesh Coupling Objects! |==" << std::endl;
  }
  else
    FOUR_C_THROW("Didn't find any mesh coupling object!");

  for (std::size_t mc = 0; mc < mc_.size(); ++mc)
  {
    if (mc_[mc]->get_averaging_strategy() != Inpar::XFEM::Xfluid_Sided &&
        mass_conservation_scaling_ != Inpar::XFEM::MassConservationScaling_only_visc)
      FOUR_C_THROW("The implementation does not what you expect!");
    else if (mc_[mc]->get_averaging_strategy() != Inpar::XFEM::Xfluid_Sided &&
             visc_stab_trace_estimate_ != Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue)
      FOUR_C_THROW("The implementation does not what you expect!");
  }

  my_sele_ids_.clear();

  prepare_iteration_step();

  last_physical_sides_ = std::pair<int, std::set<Cut::Side*>>(-2, std::set<Cut::Side*>());

  last_ele_h_ = std::pair<int, double>(-1, -1);

  print_summary_contact_gps();
}

double XFEM::XFluidContactComm::get_fsi_traction(Mortar::Element* ele,
    const Core::LinAlg::Matrix<3, 1>& xsi_parent, const Core::LinAlg::Matrix<2, 1>& xsi_boundary,
    const Core::LinAlg::Matrix<3, 1>& normal, bool& FSI_integrated,
    bool& gp_on_this_proc,  // for serial run
    double* poropressure)
{
  gp_on_this_proc = true;
  if (!fluid_init_) FOUR_C_THROW("Fluid not initialized!");
  if (ele == nullptr) FOUR_C_THROW("Contact Element not set!");

  if (parallel_)
  {
    if (contact_ele_rowmap_fluidownerbased_->lid(ele->id()) == -1)
    {
      gp_on_this_proc = false;
      return 0.0;
    }
  }

  Discret::Elements::SolidSurface* sele = get_surf_ele(ele->id());
  mcidx_ = get_surf_mc(ele->id());

  if (std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(mc_[mcidx_]) != nullptr)
    isporo_ = true;
  else
    isporo_ = false;

  // 1 // get CutSide--> BC --> VolumeCell --> FluidElement --> Check for Dofs
  Cut::SideHandle* sidehandle = cutwizard_->get_cut_side(get_surf_sid(ele->id()));
  if (sidehandle == nullptr) FOUR_C_THROW("Coundn't find Sidehandle for this structural surface!");

  std::vector<int> nds;
  int eleid;

  Cut::VolumeCell* volumecell = nullptr;
  static Core::LinAlg::Matrix<3, 1> elenormal(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<2, 1> new_xsi(xsi_boundary.data(), false);
  double distance = 0.0;
  if (!get_volumecell(sele, new_xsi, sidehandle, nds, eleid, volumecell, elenormal, x,
          FSI_integrated, distance))
  {
    gp_on_this_proc = false;
    return 0.0;
  }

  if (!volumecell)
  {
    std::cout << "==| You have a mail from XFluidContactComm: As I couldn't find an appropriate "
                 "fluid solution at your gausspoint I decided that the solution is 0.0! |=="
              << std::endl;
    return 0.0;
  }

  static std::vector<double> velpres;
  static std::vector<double> disp;
  static std::vector<double> ivel;
  Core::Elements::Element* fluidele = nullptr;
  Core::LinAlg::SerialDenseMatrix ele_xyze;
  double pres_m;
  static Core::LinAlg::Matrix<3, 1> vel_m;
  static Core::LinAlg::Matrix<3, 1> vel_s;
  static Core::LinAlg::Matrix<3, 1> velpf_s;
  static Core::LinAlg::Matrix<3, 3> vderxy_m;
  get_states(eleid, nds, sele, new_xsi, x, fluidele, ele_xyze, velpres, disp, ivel, pres_m, vel_m,
      vel_s, vderxy_m, velpf_s);

  double penalty_fac = 0.0;

  // To get the actual element normal we take the normal from the Contact, as this is the actual
  // one!
  elenormal.update(-1, normal, 0.0);

  if (mc_[mcidx_]->get_averaging_strategy() == Inpar::XFEM::Xfluid_Sided)
    get_penalty_param(fluidele, volumecell, ele_xyze, elenormal, penalty_fac, vel_m);
  else if (mc_[mcidx_]->get_averaging_strategy() == Inpar::XFEM::Embedded_Sided)
    get_penalty_param(sele, penalty_fac);
  else
    FOUR_C_THROW(
        "Your interface stress averaging strategy is not yet implemented for XFSCI and XFPSCI!");

  double visc_m;
  mc_[mcidx_]->get_viscosity_master(fluidele, visc_m);

  static double porosity = -1;
  if (isporo_)
  {
    Core::LinAlg::Matrix<3, 1> xsi3(new_xsi.data(), true);
    double J = 0;
    porosity =
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(mc_[mcidx_])->calc_porosity(sele, xsi3, J);
  }

  if (mc_[mcidx_]->get_averaging_strategy() == Inpar::XFEM::Xfluid_Sided)
  {
    if (poropressure && distance > 1e-10)
    {
      double scaling = (distance) / (mc_[mcidx_]->get_h());
      if (scaling < 1.0)
      {
        return -(*poropressure) * (scaling) +
               (1 - scaling) * XFEM::Utils::evaluate_full_traction(pres_m, vderxy_m, visc_m,
                                   penalty_fac, vel_m, vel_s, elenormal, elenormal, velpf_s,
                                   porosity);
      }
      else
      {
        return -(*poropressure);
      }
    }
    else
    {
      if (!extrapolate_to_zero_)
        return XFEM::Utils::evaluate_full_traction(pres_m, vderxy_m, visc_m, penalty_fac, vel_m,
            vel_s, elenormal, elenormal, velpf_s, porosity);
      else
      {
        double scaling = (distance) / (mc_[mcidx_]->get_h());
        if (scaling > 1.0) scaling = 1;
        return XFEM::Utils::evaluate_full_traction(pres_m, vderxy_m, visc_m, penalty_fac, vel_m,
                   vel_s, elenormal, elenormal, velpf_s, porosity) *
               (1 - scaling);
      }
    }
  }
  else
  {
    FOUR_C_THROW("Other Nitsche FSI variant than fluid-sided weighting not implemented yet!");
    return 0.0;
  }
}

bool XFEM::XFluidContactComm::check_nitsche_contact_state(CONTACT::Element* cele,
    const Core::LinAlg::Matrix<2, 1>& xsi,  // local coord on the ele element
    const double& full_fsi_traction,        // stressfluid + penalty
    double& gap                             // gap
)
{
  if (contact_strategy_fsi_)
    return contact_strategy_fsi_->check_nitsche_contact_state(cele, xsi, full_fsi_traction, gap);
  else if (contact_strategy_fpi_)
    return contact_strategy_fpi_->check_nitsche_contact_state(cele, xsi, full_fsi_traction, gap);
  else
    FOUR_C_THROW("check_nitsche_contact_state: Not adequate contact strategy is assigned!");
  return false;  // dummy to make compiler happy :)
}

bool XFEM::XFluidContactComm::get_contact_state(int sid,        // Solid Surface Element
    std::string mcname, const Core::LinAlg::Matrix<2, 1>& xsi,  // local coord on the ele element
    const double& full_fsi_traction,                            // stressfluid + penalty ...
    double& gap)
{
  if (!ele_ptrs_already_setup_) return true;

  int startgid = condition_manager_->get_mesh_coupling_start_gid(
      condition_manager_->get_coupling_index(mcname));
  CONTACT::Element* cele = get_contact_ele(sid + startgid);
  if (cele)
    return check_nitsche_contact_state(cele, xsi, full_fsi_traction, gap);
  else
  {
    gap = 1e12;
    return true;
  }
}

void XFEM::XFluidContactComm::get_states(const int fluidele_id, const std::vector<int>& fluid_nds,
    const Discret::Elements::SolidSurface* sele, const Core::LinAlg::Matrix<2, 1>& selexsi,
    const Core::LinAlg::Matrix<3, 1>& x, Core::Elements::Element*& fluidele,
    Core::LinAlg::SerialDenseMatrix& ele_xyze, std::vector<double>& velpres,
    std::vector<double>& disp, std::vector<double>& ivel, double& pres_m,
    Core::LinAlg::Matrix<3, 1>& vel_m, Core::LinAlg::Matrix<3, 1>& vel_s,
    Core::LinAlg::Matrix<3, 3>& vderxy_m, Core::LinAlg::Matrix<3, 1>& velpf_s)
{
  fluidele = fluiddis_->g_element(fluidele_id);
  Discret::Elements::Fluid* ffluidele = dynamic_cast<Discret::Elements::Fluid*>(fluidele);
  // 1 // get element states
  {
    Core::Elements::LocationArray la_f(1);
    fluidele->location_vector(*fluiddis_, fluid_nds, la_f);
    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        fluiddis_->get_state("velaf");
    velpres = Core::FE::extract_values(*matrix_state, la_f[0].lm_);

    std::vector<int> lmdisp;
    lmdisp.resize(fluid_nds.size() * 3);
    if (!ffluidele) FOUR_C_THROW("Cast to Fluidelement failed");
    if (ffluidele->is_ale())
    {
      for (std::size_t n = 0; n < fluid_nds.size(); ++n)
        for (int dof = 0; dof < 3; ++dof) lmdisp[n * 3 + dof] = la_f[0].lm_[n * 4 + dof];
      std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state_disp =
          fluiddis_->get_state("dispnp");
      disp = Core::FE::extract_values(*matrix_state_disp, lmdisp);
    }
  }
  {
    Core::Elements::LocationArray la_s(1);
    sele->location_vector(*mc_[mcidx_]->get_cutter_dis(), la_s);
    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        mc_[mcidx_]->get_cutter_dis()->get_state("ivelnp");
    ivel = Core::FE::extract_values(*matrix_state, la_s[0].lm_);
  }
  static std::vector<double> ipfvel;
  if (isporo_)
  {
    Core::Elements::LocationArray la_s(1);
    sele->location_vector(*mcfpi_ps_pf_->get_cutter_dis(), la_s);
    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        mcfpi_ps_pf_->get_cutter_dis()->get_state("ivelnp");
    ipfvel = Core::FE::extract_values(*matrix_state, la_s[0].lm_);
  }

  // 2 // get element xyze
  /// element coordinates in matrix
  ele_xyze.shape(3, fluidele->num_node());
  for (int i = 0; i < fluidele->num_node(); ++i)
  {
    for (int j = 0; j < 3; j++)
    {
      if (ffluidele->is_ale())
        ele_xyze(j, i) = fluidele->nodes()[i]->x()[j] + disp[i * 3 + j];
      else
        ele_xyze(j, i) = fluidele->nodes()[i]->x()[j];
    }
  }

  // 3 // get quantities in gp
  {
    Core::LinAlg::Matrix<3, 1> fluidele_xsi(Core::LinAlg::Initialization::zero);
    if (fluidele->shape() == Core::FE::CellType::hex8)
    {
      Core::LinAlg::Matrix<3, 8> xyze(ele_xyze.values(), true);
      // find element local position of gauss point
      std::shared_ptr<Cut::Position> pos =
          Cut::PositionFactory::build_position<3, Core::FE::CellType::hex8>(xyze, x);
      if (!pos->compute(1e-1))  // if we are a little bit outside of the element we don't care ...
      {
        FOUR_C_THROW("Couldn't compute local coordinate for fluid element!");
      }
      pos->local_coordinates(fluidele_xsi);

      static Core::LinAlg::Matrix<3, 3> xji;
      static Core::LinAlg::Matrix<3, 3> xjm;
      static Core::LinAlg::Matrix<3, 8> deriv;
      static Core::LinAlg::Matrix<8, 1> funct;
      static Core::LinAlg::Matrix<3, 8> derxy;

      // evaluate shape functions
      Core::FE::shape_function<Core::FE::CellType::hex8>(fluidele_xsi, funct);

      // evaluate the derivatives of shape functions
      Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(fluidele_xsi, deriv);
      xjm.multiply_nt(deriv, xyze);
      // double det = xji.invert(xjm); //if we need this at some point
      xji.invert(xjm);

      // compute global first derivatives
      derxy.multiply(xji, deriv);

      static Core::LinAlg::Matrix<3, 8> vel;
      static Core::LinAlg::Matrix<8, 1> pres;
      for (int n = 0; n < fluidele->num_node(); ++n)
      {
        pres(n, 0) = velpres[n * 4 + 3];
        for (int dof = 0; dof < 3; ++dof)
        {
          vel(dof, n) = velpres[n * 4 + dof];
        }
      }
      pres_m = pres.dot(funct);
      vel_m.multiply(1., vel, funct, 0.);
      vderxy_m.multiply_nt(vel, derxy);
    }
    else
      FOUR_C_THROW("fluidele is not hex8!");
  }

  // 4 // evaluate slave velocity at guasspoint
  if (sele->shape() == Core::FE::CellType::quad4)
  {
    static Core::LinAlg::Matrix<3, 4> vels;
    static Core::LinAlg::Matrix<3, 4> velpfs;
    for (int n = 0; n < sele->num_node(); ++n)
    {
      for (int dof = 0; dof < 3; ++dof)
      {
        vels(dof, n) = ivel[n * 3 + dof];
        if (isporo_) velpfs(dof, n) = ipfvel[n * 3 + dof];
      }
    }

    const int numnodes = Core::FE::num_nodes(Core::FE::CellType::quad4);
    static Core::LinAlg::Matrix<numnodes, 1> funct(Core::LinAlg::Initialization::uninitialized);
    Core::FE::shape_function_2d(funct, selexsi(0), selexsi(1), Core::FE::CellType::quad4);
    vel_s.multiply(vels, funct);
    if (isporo_) velpf_s.multiply(velpfs, funct);
  }
  else
    FOUR_C_THROW("Your Slave Element is not a quad4?!");

  return;
}

void XFEM::XFluidContactComm::get_penalty_param(Core::Elements::Element* fluidele,
    Cut::VolumeCell* volumecell, Core::LinAlg::SerialDenseMatrix& ele_xyze,
    const Core::LinAlg::Matrix<3, 1>& elenormal, double& penalty_fac,
    const Core::LinAlg::Matrix<3, 1>& vel_m)
{
  double h_k;
  double inv_h_k;
  if (last_ele_h_.first != fluidele->id() ||
      visc_stab_hk_ != Inpar::XFEM::ViscStab_hk_ele_vol_div_by_max_ele_surf)
  {
    // 1 // Get Boundary Cells and Gausspoints of this Boundarycells for this fluid element!
    std::map<int, std::vector<Cut::BoundaryCell*>> bcells;
    std::map<int, std::vector<Core::FE::GaussIntegration>> bintpoints;

    Cut::ElementHandle* cele = cutwizard_->get_element(fluidele);
    if (cele == nullptr) FOUR_C_THROW("Couldn't find cut element for ele {}", fluidele->id());

    std::vector<Cut::plain_volumecell_set> cell_sets;
    {
      std::vector<std::vector<int>> nds_sets;
      std::vector<std::vector<Core::FE::GaussIntegration>> intpoints_sets;
      cele->get_cell_sets_dof_sets_gauss_points(cell_sets, nds_sets, intpoints_sets, false);
    }

    // find the right cell_set
    std::vector<Cut::plain_volumecell_set>::iterator sit = cell_sets.end();
    // if we have just one dof in this element this isn't a loop
    for (std::vector<Cut::plain_volumecell_set>::iterator s = cell_sets.begin();
        s != cell_sets.end(); s++)
    {
      Cut::plain_volumecell_set& cells = *s;
      for (Cut::plain_volumecell_set::iterator i = cells.begin(); i != cells.end(); ++i)
      {
        if (volumecell == (*i))
        {
          sit = s;
          break;
        }
        if (sit != cell_sets.end()) break;
      }
    }

    if (sit == cell_sets.end()) FOUR_C_THROW("Couldn't identify a cell set!");

    Cut::plain_volumecell_set& cells = *sit;
    std::map<int, std::vector<Cut::BoundaryCell*>> element_bcells;
    for (Cut::plain_volumecell_set::iterator i = cells.begin(); i != cells.end(); ++i)
    {
      Cut::VolumeCell* vc = *i;
      vc->get_boundary_cells_to_be_integrated(element_bcells);
    }

    for (std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator bc = element_bcells.begin();
        bc != element_bcells.end(); ++bc)
    {
      std::vector<Cut::BoundaryCell*>& bc_new = bcells[bc->first];
      bc_new.clear();
      std::copy(bc->second.begin(), bc->second.end(), std::inserter(bc_new, bc_new.end()));
    }

    if (bcells.size() > 0)
    {
      // get boundary cell Gaussian points
      cele->boundary_cell_gauss_points_lin(bcells, bintpoints, cutwizard_->get_bc_cubaturedegree());
    }
    else
    {
      std::cout << "I didn't identify any boundary cells, this happens if I'm outside of fluid "
                   "solution elements! --> penalty ~ nue/h = nue/V*O = 0"
                << std::endl;
      penalty_fac = 0.0;
      return;
    }
    if (fluidele->shape() != Core::FE::CellType::hex8) FOUR_C_THROW("Add hex8 shapes here!");

    h_k = XFEM::Utils::compute_char_ele_length<Core::FE::CellType::hex8>(
        fluidele, ele_xyze, *condition_manager_, cells, bcells, bintpoints, visc_stab_hk_);

    inv_h_k = 1.0 / h_k;
    last_ele_h_ = std::pair<int, double>(fluidele->id(), h_k);
  }
  else
  {
    h_k = last_ele_h_.second;
    inv_h_k = 1. / h_k;
  }

  // 3 // Compute Penalty Param
  double dummy;
  double kappa_m = 1.0;
  double kappa_s = 1.0 - kappa_m;
  double visc_stab_fac = 0.0;
  mc_[mcidx_]->get_visc_penalty_stabfac(fluidele, nullptr, kappa_m, kappa_s, inv_h_k, visc_stab_fac,
      dummy, nit_stab_gamma_, nit_stab_gamma_, is_pseudo_2d_, visc_stab_trace_estimate_);

  std::shared_ptr<Core::Mat::Material> mat;
  XFEM::Utils::get_volume_cell_material(fluidele, mat);
  const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());
  if (actmat == nullptr) FOUR_C_THROW("Cast of Fluidmat failed!");

  XFEM::Utils::nit_compute_full_penalty_stabfac(
      penalty_fac,  ///< to be filled: full Nitsche's penalty term scaling (viscous+convective part)
      elenormal, h_k,
      kappa_m,  // weights (only existing for Nitsche currently!!)
      kappa_s,  // weights (only existing for Nitsche currently!!)
      vel_m, vel_m,
      visc_stab_fac,  ///< Nitsche's viscous scaling part of penalty term
      theta_ * dt_, false, actmat->density(), actmat->density(), mass_conservation_scaling_,
      mass_conservation_combination_, nit_stab_gamma_, Inpar::XFEM::ConvStabScaling_none,
      Inpar::XFEM::XFF_ConvStabScaling_none, false, false);
  return;
}

void XFEM::XFluidContactComm::get_penalty_param(
    Discret::Elements::SolidSurface* sele, double& penalty_fac)
{
  penalty_fac = nit_stab_gamma_ *
                std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(mc_[mcidx_])->get_time_fac() *
                std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(mc_[mcidx_])
                    ->get_estimate_nitsche_trace_max_eigenvalue(sele);
  return;
}

void XFEM::XFluidContactComm::setup_surf_ele_ptrs(Core::FE::Discretization& contact_interface_dis)
{
  if (ele_ptrs_already_setup_) return;

  if (!fluid_init_) FOUR_C_THROW("fluid not initialized");

  if (!mc_.size()) FOUR_C_THROW("SetupSurfElePtrs: Don't have any mesh coupling objects ....!");

  int startgid = condition_manager_->get_mesh_coupling_start_gid(
      condition_manager_->get_coupling_index(mc_[0]->get_name()));
  min_surf_id_ = mc_[0]->get_cutter_dis()->element_col_map()->min_all_gid() + startgid;
  int max_surf_gid = mc_[0]->get_cutter_dis()->element_col_map()->max_all_gid() + startgid;
  for (std::size_t mc = 1; mc < mc_.size(); ++mc)
  {
    int startgid = condition_manager_->get_mesh_coupling_start_gid(
        condition_manager_->get_coupling_index(mc_[mc]->get_name()));
    if (min_surf_id_ > mc_[mc]->get_cutter_dis()->element_col_map()->min_all_gid() + startgid)
      min_surf_id_ = mc_[mc]->get_cutter_dis()->element_col_map()->min_all_gid() + startgid;
    if (max_surf_gid < mc_[mc]->get_cutter_dis()->element_col_map()->max_all_gid() + startgid)
      max_surf_gid = mc_[mc]->get_cutter_dis()->element_col_map()->max_all_gid() + startgid;
  }
  if (mcfpi_ps_pf_ != nullptr)
  {
    int startgid = condition_manager_->get_mesh_coupling_start_gid(
        condition_manager_->get_coupling_index(mcfpi_ps_pf_->get_name()));
    if (min_surf_id_ > mcfpi_ps_pf_->get_cutter_dis()->element_col_map()->min_all_gid() + startgid)
      min_surf_id_ = mcfpi_ps_pf_->get_cutter_dis()->element_col_map()->min_all_gid() + startgid;
    if (max_surf_gid < mcfpi_ps_pf_->get_cutter_dis()->element_col_map()->max_all_gid() + startgid)
      max_surf_gid = mcfpi_ps_pf_->get_cutter_dis()->element_col_map()->max_all_gid() + startgid;
  }
  so_surf_id_to_mortar_ele_.resize(max_surf_gid - min_surf_id_ + 1, nullptr);
  min_mortar_id_ = contact_interface_dis.element_col_map()->min_all_gid();
  const int max_mortar_gid = contact_interface_dis.element_col_map()->max_all_gid();
  mortar_id_to_so_surf_ele_.resize(max_mortar_gid - min_mortar_id_ + 1, nullptr);
  mortar_id_to_somc_.resize(max_mortar_gid - min_mortar_id_ + 1, -1);
  mortar_id_to_sosid_.resize(max_mortar_gid - min_mortar_id_ + 1, -1);

  for (int i = 0; i < contact_interface_dis.element_col_map()->num_my_elements(); ++i)
  {
    CONTACT::Element* cele =
        dynamic_cast<CONTACT::Element*>(contact_interface_dis.l_col_element(i));
    if (!cele) FOUR_C_THROW("no contact element or no element at all");
    const int c_parent_id = cele->parent_element_id();
    const int c_parent_surf = cele->face_parent_number();

    for (std::size_t mc = 0; mc < mc_.size(); ++mc)
    {
      for (int j = 0; j < mc_[mc]->get_cutter_dis()->num_my_col_elements(); ++j)
      {
        int startgid = condition_manager_->get_mesh_coupling_start_gid(
            condition_manager_->get_coupling_index(mc_[mc]->get_name()));
        Discret::Elements::SolidSurface* fele = dynamic_cast<Discret::Elements::SolidSurface*>(
            mc_[mc]->get_cutter_dis()->l_col_element(j));
        if (!fele) FOUR_C_THROW("no face element or no element at all");
        const int f_parent_id = fele->parent_element_id();
        const int f_parent_surf = fele->face_parent_number();
        if (c_parent_id == f_parent_id && c_parent_surf == f_parent_surf)
        {
          mortar_id_to_so_surf_ele_[cele->id() - min_mortar_id_] = fele;
          mortar_id_to_somc_[cele->id() - min_mortar_id_] = mc;
          mortar_id_to_sosid_[cele->id() - min_mortar_id_] = fele->id() + startgid;

          so_surf_id_to_mortar_ele_[fele->id() - min_surf_id_ + startgid] = cele;
        }
      }
    }
    if (mcfpi_ps_pf_ != nullptr)
    {
      for (int j = 0; j < mcfpi_ps_pf_->get_cutter_dis()->num_my_col_elements(); ++j)
      {
        int startgid = condition_manager_->get_mesh_coupling_start_gid(
            condition_manager_->get_coupling_index(mcfpi_ps_pf_->get_name()));
        Discret::Elements::SolidSurface* fele = dynamic_cast<Discret::Elements::SolidSurface*>(
            mcfpi_ps_pf_->get_cutter_dis()->l_col_element(j));
        if (!fele) FOUR_C_THROW("no face element or no element at all");
        const int f_parent_id = fele->parent_element_id();
        const int f_parent_surf = fele->face_parent_number();

        if (c_parent_id == f_parent_id && c_parent_surf == f_parent_surf)
        {
          // mortarId_to_soSurf_ele_[cele->Id()-min_mortar_id_]=fele; //we dont need this connection
          // as this is already available from ps_ps
          so_surf_id_to_mortar_ele_[fele->id() - min_surf_id_ + startgid] = cele;
        }
      }
    }
  }
  ele_ptrs_already_setup_ = true;

  contact_strategy_fsi_ =
      dynamic_cast<CONTACT::NitscheStrategyFsi*>(&contact_strategy_);  // might be nullptr
  contact_strategy_fpi_ =
      dynamic_cast<CONTACT::NitscheStrategyFpi*>(&contact_strategy_);  // might be nullptr
}

bool XFEM::XFluidContactComm::get_volumecell(Discret::Elements::SolidSurface*& sele,
    Core::LinAlg::Matrix<2, 1>& xsi, Cut::SideHandle*& sidehandle, std::vector<int>& nds,
    int& eleid, Cut::VolumeCell*& volumecell, Core::LinAlg::Matrix<3, 1>& elenormal,
    Core::LinAlg::Matrix<3, 1>& x, bool& FSI_integrated, double& distance)
{
  distance = 0.0;
  FSI_integrated = true;
  // 1 // Compute global coord x
  volumecell = nullptr;
  if (sele->shape() == Core::FE::CellType::quad4)
  {
    const int numnodes = Core::FE::num_nodes(Core::FE::CellType::quad4);

    Core::LinAlg::SerialDenseMatrix xyze_m;
    Core::LinAlg::Matrix<numnodes, 1> funct(Core::LinAlg::Initialization::uninitialized);

    sidehandle->coordinates(xyze_m);
    Core::LinAlg::Matrix<3, numnodes> xyze(xyze_m.values(), true);
    Core::FE::shape_function_2d(funct, xsi(0), xsi(1), Core::FE::CellType::quad4);
    x.multiply(xyze, funct);
  }
  else
    FOUR_C_THROW("GetFacet: Your solid face is not a quad4, please add your element type here!");

  // 2 //Identify Subside
  Cut::Facet* facet = nullptr;

  Cut::plain_side_set subsides;
  sidehandle->collect_sides(subsides);
  int found_side = -1;
  static Core::LinAlg::Matrix<3, 1> tmpxsi(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> tmpxsi_tmp(Core::LinAlg::Initialization::zero);
  for (std::size_t ss = 0; ss < subsides.size(); ++ss)
  {
    Cut::Side* s = subsides[ss];
    if (s->local_coordinates(x, tmpxsi_tmp, true, 1e-10))
    {
      found_side = ss;
      tmpxsi = tmpxsi_tmp;
      Core::LinAlg::Matrix<2, 1> tmp2xsi(tmpxsi.data(), true);
      s->normal(tmp2xsi, elenormal, true);
      elenormal.scale(-1.0);          // flip direction
      if (fabs(tmpxsi(2, 0)) < 1e-1)  // do not search for better candidates anymore
        break;
    }
  }
  if (found_side == -1)  // do the same thing with reduced tolerance
  {
    for (std::size_t ss = 0; ss < subsides.size(); ++ss)
    {
      Cut::Side* s = subsides[ss];
      if (s->local_coordinates(x, tmpxsi_tmp, true, 1e-6))
      {
        found_side = ss;
        tmpxsi = tmpxsi_tmp;
        Core::LinAlg::Matrix<2, 1> tmp2xsi(tmpxsi.data(), true);
        s->normal(tmp2xsi, elenormal, true);
        elenormal.scale(-1.0);          // flip direction
        if (fabs(tmpxsi(2, 0)) < 1e-1)  // do not search for better candidates anymore
          break;
      }
    }
  }
  if (found_side == -1)
  {
    FOUR_C_THROW("Couldn't identify side!");
  }
  else
  {
    std::vector<Cut::Facet*> facets = subsides[found_side]->facets();

    Cut::Side* side = subsides[found_side];
    // Handle here unphysical sides ...
    if (sidehandle->isunphysical_sub_side(subsides[found_side]))
    {
      FSI_integrated = false;

      // Find Closest Point on physical boundary ...
      side = findnext_physical_side(x, subsides[found_side], sidehandle, xsi, distance);
      side->normal(xsi, elenormal, true);
      elenormal.scale(-1.0);  // flip direction
      sele =
          dynamic_cast<Discret::Elements::SolidSurface*>(condition_manager_->get_side(side->id()));
      if (!sele) FOUR_C_THROW("Couldn't Identify new sele {}", side->id());
      facets = side->facets();
    }

    if (facets.size() == 1 && !parallel_)
    {
      facet = facets[0];
    }
    else if (facets.size() > 1 || parallel_)
    {
      // a //simple case --> still all facets belong to the same volumecells
      bool SameVCs = true;
      for (std::vector<Cut::Facet*>::iterator fit = facets.begin(); fit != facets.end(); ++fit)
      {
        if (facets[0]->cells().size() != (*fit)->cells().size())  // simplest check
        {
          SameVCs = false;
          break;
        }
        else
        {
          for (std::size_t checkcell = 0; checkcell < (*fit)->cells().size(); ++checkcell)
          {
            if (facets[0]->cells()[checkcell] != (*fit)->cells()[checkcell])
            {
              SameVCs = false;
              break;
            }
          }
        }
      }

      if (SameVCs && !parallel_)  // we can take any facet ...
      {
        facet = facets[0];
      }
      else  // b // more complex need to really identify the facet ...
      {
        for (std::vector<Cut::Facet*>::iterator fit = facets.begin(); fit != facets.end(); ++fit)
        {
          Cut::Facet* afacet = *fit;
          std::vector<std::vector<Cut::Point*>> triangulation;
          if (afacet->triangulation().size())
            triangulation = afacet->triangulation();
          else if (afacet->get_split_cells().size())
            triangulation = afacet->get_split_cells();
          else if (afacet->num_points() == 3)
            triangulation.push_back(afacet->points());
          else
          {
            std::cout << "==| Warning from of your friendly XFluidContactComm: I have untriagled "
                         "faces |=="
                      << std::endl;
          }

          for (std::size_t tri = 0; tri < triangulation.size(); ++tri)
          {
            if (triangulation[tri].size() != 3)
              FOUR_C_THROW("Triangulation with another number of points than 3 ({})?",
                  triangulation[tri].size());

            // Compute local coords and take first possible facet ...
            Core::LinAlg::Matrix<3, 3> xyzf;
            triangulation[tri][0]->coordinates(xyzf.data());
            triangulation[tri][1]->coordinates(xyzf.data() + 3);
            triangulation[tri][2]->coordinates(xyzf.data() + 6);

            std::shared_ptr<Cut::Position> pos =
                Cut::PositionFactory::build_position<3, Core::FE::CellType::tri3>(xyzf, x);
            bool success = pos->compute(1e-6, true);
            if (success)
            {
              pos->local_coordinates(tmpxsi);
              if (fabs(tmpxsi(2, 0)) > 1e-3)
                FOUR_C_THROW("To far away from this facet {}!", tmpxsi(2, 0));
              facet = afacet;
              break;
            }
          }
          if (facet) break;
        }

        if (!facet && parallel_)
        {
          return false;  // in parallel this is ok
        }
        else if (!facet)
        {
          FOUR_C_THROW("Couldn't identify facet!");
        }
      }
    }
    else
    {
      if (parallel_)
      {
        return false;  // in parallel this is ok
      }
      FOUR_C_THROW("Side has no facets but is physical!");
    }
  }

  // 3 // Get Volumecells
  if (!volumecell)
  {
    for (std::size_t vc = 0; vc < facet->cells().size(); ++vc)
    {
      if (facet->cells()[vc]->position() == Cut::Point::outside)
      {
        if (volumecell)
        {
          FOUR_C_THROW("Facet has at least two volumecells which are outside!");
        }
        volumecell = facet->cells()[vc];
        if (parallel_)
        {
          if (fluiddis_->element_row_map()->lid(volumecell->parent_element()->id()) ==
              -1)  // fluidele not owned by this proc
          {
            return false;
          }
        }
      }
    }
    if (!volumecell) FOUR_C_THROW("Facet has no volumecell which is outside?");
  }

  nds = volumecell->nodal_dof_set();
  eleid = volumecell->get_parent_element_id();

  return true;
}

Cut::Side* XFEM::XFluidContactComm::findnext_physical_side(Core::LinAlg::Matrix<3, 1>& x,
    Cut::Side* initSide, Cut::SideHandle*& sidehandle, Core::LinAlg::Matrix<2, 1>& newxsi,
    double& distance)
{
  std::set<Cut::Side*> performed_sides;
  std::set<Cut::Side*> physical_sides;
  if (last_physical_sides_.first != initSide->id())
  {
    update_physical_sides(initSide, performed_sides, physical_sides);
    last_physical_sides_ = std::pair<int, std::set<Cut::Side*>>(initSide->id(), physical_sides);
  }
  else
    physical_sides = last_physical_sides_.second;

  distance = 1e200;
  Core::LinAlg::Matrix<3, 1> newx(Core::LinAlg::Initialization::zero);
  Cut::Side* newSide = nullptr;

  for (std::set<Cut::Side*>::iterator psit = physical_sides.begin(); psit != physical_sides.end();
      ++psit)
  {
    static Core::LinAlg::Matrix<3, 1> tmpx(Core::LinAlg::Initialization::zero);
    double tmpdistance = distanceto_side(x, *psit, tmpx);
    if (distance > tmpdistance)
    {
      distance = tmpdistance;
      newx.update(1.0, tmpx, 0.0);
      newSide = *psit;
    }
  }

  if (!newSide)
  {
    FOUR_C_THROW("Couldn't identify a new side (number of identified physical sides: {})!",
        physical_sides.size());
  }

  // Update global position...
  x.update(1.0, newx, 0.0);

  // compute again surface local coords ...
  sidehandle = cutwizard_->get_cut_side(newSide->id());
  if (!sidehandle)
  {
    std::cout << "The Side pointer is " << newSide << std::endl;
    std::cout << "Couldn't get Sidehandle for side " << newSide->id() << std::endl;
    // newSide->print();
    FOUR_C_THROW("Couldn't get Sidehandle for side {}", newSide->id());
  }

  Core::LinAlg::SerialDenseMatrix xyzs;
  sidehandle->coordinates(xyzs);
  if (sidehandle->shape() == Core::FE::CellType::quad4)
  {
    Core::LinAlg::Matrix<3, 4> xyze(xyzs.values(), true);
    std::shared_ptr<Cut::Position> pos =
        Cut::PositionFactory::build_position<3, Core::FE::CellType::quad4>(xyze, newx);
    pos->compute(1e-15, true);
    pos->local_coordinates(newxsi);
  }
  else
    FOUR_C_THROW("Not a quad4!");
  return newSide;
}

double XFEM::XFluidContactComm::distanceto_side(
    Core::LinAlg::Matrix<3, 1>& x, Cut::Side* side, Core::LinAlg::Matrix<3, 1>& closest_x)
{
  double distance = 1e200;
  for (std::vector<Cut::Edge*>::const_iterator it = side->edges().begin();
      it != side->edges().end(); ++it)
  {
    Cut::Edge* e = *it;
    Core::LinAlg::Matrix<3, 2> xyzl;
    e->coordinates(xyzl);
    std::shared_ptr<Cut::Position> pos =
        Cut::PositionFactory::build_position<3, Core::FE::CellType::line2>(xyzl, x);
    pos->compute(true);
    Core::LinAlg::Matrix<1, 1> rst;
    pos->local_coordinates(rst);
    if (fabs(rst(0.0)) < 1)
    {
      static Core::LinAlg::Matrix<2, 1> funct;
      // evaluate shape functions
      Core::FE::shape_function<Core::FE::CellType::line2>(rst, funct);
      static Core::LinAlg::Matrix<3, 1> posx;
      posx.multiply(xyzl, funct);
      posx.update(-1, x, 1);
      double tmpdistance = posx.norm2();
      if (distance > tmpdistance)
      {
        distance = tmpdistance;
        closest_x.multiply(xyzl, funct);
      }
    }
  }
  for (std::vector<Cut::Node*>::const_iterator nit = side->nodes().begin();
      nit != side->nodes().end(); ++nit)
  {
    Cut::Node* n = *nit;
    Core::LinAlg::Matrix<3, 1> xyzn;
    n->coordinates(xyzn.data());
    xyzn.update(-1, x, 1);
    double tmpdistance = xyzn.norm2();
    if (distance > tmpdistance)
    {
      distance = tmpdistance;
      n->coordinates(closest_x.data());
    }
  }
  return distance;
}

double XFEM::XFluidContactComm::get_h() { return mc_[0]->get_h(); }

void XFEM::XFluidContactComm::update_physical_sides(
    Cut::Side* side, std::set<Cut::Side*>& performed_sides, std::set<Cut::Side*>& physical_sides)
{
  std::vector<Cut::Side*> neibs = get_new_neighboring_sides(side, performed_sides);
  for (std::size_t sid = 0; sid < neibs.size(); ++sid)
  {
    performed_sides.insert(neibs[sid]);
    Cut::SideHandle* sh = cutwizard_->get_cut_side(neibs[sid]->id());
    if (!sh) FOUR_C_THROW("Couldn't Get Sidehandle {}!", neibs[sid]->id());
    if (sh->isunphysical_sub_side(neibs[sid]))
      update_physical_sides(neibs[sid], performed_sides, physical_sides);
    else
    {
      Core::LinAlg::Matrix<3, 1> normal;
      Core::LinAlg::Matrix<2, 1> center(Core::LinAlg::Initialization::zero);
      neibs[sid]->normal(center, normal, false);
      double norm = normal.norm2();
      if (norm > 1e-10)
        physical_sides.insert(neibs[sid]);
      else
        update_physical_sides(neibs[sid], performed_sides, physical_sides);
    }
  }
}

std::vector<Cut::Side*> XFEM::XFluidContactComm::get_new_neighboring_sides(
    Cut::Side* side, std::set<Cut::Side*>& performed_sides)
{
  std::vector<Cut::Side*> neighbors;
  for (std::vector<Cut::Node*>::const_iterator nit = side->nodes().begin();
      nit != side->nodes().end(); ++nit)
  {
    Cut::Node* n = *nit;
    for (Cut::plain_side_set::const_iterator sit = n->sides().begin(); sit != n->sides().end();
        ++sit)
    {
      Cut::Side* s = *sit;
      if (s == side) continue;
      if (s->id() < 0) continue;
      if (performed_sides.find(s) != performed_sides.end()) continue;
      if (!is_registered_surface(s->id())) continue;
      if (s->id() == side->id())  // on the same contact element
        neighbors.push_back(s);
      else  // do the contact elements share a common edge?
      {
        int common_nodes = 0;
        for (std::size_t neighbor = 0; neighbor < neighbors.size(); ++neighbor)
        {
          if (neighbors[neighbor]->id() == s->id())
          {
            neighbors.push_back(s);
            common_nodes = -1;
            break;
          }
        }
        if (common_nodes == -1) continue;  // we assigned already a side with the same id

        Core::LinAlg::SerialDenseMatrix xyze1;
        Core::LinAlg::SerialDenseMatrix xyze2;
        Cut::SideHandle* sh1 = cutwizard_->get_cut_side(side->id());
        Cut::SideHandle* sh2 = cutwizard_->get_cut_side(s->id());

        for (std::size_t nidx1 = 0; nidx1 < sh1->get_nodes().size(); ++nidx1)
          for (std::size_t nidx2 = 0; nidx2 < sh2->get_nodes().size(); ++nidx2)
          {
            if (sh1->get_nodes()[nidx1]->id() == sh2->get_nodes()[nidx2]->id())
            {
              ++common_nodes;
              break;
            }
          }
        if (common_nodes == 2)
          neighbors.push_back(s);
        else if (common_nodes > 2)
        {
          std::cout << "==| Rejected side " << s->id() << "( " << side->id()
                    << ") as there are only " << common_nodes << " common nodes! |==" << std::endl;
          s->print();
          side->print();
        }
      }
    }
  }
  return neighbors;
}

Cut::Element* XFEM::XFluidContactComm::get_next_element(
    Cut::Element* ele, std::set<Cut::Element*>& performed_elements, int& lastid)
{
  Cut::Element* newele = nullptr;
  if (lastid == -1 && ele != nullptr)
  {
    for (std::size_t s = 0; s < ele->sides().size(); ++s)
    {
      for (std::size_t e = 0; e < ele->sides()[s]->elements().size(); ++e)
      {
        Cut::Element* element = ele->sides()[s]->elements()[e];
        if (element == ele)
          break;
        else if (performed_elements.find(element) != performed_elements.end())
          break;
        else
        {
          newele = element;
          break;
        }
      }
      if (newele) break;
    }
  }
  if (!newele)  // loop structure over all elements
  {
    while (!newele)
    {
      ++lastid;
      if (lastid > 2 * fluiddis_->element_col_map()->num_global_elements())
      {
        return nullptr;
      }
      std::cout << "==| Doing the expensive Version of finding an element! |==" << lastid
                << std::endl;

      Cut::ElementHandle* elementh = cutwizard_->get_element(lastid);
      if (elementh == nullptr) continue;
      Cut::plain_element_set pes;
      elementh->collect_elements(pes);

      if (pes.size() != 1) FOUR_C_THROW("Collected Elements != 1");

      Cut::Element* element = pes[0];

      if (performed_elements.find(element) != performed_elements.end())
        continue;
      else
      {
        newele = element;
        break;
      }
    }
  }

  return newele;
}

void XFEM::XFluidContactComm::register_side_proc(int sid)
{
  if (!parallel_) return;
  if (get_contact_ele(sid)) my_sele_ids_.insert(get_contact_ele(sid)->id());
}


void XFEM::XFluidContactComm::get_cut_side_integration_points(
    int sid, Core::LinAlg::SerialDenseMatrix& coords, std::vector<double>& weights, int& npg)
{
  Cut::SideHandle* sh = cutwizard_->get_cut_side(get_surf_sid(sid));
  if (!sh) FOUR_C_THROW("Couldn't get SideHandle!");
  if (sh->shape() != Core::FE::CellType::quad4) FOUR_C_THROW("Not a quad4!");
  const int numnodes_sh = Core::FE::num_nodes(Core::FE::CellType::quad4);
  Core::LinAlg::SerialDenseMatrix xquad;
  sh->coordinates(xquad);
  Core::LinAlg::Matrix<2, numnodes_sh> deriv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<2, 2> metrictensor(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<3, 1> normal_side(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> normal_bc(Core::LinAlg::Initialization::zero);

  Cut::plain_side_set subsides;
  sh->collect_sides(subsides);
  std::vector<std::shared_ptr<Cut::Tri3BoundaryCell>> bcs;

  Core::LinAlg::SerialDenseMatrix tcoords(3, 3);
  for (Cut::plain_side_set::iterator sit = subsides.begin(); sit != subsides.end(); ++sit)
  {
    Cut::Side* side = *sit;
    side->normal(Core::LinAlg::Matrix<2, 1>(Core::LinAlg::Initialization::zero), normal_side, true);
    for (std::vector<Cut::Facet*>::const_iterator fit = side->facets().begin();
        fit != side->facets().end(); ++fit)
    {
      Cut::Facet* facet = *fit;
      if (facet->is_triangulated())
      {
        for (std::size_t triangle = 0; triangle < facet->triangulation().size(); ++triangle)
        {
          double* coord = tcoords.values();
          for (std::vector<Cut::Point*>::const_iterator tp =
                   facet->triangulation()[triangle].begin();
              tp != facet->triangulation()[triangle].end(); ++tp)
          {
            Cut::Point* p = *tp;
            p->coordinates(coord);
            coord += 3;
          }
          std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc = std::make_shared<Cut::Tri3BoundaryCell>(
              tcoords, facet, facet->triangulation()[triangle]);
          tmp_bc->normal(Core::LinAlg::Matrix<2, 1>(Core::LinAlg::Initialization::zero), normal_bc);
          if (normal_bc.dot(normal_side) < 0.0)
            bcs.push_back(tmp_bc);
          else
          {
            std::vector<Cut::Point*> points = facet->triangulation()[triangle];
            std::reverse(points.begin(), points.end());
            double* coord = tcoords.values();
            for (std::vector<Cut::Point*>::const_iterator tp =
                     facet->triangulation()[triangle].end() - 1;
                tp != facet->triangulation()[triangle].begin() - 1; --tp)
            {
              Cut::Point* p = *tp;
              p->coordinates(coord);
              coord += 3;
            }
            std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc_rev =
                std::make_shared<Cut::Tri3BoundaryCell>(tcoords, facet, points);
            bcs.push_back(tmp_bc_rev);
          }
        }
      }
      else if (facet->points().size() == 3)
      {
        facet->coordinates(tcoords.values());
        std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc =
            std::make_shared<Cut::Tri3BoundaryCell>(tcoords, facet, facet->points());
        tmp_bc->normal(Core::LinAlg::Matrix<2, 1>(Core::LinAlg::Initialization::zero), normal_bc);
        if (normal_bc.dot(normal_side) < 0.0)
          bcs.push_back(tmp_bc);
        else
        {
          std::vector<Cut::Point*> points = facet->points();
          std::reverse(points.begin(), points.end());
          double* coord = tcoords.values();
          for (std::vector<Cut::Point*>::const_iterator tp = facet->points().end() - 1;
              tp != facet->points().begin() - 1; --tp)
          {
            Cut::Point* p = *tp;
            p->coordinates(coord);
            coord += 3;
          }
          std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc_rev =
              std::make_shared<Cut::Tri3BoundaryCell>(tcoords, facet, points);
          bcs.push_back(tmp_bc_rev);
        }
      }
      else
      {
        std::cout << "==| Ignore facet |==" << std::endl;
        std::cout << "facet->GetSplitCells().size(): " << facet->get_split_cells().size()
                  << std::endl;
        facet->print(std::cout);
        if (!parallel_) FOUR_C_THROW("Ignore Facet");
      }
    }
    if (!side->facets().size())
    {
      if (!sh->isunphysical_sub_side(side))
      {
        if (parallel_)  // we are on the wrong proc ...
          continue;     // return true;
        else
          FOUR_C_THROW("There are not facets on a physical side?");
      }
      else if (side->num_nodes() == 3)
      {
        side->coordinates(tcoords.values());
        std::vector<Cut::Point*> points;
        for (unsigned p = 0; p < side->num_nodes(); ++p)
          points.push_back(side->nodes()[p]->point());
        std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc =
            std::make_shared<Cut::Tri3BoundaryCell>(tcoords, nullptr, points);
        tmp_bc->normal(Core::LinAlg::Matrix<2, 1>(Core::LinAlg::Initialization::zero), normal_bc);
        if (normal_bc.dot(normal_side) < 0.0)
          bcs.push_back(tmp_bc);
        else
        {
          std::vector<Cut::Point*> tmp_points = points;
          std::reverse(tmp_points.begin(), tmp_points.end());
          double* coord = tcoords.values();
          for (std::vector<Cut::Node*>::const_iterator tp = side->nodes().end() - 1;
              tp != side->nodes().begin() - 1; --tp)
          {
            Cut::Point* p = (*tp)->point();
            p->coordinates(coord);
            coord += 3;
          }
          std::shared_ptr<Cut::Tri3BoundaryCell> tmp_bc_rev =
              std::make_shared<Cut::Tri3BoundaryCell>(tcoords, nullptr, tmp_points);
          bcs.push_back(tmp_bc_rev);
        }
      }
      else
        FOUR_C_THROW("Unphysical Subside is not a tri3?");  // Cannot happen as these sides are
                                                            // created by the SelfCut
    }
  }

  weights.clear();
  Core::LinAlg::Matrix<3, 1> x_gp_lin(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2, 1> rst(
      Core::LinAlg::Initialization::zero);  // local coordinates w.r.t side
  double drs = 0;
  double drs_sh = 0;
  for (std::size_t bc = 0; bc < bcs.size(); ++bc)
  {
    Core::FE::GaussIntegration gi = bcs[bc]->gauss_rule(cutwizard_->get_bc_cubaturedegree());
    if (gi.num_points())
    {
      coords.reshape(weights.size() + gi.num_points(), 2);
      int idx = weights.size();
      for (Core::FE::GaussIntegration::iterator iquad = gi.begin(); iquad != gi.end(); ++iquad)
      {
        const Core::LinAlg::Matrix<2, 1> eta(iquad.point(), false);
        XFEM::Utils::compute_surface_transformation(drs, x_gp_lin, normal, bcs[bc].get(), eta);

        // find element local position of gauss point
        const Core::LinAlg::Matrix<3, numnodes_sh> xquad_m(xquad.values(), true);
        std::shared_ptr<Cut::Position> pos =
            Cut::PositionFactory::build_position<3, Core::FE::CellType::quad4>(xquad_m, x_gp_lin);
        pos->compute(true);
        pos->local_coordinates(rst);
        coords(idx, 0) = rst(0);
        coords(idx, 1) = rst(1);

        Core::FE::shape_function_2d_deriv1(
            deriv, coords(idx, 0), coords(idx, 1), Core::FE::CellType::quad4);
        Core::FE::compute_metric_tensor_for_boundary_ele<Core::FE::CellType::quad4>(
            xquad_m, deriv, metrictensor, drs_sh, nullptr);
        weights.push_back(iquad.weight() * drs / drs_sh);  // small tri3 to quad4 weight
        ++idx;
      }
    }
  }

  npg = weights.size() - 1;
  return;
}

void XFEM::XFluidContactComm::fill_complete_sele_map()
{
  if (!parallel_) return;
  if (cutwizard_ == nullptr)
    FOUR_C_THROW("XFluidContactComm::fill_complete_sele_map: CutWizard not set!");

  // We also add all unphysical sides
  for (std::size_t i = 0; i < mortar_id_to_sosid_.size(); ++i)
  {
    if (mortar_id_to_sosid_[i] == -1) continue;  // this entry is not set!
    Cut::SideHandle* sh = cutwizard_->get_cut_side(mortar_id_to_sosid_[i]);
    if (!sh)
      FOUR_C_THROW("Couldn't get Sidhandle for mortarId {}, soid {}!", i + min_mortar_id_,
          mortar_id_to_sosid_[i]);
    if (cutwizard_->get_cut_side(mortar_id_to_sosid_[i])->hasunphysical_sub_side())
    {
      my_sele_ids_.insert(get_contact_ele(mortar_id_to_sosid_[i])->id());
    }
  }
  std::vector<int> my_sele_ids(my_sele_ids_.begin(), my_sele_ids_.end());
  contact_ele_rowmap_fluidownerbased_ = std::make_shared<Core::LinAlg::Map>(
      -1, my_sele_ids.size(), my_sele_ids.data(), 0, fluiddis_->get_comm());
}

void XFEM::XFluidContactComm::prepare_time_step()
{
  higher_contact_elements_.clear();
  higher_contact_elements_comm_.clear();
}

void XFEM::XFluidContactComm::prepare_iteration_step()
{
  higher_contact_elements_comm_.clear();

  std::vector<int> src(higher_contact_elements_.begin(), higher_contact_elements_.end());

  std::vector<int> dest;
  Core::LinAlg::allreduce_vector(src, dest, fluiddis_->get_comm());

  higher_contact_elements_comm_.insert(dest.begin(), dest.end());

  if (!Core::Communication::my_mpi_rank(fluiddis_->get_comm()) &&
      higher_contact_elements_comm_.size())
  {
    std::cout << "==| Interface Elements with an increased number of Contact Gausspoins:";
    for (std::set<int>::iterator sit = higher_contact_elements_comm_.begin();
        sit != higher_contact_elements_comm_.end(); ++sit)
      std::cout << " " << get_surf_sid(*sit) << "(" << *sit << ") |";
    std::cout << "==" << std::endl;
  }
}

double XFEM::XFluidContactComm::get_fpi_pcontact_exchange_dist()
{
  return mcfpi_ps_pf_->get_fpi_pcontact_exchange_dist();
}

double XFEM::XFluidContactComm::get_fpi_pcontact_fullfraction()
{
  return mcfpi_ps_pf_->get_fpi_pcontact_fullfraction();
}

void XFEM::XFluidContactComm::print_summary_contact_gps()
{
  sum_gps_.resize(5);
  std::vector<int> g_sum_gps(5);
  g_sum_gps = Core::Communication::sum_all(sum_gps_, fluiddis_->get_comm());
  if (!Core::Communication::my_mpi_rank(fluiddis_->get_comm()))
  {
    std::cout << "===| Summary Contact GPs |===" << std::endl;
    // 0 ... Contact, 1 ... Contact_NoContactNoFSI, 2 ... Contact_NoContactFSI, 3 ... FSI_NoContact,
    // 4 ... FSI_Contact
    std::cout << "Contact: " << g_sum_gps[0] << ", Contact_noC_noFSI: " << g_sum_gps[1]
              << ", Contact_noC_FSI: " << g_sum_gps[2] << "("
              << g_sum_gps[0] + g_sum_gps[1] + g_sum_gps[2] << "), FSI_noC: " << g_sum_gps[3]
              << ", FSI_C: " << g_sum_gps[4] << "(" << g_sum_gps[3] + g_sum_gps[4] << ") == (Sum: "
              << g_sum_gps[0] + g_sum_gps[1] + g_sum_gps[2] + g_sum_gps[3] + g_sum_gps[4] << ")"
              << " === (Fair Sum: " << g_sum_gps[0] + g_sum_gps[1] + g_sum_gps[3] << ")"
              << std::endl;
    std::cout << "===| ------------------- |===" << std::endl;
  }
  sum_gps_.clear();
  sum_gps_.resize(5);
}

FOUR_C_NAMESPACE_CLOSE
