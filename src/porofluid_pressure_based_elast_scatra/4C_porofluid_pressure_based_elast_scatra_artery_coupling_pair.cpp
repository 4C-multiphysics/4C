// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_geometry_coordinate_system_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_defines.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_function.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::PoroMultiPhaseScatraArteryCouplingPair()
    : PoroMultiPhaseScatraArteryCouplingPairBase(),
      coupltype_(CouplingType::type_undefined),
      couplmethod_(Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::none),
      condname_(""),
      isinit_(false),
      ispreevaluated_(false),
      isactive_(false),
      funct_coupl_active_(false),
      diam_funct_active_(false),
      evaluate_in_ref_config_(true),
      evaluate_on_lateral_surface_(true),
      element1_(nullptr),
      element2_(nullptr),
      arterydiamref_(0.0),
      arterydiam_at_gp_(0.0),
      numdof_cont_(0),
      numdof_art_(0),
      dim1_(0),
      dim2_(0),
      numcoupleddofs_(0),
      numfluidphases_(0),
      numvolfrac_(0),
      numscalcont_(0),
      numscalart_(0),
      nds_porofluid_(-1),
      n_gp_(0),
      n_gp_per_patch_(0),
      arteryelelengthref_(0.0),
      arteryelelength_(0.0),
      jacobi_(0.0),
      pp_(0.0),
      eta_a_(0.0),
      eta_b_(0.0),
      curr_segment_length_(0.0),
      constant_part_evaluated_(false),
      coupling_element_type_(""),
      artdiam_funct_(nullptr),
      porosityname_("porosity"),
      artpressname_("p_art"),
      segmentid_(-1),
      numpatch_axi_(0),
      numpatch_rad_(0),
      timefacrhs_art_dens_(0.0),
      timefacrhs_cont_dens_(0.0),
      timefacrhs_art_(0.0),
      timefacrhs_cont_(0.0)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::init(std::vector<Core::Elements::Element const*> elements,
    const Teuchos::ParameterList& couplingparams, const Teuchos::ParameterList& fluidcouplingparams,
    const std::vector<int>& coupleddofs_cont, const std::vector<int>& coupleddofs_art,
    const std::vector<std::vector<int>>& scale_vec, const std::vector<std::vector<int>>& funct_vec,
    const std::string condname, const double penalty, const std::string couplingtype,
    const int eta_ntp)
{
  // init stuff
  couplmethod_ =
      Teuchos::getIntegralValue<Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>(
          couplingparams, "ARTERY_COUPLING_METHOD");

  condname_ = condname;

  evaluate_in_ref_config_ = fluidcouplingparams.get<bool>("EVALUATE_IN_REF_CONFIG");

  evaluate_on_lateral_surface_ = fluidcouplingparams.get<bool>("LATERAL_SURFACE_COUPLING");

  coupling_element_type_ = couplingtype;

  numpatch_axi_ = fluidcouplingparams.get<int>("NUMPATCH_AXI");
  numpatch_rad_ = fluidcouplingparams.get<int>("NUMPATCH_RAD");

  element1_ = elements[0];
  element2_ = elements[1];

  // set coupling type
  if (element1_->element_type().name() == "ArteryType" &&
      element2_->element_type().name() == "PoroFluidMultiPhaseType")
  {
    coupltype_ = type_porofluid;
    nds_porofluid_ = 0;
  }
  else if (element1_->element_type().name() == "TransportType" &&
           element2_->element_type().name() == "TransportType")
  {
    coupltype_ = type_scatra;
    nds_porofluid_ = 2;
  }
  else
  {
    FOUR_C_THROW(
        "Your selected coupling is not possible, type of element1: {}, type of element2: {}",
        element1_->element_type().name(), element2_->element_type().name());
  }

  if (couplmethod_ == Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp)
  {
    // set eta
    eta_.resize(1);
    eta_[0] = eta_ntp;

    // check couplingtype
    if (coupling_element_type_ != "ARTERY" && coupling_element_type_ != "AIRWAY")
    {
      if (coupltype_ == type_porofluid)
      {
        FOUR_C_THROW(
            "Wrong coupling type in DESIGN 1D ARTERY TO POROFLUID NONCONF COUPLING CONDITIONS.\n "
            "NTP-coupling is only possible for coupling type: "
            " 'ARTERY' or 'AIRWAY'. "
            "Your coupling type "
            "is: {}",
            coupling_element_type_);
      }
      else
      {
        FOUR_C_THROW(
            "Wrong coupling type in DESIGN 1D ARTERY TO SCATRA NONCONF COUPLING CONDITIONS.\n"
            "NTP-coupling is only possible for coupling type: "
            "'ARTERY' or 'AIRWAY'. "
            "Your coupling type "
            "is: {}",
            coupling_element_type_);
      }
    }
  }

  // get number of DOFs of artery or artery-scatra
  const Core::Nodes::Node* const* artnodes;
  artnodes = element1_->nodes();

  numdof_art_ = element1_->num_dof_per_node(*artnodes[0]);
  for (int i = 1; i < element1_->num_node(); i++)
    if (numdof_art_ != element1_->num_dof_per_node(*artnodes[i]))
      FOUR_C_THROW("It is not possible to have different number of Dofs in artery discretization");
  dim1_ = numdof_art_ * element1_->num_node();

  // get number of DOFs of continuous ele (scatra or porofluid)
  const Core::Nodes::Node* const* contnodes;
  contnodes = element2_->nodes();

  numdof_cont_ = element2_->num_dof_per_node(*contnodes[0]);
  for (int i = 1; i < element2_->num_node(); i++)
    if (numdof_cont_ != element2_->num_dof_per_node(*contnodes[i]))
      FOUR_C_THROW(
          "It is not possible to have different number of Dofs in continuous discretization");
  dim2_ = numdof_cont_ * element2_->num_node();

  // safety check
  if (numdof_art_ != (int)(scale_vec[0].size()))
    FOUR_C_THROW("Wrong size of scale-vector (artery)");
  if (numdof_art_ != (int)(funct_vec[0].size()))
    FOUR_C_THROW("Wrong size of function-vector (artery)");
  if (numdof_cont_ != (int)(scale_vec[1].size()))
    FOUR_C_THROW("Wrong size of scale-vector (continuous discretization)");
  if (numdof_cont_ != (int)(funct_vec[1].size()))
    FOUR_C_THROW("Wrong size of function-vector (continuous discretization)");

  // fill scale vector
  scale_vec_ = scale_vec;

  // fill function vector
  funct_vec_.resize(2);
  funct_vec_[0].resize(numdof_art_);
  funct_vec_[1].resize(numdof_cont_);
  fill_function_vector(funct_vec_[0], funct_vec[0], scale_vec_[0]);
  fill_function_vector(funct_vec_[1], funct_vec[1], scale_vec_[1]);

  // get the actually coupled dofs
  coupleddofs_cont_ = coupleddofs_cont;
  // Note: this will be overwritten in case of arteryscatra-scatra coupling
  volfracpressid_ = coupleddofs_cont;
  coupleddofs_art_ = coupleddofs_art;
  numcoupleddofs_ = coupleddofs_cont.size();

  // safety check
  for (int icont = 0; icont < numcoupleddofs_; icont++)
  {
    if (coupleddofs_cont_[icont] >= numdof_cont_)
    {
      FOUR_C_THROW(
          "You try to couple DOF {}, which is larger than the number of dofs of the continuous "
          "discretization",
          coupleddofs_cont_[icont] + 1);
    }
    if (coupleddofs_cont_[icont] < 0)
    {
      FOUR_C_THROW("Your coupling DOF of the continuous discretization must be >= 0, your DOF = {}",
          coupleddofs_cont_[icont] + 1);
    }
  }
  for (int iart = 0; iart < numcoupleddofs_; iart++)
  {
    if (coupleddofs_art_[iart] >= numdof_art_)
    {
      FOUR_C_THROW(
          "You try to couple DOF {}, which is larger than the number of dofs of the artery "
          "discretization",
          coupleddofs_art_[iart] + 1);
    }
    if (coupleddofs_art_[iart] < 0)
    {
      FOUR_C_THROW("Your coupling DOF of the reduced discretization must be >= 0, your DOF = {}",
          coupleddofs_art_[iart] + 1);
    }
  }

  // Set reference nodal positions for artery element
  for (unsigned int n = 0; n < numnodesart_; ++n)
  {
    const Core::Nodes::Node* node = element1_->nodes()[n];
    for (unsigned int d = 0; d < numdim_; ++d) ele1posref_(numdim_ * n + d) = node->x()[d];
  }

  // get length of 1D element
  static Core::LinAlg::Matrix<numdim_, 1> arterypos0;
  for (unsigned int d = 0; d < numdim_; ++d) arterypos0(d) = ele1posref_(d);
  static Core::LinAlg::Matrix<numdim_, 1> arterypos1;
  for (unsigned int d = 0; d < numdim_; ++d) arterypos1(d) = ele1posref_(numdim_ + d);

  static Core::LinAlg::Matrix<numdim_, 1> dist;
  dist.update(-1.0, arterypos0, 1.0, arterypos1, 0.0);
  arteryelelengthref_ = dist.norm2();

  // get initial direction of artery element
  lambda0_.update(1.0 / arteryelelengthref_, dist, 0.0);

  // Set reference nodal positions for continuous discretization element
  for (unsigned int inode = 0; inode < numnodescont_; ++inode)
  {
    const Core::Nodes::Node* node = element2_->nodes()[inode];
    for (unsigned int idim = 0; idim < numdim_; ++idim) ele2posref_(idim, inode) = node->x()[idim];
  }

  // set current nodal positions to reference nodal positions for continuous discretization
  // element
  ele2pos_.update(1.0, ele2posref_, 0.0);

  // get penalty parameter
  pp_ = penalty;

  // get out of here
  isinit_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::setup_fluid_managers_and_materials(const std::string disname,
    const double& timefacrhs_art, const double& timefacrhs_cont)
{
  // dummy parameter list
  Discret::Elements::PoroFluidMultiPhaseEleParameter* para =
      Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(disname);

  double arterydens = 0.0;

  std::shared_ptr<Mat::FluidPoroMultiPhase> multiphasemat = nullptr;
  std::shared_ptr<Mat::MatList> contscatramat = nullptr;
  switch (coupltype_)
  {
    case type_porofluid:
    {
      multiphasemat =
          std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(element2_->material(nds_porofluid_));
      if (multiphasemat == nullptr)
        FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed for artery-porofluid coupling!");
      for (int idof = 0; idof < numcoupleddofs_; idof++)
      {
        const int matid = multiphasemat->mat_id(coupleddofs_cont_[idof]);
        std::shared_ptr<Core::Mat::Material> singlemat = multiphasemat->material_by_id(matid);

        // safety check
        if (singlemat->material_type() != Core::Materials::m_fluidporo_volfracpressure &&
            singlemat->material_type() != Core::Materials::m_fluidporo_singlephase)
        {
          FOUR_C_THROW(
              "You can only couple volume fraction pressures or fluid phases in multiphase "
              "porespace, your material is of type {}",
              singlemat->material_type());
        }
      }
      // we have a coupling with scatra -> the scatra-material is the third material in the 2D/3D
      // element
      if (element2_->num_material() == 3)
      {
        if (element2_->material(2)->material_type() == Core::Materials::m_matlist or
            element2_->material(2)->material_type() == Core::Materials::m_matlist_reactions)
        {
          std::shared_ptr<Mat::MatList> scatramat =
              std::static_pointer_cast<Mat::MatList>(element2_->material(2));
          numscalcont_ = scatramat->num_mat();
        }
        else
          FOUR_C_THROW("Only MAT_matlist is valid for poromultiphase-scatra material");
      }
      // we have a coupling with artery-scatra -> the artery-scatra-material is the second
      // material in the artery element
      if (element1_->num_material() == 2)
      {
        if (element1_->material(1)->material_type() == Core::Materials::m_matlist)
        {
          std::shared_ptr<Mat::MatList> artscatramat =
              std::static_pointer_cast<Mat::MatList>(element1_->material(1));
          numscalart_ = artscatramat->num_mat();
        }
        else if (element1_->material(1)->material_type() == Core::Materials::m_scatra)
          numscalart_ = 1;
        else
          FOUR_C_THROW("Only MAT_matlist and MAT_scatra are valid for artery-scatra material");
      }

      arterymat_ = std::static_pointer_cast<Mat::Cnst1dArt>(element1_->material(0));
      if (arterymat_ == nullptr)
        FOUR_C_THROW("cast to artery material failed for porofluid-artery coupling!");
      arterydiam_at_gp_ = arterymat_->diam();
      arterydiamref_ = arterymat_->diam_initial();
      arterydens = arterymat_->density();

      break;
    }
    case type_scatra:
    {
      // check if we actually have three materials
      if (element2_->num_material() < 3) FOUR_C_THROW("no third material available");

      multiphasemat =
          std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(element2_->material(nds_porofluid_));

      if (multiphasemat == nullptr)
        FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed for arteryscatra-scatra coupling!");

      contscatramat = std::static_pointer_cast<Mat::MatList>(element2_->material(0));
      if (contscatramat == nullptr)
        FOUR_C_THROW("cast to ScatraMat failed for arteryscatra-scatra coupling!");

      for (int idof = 0; idof < numcoupleddofs_; idof++)
      {
        const int matid = contscatramat->mat_id(coupleddofs_cont_[idof]);
        std::shared_ptr<Core::Mat::Material> singlemat = contscatramat->material_by_id(matid);

        // safety check
        if (singlemat->material_type() != Core::Materials::m_scatra_multiporo_volfrac &&
            singlemat->material_type() != Core::Materials::m_scatra_multiporo_fluid)
        {
          FOUR_C_THROW(
              "You can only couple Mat::ScatraMatMultiPoroVolFrac or Mat::ScatraMatMultiPoroFluid, "
              "your material is of type {}",
              singlemat->material_type());
        }

        if (singlemat->material_type() == Core::Materials::m_scatra_multiporo_volfrac)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(singlemat);
          volfracpressid_[idof] = poromat->phase_id() + multiphasemat->num_vol_frac();
        }
      }
      // get the artery scatra-material
      if (element1_->material(0)->material_type() == Core::Materials::m_matlist)
      {
        std::shared_ptr<Mat::MatList> artscatramat =
            std::static_pointer_cast<Mat::MatList>(element1_->material(0));
        numscalart_ = artscatramat->num_mat();
      }
      else if (element1_->material(0)->material_type() == Core::Materials::m_scatra)
        numscalart_ = 1;
      else
        FOUR_C_THROW("Only MAT_matlist and MAT_scatra are valid for artery-scatra material");
      numscalcont_ = numdof_cont_;
      arterymat_ = std::static_pointer_cast<Mat::Cnst1dArt>(element1_->material(1));
      if (arterymat_ == nullptr)
        FOUR_C_THROW("cast to artery material failed for arteryscatra-scatra coupling!");
      arterydiam_at_gp_ = arterymat_->diam();
      arterydiamref_ = arterymat_->diam_initial();
      arterydens = arterymat_->density();
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }

  // take care of diameter function
  const int diam_funct_num = arterymat_->diameter_function();
  if (diam_funct_num > -1)
  {
    // no real use-case without function coupling
    if (not funct_coupl_active_)
      FOUR_C_THROW(
          "Diameter function has been defined but no exchange function has been set, this is "
          "currently not possible, if you still want a varying diameter without any exchange "
          "terms, you can still define a zero exchange term");
    diam_funct_active_ = true;
    artdiam_funct_ = &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
        diam_funct_num);
    if (coupltype_ == type_porofluid)
    {
      // cont derivatives + 1 artery pressure derivative
      diamderivs_ = std::vector<double>(numdof_cont_ + 1, 0.0);
      diam_stiffmat11_ = Core::LinAlg::SerialDenseMatrix();
      diam_stiffmat12_ = Core::LinAlg::SerialDenseMatrix();
    }
  }

  // safety checks for lateral surface coupling
  if (evaluate_on_lateral_surface_)
  {
    if (!evaluate_in_ref_config_)
      FOUR_C_THROW(
          "Evaluation in current configuration is not yet possible in combination with lateral "
          "surface coupling");
    if (diam_funct_active_)
      FOUR_C_THROW(
          "Setting a varying diameter is not yet possible in combination with lateral "
          "surface coupling");
    if (not(distype_cont == Core::FE::CellType::hex8 or distype_cont == Core::FE::CellType::tet4))
      FOUR_C_THROW("Only TET4 and HEX8 elements possible for lateral surface coupling");
  }

  numfluidphases_ = multiphasemat->num_fluid_phases();
  numvolfrac_ = multiphasemat->num_vol_frac();

  // create phase-manager
  phasemanager_ = Discret::Elements::PoroFluidManager::PhaseManagerInterface::create_phase_manager(
      *para, numdim_, multiphasemat->material_type(),
      POROFLUIDMULTIPHASE::Action::get_access_from_artcoupling, multiphasemat->num_mat(),
      multiphasemat->num_fluid_phases());

  // setup phasemanager
  phasemanager_->setup(element2_, nds_porofluid_);

  // create variablemanager
  variablemanager_ = Discret::Elements::PoroFluidManager::VariableManagerInterface<numdim_,
      numnodescont_>::create_variable_manager(*para,
      POROFLUIDMULTIPHASE::Action::get_access_from_artcoupling, multiphasemat,
      multiphasemat->num_mat(), multiphasemat->num_fluid_phases());

  // initialize the names used in functions
  initialize_function_names();

  // fill vector where to assemble rhs-(function) coupling into
  // summed up phase requires special treatment
  initialize_assemble_into_cont_dof_vector();

  // initialize the functions
  for (int i = 0; i < 2; i++)
    for (unsigned int idof = 0; idof < funct_vec_[i].size(); idof++)
      if (funct_vec_[i][idof] != nullptr) initialize_function(*funct_vec_[i][idof]);
  if (diam_funct_active_) initialize_function(*artdiam_funct_);

  // set time fac for right hand side evaluation of coupling
  set_time_fac_rhs(arterydens, *contscatramat, timefacrhs_art, timefacrhs_cont);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::pre_evaluate(std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector)
{
  if (!isinit_) FOUR_C_THROW("MeshTying Pair has not yet been initialized");

  if (couplmethod_ == Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp)
  {
    pre_evaluate_node_to_point_coupling();
  }
  else
  {
    if (evaluate_on_lateral_surface_)
      pre_evaluate_lateral_surface_coupling(*gp_vector);
    else
      pre_evaluate_centerline_coupling();
  }


  ispreevaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::pre_evaluate_lateral_surface_coupling(Core::LinAlg::MultiVector<double>& gp_vector)
{
  const int pid =
      Core::Communication::my_mpi_rank(Global::Problem::instance()->get_dis("artery")->get_comm());
  const int mylid = element1_->lid();
  if (element2_->owner() != pid) return;

  // unit radial basis vectors
  Core::LinAlg::Matrix<3, 1> unit_rad_1;
  Core::LinAlg::Matrix<3, 1> unit_rad_2;
  // unit tangential basis
  Core::LinAlg::Matrix<3, 1> tang;
  if (numdim_ != 3) FOUR_C_THROW("surface-based formulation makes only sense in 3D");
  for (int idim = 0; idim < 3; idim++) tang(idim) = lambda0_(idim);

  Core::Geo::build_orthonormal_basis_from_unit_vector(tang, unit_rad_1, unit_rad_2);

  // get radius
  const int artelematerial = coupltype_ == type_scatra ? 1 : 0;
  std::shared_ptr<Mat::Cnst1dArt> arterymat =
      std::static_pointer_cast<Mat::Cnst1dArt>(element1_->material(artelematerial));
  if (arterymat == nullptr)
    FOUR_C_THROW("cast to artery material failed for porofluid-artery coupling!");
  double radius = arterymat->diam() / 2.0;

  // size of one integration patch: 2*pi*R/numpatch_rad_*L/numpatch_axi_
  const double patch_size =
      1.0 / numpatch_axi_ * arteryelelengthref_ * 1.0 / numpatch_rad_ * 2.0 * M_PI * radius;

  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1(
      Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta
  // Coords and derivatives of 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1, double> r1(Core::LinAlg::Initialization::zero);  // = r1
  static Core::LinAlg::Matrix<numdim_, 1, double> r1_eta(
      Core::LinAlg::Initialization::zero);  // = r1,eta

  // element parameter space coordinates in 3D element
  std::vector<double> xi(3);
  // number of GPs
  n_gp_ = 0;

  // we use always 25 integration points per integration patch
  Core::FE::IntegrationPoints2D gaussPointsperPatch =
      Core::FE::IntegrationPoints2D(Core::FE::GaussRule2D::quad_25point);
  n_gp_per_patch_ = gaussPointsperPatch.nquad;
  n_gp_ = n_gp_per_patch_ * numpatch_axi_ * numpatch_rad_;
  // define Gauss points and n_gp-sized quantities
  eta_.resize(n_gp_);
  eta_s_.resize(n_gp_);
  wgp_.resize(n_gp_);
  xi_.resize(n_gp_);
  for (int i_gp = 0; i_gp < n_gp_; i_gp++) xi_[i_gp].resize(numdim_);

  // loop over all axial patches
  for (int i_ax = 0; i_ax < numpatch_axi_; i_ax++)
  {
    // loop over all radial patches
    for (int i_rad = 0; i_rad < numpatch_rad_; i_rad++)
    {
      // loop over all GPs of this patch
      for (int i_gp = 0; i_gp < n_gp_per_patch_; i_gp++)
      {
        const int gpid = i_ax * numpatch_rad_ * n_gp_per_patch_ + i_rad * n_gp_per_patch_ + i_gp;
        // axial Gauss point eta lies in [-1; 1]
        double eta = -1.0 - 1.0 / numpatch_axi_ + (i_ax + 1.0) * 2.0 / numpatch_axi_ +
                     gaussPointsperPatch.qxg[i_gp][0] * 1.0 / numpatch_axi_;

        // Update coordinates and derivatives for 1D and 2D/3D element
        get_1d_shape_functions<double>(N1, N1_eta, eta);
        compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);

        // radial Gauss point theta lies in [-pi; pi]
        double theta = (-1.0 - 1.0 / numpatch_rad_ + (i_rad + 1.0) * 2.0 / numpatch_rad_ +
                           gaussPointsperPatch.qxg[i_gp][1] * 1.0 / numpatch_rad_) *
                       M_PI;

        // get point on lateral blood vessel surface
        for (int idim = 0; idim < 3; idim++)
          r1(idim) = r1(idim) + unit_rad_1(idim) * radius * cos(theta) +
                     unit_rad_2(idim) * radius * sin(theta);

        // project into 3D domain
        bool projection_valid = false;
        projection<double>(r1, xi, projection_valid);
        eta_[gpid] = eta;
        xi_[gpid] = xi;
        // projection is valid and GP is so far unclaimed by other pair
        if (projection_valid && ((gp_vector)(gpid))[mylid] < 0.5)
        {
          isactive_ = true;
          // include jacobian
          wgp_[gpid] = gaussPointsperPatch.qwgt[i_gp] * patch_size / 4.0;
          gp_vector.SumIntoMyValue(mylid, gpid, 1.0);
        }
        else
          wgp_[gpid] = 0.0;
      }
    }
  }

  // free memory
  if (not isactive_)
  {
    std::vector<double>().swap(eta_);
    std::vector<double>().swap(wgp_);
    std::vector<std::vector<double>>().swap(xi_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::pre_evaluate_centerline_coupling()
{
  // Try to create integration segment [eta_a, eta_b]
  create_integration_segment();

  // no viable segment found
  if (!isactive_)
  {
    return;
  }

  // choice of optimal Gauss-rule:
  // basically the N^(2)*N^(2) term is crucial
  // for (bi-, tri-)linear elements (only considered so far):
  // in 2D the highest possible polynomial order for this term is 4 since N^(2) can be quadratic
  // for arbitrary integration in element
  // --> we need 3 gp for exact integration
  // in 3D the highest possible polynomial order for this term is 6 since N^(2) can be cubic for
  // arbitrary integration in element
  // --> we need 4 gp for exact integration

  Core::FE::IntegrationPoints1D gaussPoints =
      Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
  if (numdim_ == 3) gaussPoints = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

  n_gp_ = gaussPoints.nquad;
  // define Gauss points and n_gp-sized quantities
  eta_.resize(n_gp_);
  eta_s_.resize(n_gp_);
  wgp_.resize(n_gp_);
  inv_j_.resize(n_gp_);
  xi_.resize(n_gp_);
  for (int i_gp = 0; i_gp < n_gp_; i_gp++) xi_[i_gp].resize(numdim_);

  // get jacobian determinant
  const double determinant = (eta_b_ - eta_a_) / 2.0;
  jacobi_ = determinant * arteryelelengthref_ / 2.0;

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1
  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1(
      Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta
  // Coords and derivatives of 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1, double> r1(Core::LinAlg::Initialization::zero);  // = r1
  static Core::LinAlg::Matrix<numdim_, 1, double> r1_eta(
      Core::LinAlg::Initialization::zero);  // = r1,eta

  // project the Gauss points --> those have to able to be projected
  for (int i_gp = 0; i_gp < n_gp_; i_gp++)
  {
    // compute the coordinate trafo, weights and project Gauss points
    eta_[i_gp] = (eta_a_ + eta_b_) / 2.0 + gaussPoints.qxg[i_gp][0] * determinant;
    eta_s_[i_gp] = eta_[i_gp];
    wgp_[i_gp] = gaussPoints.qwgt[i_gp];

    // Update coordinates and derivatives for 1D and 2D/3D element
    get_1d_shape_functions<double>(N1, N1_eta, eta_[i_gp]);
    compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);

    bool projection_valid = false;
    projection<double>(r1, xi_[i_gp], projection_valid);
    if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");

    // compute (dX/dxi)^-1
    get_2d_3d_shape_functions<double>(N2, N2_xi, xi_[i_gp]);
    inv_j_[i_gp].multiply_nt(N2_xi, ele2pos_);
    inv_j_[i_gp].invert();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::pre_evaluate_node_to_point_coupling()
{
  xi_.resize(1);
  xi_[0].resize(numdim_);

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1
  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1(
      Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta
  // Coords and derivatives of 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1, double> r1(Core::LinAlg::Initialization::zero);  // = r1
  static Core::LinAlg::Matrix<numdim_, 1, double> r1_eta(
      Core::LinAlg::Initialization::zero);  // = r1,eta

  // Update coordinates and derivatives for 1D and 2D/3D element
  get_1d_shape_functions<double>(N1, N1_eta, eta_[0]);
  compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);

  bool projection_valid = false;
  projection<double>(r1, xi_[0], projection_valid);

  // coupling pairs is only active if projection is valid
  isactive_ = projection_valid;

  // compute (dX/dxi)^-1
  get_2d_3d_shape_functions<double>(N2, N2_xi, xi_[0]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::delete_unnecessary_gps(std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector)
{
  const int mylid = element1_->lid();
  n_gp_ = 0;
  for (int igp = 0; igp < n_gp_per_patch_ * numpatch_axi_ * numpatch_rad_; igp++)
    if (wgp_[igp] > 1e-12) n_gp_++;

  std::vector<double> wgp(n_gp_);
  std::vector<double> eta(n_gp_);
  std::vector<std::vector<double>> xi(n_gp_);

  int mygp = 0;
  for (int igp = 0; igp < n_gp_per_patch_ * numpatch_axi_ * numpatch_rad_; igp++)
  {
    if (wgp_[igp] > 1e-12)
    {
      const double scale = 1.0 / ((*gp_vector)(igp))[mylid];
      eta[mygp] = eta_[igp];
      xi[mygp] = xi_[igp];
      wgp[mygp] = wgp_[igp] * scale;
      mygp++;
    }
  }
  if (n_gp_ == 0)
    isactive_ = false;
  else
    isactive_ = true;

  // overwrite
  wgp_ = wgp;
  eta_ = eta;
  xi_ = xi;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::reset_state(std::shared_ptr<Core::FE::Discretization> contdis,
    std::shared_ptr<Core::FE::Discretization> artdis)
{
  if (!ispreevaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated");

  // get location array for continuous ele
  Core::Elements::LocationArray la_cont(contdis->num_dof_sets());
  element2_->location_vector(*contdis, la_cont, false);

  // get location array for artery ele
  Core::Elements::LocationArray la_art(artdis->num_dof_sets());
  element1_->location_vector(*artdis, la_art, false);

  switch (coupltype_)
  {
    case type_porofluid:
    {
      // extract element and node values of fluid
      variablemanager_->extract_element_and_node_values(*element2_, *contdis, la_cont, ele2pos_, 0);

      // get contelephinp_ and artelephinp_
      contelephinp_ = Core::FE::extract_values(*contdis->get_state("phinp_fluid"), la_cont[0].lm_);
      artelephinp_ =
          Core::FE::extract_values(*artdis->get_state("one_d_artery_pressure"), la_art[0].lm_);

      // extract velocity of solid phase
      extract_solid_vel(*contdis);

      // extract values of artery-scatra discretization
      if (numscalart_ > 0)
      {
        std::shared_ptr<const Core::LinAlg::Vector<double>> artscalarnp =
            artdis->get_state(ndsartery_scatra_, "one_d_artery_phinp");
        if (artscalarnp != nullptr)
        {
          Core::Elements::LocationArray la(artdis->num_dof_sets());
          element1_->location_vector(*artdis, la, false);
          // rebuild scalar vector
          eartscalarnp_.clear();
          eartscalarnp_.resize(numscalart_,
              Core::LinAlg::Matrix<numnodesart_, 1>(Core::LinAlg::Initialization::zero));
          // extract local values of artery-scatra field from global state vector
          Core::FE::extract_my_values<Core::LinAlg::Matrix<numnodesart_, 1>>(
              *artscalarnp, eartscalarnp_, la[ndsartery_scatra_].lm_);
        }
        else
          FOUR_C_THROW("Cannot get artery-scatra from artery discretization");
      }
      // extract values of continuous scatra discretization
      if (numscalcont_ > 0)
      {
        // get state vector from discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>> contscalarnp =
            contdis->get_state(3, "scalars");
        if (contscalarnp != nullptr)
        {
          Core::Elements::LocationArray la(contdis->num_dof_sets());
          element2_->location_vector(*contdis, la, false);
          // rebuild scalar vector
          econtscalarnp_.clear();
          econtscalarnp_.resize(numscalcont_,
              Core::LinAlg::Matrix<numnodescont_, 1>(Core::LinAlg::Initialization::zero));
          // extract local values of continuous-scatra field from global state vector
          Core::FE::extract_my_values<Core::LinAlg::Matrix<numnodescont_, 1>>(
              *contscalarnp, econtscalarnp_, la[3].lm_);
        }
        else
        {
          FOUR_C_THROW("Cannot get state vector 'scalars'");
        }
      }
      break;
    }
    case type_scatra:
    {
      // extract element and node values of fluid
      variablemanager_->extract_element_and_node_values(*element2_, *contdis, la_cont, ele2pos_, 2);

      // get contelephinp_ and artelephinp_
      contelephinp_ = Core::FE::extract_values(*contdis->get_state("phinp"), la_cont[0].lm_);
      artelephinp_ =
          Core::FE::extract_values(*artdis->get_state("one_d_artery_phinp"), la_art[0].lm_);

      // extract artery pressure
      std::shared_ptr<const Core::LinAlg::Vector<double>> artpressnp =
          artdis->get_state(ndsscatra_artery_, "one_d_artery_pressure");
      if (artpressnp != nullptr)
      {
        Core::Elements::LocationArray la(artdis->num_dof_sets());
        element1_->location_vector(*artdis, la, false);
        Core::FE::extract_my_values<Core::LinAlg::Matrix<numnodesart_, 1>>(
            *artpressnp, earterypressurenp_, la[ndsscatra_artery_].lm_);
      }
      else
      {
        FOUR_C_THROW("Cannot get arterypressure from artery-scatra discretization");
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
double PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
    Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
    Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
    Core::LinAlg::SerialDenseMatrix* stiffmat22, Core::LinAlg::SerialDenseMatrix* D_ele,
    Core::LinAlg::SerialDenseMatrix* M_ele, Core::LinAlg::SerialDenseVector* Kappa_ele,
    const std::vector<double>& segmentlengths)
{
  if (!ispreevaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated");

  // resize and initialize variables to zero
  if (forcevec1 != nullptr) forcevec1->size(dim1_);
  if (forcevec2 != nullptr) forcevec2->size(dim2_);

  if (stiffmat11 != nullptr) stiffmat11->shape(dim1_, dim1_);
  if (stiffmat12 != nullptr) stiffmat12->shape(dim1_, dim2_);
  if (stiffmat21 != nullptr) stiffmat21->shape(dim2_, dim1_);
  if (stiffmat22 != nullptr) stiffmat22->shape(dim2_, dim2_);

  if (arterymat_->is_collapsed()) return 0.0;

  std::vector<double> myEta;
  std::vector<std::vector<double>> myXi;

  double etaA = 0.0;
  double etaB = 0.0;
  double integrated_diam = 0.0;

  switch (couplmethod_)
  {
    case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::gpts:
    {
      // initialize eta and xi
      myEta.assign(n_gp_, double{});
      myXi.assign(n_gp_, std::vector<double>(numdim_, double{}));
      // recompute eta and xi --> see note in this function
      recompute_eta_and_xi_in_deformed_configuration(segmentlengths, myEta, myXi, etaA, etaB);
      // actual evaluate
      evaluate_gpts(myEta, myXi, segmentlengths, forcevec1, forcevec2, stiffmat11, stiffmat12,
          stiffmat21, stiffmat22);

      // case where diameter is constant
      integrated_diam = arterydiamref_ * segmentlengths[segmentid_];
      break;
    }
    case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::mp:
    {
      // initialize eta and xi
      myEta.assign(n_gp_, double{});
      myXi.assign(n_gp_, std::vector<double>(numdim_, double{}));
      // recompute eta and xi --> see note in this function
      recompute_eta_and_xi_in_deformed_configuration(segmentlengths, myEta, myXi, etaA, etaB);
      // actual evaluate
      evaluate_dm_kappa(myEta, myXi, segmentlengths, D_ele, M_ele, Kappa_ele);

      // case where diameter is constant
      integrated_diam = arterydiamref_ * segmentlengths[segmentid_];
      break;
    }
    case Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp:
    {
      // define eta and xi
      myEta = eta_;
      myXi = xi_;
      // actual evaluate
      evaluate_ntp(eta_, xi_, forcevec1, forcevec2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
      integrated_diam = arterydiamref_ * segmentlengths[0];
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type for artery to poro coupling");
      break;
  }

  // evaluate the function coupling (with possibly varying diameter)
  if (funct_coupl_active_)
    evaluate_function_coupling(myEta, myXi, segmentlengths, forcevec1, forcevec2, stiffmat11,
        stiffmat12, stiffmat21, stiffmat22, integrated_diam);

  // evaluate derivative of 1D shape function times solid velocity
  evaluated_nds_solid_vel(myEta, myXi, segmentlengths, *forcevec1, etaA, etaB);

  return integrated_diam;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_additional_linearizationof_integrated_diam(Core::LinAlg::SerialDenseMatrix*
                                                                  stiffmat11,
    Core::LinAlg::SerialDenseMatrix* stiffmat12)
{
  if (!ispreevaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated");

  if (stiffmat11 != nullptr) stiffmat11->shape(dim1_, dim1_);
  if (stiffmat12 != nullptr) stiffmat12->shape(dim1_, dim2_);

  // do not evaluate if element is collapsed
  if (arterymat_->is_collapsed()) return;

  // this is the integrated diameter over the entire element (all segments)
  const double arterydiam = arterymat_->diam();
  const double prefac = M_PI * std::pow(arterydiam, 3) / 32.0 / arterymat_->viscosity();
  // TODO: for viscosity law blood, viscosity depends on diameter, this linearization is still
  // missing

  Core::LinAlg::update(prefac, diam_stiffmat11_, 0.0, *stiffmat11);
  Core::LinAlg::update(prefac, diam_stiffmat12_, 0.0, *stiffmat12);
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
int PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::ele1_gid() const
{
  return element1_->id();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
int PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::ele2_gid() const
{
  return element2_->id();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
int PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::get_segment_id() const
{
  return segmentid_;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
double PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::calculate_vol_2d_3d() const
{
  // use one-point Gauss rule
  Core::FE::IntPointsAndWeights<numdim_> intpoints_stab(
      Discret::Elements::DisTypeToStabGaussRule<distype_cont>::rule);

  const double* gpcoord = intpoints_stab.ip().qxg[0];   // actual integration point (coords)
  const double gpweight = intpoints_stab.ip().qwgt[0];  // actual integration point (weight)

  Core::LinAlg::Matrix<numdim_, 1> xsi(gpcoord, true);
  static Core::LinAlg::Matrix<numnodescont_, 1> funct;
  static Core::LinAlg::Matrix<numdim_, numnodescont_> deriv;
  static Core::LinAlg::Matrix<numdim_, numdim_> xjm;
  static Core::LinAlg::Matrix<numdim_, numdim_> xji;

  // shape functions and their first derivatives
  Core::FE::shape_function<distype_cont>(xsi, funct);
  Core::FE::shape_function_deriv1<distype_cont>(xsi, deriv);

  //

  // get Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
  */
  xjm.multiply_nt(deriv, ele2pos_);
  double det = xji.invert(xjm);

  if (det < 1E-16) FOUR_C_THROW("GLOBAL ELEMENT ZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", det);

  // compute integration factor
  return gpweight * det;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_segment_id(const int& segmentid)
{
  segmentid_ = segmentid;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
double PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::apply_mesh_movement(const bool firstcall,
    std::shared_ptr<Core::FE::Discretization> contdis)
{
  // nodal displacement values for ALE
  Core::LinAlg::Matrix<numdim_, numnodescont_> edispnp(Core::LinAlg::Initialization::zero);

  if (!firstcall)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = contdis->get_state(1, "dispnp");
    Core::Elements::LocationArray la(contdis->num_dof_sets());
    element2_->location_vector(*contdis, la, false);

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(numdim_ * numnodescont_, -1);
    for (unsigned int inode = 0; inode < numnodescont_; ++inode)
      for (unsigned int idim = 0; idim < numdim_; ++idim)
        lmdisp[inode * numdim_ + idim] = la[1].lm_[inode * numdim_ + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<numdim_, numnodescont_>>(
        *dispnp, edispnp, lmdisp);
  }
  else
    return (eta_b_ - eta_a_) / 2.0 * arteryelelengthref_;

  // update current configuration
  ele2pos_.update(1.0, ele2posref_, 1.0, edispnp, 0.0);

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_XYZ(
      Core::LinAlg::Initialization::zero);  // = N2,X
  static Core::LinAlg::Matrix<numdim_, numdim_> defgrad(
      Core::LinAlg::Initialization::zero);      // = dx/dX = F
  static Core::LinAlg::Matrix<numdim_, 1> Ft0;  // = F*t0

  curr_segment_length_ = 0.0;
  // current segment length = \int_{\eta_a}^{eta_b} || F*t0 ||_2 ds
  // all under the assumption that artery element completely follows deformation of
  // underlying 2D/3D problem
  for (int i_gp = 0; i_gp < n_gp_; i_gp++)
  {
    // get shape functions of continuous element
    const std::vector<double> xi = xi_[i_gp];
    get_2d_3d_shape_functions<double>(N2, N2_xi, xi);
    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    N2_XYZ.multiply(inv_j_[i_gp], N2_xi);
    // dx/dX = x * N_XYZ^T
    defgrad.multiply_nt(ele2pos_, N2_XYZ);
    Ft0.multiply(defgrad, lambda0_);
    curr_segment_length_ += Ft0.norm2() * wgp_[i_gp] * jacobi_;
  }

  return curr_segment_length_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::recompute_eta_and_xi_in_deformed_configuration(const std::vector<double>& segmentlengths,
    std::vector<double>& myEta, std::vector<std::vector<double>>& myXi, double& etaA, double& etaB)
{
  // NOTE: we assume that the 1D artery element completely follows the deformation of the
  // underlying
  //       porous medium, so its length might change. Interaction between artery element and
  //       porous medium has to be evaluated in current/deformed configuration. However, Gauss
  //       points of the original projection (in reference configuration) cannot be used then
  //       anymore but we must define a new parameter space [-1, 1] which maps into the current
  //       configuration. First, we determine the new etaA and etaB of this segment as sum of
  //       segment lengths etaA_new = -1.0 + 2.0* ( \sum_{i=0}_{this_seg-1} l_i / total_ele_length
  //       ) etaB_new = -1.0 + 2.0* ( \sum_{i=0}_{this_seg} l_i / total_ele_length ) then GPs are
  //       distributed in the interval [etaA_new, etaB_new] The last step is to get the new
  //       projected xi_i in the 2D/3D parameter space of the second element For each new GP
  //       eta_new, this can be done by finding the point in reference configuration which deforms
  //       to the point in current configuration where the GP now lies as \int_{\eta_a}^{eta_s} ||
  //       F*t0 ||_2 ds where eta_s is the unknown. Linearization of this nonlinear equation
  //       within the Newton loop is done with FAD

  // not necessary if we do not take into account mesh movement or ntp-coupling
  if (evaluate_in_ref_config_)
  {
    myEta = eta_;
    myXi = xi_;
    arteryelelength_ = arteryelelengthref_;
  }
  else
  {
    // Vectors for shape functions and their derivatives
    static Core::LinAlg::Matrix<1, numnodesart_, double> N1(
        Core::LinAlg::Initialization::zero);  // = N1
    static Core::LinAlg::Matrix<1, numnodesart_, double> N1_eta(
        Core::LinAlg::Initialization::zero);  // = N1,eta
    // Coords and derivatives of 1D and 2D/3D element
    static Core::LinAlg::Matrix<numdim_, 1, double> r1(Core::LinAlg::Initialization::zero);  // = r1
    static Core::LinAlg::Matrix<numdim_, 1, double> r1_eta(
        Core::LinAlg::Initialization::zero);  // = r1,eta

    // current length of artery
    arteryelelength_ = std::accumulate(segmentlengths.begin(), segmentlengths.end(), 0.0);

    // length of segments [0, 1, ..., this_seg-1]
    double length_so_far = 0.0;
    for (int iseg = 0; iseg < segmentid_; iseg++) length_so_far += segmentlengths[iseg];

    // length of this segment
    const double curr_seg_length = segmentlengths[segmentid_];

    // get new etaA and etaB
    etaA = -1.0 + 2.0 * (length_so_far / arteryelelength_);
    etaB = -1.0 + 2.0 * ((length_so_far + curr_seg_length) / arteryelelength_);

    Core::FE::IntegrationPoints1D gaussPoints =
        Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
    if (numdim_ == 3)
      gaussPoints = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

    // distribute new Gauss points
    const double determinant = (etaB - etaA) / 2.0;
    for (int i_gp = 0; i_gp < n_gp_; i_gp++)
      myEta[i_gp] = (etaA + etaB) / 2.0 + gaussPoints.qxg[i_gp][0] * determinant;

    // solution variable for Newton loop
    FAD eta_s = 0.0;
    eta_s.diff(0, 1);  // independent variable 0 out of a total of 1

    // the GP loop
    bool converged = false;
    for (int i_gp = 0; i_gp < n_gp_; i_gp++)
    {
      // start value for Newton: last converged value of previous evaluation (should be pretty
      // close)
      eta_s.val() = eta_s_[i_gp];
      const double desired_length = curr_seg_length * (myEta[i_gp] - etaA) / (etaB - etaA);
      double val = -1.0;
      // Netwon loop
      for (int istep = 0; istep < MESHMOVEMENTMAXITER; istep++)
      {
        // integrate \int_{\eta_a}^{eta_s} || F*t0 ||_2 ds
        const FAD curr_length = integrate_length_to_eta_s(eta_s);

        val = curr_length.val() - desired_length;
        if (fabs(val) < CONVTOLNEWTONMESHMOVEMENT)
        {
          converged = true;
          break;
        }
        const double deriv = curr_length.fastAccessDx(0);
        // Newton update
        eta_s.val() -= val / deriv;
      }

      if (!converged)
        std::cout << "WARNING: could not find Gauss point position in reference configuration";
      // finally find new xi_i by projection eta_s in reference configuration

      // Update coordinates and derivatives for 1D and 2D/3D element
      get_1d_shape_functions<double>(N1, N1_eta, eta_s.val());
      compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);

      bool projection_valid = false;
      projection<double>(r1, myXi[i_gp], projection_valid);
      if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");
      // save the converged value
      eta_s_[i_gp] = eta_s.val();
    }  // GP loop
  }  // !evaluate_in_ref_config_
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_gpts(const std::vector<double>& eta, const std::vector<std::vector<double>>& xi,
    const std::vector<double>& segmentlengths, Core::LinAlg::SerialDenseVector* forcevec1,
    Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
    Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
    Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  if (numcoupleddofs_ > 0)
  {
    if (!constant_part_evaluated_)
    {
      gpts_ntp_stiffmat11_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat12_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat21_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat22_ = Core::LinAlg::SerialDenseMatrix();
    }

    // we only have to this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
      static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
          Core::LinAlg::Initialization::zero);  // = N1,eta

      static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
      static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
          Core::LinAlg::Initialization::zero);  // = N2,xi1

      gpts_ntp_stiffmat11_.shape(dim1_, dim1_);
      gpts_ntp_stiffmat12_.shape(dim1_, dim2_);
      gpts_ntp_stiffmat21_.shape(dim2_, dim1_);
      gpts_ntp_stiffmat22_.shape(dim2_, dim2_);

      const double curr_seg_length = segmentlengths[segmentid_];

      for (int i_gp = 0; i_gp < n_gp_; i_gp++)
      {
        // Get constant values from projection
        const double w_gp = wgp_[i_gp];
        const double myeta = eta[i_gp];
        const double jac = curr_seg_length / 2.0;
        const std::vector<double> myxi = xi[i_gp];

        // Update shape functions and their derivatives for 1D and 2D/3D element
        get_1d_shape_functions<double>(N1, N1_eta, myeta);
        get_2d_3d_shape_functions<double>(N2, N2_xi, myxi);

        // evaluate
        evaluate_gpts_stiff(w_gp, N1, N2, jac, pp_);
      }
    }  //! constant_part_evaluated_ or !evaluate_in_ref_config_

    update_gptsntp_stiff(*stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
    check_valid_volume_fraction_pressure_coupling(
        *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
    evaluate_gptsntp_force(
        *forcevec1, *forcevec2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_ntp(const std::vector<double>& eta, const std::vector<std::vector<double>>& xi,
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  if (numcoupleddofs_ > 0)
  {
    if (!constant_part_evaluated_)
    {
      gpts_ntp_stiffmat11_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat12_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat21_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_stiffmat22_ = Core::LinAlg::SerialDenseMatrix();
    }

    // we only have to this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
      static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
          Core::LinAlg::Initialization::zero);  // = N1,eta

      static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
      static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
          Core::LinAlg::Initialization::zero);  // = N2,xi1

      gpts_ntp_stiffmat11_.shape(dim1_, dim1_);
      gpts_ntp_stiffmat12_.shape(dim1_, dim2_);
      gpts_ntp_stiffmat21_.shape(dim2_, dim1_);
      gpts_ntp_stiffmat22_.shape(dim2_, dim2_);


      // Get constant values from projection
      const double myeta = eta[0];
      const std::vector<double> myxi = xi[0];

      // Update shape functions and their derivatives for 1D and 2D/3D element
      get_1d_shape_functions<double>(N1, N1_eta, myeta);
      get_2d_3d_shape_functions<double>(N2, N2_xi, myxi);

      // evaluate
      evaluate_ntp_stiff(N1, N2, pp_);
    }
  }  //! constant_part_evaluated_ or !evaluate_in_ref_config_

  update_gptsntp_stiff(*stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  //! safety check for coupling with additional porous network (= Artery coupling)
  if (coupling_element_type_ == "ARTERY")
    check_valid_volume_fraction_pressure_coupling(
        *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  evaluate_gptsntp_force(
      *forcevec1, *forcevec2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_dm_kappa(const std::vector<double>& eta,
    const std::vector<std::vector<double>>& xi, const std::vector<double>& segmentlengths,
    Core::LinAlg::SerialDenseMatrix* D_ele, Core::LinAlg::SerialDenseMatrix* M_ele,
    Core::LinAlg::SerialDenseVector* Kappa_ele)
{
  if (D_ele != nullptr) D_ele->shape(dim1_, dim1_);
  if (M_ele != nullptr) M_ele->shape(dim1_, dim2_);
  if (Kappa_ele != nullptr) Kappa_ele->size(dim1_);

  if (numcoupleddofs_ > 0)
  {
    // initialize
    if (!constant_part_evaluated_)
    {
      d_ = Core::LinAlg::SerialDenseMatrix();
      m_ = Core::LinAlg::SerialDenseMatrix();
    }
    // we only have to this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
      static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
          Core::LinAlg::Initialization::zero);  // = N1,eta

      static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
      static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
          Core::LinAlg::Initialization::zero);  // = N2,xi1

      d_.shape(dim1_, dim1_);
      m_.shape(dim1_, dim2_);
      kappa_.size(dim1_);

      const double curr_seg_length = segmentlengths[segmentid_];

      for (int i_gp = 0; i_gp < n_gp_; i_gp++)
      {
        // Get constant values from projection
        const double w_gp = wgp_[i_gp];
        const double myeta = eta[i_gp];
        const double jac = curr_seg_length / 2.0;
        const std::vector<double> myxi = xi[i_gp];

        // Update shape functions and their derivatives for 1D and 2D/3D element
        get_1d_shape_functions<double>(N1, N1_eta, myeta);
        get_2d_3d_shape_functions<double>(N2, N2_xi, myxi);

        evaluate_dm_kappa(w_gp, N1, N2, jac);
      }
    }  //! constant_part_evaluated_ or !evaluate_in_ref_config_

    update_dm_kappa(*D_ele, *M_ele, *Kappa_ele);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_function_coupling(const std::vector<double>& eta,
    const std::vector<std::vector<double>>& xi, const std::vector<double>& segmentlengths,
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22,
    double& integrated_diam)
{
  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numnodescont_, 1> N2_transpose(
      Core::LinAlg::Initialization::zero);  // = N2^T

  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1
  static Core::LinAlg::Matrix<numdim_, numnodescont_> derxy(
      Core::LinAlg::Initialization::zero);  // = N2,xi1

  static Core::LinAlg::Matrix<numdim_, numdim_> xjm;
  static Core::LinAlg::Matrix<numdim_, numdim_> xjm0;
  static Core::LinAlg::Matrix<numdim_, numdim_> xji;

  const double curr_seg_length = segmentlengths[segmentid_];

  // case with varying diameter and type porofluid: (integral and linearizations have to be
  // calculated) --> reset
  if (diam_funct_active_ && coupltype_ == type_porofluid)
  {
    integrated_diam = 0.0;
    diam_stiffmat11_.shape(dim1_, dim1_);
    diam_stiffmat12_.shape(dim1_, dim2_);
  }

  for (int i_gp = 0; i_gp < n_gp_; i_gp++)
  {
    // Get constant values from projection
    const double w_gp = wgp_[i_gp];
    const double myeta = eta[i_gp];
    const std::vector<double> myxi = xi[i_gp];

    const double jac = curr_seg_length / 2.0;

    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_1d_shape_functions<double>(N1, N1_eta, myeta);
    get_2d_3d_shape_functions<double>(N2, N2_xi, myxi);
    N2_transpose.update_t(N2);

    xjm.multiply_nt(N2_xi, ele2pos_);
    xjm0.multiply_nt(N2_xi, ele2posref_);

    const double det = xji.invert(xjm);
    // inverse of transposed jacobian "ds/dX"
    const double det0 = xjm0.determinant();

    derxy.multiply(xji, N2_xi);

    // determinant of deformationgradient
    // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
    const double JacobianDefGradient = det / det0;

    // clear current gauss point data for safety
    phasemanager_->clear_gp_state();
    variablemanager_->evaluate_gp_variables(N2_transpose, derxy);

    phasemanager_->evaluate_gp_state(JacobianDefGradient, *variablemanager_, nds_porofluid_);

    evaluate_function_coupling(w_gp, N1, N2, jac, *forcevec1, *forcevec2, *stiffmat11, *stiffmat12,
        *stiffmat21, *stiffmat22, integrated_diam);
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluated_nds_solid_vel(const std::vector<double>& eta,
    const std::vector<std::vector<double>>& xi, const std::vector<double>& segmentlengths,
    Core::LinAlg::SerialDenseVector& forcevec1, const double& etaA, const double& etaB)
{
  if (evaluate_in_ref_config_ || coupltype_ == type_scatra ||
      couplmethod_ == Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp)
    return;

  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_XYZ(
      Core::LinAlg::Initialization::zero);  // = N2,X
  static Core::LinAlg::Matrix<numdim_, numdim_> defgrad(
      Core::LinAlg::Initialization::zero);  // = dx/dX = F

  static Core::LinAlg::Matrix<numdim_, numdim_> xjm;
  static Core::LinAlg::Matrix<numdim_, numdim_> xjm0;
  static Core::LinAlg::Matrix<numdim_, numdim_> invJ;
  static Core::LinAlg::Matrix<numdim_, numdim_> xji;

  static Core::LinAlg::Matrix<numdim_, 1> lambda_t;  // direction in current conf.

  // Evaluate $-\int_a^b d N^(1)/ds*pi*R^2 * lambda_t*v_s ds$
  //        = $-\int_\eta_a^\eta_b d N^(1)/deta*2/L_ele*pi*R^2 * lambda_t*v_s*L_seg/2.0 d\eta$
  //        = $-\int_\eta_a^\eta_b d N^(1)/deta*pi*R^2 * lambda_t*v_s*(\eta_a-\eta_b)/2.0 d\eta$
  const double jac = (etaB - etaA) / 2.0;
  for (int i_gp = 0; i_gp < n_gp_; i_gp++)
  {
    // Get constant values from projection
    const double w_gp = wgp_[i_gp];
    const double myeta = eta[i_gp];
    const std::vector<double> myxi = xi[i_gp];

    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_1d_shape_functions<double>(N1, N1_eta, myeta);
    get_2d_3d_shape_functions<double>(N2, N2_xi, myxi);

    xjm.multiply_nt(N2_xi, ele2pos_);
    xji.invert(xjm);

    // dX/dpsi
    xjm0.multiply_nt(N2_xi, ele2posref_);
    // dpsi/dX
    // note: cannot use invJ_ here -> defined at original Gauss points
    invJ.invert(xjm0);
    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    N2_XYZ.multiply(invJ, N2_xi);
    // dx/dX = x * N_XYZ^T
    defgrad.multiply_nt(ele2pos_, N2_XYZ);

    // current direction of artery element at GP
    lambda_t.multiply(defgrad, lambda0_);
    lambda_t.scale(1.0 / lambda_t.norm2());

    std::vector<double> myvel(3, 0.0);
    for (unsigned int j = 0; j < numnodescont_; j++)
      for (unsigned int idim = 0; idim < numdim_; idim++) myvel[idim] += N2(j) * ele2vel_(idim, j);

    double lambda_t_vel = 0.0;
    for (unsigned int idim = 0; idim < numdim_; idim++)
      lambda_t_vel += myvel[idim] * lambda_t(idim);

    // TODO: here reference diameter is taken
    for (unsigned int i = 0; i < numnodesart_; i++)
      forcevec1(i) +=
          N1_eta(i) * w_gp * jac * lambda_t_vel * arterydiamref_ * arterydiamref_ * M_PI / 4.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_gpts_stiff(const double& w_gp, const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi, const double& pp)
{
  // Evaluate meshtying stiffness for artery element N_1^T * N_1
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double stiff = timefacrhs_art_ * pp * jacobi * w_gp * N1(i) * N1(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat11_(i * numdof_art_ + coupleddofs_art_[dof],
            j * numdof_art_ + coupleddofs_art_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for artery element "mixed" N_1^T * (-N_2)
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double stiff = timefacrhs_art_ * pp * jacobi * w_gp * N1(i) * (-N2(j));
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat12_(i * numdof_art_ + coupleddofs_art_[dof],
            j * numdof_cont_ + coupleddofs_cont_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for continuous element "mixed" N_2^T * (-N_1)
  for (unsigned int i = 0; i < numnodescont_; i++)
  {
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double stiff = timefacrhs_cont_ * pp * jacobi * w_gp * N2(i) * (-N1(j));
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat21_(i * numdof_cont_ + coupleddofs_cont_[dof],
            j * numdof_art_ + coupleddofs_art_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for continuous element N_2^T * N_2
  for (unsigned int i = 0; i < numnodescont_; i++)
  {
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double stiff = timefacrhs_cont_ * pp * jacobi * w_gp * N2(i) * N2(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat22_(i * numdof_cont_ + coupleddofs_cont_[dof],
            j * numdof_cont_ + coupleddofs_cont_[dof]) += stiff;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_ntp_stiff(const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& pp)
{
  // Evaluate meshtying stiffness for artery element N_1^T * N_1
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double stiff = timefacrhs_art_ * pp * N1(i) * N1(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat11_(i * numdof_art_ + coupleddofs_art_[dof],
            j * numdof_art_ + coupleddofs_art_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for artery element "mixed" N_1^T * (-N_2)
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double stiff = timefacrhs_art_ * pp * N1(i) * (-N2(j));
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat12_(i * numdof_art_ + coupleddofs_art_[dof],
            j * numdof_cont_ + coupleddofs_cont_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for continuous element "mixed" N_2^T * (-N_1)
  for (unsigned int i = 0; i < numnodescont_; i++)
  {
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double stiff = timefacrhs_cont_ * pp * N2(i) * (-N1(j));
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat21_(i * numdof_cont_ + coupleddofs_cont_[dof],
            j * numdof_art_ + coupleddofs_art_[dof]) += stiff;
    }
  }

  // Evaluate meshtying stiffness for continuous element N_2^T * N_2
  for (unsigned int i = 0; i < numnodescont_; i++)
  {
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double stiff = timefacrhs_cont_ * pp * N2(i) * N2(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        gpts_ntp_stiffmat22_(i * numdof_cont_ + coupleddofs_cont_[dof],
            j * numdof_cont_ + coupleddofs_cont_[dof]) += stiff;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_dm_kappa(const double& w_gp, const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi)
{
  // Evaluate element mortar coupling operator kappa = N_1
  for (unsigned int inode = 0; inode < numnodesart_; inode++)
  {
    const double mykappa = w_gp * jacobi * N1(inode);
    for (int dof = 0; dof < numdof_art_; dof++) kappa_(inode * numdof_art_ + dof) += mykappa;
  }

  // Evaluate element mortar coupling operator D = N_1^T * N_1
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double D = jacobi * w_gp * N1(i) * N1(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        d_(i * numdof_art_ + coupleddofs_art_[dof], j * numdof_art_ + coupleddofs_art_[dof]) += D;
    }
  }

  // Evaluate element mortar coupling operator M = N_1^T * N_2
  for (unsigned int i = 0; i < numnodesart_; i++)
  {
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double M = jacobi * w_gp * N1(i) * N2(j);
      for (int dof = 0; dof < numcoupleddofs_; dof++)
        m_(i * numdof_art_ + coupleddofs_art_[dof], j * numdof_cont_ + coupleddofs_cont_[dof]) += M;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_gptsntp_force(Core::LinAlg::SerialDenseVector& forcevec1,
    Core::LinAlg::SerialDenseVector& forcevec2, const Core::LinAlg::SerialDenseMatrix& stiffmat11,
    const Core::LinAlg::SerialDenseMatrix& stiffmat12,
    const Core::LinAlg::SerialDenseMatrix& stiffmat21,
    const Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  // Evaluate meshtying forces for artery element
  for (int i = 0; i < dim1_; i++)
    for (int j = 0; j < dim1_; j++) forcevec1(i) -= stiffmat11(i, j) * artelephinp_[j];

  for (int i = 0; i < dim1_; i++)
    for (int j = 0; j < dim2_; j++) forcevec1(i) -= stiffmat12(i, j) * contelephinp_[j];

  // Evaluate meshtying forces for continuous-dis element
  for (int i = 0; i < dim2_; i++)
    for (int j = 0; j < dim1_; j++) forcevec2(i) -= stiffmat21(i, j) * artelephinp_[j];

  for (int i = 0; i < dim2_; i++)
    for (int j = 0; j < dim2_; j++) forcevec2(i) -= stiffmat22(i, j) * contelephinp_[j];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::update_gptsntp_stiff(Core::LinAlg::SerialDenseMatrix& stiffmat11,
    Core::LinAlg::SerialDenseMatrix& stiffmat12, Core::LinAlg::SerialDenseMatrix& stiffmat21,
    Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  stiffmat11.assign(gpts_ntp_stiffmat11_);
  stiffmat12.assign(gpts_ntp_stiffmat12_);
  stiffmat21.assign(gpts_ntp_stiffmat21_);
  stiffmat22.assign(gpts_ntp_stiffmat22_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::check_valid_volume_fraction_pressure_coupling(Core::LinAlg::SerialDenseMatrix& stiffmat11,
    Core::LinAlg::SerialDenseMatrix& stiffmat12, Core::LinAlg::SerialDenseMatrix& stiffmat21,
    Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  for (int idof = 0; idof < numcoupleddofs_; idof++)
  {
    if (!variablemanager_->element_has_valid_vol_frac_pressure(
            volfracpressid_[idof] - numfluidphases_ - numvolfrac_))
    {
      // reset to zero for this dof
      for (unsigned int i = 0; i < numnodesart_; i++)
      {
        for (unsigned int j = 0; j < numnodesart_; j++)
          stiffmat11(i * numdof_art_ + coupleddofs_art_[idof],
              j * numdof_art_ + coupleddofs_art_[idof]) = 0.0;
      }

      for (unsigned int i = 0; i < numnodesart_; i++)
      {
        for (unsigned int j = 0; j < numnodescont_; j++)
          stiffmat12(i * numdof_art_ + coupleddofs_art_[idof],
              j * numdof_cont_ + coupleddofs_cont_[idof]) = 0.0;
      }

      for (unsigned int i = 0; i < numnodescont_; i++)
      {
        for (unsigned int j = 0; j < numnodesart_; j++)
          stiffmat21(i * numdof_cont_ + coupleddofs_cont_[idof],
              j * numdof_art_ + coupleddofs_art_[idof]) = 0.0;
      }

      for (unsigned int i = 0; i < numnodescont_; i++)
      {
        for (unsigned int j = 0; j < numnodescont_; j++)
          stiffmat22(i * numdof_cont_ + coupleddofs_cont_[idof],
              j * numdof_cont_ + coupleddofs_cont_[idof]) = 0.0;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::update_dm_kappa(Core::LinAlg::SerialDenseMatrix& D_ele,
    Core::LinAlg::SerialDenseMatrix& M_ele, Core::LinAlg::SerialDenseVector& Kappa_ele)
{
  D_ele.assign(d_);
  M_ele.assign(m_);
  Kappa_ele.assign(kappa_);

  for (int idof = 0; idof < numcoupleddofs_; idof++)
  {
    // this coupling is only possible if we also have an element with a valid volume fraction
    // pressure, i.e., if we also have a smeared representation of the neovasculature at this
    // point if not ---> corresponding matrices are set to zero
    if (!variablemanager_->element_has_valid_vol_frac_pressure(
            volfracpressid_[idof] - numfluidphases_ - numvolfrac_))
    {
      // reset to zero for this dof
      for (unsigned int i = 0; i < numnodesart_; i++)
        for (unsigned int j = 0; j < numnodesart_; j++)
          D_ele(i * numdof_art_ + coupleddofs_art_[idof],
              j * numdof_art_ + coupleddofs_art_[idof]) = 0.0;

      for (unsigned int i = 0; i < numnodesart_; i++)
        for (unsigned int j = 0; j < numnodescont_; j++)
          M_ele(i * numdof_art_ + coupleddofs_art_[idof],
              j * numdof_cont_ + coupleddofs_cont_[idof]) = 0.0;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_function_coupling(const double& w_gp,
    const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi,
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22,
    double& integrated_diam)
{
  // resize
  std::vector<double> artscalarnpAtGP(numscalart_, 0.0);
  std::vector<double> contscalarnpAtGP(numscalcont_, 0.0);
  double artpressAtGP = 0.0;

  // get artery values at GP
  get_artery_values_at_gp(N1, artpressAtGP, artscalarnpAtGP);
  // get scatra values at GP
  get_cont_scalar_values_at_gp(N2, contscalarnpAtGP);
  // NOTE: values of fluid held by managers

  if (diam_funct_active_)
  {
    evaluate_diam_function_and_deriv(artpressAtGP, w_gp, N1, N2, jacobi);
    // integral is only calculated in this case
    if (coupltype_ == type_porofluid) integrated_diam += w_gp * jacobi * arterydiam_at_gp_;
  }

  // artery functions
  for (int i_art = 0; i_art < numdof_art_; i_art++)
  {
    if (funct_vec_[0][i_art] != nullptr)
    {
      // resize
      std::vector<double> artderivs(numdof_art_, 0.0);
      std::vector<double> contderivs(numdof_cont_, 0.0);
      double functval = 0.0;
      // evaluate and assemble
      evaluate_function_and_deriv(*funct_vec_[0][i_art], artpressAtGP, artscalarnpAtGP,
          contscalarnpAtGP, functval, artderivs, contderivs);
      assemble_function_coupling_into_force_stiff_art(i_art, w_gp, N1, N2, jacobi,
          scale_vec_[0][i_art], functval, artderivs, contderivs, forcevec1, stiffmat11, stiffmat12);
    }
  }
  // continuous discretization functions
  for (int i_cont = 0; i_cont < numdof_cont_; i_cont++)
  {
    if (funct_vec_[1][i_cont] != nullptr)
    {
      // resize
      std::vector<double> artderivs(numdof_art_, 0.0);
      std::vector<double> contderivs(numdof_cont_, 0.0);
      double functval = 0.0;
      // evaluate and assemble
      evaluate_function_and_deriv(*funct_vec_[1][i_cont], artpressAtGP, artscalarnpAtGP,
          contscalarnpAtGP, functval, artderivs, contderivs);
      assemble_function_coupling_into_force_stiff_cont(
          cont_dofs_to_assemble_functions_into_[i_cont], w_gp, N1, N2, jacobi,
          scale_vec_[1][i_cont], timefacrhs_cont_dens_[i_cont], functval, artderivs, contderivs,
          forcevec2, stiffmat21, stiffmat22);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_diam_function_and_deriv(const double artpressnpAtGP, const double& w_gp,
    const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi)
{
  // we have to derive w.r.t. fluid variables
  std::vector<std::pair<std::string, double>> variables;
  variables.reserve(numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_ + 1);

  // reference diameter is constant
  std::vector<std::pair<std::string, double>> constants;
  constants.reserve(2);
  constants.push_back(std::pair<std::string, double>("D0", arterydiamref_));
  constants.push_back(
      std::pair<std::string, double>("Dprev", arterymat_->diam_previous_time_step()));

  // set fluid values as variables
  set_fluid_values_as_variables(variables, artpressnpAtGP);

  // evaluate the diameter at GP by evaluating the function
  arterydiam_at_gp_ = artdiam_funct_->evaluate(variables, constants, 0);

  // derivatives and linearizations are so far only calculated for coupltype porofluid
  if (coupltype_ == type_porofluid)
  {
    // function derivatives
    std::vector<double> curderivs(artdiam_funct_->evaluate_derivative(variables, constants, 0));
    // derivatives w.r.t. primary variables
    std::fill(diamderivs_.begin(), diamderivs_.end(), 0.0);

    // diameter derivatives w.r.t. saturations and pressures
    for (int doftoderive = 0; doftoderive < numfluidphases_; doftoderive++)
    {
      for (int idof = 0; idof < numfluidphases_; idof++)
        diamderivs_[doftoderive] +=
            curderivs[idof] * phasemanager_->pressure_deriv(idof, doftoderive) +
            curderivs[idof + numfluidphases_] * phasemanager_->saturation_deriv(idof, doftoderive);
      if (phasemanager_->porosity_depends_on_fluid())
        diamderivs_[doftoderive] +=
            curderivs[2 * numfluidphases_] * phasemanager_->porosity_deriv(doftoderive);
    }
    // diameter derivs w.r.t. to volume fraction phases
    for (int doftoderive = numfluidphases_; doftoderive < numdof_cont_ - numvolfrac_; doftoderive++)
    {
      // diameter derivatives w.r.t. volume fractions directly appearing
      //                             and porosity (since it depends on volfrac)
      diamderivs_[doftoderive] +=
          curderivs[doftoderive + numfluidphases_ + 1] +
          curderivs[2 * numfluidphases_] * phasemanager_->porosity_deriv(doftoderive);
    }
    // diameter derivs w.r.t. to volume fraction pressures
    for (int doftoderive = numfluidphases_ + numvolfrac_; doftoderive < numdof_cont_; doftoderive++)
      diamderivs_[doftoderive] += curderivs[doftoderive + numfluidphases_ + 1];

    // diameter derivs w.r.t. to artery pressure
    diamderivs_[numfluidphases_ + 2 * numvolfrac_] +=
        curderivs[numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_];


    // Now the derivative of the integrated (element) diameter needed in the Hagen-Poiseuille
    // terms is built and stored in the respective stiffness matrices for later use

    // pre-compute some values
    const double pressgrad = (artelephinp_[1] - artelephinp_[0]) / arteryelelength_;
    std::vector<double> dummy = {-1.0, 1.0};
    const double preprefac = pressgrad * w_gp * jacobi / arteryelelength_;

    // assemble into the respective stiffness matrices
    for (unsigned int i = 0; i < numnodesart_; i++)
    {
      // build diameter stiffness matrix w.r.t. artery primary variables
      const double prefac_art =
          dummy[i] * diamderivs_[numfluidphases_ + 2 * numvolfrac_] * preprefac;
      for (unsigned int j = 0; j < numnodesart_; j++) diam_stiffmat11_(i, j) += prefac_art * N1(j);

      // build diameter stiffness matrix w.r.t. 2D/3D primary variables
      const double prefac_cont = dummy[i] * preprefac;
      for (unsigned int j = 0; j < numnodescont_; j++)
      {
        const double prefac_cont2 = prefac_cont * N2(j);
        for (int j_cont = 0; j_cont < numdof_cont_; j_cont++)
          diam_stiffmat12_(i, j * numdof_cont_ + j_cont) += prefac_cont2 * diamderivs_[j_cont];
      }
    }
  }
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_time_fac_rhs(const double& arterydensity, Mat::MatList& contscatramat,
    const double& timefacrhs_art, const double& timefacrhs_cont)
{
  // set
  timefacrhs_art_ = timefacrhs_art;
  timefacrhs_cont_ = timefacrhs_cont;

  // resize
  timefacrhs_cont_dens_.resize(numdof_cont_);

  // fill fluid densities vector
  std::vector<double> fluiddensities(numfluidphases_ + 2 * numvolfrac_);
  for (int ifluidphase = 0; ifluidphase < numfluidphases_; ifluidphase++)
    fluiddensities[ifluidphase] = phasemanager_->density(ifluidphase);
  for (int ivolfrac = 0; ivolfrac < numvolfrac_; ivolfrac++)
  {
    fluiddensities[numfluidphases_ + ivolfrac] = phasemanager_->vol_frac_density(ivolfrac);
    fluiddensities[numfluidphases_ + numvolfrac_ + ivolfrac] =
        phasemanager_->vol_frac_density(ivolfrac);
  }

  switch (coupltype_)
  {
    case type_porofluid:
    {
      // artery
      timefacrhs_art_dens_ = timefacrhs_art_ / arterydensity;
      // continuous
      for (int idof = 0; idof < numdof_cont_; idof++)
        timefacrhs_cont_dens_[idof] = timefacrhs_cont_ / fluiddensities[idof];

      break;
    }
    case type_scatra:
    {
      // artery
      timefacrhs_art_dens_ = timefacrhs_art_ / arterydensity;
      // continuous
      for (int idof = 0; idof < numdof_cont_; idof++)
      {
        const int matid = contscatramat.mat_id(idof);
        std::shared_ptr<Core::Mat::Material> singlemat = contscatramat.material_by_id(matid);
        int phaseid = -1;
        if (singlemat->material_type() == Core::Materials::m_scatra_multiporo_fluid)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(singlemat);
          phaseid = poromat->phase_id();
        }
        else if (singlemat->material_type() == Core::Materials::m_scatra_multiporo_volfrac)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(singlemat);
          phaseid = poromat->phase_id();
        }
        else
          FOUR_C_THROW(
              "Only Mat::ScatraMatMultiPoroVolFrac and Mat::ScatraMatMultiPoroFluid, your "
              "material "
              "is of type {}",
              singlemat->material_type());
        timefacrhs_cont_dens_[idof] = timefacrhs_cont_ / fluiddensities[phaseid];
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::extract_solid_vel(Core::FE::Discretization& contdis)
{
  // no need for this
  if (evaluate_in_ref_config_) return;

  std::shared_ptr<const Core::LinAlg::Vector<double>> velocity =
      contdis.get_state(1, "velocity field");
  Core::Elements::LocationArray la(contdis.num_dof_sets());
  element2_->location_vector(contdis, la, false);

  // construct location vector for displacement related dofs
  std::vector<int> lmdisp(numdim_ * numnodescont_, -1);
  for (unsigned int inode = 0; inode < numnodescont_; ++inode)
    for (unsigned int idim = 0; idim < numdim_; ++idim)
      lmdisp[inode * numdim_ + idim] = la[1].lm_[inode * numdim_ + idim];

  // extract local values of displacement field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<numdim_, numnodescont_>>(
      *velocity, ele2vel_, lmdisp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
FAD PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::integrate_length_to_eta_s(const FAD& eta_s)
{
  FAD length = 0.0;

  // define GPs
  Core::FE::IntegrationPoints1D gaussPoints =
      Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
  if (numdim_ == 3) gaussPoints = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

  static Core::LinAlg::Matrix<1, numnodescont_, FAD> N2(
      Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_, FAD> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1

  static Core::LinAlg::Matrix<numdim_, numdim_, FAD> InvJ(
      Core::LinAlg::Initialization::zero);  // (dX/dxi)^-1
  static Core::LinAlg::Matrix<numdim_, numdim_, FAD> defGrad(
      Core::LinAlg::Initialization::zero);  // (dX/dx) = F
  static Core::LinAlg::Matrix<numdim_, numnodescont_, FAD> N2_XYZ(
      Core::LinAlg::Initialization::zero);                                               // = N2,X
  static Core::LinAlg::Matrix<numdim_, 1, FAD> Ft0(Core::LinAlg::Initialization::zero);  // = F*t0

  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, FAD> N1(Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, FAD> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta
  // Coords and derivatives of 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1, FAD> r1(Core::LinAlg::Initialization::zero);  // = r1
  static Core::LinAlg::Matrix<numdim_, 1, FAD> r1_eta(
      Core::LinAlg::Initialization::zero);  // = r1,eta

  // t0
  static Core::LinAlg::Matrix<numdim_, 1, FAD> t0;
  for (unsigned int i = 0; i < numdim_; i++) t0(i).val() = lambda0_(i);
  // ele2posref
  static Core::LinAlg::Matrix<numdim_, numnodescont_, FAD> ele2posref;
  for (unsigned int i = 0; i < numdim_; i++)
    for (unsigned int j = 0; j < numnodescont_; j++) ele2posref(i, j).val() = ele2posref_(i, j);
  // ele2pos
  static Core::LinAlg::Matrix<numdim_, numnodescont_, FAD> ele2pos;
  for (unsigned int i = 0; i < numdim_; i++)
    for (unsigned int j = 0; j < numnodescont_; j++) ele2pos(i, j).val() = ele2pos_(i, j);

  const FAD determinant = (eta_s - eta_a_) / 2.0;
  const FAD jacobi = determinant * arteryelelengthref_ / 2.0;

  // integrate from etaA to eta_s
  for (int i_gp = 0; i_gp < n_gp_; i_gp++)
  {
    const double w_gp = wgp_[i_gp];
    const FAD eta = (eta_s + eta_a_) / 2.0 + gaussPoints.qxg[i_gp][0] * determinant;
    // Update coordinates and derivatives for 1D and 2D/3D element
    get_1d_shape_functions<FAD>(N1, N1_eta, eta);
    compute_artery_coords_and_derivs_ref<FAD>(r1, r1_eta, N1, N1_eta);

    // project
    bool projection_valid = false;
    std::vector<FAD> xi(numdim_, 0.0);
    projection<FAD>(r1, xi, projection_valid);
    if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");

    get_2d_3d_shape_functions<FAD>(N2, N2_xi, xi);

    InvJ.multiply_nt(N2_xi, ele2posref);
    InvJ.invert();

    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    N2_XYZ.multiply(InvJ, N2_xi);
    // dx/dX = x * N_XYZ^T
    defGrad.multiply_nt(ele2pos, N2_XYZ);
    Ft0.multiply(defGrad, t0);
    const FAD Ft0Norm = Core::FADUtils::vector_norm(Ft0);
    // finally get the length
    length += Ft0Norm * w_gp * jacobi;
  }

  return length;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::get_artery_values_at_gp(const Core::LinAlg::Matrix<1, numnodesart_>& N1, double& artpress,
    std::vector<double>& artscalar)
{
  switch (coupltype_)
  {
    case type_porofluid:
    {
      for (unsigned int i = 0; i < numnodesart_; i++)
      {
        artpress += N1(i) * artelephinp_[i];
        for (int i_scal = 0; i_scal < numscalart_; i_scal++)
          artscalar[i_scal] += N1(i) * eartscalarnp_[i_scal](i);
      }
      break;
    }
    case type_scatra:
    {
      for (unsigned int i = 0; i < numnodesart_; i++)
      {
        artpress += N1(i) * earterypressurenp_(i);
        for (int i_art = 0; i_art < numdof_art_; i_art++)
          artscalar[i_art] += N1(i) * artelephinp_[i * numdof_art_ + i_art];
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::get_cont_scalar_values_at_gp(const Core::LinAlg::Matrix<1, numnodescont_>& N2,
    std::vector<double>& contscalarnp)
{
  switch (coupltype_)
  {
    case type_porofluid:
    {
      for (unsigned int i = 0; i < numnodescont_; i++)
      {
        for (int i_cont = 0; i_cont < numscalcont_; i_cont++)
          contscalarnp[i_cont] += N2(i) * econtscalarnp_[i_cont](i);
      }
      break;
    }
    case type_scatra:
    {
      for (unsigned int i = 0; i < numnodescont_; i++)
      {
        for (int i_cont = 0; i_cont < numdof_cont_; i_cont++)
          contscalarnp[i_cont] += N2(i) * contelephinp_[i * numdof_cont_ + i_cont];
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::assemble_function_coupling_into_force_stiff_art(const int& i_art, const double& w_gp,
    const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi, const int& scale,
    const double& functval, const std::vector<double>& artderivs,
    const std::vector<double>& contderivs, Core::LinAlg::SerialDenseVector& forcevec1,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12)
{
  const double myscale = (double)(scale);

  // rhs ---> +
  for (unsigned int i = 0; i < numnodesart_; i++)
    forcevec1(i * numdof_art_ + i_art) +=
        N1(i) * myscale * w_gp * jacobi * functval * timefacrhs_art_dens_;

  // matrix --> -
  for (unsigned int i = 0; i < numnodesart_; i++)
    for (unsigned int j = 0; j < numnodesart_; j++)
      for (int j_art = 0; j_art < numdof_art_; j_art++)
        stiffmat11(i * numdof_art_ + i_art, j * numdof_art_ + j_art) -=
            N1(i) * N1(j) * myscale * w_gp * jacobi * artderivs[j_art] * timefacrhs_art_dens_;

  // matrix --> -
  for (unsigned int i = 0; i < numnodesart_; i++)
    for (unsigned int j = 0; j < numnodescont_; j++)
      for (int j_cont = 0; j_cont < numdof_cont_; j_cont++)
        stiffmat12(i * numdof_art_ + i_art, j * numdof_cont_ + j_cont) -=
            N1(i) * N2(j) * myscale * w_gp * jacobi * contderivs[j_cont] * timefacrhs_art_dens_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::assemble_function_coupling_into_force_stiff_cont(const std::vector<int>& assembleInto,
    const double& w_gp, const Core::LinAlg::Matrix<1, numnodesart_>& N1,
    const Core::LinAlg::Matrix<1, numnodescont_>& N2, const double& jacobi, const int& scale,
    const double& timefacrhs_cont, const double& functval, const std::vector<double>& artderivs,
    const std::vector<double>& contderivs, Core::LinAlg::SerialDenseVector& forcevec2,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  const double myscale = (double)(scale);

  // rhs ---> +
  for (unsigned int i = 0; i < numnodescont_; i++)
  {
    const double rhsval = N2(i) * myscale * w_gp * jacobi * functval * timefacrhs_cont;
    for (unsigned int idof = 0; idof < assembleInto.size(); idof++)
      forcevec2(i * numdof_cont_ + assembleInto[idof]) += rhsval;
  }

  // matrix --> -
  for (unsigned int i = 0; i < numnodescont_; i++)
    for (unsigned int j = 0; j < numnodesart_; j++)
    {
      const double massmatrixfac = N2(i) * N1(j) * myscale * w_gp * jacobi * timefacrhs_cont;
      for (int j_art = 0; j_art < numdof_art_; j_art++)
      {
        const double stiffval = massmatrixfac * artderivs[j_art];
        for (unsigned int idof = 0; idof < assembleInto.size(); idof++)
          stiffmat21(i * numdof_cont_ + assembleInto[idof], j * numdof_art_ + j_art) -= stiffval;
      }
    }

  // matrix --> -
  for (unsigned int i = 0; i < numnodescont_; i++)
    for (unsigned int j = 0; j < numnodescont_; j++)
    {
      const double massmatrixfac = N2(i) * N2(j) * myscale * w_gp * jacobi * timefacrhs_cont;
      for (int j_cont = 0; j_cont < numdof_cont_; j_cont++)
      {
        const double stiffval = massmatrixfac * contderivs[j_cont];
        for (unsigned int idof = 0; idof < assembleInto.size(); idof++)
          stiffmat22(i * numdof_cont_ + assembleInto[idof], j * numdof_cont_ + j_cont) -= stiffval;
      }
    }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_function_and_deriv(const Core::Utils::FunctionOfAnything& funct,
    const double& artpressnpAtGP, const std::vector<double>& artscalarnpAtGP,
    const std::vector<double>& scalarnpAtGP, double& functval, std::vector<double>& artderivs,
    std::vector<double>& contderivs)
{
  double time = Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time();

  switch (coupltype_)
  {
    case type_porofluid:
    {
      // we have to derive w.r.t. fluid variables plus diameter
      std::vector<std::pair<std::string, double>> variables;
      variables.reserve(numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_ + 1 + 1);

      // scalar variables are constants (plus reference artery diameter and diameter of previous
      // time step)
      std::vector<std::pair<std::string, double>> constants;
      constants.reserve(numscalcont_ + numscalart_ + 2);

      set_scalar_values_as_constants(constants, artscalarnpAtGP, scalarnpAtGP);

      set_fluid_values_as_variables(variables, artpressnpAtGP);

      // set reference artery diameter as constant
      constants.emplace_back("D0", arterydiamref_);

      // set artery diameter of previous time step as constant
      constants.emplace_back("Dprev", arterymat_->diam_previous_time_step());

      // set artery diameter as variable
      variables.emplace_back("D", arterydiam_at_gp_);

      constants.emplace_back("t", time);

      // evaluate the reaction term
      functval = funct.evaluate(variables, constants, 0);
      // evaluate derivatives
      std::vector<double> curderivs(funct.evaluate_derivative(variables, constants, 0));

      evaluate_fluid_derivs(artderivs, contderivs, curderivs);

      break;
    }
    case type_scatra:
    {
      // scalars (both cont and art) are variables
      std::vector<std::pair<std::string, double>> variables;
      variables.reserve(numscalcont_ + numscalart_);

      // fluid variables are constants
      std::vector<std::pair<std::string, double>> constants;
      constants.reserve(
          numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_ + 1 + 1 + 1 + 1);

      set_scalar_values_as_variables(variables, artscalarnpAtGP, scalarnpAtGP);

      set_fluid_values_as_constants(constants, artpressnpAtGP);

      constants.emplace_back("t", time);

      // evaluate the reaction term
      functval = funct.evaluate(variables, constants, 0);
      // evaluate derivatives
      std::vector<double> curderivs(funct.evaluate_derivative(variables, constants, 0));

      evaluate_scalar_derivs(artderivs, contderivs, curderivs);

      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_scalar_values_as_constants(std::vector<std::pair<std::string, double>>& constants,
    const std::vector<double>& artscalarnpAtGP, const std::vector<double>& scalarnpAtGP)
{
  // set scalar values as constant
  for (int k = 0; k < numscalcont_; k++)
    constants.push_back(std::pair<std::string, double>(scalarnames_[k], scalarnpAtGP[k]));

  // set artery-scalar values as constant
  for (int k = 0; k < numscalart_; k++)
    constants.push_back(std::pair<std::string, double>(artscalarnames_[k], artscalarnpAtGP[k]));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_fluid_values_as_variables(std::vector<std::pair<std::string, double>>& variables,
    const double& artpressnpAtGP)
{
  // set pressure values as variable
  for (int k = 0; k < numfluidphases_; k++)
    variables.push_back(
        std::pair<std::string, double>(pressurenames_[k], phasemanager_->pressure(k)));

  // set saturation values as variable
  for (int k = 0; k < numfluidphases_; k++)
    variables.push_back(
        std::pair<std::string, double>(saturationnames_[k], phasemanager_->saturation(k)));

  // set porosity value as variable
  variables.push_back(std::pair<std::string, double>(porosityname_, phasemanager_->porosity()));

  // set volfrac values as variables
  for (int k = 0; k < numvolfrac_; k++)
    variables.push_back(
        std::pair<std::string, double>(volfracnames_[k], phasemanager_->vol_frac(k)));

  // set volfrac pressure values as variables
  for (int k = 0; k < numvolfrac_; k++)
    variables.push_back(std::pair<std::string, double>(
        volfracpressurenames_[k], phasemanager_->vol_frac_pressure(k)));

  // set artery pressure as variable
  variables.push_back(std::pair<std::string, double>(artpressname_, artpressnpAtGP));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_fluid_values_as_constants(std::vector<std::pair<std::string, double>>& constants,
    const double& artpressnpAtGP)
{
  // set pressure values as constants
  for (int k = 0; k < numfluidphases_; k++)
    constants.push_back(
        std::pair<std::string, double>(pressurenames_[k], phasemanager_->pressure(k)));

  // set saturation values as constants
  for (int k = 0; k < numfluidphases_; k++)
    constants.push_back(
        std::pair<std::string, double>(saturationnames_[k], phasemanager_->saturation(k)));

  // set porosity value as constants
  constants.push_back(std::pair<std::string, double>(porosityname_, phasemanager_->porosity()));

  // set volfrac values as constants
  for (int k = 0; k < numvolfrac_; k++)
    constants.push_back(
        std::pair<std::string, double>(volfracnames_[k], phasemanager_->vol_frac(k)));

  // set volfrac pressure values as constants
  for (int k = 0; k < numvolfrac_; k++)
    constants.push_back(std::pair<std::string, double>(
        volfracpressurenames_[k], phasemanager_->vol_frac_pressure(k)));

  // set artery pressure as constant
  constants.push_back(std::pair<std::string, double>(artpressname_, artpressnpAtGP));

  // set artery diameter as constant
  constants.push_back(std::pair<std::string, double>("D", arterydiam_at_gp_));

  // set reference artery diameter as constant
  constants.push_back(std::pair<std::string, double>("D0", arterydiamref_));

  // set artery diameter of previous time step as constant
  constants.push_back(
      std::pair<std::string, double>("Dprev", arterymat_->diam_previous_time_step()));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::set_scalar_values_as_variables(std::vector<std::pair<std::string, double>>& variables,
    const std::vector<double>& artscalarnpAtGP, const std::vector<double>& scalarnpAtGP)
{
  // set scalar values as variables
  for (int k = 0; k < numscalcont_; k++)
    variables.push_back(std::pair<std::string, double>(scalarnames_[k], scalarnpAtGP[k]));

  // set artery-scalar values as variables
  for (int k = 0; k < numscalart_; k++)
    variables.push_back(std::pair<std::string, double>(artscalarnames_[k], artscalarnpAtGP[k]));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_fluid_derivs(std::vector<double>& artderivs, std::vector<double>& contderivs,
    const std::vector<double>& functderivs)
{
  // basically dependency of exchange terms f is as follows:
  // f(D(p),g(p)) where p are generic 1D and 3D fluid primary variables
  // so derivative is df/dp = df/dg*dg/dp + df/dD*dD/dp
  // first part
  // function derivs w.r.t. to fluid phases: df/dg*dp/dp
  for (int doftoderive = 0; doftoderive < numfluidphases_; doftoderive++)
  {
    for (int idof = 0; idof < numfluidphases_; idof++)
      contderivs[doftoderive] +=
          functderivs[idof] * phasemanager_->pressure_deriv(idof, doftoderive) +
          functderivs[idof + numfluidphases_] * phasemanager_->saturation_deriv(idof, doftoderive);
    if (phasemanager_->porosity_depends_on_fluid())
      contderivs[doftoderive] +=
          functderivs[2 * numfluidphases_] * phasemanager_->porosity_deriv(doftoderive);
  }
  // function derivs w.r.t. to volume fraction phases
  for (int doftoderive = numfluidphases_; doftoderive < numdof_cont_ - numvolfrac_; doftoderive++)
  {
    // derivatives w.r.t. volume fractions directly appearing
    //                and porosity (since it depends on volfrac)
    contderivs[doftoderive] +=
        functderivs[doftoderive + numfluidphases_ + 1] +
        functderivs[2 * numfluidphases_] * phasemanager_->porosity_deriv(doftoderive);
  }
  // function derivs w.r.t. to volume fraction pressures
  for (int doftoderive = numfluidphases_ + numvolfrac_; doftoderive < numdof_cont_; doftoderive++)
    contderivs[doftoderive] += functderivs[doftoderive + numfluidphases_ + 1];

  // function derivs w.r.t. to artery pressure
  artderivs[0] += functderivs[numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_];

  // second part
  // derivatives w.r.t. to fluid diameter: df/dD*dD/dp
  if (diam_funct_active_)
  {
    // df/dD
    const double functderivdiam =
        functderivs[numfluidphases_ + numfluidphases_ + 1 + numvolfrac_ + numvolfrac_ + 1];

    // now follow the terms df/dD*dD/dp
    for (int doftoderive = 0; doftoderive < numdof_cont_; doftoderive++)
      contderivs[doftoderive] += functderivdiam * diamderivs_[doftoderive];

    // diameter derivs w.r.t. to artery pressure
    artderivs[0] += functderivdiam * diamderivs_[numfluidphases_ + 2 * numvolfrac_];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::evaluate_scalar_derivs(std::vector<double>& artderivs, std::vector<double>& contderivs,
    const std::vector<double>& functderivs)
{
  // derivatives after continuous scalars
  for (int doftoderive = 0; doftoderive < numdof_cont_; doftoderive++)
    contderivs[doftoderive] += functderivs[doftoderive];

  // derivatives after artery scalars
  for (int doftoderive = 0; doftoderive < numdof_art_; doftoderive++)
    artderivs[doftoderive] += functderivs[doftoderive + numdof_cont_];
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::create_integration_segment()
{
  if (PROJOUTPUT)
  {
    std::cout << "========================= getting intersection for artery ele " << ele1_gid()
              << std::endl;
  }

  // get the intersections
  std::vector<double> intersections = get_all_inter_sections();
  const int numintersections = intersections.size();

  if (PROJOUTPUT)
  {
    if (numintersections == 0)
      std::cout << "no intersections found" << std::endl;
    else
    {
      std::cout << "intersections are:" << std::endl;
      for (auto i = intersections.begin(); i != intersections.end(); ++i) std::cout << *i << " ";
      std::cout << " " << std::endl;
    }
  }

  std::vector<double> xi(numdim_);
  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1(
      Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, double> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta
  // Coords and derivatives of 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1, double> r1(Core::LinAlg::Initialization::zero);  // = r1
  static Core::LinAlg::Matrix<numdim_, 1, double> r1_eta(
      Core::LinAlg::Initialization::zero);  // = r1,eta
  bool projection_valid = false;

  // 1st case: no intersection found
  if (numintersections == 0)
  {
    get_1d_shape_functions<double>(N1, N1_eta, 0.0);
    compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
    projection<double>(r1, xi, projection_valid);
    // case: completely inside
    if (projection_valid)
    {
      eta_a_ = -1.0;
      eta_b_ = 1.0;
      isactive_ = true;
    }
    // case: completely outside
    else
      isactive_ = false;
  }
  // 2nd case: 1 intersection found
  else if (numintersections == 1)
  {
    // special case: eta = -1.0 lies directly at boundary of 3D element:
    if (fabs(intersections[0] + 1.0) < XIETATOL)
    {
      // first possibility: segment goes from [-1; 1], second point lies inside 3D element
      get_1d_shape_functions<double>(N1, N1_eta, 1.0);
      compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
      projection<double>(r1, xi, projection_valid);
      if (projection_valid)
      {
        eta_a_ = -1.0;
        eta_b_ = 1.0;
        isactive_ = true;
      }
      else
      {
        // we have found a segment between [-1.0; -1.0 +  XIETATOL] --> this can be sorted out
        if (PROJOUTPUT)
        {
          std::cout << "probably found a very small integration segment for artery element "
                    << ele1_gid() << " and 3D element " << ele2_gid() << " which is sorted out!"
                    << std::endl;
        }
        isactive_ = false;
      }
    }
    // special case: eta = 1.0 lies directly at boundary of 3D element:
    else if (fabs(intersections[0] - 1.0) < XIETATOL)
    {
      // first possibility: segment goes from [-1; 1], second point lies inside 3D element
      get_1d_shape_functions<double>(N1, N1_eta, -1.0);
      compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
      projection<double>(r1, xi, projection_valid);
      if (projection_valid)
      {
        eta_a_ = -1.0;
        eta_b_ = 1.0;
        isactive_ = true;
      }
      else
      {
        // we have found a segment between [1.0 - XIETATOL; 1.0] --> this can be sorted out
        if (PROJOUTPUT)
        {
          std::cout << "probably found a very small integration segment for artery element "
                    << ele1_gid() << " and 3D element " << ele2_gid() << " which is sorted out!"
                    << std::endl;
        }
        isactive_ = false;
      }
    }
    // normal case: found one intersection: check if -1.0 or 1.0 are inside
    else
    {
      get_1d_shape_functions<double>(N1, N1_eta, -1.0);
      compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
      projection<double>(r1, xi, projection_valid);
      // case: segment goes from [-1.0; intersections[0]]
      if (projection_valid)
      {
        eta_a_ = -1.0;
        eta_b_ = intersections[0];
        isactive_ = true;
      }
      else
      {
        // case: segment goes from [intersections[0]; 1.0]
        get_1d_shape_functions<double>(N1, N1_eta, 1.0);
        compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
        projection<double>(r1, xi, projection_valid);
        eta_a_ = intersections[0];
        eta_b_ = 1.0;
        isactive_ = true;
        // special case: projection lies directly at corner of element, this can be sorted out
        if (!projection_valid)
        {
          if (PROJOUTPUT)
          {
            std::cout << "original point " << intersections[0] << std::endl;
            std::cout << "Neither -1.0 nor 1.0 could be projected" << std::endl;
          }
          isactive_ = false;
        }
      }
    }
  }
  // 3rd case: two intersections found
  else if (numintersections == 2)
  {
    eta_a_ = intersections[0];
    eta_b_ = intersections[1];
    isactive_ = true;
  }
  // rest is not possible
  else
    FOUR_C_THROW(
        "Found more than two intersections for artery element {} and 2D/3D element {}, this "
        "should "
        "not be possible",
        ele1_gid(), ele2_gid());

  // safety checks
  if (isactive_)
  {
    if (eta_a_ > eta_b_)
      FOUR_C_THROW(
          "something went terribly wrong for artery element {} and 2D/3D element {}, eta_a is "
          "bigger than eta_b",
          ele1_gid(), ele2_gid());
    if (fabs(eta_a_ - eta_b_) < XIETATOL)
      FOUR_C_THROW(
          "something went terribly wrong for artery element {} and 2D/3D element {}, found "
          "extremely small integration segment",
          ele1_gid(), ele2_gid());
  }

  return;
}
/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
std::vector<double> PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art,
    distype_cont, dim>::get_all_inter_sections()
{
  std::vector<double> intersections(0);

  std::vector<double> xi(numdim_, 0.0);
  double eta = 0.0;
  bool projection_valid = true;

  switch (element2_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      for (unsigned int j = 0; j < numdim_; j++)
      {
        // project edge xi1 or xi2 = 1.0
        inter_sect_with_2d_3d(xi, eta, j, 1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, eta))
          intersections.push_back(eta);

        // project edge xi1 or xi2 = -1.0
        inter_sect_with_2d_3d(xi, eta, j, -1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, eta))
          intersections.push_back(eta);
      }
      break;
    }
    case Core::FE::CellType::hex8:
    {
      for (unsigned int j = 0; j < numdim_; j++)
      {
        // project surface xi1 or xi2 or xi3 = 1.0
        inter_sect_with_2d_3d(xi, eta, j, 1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, eta))
          intersections.push_back(eta);

        // project surface xi1 or xi2 or xi3 = -1.0
        inter_sect_with_2d_3d(xi, eta, j, -1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, eta))
          intersections.push_back(eta);
      }
      break;
    }
    case Core::FE::CellType::tet4:
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        // project surface xi1 or xi2 or xi3 = 0.0
        inter_sect_with_2d_3d(xi, eta, j, 0.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, eta))
          intersections.push_back(eta);
      }
      // project fourth surface of tetahedron xi1 + xi2 + xi3 = 1
      inter_sect_with_2d_3d(xi, eta, 3, 0.0, projection_valid);
      if (projection_valid && projection_not_yet_found(intersections, eta))
        intersections.push_back(eta);
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
  }

  std::sort(intersections.begin(), intersections.end());

  return intersections;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
bool PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::projection_not_yet_found(const std::vector<double>& intersections, const double& eta)
{
  for (unsigned int i = 0; i < intersections.size(); i++)
  {
    if (fabs(intersections[i] - eta) < XIETATOL)
    {
      if (PROJOUTPUT) std::cout << "duplicate intersection found" << std::endl;
      return false;
    }
  }
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::inter_sect_with_2d_3d(std::vector<double>& xi, double& eta, const int& fixedPar,
    const double& fixedAt, bool& projection_valid)
{
  projection_valid = true;
  bool parallel = false;

  // Initialize limit for parameter values (interval [-limit, limit])
  const double limit = 1.0 + XIETATOL;

  // reset iteration variables
  eta = 0.0;
  switch (element2_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fixedPar == 0)  // xi1 fixed
      {
        xi[0] = fixedAt;
        xi[1] = 0.0;
      }
      else if (fixedPar == 1)  // xi2 fixed
      {
        xi[0] = 0.0;
        xi[1] = fixedAt;
      }
      else
        FOUR_C_THROW("wrong input for fixedPar");
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fixedPar == 0)  // xi1 fixed
      {
        xi[0] = fixedAt;
        xi[1] = 0.0;
        xi[2] = 0.0;
      }
      else if (fixedPar == 1)  // xi2 fixed
      {
        xi[0] = 0.0;
        xi[1] = fixedAt;
        xi[2] = 0.0;
      }
      else if (fixedPar == 2)  // xi3 fixed
      {
        xi[0] = 0.0;
        xi[1] = 0.0;
        xi[2] = fixedAt;
      }
      else
        FOUR_C_THROW("wrong input for fixedPar");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (fixedPar == 0)  // xi1 fixed
      {
        xi[0] = fixedAt;
        xi[1] = 0.25;
        xi[2] = 0.25;
      }
      else if (fixedPar == 1)  // xi2 fixed
      {
        xi[0] = 0.25;
        xi[1] = fixedAt;
        xi[2] = 0.25;
      }
      else if (fixedPar == 2)  // xi3 fixed
      {
        xi[0] = 0.25;
        xi[1] = 0.25;
        xi[2] = fixedAt;
      }
      else if (fixedPar == 3)  // fourth surface xi1 + xi2 + xi3 = 1 fixed
      {
        xi[0] = 0.25;
        xi[1] = 0.25;
        xi[2] = 0.5;
      }
      else
        FOUR_C_THROW(
            "only fixedPar = 0, fixedPar = 1, fixedPar = 2 or fixedPar = 3 possible (for tet "
            "elements");
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
  }

  if (PROJOUTPUT)
  {
    std::cout << "Projection output:" << std::endl;
    std::cout << "Start parameters eta: " << eta;
    switch (element2_->shape())
    {
        // 2D case
      case Core::FE::CellType::quad4:
      {
        std::cout << ", xi1: " << xi[0] << ", xi2: " << xi[1] << std::endl;
        break;
      }
        // 3D case
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::tet4:
      {
        std::cout << ", xi1: " << xi[0] << ", xi2: " << xi[1] << ", xi3: " << xi[2] << std::endl;
        break;
      }
      default:
        FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
    }
  }

  // Initialize function f and Jacobian J for Newton iteration
  Core::LinAlg::Matrix<numdim_, 1> f(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_> J(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_> Jinv(Core::LinAlg::Initialization::zero);

  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_> N1(Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta

  static Core::LinAlg::Matrix<1, numnodescont_> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1

  // Coords and derivatives of for 1D and 2D/3D element
  static Core::LinAlg::Matrix<numdim_, 1> r1(Core::LinAlg::Initialization::zero);      // = r1
  static Core::LinAlg::Matrix<numdim_, 1> r1_eta(Core::LinAlg::Initialization::zero);  // = r1,eta

  static Core::LinAlg::Matrix<numdim_, 1> x2(Core::LinAlg::Initialization::zero);  // = x2
  static Core::LinAlg::Matrix<numdim_, numdim_> x2_xi(
      Core::LinAlg::Initialization::zero);  // = x2,xi

  // Initial scalar residual (L2-norm of f)
  double residual;

  // Local newton iteration
  // -----------------------------------------------------------------
  int iter;
  double first_residual = 1.0e-4;  // used for convergence check

  for (iter = 0; iter < PROJMAXITER; iter++)
  {
    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_1d_shape_functions<double>(N1, N1_eta, eta);
    get_2d_3d_shape_functions<double>(N2, N2_xi, xi);

    // Update coordinates and derivatives for 1D and 2D/3D element
    compute_artery_coords_and_derivs_ref<double>(r1, r1_eta, N1, N1_eta);
    compute_2d_3d_coords_and_derivs_ref<double>(x2, x2_xi, N2, N2_xi);

    // Evaluate f at current xi1, xi2, alpha
    f.clear();
    for (unsigned int i = 0; i < numdim_; i++) f(i) = x2(i) - r1(i);

    // Compute scalar residuum
    residual = 0.0;
    for (unsigned int i = 0; i < numdim_; i++) residual += f(i) * f(i);
    residual = sqrt(residual);
    if (iter == 0) first_residual = std::max(first_residual, residual);

    J.clear();

    if (fixedPar == 0)  // xi1 fixed --> we need x_{,xi2} (and x_{,xi3} in case of 3D)
    {
      for (unsigned int idim = 0; idim < numdim_; idim++)
        for (unsigned int jdim = 1; jdim < numdim_; jdim++) J(idim, jdim - 1) = x2_xi(idim, jdim);
    }
    else if (fixedPar == 1)  // xi2 fixed --> we need x_{,xi1} (and x_{,xi3} in case of 3D)
    {
      switch (element2_->shape())
      {
          // 2D case
        case Core::FE::CellType::quad4:
        {
          for (unsigned int jdim = 0; jdim < numdim_; jdim++) J(jdim, 0) = x2_xi(jdim, 0);
          break;
        }
          // 3D case
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        {
          for (unsigned int jdim = 0; jdim < numdim_; jdim++)
          {
            J(jdim, 0) = x2_xi(jdim, 0);
            J(jdim, 1) = x2_xi(jdim, 2);
          }
          break;
        }
        default:
          FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
      }
    }
    else if (fixedPar == 2)  // xi3 fixed  --> we need x_{,xi1} (and x_{,xi2} in case of 3D)
    {
      for (unsigned int idim = 0; idim < numdim_; idim++)
        for (unsigned int jdim = 0; jdim < numdim_ - 1; jdim++) J(idim, jdim) = x2_xi(idim, jdim);
    }
    else if (fixedPar == 3)  // xi3 fixed at xi3 = 1.0 - xi1 - xi2
    {
      for (unsigned int idim = 0; idim < numdim_; idim++)
      {
        // df/dxi1 = df/dxi1 - df/dxi3
        J(idim, 0) = x2_xi(idim, 0) - x2_xi(idim, 2);
        // df/dxi_2 = df/dxi2 - df/dxi3
        J(idim, 1) = x2_xi(idim, 1) - x2_xi(idim, 2);
      }
    }
    else
      FOUR_C_THROW(
          "only fixedPar = 0, fixedPar = 1, fixedPar = 2 or fixedPar = 3 possible (for tet "
          "elements)");

    // fill dr_deta into Jacobian
    for (unsigned int idim = 0; idim < numdim_; idim++) J(idim, numdim_ - 1) = -r1_eta(idim);

    double jacdet = J.determinant();

    // If det_J = 0 we assume, that the artery and the surface edge are parallel.
    // These projection is not needed due the fact that the contact interval can also be
    // identified by other projections
    parallel = fabs(jacdet) < COLINEARTOL * first_residual;
    if (!parallel) jacdet = J.invert();

    // Check if the local Newton iteration has converged
    if (residual < CONVTOLNEWTONPROJ * first_residual && !parallel)
    {
      if (PROJOUTPUT)
      {
        std::cout << "Local Newton iteration converged after " << iter << " iterations"
                  << std::endl;
        std::cout << "Found point at xi1: " << xi[0] << ", xi2: " << xi[1];
        if (numdim_ == 3) std::cout << ", xi3: " << xi[2];
        std::cout << ", eta: " << eta << " with residual: " << residual << std::endl;
        std::cout << "r1:\n" << r1 << ", x2:\n" << x2 << std::endl;
      }
      // Local Newton iteration converged
      break;
    }
    else if (PROJOUTPUT && iter > 0)
    {
      std::cout << "New point at xi1: " << xi[0] << ", xi2: " << xi[0];
      if (numdim_ == 3) std::cout << ", xi3: " << xi[2];
      std::cout << ", eta: " << eta << " with residual: " << residual << std::endl;
    }

    // Singular J
    if (parallel)
    {
      // Sort out
      if (PROJOUTPUT)
      {
        std::cout << "elementscolinear: det_J = " << jacdet << std::endl;
      }
      break;
    }
    // Regular J (inversion possible)

    if (fixedPar == 0)  // xi1 fixed --> we have to update xi2 (and xi3 in case of 3D)
    {
      for (unsigned int idim = 1; idim < numdim_; idim++)
        for (unsigned int jdim = 0; jdim < numdim_; jdim++)
          xi[idim] += -J(idim - 1, jdim) * f(jdim);
    }
    else if (fixedPar == 1)  // xi2 fixed --> we have to update xi1 (and xi3 in case of 3D)
    {
      switch (element2_->shape())
      {
          // 2D case
        case Core::FE::CellType::quad4:
        {
          for (unsigned int jdim = 0; jdim < numdim_; jdim++) xi[0] += -J(0, jdim) * f(jdim);
          break;
        }
          // 3D case
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        {
          for (unsigned int jdim = 0; jdim < numdim_; jdim++)
          {
            xi[0] += -J(0, jdim) * f(jdim);
            xi[2] += -J(1, jdim) * f(jdim);
          }
          break;
        }
        default:
          FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
      }
    }
    else if (fixedPar == 2)  // xi3 fixed --> we have to update xi1 (and xi2 in case of 3D)
    {
      for (unsigned int idim = 0; idim < numdim_ - 1; idim++)
        for (unsigned int jdim = 0; jdim < numdim_; jdim++) xi[idim] += -J(idim, jdim) * f(jdim);
    }
    else if (fixedPar == 3)
    {
      // xi3 fixed at 1.0 - xi1 - xi2
      for (unsigned int idim = 0; idim < numdim_ - 1; idim++)
        for (unsigned int jdim = 0; jdim < numdim_; jdim++) xi[idim] += -J(idim, jdim) * f(jdim);
      xi[2] = 1.0 - xi[0] - xi[1];
    }
    else
      FOUR_C_THROW(
          "only fixedPar = 0, fixedPar = 1, fixedPar = 2 or fixedPar = 3 possible (for tet "
          "elements)");

    // update also eta
    for (unsigned int jdim = 0; jdim < numdim_; jdim++) eta += -J(numdim_ - 1, jdim) * f(jdim);
  }

  // Local Newton iteration unconverged after PROJMAXITER
  if (residual > CONVTOLNEWTONPROJ * first_residual || parallel)
  {
    for (unsigned int idim = 0; idim < numdim_; idim++) xi[idim] = 1e+12;
    eta = 1e+12;

    if (PROJOUTPUT)
      std::cout << "Local Newton iteration unconverged (!) after " << iter + 1 << " iterations"
                << std::endl;
  }

  switch (element2_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fabs(xi[0]) > limit || fabs(xi[1]) > limit || fabs(eta) > limit) projection_valid = false;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fabs(xi[0]) > limit || fabs(xi[1]) > limit || fabs(xi[2]) > limit || fabs(eta) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (xi[0] < -XIETATOL || xi[1] < -XIETATOL || xi[2] < -XIETATOL ||
          xi[0] + xi[1] + xi[2] > limit || fabs(eta) > limit)
        projection_valid = false;
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8 and tet4 are valid so far for second element");
  }

  if (PROJOUTPUT)
  {
    if (projection_valid)
      std::cout << "Projection allowed" << std::endl;
    else
      std::cout << "Projection not allowed" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
template <typename T>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::projection(Core::LinAlg::Matrix<numdim_, 1, T>& r1, std::vector<T>& xi,
    bool& projection_valid)
{
  projection_valid = true;
  bool parallel = false;

  // Initialize limit for parameter values (interval [-limit, limit])
  const double limit = 1.0 + XIETATOL;

  switch (element2_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      xi[0] = 0.0;
      xi[1] = 0.0;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      xi[0] = 0.0;
      xi[1] = 0.0;
      xi[2] = 0.0;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      xi[0] = 0.25;
      xi[1] = 0.25;
      xi[2] = 0.25;
      break;
    }
    case Core::FE::CellType::tet10:
    {
      xi[0] = 0.25;
      xi[1] = 0.25;
      xi[2] = 0.25;
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
  }

  if (PROJOUTPUT)
  {
    std::cout << "Projection output:" << std::endl;
    std::cout << "Start parameters ";
    switch (element2_->shape())
    {
        // 2D case
      case Core::FE::CellType::quad4:
      {
        std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << std::endl;
        break;
      }
        // 3D case
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << ", xi3: " << xi[2] << std::endl;
        break;
      }
      default:
        FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
    }
  }

  // Initialize function f and Jacobian J for Newton iteration
  Core::LinAlg::Matrix<numdim_, 1, T> f(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_, T> J(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_, T> Jinv(Core::LinAlg::Initialization::zero);

  // Vectors for shape functions and their derivatives
  static Core::LinAlg::Matrix<1, numnodesart_, T> N1(Core::LinAlg::Initialization::zero);  // = N1
  static Core::LinAlg::Matrix<1, numnodesart_, T> N1_eta(
      Core::LinAlg::Initialization::zero);  // = N1,eta

  static Core::LinAlg::Matrix<1, numnodescont_, T> N2(Core::LinAlg::Initialization::zero);  // = N2
  static Core::LinAlg::Matrix<numdim_, numnodescont_, T> N2_xi(
      Core::LinAlg::Initialization::zero);  // = N2,xi1

  static Core::LinAlg::Matrix<numdim_, 1, T> x2(Core::LinAlg::Initialization::zero);  // = x2
  static Core::LinAlg::Matrix<numdim_, numdim_, T> x2_xi(
      Core::LinAlg::Initialization::zero);  // = x2,xi

  // Initial scalar residual (L2-norm of f)
  T residual;

  // Local newton iteration
  // -----------------------------------------------------------------

  int iter;
  double first_residual = 1.0e-4;  // used for convergence check

  for (iter = 0; iter < PROJMAXITER; iter++)
  {
    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_2d_3d_shape_functions<T>(N2, N2_xi, xi);

    // Update coordinates and derivatives for 1D and 2D/3D element
    compute_2d_3d_coords_and_derivs_ref<T>(x2, x2_xi, N2, N2_xi);

    // Evaluate f at current xi1, xi2, alpha
    f.clear();
    for (unsigned int i = 0; i < numdim_; i++) f(i) = x2(i) - r1(i);

    residual = Core::FADUtils::vector_norm(f);
    if (iter == 0)
      first_residual = std::max(first_residual, Core::FADUtils::cast_to_double(residual));

    // Reset matrices
    for (unsigned int i = 0; i < numdim_; i++)
      for (unsigned int j = 0; j < numdim_; j++) J(i, j) = x2_xi(i, j);

    const double jacdet = Core::FADUtils::cast_to_double<T, numdim_, numdim_>(J).determinant();

    // If det_J = 0 we assume, that the artery element and the surface edge are parallel.
    // These projection is not needed due the fact that the contact interval can also be
    // identified by two contact interval borders found with the GetContactLines method
    parallel = fabs(jacdet) < COLINEARTOL * first_residual;
    if (!parallel) J.invert();

    // Check if the local Newton iteration has converged
    // If the start point fulfills the orthogonalty conditions (residual < CONVTOLNEWTONPROJ*
    // first_residual), we also check if the artery element and the surface edge are parallel.
    // This is done by calculating det_J before checking if the local Newton iteration has
    // converged by fulfilling the condition residual < CONVTOLNEWTONPROJ*first_residual
    if (residual < CONVTOLNEWTONPROJ * first_residual && !parallel)
    {
      if (PROJOUTPUT)
      {
        std::cout << "Local Newton iteration converged after " << iter << " iterations"
                  << std::endl;
        std::cout << "Found point at ";
        switch (element2_->shape())
        {
            // 2D case
          case Core::FE::CellType::quad4:
          {
            std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << std::endl;
            break;
          }
            // 3D case
          case Core::FE::CellType::hex8:
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::tet10:
          {
            std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << ", xi3: " << xi[2] << std::endl;
            break;
          }
          default:
            FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
        }
        std::cout << " with residual: " << residual << std::endl;
        std::cout << "r1:\n" << r1 << ", x2:\n" << x2 << std::endl;
      }
      // Local Newton iteration converged
      break;
    }
    else if (PROJOUTPUT && iter > 0)
    {
      std::cout << "New point at xi1: ";
      switch (element2_->shape())
      {
          // 2D case
        case Core::FE::CellType::quad4:
        {
          std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << std::endl;
          break;
        }
          // 3D case
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        case Core::FE::CellType::tet10:
        {
          std::cout << "xi1: " << xi[0] << ", xi2: " << xi[1] << ", xi3: " << xi[2] << std::endl;
          break;
        }
        default:
          FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
      }
      std::cout << " with residual: " << residual << std::endl;
    }

    // Singular J
    if (parallel)
    {
      // Sort out
      if (PROJOUTPUT)
      {
        std::cout << "elementscolinear: det_J = " << jacdet << std::endl;
      }
      break;
    }
    // Regular J (inversion possible)
    for (unsigned int idim = 0; idim < numdim_; idim++)
      for (unsigned int jdim = 0; jdim < numdim_; jdim++) xi[idim] += -J(idim, jdim) * f(jdim);

    // xi1 += -J(0, 0) * f(0) - J(0, 1) * f(1) - J(0, 2) * f(2);
    // xi2 += -J(1, 0) * f(0) - J(1, 1) * f(1) - J(1, 2) * f(2);
    // xi3 += -J(2, 0) * f(0) - J(2, 1) * f(1) - J(2, 2) * f(2);
  }
  // -----------------------------------------------------------------
  // End: Local Newton iteration

  // Local Newton iteration unconverged after PROJMAXITER
  if (residual > CONVTOLNEWTONPROJ * first_residual || parallel)
  {
    for (unsigned int idim = 0; idim < numdim_; idim++) xi[idim] = 1e+12;

    if (PROJOUTPUT)
      std::cout << "Local Newton iteration unconverged (!) after " << iter + 1 << " iterations"
                << std::endl;
  }

  // check if xi lies inside element
  switch (element2_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fabs(xi[0]) > limit || fabs(xi[1]) > limit) projection_valid = false;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fabs(xi[0]) > limit || fabs(xi[1]) > limit || fabs(xi[2]) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (xi[0] < -XIETATOL || xi[1] < -XIETATOL || xi[2] < -XIETATOL ||
          xi[0] + xi[1] + xi[2] > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet10:  // TODO: Is this correct?
    {
      if (xi[0] < -XIETATOL || xi[1] < -XIETATOL || xi[2] < -XIETATOL ||
          xi[0] + xi[1] + xi[2] > limit)
        projection_valid = false;
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
  }

  if (PROJOUTPUT)
  {
    if (projection_valid)
      std::cout << "Projection allowed" << std::endl;
    else
      std::cout << "Projection not allowed" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
template <typename T>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::get_1d_shape_functions(Core::LinAlg::Matrix<1, numnodesart_, T>& N1,
    Core::LinAlg::Matrix<1, numnodesart_, T>& N1_eta, const T& eta)
{
  // Clear shape functions and derivatives
  N1.clear();
  N1_eta.clear();

  // Get discretization type
  const Core::FE::CellType distype = element1_->shape();

  // Get values and derivatives of shape functions
  Core::FE::shape_function_1d(N1, eta, distype);
  Core::FE::shape_function_1d_deriv1(N1_eta, eta, distype);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
template <typename T>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::get_2d_3d_shape_functions(Core::LinAlg::Matrix<1, numnodescont_, T>& N2,
    Core::LinAlg::Matrix<numdim_, numnodescont_, T>& N2_xi, const std::vector<T>& xi)
{
  // Clear shape functions and derivatives
  N2.clear();
  N2_xi.clear();

  switch (element2_->shape())
  {
      // 2D case
    case Core::FE::CellType::quad4:
    {
      Core::FE::shape_function_2d(N2, xi[0], xi[1], distype_cont);
      Core::FE::shape_function_2d_deriv1(N2_xi, xi[0], xi[1], distype_cont);
      break;
    }
      // 3D case
    case Core::FE::CellType::hex8:
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::tet10:
    {
      Core::FE::shape_function_3d(N2, xi[0], xi[1], xi[2], distype_cont);
      Core::FE::shape_function_3d_deriv1(N2_xi, xi[0], xi[1], xi[2], distype_cont);
      break;
    }
    default:
      FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
  }

  return;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
template <typename T>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::compute_artery_coords_and_derivs_ref(Core::LinAlg::Matrix<numdim_, 1, T>& r1,
    Core::LinAlg::Matrix<numdim_, 1, T>& r1_eta, const Core::LinAlg::Matrix<1, numnodesart_, T>& N1,
    const Core::LinAlg::Matrix<1, numnodesart_, T>& N1_eta)
{
  r1.clear();
  r1_eta.clear();

  for (unsigned int j = 0; j < numnodesart_; j++)
  {
    for (unsigned int idim = 0; idim < numdim_; idim++)
    {
      r1(idim) += N1(j) * ele1posref_(numdim_ * j + idim);
      r1_eta(idim) += N1_eta(j) * ele1posref_(numdim_ * j + idim);
    }
  }
  return;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
template <typename T>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::compute_2d_3d_coords_and_derivs_ref(Core::LinAlg::Matrix<numdim_, 1, T>& x2,
    Core::LinAlg::Matrix<numdim_, numdim_, T>& x2_xi,
    const Core::LinAlg::Matrix<1, numnodescont_, T>& N2,
    const Core::LinAlg::Matrix<numdim_, numnodescont_, T>& N2_xi)
{
  x2.clear();
  x2_xi.clear();

  for (unsigned int j = 0; j < numnodescont_; j++)
  {
    for (unsigned int idim = 0; idim < numdim_; idim++)
    {
      x2(idim) += N2(j) * ele2posref_(idim, j);
      for (unsigned int jdim = 0; jdim < numdim_; jdim++)
      {
        x2_xi(idim, jdim) += N2_xi(jdim, j) * ele2posref_(idim, j);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::fill_function_vector(std::vector<const Core::Utils::FunctionOfAnything*>& my_funct_vec,
    const std::vector<int>& funct_vec, const std::vector<int>& scale_vec)
{
  for (unsigned int i = 0; i < funct_vec.size(); i++)
  {
    if (funct_vec[i] >= 0 && abs(scale_vec[i]) > 0)
    {
      my_funct_vec.at(i) =
          &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
              funct_vec[i]);
      funct_coupl_active_ = true;
    }
    else
      my_funct_vec.at(i) = nullptr;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::initialize_function(const Core::Utils::FunctionOfAnything& funct)
{
  // safety check
  if (funct.number_components() != 1)
    FOUR_C_THROW("expected only one component for coupling function!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::initialize_function_names()
{
  pressurenames_.resize(numfluidphases_);
  saturationnames_.resize(numfluidphases_);
  volfracnames_.resize(numvolfrac_);
  volfracpressurenames_.resize(numvolfrac_);
  scalarnames_.resize(numscalcont_);
  artscalarnames_.resize(numscalart_);

  for (int k = 0; k < numscalcont_; k++)
  {
    // add scalar names
    {
      std::ostringstream temp;
      temp << k + 1;
      scalarnames_[k] = "phi" + temp.str();
    }
  }

  for (int k = 0; k < numscalart_; k++)
  {
    // add artery-scalar names
    {
      std::ostringstream temp;
      temp << k + 1;
      artscalarnames_[k] = "phi_art" + temp.str();
    }
  }

  for (int k = 0; k < numfluidphases_; k++)
  {
    // add pressure names
    {
      std::ostringstream temp;
      temp << k + 1;
      pressurenames_[k] = "p" + temp.str();
    }

    // add saturation names
    {
      std::ostringstream temp;
      temp << k + 1;
      saturationnames_[k] = "S" + temp.str();
    }
  }

  // add additional volume fractions
  for (int k = 0; k < numvolfrac_; k++)
  {
    // add volume fraction names
    {
      std::ostringstream temp;
      temp << k + 1;
      volfracnames_[k] = "VF" + temp.str();
    }
    // add volume fraction pressure names
    {
      std::ostringstream temp;
      temp << k + 1;
      volfracpressurenames_[k] = "VFP" + temp.str();
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_art, Core::FE::CellType distype_cont, int dim>
void PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<distype_art, distype_cont,
    dim>::initialize_assemble_into_cont_dof_vector()
{
  cont_dofs_to_assemble_functions_into_.resize(numdof_cont_);

  // just standard, each dof assembles into own equation
  for (int icont = 0; icont < numdof_cont_; icont++)
  {
    std::vector<int> thisdof = {icont};
    cont_dofs_to_assemble_functions_into_[icont] = thisdof;
  }

  switch (coupltype_)
  {
    case type_porofluid:
    {
      // special case for phases [0, ..., numfluidphases - 2]:
      // those have to assemble into the summed up fluid phase
      // see also porofluid_evaluator
      for (int curphase = 0; curphase < numfluidphases_ - 1; curphase++)
        cont_dofs_to_assemble_functions_into_[curphase].push_back(numfluidphases_ - 1);
      break;
    }
    case type_scatra:
    {
      // do nothing
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type");
      break;
  }
}


// explicit template instantiations
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::quad4, 1>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::hex8, 1>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet4, 1>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet10, 1>;

template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::quad4, 2>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::hex8, 2>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet4, 2>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet10, 2>;

template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::quad4, 3>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::hex8, 3>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet4, 3>;
template class PoroMultiPhaseScaTra::PoroMultiPhaseScatraArteryCouplingPair<
    Core::FE::CellType::line2, Core::FE::CellType::tet10, 3>;

FOUR_C_NAMESPACE_CLOSE
