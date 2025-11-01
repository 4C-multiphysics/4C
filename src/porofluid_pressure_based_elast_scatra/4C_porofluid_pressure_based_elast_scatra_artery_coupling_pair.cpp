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
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery, dis_type_homogenized,
    dim>::PorofluidElastScatraArteryCouplingPair()
    : PorofluidElastScatraArteryCouplingPairBase(),
      coupling_type_(CouplingType::undefined),
      coupling_method_(ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::none),
      condition_name_(""),
      is_init_(false),
      is_pre_evaluated_(false),
      is_active_(false),
      function_coupling_active_(false),
      variable_diameter_active_(false),
      evaluate_in_ref_config_(true),
      evaluate_on_lateral_surface_(true),
      artery_element_(nullptr),
      homogenized_element_(nullptr),
      artery_diameter_ref_(0.0),
      artery_diameter_at_gp_(0.0),
      num_dof_homogenized_(0),
      num_dof_artery_(0),
      dim_artery_(0),
      dim_homogenized_(0),
      num_coupled_dofs_(0),
      num_fluid_phases_(0),
      num_volfracs_(0),
      num_scalars_homogenized_(0),
      num_scalars_artery_(0),
      nds_porofluid_(-1),
      num_gp_(0),
      num_gp_per_patch_(0),
      artery_ele_length_ref_(0.0),
      artery_ele_length_(0.0),
      jacobian_determinant_(0.0),
      penalty_parameter_(0.0),
      artery_segment_start_(0.0),
      artery_segment_end_(0.0),
      current_segment_length_(0.0),
      constant_part_evaluated_(false),
      coupling_element_type_(""),
      artery_diameter_funct_(nullptr),
      porosity_name_("porosity"),
      artery_pressure_name_("p_art"),
      segment_id_(-1),
      num_patches_axial_(0),
      num_patches_radial_(0),
      timefacrhs_artery_density_(0.0),
      timefacrhs_homogenized_density_(0.0),
      timefacrhs_artery_(0.0),
      timefacrhs_homogenized_(0.0)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::init(const std::vector<Core::Elements::Element const*> elements,
    const Teuchos::ParameterList& coupling_params,
    const Teuchos::ParameterList& porofluid_coupling_params,
    const std::vector<int>& coupled_dofs_homogenized, const std::vector<int>& coupled_dofs_artery,
    const std::vector<std::vector<int>>& scale_vector,
    const std::vector<std::vector<int>>& function_id_vector, const std::string condition_name,
    const double penalty_parameter, const std::string coupling_type, const int eta_ntp)
{
  coupling_method_ =
      Teuchos::getIntegralValue<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
          coupling_params, "coupling_method");

  condition_name_ = condition_name;

  evaluate_in_ref_config_ =
      porofluid_coupling_params.get<bool>("evaluate_in_reference_configuration");

  evaluate_on_lateral_surface_ = porofluid_coupling_params.get<bool>("lateral_surface_coupling");

  coupling_element_type_ = coupling_type;

  num_patches_axial_ =
      porofluid_coupling_params.sublist("integration_patches").get<int>("number_of_patches_axial");
  num_patches_radial_ =
      porofluid_coupling_params.sublist("integration_patches").get<int>("number_of_patches_radial");

  artery_element_ = elements[0];
  homogenized_element_ = elements[1];

  // set coupling type
  if (artery_element_->element_type().name() == "ArteryType" &&
      homogenized_element_->element_type().name() == "PoroFluidMultiPhaseType")
  {
    coupling_type_ = CouplingType::porofluid;
    nds_porofluid_ = 0;
  }
  else if (artery_element_->element_type().name() == "TransportType" &&
           homogenized_element_->element_type().name() == "TransportType")
  {
    coupling_type_ = CouplingType::scatra;
    nds_porofluid_ = 2;
  }
  else
  {
    FOUR_C_THROW(
        "Your selected coupling is not possible, type of element 1: {}, type of element 2: {}",
        artery_element_->element_type().name(), homogenized_element_->element_type().name());
  }

  if (coupling_method_ == ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point)
  {
    // set eta
    gp_coords_artery_.resize(1);
    gp_coords_artery_[0] = eta_ntp;

    // check coupling type
    if (coupling_element_type_ != "ARTERY" && coupling_element_type_ != "AIRWAY")
    {
      if (coupling_type_ == CouplingType::porofluid)
      {
        FOUR_C_THROW(
            "Wrong coupling type in DESIGN 1D ARTERY TO POROFLUID NONCONF COUPLING CONDITIONS.\n "
            "NTP-coupling is only possible for coupling type: 'ARTERY' or 'AIRWAY'. "
            "Your coupling type is: {}",
            coupling_element_type_);
      }
      else
      {
        FOUR_C_THROW(
            "Wrong coupling type in DESIGN 1D ARTERY TO SCATRA NONCONF COUPLING CONDITIONS.\n"
            "NTP-coupling is only possible for coupling type: 'ARTERY' or 'AIRWAY'. "
            "Your coupling type is: {}",
            coupling_element_type_);
      }
    }
  }

  // get number of DOFs of artery or artery-scatra
  const Core::Nodes::Node* const* artery_nodes = artery_element_->nodes();

  num_dof_artery_ = artery_element_->num_dof_per_node(*artery_nodes[0]);
  for (int i = 1; i < artery_element_->num_node(); i++)
    if (num_dof_artery_ != artery_element_->num_dof_per_node(*artery_nodes[i]))
      FOUR_C_THROW("It is not possible to have different number of Dofs in artery discretization");
  dim_artery_ = num_dof_artery_ * artery_element_->num_node();

  // get number of DOFs of homogenized ele (scatra or porofluid)
  const Core::Nodes::Node* const* homogenized_nodes = homogenized_element_->nodes();

  num_dof_homogenized_ = homogenized_element_->num_dof_per_node(*homogenized_nodes[0]);
  for (int i = 1; i < homogenized_element_->num_node(); i++)
  {
    if (num_dof_homogenized_ != homogenized_element_->num_dof_per_node(*homogenized_nodes[i]))
      FOUR_C_THROW(
          "It is not possible to have different number of Dofs in homogenized discretization");
  }
  dim_homogenized_ = num_dof_homogenized_ * homogenized_element_->num_node();

  // safety check
  if (std::cmp_not_equal(num_dof_artery_, (scale_vector[0].size())))
    FOUR_C_THROW("Wrong size of scale-vector (artery)");
  if (std::cmp_not_equal(num_dof_artery_, (function_id_vector[0].size())))
    FOUR_C_THROW("Wrong size of function-vector (artery)");
  if (std::cmp_not_equal(num_dof_homogenized_, (scale_vector[1].size())))
    FOUR_C_THROW("Wrong size of scale-vector (homogenized discretization)");
  if (std::cmp_not_equal(num_dof_homogenized_, (function_id_vector[1].size())))
    FOUR_C_THROW("Wrong size of function-vector (homogenized discretization)");

  // fill scale vector
  scale_vector_ = scale_vector;

  // fill function vector
  function_vector_.resize(2);
  function_vector_[0].resize(num_dof_artery_);
  function_vector_[1].resize(num_dof_homogenized_);
  fill_function_vector(function_vector_[0], function_id_vector[0], scale_vector_[0]);
  fill_function_vector(function_vector_[1], function_id_vector[1], scale_vector_[1]);

  // get the actually coupled dofs
  coupled_dofs_homogenized_ = coupled_dofs_homogenized;
  // Note: this will be overwritten in case of arteryscatra-scatra coupling
  volfrac_pressure_id_ = coupled_dofs_homogenized;
  coupled_dofs_artery_ = coupled_dofs_artery;
  num_coupled_dofs_ = static_cast<int>(coupled_dofs_homogenized.size());

  // safety check
  for (int i_homo = 0; i_homo < num_coupled_dofs_; i_homo++)
  {
    if (coupled_dofs_homogenized_[i_homo] >= num_dof_homogenized_)
    {
      FOUR_C_THROW(
          "You try to couple DOF {}, which is larger than the number of dofs of the homogenized "
          "discretization.",
          coupled_dofs_homogenized_[i_homo] + 1);
    }
    if (coupled_dofs_homogenized_[i_homo] < 0)
    {
      FOUR_C_THROW(
          "Your coupling DOF of the homogenized discretization must be >= 0, your DOF = {}.",
          coupled_dofs_homogenized_[i_homo] + 1);
    }
  }
  for (int i_art = 0; i_art < num_coupled_dofs_; i_art++)
  {
    if (coupled_dofs_artery_[i_art] >= num_dof_artery_)
    {
      FOUR_C_THROW(
          "You try to couple DOF {}, which is larger than the number of dofs of the artery "
          "discretization.",
          coupled_dofs_artery_[i_art] + 1);
    }
    if (coupled_dofs_artery_[i_art] < 0)
    {
      FOUR_C_THROW("Your coupling DOF of the reduced discretization must be >= 0, your DOF = {}.",
          coupled_dofs_artery_[i_art] + 1);
    }
  }

  // Set reference nodal positions for artery element
  for (unsigned int n = 0; n < num_nodes_artery_; ++n)
  {
    const Core::Nodes::Node* node = artery_element_->nodes()[n];
    for (unsigned int d = 0; d < num_dim_; ++d)
      nodal_coords_artery_ele_ref_(num_dim_ * n + d) = node->x()[d];
  }

  // get length of 1D element
  Core::LinAlg::Matrix<num_dim_, 1> artery_coords_start;
  for (unsigned int d = 0; d < num_dim_; ++d)
    artery_coords_start(d) = nodal_coords_artery_ele_ref_(d);
  Core::LinAlg::Matrix<num_dim_, 1> artery_coords_end;
  for (unsigned int d = 0; d < num_dim_; ++d)
    artery_coords_end(d) = nodal_coords_artery_ele_ref_(num_dim_ + d);

  Core::LinAlg::Matrix<num_dim_, 1> distance;
  distance.update(-1.0, artery_coords_start, 1.0, artery_coords_end, 0.0);
  artery_ele_length_ref_ = distance.norm2();

  // get initial orientation of the artery element
  initial_artery_orientation_.update(1.0 / artery_ele_length_ref_, distance, 0.0);

  // Set reference nodal positions for homogenized discretization element
  for (unsigned int inode = 0; inode < num_nodes_homogenized_; ++inode)
  {
    const Core::Nodes::Node* node = homogenized_element_->nodes()[inode];
    for (unsigned int i_dim = 0; i_dim < num_dim_; ++i_dim)
      nodal_coords_homogenized_ele_ref_(i_dim, inode) = node->x()[i_dim];
  }

  // set current nodal positions to reference nodal positions for homogenized discretization
  // element
  nodal_coords_homogenized_ele_.update(1.0, nodal_coords_homogenized_ele_ref_, 0.0);

  penalty_parameter_ = penalty_parameter;

  is_init_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::setup_fluid_managers_and_materials(const std::string dis_name,
    const double& timefacrhs_artery, const double& timefacrhs_homogenized)
{
  // dummy parameter list
  Discret::Elements::PoroFluidMultiPhaseEleParameter* params =
      Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(dis_name);

  double artery_density = 0.0;

  std::shared_ptr<Mat::FluidPoroMultiPhase> porofluid_material = nullptr;
  std::shared_ptr<Mat::MatList> homogenized_scatra_material = nullptr;
  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      porofluid_material = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(
          homogenized_element_->material(nds_porofluid_));
      if (porofluid_material == nullptr)
        FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed for artery-porofluid coupling!");
      for (int i_dof = 0; i_dof < num_coupled_dofs_; i_dof++)
      {
        const int material_id = porofluid_material->mat_id(coupled_dofs_homogenized_[i_dof]);

        // safety check
        if (const std::shared_ptr<Core::Mat::Material> single_phase_material =
                porofluid_material->material_by_id(material_id);
            single_phase_material->material_type() !=
                Core::Materials::m_fluidporo_volfracpressure &&
            single_phase_material->material_type() != Core::Materials::m_fluidporo_singlephase &&
            single_phase_material->material_type() !=
                Core::Materials::m_fluidporo_volfrac_pressure_blood_lung)

        {
          FOUR_C_THROW(
              "You can only couple volume fraction pressures or fluid phases in multi-phase "
              "pore space. Your material is of type {}.",
              single_phase_material->material_type());
        }
      }
      // coupling with scatra: the scatra-material is the third material in the 2D/3D element
      if (homogenized_element_->num_material() == 3)
      {
        if (const std::shared_ptr<Mat::MatList> scatra_material =
                std::static_pointer_cast<Mat::MatList>(homogenized_element_->material(2));
            scatra_material->material_type() == Core::Materials::m_matlist or
            scatra_material->material_type() == Core::Materials::m_matlist_reactions)
        {
          num_scalars_homogenized_ = scatra_material->num_mat();
        }
        else
        {
          FOUR_C_THROW(
              "Only MAT_matlist or MAT_matlist_reactions are valid as scatra material in the "
              "homogenized domain with porofluid-artery coupling. Your material is of type {}.",
              scatra_material->material_type());
        }
      }
      // coupling with artery-scatra: the artery-scatra-material is the second material in the
      // artery element
      if (artery_element_->num_material() == 2)
      {
        if (const std::shared_ptr<Mat::MatList> artery_scatra_material =
                std::static_pointer_cast<Mat::MatList>(artery_element_->material(1));
            artery_element_->material(1)->material_type() == Core::Materials::m_matlist)
        {
          num_scalars_artery_ = artery_scatra_material->num_mat();
        }
        else if (artery_element_->material(1)->material_type() == Core::Materials::m_scatra)
        {
          num_scalars_artery_ = 1;
        }
        else
        {
          FOUR_C_THROW(
              "Only MAT_matlist and MAT_scatra are valid as scatra material in the artery domain "
              "with porofluid-artery coupling. Your material is of type {}.",
              artery_element_->material(1)->material_type());
        }
      }

      artery_material_ = std::static_pointer_cast<Mat::Cnst1dArt>(artery_element_->material(0));
      if (artery_material_ == nullptr)
        FOUR_C_THROW("cast to artery material failed for porofluid-artery coupling!");
      artery_diameter_at_gp_ = artery_material_->diam();
      artery_diameter_ref_ = artery_material_->diam_initial();
      artery_density = artery_material_->density();

      break;
    }
    case CouplingType::scatra:
    {
      // check if we actually have three materials
      if (homogenized_element_->num_material() < 3)
        FOUR_C_THROW("No third material available in Mat::FluidPoroMultiPhase.");

      porofluid_material = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(
          homogenized_element_->material(nds_porofluid_));
      if (porofluid_material == nullptr)
        FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed for arteryscatra-scatra coupling!");

      homogenized_scatra_material =
          std::static_pointer_cast<Mat::MatList>(homogenized_element_->material(0));
      if (homogenized_scatra_material == nullptr)
        FOUR_C_THROW("cast to ScatraMat failed for arteryscatra-scatra coupling!");

      for (int i_dof = 0; i_dof < num_coupled_dofs_; i_dof++)
      {
        const int material_id =
            homogenized_scatra_material->mat_id(coupled_dofs_homogenized_[i_dof]);
        std::shared_ptr<Core::Mat::Material> single_scatra_material =
            homogenized_scatra_material->material_by_id(material_id);

        // safety check
        if (single_scatra_material->material_type() !=
                Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based &&
            single_scatra_material->material_type() !=
                Core::Materials::m_scatra_in_fluid_porofluid_pressure_based)
        {
          FOUR_C_THROW(
              "You can only couple Mat::ScatraMatMultiPoroVolFrac or Mat::ScatraMatMultiPoroFluid. "
              "Your material on the homogenized domain is of type {}.",
              single_scatra_material->material_type());
        }

        if (single_scatra_material->material_type() ==
            Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& scatra_volfrac_material =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(
                  single_scatra_material);

          if (porofluid_material->num_mat() ==
              porofluid_material->num_fluid_phases() + porofluid_material->num_vol_frac())
          {
            volfrac_pressure_id_[i_dof] = scatra_volfrac_material->phase_id();
          }
          else if (porofluid_material->num_mat() ==
                   porofluid_material->num_fluid_phases() + 2 * porofluid_material->num_vol_frac())
          {
            volfrac_pressure_id_[i_dof] =
                scatra_volfrac_material->phase_id() + porofluid_material->num_vol_frac();
          }
          else
            FOUR_C_THROW("Internal Error!");
        }
      }

      // get the artery scatra-material
      const std::shared_ptr<Mat::MatList> artery_scatra_material =
          std::static_pointer_cast<Mat::MatList>(artery_element_->material(0));
      if (artery_scatra_material->material_type() == Core::Materials::m_matlist)
      {
        num_scalars_artery_ = artery_scatra_material->num_mat();
      }
      else if (artery_scatra_material->material_type() == Core::Materials::m_scatra)
      {
        num_scalars_artery_ = 1;
      }
      else
      {
        FOUR_C_THROW(
            "Only MAT_matlist and MAT_scatra are valid for artery-scatra material. Your material "
            "on the artery domain is of type {}.",
            artery_scatra_material->material_type());
      }

      num_scalars_homogenized_ = num_dof_homogenized_;
      artery_material_ = std::static_pointer_cast<Mat::Cnst1dArt>(artery_element_->material(1));
      if (artery_material_ == nullptr)
        FOUR_C_THROW("Cast to artery material failed for arteryscatra-scatra coupling!");
      artery_diameter_at_gp_ = artery_material_->diam();
      artery_diameter_ref_ = artery_material_->diam_initial();
      artery_density = artery_material_->density();
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }

  // take care of diameter function
  if (const int diameter_function_id = artery_material_->diameter_function();
      diameter_function_id > -1)
  {
    // no real use-case without function coupling
    if (not function_coupling_active_)
    {
      FOUR_C_THROW(
          "Diameter function has been defined but no exchange function has been set. This is "
          "currently not possible. If you want a variable diameter without any exchange terms, you "
          "must define a zero exchange term.");
    }
    variable_diameter_active_ = true;
    artery_diameter_funct_ =
        &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
            diameter_function_id);
    if (coupling_type_ == CouplingType::porofluid)
    {
      // homogenized derivatives + 1 artery pressure derivative
      diameter_derivs_ = std::vector(num_dof_homogenized_ + 1, 0.0);
      diameter_ele_matrix_artery_artery_ = Core::LinAlg::SerialDenseMatrix();
      diameter_ele_matrix_artery_homogenized_ = Core::LinAlg::SerialDenseMatrix();
    }
  }

  // safety checks for lateral surface coupling
  if (evaluate_on_lateral_surface_)
  {
    if (!evaluate_in_ref_config_)
    {
      FOUR_C_THROW(
          "Evaluation in current configuration is not yet possible in combination with lateral "
          "surface coupling.");
    }
    if (variable_diameter_active_)
    {
      FOUR_C_THROW(
          "Setting a variable diameter is not yet possible in combination with lateral "
          "surface coupling.");
    }
    if (dis_type_homogenized != Core::FE::CellType::hex8 &&
        dis_type_homogenized != Core::FE::CellType::tet4)
      FOUR_C_THROW("Only TET4 and HEX8 elements possible for lateral surface coupling.");
  }

  num_fluid_phases_ = porofluid_material->num_fluid_phases();
  num_volfracs_ = porofluid_material->num_vol_frac();

  // create phase-manager
  phase_manager_ = Discret::Elements::PoroFluidManager::PhaseManagerInterface::create_phase_manager(
      *params, num_dim_, porofluid_material->material_type(), get_access_from_artcoupling,
      porofluid_material->num_mat(), porofluid_material->num_fluid_phases(),
      porofluid_material->num_vol_frac());

  // setup phase-manager
  phase_manager_->setup(homogenized_element_, nds_porofluid_);

  // create variable-manager
  variable_manager_ = Discret::Elements::PoroFluidManager::VariableManagerInterface<num_dim_,
      num_nodes_homogenized_>::create_variable_manager(*params, get_access_from_artcoupling,
      porofluid_material, porofluid_material->num_mat(), porofluid_material->num_fluid_phases(),
      porofluid_material->num_vol_frac());

  // initialize the names used in functions
  initialize_function_names();

  // fill vector where to assemble rhs-(function) coupling into summed up phase requires special
  // treatment
  initialize_assemble_into_homogenized_dof_vector();

  // initialize the functions
  for (int i = 0; i < 2; i++)
    for (const auto& i_dof : function_vector_[i])
      if (i_dof != nullptr) initialize_function(*i_dof);
  if (variable_diameter_active_) initialize_function(*artery_diameter_funct_);

  // set time fac for right-hand side evaluation of coupling
  set_time_fac_rhs(
      artery_density, homogenized_scatra_material.get(), timefacrhs_artery, timefacrhs_homogenized);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::pre_evaluate(std::shared_ptr<Core::LinAlg::MultiVector<double>>
        gp_vector)
{
  if (!is_init_) FOUR_C_THROW("MeshTying Pair has not yet been initialized.");

  if (coupling_method_ == ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point)
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

  is_pre_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::pre_evaluate_lateral_surface_coupling(Core::LinAlg::MultiVector<double>&
        gauss_point_vector)
{
  const int my_mpi_rank =
      Core::Communication::my_mpi_rank(Global::Problem::instance()->get_dis("artery")->get_comm());
  const int my_lid = artery_element_->lid();
  if (homogenized_element_->owner() != my_mpi_rank) return;

  // unit radial basis vectors
  Core::LinAlg::Matrix<3, 1> unit_radial_basis_1;
  Core::LinAlg::Matrix<3, 1> unit_radial_basis_2;
  // unit tangential basis
  Core::LinAlg::Matrix<3, 1> unit_tangent_basis;
  if (num_dim_ != 3) FOUR_C_THROW("Surface-based formulation makes only sense in 3D.");
  for (int i_dim = 0; i_dim < 3; i_dim++)
    unit_tangent_basis(i_dim) = initial_artery_orientation_(i_dim);

  Core::Geo::build_orthonormal_basis_from_unit_vector(
      unit_tangent_basis, unit_radial_basis_1, unit_radial_basis_2);

  // get radius
  const int artery_ele_material = coupling_type_ == CouplingType::scatra ? 1 : 0;
  const std::shared_ptr<Mat::Cnst1dArt> artery_material =
      std::static_pointer_cast<Mat::Cnst1dArt>(artery_element_->material(artery_ele_material));
  if (artery_material == nullptr)
    FOUR_C_THROW("Cast to artery material failed for porofluid-artery coupling!");
  const double radius = artery_material->diam() / 2.0;

  // size of one integration patch: 2 * pi * R/num_patches_radial_ * L/num_patches_axial_
  const double patch_size = 1.0 / num_patches_axial_ * artery_ele_length_ref_ * 1.0 /
                            num_patches_radial_ * 2.0 * std::numbers::pi * radius;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv(
      Core::LinAlg::Initialization::zero);
  // Coordinates and derivatives of the artery element
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv(Core::LinAlg::Initialization::zero);

  // element parameter space coordinates in 3D element
  std::vector<double> local_coordinate(3);
  // number of GPs
  num_gp_ = 0;

  // we always use 25 integration points per integration patch
  const auto gauss_points_per_patch =
      Core::FE::IntegrationPoints2D(Core::FE::GaussRule2D::quad_25point);
  num_gp_per_patch_ = gauss_points_per_patch.nquad;
  num_gp_ = num_gp_per_patch_ * num_patches_axial_ * num_patches_radial_;
  // define Gauss points and n_gp-sized quantities
  gp_coords_artery_.resize(num_gp_);
  previous_gp_coords_deformed_.resize(num_gp_);
  gp_weights_.resize(num_gp_);
  gp_coords_homogenized_.resize(num_gp_);
  for (int i_gp = 0; i_gp < num_gp_; i_gp++) gp_coords_homogenized_[i_gp].resize(num_dim_);

  // loop over all axial patches
  for (int i_ax = 0; i_ax < num_patches_axial_; i_ax++)
  {
    // loop over all radial patches
    for (int i_rad = 0; i_rad < num_patches_radial_; i_rad++)
    {
      // loop over all GPs of this patch
      for (int i_gp = 0; i_gp < num_gp_per_patch_; i_gp++)
      {
        // axial Gauss point eta lies in [-1; 1]
        const double eta = -1.0 - 1.0 / num_patches_axial_ +
                           (i_ax + 1.0) * 2.0 / num_patches_axial_ +
                           gauss_points_per_patch.qxg[i_gp][0] * 1.0 / num_patches_axial_;

        // Update coordinates and derivatives for 1D and 2D/3D element
        get_artery_shape_functions<double>(
            shape_functions_artery, shape_functions_artery_deriv, eta);
        compute_artery_coords_and_derivs_ref<double>(coords_artery, coords_artery_deriv,
            shape_functions_artery, shape_functions_artery_deriv);

        // radial Gauss point theta lies in [-pi; pi]
        const double theta =
            (-1.0 - 1.0 / num_patches_radial_ + (i_rad + 1.0) * 2.0 / num_patches_radial_ +
                gauss_points_per_patch.qxg[i_gp][1] * 1.0 / num_patches_radial_) *
            std::numbers::pi;

        // get point on lateral blood vessel surface
        for (int i_dim = 0; i_dim < 3; i_dim++)
        {
          coords_artery(i_dim) = coords_artery(i_dim) +
                                 unit_radial_basis_1(i_dim) * radius * cos(theta) +
                                 unit_radial_basis_2(i_dim) * radius * sin(theta);
        }

        // project into 3D domain
        bool projection_valid = false;
        projection<double>(coords_artery, local_coordinate, projection_valid);

        const int gp_id =
            i_ax * num_patches_radial_ * num_gp_per_patch_ + i_rad * num_gp_per_patch_ + i_gp;
        gp_coords_artery_[gp_id] = eta;
        gp_coords_homogenized_[gp_id] = local_coordinate;
        // the projection is valid and GP is so far unclaimed by another pair
        if (projection_valid && (gauss_point_vector)(gp_id).local_values_as_span()[my_lid] < 0.5)
        {
          is_active_ = true;
          // include jacobian
          gp_weights_[gp_id] = gauss_points_per_patch.qwgt[i_gp] * patch_size / 4.0;
          gauss_point_vector.sum_into_local_value(my_lid, gp_id, 1.0);
        }
        else
        {
          gp_weights_[gp_id] = 0.0;
        }
      }
    }
  }

  // free memory
  if (not is_active_)
  {
    std::vector<double>().swap(gp_coords_artery_);
    std::vector<double>().swap(gp_weights_);
    std::vector<std::vector<double>>().swap(gp_coords_homogenized_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::pre_evaluate_centerline_coupling()
{
  // Try to create the integration segment [eta_start, eta_end]
  create_integration_segment();

  // no viable segment found
  if (!is_active_)
  {
    return;
  }

  // Choice of optimal Gauss-point rule: basically the N^(2)*N^(2) term is crucial.
  // For (bi-, tri-)linear elements (only considered so far):
  // In 2D, the highest possible polynomial order for this term is 4 since N^(2) can be quadratic
  // for arbitrary integration in the element. --> We need 3 Gauss points for exact integration.
  // In 3D, the highest possible polynomial order for this term is 6 since N^(2) can be cubic for
  // arbitrary integration in the element. --> We need 4 Gauss points for exact integration.

  auto gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
  if (num_dim_ == 3)
    gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

  num_gp_ = gauss_points.nquad;
  // define Gauss points and n_gp-sized quantities
  gp_coords_artery_.resize(num_gp_);
  previous_gp_coords_deformed_.resize(num_gp_);
  gp_weights_.resize(num_gp_);
  inverse_jacobian_matrix_.resize(num_gp_);
  gp_coords_homogenized_.resize(num_gp_);
  for (int i_gp = 0; i_gp < num_gp_; i_gp++) gp_coords_homogenized_[i_gp].resize(num_dim_);

  // get jacobian determinant
  const double determinant = (artery_segment_end_ - artery_segment_start_) / 2.0;
  jacobian_determinant_ = determinant * artery_ele_length_ref_ / 2.0;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;
  // Coordinates and derivatives of the artery element in reference configuration
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_ref;
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv_ref;

  // project the Gauss points --> those have to be able to be projected
  for (int i_gp = 0; i_gp < num_gp_; i_gp++)
  {
    // compute the coordinate transformation, Gauss point weights and project Gauss points
    gp_coords_artery_[i_gp] = (artery_segment_start_ + artery_segment_end_) / 2.0 +
                              gauss_points.qxg[i_gp][0] * determinant;
    previous_gp_coords_deformed_[i_gp] = gp_coords_artery_[i_gp];
    gp_weights_[i_gp] = gauss_points.qwgt[i_gp];

    // Update coordinates and derivatives for 1D and 2D/3D element
    get_artery_shape_functions<double>(
        shape_functions_artery, shape_functions_artery_deriv, gp_coords_artery_[i_gp]);
    compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
        shape_functions_artery, shape_functions_artery_deriv);

    bool projection_valid = false;
    projection<double>(coords_artery_ref, gp_coords_homogenized_[i_gp], projection_valid);
    if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");

    // compute (dX/dxi)^-1
    get_homogenized_shape_functions<double>(shape_functions_homogenized,
        shape_functions_homogenized_deriv, gp_coords_homogenized_[i_gp]);
    inverse_jacobian_matrix_[i_gp].multiply_nt(
        shape_functions_homogenized_deriv, nodal_coords_homogenized_ele_);
    inverse_jacobian_matrix_[i_gp].invert();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::pre_evaluate_node_to_point_coupling()
{
  gp_coords_homogenized_.resize(1);
  gp_coords_homogenized_[0].resize(num_dim_);

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;
  // Coordinates and derivatives of the artery element in reference configuration
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_ref;
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv_ref;

  // Update coordinates and derivatives for 1D and 2D/3D element
  get_artery_shape_functions<double>(
      shape_functions_artery, shape_functions_artery_deriv, gp_coords_artery_[0]);
  compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
      shape_functions_artery, shape_functions_artery_deriv);

  bool projection_valid = false;
  projection<double>(coords_artery_ref, gp_coords_homogenized_[0], projection_valid);

  // coupling pairs is only active if projection is valid
  is_active_ = projection_valid;

  // compute (dX/dxi)^-1
  get_homogenized_shape_functions<double>(
      shape_functions_homogenized, shape_functions_homogenized_deriv, gp_coords_homogenized_[0]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::delete_unnecessary_gps(const std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector)
{
  const int my_lid = artery_element_->lid();
  num_gp_ = 0;
  for (int igp = 0; igp < num_gp_per_patch_ * num_patches_axial_ * num_patches_radial_; igp++)
    if (gp_weights_[igp] > 1e-12) num_gp_++;

  std::vector<double> gp_weights(num_gp_);
  std::vector<double> gp_coords_artery(num_gp_);
  std::vector<std::vector<double>> gp_coords_homogenized(num_gp_);

  int current_gp = 0;
  for (int igp = 0; igp < num_gp_per_patch_ * num_patches_axial_ * num_patches_radial_; igp++)
  {
    if (gp_weights_[igp] > 1e-12)
    {
      const double scale = 1.0 / (*gp_vector)(igp).local_values_as_span()[my_lid];
      gp_coords_artery[current_gp] = gp_coords_artery_[igp];
      gp_coords_homogenized[current_gp] = gp_coords_homogenized_[igp];
      gp_weights[current_gp] = gp_weights_[igp] * scale;
      current_gp++;
    }
  }
  if (num_gp_ == 0)
    is_active_ = false;
  else
    is_active_ = true;

  // overwrite
  gp_weights_ = gp_weights;
  gp_coords_artery_ = gp_coords_artery;
  gp_coords_homogenized_ = gp_coords_homogenized;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::reset_state(const std::shared_ptr<Core::FE::Discretization>
                                                homogenized_dis,
    const std::shared_ptr<Core::FE::Discretization> artery_dis)
{
  if (!is_pre_evaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated.");

  // get location array of the homogenized element
  Core::Elements::LocationArray location_array_homogenized(homogenized_dis->num_dof_sets());
  homogenized_element_->location_vector(*homogenized_dis, location_array_homogenized);

  // get location array for the artery element
  Core::Elements::LocationArray location_array_artery(artery_dis->num_dof_sets());
  artery_element_->location_vector(*artery_dis, location_array_artery);

  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      // extract element and node values of fluid
      variable_manager_->extract_element_and_node_values(*homogenized_element_, *homogenized_dis,
          location_array_homogenized, nodal_coords_homogenized_ele_, 0);

      // get phinp from homogenized and artery elements
      phi_np_homogenized_ele_ = Core::FE::extract_values(
          *homogenized_dis->get_state("phinp_fluid"), location_array_homogenized[0].lm_);
      phi_np_artery_ele_ = Core::FE::extract_values(
          *artery_dis->get_state("one_d_artery_pressure"), location_array_artery[0].lm_);

      // extract velocity of solid phase
      extract_velocity_solid_phase(*homogenized_dis);

      // extract values of artery-scatra discretization
      if (num_scalars_artery_ > 0)
      {
        const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_scalar_np =
            artery_dis->get_state(nds_artery_scatra_, "one_d_artery_phinp");
        if (artery_scalar_np != nullptr)
        {
          Core::Elements::LocationArray location_array(artery_dis->num_dof_sets());
          artery_element_->location_vector(*artery_dis, location_array);
          // rebuild scalar vector
          nodal_artery_scalar_np_.clear();
          nodal_artery_scalar_np_.resize(num_scalars_artery_,
              Core::LinAlg::Matrix<num_nodes_artery_, 1>(Core::LinAlg::Initialization::zero));
          // extract local values of artery-scatra field from global state vector
          Core::FE::extract_my_values<Core::LinAlg::Matrix<num_nodes_artery_, 1>>(
              *artery_scalar_np, nodal_artery_scalar_np_, location_array[nds_artery_scatra_].lm_);
        }
        else
        {
          FOUR_C_THROW("Cannot get artery-scatra from artery discretization.");
        }
      }
      // extract values of homogenized scatra discretization
      if (num_scalars_homogenized_ > 0)
      {
        // get state vector from discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_scalar_np =
            homogenized_dis->get_state(3, "scalars");
        if (homogenized_scalar_np != nullptr)
        {
          Core::Elements::LocationArray location_array(homogenized_dis->num_dof_sets());
          homogenized_element_->location_vector(*homogenized_dis, location_array);
          // rebuild scalar vector
          nodal_homogenized_scalar_np_.clear();
          nodal_homogenized_scalar_np_.resize(num_scalars_homogenized_,
              Core::LinAlg::Matrix<num_nodes_homogenized_, 1>(Core::LinAlg::Initialization::zero));
          // extract local values of homogenized scatra field from global state vector
          Core::FE::extract_my_values<Core::LinAlg::Matrix<num_nodes_homogenized_, 1>>(
              *homogenized_scalar_np, nodal_homogenized_scalar_np_, location_array[3].lm_);
        }
        else
        {
          FOUR_C_THROW("Cannot get state vector 'scalars'.");
        }
      }
      break;
    }
    case CouplingType::scatra:
    {
      // extract element and node values of fluid
      variable_manager_->extract_element_and_node_values(*homogenized_element_, *homogenized_dis,
          location_array_homogenized, nodal_coords_homogenized_ele_, 2);

      // get phi_np from homogenized and artery elements
      phi_np_homogenized_ele_ = Core::FE::extract_values(
          *homogenized_dis->get_state("phinp"), location_array_homogenized[0].lm_);
      phi_np_artery_ele_ = Core::FE::extract_values(
          *artery_dis->get_state("one_d_artery_phinp"), location_array_artery[0].lm_);

      // extract artery pressure
      const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_pressure_np =
          artery_dis->get_state(nds_scatra_artery_, "one_d_artery_pressure");
      if (artery_pressure_np != nullptr)
      {
        Core::Elements::LocationArray la(artery_dis->num_dof_sets());
        artery_element_->location_vector(*artery_dis, la);
        Core::FE::extract_my_values<Core::LinAlg::Matrix<num_nodes_artery_, 1>>(
            *artery_pressure_np, nodal_artery_pressure_np_, la[nds_scatra_artery_].lm_);
      }
      else
      {
        FOUR_C_THROW("Cannot get artery pressure from artery-scatra discretization.");
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
double PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate(Core::LinAlg::SerialDenseVector* ele_rhs_artery,
    Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized,
    Core::LinAlg::SerialDenseMatrix* D_ele, Core::LinAlg::SerialDenseMatrix* M_ele,
    Core::LinAlg::SerialDenseVector* Kappa_ele, const std::vector<double>& segment_lengths)
{
  if (!is_pre_evaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated.");

  // resize and initialize variables to zero
  if (ele_rhs_artery != nullptr) ele_rhs_artery->size(dim_artery_);
  if (ele_rhs_homogenized != nullptr) ele_rhs_homogenized->size(dim_homogenized_);

  if (ele_matrix_artery_artery != nullptr)
    ele_matrix_artery_artery->shape(dim_artery_, dim_artery_);
  if (ele_matrix_artery_homogenized != nullptr)
    ele_matrix_artery_homogenized->shape(dim_artery_, dim_homogenized_);
  if (ele_matrix_homogenized_artery != nullptr)
    ele_matrix_homogenized_artery->shape(dim_homogenized_, dim_artery_);
  if (ele_matrix_homogenized_homogenized != nullptr)
    ele_matrix_homogenized_homogenized->shape(dim_homogenized_, dim_homogenized_);

  if (artery_material_->is_collapsed()) return 0.0;

  std::vector<double> current_gp_coord_artery;
  std::vector<std::vector<double>> current_gp_coords_homogenized;

  double artery_segment_start = 0.0;
  double artery_segment_end = 0.0;
  double integrated_diameter = 0.0;

  switch (coupling_method_)
  {
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gauss_point_to_segment:
    {
      // initialize Gauss point coordinates
      current_gp_coord_artery.assign(num_gp_, double{});
      current_gp_coords_homogenized.assign(num_gp_, std::vector(num_dim_, double{}));
      // recompute Gauss point coordinated in deformed configuration --> see notes in this function
      recompute_gp_coords_in_deformed_configuration(segment_lengths, current_gp_coord_artery,
          current_gp_coords_homogenized, artery_segment_start, artery_segment_end);

      evaluate_gpts(current_gp_coord_artery, current_gp_coords_homogenized, segment_lengths,
          ele_rhs_artery, ele_rhs_homogenized, ele_matrix_artery_artery,
          ele_matrix_artery_homogenized, ele_matrix_homogenized_artery,
          ele_matrix_homogenized_homogenized);

      // case where diameter is constant
      integrated_diameter = artery_diameter_ref_ * segment_lengths[segment_id_];
      break;
    }
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty:
    {
      // initialize Gauss point coordinates
      current_gp_coord_artery.assign(num_gp_, double{});
      current_gp_coords_homogenized.assign(num_gp_, std::vector(num_dim_, double{}));
      // recompute Gauss point coordinated in deformed configuration --> see notes in this function
      recompute_gp_coords_in_deformed_configuration(segment_lengths, current_gp_coord_artery,
          current_gp_coords_homogenized, artery_segment_start, artery_segment_end);

      evaluate_mortar_coupling(current_gp_coord_artery, current_gp_coords_homogenized,
          segment_lengths, D_ele, M_ele, Kappa_ele);

      // case where diameter is constant
      integrated_diameter = artery_diameter_ref_ * segment_lengths[segment_id_];
      break;
    }
    case ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point:
    {
      // define Gauss point coordinates
      current_gp_coord_artery = gp_coords_artery_;
      current_gp_coords_homogenized = gp_coords_homogenized_;

      evaluate_ntp(gp_coords_artery_, gp_coords_homogenized_, ele_rhs_artery, ele_rhs_homogenized,
          ele_matrix_artery_artery, ele_matrix_artery_homogenized, ele_matrix_homogenized_artery,
          ele_matrix_homogenized_homogenized);
      integrated_diameter = artery_diameter_ref_ * segment_lengths[0];
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type for artery to poro coupling");
      break;
  }

  // evaluate the function coupling (with possibly variable diameter)
  if (function_coupling_active_)
  {
    evaluate_function_coupling(current_gp_coord_artery, current_gp_coords_homogenized,
        segment_lengths, ele_rhs_artery, ele_rhs_homogenized, ele_matrix_artery_artery,
        ele_matrix_artery_homogenized, ele_matrix_homogenized_artery,
        ele_matrix_homogenized_homogenized, integrated_diameter);
  }

  // evaluate derivative of 1D shape function times solid velocity
  evaluate_nds_solid_velocity(current_gp_coord_artery, current_gp_coords_homogenized,
      *ele_rhs_artery, artery_segment_start, artery_segment_end);

  return integrated_diameter;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::evaluate_additional_linearization_of_integrated_diameter(Core::LinAlg::SerialDenseMatrix*
                                                                       ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized)
{
  if (!is_pre_evaluated_) FOUR_C_THROW("MeshTying Pair has not yet been pre-evaluated.");

  if (ele_matrix_artery_artery != nullptr)
    ele_matrix_artery_artery->shape(dim_artery_, dim_artery_);
  if (ele_matrix_artery_homogenized != nullptr)
    ele_matrix_artery_homogenized->shape(dim_artery_, dim_homogenized_);

  // do not evaluate if the element is collapsed
  if (artery_material_->is_collapsed()) return;

  // this is the integrated diameter over the entire element (all segments)
  const double artery_diameter = artery_material_->diam();
  const double pre_factor =
      std::numbers::pi * std::pow(artery_diameter, 3) / 32.0 / artery_material_->viscosity();
  // TODO: for viscosity law blood, viscosity depends on diameter, linearization is still missing

  Core::LinAlg::update(
      pre_factor, diameter_ele_matrix_artery_artery_, 0.0, *ele_matrix_artery_artery);
  Core::LinAlg::update(
      pre_factor, diameter_ele_matrix_artery_homogenized_, 0.0, *ele_matrix_artery_homogenized);
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
int PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery, dis_type_homogenized,
    dim>::artery_ele_gid() const
{
  return artery_element_->id();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
int PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery, dis_type_homogenized,
    dim>::homogenized_ele_gid() const
{
  return homogenized_element_->id();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
int PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery, dis_type_homogenized,
    dim>::get_segment_id() const
{
  return segment_id_;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
double PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::calculate_volume_homogenized_element() const
{
  // use one-point Gauss rule
  Core::FE::IntPointsAndWeights<num_dim_> integration_points_stab(
      Discret::Elements::DisTypeToStabGaussRule<dis_type_homogenized>::rule);

  const double* gp_coords =
      integration_points_stab.ip().qxg[0];  // actual integration point (coords)
  const double gp_weights =
      integration_points_stab.ip().qwgt[0];  // actual integration point (weight)

  Core::LinAlg::Matrix<num_dim_, 1> local_coords(gp_coords, true);
  Core::LinAlg::Matrix<num_nodes_homogenized_, 1> shape_function;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_function_deriv;
  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_> inverse_jacobian_matrix;

  // evaluate shape functions and their first derivatives at the Gauss point
  Core::FE::shape_function<dis_type_homogenized>(local_coords, shape_function);
  Core::FE::shape_function_deriv1<dis_type_homogenized>(local_coords, shape_function_deriv);

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
  jacobian_matrix.multiply_nt(shape_function_deriv, nodal_coords_homogenized_ele_);
  double jacobian_determinant = inverse_jacobian_matrix.invert(jacobian_matrix);

  if (jacobian_determinant < 1E-16)
    FOUR_C_THROW("GLOBAL ELEMENT ZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", jacobian_determinant);

  // compute integration factor
  return gp_weights * jacobian_determinant;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::set_segment_id(const int& segment_id)
{
  segment_id_ = segment_id;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
double PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::apply_mesh_movement(const bool first_call,
    const std::shared_ptr<Core::FE::Discretization> homogenized_dis)
{
  // nodal displacement values for ALE
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> ele_disp_np;

  if (!first_call)
  {
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp_np =
        homogenized_dis->get_state(1, "dispnp");
    Core::Elements::LocationArray la(homogenized_dis->num_dof_sets());
    homogenized_element_->location_vector(*homogenized_dis, la);

    // construct location vector for displacement related dofs
    std::vector lm_disp(num_dim_ * num_nodes_homogenized_, -1);
    for (unsigned int inode = 0; inode < num_nodes_homogenized_; ++inode)
      for (unsigned int i_dim = 0; i_dim < num_dim_; ++i_dim)
        lm_disp[inode * num_dim_ + i_dim] = la[1].lm_[inode * num_dim_ + i_dim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_>>(
        *disp_np, ele_disp_np, lm_disp);
  }
  else
  {
    return (artery_segment_end_ - artery_segment_start_) / 2.0 * artery_ele_length_ref_;
  }

  // update current configuration
  nodal_coords_homogenized_ele_.update(
      1.0, nodal_coords_homogenized_ele_ref_, 1.0, ele_disp_np, 0.0);

  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv_xyz;
  // deformation gradient: dx/dX = F
  Core::LinAlg::Matrix<num_dim_, num_dim_> deformation_gradient;
  // deformed artery orientation: F * initial orientation (t0)
  Core::LinAlg::Matrix<num_dim_, 1> deformed_artery_orientation;

  current_segment_length_ = 0.0;
  // current segment length = \int_{\eta_a}^{eta_b} || F*t0 ||_2 ds
  // all under the assumption that the artery element completely follows the deformation of
  // the underlying 2D/3D problem
  for (int i_gp = 0; i_gp < num_gp_; i_gp++)
  {
    // get shape functions of the homogenized element
    const std::vector<double> gp_coord_homogenized = gp_coords_homogenized_[i_gp];
    get_homogenized_shape_functions<double>(
        shape_functions_homogenized, shape_functions_homogenized_deriv, gp_coord_homogenized);
    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    shape_functions_homogenized_deriv_xyz.multiply(
        inverse_jacobian_matrix_[i_gp], shape_functions_homogenized_deriv);
    // dx/dX = x * N_XYZ^T
    deformation_gradient.multiply_nt(
        nodal_coords_homogenized_ele_, shape_functions_homogenized_deriv_xyz);
    deformed_artery_orientation.multiply(deformation_gradient, initial_artery_orientation_);
    current_segment_length_ +=
        deformed_artery_orientation.norm2() * gp_weights_[i_gp] * jacobian_determinant_;
  }

  return current_segment_length_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::recompute_gp_coords_in_deformed_configuration(const std::vector<double>& segment_lengths,
    std::vector<double>& gp_coords_artery, std::vector<std::vector<double>>& gp_coords_homogenized,
    double& artery_coords_start, double& artery_coords_end)
{
  // NOTE: We assume that the 1D artery element completely follows the deformation of the
  // underlying porous medium and hence its length might change. Interaction between artery element
  // and porous medium has to be evaluated in current/deformed configuration. However, Gauss points
  // of the original projection (in reference configuration) cannot be used anymore, but we must
  // define a new parameter space [-1, 1] which maps into the current configuration.
  // First, we determine the new start and end coordinates of this segment as sum of segment lengths
  //     artery_coord_start_new = -1.0 + 2.0 * (\sum_{i=0}_{this_seg-1} l_i / total_ele_length)
  //     artery_coord_end_new = -1.0 + 2.0 * (\sum_{i=0}_{this_seg} l_i / total_ele_length)
  // Second, GPs are distributed in this interval.
  // The last step is to get the new projected coordinates in the 2D/3D parameter space of the
  // homogenized element. For each new GP, this can be done by finding the point in reference
  // configuration which deforms to the point in current configuration where the GP now lies as
  // \int_{\eta_a}^{eta_s} || F*t0 ||_2 ds
  // where eta_s is unknown. Linearization of this nonlinear equation within the Newton loop is
  // done with FAD.

  // not necessary if we do not take into account mesh movement or node-to-point-coupling
  if (evaluate_in_ref_config_)
  {
    gp_coords_artery = gp_coords_artery_;
    gp_coords_homogenized = gp_coords_homogenized_;
    artery_ele_length_ = artery_ele_length_ref_;
  }
  else
  {
    // Vectors for shape functions and their derivatives
    Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
    Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;
    // Coords and derivatives of the artery element in reference configuration
    Core::LinAlg::Matrix<num_dim_, 1> coords_artery_ref;
    Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv_ref;

    // current length of artery
    artery_ele_length_ = std::accumulate(segment_lengths.begin(), segment_lengths.end(), 0.0);

    // length of segments [0, 1, ..., this_seg-1]
    double length_so_far = 0.0;
    for (int i_seg = 0; i_seg < segment_id_; i_seg++) length_so_far += segment_lengths[i_seg];

    // length of this segment
    const double curr_seg_length = segment_lengths[segment_id_];

    // get new Gauss point coordinates
    artery_coords_start = -1.0 + 2.0 * (length_so_far / artery_ele_length_);
    artery_coords_end = -1.0 + 2.0 * ((length_so_far + curr_seg_length) / artery_ele_length_);

    auto gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
    if (num_dim_ == 3)
      gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

    // distribute new Gauss points
    const double determinant = (artery_coords_end - artery_coords_start) / 2.0;
    for (int i_gp = 0; i_gp < num_gp_; i_gp++)
      gp_coords_artery[i_gp] =
          (artery_coords_start + artery_coords_end) / 2.0 + gauss_points.qxg[i_gp][0] * determinant;

    // solution variable for Newton loop
    FAD gp_coords_deformed = 0.0;
    gp_coords_deformed.diff(0, 1);  // independent variable 0 out of a total of 1

    // Gauss point loop
    bool converged = false;
    for (int i_gp = 0; i_gp < num_gp_; i_gp++)
    {
      // start value for Newton: last converged value of previous evaluation
      // (should be pretty close)
      gp_coords_deformed.val() = previous_gp_coords_deformed_[i_gp];
      const double desired_length = curr_seg_length *
                                    (gp_coords_artery[i_gp] - artery_coords_start) /
                                    (artery_coords_end - artery_coords_start);
      double value = -1.0;
      // Newton loop
      for (int i_step = 0; i_step < 10; i_step++)
      {
        // integrate \int_{\eta_start}^{eta_def} || F*t0 ||_2 ds
        const FAD current_length = integrate_length_to_deformed_coords(gp_coords_deformed);

        value = current_length.val() - desired_length;

        if (fabs(value) < 1.0e-9)
        {
          converged = true;
          break;
        }
        const double deriv = current_length.fastAccessDx(0);
        // Newton update
        gp_coords_deformed.val() -= value / deriv;
      }
      if (!converged)
        std::cout << "WARNING: could not find Gauss point position in reference configuration.";
      // save the converged value
      previous_gp_coords_deformed_[i_gp] = gp_coords_deformed.val();

      // Finally, find new coordinate in the homogenized domain by projection eta_s in reference
      // configuration.

      // Update coordinates and derivatives for 1D and 2D/3D element
      get_artery_shape_functions<double>(
          shape_functions_artery, shape_functions_artery_deriv, gp_coords_deformed.val());
      compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
          shape_functions_artery, shape_functions_artery_deriv);

      bool projection_valid = false;
      projection<double>(coords_artery_ref, gp_coords_homogenized[i_gp], projection_valid);
      if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_gpts(const std::vector<double>& gp_coords_artery,
    const std::vector<std::vector<double>>& gp_coords_homogenized,
    const std::vector<double>& segment_lengths, Core::LinAlg::SerialDenseVector* ele_rhs_artery,
    Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized)
{
  if (num_coupled_dofs_ > 0)
  {
    if (!constant_part_evaluated_)
    {
      gpts_ntp_ele_matrix_artery_artery_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_artery_homogenized_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_homogenized_artery_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_homogenized_homogenized_ = Core::LinAlg::SerialDenseMatrix();
    }

    // we only have to do this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

      Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
      Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;

      gpts_ntp_ele_matrix_artery_artery_.shape(dim_artery_, dim_artery_);
      gpts_ntp_ele_matrix_artery_homogenized_.shape(dim_artery_, dim_homogenized_);
      gpts_ntp_ele_matrix_homogenized_artery_.shape(dim_homogenized_, dim_artery_);
      gpts_ntp_ele_matrix_homogenized_homogenized_.shape(dim_homogenized_, dim_homogenized_);

      const double current_segment_length = segment_lengths[segment_id_];

      for (int i_gp = 0; i_gp < num_gp_; i_gp++)
      {
        // Get constant values from projection
        const double current_gp_weight = gp_weights_[i_gp];
        const double current_gp_coord_artery = gp_coords_artery[i_gp];
        const std::vector<double>& current_gp_coords_homogenized = gp_coords_homogenized[i_gp];
        const double jacobian_determinant = current_segment_length / 2.0;

        // Update shape functions of the artery and homogenized element
        get_artery_shape_functions<double>(
            shape_functions_artery, shape_functions_artery_deriv, current_gp_coord_artery);
        get_homogenized_shape_functions<double>(shape_functions_homogenized,
            shape_functions_homogenized_deriv, current_gp_coords_homogenized);

        evaluate_gpts_element_matrix(current_gp_weight, shape_functions_artery,
            shape_functions_homogenized, jacobian_determinant, penalty_parameter_);
      }
    }

    update_gpts_ntp_element_matrix(*ele_matrix_artery_artery, *ele_matrix_artery_homogenized,
        *ele_matrix_homogenized_artery, *ele_matrix_homogenized_homogenized);
    check_valid_volume_fraction_pressure_coupling(*ele_matrix_artery_artery,
        *ele_matrix_artery_homogenized, *ele_matrix_homogenized_artery,
        *ele_matrix_homogenized_homogenized);
    evaluate_gpts_ntp_ele_rhs(*ele_rhs_artery, *ele_rhs_homogenized, *ele_matrix_artery_artery,
        *ele_matrix_artery_homogenized, *ele_matrix_homogenized_artery,
        *ele_matrix_homogenized_homogenized);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_ntp(const std::vector<double>& gp_coords_artery,
    const std::vector<std::vector<double>>& gp_coords_homogenized,
    Core::LinAlg::SerialDenseVector* ele_rhs_artery,
    Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized)
{
  if (num_coupled_dofs_ > 0)
  {
    if (!constant_part_evaluated_)
    {
      gpts_ntp_ele_matrix_artery_artery_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_artery_homogenized_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_homogenized_artery_ = Core::LinAlg::SerialDenseMatrix();
      gpts_ntp_ele_matrix_homogenized_homogenized_ = Core::LinAlg::SerialDenseMatrix();
    }

    // we only have to this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

      Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
      Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;

      gpts_ntp_ele_matrix_artery_artery_.shape(dim_artery_, dim_artery_);
      gpts_ntp_ele_matrix_artery_homogenized_.shape(dim_artery_, dim_homogenized_);
      gpts_ntp_ele_matrix_homogenized_artery_.shape(dim_homogenized_, dim_artery_);
      gpts_ntp_ele_matrix_homogenized_homogenized_.shape(dim_homogenized_, dim_homogenized_);


      // Get constant values from projection
      const double current_gp_coord_artery = gp_coords_artery[0];
      const std::vector<double>& current_gp_coords_homogenized = gp_coords_homogenized[0];

      // Update shape functions and their derivatives for 1D and 2D/3D element
      get_artery_shape_functions<double>(
          shape_functions_artery, shape_functions_artery_deriv, current_gp_coord_artery);
      get_homogenized_shape_functions<double>(shape_functions_homogenized,
          shape_functions_homogenized_deriv, current_gp_coords_homogenized);

      // evaluate
      evaluate_ntp_element_matrix(
          shape_functions_artery, shape_functions_homogenized, penalty_parameter_);
    }
  }

  update_gpts_ntp_element_matrix(*ele_matrix_artery_artery, *ele_matrix_artery_homogenized,
      *ele_matrix_homogenized_artery, *ele_matrix_homogenized_homogenized);


  evaluate_gpts_ntp_ele_rhs(*ele_rhs_artery, *ele_rhs_homogenized, *ele_matrix_artery_artery,
      *ele_matrix_artery_homogenized, *ele_matrix_homogenized_artery,
      *ele_matrix_homogenized_homogenized);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_mortar_coupling(const std::vector<double>&
                                                             gp_coords_artery,
    const std::vector<std::vector<double>>& gp_coords_homogenized,
    const std::vector<double>& segment_lengths, Core::LinAlg::SerialDenseMatrix* mortar_matrix_d,
    Core::LinAlg::SerialDenseMatrix* mortar_matrix_m,
    Core::LinAlg::SerialDenseVector* mortar_vector_kappa)
{
  if (mortar_matrix_d != nullptr) mortar_matrix_d->shape(dim_artery_, dim_artery_);
  if (mortar_matrix_m != nullptr) mortar_matrix_m->shape(dim_artery_, dim_homogenized_);
  if (mortar_vector_kappa != nullptr) mortar_vector_kappa->size(dim_artery_);

  if (num_coupled_dofs_ > 0)
  {
    // initialize
    if (!constant_part_evaluated_)
    {
      mortar_matrix_d_ = Core::LinAlg::SerialDenseMatrix();
      mortar_matrix_m_ = Core::LinAlg::SerialDenseMatrix();
    }
    // we only have to this once if evaluated in reference configuration
    if (!constant_part_evaluated_ or !evaluate_in_ref_config_)
    {
      // Vectors for shape functions and their derivatives
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
      Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

      Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
      Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;

      mortar_matrix_d_.shape(dim_artery_, dim_artery_);
      mortar_matrix_m_.shape(dim_artery_, dim_homogenized_);
      mortar_vector_kappa_.size(dim_artery_);

      const double current_segment_length = segment_lengths[segment_id_];

      for (int i_gp = 0; i_gp < num_gp_; i_gp++)
      {
        // Get constant values from projection
        const double current_gp_weight = gp_weights_[i_gp];
        const double current_gp_coord_artery = gp_coords_artery[i_gp];
        const std::vector<double>& current_gp_coords_homogenized = gp_coords_homogenized[i_gp];
        const double jacobian_determinant = current_segment_length / 2.0;

        // Update shape functions and their derivatives for 1D and 2D/3D element
        get_artery_shape_functions<double>(
            shape_functions_artery, shape_functions_artery_deriv, current_gp_coord_artery);
        get_homogenized_shape_functions<double>(shape_functions_homogenized,
            shape_functions_homogenized_deriv, current_gp_coords_homogenized);

        evaluate_mortar_matrices_and_vector(current_gp_weight, shape_functions_artery,
            shape_functions_homogenized, jacobian_determinant);
      }
    }

    update_mortar_matrices_and_vector(*mortar_matrix_d, *mortar_matrix_m, *mortar_vector_kappa);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_function_coupling(const std::vector<double>&
                                                               gp_coords_artery,
    const std::vector<std::vector<double>>& gp_coords_homogenized,
    const std::vector<double>& segment_lengths, Core::LinAlg::SerialDenseVector* ele_rhs_artery,
    Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized,
    double& integrated_diameter)
{
  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_nodes_homogenized_, 1> shape_functions_homogenized_transpose;

  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv_xyz;

  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix_ref;
  Core::LinAlg::Matrix<num_dim_, num_dim_> inverse_jacobian_matrix;

  const double current_segment_length = segment_lengths[segment_id_];

  // case with variable diameter and type porofluid:
  // integral and linearizations have to be calculated --> reset
  if (variable_diameter_active_ && coupling_type_ == CouplingType::porofluid)
  {
    integrated_diameter = 0.0;
    diameter_ele_matrix_artery_artery_.shape(dim_artery_, dim_artery_);
    diameter_ele_matrix_artery_homogenized_.shape(dim_artery_, dim_homogenized_);
  }

  for (int i_gp = 0; i_gp < num_gp_; i_gp++)
  {
    // clear current gauss point data for safety
    phase_manager_->clear_gp_state();

    // Get constant values from projection
    const double current_gp_weight = gp_weights_[i_gp];
    const double current_gp_coord_artery = gp_coords_artery[i_gp];
    const std::vector<double>& current_gp_coords_homogenized = gp_coords_homogenized[i_gp];

    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_artery_shape_functions<double>(
        shape_functions_artery, shape_functions_artery_deriv, current_gp_coord_artery);
    get_homogenized_shape_functions<double>(shape_functions_homogenized,
        shape_functions_homogenized_deriv, current_gp_coords_homogenized);

    jacobian_matrix.multiply_nt(shape_functions_homogenized_deriv, nodal_coords_homogenized_ele_);
    jacobian_matrix_ref.multiply_nt(
        shape_functions_homogenized_deriv, nodal_coords_homogenized_ele_ref_);

    shape_functions_homogenized_deriv_xyz.multiply(
        inverse_jacobian_matrix, shape_functions_homogenized_deriv);
    shape_functions_homogenized_transpose.update_t(shape_functions_homogenized);
    variable_manager_->evaluate_gp_variables(
        shape_functions_homogenized_transpose, shape_functions_homogenized_deriv_xyz);

    const double jacobian_determinant = inverse_jacobian_matrix.invert(jacobian_matrix);
    // inverse of transposed jacobian "ds/dX"
    const double jacobian_determinant_ref = jacobian_matrix_ref.determinant();
    // determinant of deformation gradient
    // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
    const double determinant_deformation_gradient = jacobian_determinant / jacobian_determinant_ref;
    phase_manager_->evaluate_gp_state(
        determinant_deformation_gradient, *variable_manager_, nds_porofluid_);

    const double jacobian_determinant_artery = current_segment_length / 2.0;
    evaluate_function_coupling(current_gp_weight, shape_functions_artery,
        shape_functions_artery_deriv, shape_functions_homogenized, jacobian_determinant_artery,
        *ele_rhs_artery, *ele_rhs_homogenized, *ele_matrix_artery_artery,
        *ele_matrix_artery_homogenized, *ele_matrix_homogenized_artery,
        *ele_matrix_homogenized_homogenized, integrated_diameter);
  }
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_nds_solid_velocity(const std::vector<double>&
                                                                gp_coords_artery,
    const std::vector<std::vector<double>>& gp_coords_homogenized,
    Core::LinAlg::SerialDenseVector& ele_rhs_artery, const double& artery_segment_start,
    const double& artery_segment_end)
{
  if (evaluate_in_ref_config_ || coupling_type_ == CouplingType::scatra ||
      coupling_method_ == ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point)
    return;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_xyz;
  Core::LinAlg::Matrix<num_dim_, num_dim_> deformation_gradient;

  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix_ref;
  Core::LinAlg::Matrix<num_dim_, num_dim_> inverse_jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_> inverse_jacobian_matrix_ref;

  // artery orientation in current configuration (lambda_t)
  Core::LinAlg::Matrix<num_dim_, 1> current_artery_orientation;

  // Evaluate
  // $-\int_a^b d N^(1)/ds*pi*R^2 * lambda_t*v_s ds$
  //  = $-\int_\eta_a^\eta_b d N^(1)/deta*2/L_ele*pi*R^2 * lambda_t * v_s * L_seg/2.0 d\eta$
  //  = $-\int_\eta_a^\eta_b d N^(1)/deta*pi*R^2 * lambda_t * v_s * (\eta_a-\eta_b)/2.0 d\eta$
  const double jacobian_determinant_artery = (artery_segment_end - artery_segment_start) / 2.0;

  for (int i_gp = 0; i_gp < num_gp_; i_gp++)
  {
    // Get constant values from projection
    const double current_gp_weight = gp_weights_[i_gp];
    const double current_gp_coord_artery = gp_coords_artery[i_gp];
    const std::vector<double>& current_gp_coords_homogenized = gp_coords_homogenized[i_gp];

    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_artery_shape_functions<double>(
        shape_functions_artery, shape_functions_artery_deriv, current_gp_coord_artery);
    get_homogenized_shape_functions<double>(shape_functions_homogenized,
        shape_functions_homogenized_deriv, current_gp_coords_homogenized);

    jacobian_matrix.multiply_nt(shape_functions_homogenized_deriv, nodal_coords_homogenized_ele_);
    inverse_jacobian_matrix.invert(jacobian_matrix);

    // dX/dpsi
    jacobian_matrix_ref.multiply_nt(
        shape_functions_homogenized_deriv, nodal_coords_homogenized_ele_ref_);
    // dpsi/dX
    // note: cannot use invJ_ here -> defined at original Gauss points
    inverse_jacobian_matrix_ref.invert(jacobian_matrix_ref);
    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    shape_functions_homogenized_xyz.multiply(
        inverse_jacobian_matrix_ref, shape_functions_homogenized_deriv);
    // dx/dX = x * N_XYZ^T
    deformation_gradient.multiply_nt(
        nodal_coords_homogenized_ele_, shape_functions_homogenized_xyz);

    // current orientation of the artery element at GP
    current_artery_orientation.multiply(deformation_gradient, initial_artery_orientation_);
    current_artery_orientation.scale(1.0 / current_artery_orientation.norm2());

    std::vector velocity(3, 0.0);
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
      for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
        velocity[i_dim] +=
            shape_functions_homogenized(j) * nodal_velocity_homogenized_ele_(i_dim, j);

    // compute the scalar product of velocity and current artery orientation (lambda_t * v_s)
    double velocity_x_current_artery_orientation = 0.0;
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
      velocity_x_current_artery_orientation += velocity[i_dim] * current_artery_orientation(i_dim);

    // TODO: here reference diameter is used
    for (unsigned int i = 0; i < num_nodes_artery_; i++)
    {
      ele_rhs_artery(i) += shape_functions_artery_deriv(i) * current_gp_weight *
                           jacobian_determinant_artery * velocity_x_current_artery_orientation *
                           artery_diameter_ref_ * artery_diameter_ref_ * std::numbers::pi / 4.0;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_gpts_element_matrix(const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobian_matrix, const double& penalty_parameter)
{
  // Evaluate meshtying element matrix contribution for artery element N_1^T * N_1
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double contribution = timefacrhs_artery_ * penalty_parameter * jacobian_matrix *
                                  gp_weight * shape_functions_artery(i) * shape_functions_artery(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_artery_artery_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_artery_ + coupled_dofs_artery_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for artery element "mixed" N_1^T * (-N_2)
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double contribution = timefacrhs_artery_ * penalty_parameter * jacobian_matrix *
                                  gp_weight * shape_functions_artery(i) *
                                  (-shape_functions_homogenized(j));
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_artery_homogenized_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_homogenized_ + coupled_dofs_homogenized_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for homogenized element "mixed" N_2^T * (-N_1)
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double contribution = timefacrhs_homogenized_ * penalty_parameter * jacobian_matrix *
                                  gp_weight * shape_functions_homogenized(i) *
                                  (-shape_functions_artery(j));
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_homogenized_artery_(
            i * num_dof_homogenized_ + coupled_dofs_homogenized_[dof],
            j * num_dof_artery_ + coupled_dofs_artery_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for homogenized element N_2^T * N_2
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double contribution = timefacrhs_homogenized_ * penalty_parameter * jacobian_matrix *
                                  gp_weight * shape_functions_homogenized(i) *
                                  shape_functions_homogenized(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_homogenized_homogenized_(
            i * num_dof_homogenized_ + coupled_dofs_homogenized_[dof],
            j * num_dof_homogenized_ + coupled_dofs_homogenized_[dof]) += contribution;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::evaluate_ntp_element_matrix(const Core::LinAlg::Matrix<1, num_nodes_artery_>&
                                          shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& penalty_parameter)
{
  // Evaluate meshtying element matrix contribution for artery element N_1^T * N_1
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double contribution = timefacrhs_artery_ * penalty_parameter *
                                  shape_functions_artery(i) * shape_functions_artery(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_artery_artery_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_artery_ + coupled_dofs_artery_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for artery element "mixed" N_1^T * (-N_2)
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double contribution = timefacrhs_artery_ * penalty_parameter *
                                  shape_functions_artery(i) * (-shape_functions_homogenized(j));
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_artery_homogenized_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_homogenized_ + coupled_dofs_homogenized_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for homogenized element "mixed" N_2^T * (-N_1)
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double contribution = timefacrhs_homogenized_ * penalty_parameter *
                                  shape_functions_homogenized(i) * (-shape_functions_artery(j));
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_homogenized_artery_(
            i * num_dof_homogenized_ + coupled_dofs_homogenized_[dof],
            j * num_dof_artery_ + coupled_dofs_artery_[dof]) += contribution;
    }
  }

  // Evaluate meshtying element matrix contribution for homogenized element N_2^T * N_2
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double contribution = timefacrhs_homogenized_ * penalty_parameter *
                                  shape_functions_homogenized(i) * shape_functions_homogenized(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        gpts_ntp_ele_matrix_homogenized_homogenized_(
            i * num_dof_homogenized_ + coupled_dofs_homogenized_[dof],
            j * num_dof_homogenized_ + coupled_dofs_homogenized_[dof]) += contribution;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_mortar_matrices_and_vector(const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobian_matrix)
{
  // Evaluate element mortar coupling operator kappa = N_1
  for (unsigned int inode = 0; inode < num_nodes_artery_; inode++)
  {
    const double kappa = gp_weight * jacobian_matrix * shape_functions_artery(inode);
    for (int dof = 0; dof < num_dof_artery_; dof++)
      mortar_vector_kappa_(inode * num_dof_artery_ + dof) += kappa;
  }

  // Evaluate element mortar coupling operator D = N_1^T * N_1
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double D =
          jacobian_matrix * gp_weight * shape_functions_artery(i) * shape_functions_artery(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        mortar_matrix_d_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_artery_ + coupled_dofs_artery_[dof]) += D;
    }
  }

  // Evaluate element mortar coupling operator M = N_1^T * N_2
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double M =
          jacobian_matrix * gp_weight * shape_functions_artery(i) * shape_functions_homogenized(j);
      for (int dof = 0; dof < num_coupled_dofs_; dof++)
        mortar_matrix_m_(i * num_dof_artery_ + coupled_dofs_artery_[dof],
            j * num_dof_homogenized_ + coupled_dofs_homogenized_[dof]) += M;
    }
  }

  constant_part_evaluated_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_gpts_ntp_ele_rhs(Core::LinAlg::SerialDenseVector&
                                                              ele_rhs_artery,
    Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
    const Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
    const Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
    const Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
    const Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized) const
{
  // Evaluate meshtying rhs contributions for artery element
  for (int i = 0; i < dim_artery_; i++)
    for (int j = 0; j < dim_artery_; j++)
      ele_rhs_artery(i) -= ele_matrix_artery_artery(i, j) * phi_np_artery_ele_[j];

  for (int i = 0; i < dim_artery_; i++)
    for (int j = 0; j < dim_homogenized_; j++)
      ele_rhs_artery(i) -= ele_matrix_artery_homogenized(i, j) * phi_np_homogenized_ele_[j];

  // Evaluate meshtying rhs contributions for homogenized element
  for (int i = 0; i < dim_homogenized_; i++)
    for (int j = 0; j < dim_artery_; j++)
      ele_rhs_homogenized(i) -= ele_matrix_homogenized_artery(i, j) * phi_np_artery_ele_[j];

  for (int i = 0; i < dim_homogenized_; i++)
  {
    for (int j = 0; j < dim_homogenized_; j++)
      ele_rhs_homogenized(i) -=
          ele_matrix_homogenized_homogenized(i, j) * phi_np_homogenized_ele_[j];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::update_gpts_ntp_element_matrix(Core::LinAlg::SerialDenseMatrix&
                                                                   ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized) const
{
  ele_matrix_artery_artery.assign(gpts_ntp_ele_matrix_artery_artery_);
  ele_matrix_artery_homogenized.assign(gpts_ntp_ele_matrix_artery_homogenized_);
  ele_matrix_homogenized_artery.assign(gpts_ntp_ele_matrix_homogenized_artery_);
  ele_matrix_homogenized_homogenized.assign(gpts_ntp_ele_matrix_homogenized_homogenized_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::check_valid_volume_fraction_pressure_coupling(Core::LinAlg::SerialDenseMatrix&
                                                            ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized)
{
  for (int i_dof = 0; i_dof < num_coupled_dofs_; i_dof++)
  {
    if (!variable_manager_->element_has_valid_vol_frac_pressure(
            volfrac_pressure_id_[i_dof] - num_fluid_phases_ - num_volfracs_))
    {
      // reset to zero for this dof
      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_artery_; j++)
          ele_matrix_artery_artery(i * num_dof_artery_ + coupled_dofs_artery_[i_dof],
              j * num_dof_artery_ + coupled_dofs_artery_[i_dof]) = 0.0;
      }

      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
          ele_matrix_artery_homogenized(i * num_dof_artery_ + coupled_dofs_artery_[i_dof],
              j * num_dof_homogenized_ + coupled_dofs_homogenized_[i_dof]) = 0.0;
      }

      for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_artery_; j++)
          ele_matrix_homogenized_artery(i * num_dof_homogenized_ + coupled_dofs_homogenized_[i_dof],
              j * num_dof_artery_ + coupled_dofs_artery_[i_dof]) = 0.0;
      }

      for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
          ele_matrix_homogenized_homogenized(
              i * num_dof_homogenized_ + coupled_dofs_homogenized_[i_dof],
              j * num_dof_homogenized_ + coupled_dofs_homogenized_[i_dof]) = 0.0;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::update_mortar_matrices_and_vector(Core::LinAlg::SerialDenseMatrix&
                                                                      mortar_matrix_d,
    Core::LinAlg::SerialDenseMatrix& mortar_matrix_m,
    Core::LinAlg::SerialDenseVector& mortar_vector_kappa)
{
  mortar_matrix_d.assign(mortar_matrix_d_);
  mortar_matrix_m.assign(mortar_matrix_m_);
  mortar_vector_kappa.assign(mortar_vector_kappa_);

  for (int i_dof = 0; i_dof < num_coupled_dofs_; i_dof++)
  {
    // This coupling is only possible if we also have an element with valid volume fraction
    // pressure, i.e., if we also have a homogenized representation of the neovasculature at this
    // point. If this is not the case, the corresponding matrices are set to zero.
    if (!variable_manager_->element_has_valid_vol_frac_pressure(
            volfrac_pressure_id_[i_dof] - num_fluid_phases_ - num_volfracs_))
    {
      // reset to zero for this dof
      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_artery_; j++)
          mortar_matrix_d(i * num_dof_artery_ + coupled_dofs_artery_[i_dof],
              j * num_dof_artery_ + coupled_dofs_artery_[i_dof]) = 0.0;
      }

      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
          mortar_matrix_m(i * num_dof_artery_ + coupled_dofs_artery_[i_dof],
              j * num_dof_homogenized_ + coupled_dofs_homogenized_[i_dof]) = 0.0;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_function_coupling(const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery_deriv,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobi, Core::LinAlg::SerialDenseVector& ele_rhs_artery,
    Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized,
    double& integrated_diameter)
{
  std::vector artery_scalar_np_at_gp(num_scalars_artery_, 0.0);
  std::vector homogenized_scalar_np_at_gp(num_scalars_homogenized_, 0.0);
  double artery_pressure_at_gp = 0.0;
  double artery_pressure_gradient_at_gp = 0.0;

  // get artery values at GP
  get_artery_values_at_gp(shape_functions_artery, shape_functions_artery_deriv,
      artery_pressure_at_gp, artery_pressure_gradient_at_gp, artery_scalar_np_at_gp);
  const auto artery_element_flow_rate = evaluate_artery_flow_rate(artery_pressure_gradient_at_gp);

  // get scatra values at GP
  get_homogenized_scalar_values_at_gp(shape_functions_homogenized, homogenized_scalar_np_at_gp);
  // NOTE: values of fluid phases held by managers

  if (variable_diameter_active_)
  {
    evaluate_diameter_function_and_deriv(artery_pressure_at_gp, gp_weight, shape_functions_artery,
        shape_functions_homogenized, jacobi);
    // integral is only calculated in this case
    if (coupling_type_ == CouplingType::porofluid)
      integrated_diameter += gp_weight * jacobi * artery_diameter_at_gp_;
  }

  // artery functions
  for (int i_art = 0; i_art < num_dof_artery_; i_art++)
  {
    if (function_vector_[0][i_art] != nullptr)
    {
      std::vector artery_derivs(num_dof_artery_, 0.0);
      std::vector homogenized_derivs(num_dof_homogenized_, 0.0);
      double function_value = 0.0;

      // evaluate and assemble
      evaluate_function_and_deriv(*function_vector_[0][i_art], artery_pressure_at_gp,
          artery_element_flow_rate, artery_scalar_np_at_gp, homogenized_scalar_np_at_gp,
          function_value, artery_derivs, homogenized_derivs);
      assemble_function_coupling_into_ele_matrix_rhs_artery(i_art, gp_weight,
          shape_functions_artery, shape_functions_homogenized, jacobi, scale_vector_[0][i_art],
          function_value, artery_derivs, homogenized_derivs, ele_rhs_artery,
          ele_matrix_artery_artery, ele_matrix_artery_homogenized);
    }
  }
  // homogenized discretization functions
  for (int i_homo = 0; i_homo < num_dof_homogenized_; i_homo++)
  {
    if (function_vector_[1][i_homo] != nullptr)
    {
      std::vector artery_derivs(num_dof_artery_, 0.0);
      std::vector homogenized_derivs(num_dof_homogenized_, 0.0);
      double function_value = 0.0;

      // evaluate and assemble
      evaluate_function_and_deriv(*function_vector_[1][i_homo], artery_pressure_at_gp,
          artery_element_flow_rate, artery_scalar_np_at_gp, homogenized_scalar_np_at_gp,
          function_value, artery_derivs, homogenized_derivs);
      assemble_function_coupling_into_ele_matrix_rhs_homogenized(
          homogenized_dofs_to_assemble_functions_into_[i_homo], gp_weight, shape_functions_artery,
          shape_functions_homogenized, jacobi, scale_vector_[1][i_homo],
          timefacrhs_homogenized_density_[i_homo], function_value, artery_derivs,
          homogenized_derivs, ele_rhs_homogenized, ele_matrix_homogenized_artery,
          ele_matrix_homogenized_homogenized);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_diameter_function_and_deriv(const double
                                                                         artery_pressure_np_at_gp,
    const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobian_determinant)
{
  // we have to derive w.r.t. fluid variables
  std::vector<std::pair<std::string, double>> variables;
  variables.reserve(num_fluid_phases_ + num_fluid_phases_ + 1 + num_volfracs_ + num_volfracs_ + 1);

  // reference diameter is constant
  std::vector<std::pair<std::string, double>> constants;
  constants.reserve(2);
  constants.emplace_back("D0", artery_diameter_ref_);
  constants.emplace_back("Dprev", artery_material_->diam_previous_time_step());

  // set fluid values as variables
  set_fluid_values_as_variables(variables, artery_pressure_np_at_gp);

  // evaluate the diameter at GP by evaluating the function
  artery_diameter_at_gp_ = artery_diameter_funct_->evaluate(variables, constants, 0);

  // derivatives and linearizations are so far only calculated for coupling type porofluid
  if (coupling_type_ == CouplingType::porofluid)
  {
    // function derivatives
    const std::vector derivs(artery_diameter_funct_->evaluate_derivative(variables, constants, 0));
    // derivatives w.r.t. primary variables
    std::ranges::fill(diameter_derivs_.begin(), diameter_derivs_.end(), 0.0);

    // diameter derivatives w.r.t. saturations and pressures
    for (int dof_to_derive = 0; dof_to_derive < num_fluid_phases_; dof_to_derive++)
    {
      for (int i_dof = 0; i_dof < num_fluid_phases_; i_dof++)
      {
        diameter_derivs_[dof_to_derive] +=
            derivs[i_dof] * phase_manager_->pressure_deriv(i_dof, dof_to_derive) +
            derivs[i_dof + num_fluid_phases_] *
                phase_manager_->saturation_deriv(i_dof, dof_to_derive);
      }
      if (phase_manager_->porosity_depends_on_fluid())
        diameter_derivs_[dof_to_derive] +=
            derivs[2 * num_fluid_phases_] * phase_manager_->porosity_deriv(dof_to_derive);
    }
    // diameter derivatives w.r.t. to volume fraction phases
    for (int dof_to_derive = num_fluid_phases_;
        dof_to_derive < num_dof_homogenized_ - num_volfracs_; dof_to_derive++)
    {
      // diameter derivatives w.r.t. volume fractions directly appearing and porosity
      // (because it depends on volfrac)
      diameter_derivs_[dof_to_derive] +=
          derivs[dof_to_derive + num_fluid_phases_ + 1] +
          derivs[2 * num_fluid_phases_] * phase_manager_->porosity_deriv(dof_to_derive);
    }
    // diameter derivatives w.r.t. to volume fraction pressures
    for (int dof_to_derive = num_fluid_phases_ + num_volfracs_;
        dof_to_derive < num_dof_homogenized_; dof_to_derive++)
      diameter_derivs_[dof_to_derive] += derivs[dof_to_derive + num_fluid_phases_ + 1];

    // diameter derivs w.r.t. to artery pressure
    diameter_derivs_[num_fluid_phases_ + 2 * num_volfracs_] +=
        derivs[num_fluid_phases_ + num_fluid_phases_ + 1 + num_volfracs_ + num_volfracs_];

    // Now the derivative of the integrated (element) diameter needed in the Hagen-Poiseuille
    // terms is built and stored in the respective element matrices for later use.

    // pre-compute some values
    const double pressure_gradient =
        (phi_np_artery_ele_[1] - phi_np_artery_ele_[0]) / artery_ele_length_;
    const std::vector sign = {-1.0, 1.0};
    const double preprefac =
        pressure_gradient * gp_weight * jacobian_determinant / artery_ele_length_;

    // assemble into the respective element matrices
    for (unsigned int i = 0; i < num_nodes_artery_; i++)
    {
      // build diameter element matrix w.r.t. artery primary variables
      const double prefac_artery =
          sign[i] * diameter_derivs_[num_fluid_phases_ + 2 * num_volfracs_] * preprefac;
      for (unsigned int j = 0; j < num_nodes_artery_; j++)
        diameter_ele_matrix_artery_artery_(i, j) += prefac_artery * shape_functions_artery(j);

      // build diameter element matrix w.r.t. 2D/3D primary variables
      const double prefac_homogenized = sign[i] * preprefac;
      for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
      {
        const double prefac_homogenized_2 = prefac_homogenized * shape_functions_homogenized(j);
        for (int j_home = 0; j_home < num_dof_homogenized_; j_home++)
          diameter_ele_matrix_artery_homogenized_(i, j * num_dof_homogenized_ + j_home) +=
              prefac_homogenized_2 * diameter_derivs_[j_home];
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::set_time_fac_rhs(const double& artery_density,
    const Mat::MatList* scatra_material_homogenized, const double& timefacrhs_artery,
    const double& timefacrhs_homogenized)
{
  timefacrhs_artery_ = timefacrhs_artery;
  timefacrhs_homogenized_ = timefacrhs_homogenized;

  timefacrhs_homogenized_density_.resize(num_dof_homogenized_);

  // fill fluid densities vector
  std::vector<double> fluid_densities{};

  if (phase_manager_->total_num_dof() ==
      phase_manager_->num_fluid_phases() + 2 * phase_manager_->num_vol_frac())
  {
    fluid_densities.resize(num_fluid_phases_ + 2 * num_volfracs_);
    for (int i_fluid_phase = 0; i_fluid_phase < num_fluid_phases_; i_fluid_phase++)
      fluid_densities[i_fluid_phase] = phase_manager_->density(i_fluid_phase);
    for (int i_volfrac = 0; i_volfrac < num_volfracs_; i_volfrac++)
    {
      fluid_densities[num_fluid_phases_ + i_volfrac] = phase_manager_->vol_frac_density(i_volfrac);
      fluid_densities[num_fluid_phases_ + num_volfracs_ + i_volfrac] =
          phase_manager_->vol_frac_density(i_volfrac);
    }
  }

  else if (phase_manager_->total_num_dof() ==
           phase_manager_->num_fluid_phases() + phase_manager_->num_vol_frac())
  {
    fluid_densities.resize(num_fluid_phases_ + num_volfracs_);
    for (int i_fluid_phase = 0; i_fluid_phase < num_fluid_phases_; i_fluid_phase++)
      fluid_densities[i_fluid_phase] = phase_manager_->density(i_fluid_phase);

    fluid_densities[num_fluid_phases_] = phase_manager_->vol_frac_density(0);
  }
  else
  {
    FOUR_C_THROW("Internal Error!");
  }



  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      // artery
      timefacrhs_artery_density_ = timefacrhs_artery_ / artery_density;
      // homogenized
      for (int i_dof = 0; i_dof < num_dof_homogenized_; i_dof++)
        timefacrhs_homogenized_density_[i_dof] = timefacrhs_homogenized_ / fluid_densities[i_dof];

      break;
    }
    case CouplingType::scatra:
    {
      // artery
      timefacrhs_artery_density_ = timefacrhs_artery_ / artery_density;
      // homogenized
      for (int i_dof = 0; i_dof < num_dof_homogenized_; i_dof++)
      {
        const int material_id = scatra_material_homogenized->mat_id(i_dof);
        std::shared_ptr<Core::Mat::Material> single_phase_material =
            scatra_material_homogenized->material_by_id(material_id);
        int phase_id = -1;
        if (single_phase_material->material_type() ==
            Core::Materials::m_scatra_in_fluid_porofluid_pressure_based)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(single_phase_material);
          phase_id = poromat->phase_id();
        }
        else if (single_phase_material->material_type() ==
                 Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based)
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(
                  single_phase_material);
          phase_id = poromat->phase_id();
        }
        else
        {
          FOUR_C_THROW(
              "Only Mat::ScatraMatMultiPoroVolFrac and Mat::ScatraMatMultiPoroFluid. Your material "
              "is of type {}.",
              single_phase_material->material_type());
        }
        timefacrhs_homogenized_density_[i_dof] =
            timefacrhs_homogenized_ / fluid_densities[phase_id];
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::extract_velocity_solid_phase(const Core::FE::Discretization&
        homogenized_dis)
{
  // no need for this
  if (evaluate_in_ref_config_) return;

  const std::shared_ptr<const Core::LinAlg::Vector<double>> velocity =
      homogenized_dis.get_state(1, "velocity field");
  Core::Elements::LocationArray location_array(homogenized_dis.num_dof_sets());
  homogenized_element_->location_vector(homogenized_dis, location_array);

  // construct location vector for displacement related dofs
  std::vector lm_disp(num_dim_ * num_nodes_homogenized_, -1);
  for (unsigned int inode = 0; inode < num_nodes_homogenized_; ++inode)
    for (unsigned int i_dim = 0; i_dim < num_dim_; ++i_dim)
      lm_disp[inode * num_dim_ + i_dim] = location_array[1].lm_[inode * num_dim_ + i_dim];

  // extract local values of displacement field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_>>(
      *velocity, nodal_velocity_homogenized_ele_, lm_disp);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
FAD PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery, dis_type_homogenized,
    dim>::integrate_length_to_deformed_coords(const FAD& deformed_coords)
{
  FAD length = 0.0;

  // define Gauss points
  auto gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_3point);
  if (num_dim_ == 3)
    gauss_points = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_4point);

  Core::LinAlg::Matrix<1, num_nodes_homogenized_, FAD> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, FAD> shape_functions_homogenized_deriv;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_, FAD> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_, FAD> shape_functions_artery_deriv;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, FAD> shape_functions_homogenized_xyz;

  Core::LinAlg::Matrix<num_dim_, num_dim_, FAD> inverse_jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_, FAD> deformation_gradient;  // (dX/dx) = F

  // Coords and derivatives of the artery element in reference configuration
  Core::LinAlg::Matrix<num_dim_, 1, FAD> coord_artery_ref;
  Core::LinAlg::Matrix<num_dim_, 1, FAD> coord_artery_deriv_ref;

  // initial artery orientation (t0)
  Core::LinAlg::Matrix<num_dim_, 1, FAD> t0;
  for (unsigned int i = 0; i < num_dim_; i++) t0(i).val() = initial_artery_orientation_(i);

  // position of the homogenized element in reference configuration
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, FAD> homogenized_element_position_ref;
  for (unsigned int i = 0; i < num_dim_; i++)
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
      homogenized_element_position_ref(i, j).val() = nodal_coords_homogenized_ele_ref_(i, j);

  // position of the homogenized element in current configuration
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, FAD> homogenized_element_position;
  for (unsigned int i = 0; i < num_dim_; i++)
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
      homogenized_element_position(i, j).val() = nodal_coords_homogenized_ele_(i, j);

  const FAD determinant = (deformed_coords - artery_segment_start_) / 2.0;
  const FAD jacobian_determinant = determinant * artery_ele_length_ref_ / 2.0;

  // integrate from etaA to eta_s
  for (int i_gp = 0; i_gp < num_gp_; i_gp++)
  {
    const double gp_weight = gp_weights_[i_gp];
    const FAD eta =
        (deformed_coords + artery_segment_start_) / 2.0 + gauss_points.qxg[i_gp][0] * determinant;
    // Update coordinates and derivatives for 1D and 2D/3D element
    get_artery_shape_functions<FAD>(shape_functions_artery, shape_functions_artery_deriv, eta);
    compute_artery_coords_and_derivs_ref<FAD>(coord_artery_ref, coord_artery_deriv_ref,
        shape_functions_artery, shape_functions_artery_deriv);

    // project
    bool projection_valid = false;
    std::vector<FAD> coords_homogenized(num_dim_, 0.0);
    projection<FAD>(coord_artery_ref, coords_homogenized, projection_valid);
    if (!projection_valid) FOUR_C_THROW("Gauss point could not be projected");

    get_homogenized_shape_functions<FAD>(
        shape_functions_homogenized, shape_functions_homogenized_deriv, coords_homogenized);

    inverse_jacobian_matrix.multiply_nt(
        shape_functions_homogenized_deriv, homogenized_element_position_ref);
    inverse_jacobian_matrix.invert();

    // dN/dX = dN/dxi * dxi/dX = dN/dxi * (dX/dxi)^-1
    shape_functions_homogenized_xyz.multiply(
        inverse_jacobian_matrix, shape_functions_homogenized_deriv);
    // dx/dX = x * N_XYZ^T
    deformation_gradient.multiply_nt(homogenized_element_position, shape_functions_homogenized_xyz);

    // deformation gradient * initial orientation of artery (F*t0)
    Core::LinAlg::Matrix<num_dim_, 1, FAD> F_t0(Core::LinAlg::Initialization::zero);
    F_t0.multiply(deformation_gradient, t0);
    const FAD F_t0_norm = Core::FADUtils::vector_norm(F_t0);

    // finally, get the length
    length += F_t0_norm * gp_weight * jacobian_determinant;
  }

  return length;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::get_artery_values_at_gp(const Core::LinAlg::Matrix<1, num_nodes_artery_>&
                                      shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery_deriv,
    double& artery_pressure, double& artery_pressure_gradient, std::vector<double>& artery_scalars)
{
  double artery_pressure_deriv = 0.0;

  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        artery_pressure += shape_functions_artery(i) * phi_np_artery_ele_[i];
        artery_pressure_deriv += shape_functions_artery_deriv(i) * phi_np_artery_ele_[i];
        for (int i_scal = 0; i_scal < num_scalars_artery_; i_scal++)
          artery_scalars[i_scal] += shape_functions_artery(i) * nodal_artery_scalar_np_[i_scal](i);
      }
      break;
    }
    case CouplingType::scatra:
    {
      for (unsigned int i = 0; i < num_nodes_artery_; i++)
      {
        artery_pressure += shape_functions_artery(i) * nodal_artery_pressure_np_(i);
        artery_pressure_deriv += shape_functions_artery_deriv(i) * nodal_artery_pressure_np_(i);
        for (int i_art = 0; i_art < num_dof_artery_; i_art++)
          artery_scalars[i_art] +=
              shape_functions_artery(i) * phi_np_artery_ele_[i * num_dof_artery_ + i_art];
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
  // the artery pressure gradient is the derivative with respect to the parameter space coordinate
  // multiplied with the inverse Jacobian matrix (in case of 1D line elements: dxi/ds = 2 / L)
  // grad p = (dp/dxi) * (dxi/ds) = (dp/dxi) * (2 / L)
  artery_pressure_gradient = (2.0 / artery_ele_length_) * artery_pressure_deriv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::get_homogenized_scalar_values_at_gp(const Core::LinAlg::Matrix<1, num_nodes_homogenized_>&
                                                  shape_functions_homogenized,
    std::vector<double>& scalars_homogenized_np)
{
  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
      {
        for (int i_homo = 0; i_homo < num_scalars_homogenized_; i_homo++)
          scalars_homogenized_np[i_homo] +=
              shape_functions_homogenized(i) * nodal_homogenized_scalar_np_[i_homo](i);
      }
      break;
    }
    case CouplingType::scatra:
    {
      for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
      {
        for (int i_homo = 0; i_homo < num_dof_homogenized_; i_homo++)
        {
          scalars_homogenized_np[i_homo] +=
              shape_functions_homogenized(i) *
              phi_np_homogenized_ele_[i * num_dof_homogenized_ + i_homo];
        }
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::assemble_function_coupling_into_ele_matrix_rhs_artery(const int&
                                                                                          i_art,
    const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobian_determinant, const int& scale, const double& function_value,
    const std::vector<double>& artery_derivs, const std::vector<double>& homogenized_derivs,
    Core::LinAlg::SerialDenseVector& ele_rhs_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized)
{
  const double my_scale = scale;

  // rhs ---> +
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    ele_rhs_artery(i * num_dof_artery_ + i_art) += shape_functions_artery(i) * my_scale *
                                                   gp_weight * jacobian_determinant *
                                                   function_value * timefacrhs_artery_density_;
  }

  // matrix --> -
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      for (int j_art = 0; j_art < num_dof_artery_; j_art++)
      {
        ele_matrix_artery_artery(i * num_dof_artery_ + i_art, j * num_dof_artery_ + j_art) -=
            shape_functions_artery(i) * shape_functions_artery(j) * my_scale * gp_weight *
            jacobian_determinant * artery_derivs[j_art] * timefacrhs_artery_density_;
      }
    }
  }

  // matrix --> -
  for (unsigned int i = 0; i < num_nodes_artery_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      for (int j_home = 0; j_home < num_dof_homogenized_; j_home++)
      {
        ele_matrix_artery_homogenized(
            i * num_dof_artery_ + i_art, j * num_dof_homogenized_ + j_home) -=
            shape_functions_artery(i) * shape_functions_homogenized(j) * my_scale * gp_weight *
            jacobian_determinant * homogenized_derivs[j_home] * timefacrhs_artery_density_;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::assemble_function_coupling_into_ele_matrix_rhs_homogenized(const std::vector<int>&
                                                                         assemble_into,
    const double& gp_weight,
    const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
    const double& jacobian_determinant, const int& scale, const double& timefacrhs_homogenized,
    const double& function_value, const std::vector<double>& artery_derivs,
    const std::vector<double>& homogenized_derivs,
    Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
    Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized)
{
  const double my_scale = scale;

  // rhs ---> +
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    const double rhs_value = shape_functions_homogenized(i) * my_scale * gp_weight *
                             jacobian_determinant * function_value * timefacrhs_homogenized;
    for (const int i_dof : assemble_into)
      ele_rhs_homogenized(i * num_dof_homogenized_ + i_dof) += rhs_value;
  }

  // matrix --> -
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_artery_; j++)
    {
      const double matrix_fac = shape_functions_homogenized(i) * shape_functions_artery(j) *
                                my_scale * gp_weight * jacobian_determinant *
                                timefacrhs_homogenized;
      for (int j_art = 0; j_art < num_dof_artery_; j_art++)
      {
        const double matrix_value = matrix_fac * artery_derivs[j_art];
        for (const int i_dof : assemble_into)
          ele_matrix_homogenized_artery(
              i * num_dof_homogenized_ + i_dof, j * num_dof_artery_ + j_art) -= matrix_value;
      }
    }
  }

  // matrix --> -
  for (unsigned int i = 0; i < num_nodes_homogenized_; i++)
  {
    for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
    {
      const double matrix_fac = shape_functions_homogenized(i) * shape_functions_homogenized(j) *
                                my_scale * gp_weight * jacobian_determinant *
                                timefacrhs_homogenized;
      for (int j_home = 0; j_home < num_dof_homogenized_; j_home++)
      {
        const double matrix_value = matrix_fac * homogenized_derivs[j_home];
        for (const int i_dof : assemble_into)
          ele_matrix_homogenized_homogenized(
              i * num_dof_homogenized_ + i_dof, j * num_dof_homogenized_ + j_home) -= matrix_value;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_function_and_deriv(const Core::Utils::FunctionOfAnything&
                                                                function,
    const double& artery_pressure, const double& artery_element_flow_rate,
    const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars,
    double& function_value, std::vector<double>& artery_derivs,
    std::vector<double>& homogenized_derivs)
{
  double time = Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time();

  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      // we have to derive w.r.t. fluid variables plus diameter
      std::vector<std::pair<std::string, double>> variables;
      variables.reserve(
          num_fluid_phases_ + num_fluid_phases_ + 1 + num_volfracs_ + num_volfracs_ + 1 + 1);

      // scalar variables are constants
      // (plus reference artery diameter and diameter of the previous time step)
      std::vector<std::pair<std::string, double>> constants;
      constants.reserve(num_scalars_homogenized_ + num_scalars_artery_ + 2);

      set_scalar_values_as_constants(constants, artery_scalars, homogenized_scalars);

      set_fluid_values_as_variables(variables, artery_pressure);

      // set reference artery diameter as constant
      constants.emplace_back("D0", artery_diameter_ref_);

      // set artery diameter of the previous time step as constant
      constants.emplace_back("Dprev", artery_material_->diam_previous_time_step());

      // set artery diameter as variable
      variables.emplace_back("D", artery_diameter_at_gp_);

      constants.emplace_back("t", time);
      variables.emplace_back("Q_art", artery_element_flow_rate);

      // evaluate the reaction term
      function_value = function.evaluate(variables, constants, 0);
      // evaluate derivatives
      const std::vector function_derivs(function.evaluate_derivative(variables, constants, 0));

      evaluate_fluid_derivs(artery_derivs, homogenized_derivs, function_derivs);

      break;
    }
    case CouplingType::scatra:
    {
      // scalars (both homogenized and artery) are variables
      std::vector<std::pair<std::string, double>> variables;
      variables.reserve(num_scalars_homogenized_ + num_scalars_artery_);

      // fluid variables are constants
      std::vector<std::pair<std::string, double>> constants;
      constants.reserve(num_fluid_phases_ + num_fluid_phases_ + 1 + num_volfracs_ + num_volfracs_ +
                        1 + 1 + 1 + 1);

      set_scalar_values_as_variables(variables, artery_scalars, homogenized_scalars);

      set_fluid_values_as_constants(constants, artery_pressure);

      constants.emplace_back("t", time);
      variables.emplace_back("Q_art", artery_element_flow_rate);

      // evaluate the reaction term
      function_value = function.evaluate(variables, constants, 0);
      // evaluate derivatives
      const std::vector function_derivs(function.evaluate_derivative(variables, constants, 0));

      evaluate_scalar_derivs(artery_derivs, homogenized_derivs, function_derivs);

      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
double PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_artery_flow_rate(const double artery_pressure_gradient)
    const
{
  const double radius = artery_diameter_at_gp_ / 2.0;

  // Hagen-Poiseuille equation
  const double artery_element_flow_rate = -artery_pressure_gradient * std::numbers::pi * radius *
                                          radius * radius * radius /
                                          (8.0 * artery_material_->viscosity());

  return artery_element_flow_rate;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::set_scalar_values_as_constants(std::vector<std::pair<std::string, double>>& constants,
    const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars)
{
  // set homogenized scalar values as constant
  for (int k = 0; k < num_scalars_homogenized_; k++)
    constants.emplace_back(scalar_names_[k], homogenized_scalars[k]);

  // set artery-scalar values as constant
  for (int k = 0; k < num_scalars_artery_; k++)
    constants.emplace_back(artery_scalar_names_[k], artery_scalars[k]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::set_fluid_values_as_variables(std::vector<std::pair<std::string, double>>& variables,
    const double& artery_pressure)
{
  // set pressure values as variable
  for (int k = 0; k < num_fluid_phases_; k++)
    variables.emplace_back(pressure_names_[k], phase_manager_->pressure(k));

  // set saturation values as variable
  for (int k = 0; k < num_fluid_phases_; k++)
    variables.emplace_back(saturation_names_[k], phase_manager_->saturation(k));

  // set porosity value as variable
  variables.emplace_back(porosity_name_, phase_manager_->porosity());

  // set volfrac values as variables
  for (int k = 0; k < num_volfracs_; k++)
    variables.emplace_back(volfrac_names_[k], phase_manager_->vol_frac(k));

  // set volfrac pressure values as variables
  for (int k = 0; k < num_volfracs_; k++)
    variables.emplace_back(volfrac_pressure_names_[k], phase_manager_->vol_frac_pressure(k));

  // set artery pressure as variable
  variables.emplace_back(artery_pressure_name_, artery_pressure);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::set_fluid_values_as_constants(std::vector<std::pair<std::string, double>>& constants,
    const double& artery_pressure)
{
  // set pressure values as constants
  for (int k = 0; k < num_fluid_phases_; k++)
    constants.emplace_back(pressure_names_[k], phase_manager_->pressure(k));

  // set saturation values as constants
  for (int k = 0; k < num_fluid_phases_; k++)
    constants.emplace_back(saturation_names_[k], phase_manager_->saturation(k));

  // set porosity value as constants
  constants.emplace_back(porosity_name_, phase_manager_->porosity());

  // set volfrac values as constants
  for (int k = 0; k < num_volfracs_; k++)
    constants.emplace_back(volfrac_names_[k], phase_manager_->vol_frac(k));

  // set volfrac pressure values as constants
  for (int k = 0; k < num_volfracs_; k++)
    constants.emplace_back(volfrac_pressure_names_[k], phase_manager_->vol_frac_pressure(k));

  // set artery pressure as constant
  constants.emplace_back(artery_pressure_name_, artery_pressure);

  // set artery diameter as constant
  constants.emplace_back("D", artery_diameter_at_gp_);

  // set reference artery diameter as constant
  constants.emplace_back("D0", artery_diameter_ref_);

  // set artery diameter of the previous time step as constant
  constants.emplace_back("Dprev", artery_material_->diam_previous_time_step());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::set_scalar_values_as_variables(std::vector<std::pair<std::string, double>>& variables,
    const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars)
{
  // set scalar values as variables
  for (int k = 0; k < num_scalars_homogenized_; k++)
    variables.emplace_back(scalar_names_[k], homogenized_scalars[k]);

  // set artery-scalar values as variables
  for (int k = 0; k < num_scalars_artery_; k++)
    variables.emplace_back(artery_scalar_names_[k], artery_scalars[k]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_fluid_derivs(std::vector<double>& artery_derivs,
    std::vector<double>& homogenized_derivs, const std::vector<double>& function_derivs) const
{
  // Dependency of exchange terms f is as follows:
  // f(D(p),g(p)) where p are generic 1D and 3D fluid primary variables.
  // So, the derivative is df/dp = df/dg*dg/dp + df/dD*dD/dp.
  // First part: function derivs w.r.t. to fluid phases: df/dg * dp/dp.
  for (int dof_to_derive = 0; dof_to_derive < num_fluid_phases_; dof_to_derive++)
  {
    for (int i_dof = 0; i_dof < num_fluid_phases_; i_dof++)
    {
      homogenized_derivs[dof_to_derive] +=
          function_derivs[i_dof] * phase_manager_->pressure_deriv(i_dof, dof_to_derive) +
          function_derivs[i_dof + num_fluid_phases_] *
              phase_manager_->saturation_deriv(i_dof, dof_to_derive);
    }
    if (phase_manager_->porosity_depends_on_fluid())
      homogenized_derivs[dof_to_derive] +=
          function_derivs[2 * num_fluid_phases_] * phase_manager_->porosity_deriv(dof_to_derive);
  }
  // function derivs w.r.t. to volume fraction phases
  for (int dof_to_derive = num_fluid_phases_; dof_to_derive < num_dof_homogenized_ - num_volfracs_;
      dof_to_derive++)
  {
    // derivatives w.r.t. volume fractions directly appearing and porosity
    // (since it depends on volfrac)
    homogenized_derivs[dof_to_derive] +=
        function_derivs[dof_to_derive + num_fluid_phases_ + 1] +
        function_derivs[2 * num_fluid_phases_] * phase_manager_->porosity_deriv(dof_to_derive);
  }
  // function derivs w.r.t. to volume fraction pressures
  for (int dof_to_derive = num_fluid_phases_ + num_volfracs_; dof_to_derive < num_dof_homogenized_;
      dof_to_derive++)
    homogenized_derivs[dof_to_derive] += function_derivs[dof_to_derive + num_fluid_phases_ + 1];

  // function derivs w.r.t. to artery pressure
  artery_derivs[0] +=
      function_derivs[num_fluid_phases_ + num_fluid_phases_ + 1 + num_volfracs_ + num_volfracs_];

  // Second part: derivatives w.r.t. to fluid diameter: df/dD * dD/dp.
  if (variable_diameter_active_)
  {
    // df/dD
    const double funct_deriv_diameter = function_derivs[num_fluid_phases_ + num_fluid_phases_ + 1 +
                                                        num_volfracs_ + num_volfracs_ + 1];

    // now follow the terms df/dD*dD/dp
    for (int dof_to_derive = 0; dof_to_derive < num_dof_homogenized_; dof_to_derive++)
      homogenized_derivs[dof_to_derive] += funct_deriv_diameter * diameter_derivs_[dof_to_derive];

    // diameter derivs w.r.t. to artery pressure
    artery_derivs[0] +=
        funct_deriv_diameter * diameter_derivs_[num_fluid_phases_ + 2 * num_volfracs_];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::evaluate_scalar_derivs(std::vector<double>& artery_derivs,
    std::vector<double>& homogenized_derivs, const std::vector<double>& function_derivs) const
{
  // derivatives after homogenized scalars
  for (int dof_to_derive = 0; dof_to_derive < num_dof_homogenized_; dof_to_derive++)
    homogenized_derivs[dof_to_derive] += function_derivs[dof_to_derive];

  // derivatives after artery scalars
  for (int dof_to_derive = 0; dof_to_derive < num_dof_artery_; dof_to_derive++)
    artery_derivs[dof_to_derive] += function_derivs[dof_to_derive + num_dof_homogenized_];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::create_integration_segment()
{
  if (projection_output)
  {
    std::cout << "---- Getting intersections for artery element ----" << artery_ele_gid() << "\n";
  }

  std::vector<double> intersections = get_all_intersections();
  const auto num_intersections = intersections.size();

  if (projection_output)
  {
    if (num_intersections == 0)
      std::cout << "No intersections found. \n";
    else
    {
      std::cout << "Intersections are:" << '\n';
      for (double& intersection : intersections) std::cout << intersection << " ";
      std::cout << "\n";
    }
  }

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

  // Coords and derivatives of 1D and 2D/3D element
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_ref;
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv_ref;
  std::vector<double> coords_homogenized(num_dim_);
  bool projection_valid = false;

  // 1st case: no intersection found
  if (num_intersections == 0)
  {
    get_artery_shape_functions<double>(shape_functions_artery, shape_functions_artery_deriv, 0.0);
    compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
        shape_functions_artery, shape_functions_artery_deriv);
    projection<double>(coords_artery_ref, coords_homogenized, projection_valid);

    // case: completely inside
    if (projection_valid)
    {
      artery_segment_start_ = -1.0;
      artery_segment_end_ = 1.0;
      is_active_ = true;
    }

    // case: completely outside
    else
    {
      is_active_ = false;
    }
  }
  // 2nd case: 1 intersection found
  else if (num_intersections == 1)
  {
    // special case: the artery start lies directly at the boundary of 3D element
    if (fabs(intersections[0] + 1.0) < 1.0e-9)
    {
      // first possibility: segment goes from [-1; 1], second point lies inside 3D element
      get_artery_shape_functions<double>(shape_functions_artery, shape_functions_artery_deriv, 1.0);
      compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
          shape_functions_artery, shape_functions_artery_deriv);
      projection<double>(coords_artery_ref, coords_homogenized, projection_valid);
      if (projection_valid)
      {
        artery_segment_start_ = -1.0;
        artery_segment_end_ = 1.0;
        is_active_ = true;
      }
      else
      {
        // found a segment between [-1.0; -1.0 + projection_tolerance] --> can be neglected
        if (projection_output)
        {
          std::cout << "Probably found a very small integration segment for artery element "
                    << artery_ele_gid() << " and 3D element " << homogenized_ele_gid() << ". "
                    << "This segment is neglected. \n";
        }
        is_active_ = false;
      }
    }
    // special case: the artery end lies directly at the boundary of 3D element
    else if (fabs(intersections[0] - 1.0) < 1.0e-9)
    {
      // first possibility: segment goes from [-1; 1], second point lies inside 3D element
      get_artery_shape_functions<double>(
          shape_functions_artery, shape_functions_artery_deriv, -1.0);
      compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
          shape_functions_artery, shape_functions_artery_deriv);
      projection<double>(coords_artery_ref, coords_homogenized, projection_valid);
      if (projection_valid)
      {
        artery_segment_start_ = -1.0;
        artery_segment_end_ = 1.0;
        is_active_ = true;
      }
      else
      {
        // found a segment between [1.0 - projection_tolerance; 1.0] --> can be neglected
        if (projection_output)
        {
          std::cout << "Probably found a very small integration segment for artery element "
                    << artery_ele_gid() << " and 3D element " << homogenized_ele_gid() << ". "
                    << "This segment is neglected. \n";
        }
        is_active_ = false;
      }
    }
    // normal case: found one intersection: check if -1.0 or 1.0 are inside
    else
    {
      get_artery_shape_functions<double>(
          shape_functions_artery, shape_functions_artery_deriv, -1.0);
      compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
          shape_functions_artery, shape_functions_artery_deriv);
      projection<double>(coords_artery_ref, coords_homogenized, projection_valid);

      // case: the segment goes from [-1.0; intersections[0]]
      if (projection_valid)
      {
        artery_segment_start_ = -1.0;
        artery_segment_end_ = intersections[0];
        is_active_ = true;
      }
      else
      {
        // case: the segment goes from [intersections[0]; 1.0]
        get_artery_shape_functions<double>(
            shape_functions_artery, shape_functions_artery_deriv, 1.0);
        compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
            shape_functions_artery, shape_functions_artery_deriv);
        projection<double>(coords_artery_ref, coords_homogenized, projection_valid);
        artery_segment_start_ = intersections[0];
        artery_segment_end_ = 1.0;
        is_active_ = true;

        // special case: projection lies directly in the corner of element --> can be neglected
        if (!projection_valid)
        {
          if (projection_output)
          {
            std::cout << "Original point " << intersections[0] << '\n';
            std::cout << "Neither -1.0 nor 1.0 could be projected." << '\n';
          }
          is_active_ = false;
        }
      }
    }
  }
  // 3rd case: two intersections found
  else if (num_intersections == 2)
  {
    artery_segment_start_ = intersections[0];
    artery_segment_end_ = intersections[1];
    is_active_ = true;
  }
  else
  {
    FOUR_C_THROW(
        "Found more than two intersections for artery element {} and 2D/3D element {}. "
        "This should not happen.",
        artery_ele_gid(), homogenized_ele_gid());
  }

  // safety checks
  if (is_active_)
  {
    if (artery_segment_start_ > artery_segment_end_)
    {
      FOUR_C_THROW(
          "Something went terribly wrong for artery element {} and 2D/3D element {}: "
          "artery_start is bigger than artery_end. This should not happen.",
          artery_ele_gid(), homogenized_ele_gid());
    }
    if (fabs(artery_segment_start_ - artery_segment_end_) < 1.0e-9)
    {
      FOUR_C_THROW(
          "Something went terribly wrong for artery element {} and 2D/3D element {}."
          "Extremely small integration segments were detected. This should not happen.",
          artery_ele_gid(), homogenized_ele_gid());
    }
  }
}
/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
std::vector<double> PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::get_all_intersections()
{
  std::vector<double> intersections;

  std::vector coords_homogenized(num_dim_, 0.0);
  double coords_artery = 0.0;
  bool projection_valid = true;

  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      for (unsigned int j = 0; j < num_dim_; j++)
      {
        // project edge at coords_homogenized = 1.0
        intersect_with_homogenized_element(
            coords_homogenized, coords_artery, j, 1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, coords_artery))
          intersections.push_back(coords_artery);

        // project edge at coords_homogenized = -1.0
        intersect_with_homogenized_element(
            coords_homogenized, coords_artery, j, -1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, coords_artery))
          intersections.push_back(coords_artery);
      }
      break;
    }
    case Core::FE::CellType::hex8:
    {
      for (unsigned int j = 0; j < num_dim_; j++)
      {
        // project surface at coords_homogenized = 1.0
        intersect_with_homogenized_element(
            coords_homogenized, coords_artery, j, 1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, coords_artery))
          intersections.push_back(coords_artery);

        // project surface at coords_homogenized = -1.0
        intersect_with_homogenized_element(
            coords_homogenized, coords_artery, j, -1.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, coords_artery))
          intersections.push_back(coords_artery);
      }
      break;
    }
    case Core::FE::CellType::tet4:
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        // project surface at coords_homogenized = 0.0
        intersect_with_homogenized_element(
            coords_homogenized, coords_artery, j, 0.0, projection_valid);
        if (projection_valid && projection_not_yet_found(intersections, coords_artery))
          intersections.push_back(coords_artery);
      }
      // project fourth surface of tetrahedron at coords_homogenized =  1.0
      intersect_with_homogenized_element(
          coords_homogenized, coords_artery, 3, 0.0, projection_valid);
      if (projection_valid && projection_not_yet_found(intersections, coords_artery))
        intersections.push_back(coords_artery);
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }

  std::ranges::sort(intersections.begin(), intersections.end());

  return intersections;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
bool PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::projection_not_yet_found(const std::vector<double>& intersections,
    const double& eta)
{
  for (const double intersection : intersections)
  {
    if (fabs(intersection - eta) < 1.0e-9)
    {
      if (projection_output) std::cout << "Duplicate intersection found. \n";
      return false;
    }
  }
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::intersect_with_homogenized_element(std::vector<double>&
                                                                       coords_homogenized,
    double& coords_artery, const int& fixed_coord_idx, const double& fixed_at,
    bool& projection_valid)
{
  projection_valid = true;
  bool parallel = false;

  // Initialize limit for parameter values (interval [-limit, limit])
  constexpr double limit = 1.0 + 1.0e-9;

  // reset iteration variables
  coords_artery = 0.0;
  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fixed_coord_idx == 0)  // xi_1 fixed
      {
        coords_homogenized[0] = fixed_at;
        coords_homogenized[1] = 0.0;
      }
      else if (fixed_coord_idx == 1)  // xi_2 fixed
      {
        coords_homogenized[0] = 0.0;
        coords_homogenized[1] = fixed_at;
      }
      else
      {
        FOUR_C_THROW("Wrong input for fixed_coord_idx.");
      }
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fixed_coord_idx == 0)  // xi_1 fixed
      {
        coords_homogenized[0] = fixed_at;
        coords_homogenized[1] = 0.0;
        coords_homogenized[2] = 0.0;
      }
      else if (fixed_coord_idx == 1)  // xi_2 fixed
      {
        coords_homogenized[0] = 0.0;
        coords_homogenized[1] = fixed_at;
        coords_homogenized[2] = 0.0;
      }
      else if (fixed_coord_idx == 2)  // xi_3 fixed
      {
        coords_homogenized[0] = 0.0;
        coords_homogenized[1] = 0.0;
        coords_homogenized[2] = fixed_at;
      }
      else
      {
        FOUR_C_THROW("Wrong input for fixed_coord_idx.");
      }
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (fixed_coord_idx == 0)  // xi_1 fixed
      {
        coords_homogenized[0] = fixed_at;
        coords_homogenized[1] = 0.25;
        coords_homogenized[2] = 0.25;
      }
      else if (fixed_coord_idx == 1)  // xi_2 fixed
      {
        coords_homogenized[0] = 0.25;
        coords_homogenized[1] = fixed_at;
        coords_homogenized[2] = 0.25;
      }
      else if (fixed_coord_idx == 2)  // xi_3 fixed
      {
        coords_homogenized[0] = 0.25;
        coords_homogenized[1] = 0.25;
        coords_homogenized[2] = fixed_at;
      }
      else if (fixed_coord_idx == 3)  // fourth surface xi_1 + xi_2 + xi_3 = 1 fixed
      {
        coords_homogenized[0] = 0.25;
        coords_homogenized[1] = 0.25;
        coords_homogenized[2] = 0.5;
      }
      else
      {
        FOUR_C_THROW(
            "For TET elements, only fixed_coord_idx = 0, fixed_coord_idx = 1, fixed_coord_idx = 2 "
            "or fixed_coord_idx = 3 are possible.");
      }
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }

  if (projection_output)
  {
    std::cout << "Projection output:" << '\n';
    std::cout << "Start parameters coords_artery: " << coords_artery;
    switch (homogenized_element_->shape())
    {
      case Core::FE::CellType::quad4:
      {
        // 2D case
        std::cout << ", coords_homogenized 1: " << coords_homogenized[0]
                  << ", 2: " << coords_homogenized[1] << '\n';
        break;
      }
        // 3D case
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::tet4:
      {
        std::cout << ", coords_homogenized 1: " << coords_homogenized[0]
                  << ", 2: " << coords_homogenized[1] << ", 3: " << coords_homogenized[2] << '\n';
        break;
      }
      default:
        FOUR_C_THROW(
            "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
    }
  }

  // Initialize distance function and Jacobian for Newton iteration
  Core::LinAlg::Matrix<num_dim_, 1> distance;
  Core::LinAlg::Matrix<num_dim_, num_dim_> jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_> inverse_jacobian_matrix;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_> shape_functions_artery_deriv;

  Core::LinAlg::Matrix<1, num_nodes_homogenized_> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> shape_functions_homogenized_deriv;

  // Coords and derivatives of for 1D and 2D/3D element in reference configuration
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_ref;
  Core::LinAlg::Matrix<num_dim_, 1> coords_artery_deriv_ref;

  Core::LinAlg::Matrix<num_dim_, 1> coords_homogenized_ref;
  Core::LinAlg::Matrix<num_dim_, num_dim_> coords_homogenized_deriv_ref;

  // Initial scalar residual (L2-norm of distance)
  double residual;

  // Local newton iteration
  int iter;
  double first_residual = 1.0e-4;  // used for convergence check

  for (iter = 0; iter < projection_max_iter; iter++)
  {
    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_artery_shape_functions<double>(
        shape_functions_artery, shape_functions_artery_deriv, coords_artery);
    get_homogenized_shape_functions<double>(
        shape_functions_homogenized, shape_functions_homogenized_deriv, coords_homogenized);

    // Update coordinates and derivatives for 1D and 2D/3D element
    compute_artery_coords_and_derivs_ref<double>(coords_artery_ref, coords_artery_deriv_ref,
        shape_functions_artery, shape_functions_artery_deriv);
    compute_homogenized_coords_and_derivs_ref<double>(coords_homogenized_ref,
        coords_homogenized_deriv_ref, shape_functions_homogenized,
        shape_functions_homogenized_deriv);

    // Evaluate distance in reference configuration
    distance.clear();
    for (unsigned int i = 0; i < num_dim_; i++)
      distance(i) = coords_homogenized_ref(i) - coords_artery_ref(i);

    // Compute scalar residuum
    residual = 0.0;
    for (unsigned int i = 0; i < num_dim_; i++) residual += distance(i) * distance(i);
    residual = sqrt(residual);
    if (iter == 0) first_residual = std::max(first_residual, residual);

    jacobian_matrix.clear();

    if (fixed_coord_idx == 0)  // xi_1 fixed: we need x_{,xi_2} (and x_{,xi_3} in case of 3D)
    {
      for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
        for (unsigned int j_dim = 1; j_dim < num_dim_; j_dim++)
          jacobian_matrix(i_dim, j_dim - 1) = coords_homogenized_deriv_ref(i_dim, j_dim);
    }
    else if (fixed_coord_idx == 1)  // xi_2 fixed: we need x_{,xi_1} (and x_{,xi_3} in case of 3D)
    {
      switch (homogenized_element_->shape())
      {
        case Core::FE::CellType::quad4:
        {
          for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
            jacobian_matrix(j_dim, 0) = coords_homogenized_deriv_ref(j_dim, 0);
          break;
        }
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        {
          for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
          {
            jacobian_matrix(j_dim, 0) = coords_homogenized_deriv_ref(j_dim, 0);
            jacobian_matrix(j_dim, 1) = coords_homogenized_deriv_ref(j_dim, 2);
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
      }
    }
    else if (fixed_coord_idx == 2)  // xi_3 fixed: we need x_{,xi_1} (and x_{,xi_2} in case of 3D)
    {
      for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
        for (unsigned int j_dim = 0; j_dim < num_dim_ - 1; j_dim++)
          jacobian_matrix(i_dim, j_dim) = coords_homogenized_deriv_ref(i_dim, j_dim);
    }
    else if (fixed_coord_idx == 3)  // xi_3 fixed at xi_3 = 1.0 - xi_1 - xi_2
    {
      for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
      {
        // df/dxi1 = df/dxi1 - df/dxi3
        jacobian_matrix(i_dim, 0) =
            coords_homogenized_deriv_ref(i_dim, 0) - coords_homogenized_deriv_ref(i_dim, 2);
        // df/dxi_2 = df/dxi2 - df/dxi3
        jacobian_matrix(i_dim, 1) =
            coords_homogenized_deriv_ref(i_dim, 1) - coords_homogenized_deriv_ref(i_dim, 2);
      }
    }
    else
    {
      FOUR_C_THROW(
          "For TET elements, only fixed_coord_idx = 0, fixed_coord_idx = 1, fixed_coord_idx = 2 "
          "or fixed_coord_idx = 3 are possible.");
    }

    // fill Jacobian matrix
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
      jacobian_matrix(i_dim, num_dim_ - 1) = -coords_artery_deriv_ref(i_dim);

    double jacobian_determinant = jacobian_matrix.determinant();

    // If det_J = 0, we assume that the artery and the surface edge are parallel. This projection
    // is not needed because the contact interval can also be identified by other projections.
    parallel = fabs(jacobian_determinant) < tol_collinear * first_residual;
    if (!parallel) jacobian_determinant = jacobian_matrix.invert();

    // Check if the local Newton iteration has converged
    if (residual < tol_projection * first_residual && !parallel)
    {
      if (projection_output)
      {
        std::cout << "Local Newton iteration converged after " << iter << " iterations \n";
        std::cout << "Found point at coords_homogenized 1: " << coords_homogenized[0]
                  << ", 2: " << coords_homogenized[1];
        if (num_dim_ == 3) std::cout << ", 3: " << coords_homogenized[2];
        std::cout << ", coords_artery: " << coords_artery << " with residual: " << residual << '\n';
        std::cout << "In reference configuration: \n";
        std::cout << "Coords_artery: " << coords_artery_ref
                  << ",\n Coords_homogenized: " << coords_homogenized_ref << '\n';
      }
      // Local Newton iteration converged
      break;
    }
    if (projection_output && iter > 0)
    {
      std::cout << "New point at coords_homogenized 1: " << coords_homogenized[0]
                << ", 2: " << coords_homogenized[0];
      if (num_dim_ == 3) std::cout << ", 3: " << coords_homogenized[2];
      std::cout << ", coords_artery: " << coords_artery << " with residual: " << residual << '\n';
    }

    // Singular Jacobian matrix
    if (parallel)
    {
      // Sort out
      if (projection_output)
      {
        std::cout << "Elements are collinear: det_J = " << jacobian_determinant << '\n';
      }
      break;
    }

    if (fixed_coord_idx == 0)  // xi_1 fixed: we have to update xi_2 (and xi_3 in case of 3D)
    {
      for (unsigned int i_dim = 1; i_dim < num_dim_; i_dim++)
        for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
          coords_homogenized[i_dim] += -jacobian_matrix(i_dim - 1, j_dim) * distance(j_dim);
    }
    else if (fixed_coord_idx == 1)  // xi_2 fixed: we have to update xi_1 (and xi_3 in case of 3D)
    {
      switch (homogenized_element_->shape())
      {
        case Core::FE::CellType::quad4:
        {
          for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
            coords_homogenized[0] += -jacobian_matrix(0, j_dim) * distance(j_dim);
          break;
        }
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        {
          for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
          {
            coords_homogenized[0] += -jacobian_matrix(0, j_dim) * distance(j_dim);
            coords_homogenized[2] += -jacobian_matrix(1, j_dim) * distance(j_dim);
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
      }
    }
    else if (fixed_coord_idx == 2)  // xi_3 fixed: we have to update xi_1 (and xi_2 in case of 3D)
    {
      for (unsigned int i_dim = 0; i_dim < num_dim_ - 1; i_dim++)
        for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
          coords_homogenized[i_dim] += -jacobian_matrix(i_dim, j_dim) * distance(j_dim);
    }
    else if (fixed_coord_idx == 3)  // xi_3 fixed at xi_3 = 1.0 - xi_1 - xi_2
    {
      for (unsigned int i_dim = 0; i_dim < num_dim_ - 1; i_dim++)
        for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
          coords_homogenized[i_dim] += -jacobian_matrix(i_dim, j_dim) * distance(j_dim);
      coords_homogenized[2] = 1.0 - coords_homogenized[0] - coords_homogenized[1];
    }
    else
    {
      FOUR_C_THROW(
          "For TET elements, only fixed_coord_idx = 0, fixed_coord_idx = 1, fixed_coord_idx = 2 "
          "or fixed_coord_idx = 3 are possible.");
    }

    // update also coords_artery
    for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
      coords_artery += -jacobian_matrix(num_dim_ - 1, j_dim) * distance(j_dim);
  }

  // Local Newton iteration unconverged
  if (residual > tol_projection * first_residual || parallel)
  {
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++) coords_homogenized[i_dim] = 1e+12;
    coords_artery = 1e+12;

    if (projection_output)
      std::cout << "Local Newton iteration unconverged after " << iter + 1 << " iterations. \n";
  }

  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fabs(coords_homogenized[0]) > limit || fabs(coords_homogenized[1]) > limit ||
          fabs(coords_artery) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fabs(coords_homogenized[0]) > limit || fabs(coords_homogenized[1]) > limit ||
          fabs(coords_homogenized[2]) > limit || fabs(coords_artery) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (coords_homogenized[0] < -1.0e-9 || coords_homogenized[1] < -1.0e-9 ||
          coords_homogenized[2] < -1.0e-9 ||
          coords_homogenized[0] + coords_homogenized[1] + coords_homogenized[2] > limit ||
          fabs(coords_artery) > limit)
        projection_valid = false;
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }

  if (projection_output)
  {
    if (projection_valid)
      std::cout << "Projection allowed. \n";
    else
      std::cout << "Projection not allowed. \n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
template <typename T>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::projection(Core::LinAlg::Matrix<num_dim_, 1, T>& coords_artery_ref,
    std::vector<T>& coords_homogenized, bool& projection_valid)
{
  projection_valid = true;
  bool parallel = false;

  // Initialize limit for parameter values (interval [-limit, limit])
  constexpr double limit = 1.0 + 1.0e-9;

  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      coords_homogenized[0] = 0.0;
      coords_homogenized[1] = 0.0;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      coords_homogenized[0] = 0.0;
      coords_homogenized[1] = 0.0;
      coords_homogenized[2] = 0.0;
      break;
    }
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::tet10:
    {
      coords_homogenized[0] = 0.25;
      coords_homogenized[1] = 0.25;
      coords_homogenized[2] = 0.25;
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }

  if constexpr (projection_output)
  {
    std::cout << "Gauss point projection output:" << '\n';
    std::cout << "Start parameters ";
    switch (homogenized_element_->shape())
    {
      case Core::FE::CellType::quad4:
      {
        std::cout << "coords_homogenized 1: " << coords_homogenized[0]
                  << ", 2: " << coords_homogenized[1] << '\n';
        break;
      }
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        std::cout << "coords_homogenized 1: " << coords_homogenized[0]
                  << ", 2: " << coords_homogenized[1] << ", 3: " << coords_homogenized[2] << '\n';
        break;
      }
      default:
        FOUR_C_THROW(
            "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
    }
  }

  // Initialize distance function and Jacobian for Newton iteration
  Core::LinAlg::Matrix<num_dim_, 1, T> distance;
  Core::LinAlg::Matrix<num_dim_, num_dim_, T> jacobian_matrix;
  Core::LinAlg::Matrix<num_dim_, num_dim_, T> inverse_jacobian_matrix;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, num_nodes_artery_, T> shape_functions_artery;
  Core::LinAlg::Matrix<1, num_nodes_artery_, T> shape_functions_artery_deriv;

  Core::LinAlg::Matrix<1, num_nodes_homogenized_, T> shape_functions_homogenized;
  Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, T> shape_functions_homogenized_deriv;

  Core::LinAlg::Matrix<num_dim_, 1, T> coords_homogenized_ref;
  Core::LinAlg::Matrix<num_dim_, num_dim_, T> coords_homogenized_deriv_ref;

  T residual;

  // Local newton iteration
  int iter;
  double first_residual = 1.0e-4;  // used for convergence check

  for (iter = 0; iter < projection_max_iter; iter++)
  {
    // Update shape functions and their derivatives for 1D and 2D/3D element
    get_homogenized_shape_functions<T>(
        shape_functions_homogenized, shape_functions_homogenized_deriv, coords_homogenized);

    // Update coordinates and derivatives for 1D and 2D/3D element in reference configuration
    compute_homogenized_coords_and_derivs_ref<T>(coords_homogenized_ref,
        coords_homogenized_deriv_ref, shape_functions_homogenized,
        shape_functions_homogenized_deriv);

    // Evaluate distance in reference configuration
    distance.clear();
    for (unsigned int i = 0; i < num_dim_; i++)
      distance(i) = coords_homogenized_ref(i) - coords_artery_ref(i);

    residual = Core::FADUtils::vector_norm(distance);
    if (iter == 0)
      first_residual = std::max(first_residual, Core::FADUtils::cast_to_double(residual));

    // Reset matrices
    for (unsigned int i = 0; i < num_dim_; i++)
      for (unsigned int j = 0; j < num_dim_; j++)
        jacobian_matrix(i, j) = coords_homogenized_deriv_ref(i, j);

    const double jacobian_determinant =
        Core::FADUtils::cast_to_double<T, num_dim_, num_dim_>(jacobian_matrix).determinant();

    // If det_J = 0, we assume that the artery element and the surface edge are parallel. This
    // projection is not needed because the contact interval can also be identified by two
    // contact interval borders found with the GetContactLines method.
    parallel = fabs(jacobian_determinant) < tol_collinear * first_residual;
    if (!parallel) jacobian_matrix.invert();

    // Check if the local Newton iteration has converged:
    // If the start point fulfills the orthogonality conditions (residual < newton_projection_tol *
    // first_residual), we also check if the artery element and the surface edge are parallel.
    // This is done by calculating det_J before checking if the local Newton iteration has
    // converged by fulfilling the condition residual < newton_projection_tol * first_residual.
    if (residual < tol_projection * first_residual && !parallel)
    {
      if (projection_output)
      {
        std::cout << "Local Newton iteration converged after " << iter << " iterations. \n";
        std::cout << "Found point at ";
        switch (homogenized_element_->shape())
        {
          case Core::FE::CellType::quad4:
          {
            std::cout << "coords_homogenized 1: " << coords_homogenized[0]
                      << ", 2: " << coords_homogenized[1] << '\n';
            break;
          }
          case Core::FE::CellType::hex8:
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::tet10:
          {
            std::cout << "coords_homogenized 1: " << coords_homogenized[0]
                      << ", 2: " << coords_homogenized[1] << ", 3: " << coords_homogenized[2]
                      << '\n';
            break;
          }
          default:
            FOUR_C_THROW("Only quad4, hex8, tet4 and tet10 are valid so far for second element");
        }
        std::cout << "with residual: " << residual << '\n';
        std::cout << "In reference configuration: \n";
        std::cout << "Coords_artery: " << coords_artery_ref
                  << ",\n Coords_homogenized: " << coords_homogenized_ref << '\n';
      }
      // Local Newton iteration converged
      break;
    }

    if (projection_output && iter > 0)
    {
      std::cout << "New point at coords_homogenized: ";
      switch (homogenized_element_->shape())
      {
        case Core::FE::CellType::quad4:
        {
          std::cout << "1: " << coords_homogenized[0] << ", 2: " << coords_homogenized[1] << '\n';
          break;
        }
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::tet4:
        case Core::FE::CellType::tet10:
        {
          std::cout << "1: " << coords_homogenized[0] << ", 2: " << coords_homogenized[1]
                    << ", 3: " << coords_homogenized[2] << '\n';
          break;
        }
        default:
          FOUR_C_THROW(
              "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
      }
      std::cout << " with residual: " << residual << '\n';
    }

    // Singular Jacobian matrix
    if (parallel)
    {
      // Sort out
      if (projection_output)
      {
        std::cout << "Elements are collinear: det_J = " << jacobian_determinant << '\n';
      }
      break;
    }

    // Regular Jacobian matrix (inversion possible)
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
      for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
        coords_homogenized[i_dim] += -jacobian_matrix(i_dim, j_dim) * distance(j_dim);

    // xi1 += -J(0, 0) * f(0) - J(0, 1) * f(1) - J(0, 2) * f(2);
    // xi2 += -J(1, 0) * f(0) - J(1, 1) * f(1) - J(1, 2) * f(2);
    // xi3 += -J(2, 0) * f(0) - J(2, 1) * f(1) - J(2, 2) * f(2);
  }

  // Local Newton iteration unconverged
  if (residual > tol_projection * first_residual || parallel)
  {
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++) coords_homogenized[i_dim] = 1e+12;

    if (projection_output)
      std::cout << "Local Newton iteration unconverged after " << iter + 1 << " iterations. \n";
  }

  // check if xi lies inside the element
  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      if (fabs(coords_homogenized[0]) > limit || fabs(coords_homogenized[1]) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::hex8:
    {
      if (fabs(coords_homogenized[0]) > limit || fabs(coords_homogenized[1]) > limit ||
          fabs(coords_homogenized[2]) > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (coords_homogenized[0] < -1.0e-9 || coords_homogenized[1] < -1.0e-9 ||
          coords_homogenized[2] < -1.0e-9 ||
          coords_homogenized[0] + coords_homogenized[1] + coords_homogenized[2] > limit)
        projection_valid = false;
      break;
    }
    case Core::FE::CellType::tet10:  // TODO: Is this correct?
    {
      if (coords_homogenized[0] < -1.0e-9 || coords_homogenized[1] < -1.0e-9 ||
          coords_homogenized[2] < -1.0e-9 ||
          coords_homogenized[0] + coords_homogenized[1] + coords_homogenized[2] > limit)
        projection_valid = false;
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }

  if (projection_output)
  {
    if (projection_valid)
      std::cout << "Projection allowed. \n";
    else
      std::cout << "Projection not allowed \n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
template <typename T>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::get_artery_shape_functions(Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function,
    Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function_deriv, const T& coordinate)
{
  // Clear shape functions and derivatives
  shape_function.clear();
  shape_function_deriv.clear();

  // Get discretization type
  const Core::FE::CellType dis_type = artery_element_->shape();

  // Get values and derivatives of shape functions
  Core::FE::shape_function_1d(shape_function, coordinate, dis_type);
  Core::FE::shape_function_1d_deriv1(shape_function_deriv, coordinate, dis_type);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
template <typename T>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::get_homogenized_shape_functions(Core::LinAlg::Matrix<1, num_nodes_homogenized_, T>&
                                              shape_function,
    Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, T>& shape_function_deriv,
    const std::vector<T>& coordinate)
{
  // Clear shape functions and derivatives
  shape_function.clear();
  shape_function_deriv.clear();

  switch (homogenized_element_->shape())
  {
    case Core::FE::CellType::quad4:
    {
      // 2D case
      Core::FE::shape_function_2d(
          shape_function, coordinate[0], coordinate[1], dis_type_homogenized);
      Core::FE::shape_function_2d_deriv1(
          shape_function_deriv, coordinate[0], coordinate[1], dis_type_homogenized);
      break;
    }
    case Core::FE::CellType::hex8:
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::tet10:
    {
      Core::FE::shape_function_3d(
          shape_function, coordinate[0], coordinate[1], coordinate[2], dis_type_homogenized);
      Core::FE::shape_function_3d_deriv1(
          shape_function_deriv, coordinate[0], coordinate[1], coordinate[2], dis_type_homogenized);
      break;
    }
    default:
      FOUR_C_THROW(
          "Only quad4, hex8 and tet4 are valid element types for the homogenized element.");
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
template <typename T>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::compute_artery_coords_and_derivs_ref(Core::LinAlg::Matrix<num_dim_,
                                                                         1, T>& coordinates_ref,
    Core::LinAlg::Matrix<num_dim_, 1, T>& coordinates_deriv_ref,
    const Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function,
    const Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function_deriv)
{
  coordinates_ref.clear();
  coordinates_deriv_ref.clear();

  for (unsigned int j = 0; j < num_nodes_artery_; j++)
  {
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
    {
      coordinates_ref(i_dim) +=
          shape_function(j) * nodal_coords_artery_ele_ref_(num_dim_ * j + i_dim);
      coordinates_deriv_ref(i_dim) +=
          shape_function_deriv(j) * nodal_coords_artery_ele_ref_(num_dim_ * j + i_dim);
    }
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
template <typename T>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::compute_homogenized_coords_and_derivs_ref(Core::LinAlg::Matrix<num_dim_, 1, T>&
                                                        coordinates_ref,
    Core::LinAlg::Matrix<num_dim_, num_dim_, T>& coordinates_deriv_ref,
    const Core::LinAlg::Matrix<1, num_nodes_homogenized_, T>& shape_function,
    const Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, T>& shape_function_deriv)
{
  coordinates_ref.clear();
  coordinates_deriv_ref.clear();

  for (unsigned int j = 0; j < num_nodes_homogenized_; j++)
  {
    for (unsigned int i_dim = 0; i_dim < num_dim_; i_dim++)
    {
      coordinates_ref(i_dim) += shape_function(j) * nodal_coords_homogenized_ele_ref_(i_dim, j);
      for (unsigned int j_dim = 0; j_dim < num_dim_; j_dim++)
      {
        coordinates_deriv_ref(i_dim, j_dim) +=
            shape_function_deriv(j_dim, j) * nodal_coords_homogenized_ele_ref_(i_dim, j);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized,
    dim>::fill_function_vector(std::vector<const Core::Utils::FunctionOfAnything*>& function_vector,
    const std::vector<int>& function_id_vector, const std::vector<int>& scale_vector)
{
  for (unsigned int i = 0; i < function_id_vector.size(); i++)
  {
    if (function_id_vector[i] >= 0 && abs(scale_vector[i]) > 0)
    {
      function_vector.at(i) =
          &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
              function_id_vector[i]);
      function_coupling_active_ = true;
    }
    else
    {
      function_vector.at(i) = nullptr;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::initialize_function(const Core::Utils::FunctionOfAnything& funct)
{
  if (funct.number_components() != 1)
    FOUR_C_THROW("Expected only one component for coupling function!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::initialize_function_names()
{
  pressure_names_.resize(num_fluid_phases_);
  saturation_names_.resize(num_fluid_phases_);
  volfrac_names_.resize(num_volfracs_);
  volfrac_pressure_names_.resize(num_volfracs_);
  scalar_names_.resize(num_scalars_homogenized_);
  artery_scalar_names_.resize(num_scalars_artery_);

  for (int k = 0; k < num_scalars_homogenized_; k++)
  {
    // add scalar names
    scalar_names_[k] = "phi" + std::to_string(k + 1);
  }

  for (int k = 0; k < num_scalars_artery_; k++)
  {
    // add artery-scalar names
    artery_scalar_names_[k] = "phi_art" + std::to_string(k + 1);
  }

  for (int k = 0; k < num_fluid_phases_; k++)
  {
    // add pressure names
    pressure_names_[k] = "p" + std::to_string(k + 1);

    // add saturation names
    saturation_names_[k] = "S" + std::to_string(k + 1);
  }

  // add additional volume fractions
  for (int k = 0; k < num_volfracs_; k++)
  {
    // add volume fraction names
    volfrac_names_[k] = "VF" + std::to_string(k + 1);

    // add volume fraction pressure names
    volfrac_pressure_names_[k] = "VFP" + std::to_string(k + 1);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
void PoroPressureBased::PorofluidElastScatraArteryCouplingPair<dis_type_artery,
    dis_type_homogenized, dim>::initialize_assemble_into_homogenized_dof_vector()
{
  homogenized_dofs_to_assemble_functions_into_.resize(num_dof_homogenized_);

  for (int i_homo = 0; i_homo < num_dof_homogenized_; i_homo++)
  {
    const std::vector current_dof = {i_homo};
    homogenized_dofs_to_assemble_functions_into_[i_homo] = current_dof;
  }

  switch (coupling_type_)
  {
    case CouplingType::porofluid:
    {
      // Special case for phases [0, ..., num_fluid_phases - 2]:
      // those have to assemble into the summed up fluid phase (see also porofluid_evaluator).
      for (int cur_phase = 0; cur_phase < num_fluid_phases_ - 1; cur_phase++)
        homogenized_dofs_to_assemble_functions_into_[cur_phase].push_back(num_fluid_phases_ - 1);
      break;
    }
    case CouplingType::scatra:
    {
      // do nothing
      break;
    }
    default:
      FOUR_C_THROW("Unknown coupling type.");
      break;
  }
}


// explicit template instantiations
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::quad4, 1>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::hex8, 1>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet4, 1>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet10, 1>;

template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::quad4, 2>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::hex8, 2>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet4, 2>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet10, 2>;

template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::quad4, 3>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::hex8, 3>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet4, 3>;
template class PoroPressureBased::PorofluidElastScatraArteryCouplingPair<Core::FE::CellType::line2,
    Core::FE::CellType::tet10, 3>;

FOUR_C_NAMESPACE_CLOSE
