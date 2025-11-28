// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_springdashpot.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_contact_interface.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element_dof_matrix.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_scatra.hpp"
#include "4C_shell7p_line.hpp"
#include "4C_truss3.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <concepts>
#include <tuple>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using supported_celltypes =
      Core::FE::CelltypeSequence<Core::FE::CellType::quad4, Core::FE::CellType::quad8,
          Core::FE::CellType::quad9, Core::FE::CellType::tri3, Core::FE::CellType::tri6>;

  using supported_linetypes =
      Core::FE::CelltypeSequence<Core::FE::CellType::line2, Core::FE::CellType::line3>;

  double scale_by_optional_function(double baseline_value, int num_func, double total_time)
  {
    return num_func ? Global::Problem::instance()
                              ->function_by_id<Core::Utils::FunctionOfTime>(num_func)
                              .evaluate(total_time) *
                          baseline_value
                    : baseline_value;
  }

  template <typename T>
  concept SpringDashpotEvaluable = requires(T t, double delta_displacement, double velocity) {
    { t(delta_displacement, velocity) } -> std::convertible_to<std::tuple<double, double, double>>;
  };

  template <SpringDashpotEvaluable SpringDashpot, Core::FE::CellType celltype,
      unsigned int num_dof_per_node, unsigned int dim>
  void add_robin_spring_dashpot_contribution_xyz(const Core::Elements::Element& surface_element,
      const Core::Elements::ElementNodes<celltype, dim> element_nodes,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>& displacements,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>&
          displacement_offset,
      const std::vector<double>& constant_offset,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>& velocity,
      const std::vector<SpringDashpot>& spring_dashpot_evaluator, const double time_factor,
      const Core::FE::GaussIntegration& gauss_integration, const std::vector<bool>& onoff,
      std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype), 1>>&
          residual_vector,
      std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype),
          num_dof_per_node * Core::FE::num_nodes(celltype)>>& stiffness_matrix)
  {
    Core::Elements::for_each_gauss_point<celltype>(element_nodes, gauss_integration,
        [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
            const Core::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const Core::Elements::JacobianMapping<celltype, dim>& jacobian_mapping,
            double integration_factor, int gp)
        {
          Core::LinAlg::Matrix<num_dof_per_node, 1> displacement_at_gp;
          displacement_at_gp.multiply_nn(displacements, shape_functions.values);

          Core::LinAlg::Matrix<num_dof_per_node, 1> offset_at_gp;
          offset_at_gp.multiply_nn(displacement_offset, shape_functions.values);

          Core::LinAlg::Matrix<num_dof_per_node, 1> velocity_at_gp;
          velocity_at_gp.multiply_nn(velocity, shape_functions.values);

          for (unsigned axis_direction = 0; axis_direction < dim; ++axis_direction)
          {
            if (!onoff[axis_direction]) continue;

            FOUR_C_ASSERT(axis_direction < spring_dashpot_evaluator.size(),
                "Not enough spring-dashpot evaluators given!");

            const auto& [force, force_deriv_disp, force_deriv_vel] =
                spring_dashpot_evaluator[axis_direction](displacement_at_gp(axis_direction) +
                                                             offset_at_gp(axis_direction) -
                                                             constant_offset[axis_direction],
                    velocity_at_gp(axis_direction));

            // evaluate residual force
            if (residual_vector)
            {
              for (int node = 0; node < Core::FE::num_nodes(celltype); ++node)
                residual_vector.value()(node * num_dof_per_node + axis_direction) +=
                    shape_functions.values(node) * force * integration_factor;
            }

            // evaluate stiffness matrix
            if (stiffness_matrix)
            {
              for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
              {
                for (int j = 0; j < Core::FE::num_nodes(celltype); ++j)
                {
                  (*stiffness_matrix)(i* num_dof_per_node + axis_direction,
                      j * num_dof_per_node + axis_direction) +=
                      shape_functions.values(i) * shape_functions.values(j) *
                      (force_deriv_disp + force_deriv_vel * time_factor) * integration_factor;
                }
              }
            }
          }
        });
  }

  template <Core::FE::CellType celltype, unsigned int dim>
  Core::LinAlg::Matrix<dim, 1> get_normal(
      const Core::Elements::JacobianMapping<celltype, dim>& jacobian_mapping)
    requires(Core::FE::dim<celltype> == 2 && dim == 3)
  {
    Core::LinAlg::Matrix<dim, 1> normal;
    normal(0) = jacobian_mapping(0, 1) * jacobian_mapping(1, 2) -
                jacobian_mapping(0, 2) * jacobian_mapping(1, 1);
    normal(1) = jacobian_mapping(0, 2) * jacobian_mapping(1, 0) -
                jacobian_mapping(0, 0) * jacobian_mapping(1, 2);
    normal(2) = jacobian_mapping(0, 0) * jacobian_mapping(1, 1) -
                jacobian_mapping(0, 1) * jacobian_mapping(1, 0);

    normal.scale(1.0 / normal.norm2());

    return normal;
  }

  template <SpringDashpotEvaluable SpringDashpot, Core::FE::CellType celltype,
      unsigned int num_dof_per_node, unsigned int dim>
  void add_robin_spring_dashpot_contribution_ref_normal(const Core::Elements::Element& element,
      const Core::Elements::ElementNodes<celltype, dim> element_nodes,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>& displacements,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>&
          displacement_offset,
      double constant_offset,
      const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>& velocity,
      const SpringDashpot& spring_dashpot_evaluator, const double time_factor,
      const Core::FE::GaussIntegration& gauss_integration,
      std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype), 1>>&
          residual_vector,
      std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype),
          num_dof_per_node * Core::FE::num_nodes(celltype)>>& stiffness_matrix)
  {
    // compute nodal normals of the element
    Core::Elements::for_each_gauss_point<celltype>(element_nodes, gauss_integration,
        [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
            const Core::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const Core::Elements::JacobianMapping<celltype, dim>& jacobian_mapping,
            double integration_factor, int gp)
        {
          Core::LinAlg::Matrix<num_dof_per_node, 1> displacement_at_gp;
          displacement_at_gp.multiply_nn(displacements, shape_functions.values);

          Core::LinAlg::Matrix<num_dof_per_node, 1> offset_at_gp;
          offset_at_gp.multiply_nn(displacement_offset, shape_functions.values);

          Core::LinAlg::Matrix<num_dof_per_node, 1> velocity_at_gp;
          velocity_at_gp.multiply_nn(velocity, shape_functions.values);

          Core::LinAlg::Matrix<dim, 1> normal = get_normal<celltype>(jacobian_mapping);

          Core::LinAlg::Matrix<num_dof_per_node, 1> normal_at_gp(
              Core::LinAlg::Initialization::zero);

          // For shell7p elements, copy the normal vector twice (for translations of
          // displacements and director vector)
          if (element.element_type() == Discret::Elements::Shell7pType::instance() ||
              element.element_type() == Discret::Elements::Shell7pScatraType::instance())
          {
            std::copy_n(normal.data(), 3, normal_at_gp.data());
            std::copy_n(normal.data(), 3, normal_at_gp.data() + 3);
          }
          else
          {
            std::copy_n(normal.data(), dim, normal_at_gp.data());
          }

          double displacement_ref_normal = displacement_at_gp.dot(normal_at_gp);
          double offset_ref_normal = offset_at_gp.dot(normal_at_gp);
          double velocity_ref_normal = velocity_at_gp.dot(normal_at_gp);

          const auto& [force, force_deriv_disp, force_deriv_vel] = spring_dashpot_evaluator(
              displacement_ref_normal + offset_ref_normal - constant_offset, velocity_ref_normal);

          // evaluate residual force
          if (residual_vector)
          {
            for (unsigned axis_direction = 0; axis_direction < dim; ++axis_direction)
            {
              for (int node = 0; node < Core::FE::num_nodes(celltype); ++node)
              {
                residual_vector.value()(node * num_dof_per_node + axis_direction) +=
                    shape_functions.values(node) * force * normal_at_gp(axis_direction) *
                    integration_factor;
              }
            }
          }

          // evaluate stiffness matrix
          if (stiffness_matrix)
          {
            for (unsigned axis_direction1 = 0; axis_direction1 < dim; ++axis_direction1)
            {
              for (unsigned axis_direction2 = 0; axis_direction2 < dim; ++axis_direction2)
              {
                for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
                {
                  for (int j = 0; j < Core::FE::num_nodes(celltype); ++j)
                  {
                    (*stiffness_matrix)(i* num_dof_per_node + axis_direction1,
                        j * num_dof_per_node + axis_direction2) +=
                        shape_functions.values(i) * shape_functions.values(j) *
                        (force_deriv_disp + force_deriv_vel * time_factor) *
                        normal_at_gp(axis_direction1) * normal_at_gp(axis_direction2) *
                        integration_factor;
                  }
                }
              }
            }
          }
        });
  }

  std::vector<std::function<std::tuple<double, double, double>(double, double)>>
  create_spring_dashpot_evaluators(const std::vector<bool>& onoff_bool,
      const std::vector<double>& dashpotvisc, const std::vector<int>& numfuncvisco,
      const std::vector<int>& numfuncnonlinstiff, const std::vector<double>& springstiff,
      const std::vector<int>& numfuncstiff, double total_time)
  {
    FOUR_C_ASSERT_ALWAYS(onoff_bool.size() <= 3, "Number of dofs should not be larger than 3");

    std::vector<std::function<std::tuple<double, double, double>(double, double)>>
        spring_dashpot_evaluators(onoff_bool.size());

    for (std::size_t i = 0; i < onoff_bool.size(); ++i)
    {
      if (!onoff_bool[i]) continue;

      const double dashpot_viscosity =
          scale_by_optional_function(dashpotvisc[i], numfuncvisco[i], total_time);

      if (numfuncnonlinstiff[i] > 0)
      {
        const auto& nonlinear_spring =
            Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                numfuncnonlinstiff[i]);

        spring_dashpot_evaluators[i] = [=, &nonlinear_spring](double delta_displacement,
                                           double velocity) -> std::tuple<double, double, double>
        {
          std::array<double, 3> displ = {std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
          displ[i] = delta_displacement;

          const double force =
              nonlinear_spring.evaluate(displ, total_time, 0) + dashpot_viscosity * velocity;

          const double force_derivative_wrt_displ =
              nonlinear_spring.evaluate_spatial_derivative(displ, total_time, 0)[i];

          return {force, force_derivative_wrt_displ, dashpot_viscosity};
        };
      }
      else
      {
        const double spring_stiffness =
            scale_by_optional_function(springstiff[i], numfuncstiff[i], total_time);

        spring_dashpot_evaluators[i] = [=](double delta_displacement,
                                           double velocity) -> std::tuple<double, double, double>
        {
          const double force = spring_stiffness * delta_displacement + dashpot_viscosity * velocity;
          return {force, spring_stiffness, dashpot_viscosity};
        };
      }
    }

    return spring_dashpot_evaluators;
  }

  template <unsigned int num_dof_per_node, unsigned int num_dim, typename DirectionT>
  void evaluate_robin_surface_element(const std::shared_ptr<Core::Elements::Element>& element,
      Core::FE::Discretization* actdisc, const std::vector<double>& displacements,
      const std::vector<double>& displacement_offset, const std::vector<double>& velocities,
      const std::vector<double>& scaled_constant_offset,
      const std::vector<std::function<std::tuple<double, double, double>(double, double)>>&
          spring_dashpot_evaluators,
      double time_fac, const std::vector<bool>& onoff_bool, Core::LinAlg::SerialDenseVector& res,
      Core::LinAlg::SerialDenseMatrix& elestiff, bool assvec, bool assmat, DirectionT direction)
  {
    Core::FE::cell_type_switch<supported_celltypes>(element->shape(),
        [&](auto celltype_t)
        {
          constexpr Core::FE::CellType celltype = celltype_t();
          const Core::Elements::ElementNodes<celltype, num_dim> element_nodes =
              Core::Elements::evaluate_element_nodes<celltype, num_dim>(*actdisc, *element);

          std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype), 1>>
              residual_vector;
          if (assvec)
          {
            residual_vector.emplace(res.values(), true);
            residual_vector.value().clear();
          }

          std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype),
              num_dof_per_node * Core::FE::num_nodes(celltype)>>
              stiffness_matrix;
          if (assmat)
          {
            stiffness_matrix.emplace(elestiff.values(), true);
            (*stiffness_matrix).clear();
          }

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>
              ele_displacements =
                  Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(displacements);

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)> nodal_offset =
              Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(displacement_offset);

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>
              velocity_matrix =
                  Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(velocities);

          Core::FE::GaussIntegration gauss_integration =
              Core::FE::create_gauss_integration<celltype>(
                  Discret::Elements::DisTypeToOptGaussRule<celltype>::rule);

          if (direction == DirectionT::xyz)
          {
            FOUR_C_ASSERT_ALWAYS(onoff_bool.size() == 3,
                "Number of given functions must be 3 when using the xyz-direction of the Robin "
                "Spring-Dashpot BC!");

            add_robin_spring_dashpot_contribution_xyz(*element, element_nodes, ele_displacements,
                nodal_offset, scaled_constant_offset, velocity_matrix, spring_dashpot_evaluators,
                time_fac, gauss_integration, onoff_bool, residual_vector, stiffness_matrix);
          }
          else if (direction == DirectionT::refsurfnormal)
          {
            FOUR_C_ASSERT_ALWAYS(onoff_bool.size() == 1,
                "Number of given functions must be 1 when using the reference surface normal "
                "of the Robin Spring-Dashpot BC!");

            add_robin_spring_dashpot_contribution_ref_normal(*element, element_nodes,
                ele_displacements, nodal_offset, scaled_constant_offset[0], velocity_matrix,
                spring_dashpot_evaluators[0], time_fac, gauss_integration, residual_vector,
                stiffness_matrix);
          }
          else
          {
            FOUR_C_THROW("Robin surface direction type {} not implemented!", direction);
          }
        });
  }

  template <unsigned int num_dof_per_node, unsigned int num_dim, typename DirectionT>
  void evaluate_robin_line_element(const std::shared_ptr<Core::Elements::Element>& element,
      Core::FE::Discretization* actdisc, const std::vector<double>& displacements,
      const std::vector<double>& displacement_offset, const std::vector<double>& velocities,
      const std::vector<double>& scaled_constant_offset,
      const std::vector<std::function<std::tuple<double, double, double>(double, double)>>&
          spring_dashpot_evaluators,
      double time_fac, const std::vector<bool>& onoff_bool, Core::LinAlg::SerialDenseVector& res,
      Core::LinAlg::SerialDenseMatrix& elestiff, bool assvec, bool assmat, DirectionT direction)
  {
    Core::FE::cell_type_switch<supported_linetypes>(element->shape(),
        [&](auto celltype_t)
        {
          constexpr Core::FE::CellType celltype = celltype_t();
          const Core::Elements::ElementNodes<celltype, num_dim> element_nodes =
              Core::Elements::evaluate_element_nodes<celltype, num_dim>(*actdisc, *element);

          std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype), 1>>
              residual_vector;
          if (assvec)
          {
            residual_vector.emplace(res.values(), true);
            residual_vector.value().clear();
          }

          std::optional<Core::LinAlg::Matrix<num_dof_per_node * Core::FE::num_nodes(celltype),
              num_dof_per_node * Core::FE::num_nodes(celltype)>>
              stiffness_matrix;
          if (assmat)
          {
            stiffness_matrix.emplace(elestiff.values(), true);
            (*stiffness_matrix).clear();
          }

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>
              ele_displacements =
                  Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(displacements);

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)> nodal_offset =
              Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(displacement_offset);

          const Core::LinAlg::Matrix<num_dof_per_node, Core::FE::num_nodes(celltype)>
              velocity_matrix =
                  Core::FE::get_element_dof_matrix<celltype, num_dof_per_node>(velocities);

          Core::FE::GaussIntegration gauss_integration =
              Core::FE::create_gauss_integration<celltype>(
                  Discret::Elements::DisTypeToOptGaussRule<celltype>::rule);

          if (direction == DirectionT::xyz)
          {
            FOUR_C_ASSERT_ALWAYS(onoff_bool.size() == 3,
                "Number of given functions must be 3 when using the xyz-direction of the Robin "
                "Spring-Dashpot BC!");

            add_robin_spring_dashpot_contribution_xyz(*element, element_nodes, ele_displacements,
                nodal_offset, scaled_constant_offset, velocity_matrix, spring_dashpot_evaluators,
                time_fac, gauss_integration, onoff_bool, residual_vector, stiffness_matrix);
          }
          else
          {
            FOUR_C_THROW(
                "Only xyz-direction is supported for line spring dashpot conditions currently!");
          }
        });
  }
}  // namespace

/*----------------------------------------------------------------------*
 |                                                         pfaller Apr15|
 *----------------------------------------------------------------------*/
Constraints::SpringDashpot::SpringDashpot(
    std::shared_ptr<Core::FE::Discretization> dis, const Core::Conditions::Condition& cond)
    : actdisc_(std::move(dis)),
      spring_(&cond),
      stiff_tens_((spring_->parameters().get<std::vector<double>>("STIFF"))[0]),
      stiff_comp_((spring_->parameters().get<std::vector<double>>("STIFF"))[0]),
      offset_((spring_->parameters().get<std::vector<double>>("DISPLOFFSET"))[0]),
      viscosity_((spring_->parameters().get<std::vector<double>>("VISCO"))[0]),
      coupling_(spring_->parameters().get<std::optional<int>>("COUPLING").value_or(-1)),
      nodes_(spring_->get_nodes()),
      area_(),
      gap0_(),
      gap_(),
      gapdt_(),
      dgap_(),
      normals_(),
      dnormals_(),
      offset_prestr_(),
      offset_prestr_new_(nullptr)
{
  offset_prestr_new_ = std::make_shared<Core::LinAlg::Vector<double>>(*actdisc_->dof_row_map());
  offset_prestr_new_->put_scalar(0.0);

  // set type of this spring
  set_spring_type();

  if (springtype_ != RobinSpringDashpotType::cursurfnormal && coupling_ >= 0)
  {
    FOUR_C_THROW(
        "Coupling of spring dashpot to reference surface only possible for DIRECTION "
        "cursurfnormal.");
  }

  if (springtype_ == RobinSpringDashpotType::cursurfnormal && coupling_ == -1)
    FOUR_C_THROW("Coupling id necessary for DIRECTION cursurfnormal.");

  // safety checks of input
  const auto* springstiff = &spring_->parameters().get<std::vector<double>>("STIFF");
  const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
  const auto* numfuncnonlinstiff = &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");

  for (unsigned i = 0; i < (*numfuncnonlinstiff).size(); ++i)
  {
    if ((*numfuncnonlinstiff)[i] != 0 and ((*springstiff)[i] != 0 or (*numfuncstiff)[i] != 0))
      FOUR_C_THROW("Must not apply nonlinear stiffness and linear stiffness");
  }

  // ToDo: delete rest until return statement!

  // get normal vectors if necessary
  if (springtype_ == RobinSpringDashpotType::cursurfnormal)
  {
    // get geometry
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = spring_->geometry();
    // calculate nodal area
    if (!Core::Communication::my_mpi_rank(actdisc_->get_comm()))
      Core::IO::cout << "Computing area for spring dashpot condition...\n";
    get_area(geom);
    initialize_cur_surf_normal();
  }

  // ToDo: do we really need this??
  // initialize prestressing offset
  initialize_prestr_offset();
}

// NEW version, consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 * Integrate a Surface Robin boundary condition (public)       mhv 08/16|
 * ---------------------------------------------------------------------*/
void Constraints::SpringDashpot::evaluate_robin(std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
    std::shared_ptr<Core::LinAlg::Vector<double>> fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const Core::LinAlg::Vector<double>& velo, Teuchos::ParameterList p)
{
  // reset last Newton step
  springstress_.clear();

  const bool assvec = fint != nullptr;
  const bool assmat = stiff != nullptr;

  FOUR_C_ASSERT(assvec || assmat,
      "You neither want to evaluate the residual vector nor its linearization! Calling this "
      "function has, therefore, no effect!");

  Core::LinAlg::Vector<double> disp_with_ghosted(*actdisc_->dof_col_map(), true);
  Core::LinAlg::Vector<double> velo_with_ghosted(*actdisc_->dof_col_map(), true);
  Core::LinAlg::Vector<double> offset_with_ghosted(*actdisc_->dof_col_map(), true);

  Core::LinAlg::export_to(*disp, disp_with_ghosted);
  Core::LinAlg::export_to(velo, velo_with_ghosted);
  Core::LinAlg::export_to(*offset_prestr_new_, offset_with_ghosted);

  // get values and switches from the condition
  const auto& onoff = spring_->parameters().get<std::vector<int>>("ONOFF");
  const auto& springstiff = spring_->parameters().get<std::vector<double>>("STIFF");
  const auto& numfuncstiff = spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
  const auto& dashpotvisc = spring_->parameters().get<std::vector<double>>("VISCO");
  const auto& numfuncvisco = spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
  const auto& constant_offset = spring_->parameters().get<std::vector<double>>("DISPLOFFSET");
  const auto& numfuncdisploffset =
      spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
  const auto& numfuncnonlinstiff = spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
  const auto& direction = spring_->parameters().get<RobinSpringDashpotType>("DIRECTION");

  // time-integration factor for stiffness contribution of dashpot, d(v_{n+1})/d(d_{n+1})
  const double time_fac = p.get("time_fac", 0.0);
  const double total_time = p.get("total time", 0.0);

  std::vector<bool> onoff_bool(onoff.size(), true);
  std::transform(onoff.begin(), onoff.end(), onoff_bool.begin(), [](int i) { return i == 1; });

  switch (spring_->g_type())
  {
    case Core::Conditions::geometry_type_surface:
    {
      const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = spring_->geometry();

      // Build spring-dashpot evaluator functions once per geometry
      auto spring_dashpot_evaluators = create_spring_dashpot_evaluators(onoff_bool, dashpotvisc,
          numfuncvisco, numfuncnonlinstiff, springstiff, numfuncstiff, total_time);

      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      for (auto& curr : geom)
      {
        // get element location vector and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        curr.second->location_vector(*actdisc_, lm, lmowner, lmstride);

        // Functions that evaluate the spring-dashpot force
        FOUR_C_ASSERT_ALWAYS(onoff.size() <= 3, "Number of dofs should not be larger than 3");

        // extract velocity
        std::vector<double> velocities = Core::FE::extract_values(velo_with_ghosted, lm);

        std::vector<double> displacements = Core::FE::extract_values(disp_with_ghosted, lm);

        std::vector<double> displacement_offset = Core::FE::extract_values(offset_with_ghosted, lm);

        Core::LinAlg::SerialDenseMatrix elestiff;
        Core::LinAlg::SerialDenseVector res;
        res.size(lm.size());
        elestiff.shape(lm.size(), lm.size());

        std::vector<double> scaled_constant_offset(onoff.size());
        for (unsigned axis_direction = 0; axis_direction < onoff.size(); ++axis_direction)
        {
          scaled_constant_offset[axis_direction] = scale_by_optional_function(
              constant_offset[axis_direction], numfuncdisploffset[axis_direction], total_time);
        }

        // check if element is either Shell7p or Shell7pScatra
        bool is_shell7p_element =
            (curr.second->element_type() == Discret::Elements::Shell7pType::instance()) ||
            (curr.second->element_type() == Discret::Elements::Shell7pScatraType::instance());

        if (is_shell7p_element)
        {
          // shell7p elements: 6 dofs per node, 3 spatial dimensions
          const int numdim = 3;
          const int numdf = 6;

          evaluate_robin_surface_element<numdf, numdim>(curr.second, actdisc_.get(), displacements,
              displacement_offset, velocities, scaled_constant_offset, spring_dashpot_evaluators,
              time_fac, onoff_bool, res, elestiff, assvec, assmat, direction);

          if (assvec) Core::LinAlg::assemble(*fint, res, lm, lmowner);
          if (assmat) stiff->assemble(curr.second->id(), lmstride, elestiff, lm, lmowner);

          // save spring stress for postprocessing
          std::vector<double> stress(numdim, 0.0);

          for (int node = 0; node < curr.second->num_node(); ++node)
          {
            for (int dim = 0; dim < numdim; dim++) stress[dim] = res[node * numdf + dim];
            springstress_.insert(
                std::pair<int, std::vector<double>>(curr.second->node_ids()[node], stress));
          }
        }
        else
        {
          // other element types: 3 dofs per node, 3 spatial dimensions
          const int numdim = 3;
          const int numdf = 3;

          evaluate_robin_surface_element<numdf, numdim>(curr.second, actdisc_.get(), displacements,
              displacement_offset, velocities, scaled_constant_offset, spring_dashpot_evaluators,
              time_fac, onoff_bool, res, elestiff, assvec, assmat, direction);

          if (assvec) Core::LinAlg::assemble(*fint, res, lm, lmowner);
          if (assmat) stiff->assemble(curr.second->id(), lmstride, elestiff, lm, lmowner);

          // save spring stress for postprocessing
          std::vector<double> stress(numdim, 0.0);

          for (int node = 0; node < curr.second->num_node(); ++node)
          {
            for (int dim = 0; dim < numdim; dim++) stress[dim] = res[node * numdf + dim];
            springstress_.insert(
                std::pair<int, std::vector<double>>(curr.second->node_ids()[node], stress));
          }
        }
      } /* end of loop over geometry */
      break;
    }
    case Core::Conditions::geometry_type_line:
    {
      const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = spring_->geometry();

      // Build spring-dashpot evaluator functions once per geometry
      auto spring_dashpot_evaluators = create_spring_dashpot_evaluators(onoff_bool, dashpotvisc,
          numfuncvisco, numfuncnonlinstiff, springstiff, numfuncstiff, total_time);

      for (auto& curr : geom)
      {
        // get element location vector and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        curr.second->location_vector(*actdisc_, lm, lmowner, lmstride);

        // extract velocity
        std::vector<double> velocities = Core::FE::extract_values(velo_with_ghosted, lm);
        std::vector<double> displacements = Core::FE::extract_values(disp_with_ghosted, lm);
        std::vector<double> displacement_offset = Core::FE::extract_values(offset_with_ghosted, lm);

        Core::LinAlg::SerialDenseMatrix elestiff;
        Core::LinAlg::SerialDenseVector res;
        res.size(lm.size());
        elestiff.shape(lm.size(), lm.size());

        std::vector<double> scaled_constant_offset(onoff.size());
        for (unsigned axis_direction = 0; axis_direction < onoff.size(); ++axis_direction)
        {
          scaled_constant_offset[axis_direction] = scale_by_optional_function(
              constant_offset[axis_direction], numfuncdisploffset[axis_direction], total_time);
        }

        // detect shell7p line elements (6 DOFs) or treat as generic 3-DOF line
        if (curr.second->element_type() == Discret::Elements::Shell7pLineType::instance())
        {
          // shell7p line: 6 dofs per node
          const int numdim = 3;
          const int numdf = 6;

          evaluate_robin_line_element<numdf, numdim>(curr.second, actdisc_.get(), displacements,
              displacement_offset, velocities, scaled_constant_offset, spring_dashpot_evaluators,
              time_fac, onoff_bool, res, elestiff, assvec, assmat, direction);

          if (assvec) Core::LinAlg::assemble(*fint, res, lm, lmowner);
          if (assmat) stiff->assemble(curr.second->id(), lmstride, elestiff, lm, lmowner);

          // save spring stress for postprocessing
          std::vector<double> stress(numdim, 0.0);

          for (int node = 0; node < curr.second->num_node(); ++node)
          {
            for (int dim = 0; dim < numdim; dim++) stress[dim] = res[node * numdf + dim];
            springstress_.insert(
                std::pair<int, std::vector<double>>(curr.second->node_ids()[node], stress));
          }
        }
        else
        {
          // generic line element: 3 dofs per node
          const int numdim = 3;
          const int numdf = 3;

          evaluate_robin_line_element<numdf, numdim>(curr.second, actdisc_.get(), displacements,
              displacement_offset, velocities, scaled_constant_offset, spring_dashpot_evaluators,
              time_fac, onoff_bool, res, elestiff, assvec, assmat, direction);

          if (assvec) Core::LinAlg::assemble(*fint, res, lm, lmowner);
          if (assmat) stiff->assemble(curr.second->id(), lmstride, elestiff, lm, lmowner);

          // save spring stress for postprocessing
          std::vector<double> stress(numdim, 0.0);

          for (int node = 0; node < curr.second->num_node(); ++node)
          {
            for (int dim = 0; dim < numdim; dim++) stress[dim] = res[node * numdf + dim];
            springstress_.insert(
                std::pair<int, std::vector<double>>(curr.second->node_ids()[node], stress));
          }
        }
      } /* end of loop over geometry */
      break;
    }
    case Core::Conditions::geometry_type_point:
    {
      if (direction == RobinSpringDashpotType::xyz)
      {
        // get all nodes of this condition and check, if it's just one -> get this node
        const auto* nodes_cond = spring_->get_nodes();
        if (nodes_cond->size() != 1)
          FOUR_C_THROW("Point Robin condition must be defined on one node.");
        const int node_gid = nodes_cond->at(0);
        auto* node = actdisc_->g_node(node_gid);

        // get adjacent element of this node and check if it's just one -> get this element and cast
        // it to truss element
        if (node->num_element() != 1) FOUR_C_THROW("Node may only have one element");
        auto* truss_ele =
            dynamic_cast<Discret::Elements::Truss3*>(node->adjacent_elements()[0].user_element());
        if (truss_ele == nullptr)
        {
          FOUR_C_THROW(
              "Currently, only Truss Elements are allowed to evaluate point Robin Condition. Cast "
              "to "
              "Truss Element failed.");
        }

        // dofs of this node
        auto dofs_gid = actdisc_->dof(0, node);

        // get cross section for integration of this element
        const double cross_section = truss_ele->cross_section();

        for (size_t dof = 0; dof < onoff.size(); ++dof)
        {
          const int dof_onoff = onoff[dof];
          if (dof_onoff == 0) continue;

          const int dof_gid = dofs_gid[dof];
          const int dof_lid = actdisc_->dof_row_map()->lid(dof_gid);

          const double dof_disp = disp->local_values_as_span()[dof_lid];
          const double dof_vel = (velo).local_values_as_span()[dof_lid];

          // compute stiffness, viscosity, and initial offset from functions
          const double dof_stiffness =
              (numfuncstiff)[dof] != 0
                  ? springstiff[dof] *
                        Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>(numfuncstiff[dof])
                            .evaluate(total_time)
                  : springstiff[dof];
          const double dof_viscosity =
              numfuncvisco[dof] != 0
                  ? dashpotvisc[dof] *
                        Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>(numfuncvisco[dof])
                            .evaluate(total_time)
                  : dashpotvisc[dof];
          const double dof_disploffset =
              numfuncdisploffset[dof] != 0
                  ? constant_offset[dof] *
                        Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>(numfuncdisploffset[dof])
                            .evaluate(total_time)
                  : constant_offset[dof];

          // displacement related forces and derivatives
          double force_disp = 0.0;
          double force_disp_deriv = 0.0;
          if ((numfuncnonlinstiff)[dof] == 0)
          {
            force_disp = dof_stiffness * (dof_disp - dof_disploffset);
            force_disp_deriv = dof_stiffness;
          }
          else
          {
            std::array<double, 3> displ = {disp->local_values_as_span()[0],
                disp->local_values_as_span()[1], (*disp).local_values_as_span()[2]};
            force_disp = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                 (numfuncnonlinstiff)[dof] - 1)
                             .evaluate(displ, total_time, 0);

            force_disp_deriv = (Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                        (numfuncnonlinstiff)[dof] - 1)
                    .evaluate_spatial_derivative(displ, total_time, 0))[dof];
          }

          // velocity related forces and derivatives
          const double force_vel = dof_viscosity * dof_vel;
          const double force_vel_deriv = dof_viscosity;

          const double force = force_disp + force_vel;
          const double stiffness = force_disp_deriv + force_vel_deriv * time_fac;

          // assemble contributions into force vector and stiffness matrix
          (*fint).get_values()[dof_lid] += force * cross_section;
          if (stiff != nullptr) stiff->assemble(-stiffness * cross_section, dof_gid, dof_gid);
        }
      }
      else
      {
        FOUR_C_THROW(
            "Only 'xyz' for 'DIRECTION' supported in 'DESIGN POINT ROBIN SPRING DASHPOT "
            "CONDITIONS'");
      }
      break;
    }
    default:
      FOUR_C_THROW("Geometry type for spring dashpot must either be either 'Surface' or 'Point'.");
  }
}


// old version, NOT consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 |                                                         pfaller Mar16|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::evaluate_force(Core::LinAlg::Vector<double>& fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const Core::LinAlg::Vector<double>& vel, const Teuchos::ParameterList& p)
{
  if (disp == nullptr) FOUR_C_THROW("Cannot find displacement state in discretization");

  if (springtype_ == RobinSpringDashpotType::cursurfnormal) get_cur_normals(disp, p);

  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->my_gid(node_gid))
    {
      int gid = node_gid;
      Core::Nodes::Node* node = actdisc_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find global node {}", gid);

      // get nodal values
      const double nodalarea = area_[gid];               // nodal area
      const std::vector<double> normal = normals_[gid];  // normalized nodal normal
      const std::vector<double> offsetprestr =
          offset_prestr_[gid];  // get nodal displacement values of last time step for MULF offset

      const int numdof = actdisc_->num_dof(0, node);
      FOUR_C_ASSERT(numdof == 3, "Only implemented for 3 DoFs per node.");
      std::vector<int> dofs = actdisc_->dof(0, node);

      // initialize
      double gap = 0.;          // displacement
      double gapdt = 0.;        // velocity
      double springstiff = 0.;  // spring stiffness

      // calculation of normals and displacements differs for each spring variant
      switch (springtype_)
      {
        case RobinSpringDashpotType::xyz:  // spring dashpot acts in every surface dof direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case RobinSpringDashpotType::refsurfnormal:  // spring dashpot acts in refnormal direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case RobinSpringDashpotType::cursurfnormal:  // spring dashpot acts in curnormal direction

          // safety checks
          const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
          const auto* numfuncvisco = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
          const auto* numfuncdisploffset =
              &spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
          const auto* numfuncnonlinstiff =
              &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
          for (int dof_numfuncstiff : *numfuncstiff)
          {
            if (dof_numfuncstiff != 0)
            {
              FOUR_C_THROW(
                  "temporal dependence of stiffness not implemented for current surface "
                  "evaluation");
            }
          }
          for (int dof_numfuncvisco : *numfuncvisco)
          {
            if (dof_numfuncvisco != 0)
              FOUR_C_THROW(
                  "temporal dependence of damping not implemented for current surface evaluation");
          }
          for (int dof_numfuncdisploffset : *numfuncdisploffset)
          {
            if (dof_numfuncdisploffset != 0)
              FOUR_C_THROW(
                  "temporal dependence of offset not implemented for current surface evaluation");
          }
          for (int dof_numfuncnonlinstiff : *numfuncnonlinstiff)
          {
            if (dof_numfuncnonlinstiff != 0)
              FOUR_C_THROW("Nonlinear spring not implemented for current surface evaluation");
          }

          // spring displacement
          gap = gap_[gid];
          //        gapdt = gapdt_[gid]; // unused ?!?

          // select spring stiffnes
          springstiff = select_stiffness(gap);

          // assemble into residual vector
          std::vector<double> out_vec(numdof, 0.);
          for (int k = 0; k < numdof; ++k)
          {
            // force
            const double val =
                -nodalarea *
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
            fint.sum_into_global_values(1, &val, &dofs[k]);

            // store spring stress for output
            out_vec[k] =
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
          }
          // add to output
          springstress_.insert(std::pair<int, std::vector<double>>(gid, out_vec));
          break;
      }
    }  // node owned by processor
  }  // loop over nodes
}


// old version, NOT consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 |                                                         pfaller mar16|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::evaluate_force_stiff(Core::LinAlg::SparseMatrix& stiff,
    Core::LinAlg::Vector<double>& fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const Core::LinAlg::Vector<double>& vel, Teuchos::ParameterList p)
{
  if (disp == nullptr) FOUR_C_THROW("Cannot find displacement state in discretization");

  if (springtype_ == RobinSpringDashpotType::cursurfnormal)
  {
    get_cur_normals(disp, p);
    stiff.un_complete();  // sparsity pattern might change
  }

  // time-integration factor for stiffness contribution of dashpot, d(v_{n+1})/d(d_{n+1})
  const double dt = p.get("dt", 1.0);

  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->my_gid(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node {}", node_gid);

      // get nodal values
      const double nodalarea = area_[node_gid];               // nodal area
      const std::vector<double> normal = normals_[node_gid];  // normalized nodal normal
      const std::vector<double> offsetprestr =
          offset_prestr_[node_gid];  // get nodal displacement values of last time step for MULF
                                     // offset

      const int numdof = actdisc_->num_dof(0, node);
      FOUR_C_ASSERT(numdof == 3, "Only implemented for 3 DoFs per node.");
      std::vector<int> dofs = actdisc_->dof(0, node);

      // initialize
      double gap = 0.;          // displacement
      double gapdt = 0.;        // velocity
      double springstiff = 0.;  // spring stiffness

      // calculation of normals and displacements differs for each spring variant
      switch (springtype_)
      {
        case RobinSpringDashpotType::xyz:  // spring dashpot acts in every surface dof direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case RobinSpringDashpotType::refsurfnormal:  // spring dashpot acts in refnormal direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case RobinSpringDashpotType::cursurfnormal:  // spring dashpot acts in curnormal direction

          // safety checks
          const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
          const auto* numfuncvisco = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
          const auto* numfuncdisploffset =
              &spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
          const auto* numfuncnonlinstiff =
              &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
          for (int dof_numfuncstiff : *numfuncstiff)
          {
            if (dof_numfuncstiff != 0)
            {
              FOUR_C_THROW(
                  "temporal dependence of stiffness not implemented for current surface "
                  "evaluation");
            }
          }
          for (int dof_numfuncvisco : *numfuncvisco)
          {
            if (dof_numfuncvisco != 0)
              FOUR_C_THROW(
                  "temporal dependence of damping not implemented for current surface evaluation");
          }
          for (int dof_numfuncdisploffset : *numfuncdisploffset)
          {
            if (dof_numfuncdisploffset != 0)
              FOUR_C_THROW(
                  "temporal dependence of offset not implemented for current surface evaluation");
          }
          for (int dof_numfuncnonlinstiff : *numfuncnonlinstiff)
          {
            if (dof_numfuncnonlinstiff != 0)
              FOUR_C_THROW("Nonlinear spring not implemented for current surface evaluation");
          }

          // spring displacement
          gap = gap_[node_gid];
          gapdt = gapdt_[node_gid];

          // select spring stiffnes
          springstiff = select_stiffness(gap);

          // assemble into residual vector and stiffness matrix
          std::vector<double> out_vec(numdof, 0.);
          for (int k = 0; k < numdof; ++k)
          {
            // force
            const double val =
                -nodalarea *
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
            fint.sum_into_global_values(1, &val, &dofs[k]);

            // stiffness
            std::map<int, double> dgap = dgap_[node_gid];
            std::vector<Core::Gen::Pairedvector<int, double>> dnormal = dnormals_[node_gid];

            // check if projection exists
            if (!dnormal.empty() && !dgap.empty())
            {
              // linearize gap
              for (auto& i : dgap)
              {
                const double dval = -nodalarea *
                                    (springstiff * (i.second) + viscosity_ * (i.second) / dt) *
                                    normal[k];
                stiff.assemble(dval, dofs[k], i.first);
              }

              // linearize normal
              for (auto& i : dnormal[k])
              {
                const double dval =
                    -nodalarea *
                    (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) *
                    (i.second);
                stiff.assemble(dval, dofs[k], i.first);
              }
            }
            // store negative value of internal force for output (=reaction force)
            out_vec[k] = -val;
          }
          // add to output
          springstress_.insert(std::pair<int, std::vector<double>>(node_gid, out_vec));
          break;
      }
    }  // node owned by processor
  }  // loop over nodes

  if (springtype_ == RobinSpringDashpotType::cursurfnormal)
    stiff.complete();  // sparsity pattern might have changed
}

/*----------------------------------------------------------------------*
 |                                                         pfaller Mar16|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::reset_newton()
{
  // all springs
  gap_.clear();
  gapdt_.clear();
  springstress_.clear();

  // only curnormal
  if (springtype_ == RobinSpringDashpotType::cursurfnormal)
  {
    dgap_.clear();
    normals_.clear();
    dnormals_.clear();
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::reset_prestress(const Core::LinAlg::Vector<double>& dis)
{
  // this should be sufficient, no need to loop over nodes anymore
  offset_prestr_new_->update(1.0, dis, 1.0);

  // loop over all nodes only necessary for cursurfnormal which does not use consistent integration
  if (springtype_ == RobinSpringDashpotType::cursurfnormal)
  {
    for (int node_gid : *nodes_)
    {
      // nodes owned by processor
      if (actdisc_->node_row_map()->my_gid(node_gid))
      {
        Core::Nodes::Node* node = actdisc_->g_node(node_gid);
        if (!node) FOUR_C_THROW("Cannot find global node {}", node_gid);

        const int numdof = actdisc_->num_dof(0, node);
        FOUR_C_ASSERT(numdof == 3, "Only implemented for 3 DoFs per node.");
        std::vector<int> dofs = actdisc_->dof(0, node);

        // initialize. calculation of displacements differs for each spring variant
        std::vector<double> uoff(numdof, 0.0);  // displacement vector of condition nodes
        offset_prestr_.insert(std::pair<int, std::vector<double>>(node_gid, uoff));

      }  // node owned by processor
    }  // loop over nodes
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::set_restart(Core::LinAlg::Vector<double>& vec)
{
  offset_prestr_new_->update(1.0, vec, 0.0);
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::set_restart_old(Core::LinAlg::MultiVector<double>& vec)
{
  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->my_gid(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node {}", node_gid);

      [[maybe_unused]] const int numdof = actdisc_->num_dof(0, node);
      FOUR_C_ASSERT(numdof == 3, "Only implemented for 3 DoFs per node.");
      std::vector<int> dofs = actdisc_->dof(0, node);


      if (springtype_ == RobinSpringDashpotType::refsurfnormal ||
          springtype_ == RobinSpringDashpotType::xyz)
      {
        // import spring offset length
        for (auto& i : offset_prestr_)
        {
          // global id -> local id
          const int lid = vec.get_map().lid(i.first);
          // local id on processor
          if (lid >= 0)
          {
            // copy all components of spring offset length vector
            (i.second)[0] = ((vec)(0)).local_values_as_span()[lid];
            (i.second)[1] = ((vec)(1)).local_values_as_span()[lid];
            (i.second)[2] = ((vec)(2)).local_values_as_span()[lid];
          }
        }
      }

    }  // node owned by processor
  }  // loop over nodes
}

/*----------------------------------------------------------------------*
 |                                                         pfaller Jan14|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::output_gap_normal(Core::LinAlg::Vector<double>& gap,
    Core::LinAlg::MultiVector<double>& normals, Core::LinAlg::MultiVector<double>& stress) const
{
  // export gap function
  for (const auto& i : gap_)
  {
    // global id -> local id
    const int lid = gap.get_map().lid(i.first);
    // local id on processor
    if (lid >= 0) (gap).get_values()[lid] += i.second;
  }

  // export normal
  for (const auto& normal : normals_)
  {
    // global id -> local id
    const int lid = normals.get_map().lid(normal.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of normal vector
      ((normals)(0)).get_values()[lid] += (normal.second).at(0);
      ((normals)(1)).get_values()[lid] += (normal.second).at(1);
      ((normals)(2)).get_values()[lid] += (normal.second).at(2);
    }
  }

  // export spring stress
  for (const auto& i : springstress_)
  {
    // global id -> local id
    const int lid = stress.get_map().lid(i.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of normal vector
      ((stress)(0)).get_values()[lid] += (i.second).at(0);
      ((stress)(1)).get_values()[lid] += (i.second).at(1);
      ((stress)(2)).get_values()[lid] += (i.second).at(2);
    }
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv Dec15|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::output_prestr_offset(
    Core::LinAlg::Vector<double>& springprestroffset) const
{
  springprestroffset.update(1.0, *offset_prestr_new_, 0.0);
}

/*----------------------------------------------------------------------*
 |                                                             mhv Dec15|
 *----------------------------------------------------------------------*/
void Constraints::SpringDashpot::output_prestr_offset_old(
    Core::LinAlg::MultiVector<double>& springprestroffset) const
{
  // export spring offset length
  for (const auto& i : offset_prestr_)
  {
    // global id -> local id
    const int lid = springprestroffset.get_map().lid(i.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of spring offset length vector
      ((springprestroffset)(0)).get_values()[lid] = (i.second)[0];
      ((springprestroffset)(1)).get_values()[lid] = (i.second)[1];
      ((springprestroffset)(2)).get_values()[lid] = (i.second)[2];
    }
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::initialize_cur_surf_normal()
{
  // create MORTAR interface
  mortar_ = std::make_shared<Adapter::CouplingNonLinMortar>(Global::Problem::instance()->n_dim(),
      Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  // create CONTACT elements at interface for normal and gap calculation
  mortar_->setup_spring_dashpot(actdisc_, actdisc_, *spring_, coupling_, actdisc_->get_comm());

  // create temp vectors for gap initialization
  std::map<int, std::map<int, double>> tmpdgap_;
  std::map<int, std::vector<double>> tmpnormals_;
  std::map<int, std::vector<Core::Gen::Pairedvector<int, double>>> tmpdnormals_;

  // empty displacement vector
  std::shared_ptr<Core::LinAlg::Vector<double>> disp;
  disp = std::make_shared<Core::LinAlg::Vector<double>>(*(actdisc_->dof_row_map()), true);

  // initialize gap in reference configuration
  mortar_->interface()->evaluate_distances(disp, tmpnormals_, tmpdnormals_, gap0_, tmpdgap_);
}

// ToDo: this function should vanish completely
// obsolete when using new EvaluateRobin function!
/*-----------------------------------------------------------------------*
|(private) adapted from mhv 01/14                           pfaller Apr15|
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::get_area(
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom)
{
  for (const auto& ele : geom)
  {
    Core::Elements::Element& element = *ele.second.get();

    double area = 0.0;
    Core::FE::cell_type_switch<supported_celltypes>(element.shape(),
        [&](auto celltype_t)
        {
          constexpr Core::FE::CellType celltype = celltype_t();
          constexpr int num_dof_per_node = 3;

          const Core::Elements::ElementNodes<celltype, num_dof_per_node> element_nodes =
              Core::Elements::evaluate_element_nodes<celltype, num_dof_per_node>(
                  *actdisc_, *ele.second);

          Core::FE::GaussIntegration gauss_integration =
              Core::FE::create_gauss_integration<celltype>(
                  Discret::Elements::DisTypeToOptGaussRule<celltype>::rule);
          Core::Elements::for_each_gauss_point<celltype>(element_nodes, gauss_integration,
              [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
                  const Core::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
                  const Core::Elements::JacobianMapping<celltype, 3>& jacobian_mapping,
                  double integration_factor, int gp) { area += integration_factor; });
        });

    // loop over all nodes of the element that share the area
    // do only contribute to my own row nodes
    double apernode = 0.;
    for (int i = 0; i < element.num_node(); ++i)
    {
      /* here we have to take care to assemble the right stiffness to the nodes!!! (mhv 05/2014):
          we do some sort of "manual" gauss integration here since we have to pay attention to
         assemble the correct stiffness in case of quadratic surface elements*/

      switch (element.shape())
      {
        case Core::FE::CellType::tri3:
          apernode = area / element.num_node();
          break;
        case Core::FE::CellType::tri6:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = 0.;
          double int_N_edgemidnode = 1. / 6.;

          int numcornernode = 3;
          int numedgemidnode = 3;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode;
          double a_inv_weight = area / weight;

          if (i < 3)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::quad4:
          apernode = area / element.num_node();
          break;
        case Core::FE::CellType::quad8:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = -1. / 3.;
          double int_N_edgemidnode = 4. / 3.;

          int numcornernode = 4;
          int numedgemidnode = 4;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode;
          double a_inv_weight = area / weight;

          if (i < 4)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::quad9:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = 1. / 9.;
          double int_N_edgemidnode = 4. / 9.;
          double int_N_centermidnode = 16. / 9.;

          int numcornernode = 4;
          int numedgemidnode = 4;
          int numcentermidnode = 1;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode +
                          numcentermidnode * int_N_centermidnode;
          double a_inv_weight = area / weight;

          if (i < 4)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else if (i == 8)  // center mid node
            apernode = int_N_centermidnode * area / weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::nurbs9:
          FOUR_C_THROW(
              "Not yet implemented for Nurbs! To do: Apply the correct weighting of the area per "
              "node!");
          break;
        default:
          FOUR_C_THROW("shape type unknown!\n");
          break;
      }

      const int gid = element.nodes()[i]->id();
      if (!actdisc_->node_row_map()->my_gid(gid)) continue;

      // store area in map (gid, area). erase old value before adding new one
      const double newarea = area_[gid] + apernode;
      area_.erase(gid);
      area_.insert(std::pair<int, double>(gid, newarea));
    }
  }  // for (ele=geom.begin(); ele != geom.end(); ++ele)
}


/*-----------------------------------------------------------------------*
|(private)                                                    mhv 12/2015|
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::initialize_prestr_offset()
{
  offset_prestr_.clear();

  for (int node_gid : *nodes_)
  {
    if (actdisc_->node_row_map()->my_gid(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node {}", node_gid);

      int numdof = actdisc_->num_dof(0, node);
      std::vector<int> dofs = actdisc_->dof(0, node);

      FOUR_C_ASSERT((numdof == 3 || numdof == 6), "Only implemented for 3 or 6 DoFs per node.");

      std::vector<double> temp(numdof, 0.0);

      // insert to map
      offset_prestr_.insert(std::pair<int, std::vector<double>>(node_gid, temp));
    }
  }
}


/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::get_cur_normals(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& disp, Teuchos::ParameterList p)
{
  // get current time step size
  const double dt = p.get("dt", 1.0);

  // temp nodal gap
  std::map<int, double> tmpgap;

  // calculate normals and gap using CONTACT elements
  mortar_->interface()->evaluate_distances(disp, normals_, dnormals_, tmpgap, dgap_);

  // subtract reference gap from current gap (gap in reference configuration is zero everywhere)
  for (auto& i : tmpgap)
  {
    auto j = gap0_.find(i.first);
    if (j == gap0_.end()) gap_[i.first] = i.second;
    //      FOUR_C_THROW("The maps of reference gap and current gap are inconsistent.");
    else
      gap_[i.first] = i.second - j->second;

    // calculate gap velocity via local finite difference (not the best way but also not the worst)
    gapdt_[i.first] = (gap_[i.first] - gapn_[i.first]) / dt;
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::set_spring_type()
{
  // get spring direction from condition
  springtype_ = spring_->parameters().get<RobinSpringDashpotType>("DIRECTION");
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Constraints::SpringDashpot::update()
{
  // store current time step
  gapn_ = gap_;
}

void Constraints::SpringDashpot::reset_step_state() { gap_ = gapn_; }

FOUR_C_NAMESPACE_CLOSE
