// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele_calc.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


template <Core::FE::CellType distype>
Discret::Elements::Shell7pEleCalc<distype>::Shell7pEleCalc()
    : Discret::Elements::Shell7pEleCalcInterface::Shell7pEleCalcInterface(),
      intpoints_midsurface_(
          Shell::create_gauss_integration_points<distype>(Shell::get_gauss_rule<distype>()))
{
  cur_thickness_.resize(intpoints_midsurface_.num_points(), shell_data_.thickness);
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::setup(Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::IO::InputParameterContainer& container,
    const Solid::Elements::ShellLockingTypes& locking_types,
    const Solid::Elements::ShellData& shell_data)
{
  shell_data_ = shell_data;
  // initialize current thickness at all gp
  cur_thickness_.resize(intpoints_midsurface_.num_points(), shell_data_.thickness);
  //  set up of materials with GP data (e.g., history variables)
  solid_material.setup(intpoints_midsurface_.num_points(), read_fibers(container),
      read_coordinate_system(container));
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, shell_data_.sdc);
  add_to_pack(data, shell_data_.thickness);
  add_to_pack(data, shell_data_.num_ans);
  add_to_pack(data, cur_thickness_);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, shell_data_.sdc);
  extract_from_pack(buffer, shell_data_.thickness);
  extract_from_pack(buffer, shell_data_.num_ans);
  extract_from_pack(buffer, cur_thickness_);
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::material_post_setup(
    Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  // element/nodal wise defined data
  Teuchos::ParameterList params{};

  // Call post_setup of material
  solid_material.post_setup(params, ele.id());
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::reset_to_last_converged(
    Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  solid_material.reset_step();
}

template <Core::FE::CellType distype>
double Discret::Elements::Shell7pEleCalc<distype>::calculate_internal_energy(
    Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params)
{
  // need update
  double intenergy = 0.0;

  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  std::shared_ptr<const Core::LinAlg::Vector<double>> res =
      discretization.get_state("residual displacement");
  if (disp == nullptr || res == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;

  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // Assumed Natural Strains (ANS) Technology to remedy transverse shear strain locking
  std::vector<double> shape_functions_ans(true);

  // for a_13 and a_23 each
  const int total_ansq = 2 * shell_data_.num_ans;
  std::vector<Shell::ShapefunctionsAndDerivatives<distype>> shapefunctions_collocation(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_reference(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_current(total_ansq);

  if (shell_data_.num_ans > 0)
  {
    Shell::setup_ans(shapefunctions_collocation, metrics_collocation_reference,
        metrics_collocation_current, nodal_coordinates, total_ansq);
  }

  // init metric tensor and basis vectors of element mid-surface
  Shell::BasisVectorsAndMetrics<distype> a_reference;
  Shell::BasisVectorsAndMetrics<distype> a_current;

  // init metric tensor and basis vectors of element shell body
  Shell::BasisVectorsAndMetrics<distype> g_reference;
  Shell::BasisVectorsAndMetrics<distype> g_current;

  const double* total_time =
      params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
  const double* time_step_size =
      params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;

  Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
      [&](const std::array<double, 2>& xi_gp,
          const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
          Shell::BasisVectorsAndMetrics<distype>& a_current,
          Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
      {
        double integration_factor = gpweight * da;

        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        // integration loop in thickness direction, here we prescribe 2 integration points
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;

          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          // modify the current covariant metric tensor to neglect the quadratic terms in thickness
          // directions
          Shell::modify_covariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);

          // evaluate Green-Lagrange strains and deformation gradient in curvilinear coordinate
          // system
          auto strains = evaluate_strains(g_reference, g_current);

          // transform gl strains to cartesian system
          strains.gl_strain_ =
              Shell::transform_green_lagrange_strain_to_cartesian(strains.gl_strain_, g_reference);

          // call material for evaluation of strain energy function
          Core::LinAlg::Tensor<double, 3> xi = {{xi_gp[0], xi_gp[1], 0.0}};
          Mat::EvaluationContext context{.total_time = total_time,
              .time_step_size = time_step_size,
              .xi = &xi,
              .ref_coords = nullptr};
          double psi = solid_material.strain_energy(strains.gl_strain_, context, gp, ele.id());

          double thickness = 0.0;
          for (int i = 0; i < Shell::Internal::num_node<distype>; ++i)
            thickness += thickness * shape_functions.shapefunctions_(i);

          intenergy += psi * integration_factor * 0.5 * thickness;
        }
      });

  return intenergy;
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::calculate_stresses_strains(
    Core::Elements::Element& ele, Mat::So3Material& solid_material, const ShellStressIO& stressIO,
    const ShellStrainIO& strainIO, const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params)
{
  std::vector<char>& serialized_stress_data = stressIO.mutable_data;
  std::vector<char>& serialized_strain_data = strainIO.mutable_data;
  Core::LinAlg::SerialDenseMatrix stress_data(
      intpoints_midsurface_.num_points(), Mat::NUM_STRESS_3D);
  Core::LinAlg::SerialDenseMatrix strain_data(
      intpoints_midsurface_.num_points(), Mat::NUM_STRESS_3D);


  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  std::shared_ptr<const Core::LinAlg::Vector<double>> res =
      discretization.get_state("residual displacement");
  if (disp == nullptr || res == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;

  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // Assumed Natural Strains (ANS) Technology to remedy transverse shear strain locking
  // for a_13 and a_23 each
  const int total_ansq = 2 * shell_data_.num_ans;
  std::vector<Shell::ShapefunctionsAndDerivatives<distype>> shapefunctions_collocation(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_reference(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_current(total_ansq);

  if (shell_data_.num_ans > 0)
  {
    Shell::setup_ans(shapefunctions_collocation, metrics_collocation_reference,
        metrics_collocation_current, nodal_coordinates, total_ansq);
  }

  // metric of element at centroid point (for EAS)
  std::array<double, 2> centroid_point = {0.0, 0.0};
  Shell::ShapefunctionsAndDerivatives<distype> shapefunctions_centroid =
      Shell::evaluate_shapefunctions_and_derivs<distype>(centroid_point);
  Shell::BasisVectorsAndMetrics<distype> metrics_centroid_reference;
  Shell::BasisVectorsAndMetrics<distype> metrics_centroid_current;

  Shell::evaluate_metrics(shapefunctions_centroid, metrics_centroid_reference,
      metrics_centroid_current, nodal_coordinates, 0.0);

  // init metric tensor and basis vectors of element mid-surface
  Shell::BasisVectorsAndMetrics<distype> a_reference;
  Shell::BasisVectorsAndMetrics<distype> a_current;
  // init metric tensor and basis vectors of element shell body
  Shell::BasisVectorsAndMetrics<distype> g_reference;
  Shell::BasisVectorsAndMetrics<distype> g_current;

  const double* total_time =
      params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
  const double* time_step_size =
      params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;
  Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
      [&](const std::array<double, 2>& xi_gp,
          const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
          Shell::BasisVectorsAndMetrics<distype>& a_current,
          Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
      {
        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        // integration loop in thickness direction, here we prescribe 2 integration points
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;
          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          // modify the current covariant metric tensor to neglect the quadratic terms in thickness
          // directions
          Shell::modify_covariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);

          // evaluate Green-Lagrange strains and deformation gradient in curvilinear coordinate
          // system
          auto strains = evaluate_strains(g_reference, g_current);

          // transform gl strains to cartesian system
          strains.gl_strain_ =
              Shell::transform_green_lagrange_strain_to_cartesian(strains.gl_strain_, g_reference);

          Core::LinAlg::Tensor<double, 3> xi = {{xi_gp[0], xi_gp[1], 0.0}};
          Mat::EvaluationContext context{.total_time = total_time,
              .time_step_size = time_step_size,
              .xi = &xi,
              .ref_coords = nullptr};
          // evaluate stress in global cartesian system
          auto stress = Shell::evaluate_material_stress_cartesian_system<Shell::Internal::num_dim>(
              solid_material, strains.defgrd_, strains.gl_strain_, params, context, gp, ele.id());
          Shell::assemble_strain_type_to_matrix_row(
              strains.gl_strain_, strains.defgrd_, strainIO.type, strain_data, gp, 0.5);
          Shell::assemble_stress_type_to_matrix_row(
              strains.defgrd_, stress, stressIO.type, stress_data, gp, 0.5);
        }
      });
  Shell::serialize(stress_data, serialized_stress_data);
  Shell::serialize(strain_data, serialized_strain_data);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::evaluate_nonlinear_force_stiffness_mass(
    Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector* force_vector,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix, Core::LinAlg::SerialDenseMatrix* mass_matrix)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  std::shared_ptr<const Core::LinAlg::Vector<double>> res =
      discretization.get_state("residual displacement");
  if (disp == nullptr || res == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;


  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // Assumed Natural Strains (ANS) Technology to remedy transverse shear strain locking
  // for a_13 and a_23 each
  const int total_ansq = 2 * shell_data_.num_ans;
  std::vector<Shell::ShapefunctionsAndDerivatives<distype>> shapefunctions_collocation(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_reference(total_ansq);
  std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_current(total_ansq);

  if (shell_data_.num_ans > 0)
  {
    Shell::setup_ans(shapefunctions_collocation, metrics_collocation_reference,
        metrics_collocation_current, nodal_coordinates, total_ansq);
  }

  // init metric tensor and basis vectors of element mid-surface
  Shell::BasisVectorsAndMetrics<distype> a_reference;
  Shell::BasisVectorsAndMetrics<distype> a_current;

  // init metric tensor and basis vectors of element shell body
  Shell::BasisVectorsAndMetrics<distype> g_reference;
  Shell::BasisVectorsAndMetrics<distype> g_current;

  constexpr auto num_internal_variables = Shell::Internal::num_internal_variables;

  Shell::StressResultants stress_resultants{};

  const double* total_time =
      params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
  const double* time_step_size =
      params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;
  Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
      [&](const std::array<double, 2>& xi_gp,
          const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
          Shell::BasisVectorsAndMetrics<distype>& a_current,
          Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
      {
        double integration_factor = gpweight * da;

        // update current thickness at gauss point
        cur_thickness_[gp] = Shell::update_gauss_point_thickness<distype>(
            nodal_coordinates.a3_curr_, shape_functions.shapefunctions_);

        // reset mid-surface material tensor and stress resultants to zero
        stress_resultants.dmat_.shape(num_internal_variables, num_internal_variables);
        stress_resultants.stress_.size(num_internal_variables);

        // init mass matrix variables
        Shell::MassMatrixVariables mass_matrix_variables;

        // calculate B-operator for compatible strains (displacement)
        Core::LinAlg::SerialDenseMatrix Bop = Shell::calc_b_operator<distype>(
            a_current.covariant_, a_current.partial_derivative_, shape_functions);

        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        // modifications due to ANS with B-bar method (Hughes (1980))
        std::invoke(
            [&]()
            {
              if (shell_data_.num_ans > 0)
              {
                modify_b_operator_ans(Bop, shape_functions_ans, shapefunctions_collocation,
                    metrics_collocation_current, shell_data_.num_ans);
              }
            });


        // integration loop in thickness direction, here we prescribe 2 integration points
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;
          double factor = intpoints_thickness_.qwgt[gpt];

          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          // modify the current covariant metric tensor to neglect the quadratic terms in thickness
          // directions
          Shell::modify_covariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);

          // calc shell shifter and put it in the integration factor
          const double shifter = (1.0 / condfac) * (g_reference.detJ_ / da);
          factor *= shifter;

          // evaluate Green-Lagrange strains and deformation gradient in curvilinear coordinate
          // system
          auto strains = evaluate_strains(g_reference, g_current);

          // transform gl strains to cartesian system
          strains.gl_strain_ =
              Shell::transform_green_lagrange_strain_to_cartesian(strains.gl_strain_, g_reference);

          Core::LinAlg::Tensor<double, 3> xi = {{xi_gp[0], xi_gp[1], 0.0}};
          Mat::EvaluationContext context{.total_time = total_time,
              .time_step_size = time_step_size,
              .xi = &xi,
              .ref_coords = nullptr};

          auto stress = Shell::evaluate_material_stress_cartesian_system<Shell::Internal::num_dim>(
              solid_material, strains.defgrd_, strains.gl_strain_, params, context, gp, ele.id());
          Shell::map_material_stress_to_curvilinear_system(stress, g_reference);
          Shell::thickness_integration<distype>(stress_resultants, stress, factor, zeta);

          // thickness integration of mass matrix variables
          if (mass_matrix != nullptr)
          {
            double tmp_integration_factor = intpoints_thickness_.qwgt[gpt] * g_reference.detJ_;
            mass_matrix_variables.factor_v_ += tmp_integration_factor;
            mass_matrix_variables.factor_w_ += tmp_integration_factor *
                                               intpoints_thickness_.qxg[gpt][0] *
                                               intpoints_thickness_.qxg[gpt][0];
            mass_matrix_variables.factor_vw_ +=
                tmp_integration_factor * intpoints_thickness_.qxg[gpt][0];
          }
        }
        // add stiffness matrix
        if (stiffness_matrix != nullptr)
        {
          // elastic stiffness matrix Ke
          Shell::add_elastic_stiffness_matrix<distype>(
              Bop, stress_resultants.dmat_, integration_factor, *stiffness_matrix);
          // geometric stiffness matrix Kg
          Shell::add_geometric_stiffness_matrix<distype>(shapefunctions_collocation,
              shape_functions_ans, shape_functions, stress_resultants.stress_, shell_data_.num_ans,
              integration_factor, *stiffness_matrix);
        }
        // add internal force vector
        if (force_vector != nullptr)
        {
          Shell::add_internal_force_vector<distype>(
              Bop, stress_resultants.stress_, integration_factor, *force_vector);
        }
        // add internal mass_matrix
        if (mass_matrix != nullptr)
        {
          double density = solid_material.density(gp);
          mass_matrix_variables.factor_v_ *= gpweight * density;
          mass_matrix_variables.factor_w_ *= gpweight * density;
          mass_matrix_variables.factor_vw_ *= gpweight * density;
          Shell::add_mass_matrix(
              shape_functions, mass_matrix_variables, shell_data_.thickness, *mass_matrix);
        }
      });
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::recover(Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params, Solid::Elements::ParamsInterface& str_interface)
{
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::update(Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement' ");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);

  // calculate and update inelastic deformation gradient if needed
  if (solid_material.uses_extended_update())
  {
    const double condfac = shell_data_.sdc;

    // init gauss point in thickness direction that will be modified via SDC
    double zeta = 0.0;

    // get nodal coordinates
    Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
        ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

    // Assumed Natural Strains (ANS) Technology to remedy transverse shear strain locking
    // for a_13 and a_23 each
    const int total_ansq = 2 * shell_data_.num_ans;
    std::vector<Shell::ShapefunctionsAndDerivatives<distype>> shapefunctions_collocation(
        total_ansq);
    std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_reference(total_ansq);
    std::vector<Shell::BasisVectorsAndMetrics<distype>> metrics_collocation_current(total_ansq);

    if (shell_data_.num_ans > 0)
    {
      Shell::setup_ans(shapefunctions_collocation, metrics_collocation_reference,
          metrics_collocation_current, nodal_coordinates, total_ansq);
    }
    // init metric tensor and basis vectors of mid-surface
    Shell::BasisVectorsAndMetrics<distype> a_reference;
    Shell::BasisVectorsAndMetrics<distype> a_current;

    // init metric tensor and basis vectors of shell body
    Shell::BasisVectorsAndMetrics<distype> g_reference;
    Shell::BasisVectorsAndMetrics<distype> g_current;

    const double* total_time =
        params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
    const double* time_step_size =
        params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;
    Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
        [&](const std::array<double, 2>& xi_gp,
            const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
            Shell::BasisVectorsAndMetrics<distype>& a_current,
            Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
        {
          const std::vector<double> shape_functions_ans =
              Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

          // integration loop in thickness direction, here we prescribe 2 integration points
          for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
          {
            zeta = intpoints_thickness_.qxg[gpt][0] / condfac;

            Shell::evaluate_metrics(
                shape_functions, g_reference, g_current, nodal_coordinates, zeta);

            // modify the current covariant metric tensor to neglect the quadratic terms in
            // thickness directions
            Shell::modify_covariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
                shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
                shell_data_.num_ans);

            // evaluate Green-Lagrange strains and deformation gradient in curvilinear coordinate
            // system
            auto strains = evaluate_strains(g_reference, g_current);

            // transform gl strains to cartesian system
            strains.gl_strain_ = Shell::transform_green_lagrange_strain_to_cartesian(
                strains.gl_strain_, g_reference);

            Core::LinAlg::Tensor<double, 3> xi = {{xi_gp[0], xi_gp[1], 0.0}};
            Mat::EvaluationContext context{.total_time = total_time,
                .time_step_size = time_step_size,
                .xi = &xi,
                .ref_coords = nullptr};
            solid_material.update(strains.defgrd_, gp, params, context, ele.id());
          }
        });
  }
  solid_material.update();
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalc<distype>::vis_data(
    const std::string& name, std::vector<double>& data)
{
  if (name == "thickness")
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (auto& thickness_data : cur_thickness_)
    {
      data[0] += thickness_data;
    }
    data[0] = data[0] / intpoints_midsurface_.num_points();
  }

}  // vis_data()

// template classes
template class Discret::Elements::Shell7pEleCalc<Core::FE::CellType::quad4>;
template class Discret::Elements::Shell7pEleCalc<Core::FE::CellType::quad8>;
template class Discret::Elements::Shell7pEleCalc<Core::FE::CellType::quad9>;
template class Discret::Elements::Shell7pEleCalc<Core::FE::CellType::tri3>;
template class Discret::Elements::Shell7pEleCalc<Core::FE::CellType::tri6>;

FOUR_C_NAMESPACE_CLOSE
