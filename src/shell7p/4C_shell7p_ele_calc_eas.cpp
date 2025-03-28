// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele_calc_eas.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_calc_eas_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{

  /*!
   * @brief Evaluates the enhanced strains scalar increment alpha
   *
   * @tparam distype : discretization type
   * @param old_eas_data (in) : EAS iteration data
   * @param neas (in) : Number of EAS parameter
   * @param residual (in) : Residual displacement increment
   * @param alpha_inc (out) : Enhanced strains scalar increment
   */
  template <Core::FE::CellType distype>
  void evaluate_alpha_increment(Discret::Elements::ShellEASIterationData& old_eas_data,
      const int& neas, const std::vector<double>& residual,
      Core::LinAlg::SerialDenseMatrix& delta_alpha)
  {
    // we need the (residual) displacement at the previous step
    Core::LinAlg::SerialDenseVector disp_inc(
        Discret::Elements::Shell::Internal::numdofperelement<distype>);
    for (int i = 0; i < Discret::Elements::Shell::Internal::numdofperelement<distype>; ++i)
      disp_inc(i) = residual[i];

    Core::LinAlg::SerialDenseMatrix eashelp(neas, 1);
    // make multiplication eashelp = - old L^T * disp_incr[kstep]
    eashelp.multiply(
        Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, old_eas_data.transL_, disp_inc, 0.0);
    // add old RTilde to eashelp
    eashelp += old_eas_data.RTilde_;
    // make multiplication alpha_inc = - old invDTilde * eashelp
    delta_alpha.multiply(
        Teuchos::NO_TRANS, Teuchos::NO_TRANS, -1.0, old_eas_data.invDTilde_, eashelp, 0.0);
  }  // end of EvaluateAlphaIncrement

  /*!
   * @brief Integrate the EAS data
   *
   *  Function needs to be called for every gaussian point
   *
   * @tparam distype : discretization type
   * @param stress_enh (in) : An object holding the enhanced stress resultants
   * @param M (in) : EAS shapefunctions
   * @param Bop (in) : B-operator matrix
   * @param eas_data (in/out) : An object holding the EAS data
   * @param integration_factor (in) : Integration factor
   */
  template <Core::FE::CellType distype>
  void integrate_eas(const Discret::Elements::Shell::StressEnhanced& stress_enh,
      const Core::LinAlg::SerialDenseMatrix& M, const Core::LinAlg::SerialDenseMatrix& Bop,
      Discret::Elements::ShellEASIterationData& eas_data, const double& integration_factor,
      const int& neas)
  {
    // integrate D_Tilde += M^T * D * M  * detJ * w(gp)
    // IMPORTANT: here we save D_Tilde in invDTilde_, since after the loop over all Gaussian points,
    // we invert the matrix. At this point, this is still D_Tilde and NOT invD_Tilde
    Core::LinAlg::SerialDenseMatrix MTDmat(
        neas, Discret::Elements::Shell::Internal::num_internal_variables);
    MTDmat.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, M, stress_enh.dmat_, 0.0);

    eas_data.invDTilde_.multiply(
        Teuchos::NO_TRANS, Teuchos::NO_TRANS, integration_factor, MTDmat, M, 1.);

    //  integrate transL (L^T) += M^T * D * B * detJ * w(gp)
    eas_data.transL_.multiply(
        Teuchos::NO_TRANS, Teuchos::NO_TRANS, integration_factor, MTDmat, Bop, 1.);

    //  integrate Rtilde (R_Tilde) : Rtilde  += M^T * stress_r * detJ * w(gp)
    eas_data.RTilde_.multiply(
        Teuchos::TRANS, Teuchos::NO_TRANS, integration_factor, M, stress_enh.stress_, 1.);
  }
}  // namespace


template <Core::FE::CellType distype>
Discret::Elements::Shell7pEleCalcEas<distype>::Shell7pEleCalcEas()
    : Discret::Elements::Shell7pEleCalcInterface::Shell7pEleCalcInterface(),
      intpoints_midsurface_(
          Shell::create_gauss_integration_points<distype>(Shell::get_gauss_rule<distype>()))
{
  old_step_length_ = 0.0;
  cur_thickness_.resize(intpoints_midsurface_.num_points(), shell_data_.thickness);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::setup(Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::IO::InputParameterContainer& container,
    const Solid::Elements::ShellLockingTypes& locking_types,
    const Solid::Elements::ShellData& shell_data)
{
  shell_data_ = shell_data;
  cur_thickness_.resize(intpoints_midsurface_.num_points(), shell_data_.thickness);
  locking_types_ = locking_types;

  // init sizes of EAS data for integration
  eas_iteration_data_.alpha_.shape(locking_types.total, 1);
  eas_iteration_data_.RTilde_.shape(locking_types_.total, 1);
  eas_iteration_data_.invDTilde_.shape(locking_types_.total, locking_types_.total);
  eas_iteration_data_.transL_.shape(
      locking_types_.total, Shell::Internal::numdofperelement<distype>);

  //  set up of materials with GP data (e.g., history variables)
  solid_material.setup(intpoints_midsurface_.num_points(), container);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, shell_data_.sdc);
  add_to_pack(data, shell_data_.thickness);
  add_to_pack(data, shell_data_.num_ans);

  add_to_pack(data, eas_iteration_data_.alpha_);
  add_to_pack(data, eas_iteration_data_.RTilde_);
  add_to_pack(data, eas_iteration_data_.invDTilde_);
  add_to_pack(data, eas_iteration_data_.transL_);

  // number of total EAS parameters
  add_to_pack(data, locking_types_.membrane);
  add_to_pack(data, locking_types_.bending);
  add_to_pack(data, locking_types_.thickness);
  add_to_pack(data, locking_types_.transverse_shear_strain_const);
  add_to_pack(data, locking_types_.transverse_shear_strain_lin);
  add_to_pack(data, locking_types_.total);

  add_to_pack(data, old_step_length_);
  add_to_pack(data, cur_thickness_);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, shell_data_.sdc);
  extract_from_pack(buffer, shell_data_.thickness);
  extract_from_pack(buffer, shell_data_.num_ans);

  extract_from_pack(buffer, eas_iteration_data_.alpha_);
  extract_from_pack(buffer, eas_iteration_data_.RTilde_);
  extract_from_pack(buffer, eas_iteration_data_.invDTilde_);
  extract_from_pack(buffer, eas_iteration_data_.transL_);
  // number of total EAS parameters
  extract_from_pack(buffer, locking_types_.membrane);
  extract_from_pack(buffer, locking_types_.bending);
  extract_from_pack(buffer, locking_types_.thickness);
  extract_from_pack(buffer, locking_types_.transverse_shear_strain_const);
  extract_from_pack(buffer, locking_types_.transverse_shear_strain_lin);
  extract_from_pack(buffer, locking_types_.total);

  extract_from_pack(buffer, old_step_length_);
  extract_from_pack(buffer, cur_thickness_);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::material_post_setup(
    Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  // element/nodal wise defined data
  Teuchos::ParameterList params{};
  // Call post_setup of material
  solid_material.post_setup(params, ele.id());
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::reset_to_last_converged(
    Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  solid_material.reset_step();
}

template <Core::FE::CellType distype>
double Discret::Elements::Shell7pEleCalcEas<distype>::calculate_internal_energy(
    Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params)
{
  double intenergy = 0.0;

  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  std::shared_ptr<const Core::LinAlg::Vector<double>> res =
      discretization.get_state("residual displacement");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;

  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // Enhanced Assumed Strain (EAS) Technology: declare, initialize, set up, and alpha history
  Core::LinAlg::SerialDenseMatrix delta_alpha(locking_types_.total, 1);

  // EAS Update of alphas: the current alphas are (re-)evaluated out of DTilde and L^T of previous
  // step to avoid additional element call
  if (not ele.is_params_interface())
  {
    // compute the EAS increment delta_alpha
    evaluate_alpha_increment<distype>(
        eas_iteration_data_, locking_types_.total, residual, delta_alpha);
    // update alpha += 1.0 * delta_alpha
    eas_iteration_data_.alpha_ += delta_alpha;
  }

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
  const std::array<double, 2> centroid_point = {0.0, 0.0};
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

  Discret::Elements::Shell::Strains strains;
  // init enhanced strain for shell
  Core::LinAlg::SerialDenseVector strain_enh(Shell::Internal::num_internal_variables);

  // init EAS shape function matrix
  Core::LinAlg::SerialDenseMatrix M(Shell::Internal::num_internal_variables, locking_types_.total);

  Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
      [&](const std::array<double, 2>& xi_gp,
          const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
          Shell::BasisVectorsAndMetrics<distype>& a_current,
          Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
      {
        double integration_factor = gpweight * da;

        //  make shape functions for incompatible strains
        M = Shell::EAS::evaluate_eas_shape_functions(
            xi_gp, locking_types_, a_reference, metrics_centroid_reference);
        Shell::EAS::evaluate_eas_strains(strain_enh, eas_iteration_data_.alpha_, M);

        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        // integration loop in thickness direction, here we prescribe 2 integration points
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;

          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          // modify the current kovariant metric tensor to neglect the quadratic terms in thickness
          // directions
          Shell::modify_kovariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);

          // change to current metrics due to eas
          Shell::EAS::update_current_metrics_eas(g_current, strain_enh, zeta);

          // evaluate Green-Lagrange strains and deformation gradient in cartesian coordinate system
          strains = Shell::evaluate_strains(g_reference, g_current);

          // call material for evaluation of strain energy function
          double psi = 0.0;

          solid_material.strain_energy(strains.gl_strain_, psi, gp, ele.id());

          double thickness = 0.0;
          for (int i = 0; i < Shell::Internal::num_node<distype>; ++i)
            thickness += thickness * shape_functions.shapefunctions_(i);

          intenergy += psi * integration_factor * 0.5 * thickness;
        }
      });

  return intenergy;
}


template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::calculate_stresses_strains(
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
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;

  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // Enhanced Assumed Strain (EAS) Technology: declare, initialize, set up, and alpha history

  // EAS Update of alphas: the current alphas are (re-)evaluated out of DTilde and L^T of previous
  // step to avoid additional element call
  Core::LinAlg::SerialDenseMatrix delta_alpha(locking_types_.total, 1);

  if (not ele.is_params_interface())
  {
    // compute the EAS increment delta_alpha
    evaluate_alpha_increment<distype>(
        eas_iteration_data_, locking_types_.total, residual, delta_alpha);
    // update alpha += 1.0 * delta_alpha
    eas_iteration_data_.alpha_ += delta_alpha;
  }

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
  const std::array<double, 2> centroid_point = {0.0, 0.0};
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

  // init enhanced strains for shell
  Core::LinAlg::SerialDenseVector strain_enh(Shell::Internal::num_internal_variables);
  Shell::StressEnhanced stress_enh;

  // init EAS shape function matrix
  Core::LinAlg::SerialDenseMatrix M(Shell::Internal::num_internal_variables, locking_types_.total);

  Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
      [&](const std::array<double, 2>& xi_gp,
          const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
          Shell::BasisVectorsAndMetrics<distype>& a_current,
          Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
      {
        //  evaluate shape functions for incompatible strains
        M = Shell::EAS::evaluate_eas_shape_functions(
            xi_gp, locking_types_, a_reference, metrics_centroid_reference);

        Shell::EAS::evaluate_eas_strains(strain_enh, eas_iteration_data_.alpha_, M);

        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        // integration loop in thickness direction, here we prescribe 2 integration points to avoid
        // nonlinear poisson stiffening
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;
          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          // modify the current kovariant metric tensor to neglect the quadratic terms in thickness
          // directions
          Shell::modify_kovariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);

          // change to current metrics due to eas
          Shell::EAS::update_current_metrics_eas(g_current, strain_enh, zeta);

          // evaluate Green-Lagrange strains and deformationgradient in cartesian coordinate system
          auto strains = Shell::evaluate_strains(g_reference, g_current);

          // update the deformation gradient
          Core::LinAlg::Matrix<Shell::Internal::num_dim, Shell::Internal::num_dim> defgrd_enh(
              Core::LinAlg::Initialization::uninitialized);
          Shell::calc_consistent_defgrd<Shell::Internal::num_dim>(
              strains.defgrd_, strains.gl_strain_, defgrd_enh);
          strains.defgrd_ = defgrd_enh;

          // evaluate stress in local cartesian system
          auto stress = Shell::evaluate_material_stress_cartesian_system<Shell::Internal::num_dim>(
              solid_material, strains, params, gp, ele.id());
          Shell::assemble_strain_type_to_matrix_row<distype>(
              strains, strainIO.type, strain_data, gp, 0.5);
          Shell::assemble_stress_type_to_matrix_row<distype>(
              strains, stress, stressIO.type, stress_data, gp, 0.5);
        }
      });
  Shell::serialize(stress_data, serialized_stress_data);
  Shell::serialize(strain_data, serialized_strain_data);
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::evaluate_nonlinear_force_stiffness_mass(
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
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // init gauss point in thickness direction that will be modified via SDC
  double zeta = 0.0;

  // init scale factor for scaled director approach (SDC)
  const double condfac = shell_data_.sdc;

  // get nodal coordinates
  Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
      ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

  // Enhanced Assumed Strain (EAS) Technology: declare, initialize, set up, and alpha history

  // EAS Update of alphas: the current alphas are (re-)evaluated out of DTilde and L^T of previous
  // step to avoid additional element call
  Core::LinAlg::SerialDenseMatrix delta_alpha(locking_types_.total, 1);

  if (not ele.is_params_interface())
  {
    // compute the EAS increment delta_alpha
    evaluate_alpha_increment<distype>(
        eas_iteration_data_, locking_types_.total, residual, delta_alpha);
    // update alpha += 1.0 * delta_alpha
    eas_iteration_data_.alpha_ += delta_alpha;
  }

  // clear EAS data for integration
  eas_iteration_data_.RTilde_.shape(locking_types_.total, 1);
  eas_iteration_data_.invDTilde_.shape(locking_types_.total, locking_types_.total);
  eas_iteration_data_.transL_.shape(
      locking_types_.total, Shell::Internal::numdofperelement<distype>);

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
  const std::array<double, 2> centroid_point = {0.0, 0.0};
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

  // init enhanced strain for shell
  constexpr auto num_internal_variables = Shell::Internal::num_internal_variables;
  Core::LinAlg::SerialDenseVector strain_enh(num_internal_variables);
  Shell::StressEnhanced stress_enh;

  // init EAS shape function matrix
  Core::LinAlg::SerialDenseMatrix M(num_internal_variables, locking_types_.total);

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
        stress_enh.dmat_.shape(num_internal_variables, num_internal_variables);
        stress_enh.stress_.size(num_internal_variables);

        // init mass matrix variables
        Shell::MassMatrixVariables mass_matrix_variables;

        //  evaluate shape functions for incompatible strains
        M = Shell::EAS::evaluate_eas_shape_functions(
            xi_gp, locking_types_, a_reference, metrics_centroid_reference);
        Shell::EAS::evaluate_eas_strains(strain_enh, eas_iteration_data_.alpha_, M);

        // calculate B-operator for compatible strains (displacement)
        Core::LinAlg::SerialDenseMatrix Bop = Shell::calc_b_operator<distype>(
            a_current.kovariant_, a_current.partial_derivative_, shape_functions);

        const std::vector<double> shape_functions_ans =
            Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

        std::invoke(
            [&]()
            {
              if (shell_data_.num_ans > 0)
              {
                Shell::modify_b_operator_ans(Bop, shape_functions_ans, shapefunctions_collocation,
                    metrics_collocation_current, shell_data_.num_ans);
              }
            });

        // integration loop in thickness direction, here we prescribe 2 integration points to avoid
        // nonlinear poisson stiffening
        for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
        {
          zeta = intpoints_thickness_.qxg[gpt][0] / condfac;
          double factor = intpoints_thickness_.qwgt[gpt];

          // evaluate metric tensor at gp in shell body
          Shell::evaluate_metrics(shape_functions, g_reference, g_current, nodal_coordinates, zeta);

          Shell::modify_kovariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
              shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
              shell_data_.num_ans);


          // calc shell shifter and put it in the integration factor
          factor *= (1.0 / condfac) * (g_reference.detJ_ / da);

          // change to current metrics due to EAS
          Shell::EAS::update_current_metrics_eas(g_current, strain_enh, zeta);

          // evaluate Green-Lagrange strains and deformation gradient in cartesian coordinate system
          auto strains = Shell::evaluate_strains(g_reference, g_current);

          // update the deformation gradient (if needed?)
          if (solid_material.needs_defgrd())
          {
            Core::LinAlg::Matrix<Shell::Internal::num_dim, Shell::Internal::num_dim> defgrd_enh(
                Core::LinAlg::Initialization::uninitialized);
            Shell::calc_consistent_defgrd<Shell::Internal::num_dim>(
                strains.defgrd_, strains.gl_strain_, defgrd_enh);
            strains.defgrd_ = defgrd_enh;
          }

          auto stress = Shell::evaluate_material_stress_cartesian_system<Shell::Internal::num_dim>(
              solid_material, strains, params, gp, ele.id());
          Shell::map_material_stress_to_curvilinear_system(stress, g_reference);
          Shell::thickness_integration<distype>(stress_enh, stress, factor, zeta);
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

        // integration of EAS matrices
        integrate_eas<distype>(
            stress_enh, M, Bop, eas_iteration_data_, integration_factor, locking_types_.total);

        // add stiffness matrix
        if (stiffness_matrix != nullptr)
        {
          // elastic stiffness matrix Ke
          Shell::add_elastic_stiffness_matrix<distype>(
              Bop, stress_enh.dmat_, integration_factor, *stiffness_matrix);
          // geometric stiffness matrix Kg
          Shell::add_geometric_stiffness_matrix(shapefunctions_collocation, shape_functions_ans,
              shape_functions, stress_enh.stress_, shell_data_.num_ans, integration_factor,
              *stiffness_matrix);
        }
        // add internal force vector
        if (force_vector != nullptr)
        {
          Shell::add_internal_force_vector<distype>(
              Bop, stress_enh.stress_, integration_factor, *force_vector);
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

  // compute inverse of DTilde = invDTilde
  Core::LinAlg::symmetric_inverse(eas_iteration_data_.invDTilde_, locking_types_.total);

  // compute  L * DTilde^-1  which is later needed for force and stiffness update
  Core::LinAlg::SerialDenseMatrix LinvDTilde(
      Shell::Internal::numdofperelement<distype>, locking_types_.total);
  LinvDTilde.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1., eas_iteration_data_.transL_,
      eas_iteration_data_.invDTilde_, 0.);
  if (stiffness_matrix != nullptr)
  {
    Shell::EAS::add_eas_stiffness_matrix(
        LinvDTilde, eas_iteration_data_.transL_, *stiffness_matrix);
  }

  if (force_vector != nullptr)
  {
    Shell::EAS::add_eas_internal_force(LinvDTilde, eas_iteration_data_.RTilde_, *force_vector);
  }

  if (stiffness_matrix != nullptr)
  {
    // make stiffness matrix absolute symmetric
    for (int i = 0; i < Shell::Internal::numdofperelement<distype>; ++i)
    {
      for (int j = i + 1; j < Shell::Internal::numdofperelement<distype>; ++j)
      {
        const double average = 0.5 * ((*stiffness_matrix)(i, j) + (*stiffness_matrix)(j, i));
        (*stiffness_matrix)(i, j) = average;
        (*stiffness_matrix)(j, i) = average;
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::recover(Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params, Solid::Elements::ParamsInterface& interface_ptr)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> res =
      discretization.get_state("residual displacement");
  if (res == nullptr) FOUR_C_THROW("Cannot get residual displacement state vector");
  std::vector<double> residual = Core::FE::extract_values(*res, dof_index_array);

  // get access to the interface parameters
  double step_length = interface_ptr.get_step_length();

  // access general EAS history stuff stored in element
  Core::LinAlg::SerialDenseMatrix delta_alpha(locking_types_.total, 1);

  // if it is a default step, we have to recover the condensed solution vectors
  if (interface_ptr.is_default_step())
  {
    // first, store the eas state of the previous accepted Newton step
    interface_ptr.sum_into_my_previous_sol_norm(NOX::Nln::StatusTest::quantity_eas,
        locking_types_.total, eas_iteration_data_.alpha_[0], ele.owner());

    // compute the EAS increment delta_alpha
    evaluate_alpha_increment<distype>(
        eas_iteration_data_, locking_types_.total, residual, delta_alpha);
    // update alpha += step_length * delta_alpha
    delta_alpha.scale(step_length);
    eas_iteration_data_.alpha_ += delta_alpha;
  }
  // if it is no default step, we can correct the update and the current eas state without the
  // need for any matrix-vector products.
  else
  {
    // The first step has to be a default step!
    if (old_step_length_ < 0.0) FOUR_C_THROW("The old step length was not defined!");
    // if this is no full step, we have to adjust the length of the enhanced assumed strain
    // incremental step.
    // undo the previous step:
    //            alpha_new = alpha_old - old_step * delta_alpha
    // and update the solution variable with the new step length:
    //           alpha_new = alpha_new + new_step * delta_alpha
    delta_alpha.scale(step_length - old_step_length_);
    eas_iteration_data_.alpha_ += delta_alpha;
  }

  // Check if delta alpha is tested and if yes, calculate the element
  // contribution to the norm
  interface_ptr.sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas, locking_types_.total,
      delta_alpha[0], eas_iteration_data_.alpha_[0], step_length, ele.owner());

  // save the old step length
  old_step_length_ = step_length;
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::update(Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
    const Core::LinAlg::SerialDenseMatrix& nodal_directors, const std::vector<int>& dof_index_array,
    Teuchos::ParameterList& params)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement");
  if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement' ");
  std::vector<double> displacement = Core::FE::extract_values(*disp, dof_index_array);

  // No need to update alpha here. Update is called to copy states from t_{n+1} to
  // t_{n} after the time step and output Hence, there are no more Newton iterations that would
  // require an update of alpha

  // calculate and update inelastic deformation gradient if needed
  if (solid_material.uses_extended_update())
  {
    // init scale factor for scaled director approach (SDC)
    const double condfac = shell_data_.sdc;

    // init gauss point in thickness direction that will be modified via SDC
    double zeta = 0.0;

    // get nodal coordinates
    Shell::NodalCoordinates<distype> nodal_coordinates = Shell::evaluate_nodal_coordinates<distype>(
        ele.nodes(), displacement, shell_data_.thickness, nodal_directors, condfac);

    // metric of element centroid point (for EAS)
    const std::array<double, 2> centroid_point = {0.0, 0.0};
    Shell::ShapefunctionsAndDerivatives<distype> shapefunctions_centroid =
        Shell::evaluate_shapefunctions_and_derivs<distype>(centroid_point);
    Shell::BasisVectorsAndMetrics<distype> metrics_centroid_reference;
    Shell::BasisVectorsAndMetrics<distype> metrics_centroid_current;

    Shell::evaluate_metrics(shapefunctions_centroid, metrics_centroid_reference,
        metrics_centroid_current, nodal_coordinates, 0.0);

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

    // enhanced strain for shell
    constexpr auto num_internal_variables = Shell::Internal::num_internal_variables;
    Core::LinAlg::SerialDenseVector strain_enh(num_internal_variables);

    // init EAS shape function matrix
    Core::LinAlg::SerialDenseMatrix M(num_internal_variables, num_internal_variables);

    Shell::for_each_gauss_point<distype>(nodal_coordinates, intpoints_midsurface_,
        [&](const std::array<double, 2>& xi_gp,
            const Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
            Shell::BasisVectorsAndMetrics<distype>& a_current,
            Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp)
        {
          //  make shape functions for incompatible strains  M
          M = Shell::EAS::evaluate_eas_shape_functions(
              xi_gp, locking_types_, a_reference, metrics_centroid_reference);
          Shell::EAS::evaluate_eas_strains(strain_enh, eas_iteration_data_.alpha_, M);

          const std::vector<double> shape_functions_ans =
              Shell::get_shapefunctions_for_ans<distype>(xi_gp, shell_data_.num_ans);

          // integration loop in thickness direction, here we prescribe 2 integration points
          for (int gpt = 0; gpt < intpoints_thickness_.num_points(); ++gpt)
          {
            zeta = intpoints_thickness_.qxg[gpt][0] / condfac;

            Shell::evaluate_metrics(
                shape_functions, g_reference, g_current, nodal_coordinates, zeta);

            // modify the current kovariant metric tensor to neglect the quadratic terms in
            // thickness directions
            Shell::modify_kovariant_metrics(g_reference, g_current, a_reference, a_current, zeta,
                shape_functions_ans, metrics_collocation_reference, metrics_collocation_current,
                shell_data_.num_ans);


            Shell::EAS::update_current_metrics_eas(g_current, strain_enh, zeta);

            auto strains = evaluate_strains(g_reference, g_current);

            // calculate deformation gradient consistent with modified GL strain tensor
            if (solid_material.needs_defgrd())
            {
              // update the deformation gradient (if needed)
              Core::LinAlg::Matrix<Discret::Elements::Shell::Internal::num_dim,
                  Discret::Elements::Shell::Internal::num_dim>
                  defgrd_enh(Core::LinAlg::Initialization::uninitialized);
              Shell::calc_consistent_defgrd<Shell::Internal::num_dim>(
                  strains.defgrd_, strains.gl_strain_, defgrd_enh);
              strains.defgrd_ = defgrd_enh;
            }
            solid_material.update(strains.defgrd_, gp, params, ele.id());
          }
        });
  }
  solid_material.update();
}

template <Core::FE::CellType distype>
void Discret::Elements::Shell7pEleCalcEas<distype>::vis_data(
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
template class Discret::Elements::Shell7pEleCalcEas<Core::FE::CellType::quad4>;
template class Discret::Elements::Shell7pEleCalcEas<Core::FE::CellType::quad8>;
template class Discret::Elements::Shell7pEleCalcEas<Core::FE::CellType::quad9>;
template class Discret::Elements::Shell7pEleCalcEas<Core::FE::CellType::tri3>;
template class Discret::Elements::Shell7pEleCalcEas<Core::FE::CellType::tri6>;

FOUR_C_NAMESPACE_CLOSE
