// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_utils_tensor_interpolation.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_utils_fad.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*!
   * @brief Order the eigenpairs of a given matrix w.r.t. the eigenpairs of a reference
   * matrix to yield minimal rotations between corresponding eigenvectors (eigenvalues assumed
   * to already be sorted from highest to lowest in the eigenpairs)
   *
   * @note This ordering procedure is relevant in case of multiple eigenvalues, for which the
   * eigenpairs have to be ordered properly w.r.t. reference eigenpairs
   * For further information, refer to:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for
   * Invertible Second-Order Tensors, Int J Number Methods Eng. 2024, 125, 10.1002/nme.7373,
   * Section 5.1
   *
   * @param[in]  ref_eigenpairs  eigenpairs of the reference matrix
   * @param[in|out]  eigenpairs  eigenpairs to be sorted w.r.t. reference matrix
   */
  void order_eigenpairs_wrt_reference(
      const std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& ref_eigenpairs,
      std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& eigenpairs)
  {
    // auxiliaries
    Core::LinAlg::Matrix<3, 1> temp3x1(Core::LinAlg::Initialization::zero);

    // loop through reference eigenpairs, determine the corresponding eigenpairs
    int tensor_ind;
    Core::LinAlg::Matrix<1, 1> max_scalar_prod;
    Core::LinAlg::Matrix<1, 1> next_scalar_prod;
    for (int i = 0; i < 2; ++i)
    {
      // set the tensor_ind to the current index, before verifying for multiple eigenvalues
      tensor_ind = i;

      // determine the current scalar product of the tensor eigenvector and the reference
      // eigenvector
      max_scalar_prod.multiply_tn(
          1.0 / (eigenpairs[tensor_ind].second.norm2() * ref_eigenpairs[i].second.norm2()),
          eigenpairs[tensor_ind].second, ref_eigenpairs[i].second, 0.0);

      // check for multiple eigenvalues: loop through the next eigenpairs of the considered tensor
      for (int j = i + 1; j < 3; ++j)
      {
        if (std::abs(eigenpairs[j].first - eigenpairs[tensor_ind].first) < 1.0e-8)
        {
          // multiple eigenvalue found
          // now we check whether the absolute value of the scalar product is larger than the
          // current max_scalar_prod

          // compute this scalar product
          next_scalar_prod.multiply_tn(
              1.0 / (eigenpairs[j].second.norm2() * ref_eigenpairs[i].second.norm2()),
              eigenpairs[j].second, ref_eigenpairs[i].second, 0.0);

          // if the absolute value of the current scalar product is larger: set tensor_ind to
          // the current index and update the maximum scalar product
          if (std::abs(next_scalar_prod(0)) > std::abs(max_scalar_prod(0)))
          {
            tensor_ind = j;

            // update maximum scalar product
            max_scalar_prod(0) = next_scalar_prod(0);
          }
        }
      }

      // the correct tensor_ind has been now determined;
      // now set the sign of the according vector so, as to minimize rotation wrt corresponding
      // reference matrix eigenvector
      if (max_scalar_prod(0) < 0.0)
      {
        eigenpairs[tensor_ind].second.update(0.0, temp3x1, -1.0);
      }

      // interchange the EV of i and tensor_ind if they are not the same (no need to interchange the
      // eigenvalues, as these are the same)
      if (tensor_ind != i)
      {
        temp3x1.update(1.0, eigenpairs[i].second, 0.0);        // save EV at index i
        eigenpairs[i].second = eigenpairs[tensor_ind].second;  // interchange EV
        eigenpairs[tensor_ind].second = temp3x1;
      }
    }

    // the last eigenvector is determined based on the first two to yield
    // det(...) = 1 for the corresponding eigenvector matrix
    eigenpairs[2].second(0) = eigenpairs[0].second(1) * eigenpairs[1].second(2) -
                              eigenpairs[0].second(2) * eigenpairs[1].second(1);
    eigenpairs[2].second(1) = eigenpairs[0].second(2) * eigenpairs[1].second(0) -
                              eigenpairs[0].second(0) * eigenpairs[1].second(2);
    eigenpairs[2].second(2) = eigenpairs[0].second(0) * eigenpairs[1].second(1) -
                              eigenpairs[0].second(1) * eigenpairs[1].second(0);
  }


  /*!
   * @brief Align the eigenpairs of the base matrix (nearest to the interpolation point) in case
   *  of multiple eigenvalues
   *
   *  The eigenpairs of the base matrix are reordered in case of multiple eigenvalues to yield
   *  minimal rotations w.r.t. the eigenpairs of the other matrices.
   *  Theoretically, some matrices will be favored in this reordering process, since there are
   *  max. 6 possible ways to reorder the eigenvectors of the base matrix (for a triple
   * eigenvalue). The following criteria determine the reordering result (priority: 1-> highest):
   *  1. Distance of the location point (the matrix whose location lies nearest to the base
   * matrix is favored)
   *  2. Highest eigenvalue (the matrix with the overall highest eigenvalue is favored in the
   * reordering process)
   *
   * @param[in|out]  spectral_pairs  all spectral pairs (eigenvalue, eigenvector) of all
   *                                 available matrices used for interpolation
   * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
   * @param[in]  base_ind  index of the base matrix within spectral_pairs
   */
  template <unsigned int loc_dim>
  void align_eigenpairs_of_base_matrix(
      std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>& spectral_pairs,
      const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs, const unsigned int& base_ind)
  {
    // difference vector between the locations of matrices and the location of the determined base
    // matrix
    Core::LinAlg::Matrix<loc_dim, 1> diff_locs(Core::LinAlg::Initialization::zero);

    // find matrix(or matrices) closest to the basis matrix
    double min_distance = -1.0;
    std::vector<unsigned int> closest_matrix_inds;

    // vector of maximum relevant eigenvalues (for which the corresponding base eigenvalues have a
    // multiplicity higher than 1)
    std::vector<double> max_eigenval(spectral_pairs.size(), 0.0);

    // eigenvalue multiplicity
    unsigned int multp = 0;

    // loop through the eigenvalues of the base matrix
    for (int i = 0; i < 3; ++i)
    {
      // for the first eigenvalues: determine matrix closest to the base matrix and set first
      // eigenvalues as the maximum eigenvalues
      if (i == 0)
      {
        // loop through the matrices
        for (unsigned int m = 0; m < ref_locs.size(); ++m)
        {
          // --> find matrix closest to the base matrix (based on location)

          // skip base matrix
          if (m == base_ind) continue;

          // compute difference vector
          diff_locs.update(1.0, ref_locs[m], -1.0, ref_locs[base_ind], 0.0);

          // first matrix (excluding the base matrix) is added as closest in the beginning
          if (min_distance < 0)
          {
            min_distance = diff_locs.norm2();
            closest_matrix_inds.push_back(m);
            continue;
          }

          // new closest matrix found
          if (min_distance > diff_locs.norm2())
          {
            min_distance = diff_locs.norm2();
            closest_matrix_inds.clear();
            closest_matrix_inds.push_back(m);
          }

          // another matrix found with the same minimum distance
          else if (std::abs(min_distance - diff_locs.norm2()) <
                   1.0e-10 * (std::abs(min_distance) + std::abs(diff_locs.norm2())))
          {
            closest_matrix_inds.push_back(m);
          }

          // --> set first eigenvalues as the maximum eigenvalues initially
          max_eigenval[m] = spectral_pairs[m][0].first;
        }
        multp = 1;
        continue;
      }

      // if the current eigenvalue corresponds to the last one
      if (std::abs(spectral_pairs[base_ind][i].first - spectral_pairs[base_ind][i - 1].first) <
          1.0e-8)
      {
        // increment multiplicity
        multp += 1;

        // for each of the matrices: get maximum eigenvalue, accounting also for the currently saved
        // value
        std::transform(spectral_pairs.begin(), spectral_pairs.end(), max_eigenval.begin(),
            max_eigenval.begin(),
            [i](const std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& arr,
                double curr_val) { return std::max(arr[i].first, curr_val); });
        continue;
      }

      // if we encounter a new eigenvalue
      if (multp == 1)
      {
        // set maximum eigenvalues to the current eigenvalues
        std::transform(spectral_pairs.begin(), spectral_pairs.end(), max_eigenval.begin(),
            [i](const std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& arr)
            { return arr[i].first; });
        continue;
      }
      // else we are finished, there are maximum three eigenvalues
      break;
    }

    // for multiple eigenvalues: reorder the eigenpairs of the base matrix with respect to the
    // favored matrix
    if (multp > 1)
    {
      // index of the favored matrix
      unsigned int ind_fav_matrix;

      // vector containing the eigenvalues of the closest matrices (w.r.t. base matrix)
      std::vector<unsigned int> eigenval_closest_matrix_inds(closest_matrix_inds.size());
      std::transform(closest_matrix_inds.begin(), closest_matrix_inds.end(),
          eigenval_closest_matrix_inds.begin(),
          [max_eigenval](int index) { return max_eigenval[index]; });

      // get index of the favored matrix
      ind_fav_matrix = closest_matrix_inds[std::distance(eigenval_closest_matrix_inds.begin(),
          std::max_element(
              eigenval_closest_matrix_inds.begin(), eigenval_closest_matrix_inds.end()))];

      // align eigenpairs of the base matrix with its reference (favored matrix)
      order_eigenpairs_wrt_reference(spectral_pairs[ind_fav_matrix], spectral_pairs[base_ind]);
    }
  }
}  // namespace


template <unsigned int loc_dim>
Core::LinAlg::Matrix<3, 3>
Core::LinAlg::SecondOrderTensorInterpolator<loc_dim>::get_interpolated_matrix(
    const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
    const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
    const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc)
{
  // declare output variable
  Core::LinAlg::Matrix<3, 3> output(Core::LinAlg::Initialization::zero);

  // assert if the number of input matrices does not match the number of input locations
  FOUR_C_ASSERT(ref_matrices.size() == ref_locs.size(),
      "The number of given reference matrices does not match the number of given reference "
      "locations");

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<3, 1> temp3x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<loc_dim, 1> diff_locs(Core::LinAlg::Initialization::zero);

  // interpolation setting: exponential decay factor of the weighting function in Satheesh,
  // 2024, 10.1002/nme.7373, Eq. (21)
  double c = 10.0;

  // index of the base matrix, which is the matrix located nearest to the
  // interpolation location
  unsigned int base_ind = 0;
  // determine the base index
  diff_locs.update(1.0, ref_locs[0], -1.0, interp_loc, 0.0);
  double dist_to_interp_loc = diff_locs.norm2();
  for (unsigned int i = 1; i < ref_locs.size(); ++i)
  {
    // get difference of the current reference location w.r.t. interpolation location
    diff_locs.update(1.0, ref_locs[i], -1.0, interp_loc, 0.0);
    // check if the determined distance is smaller than the current minimum
    if (diff_locs.norm2() < dist_to_interp_loc)
    {
      // update index and distance
      base_ind = i;
      dist_to_interp_loc = diff_locs.norm2();
    }
  }

  // --> declare vectors of variables be updated while looping through reference matrices

  // all \f$ \boldsymbol{U}_j \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_U(ref_matrices.size());

  // all \f$ \boldsymbol{R}_j \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_R(ref_matrices.size());

  // all \f$ \boldsymbol{Q}_j \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_Q(ref_matrices.size());

  // all \f$ \boldsymbol{R}_j^{\text{r}} \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_R_rel(ref_matrices.size());

  // all \f$ \boldsymbol{Q}_j^{\text{r}} \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_Q_rel(ref_matrices.size());

  // all \f$ \boldsymbol{\theta}_j^{\text{r}} \f$ associated to \f$ \boldsymbol{R}_j^{\text{r}} \f$
  std::vector<Core::LinAlg::Matrix<3, 1>> all_rot_vect_R_rel(ref_matrices.size());

  // all \f$ \boldsymbol{\theta}_j^{\text{r}} \f$ associated to \f$ \boldsymbol{Q}_j^{\text{r}} \f$
  std::vector<Core::LinAlg::Matrix<3, 1>> all_rot_vect_Q_rel(ref_matrices.size());

  // all \f$ \boldsymbol{\lambda}_j \f$
  std::vector<Core::LinAlg::Matrix<3, 3>> all_lambda(ref_matrices.size());

  // all spectral pairs of eigenvalue, eigenvector of all matrices
  std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>> all_spectral_pairs(
      ref_matrices.size());

  // all unnormalized weights \f$ w_j  \f$
  std::vector<double> all_unnorm_weights(ref_matrices.size());
  double sum_of_unnorm_weights = 0.0;

  // all normalized weights \f$ \tilde{w}_j  \f$
  std::vector<double> all_norm_weights(ref_matrices.size());

  // first loop through the reference matrices
  for (unsigned int i = 0; i < ref_locs.size(); ++i)
  {
    // STEP 1 (Satheesh, 2024, 10.1002/nme.7373, Section 2.5): polar decomposition of each matrix
    matrix_3x3_polar_decomposition(
        ref_matrices[i], all_R[i], all_U[i], all_lambda[i], all_spectral_pairs[i]);

    // STEP 3 (Satheesh, 2024, 10.1002/nme.7373, Section 2.5): interpolate rotation matrices
    // get relative rotation matrices
    all_R_rel[i].multiply_tn(1.0, all_R[base_ind], all_R[i], 0.0);

    // compute relative rotation vectors from the relative rotation matrices
    all_rot_vect_R_rel[i] = calc_rot_vect_from_rot_matrix(all_R_rel[i]);

    // compute unnormalized weights of interpolation points
    diff_locs.update(1.0, ref_locs[i], -1.0, interp_loc, 0.0);
    all_unnorm_weights[i] = std::exp(-c * diff_locs.norm2());
    sum_of_unnorm_weights += all_unnorm_weights[i];
  }

  // align eigenpairs of the base matrix in case of multiple eigenvalues
  align_eigenpairs_of_base_matrix(all_spectral_pairs, ref_locs, base_ind);
  // also adapt \f$ Q \f$ of the base matrix to the new ordering
  for (unsigned int r = 0; r < 3; ++r)
  {
    all_Q[base_ind](r, 0) = all_spectral_pairs[base_ind][r].second(0);
    all_Q[base_ind](r, 1) = all_spectral_pairs[base_ind][r].second(1);
    all_Q[base_ind](r, 2) = all_spectral_pairs[base_ind][r].second(2);
  }

  // get relative rotation matrix of base_ind
  all_Q_rel[base_ind].multiply_tn(1.0, all_Q[base_ind], all_Q[base_ind], 0.0);  // unit tensor

  // compute relative rotation vector from the relative rotation matrix
  all_rot_vect_Q_rel[base_ind] = calc_rot_vect_from_rot_matrix(all_Q_rel[base_ind]);  // zero vector

  // get \f$ \boldsymbol{Q} \f$, normalized weights and build \f$ \boldsymbol{P} \f$ matrix along
  // with \f$ \boldsymbol{b}^i
  // \f$ vectors, as shown in Satheesh, 2024, 10.1002/nme.7373, Eq. (22) and Eq. (37)

  // \f$ \boldsymbol{p}_j \f$: p vector of monomial values
  Core::LinAlg::SerialDenseVector p_vec(polynomial_space_.size());
  polynomial_space_.evaluate(interp_loc, p_vec);

  // length of p vector
  unsigned int m = p_vec.length();
  FOUR_C_ASSERT(m <= ref_locs.size(),
      "The number of reference matrices is too small for the desired interpolation scheme!");

  // interpolation matrix of monomial values \f$ \bm{P} \f$
  Core::LinAlg::SerialDenseMatrix P(m, m, true);

  // \f$ \bm{b}^i \f$ for Q matrices
  Core::LinAlg::SerialDenseMatrix b_Q(m, 3, true);

  // \f$ \bm{b}^i \f$ for R matrices
  Core::LinAlg::SerialDenseMatrix b_R(m, 3, true);

  // \f$ \log{\bm{\lambda}_{\text{x}_p}} \f$
  Core::LinAlg::Matrix<3, 1> ln_lambda_interp(Core::LinAlg::Initialization::zero);

  // loop once more over all matrices with the updated info
  for (unsigned int i = 0; i < ref_locs.size(); ++i)
  {
    if (i != base_ind)
    {
      // order eigenpairs to give minimal eigenvector rotation w.r.t. base matrix
      order_eigenpairs_wrt_reference(all_spectral_pairs[base_ind], all_spectral_pairs[i]);

      // STEP 2 (Satheesh, 2024, 10.1002/nme.7373, Section 2.5): get rotation matrices, relative
      // rotation matrices and relative rotation vectors rotation tensors
      for (int r = 0; r < 3; ++r)
      {
        all_Q[i](r, 0) = all_spectral_pairs[i][r].second(0);
        all_Q[i](r, 1) = all_spectral_pairs[i][r].second(1);
        all_Q[i](r, 2) = all_spectral_pairs[i][r].second(2);
      }
      // get relative rotation matrices
      all_Q_rel[i].multiply_tn(1.0, all_Q[base_ind], all_Q[i], 0.0);

      // compute relative rotation vectors from the relative rotation matrices
      all_rot_vect_Q_rel[i] = calc_rot_vect_from_rot_matrix(all_Q_rel[i]);
    }

    // compute normalized weights
    all_norm_weights[i] = all_unnorm_weights[i] / sum_of_unnorm_weights;

    // compute vector and conversion to matrix
    polynomial_space_.evaluate(ref_locs[i], p_vec);

    // P matrix
    P.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, all_norm_weights[i], p_vec, p_vec, 1.0);

    // RHS of the rotation interpolation
    b_Q.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, all_norm_weights[i], p_vec,
        Core::LinAlg::SerialDenseVector(Teuchos::Copy, all_rot_vect_Q_rel[i].data(), 3), 1.0);
    b_R.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, all_norm_weights[i], p_vec,
        Core::LinAlg::SerialDenseVector(Teuchos::Copy, all_rot_vect_R_rel[i].data(), 3), 1.0);

    // compute contribution to the natural logarithm of the interpolated eigenvalues (logarithmic
    // weighted average approach in Satheesh 2024, Section 2.4.1)
    ln_lambda_interp(0) += all_norm_weights[i] * std::log(all_spectral_pairs[i][0].first);
    ln_lambda_interp(1) += all_norm_weights[i] * std::log(all_spectral_pairs[i][1].first);
    ln_lambda_interp(2) += all_norm_weights[i] * std::log(all_spectral_pairs[i][2].first);
  }

  // declare all monomial coefficients
  Core::LinAlg::SerialDenseMatrix a_Q(m, 3, true);
  Core::LinAlg::SerialDenseMatrix a_R(m, 3, true);

  // setup solver
  Core::LinAlg::SerialDenseMatrix copy_P = Core::LinAlg::SerialDenseMatrix(P);
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> solver;

  // solve for the coefficients of Q
  solver.setMatrix(Teuchos::rcpFromRef(P));
  solver.setVectors(Teuchos::rcpFromRef(a_Q), Teuchos::rcpFromRef(b_Q));
  solver.factorWithEquilibration(true);
  solver.solveToRefinedSolution(true);
  if (solver.factor() or solver.solve())
    FOUR_C_THROW(
        "Solution of linear system of equations during second-order tensor interpolation failed "
        "(for rotation matrix Q)!");

  // solve for the coefficients of R
  solver.setMatrix(Teuchos::rcpFromRef(copy_P));
  solver.setVectors(Teuchos::rcpFromRef(a_R), Teuchos::rcpFromRef(b_R));
  solver.factorWithEquilibration(true);
  solver.solveToRefinedSolution(true);
  if (solver.factor() or solver.solve())
    FOUR_C_THROW(
        "Solution of linear system of equations during second-order tensor interpolation failed "
        "(for rotation matrix R)!");


  // get relative rotation vectors at interpolation points...
  Core::LinAlg::SerialDenseVector rot_serial_dense_vec(3, true);

  // evaluate polynomial space of the interpolation location
  polynomial_space_.evaluate(interp_loc, p_vec);

  // ...for Q
  rot_serial_dense_vec.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, a_Q, p_vec, 0.0);
  Core::LinAlg::Matrix<3, 1> rot_vect_Q_rel_interp(Core::LinAlg::Initialization::zero);
  rot_vect_Q_rel_interp(0) = rot_serial_dense_vec(0);
  rot_vect_Q_rel_interp(1) = rot_serial_dense_vec(1);
  rot_vect_Q_rel_interp(2) = rot_serial_dense_vec(2);

  // ...for R
  rot_serial_dense_vec.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, a_R, p_vec, 0.0);
  Core::LinAlg::Matrix<3, 1> rot_vect_R_rel_interp(Core::LinAlg::Initialization::zero);
  rot_vect_R_rel_interp(0) = rot_serial_dense_vec(0);
  rot_vect_R_rel_interp(1) = rot_serial_dense_vec(1);
  rot_vect_R_rel_interp(2) = rot_serial_dense_vec(2);

  // get corresponding relative rotation matrices
  Core::LinAlg::Matrix<3, 3> Q_rel_interp = calc_rot_matrix_from_rot_vect(rot_vect_Q_rel_interp);
  Core::LinAlg::Matrix<3, 3> R_rel_interp = calc_rot_matrix_from_rot_vect(rot_vect_R_rel_interp);

  // compute interpolated rotation tensors (absolute)
  Core::LinAlg::Matrix<3, 3> Q_interp(Core::LinAlg::Initialization::zero);
  Q_interp.multiply_nn(1.0, all_Q[base_ind], Q_rel_interp, 0.0);
  Core::LinAlg::Matrix<3, 3> R_interp(Core::LinAlg::Initialization::zero);
  R_interp.multiply_nn(1.0, all_R[base_ind], R_rel_interp, 0.0);

  // STEP 4 (Satheesh, 2024, 10.1002/nme.7373, Section 2.5): interpolate eigenvalue tensor
  // we use the logarithmic weighted average to ensure a positive-definite lambda matrix
  Core::LinAlg::Matrix<3, 3> lambda_interp(Core::LinAlg::Initialization::zero);
  lambda_interp(0, 0) = std::exp(ln_lambda_interp(0));
  lambda_interp(1, 1) = std::exp(ln_lambda_interp(1));
  lambda_interp(2, 2) = std::exp(ln_lambda_interp(2));


  // Finally: build the output interpolation tensor from its interpolated components
  output.multiply_nn(1.0, lambda_interp, Q_interp, 0.0);
  temp3x3.multiply_tn(1.0, Q_interp, output, 0.0);
  output.multiply_nn(1.0, R_interp, temp3x3, 0.0);

  return output;
}


void Core::LinAlg::matrix_3x3_polar_decomposition(const Core::LinAlg::Matrix<3, 3>& inp_matrix,
    Core::LinAlg::Matrix<3, 3>& R_matrix, Core::LinAlg::Matrix<3, 3>& U_matrix,
    Core::LinAlg::Matrix<3, 3>& eigenval_matrix,
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& spectral_pairs)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> temp3x1(Core::LinAlg::Initialization::zero);

  // compute squared stretch tensors U
  Core::LinAlg::Matrix<3, 3> U_squared(Core::LinAlg::Initialization::zero);
  U_squared.multiply_tn(1.0, inp_matrix, inp_matrix, 0.0);

  // decompose squared stretch tensors U
  Core::LinAlg::Matrix<3, 3> eigenvectors_U(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> eigenvalues_U(Core::LinAlg::Initialization::zero);
  temp3x3.clear();
  Core::LinAlg::syev(U_squared, temp3x3, eigenvectors_U);
  eigenvalues_U(0, 0) = std::sqrt(temp3x3(0, 0));
  eigenvalues_U(1, 1) = std::sqrt(temp3x3(1, 1));
  eigenvalues_U(2, 2) = std::sqrt(temp3x3(2, 2));
  // scale eigenvectors matrix to yield determinant 1
  eigenvectors_U.scale(Core::FADUtils::signum(eigenvectors_U.determinant()) * 1.0 /
                       std::pow(std::abs(eigenvectors_U.determinant()), 1.0 / 3.0));

  // compute resulting stretch and rotation tensors, along with the eigenvalue matrix
  temp3x3.multiply_nn(1.0, eigenvectors_U, eigenvalues_U, 0.0);
  U_matrix.multiply_nt(1.0, temp3x3, eigenvectors_U, 0.0);
  temp3x3.invert(U_matrix);
  R_matrix.multiply_nn(1.0, inp_matrix, temp3x3, 0.0);
  eigenval_matrix = eigenvalues_U;

  // sort the eigenpairs in descending order of eigenvalues: from the largest to the smallest
  // eigenvalue
  for (int i = 2; i >= 0; --i)
  {
    // determine spectral pairs
    temp3x1.clear();
    temp3x1(0) = eigenvectors_U(0, i);
    temp3x1(1) = eigenvectors_U(1, i);
    temp3x1(2) = eigenvectors_U(2, i);
    spectral_pairs[2 - i] = std::make_pair(eigenvalues_U(i, i), temp3x1);
  }
  // correct signs to ensure det(...) = 1
  spectral_pairs[2].second(0) = spectral_pairs[0].second(1) * spectral_pairs[1].second(2) -
                                spectral_pairs[0].second(2) * spectral_pairs[1].second(1);
  spectral_pairs[2].second(1) = spectral_pairs[0].second(2) * spectral_pairs[1].second(0) -
                                spectral_pairs[0].second(0) * spectral_pairs[1].second(2);
  spectral_pairs[2].second(2) = spectral_pairs[0].second(0) * spectral_pairs[1].second(1) -
                                spectral_pairs[0].second(1) * spectral_pairs[1].second(0);
}


Core::LinAlg::Matrix<3, 1> Core::LinAlg::calc_rot_vect_from_rot_matrix(
    const Core::LinAlg::Matrix<3, 3>& rot_matrix)
{
  // declare output rotation vector
  Core::LinAlg::Matrix<3, 1> rot_vect(Core::LinAlg::Initialization::zero);

  // --> interpolate quaternion from rotation matrix using Spurrier's algorithm

  // utilities
  Core::LinAlg::Matrix<4, 1> tensor_characteristics(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> quaternion(Core::LinAlg::Initialization::zero);
  double max_value = 0.0;

  // compute tensor characteristics for the algorithm
  tensor_characteristics(0) = rot_matrix(0, 0) + rot_matrix(1, 1) + rot_matrix(2, 2);
  tensor_characteristics(1) = rot_matrix(0, 0);
  tensor_characteristics(2) = rot_matrix(1, 1);
  tensor_characteristics(3) = rot_matrix(2, 2);

  // get the max. value of the computed tensor characteristics
  max_value = tensor_characteristics.max_value();

  // get corresponding quaternion based on the max. value
  if (max_value == tensor_characteristics(0))
  {
    quaternion(0) = 1.0 / 2.0 * std::sqrt(1.0 + max_value);
    quaternion(1) = 1.0 / 4.0 * (rot_matrix(2, 1) - rot_matrix(1, 2)) / quaternion(0);
    quaternion(2) = 1.0 / 4.0 * (rot_matrix(0, 2) - rot_matrix(2, 0)) / quaternion(0);
    quaternion(3) = 1.0 / 4.0 * (rot_matrix(1, 0) - rot_matrix(0, 1)) / quaternion(0);
  }
  else if (max_value == tensor_characteristics(1))
  {
    quaternion(1) =
        std::sqrt(1.0 / 2.0 * rot_matrix(0, 0) + 1.0 / 4.0 * (1.0 - tensor_characteristics(0)));

    quaternion(0) = 1.0 / 4.0 * (rot_matrix(2, 1) - rot_matrix(1, 2)) / quaternion(1);

    quaternion(2) = 1.0 / 4.0 * (rot_matrix(1, 0) + rot_matrix(0, 1)) / quaternion(1);
    quaternion(3) = 1.0 / 4.0 * (rot_matrix(2, 0) + rot_matrix(0, 2)) / quaternion(1);
  }
  else if (max_value == tensor_characteristics(2))
  {
    quaternion(2) =
        std::sqrt(1.0 / 2.0 * rot_matrix(1, 1) + 1.0 / 4.0 * (1.0 - tensor_characteristics(0)));

    quaternion(0) = 1.0 / 4.0 * (rot_matrix(0, 2) - rot_matrix(2, 0)) / quaternion(2);

    quaternion(1) = 1.0 / 4.0 * (rot_matrix(0, 1) + rot_matrix(1, 0)) / quaternion(2);
    quaternion(3) = 1.0 / 4.0 * (rot_matrix(2, 1) + rot_matrix(1, 2)) / quaternion(2);
  }
  else if (max_value == tensor_characteristics(3))
  {
    quaternion(3) =
        std::sqrt(1.0 / 2.0 * rot_matrix(2, 2) + 1.0 / 4.0 * (1.0 - tensor_characteristics(0)));

    quaternion(0) = 1.0 / 4.0 * (rot_matrix(1, 0) - rot_matrix(0, 1)) / quaternion(3);

    quaternion(1) = 1.0 / 4.0 * (rot_matrix(0, 2) + rot_matrix(2, 0)) / quaternion(3);
    quaternion(2) = 1.0 / 4.0 * (rot_matrix(1, 2) + rot_matrix(2, 1)) / quaternion(3);
  }
  // compute unit quaternion
  Core::LinAlg::Matrix<4, 1> unit_quaternion(Core::LinAlg::Initialization::zero);
  unit_quaternion.update(1.0 / quaternion.norm2(), quaternion, 0.0);

  // extract rotation vector
  rot_vect.clear();
  if (1.0 - unit_quaternion(0) > 1.0e-16)  // otherwise rot_vect = 0
  {
    double rot_theta = 2.0 * std::acos(unit_quaternion(0));
    rot_vect(0) = rot_theta / std::sin(rot_theta / 2.0) * unit_quaternion(1);
    rot_vect(1) = rot_theta / std::sin(rot_theta / 2.0) * unit_quaternion(2);
    rot_vect(2) = rot_theta / std::sin(rot_theta / 2.0) * unit_quaternion(3);
  }

  return rot_vect;
}

Core::LinAlg::Matrix<3, 3> Core::LinAlg::calc_rot_matrix_from_rot_vect(
    const Core::LinAlg::Matrix<3, 1>& rot_vect)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;

  // declare output matrix
  Core::LinAlg::Matrix<3, 3> rot_matrix(Core::LinAlg::Initialization::zero);

  // add unit tensor to the rotation tensor
  rot_matrix.update(1.0, id3x3, 0.0);

  // get angle
  double angle = rot_vect.norm2();

  // check whether angle is larger than 0.0 -> in that case compute the further terms for the
  // rotation tensor
  if (angle > 1.0e-16)
  {
    // get normalized rotation vector
    Core::LinAlg::Matrix<3, 1> norm_rot_vect(Core::LinAlg::Initialization::zero);
    norm_rot_vect.update(1.0 / angle, rot_vect, 0.0);

    // compute S matrix with components from the normalized rotation vector
    Core::LinAlg::Matrix<3, 3> S(Core::LinAlg::Initialization::zero);
    S(0, 1) = -norm_rot_vect(2);
    S(1, 0) = -S(0, 1);
    S(0, 2) = norm_rot_vect(1);
    S(2, 0) = -S(0, 2);
    S(1, 2) = -norm_rot_vect(0);
    S(2, 1) = -S(1, 2);

    // update rotation matrix
    rot_matrix.update(std::sin(angle), S, 1.0);

    // \f$ \bm{S} \bm{S} \f$
    Core::LinAlg::Matrix<3, 3> SS(Core::LinAlg::Initialization::zero);
    SS.multiply_nn(1.0, S, S, 0.0);

    // update rotation matrix
    rot_matrix.update(1.0 - std::cos(angle), SS, 1.0);
  }

  return rot_matrix;
}

template <>
Core::LinAlg::Matrix<3, 3> Core::LinAlg::SecondOrderTensorInterpolator<1>::get_interpolated_matrix(
    const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
    const std::vector<double>& ref_locs, const double interp_loc)
{
  // auxiliaries
  Core::LinAlg::Matrix<1, 1> temp_matrix;

  // pack reference locations to matrices
  std::vector<Core::LinAlg::Matrix<1, 1>> converted_ref_locs;
  for (auto ref_loc : ref_locs)
  {
    // save scalar to 1x1 matrix
    temp_matrix(0) = ref_loc;

    // add 1x1 matrix to its corresponding vector
    converted_ref_locs.push_back(temp_matrix);
  }

  // pack interpolation location to matrix
  Core::LinAlg::Matrix<1, 1> converted_interp_loc(Core::LinAlg::Initialization::zero);
  converted_interp_loc(0) = interp_loc;

  // call the general constructor with the matrix expressions
  return get_interpolated_matrix(ref_matrices, converted_ref_locs, converted_interp_loc);
}


// explicit instantiation of template functions
template class Core::LinAlg::SecondOrderTensorInterpolator<1>;
template class Core::LinAlg::SecondOrderTensorInterpolator<2>;
template class Core::LinAlg::SecondOrderTensorInterpolator<3>;

FOUR_C_NAMESPACE_CLOSE
