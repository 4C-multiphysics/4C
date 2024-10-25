// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FIXEDSIZEMATRIX_VOIGT_NOTATION_HPP
#define FOUR_C_LINALG_FIXEDSIZEMATRIX_VOIGT_NOTATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_four_tensor.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg::Voigt
{
  /*! Voigt notation types
   *
   * This enum can be used whenever the distinction between stress- and strain-like Voigt notation
   * is important. The classical Voigt notation is only meaningful for symmetric tensors.
   */
  enum class NotationType
  {
    /// stress-like Voigt notation
    ///
    /// A vector in stress-like Voigt notation, contains the off-diagonal values only once.
    stress,
    /// strain-like Voigt notation
    ///
    /// A vector in strain-like Voigt notation, contains the sum of corresponding off-diagonal
    /// values.
    strain
  };

  /*--------------------------------------------------------------------------*/
  /** \brief Utility routines for the perturbed Voigt tensor notation
   *
   * \tparam type specific NotationType this class operates on
   */
  template <NotationType type>
  class VoigtUtils
  {
   public:
    /// instantiation is forbidden
    VoigtUtils() = delete;

    /** \brief compute power of a symmetric 3x3 matrix in perturbed Voigt notation
     *
     *  \f[
     *  [vtensorpow]_{AB} = [vtensor^{pow}]_{AB}
     *  \f]
     *
     *  \param[in] pow          positive integer exponent
     *  \param[in] vtensor      input tensor in Voigt <type> notation
     *  \param[in] vtensor_pow  result, i.e. input tensor to the given power
     */
    static void power_of_symmetric_tensor(unsigned pow, const Core::LinAlg::Matrix<6, 1>& strain,
        Core::LinAlg::Matrix<6, 1>& strain_pow);

    /** \brief Compute the inverse tensor in perturbed Voigt notation
     *
     *  \param[in]  vtensor      tensor in Voigt <type> notation
     *  \param[out] vtensor_inv  inverse tensor in Voigt <type> notation
     */
    static void inverse_tensor(
        const Core::LinAlg::Matrix<6, 1>& tens, Core::LinAlg::Matrix<6, 1>& tens_inv);

    /**
     * \brief Compute the determinant of a matrix in Voigt <type> notation
     *
     * @param vtensor Tensor in Voigt <type> notation
     */
    static inline double determinant(const Core::LinAlg::Matrix<6, 1>& vtensor)
    {
      return triple_entry_product(vtensor, 0, 1, 2) + 2 * triple_entry_product(vtensor, 3, 4, 5) -
             triple_entry_product(vtensor, 1, 5, 5) - triple_entry_product(vtensor, 2, 3, 3) -
             triple_entry_product(vtensor, 0, 4, 4);
    }

    /** \brief Compute the three principal invariants of a matrix in Voigt <type> notation
     *
     * @param[out] prinv the three principal invariants
     * @param vtensor tensor in Voigt <type> notation
     */
    static inline void invariants_principal(
        Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<6, 1>& vtensor)
    {
      // 1st invariant, trace tens
      prinv(0) = vtensor(0) + vtensor(1) + vtensor(2);
      // 2nd invariant, 0.5( (trace(tens))^2 - trace(tens^2))
      prinv(1) = 0.5 * (prinv(0) * prinv(0) - vtensor(0) * vtensor(0) - vtensor(1) * vtensor(1) -
                           vtensor(2) * vtensor(2)) -
                 vtensor(3) * vtensor(3) * unscale_factor(3) * unscale_factor(3) -
                 vtensor(4) * vtensor(4) * unscale_factor(4) * unscale_factor(4) -
                 vtensor(5) * vtensor(5) * unscale_factor(5) * unscale_factor(5);
      // 3rd invariant, determinant tens
      prinv(2) = determinant(vtensor);
    }

    /** \brief Compute the product of a tensor in perturbed Voigt notation
     *  and a vector
     *
     *  \f$ [vecres]_{A} = vtensor_{AB} vec^{B} \f$
     *
     *  \param[in]  vtensor      tensor in Voigt <type> notation
     *  \param[out] vtensor_inv  inverser tensor in Voigt <type> notation
     */
    static void multiply_tensor_vector(const Core::LinAlg::Matrix<6, 1>& strain,
        const Core::LinAlg::Matrix<3, 1>& vec, Core::LinAlg::Matrix<3, 1>& vec_res);

    /** \brief Compute the symmetric outer product of two vectors
     *
     *  \f$ [abba]_{AB} = [veca]_{A} [vecb]_{B} + [veca]_{B} [vecb]_{A} \f$
     *
     *  \param[in]  vec_a  first vector
     *  \param[in]  vec_b  second vector
     *  \param[out] ab_ba  symmetric outer product of the two input vectors
     *                     in the Voigt <type> notation
     */
    static void symmetric_outer_product(const Core::LinAlg::Matrix<3, 1>& vec_a,
        const Core::LinAlg::Matrix<3, 1>& vec_b, Core::LinAlg::Matrix<6, 1>& ab_ba);

    /*!
     * Converts a <type>-like tensor to stress-like Voigt notation
     * @param vtensor_in tensor in <type>-like Voigt notation
     * @param vtensor_out tensor in stress-like Voigt notation
     */
    static void to_stress_like(
        const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<6, 1>& vtensor_out);

    /*!
     * Converts a <type>-like tensor to strain-like Voigt notation
     * @param vtensor_in tensor in <type>-like Voigt notation
     * @param vtensor_out tensor in strain-like Voigt notation
     */
    static void to_strain_like(
        const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<6, 1>& vtensor_out);

    /*!
     * Converts a <type>-like tensor in Voigt notation to a matrix
     * @param vtensor_in tensor in <type>-like Voigt notation
     * @param tensor_out tensor as a matrix
     */
    static void vector_to_matrix(
        const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<3, 3>& tensor_out);

    /*! Copy matrix contents to type-like Voigt notation
     *
     * Matrix [A_00 A_01 A_02; A_10 A_11 A_12; A_20 A_21 A_22] is converted to the Voigt vector
     * [A_00; A_11; A_22; 0.5*k*(A_01 + A_10); 0.5*k*(A_12 + A_21); 0.5*k*(A_02 + A_20)]
     *
     * where k is the scale factor for type-like Voigt notation.
     *
     * @param tensor_in the matrix to copy from
     * @param[out] vtensor_out target tensor in <type>-like Voigt notation
     */
    template <typename T>
    static void matrix_to_vector(
        const Core::LinAlg::Matrix<3, 3, T>& tensor_in, Core::LinAlg::Matrix<6, 1, T>& vtensor_out);

    /// access scaling factors
    static inline double scale_factor(unsigned i) { return scale_fac_[i]; };

    /// access unscaling factors
    static inline double unscale_factor(unsigned i) { return unscale_fac_[i]; };

   private:
    /** \brief scale off diagonal values
     *
     *  \note This function changes the values only if the strain notation is used.
     *
     *  \param[out] tensor  scale the off-diagonal values of this tensor
     */
    static void scale_off_diagonal_vals(Core::LinAlg::Matrix<6, 1>& strain);

    /** \brief unscale off diagonal values
     *
     *  \note This function changes the values only if the strain notation is used.
     *
     *  \param[out] tensor  unscale the off-diagonal values of this tensor
     */
    static void unscale_off_diagonal_vals(Core::LinAlg::Matrix<6, 1>& strain);


    /** \brief unscale factors for the perturbed Voigt strain notation
     *
     *  These factors are meaningful if the strain convention is followed. */
    static constexpr std::array<double, 6> unscale_fac_ =
        type == NotationType::strain ? std::array{1.0, 1.0, 1.0, 0.5, 0.5, 0.5}
                                     : std::array{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    /** \brief scale factors for the perturbed Voigt stress notation
     *
     *  These factors are meaningful if the strain convention is followed. */
    static constexpr std::array<double, 6> scale_fac_ =
        type == NotationType::strain ? std::array{1.0, 1.0, 1.0, 2.0, 2.0, 2.0}
                                     : std::array{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    /**
     * \brief Compute the product of three matrix entries
     *
     * The entries are correctly scaled depending on the VoigtNotation type of the tensor.
     * @param vtensor the tensor in voigt notaion
     * @param i first entry's Voigt index
     * @param j second entry's Voigt inde
     * @param k third entry's Voigt index
     * @return product of the three entries
     */
    static inline double triple_entry_product(
        const Core::LinAlg::Matrix<6, 1>& vtensor, unsigned i, unsigned j, unsigned k)
    {
      return vtensor(i) * unscale_factor(i) * vtensor(j) * unscale_factor(j) * vtensor(k) *
             unscale_factor(k);
    }
  };


  /// convert non-symmetric 2-tensor to 9x1 vector
  ///          A_00 A_01 A_02
  /// Matrix   A_10 A_11 A_12  is converted to the vector below:
  ///          A_20 A_21 A_22
  ///
  /// Vector   V_0 = A_00; V_1 = A_11; V_2 = A_22; V_3 = A_01; V_4 = A_12; V_5 = A_02; V_6 = A_10;
  /// V_7 = A_21; V_8 = A_20
  void matrix_3x3_to_9x1(Core::LinAlg::Matrix<3, 3> const& in, Core::LinAlg::Matrix<9, 1>& out);

  /// convert 9x1 vector to non-symmetric 2-tensor
  ///
  /// Vector   V_0 = A_00; V_1 = A_11; V_2 = A_22; V_3 = A_01; V_4 = A_12; V_5 = A_02; V_6 = A_10;
  /// V_7 = A_21; V_8 = A_20
  ///
  /// is converted to
  ///
  ///          A_00 A_01 A_02
  /// Matrix   A_10 A_11 A_12
  ///          A_20 A_21 A_22
  void matrix_9x1_to_3x3(Core::LinAlg::Matrix<9, 1> const& in, Core::LinAlg::Matrix<3, 3>& out);

  /*!
   * @brief Setup 4th order tensor from 6x6 Voigt notation
   *
   * @note Setup 4th order tensor from 6x6 Voigt matrix (which has to be the representative of a 4
   * tensor with at least minor symmetries)
   *
   * @param[out] four_tensor   4th order tensor that is set up based on matrix_voigt
   * @param[in]  matrix_voigt  4th order tensor in Voigt notation with at least minor symmetries
   */
  template <int dim>
  void setup_four_tensor_from_6x6_voigt_matrix(
      Core::LinAlg::FourTensor<dim>& four_tensor, const Core::LinAlg::Matrix<6, 6>& matrix_voigt);

  /*!
   * @brief Setup 4th order tensor from 6x9 Voigt notation
   *
   *
   * @param[out] fourTensor   4th order tensor that is set up based on matrixVoigt
   * @param[in]  matrixVoigt  4th order tensor in Voigt notation with 1st minor symmetry
   */
  template <int dim>
  void setup_four_tensor_from_6x9_voigt_matrix(
      Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<6, 9>& matrixVoigt);

  /*!
   * @brief Setup 4th order tensor from 9x6 Voigt notation
   *
   *
   * @param[out] fourTensor   4th order tensor that is set up based on matrixVoigt
   * @param[in]  matrixVoigt  4th order tensor in Voigt notation with 2nd minor symmetry
   */
  template <int dim>
  void setup_four_tensor_from_9x6_voigt_matrix(
      Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<9, 6>& matrixVoigt);

  /*!
   * @brief Setup 4th order tensor from 9x9 Voigt notation
   *
   *
   * @param[out] fourTensor   4th order tensor that is set up based on matrixVoigt
   * @param[in]  matrixVoigt  4th order tensor in Voigt notation without any symmetries
   */
  template <int dim>
  void setup_four_tensor_from_9x9_voigt_matrix(
      Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<9, 9>& matrixVoigt);

  /*!
   * @brief Setup 6x6 Voigt matrix from 4th order tensor with minor symmetries
   *
   * @param[out] matrix_voigt  6x6 Voigt matrix that is set up based on four_tensor
   * @param[in]  four_tensor   4th order tensor with minor symmetries
   */
  template <int dim>
  void setup_6x6_voigt_matrix_from_four_tensor(
      Core::LinAlg::Matrix<6, 6>& matrix_voigt, const Core::LinAlg::FourTensor<dim>& four_tensor);

  /*!
   * @brief Setup 9x6 Voigt matrix from 4th order tensor with 2nd minor symmetry
   *
   * @param[out] matrixVoigt  9x6 Voigt matrix that is set up based on fourTensor
   * @param[in]  fourTensor   4th order tensor with 2nd minor symmetry (C_ijkl = C_ijlk)
   */
  template <int dim>
  void setup_9x6_voigt_matrix_from_four_tensor(
      Core::LinAlg::Matrix<9, 6>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor);

  /*!
   * @brief Setup 6x9 Voigt matrix from 4th order tensor with 1st minor symmetry
   *
   * @param[out] matrixVoigt  6x9 Voigt matrix that is set up based on fourTensor
   * @param[in]  fourTensor   4th order tensor with 1st minor symmetry (C_ijkl = C_jikl)
   */
  template <int dim>
  void setup_6x9_voigt_matrix_from_four_tensor(
      Core::LinAlg::Matrix<6, 9>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor);

  /*!
   * @brief Setup 9x9 Voigt matrix from 4th order tensor
   *
   * @param[out] matrixVoigt  9x9 Voigt matrix that is set up based on fourTensor
   * @param[in]  fourTensor   4th order tensor without symmetries
   */
  template <int dim>
  void setup_9x9_voigt_matrix_from_four_tensor(
      Core::LinAlg::Matrix<9, 9>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor);

  /**
   * \brief Identity matrix in stress/strain-like Voigt notation
   * @param id (out) : 2nd order identity tensor in stress/strain-like Voigt notation
   */
  inline void identity_matrix(Core::LinAlg::Matrix<6, 1>& id)
  {
    id.clear();
    for (unsigned i = 0; i < 3; ++i) id(i) = 1.0;
  }

  /*!
   * \brief Constructs a 4th order identity matrix with rows in <rows_notation>-type Voigt
   * notation and columns in <cols_notation>-type Voigt notation
   * @tparam rows_notation Voigt notation used for the rows
   * @tparam cols_notation  Voigt notation used for the columns
   * @param id (out) : Voigt-Matrix
   */
  template <NotationType rows_notation, NotationType cols_notation>
  void fourth_order_identity_matrix(Core::LinAlg::Matrix<6, 6>& id);

  /*!
   * @brief Modify the representation of the input 6x6 Voigt matrix, by applying scalar factors to
   * rows and columns associated with mixed indices, e.g., with (1, 2)
   *
   * @note Helpful when converting between Voigt representations, e.g., from stress-stress form to
   * stress-strain form
   * @param[in]  input  input 6x6 Voigt matrix
   * @param[in]  scalar_row  scalar factor applied to rows 4, 5, 6 of the Voigt matrix
   * @param[in]  scalar_col  scalar factor applied to columns 4, 5, 6 of the Voigt matrix
   * @returns modified 6x6 Voigt matrix
   */
  Core::LinAlg::Matrix<6, 6> modify_voigt_representation(
      const Core::LinAlg::Matrix<6, 6>& input, const double scalar_row, const double scalar_col);

  /// collection of index mappings from matrix to Voigt-notation or vice versa
  struct IndexMappings
  {
   public:
    /**
     * from 6-Voigt index to corresponding 2-tensor row
     * @param i the index of a 6x1 vector in Voigt notation
     * @return row index of the corresponding 3x3 matrix
     */
    static inline int voigt6_to_matrix_row_index(unsigned int i)
    {
      assert_range_voigt_index(i);
      static constexpr int VOIGT6ROW[6] = {0, 1, 2, 0, 1, 2};
      return VOIGT6ROW[i];
    };

    /**
     * from 6-Voigt index to corresponding 2-tensor column
     * @param i the index of a 6x1 vector in Voigt notation
     * @return column index of the corresponding 3x3 matrix
     */
    static inline int voigt6_to_matrix_column_index(unsigned int i)
    {
      assert_range_voigt_index(i);
      static constexpr int VOIGT6COL[6] = {0, 1, 2, 1, 2, 0};
      return VOIGT6COL[i];
    };

    /// from symmetric 2-tensor index pair to 6-Voigt index
    static inline int symmetric_tensor_to_voigt6_index(unsigned int row, unsigned int col)
    {
      assert_range_matrix_index(row, col);
      static constexpr int VOIGT3X3SYM[3][3] = {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}};
      return VOIGT3X3SYM[row][col];
    };

    /// from non-symmetric 2-tensor index pair to 9-Voigt index
    static inline int non_symmetric_tensor_to_voigt9_index(unsigned int row, unsigned int col)
    {
      assert_range_matrix_index(row, col);
      static constexpr int VOIGT3X3NONSYM[3][3] = {{0, 3, 5}, {6, 1, 4}, {8, 7, 2}};
      return VOIGT3X3NONSYM[row][col];
    };

    /// from symmetric 6x6 Voigt notation matrix indices (e.g. constitutive tensor) to one of the
    /// four indices of a four tensor
    static inline int voigt_6x6_to_four_tensor_index(
        unsigned int voigt_row, unsigned int voigt_col, unsigned int target_index)
    {
      assert_range_voigt_index(voigt_row);
      assert_range_voigt_index(voigt_col);
      FOUR_C_ASSERT(target_index < 4, "target index for fourth order tensor out of range");
      static constexpr int FOURTH[6][6][4] = {
          {{0, 0, 0, 0}, {0, 0, 1, 1}, {0, 0, 2, 2}, {0, 0, 0, 1}, {0, 0, 1, 2}, {0, 0, 0, 2}},
          {{1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 2, 2}, {1, 1, 0, 1}, {1, 1, 1, 2}, {1, 1, 0, 2}},
          {{2, 2, 0, 0}, {2, 2, 1, 1}, {2, 2, 2, 2}, {2, 2, 0, 1}, {2, 2, 1, 2}, {2, 2, 0, 2}},
          {{0, 1, 0, 0}, {0, 1, 1, 1}, {0, 1, 2, 2}, {0, 1, 0, 1}, {0, 1, 1, 2}, {0, 1, 0, 2}},
          {{1, 2, 0, 0}, {1, 2, 1, 1}, {1, 2, 2, 2}, {1, 2, 0, 1}, {1, 2, 1, 2}, {1, 2, 0, 2}},
          {{0, 2, 0, 0}, {0, 2, 1, 1}, {0, 2, 2, 2}, {0, 2, 0, 1}, {0, 2, 1, 2}, {0, 2, 0, 2}}};
      return FOURTH[voigt_row][voigt_col][target_index];
    };

    /// instancing of this class is forbidden
    IndexMappings() = delete;

   private:
    static inline void assert_range_matrix_index(unsigned int row, unsigned int col)
    {
      FOUR_C_ASSERT(row < 3, "given row index out of range [0,2]");
      FOUR_C_ASSERT(col < 3, "given col index out of range [0,2]");
    }

    static inline void assert_range_voigt_index(unsigned int index)
    {
      FOUR_C_ASSERT(index < 6, "given index out of range [0,5]");
    }
  };

  //! typedefs for improved readability
  //! @{
  using Stresses = VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>;
  using Strains = VoigtUtils<Core::LinAlg::Voigt::NotationType::strain>;
  //! @}

}  // namespace Core::LinAlg::Voigt


FOUR_C_NAMESPACE_CLOSE

#endif