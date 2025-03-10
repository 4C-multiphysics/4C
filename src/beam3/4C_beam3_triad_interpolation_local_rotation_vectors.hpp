// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_TRIAD_INTERPOLATION_LOCAL_ROTATION_VECTORS_HPP
#define FOUR_C_BEAM3_TRIAD_INTERPOLATION_LOCAL_ROTATION_VECTORS_HPP

#include "4C_config.hpp"

#include "4C_beam3_triad_interpolation.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace LargeRotations
{
  /**
   * \brief a triad interpolation scheme based on local rotation vectors
   *        see Shoemake (1985) and Crisfield/Jelenic (1999) for formulae and details
   */

  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors : public LargeRotations::TriadInterpolation<T>
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}


    //! @name Constructors and destructors and related methods

    /** \brief Standard Constructor
     *
     */
    TriadInterpolationLocalRotationVectors();

    //@}

    //! @name Accessors

    /** \brief get node I which is part of the definition of the reference triad
     *
     */
    inline unsigned int node_i() const { return node_i_; }

    /** \brief get node J which is part of the definition of the reference triad
     *
     */
    inline unsigned int node_j() const { return node_j_; }

    //@}


    //! @name Derived methods

    /** \brief reset interpolation scheme with nodal quaternions
     *
     */
    void reset(std::vector<Core::LinAlg::Matrix<4, 1, T>> const& nodal_quaternions) override;

    /** \brief reset interpolation scheme with nodal triads
     *
     */
    void reset(std::vector<Core::LinAlg::Matrix<3, 3, T>> const& nodal_triads) override;

    /** \brief compute the interpolated triad at any point \xi \in [-1,1] in parameter space
     *
     */
    void get_interpolated_triad_at_xi(
        Core::LinAlg::Matrix<3, 3, T>& triad, const double xi) const override;

    /** \brief compute the interpolated quaternion at any point \xi \in [-1,1] in parameter space
     *
     */
    void get_interpolated_quaternion_at_xi(
        Core::LinAlg::Matrix<4, 1, T>& quaternion, const double xi) const override;

    //@}

    //! @name specific methods of this triad interpolation scheme (based on local rotation vectors)

    /** \brief compute the interpolated triad based on given local rotation vector
     *
     */
    void get_interpolated_triad(
        Core::LinAlg::Matrix<3, 3, T>& triad, const Core::LinAlg::Matrix<3, 1, T>& Psi_l) const;

    /** \brief compute the interpolated quaternion based on given local rotation vector
     *
     */
    void get_interpolated_quaternion(Core::LinAlg::Matrix<4, 1, T>& quaternion,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l) const;

    /** \brief compute the local rotation vector at any point \xi \in [-1,1] in parameter space
     *
     */
    void get_interpolated_local_rotation_vector_at_xi(
        Core::LinAlg::Matrix<3, 1, T>& Psi_l, const double xi) const;

    /** \brief compute the local rotation vector based on given shape function values
     *
     */
    void get_interpolated_local_rotation_vector(Core::LinAlg::Matrix<3, 1, T>& Psi_l,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i) const;


    /** \brief compute the arc-length derivative of the local rotation vector at any point
     *         \xi \in [-1,1] in parameter space
     *
     */
    void get_interpolated_local_rotation_vector_derivative_at_xi(
        Core::LinAlg::Matrix<3, 1, T>& Psi_l_s, const double jacobifac, const double xi) const;

    /** \brief compute the arc-length derivative of the local rotation vector based on given
     *         shape function values
     *
     */
    void get_interpolated_local_rotation_vector_derivative(Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const;


    /** \brief compute the generalized rotational interpolation matrices for all nodes at
     *         any point \xi \in [-1,1] in parameter space
     *
     */
    void get_nodal_generalized_rotation_interpolation_matrices_at_xi(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde, const double xi) const;

    /** \brief compute the generalized rotational interpolation matrices for all nodes
     *         based on given local rotation vector and shape function values
     *
     */
    void get_nodal_generalized_rotation_interpolation_matrices(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i) const;


    /** \brief compute the arc-length derivative of generalized rotational interpolation
     *         matrices for all nodes based on given local rotation vector and shape function values
     *
     */
    void get_nodal_generalized_rotation_interpolation_matrices_derivative(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde_prime,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const;

    /** \brief compute the arc-length derivative of generalized rotational interpolation
     *         matrices for all nodes based on given local rotation vector and shape function values
     *
     */
    void get_nodal_generalized_rotation_interpolation_matrices_derivative(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde_prime,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i_s) const;

    //@}

   private:
    //! @name Private methods

    /** \brief set the two nodes I and J that are used to define the reference triad later on
     *
     */
    void set_node_iand_j();

    //! get the interpolation scheme from the given number of nodes
    Core::FE::CellType get_dis_type() const;

    //! compute quaternion corresponding to reference triad Lambda_r according to (3.9), Jelenic
    //! 1999
    void calc_ref_quaternion(const Core::LinAlg::Matrix<4, 1, T>& Q_nodeI,
        const Core::LinAlg::Matrix<4, 1, T>& Q_nodeJ, Core::LinAlg::Matrix<4, 1, T>& Q_r) const;

    //! compute angle of relative rotation between node I and J according to (3.10), Jelenic 1999
    void calc_phi_ij(const Core::LinAlg::Matrix<4, 1, T>& Q_nodeI,
        const Core::LinAlg::Matrix<4, 1, T>& Q_nodeJ, Core::LinAlg::Matrix<3, 1, T>& Phi_IJ) const;

    //! compute nodal local rotations according to (3.8), Jelenic 1999
    void calc_psi_li(const Core::LinAlg::Matrix<4, 1, T>& Q_i,
        const Core::LinAlg::Matrix<4, 1, T>& Q_r, Core::LinAlg::Matrix<3, 1, T>& Psi_li) const;

    //! compute interpolated local relative rotation \Psi^l according to (3.11), Jelenic 1999
    void calc_psi_l(const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psi_li,
        const Core::LinAlg::Matrix<1, numnodes, double>& func,
        Core::LinAlg::Matrix<3, 1, T>& Psi_l) const;

    //! compute derivative of interpolated local relative rotation \Psi^l with respect to reference
    //! arc-length parameter s according to (3.11), Jelenic 1999
    void calc_psi_l_s(const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psi_li,
        const Core::LinAlg::Matrix<1, numnodes, double>& deriv_xi, const double& jacobi,
        Core::LinAlg::Matrix<3, 1, T>& Psi_l_s) const;

    //! compute local triad \Lambda from Crisfield 1999, eq. (4.7)
    void calc_lambda(const Core::LinAlg::Matrix<3, 1, T>& Psi_l,
        const Core::LinAlg::Matrix<4, 1, T>& Q_r, Core::LinAlg::Matrix<3, 3, T>& Lambda) const;

    //! compute quaternion equivalent to local triad \Lambda from Crisfield 1999, eq. (4.7)
    void calc_qgauss(const Core::LinAlg::Matrix<3, 1, T>& Psi_l,
        const Core::LinAlg::Matrix<4, 1, T>& Q_r, Core::LinAlg::Matrix<4, 1, T>& Qgauss) const;

    //! compute \tilde{I}^i in (3.18), page 152, Jelenic 1999, for all nodes i at a certain Gauss
    //! point
    void compute_itilde(const Core::LinAlg::Matrix<3, 1, T>& Psil,
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde,
        const Core::LinAlg::Matrix<3, 1, T>& phiIJ, const Core::LinAlg::Matrix<3, 3, T>& Lambdar,
        const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psili,
        const Core::LinAlg::Matrix<1, numnodes, double>& funct) const;

    //! compute \tilde{I}^{i'} in (3.19), page 152, Jelenic 1999 for all nodes i at a certain Gauss
    //! point
    void compute_itildeprime(const Core::LinAlg::Matrix<3, 1, T>& Psil,
        const Core::LinAlg::Matrix<3, 1, T>& Psilprime,
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itildeprime,
        const Core::LinAlg::Matrix<3, 1, T>& phiIJ, const Core::LinAlg::Matrix<3, 3, T>& Lambdar,
        const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psili,
        const Core::LinAlg::Matrix<1, numnodes, double>& funct,
        const Core::LinAlg::Matrix<1, numnodes, double>& deriv_s) const;

    //! compute matrix v_I as outlined in the equations above (3.15) on page 152 of Jelenic 1999
    void calc_v_i(
        Core::LinAlg::Matrix<3, 3, T>& vI, const Core::LinAlg::Matrix<3, 1, T>& phiIJ) const;

    //! compute matrix v_J as outlined in the equations above (3.15) on page 152 of Jelenic 1999
    void calc_v_j(
        Core::LinAlg::Matrix<3, 3, T>& vJ, const Core::LinAlg::Matrix<3, 1, T>& phiIJ) const;

    //@}


   private:
    //! @name member variables

    //! node I for determination of reference triad, eq. (3.9), (3.10), Jelenic 1999
    unsigned int node_i_;

    //! node J for determination of reference triad, eq. (3.9), (3.10), Jelenic 1999
    unsigned int node_j_;

    //! this determines the kind of shape functions which are to be applied
    Core::FE::CellType distype_;

    //! nodal triads stored as quaternions
    std::vector<Core::LinAlg::Matrix<4, 1, T>> qnode_;

    //! reference quaternion Q_r corresponding to reference triad Lambda_r
    Core::LinAlg::Matrix<4, 1, T> q_r_;

    //! local rotation angles at nodes: angles between nodal triads and reference triad
    std::vector<Core::LinAlg::Matrix<3, 1, T>> psi_li_;

    //@}
  };

}  // namespace LargeRotations

FOUR_C_NAMESPACE_CLOSE

#endif
