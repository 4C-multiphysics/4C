// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_PERIODIC_BOUNDINGBOX_HPP
#define FOUR_C_FEM_GEOMETRY_PERIODIC_BOUNDINGBOX_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_random.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::IO
{
  class DiscretizationVisualizationWriterMesh;
  class OutputControl;
}  // namespace Core::IO
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Geo
{
  namespace MeshFree
  {
    /*!
     \brief Construction of periodic boundingbox over entire considered simulation volume
     */
    class BoundingBox
    {
     public:
      BoundingBox();

      virtual ~BoundingBox() = default;

      /// initialize bounding box object
      void init(const Teuchos::ParameterList& binning_params);

      /// initialize bounding box object
      void init(Core::LinAlg::Matrix<3, 2> const& box, std::vector<bool> const& pbconoff);

      /// setup bounding box object, setup call is needed in case of box dirichlet
      void setup(const Teuchos::ParameterList& io_params,
          std::shared_ptr<Core::FE::Discretization> boundingbox_dis, MPI_Comm comm, int n_dim,
          const Core::IO::OutputControl& output_control);

      /// get edge length
      double edge_length(int dim) const { return edgelength_[dim]; }

      /// get box
      Core::LinAlg::Matrix<3, 2> const& box() const { return box_; }

      /// get flag indicating if periodic boundary conditions are active
      bool have_pbc() const { return haveperiodicbc_; }

      /// get const bounding box discretization
      Core::FE::Discretization const& bounding_box_discret() const { return *boxdiscret_; }

      /// get corner points
      double operator()(int i, int j) const { return box_(i, j); }

      /// initialize bounding box discretization
      void setup_bounding_box_discretization(
          std::shared_ptr<Core::FE::Discretization> boundingbox_dis, MPI_Comm comm,
          const int n_dim);

      /*!
      \brief shift node (if outside) back in box if periodic boundary conditions
      */
      bool shift_3d(Core::LinAlg::Matrix<3, 1>& d,
          Core::LinAlg::Matrix<3, 1> const X = Core::LinAlg::Matrix<3, 1>(
              Core::LinAlg::Initialization::zero)) const;

      /*!
      \brief get xi of intersection between two points
      */
      void get_xi_of_intersection_3d(Core::LinAlg::Matrix<3, 1> const& x1,
          Core::LinAlg::Matrix<3, 1> const& x2, Core::LinAlg::Matrix<3, 1>& xi) const;
      void get_xi_of_intersection_3d(Core::LinAlg::Matrix<3, 1> const& x1,
          Core::LinAlg::Matrix<3, 1> const& x2, Core::LinAlg::Matrix<3, 1>& xi,
          Core::LinAlg::Matrix<3, 2> const& box) const;

      /*! Check the distance to a reference point position (e.g. node of the
       * same element). If the distance is larger than half of the period
       * length, the point position has been shifted before.
       *
       * Warning: This assumes that the distance between point and reference
       *          point is not larger than half of the period length unless we
       *          shift it. For beam elements, this restricts the element length
       *          to be smaller than this value throughout the entire simulation.
       *          So far, we only check this once in the beginning.
       *
       * Note: this should be equivalent to the previously applied criterion
       *       that the distance between given point and reference point
       *       decreases by either adding or subtracting the period length. */
      void un_shift_3d(Core::LinAlg::Matrix<3, 1>& d, Core::LinAlg::Matrix<3, 1> const& ref,
          Core::LinAlg::Matrix<3, 1> const X = Core::LinAlg::Matrix<3, 1>(
              Core::LinAlg::Initialization::zero)) const;

      bool check_if_shift_between_points(Core::LinAlg::Matrix<3, 1>& d,
          Core::LinAlg::Matrix<3, 1> const& ref, std::vector<bool>& shift_in_dim,
          Core::LinAlg::Matrix<3, 1> const X = Core::LinAlg::Matrix<3, 1>(
              Core::LinAlg::Initialization::zero)) const;

      /*!
      \brief get random position inside box
      */
      void random_pos_within(Core::LinAlg::Matrix<3, 1>& pos, Core::Utils::Random* random) const;

      /*!
       \brief If necessary make the boundingbox larger to include this point as one of the corners
       of the box
       */
      void add_point(double const* x);

      /*!
       \brief Check whether "b" is within this boundingbox
       */
      bool within(const BoundingBox& b) const;

      /*!
       \brief Check if the point is within this boundingbox
       */
      bool within(const double* x, std::vector<bool>& within_in_dir) const;
      bool within(Core::LinAlg::Matrix<3, 1> const& x, std::vector<bool>& within_in_dir) const;
      bool within(Core::LinAlg::Matrix<3, 2> const& box, Core::LinAlg::Matrix<3, 1> const& x,
          std::vector<bool>& within_in_dir) const;

      /*!
       \brief Check these points are within this boundingbox
       */
      bool within(const Core::LinAlg::SerialDenseMatrix& xyz) const;

      /*!
       \brief Print the corner points of boundingbox on the screen
       */
      void print();

      /*!
       \brief get min of box in certain dim
       */
      double box_min(int dim) const { return box_min(box_, dim); }
      double box_min(Core::LinAlg::Matrix<3, 2> const& box, int dim) const { return box(dim, 0); }

      /*!
       \brief get max of box in certain dim
       */
      double box_max(int dim) const { return box_max(box_, dim); }
      double box_max(Core::LinAlg::Matrix<3, 2> const& box, int dim) const { return box(dim, 1); }

      /*!
       \brief Get the outmost point of the boundingbox
       */
      void undeformed_box_corner_point_position(int i, std::vector<double>& x) const;
      Core::LinAlg::Matrix<3, 1> undeformed_box_corner_point_position(int i) const;
      /*!
       \brief get reference position of corner point i
       */
      Core::LinAlg::Matrix<3, 1> reference_pos_of_corner_point(int i) const;

      /*!
       \brief get current position of corner point i
       */
      Core::LinAlg::Matrix<3, 1> current_position_of_corner_point(int i) const;

      /*!
       \brief print box
      */
      void print(std::ostream& out) const;

      /*!
       \brief Write output
      */
      void runtime_output_step_state(double timen, int stepn) const;

      /*!
       \brief Apply dirichlet condition according to input file
      */
      void apply_dirichlet(double timen, const Core::Utils::FunctionManager& function_manager);

      /*!
       \brief init runtime output object for bounding box discretization
      */
      void init_runtime_output(
          const Teuchos::ParameterList& io_params, const Core::IO::OutputControl& output_control);

      //! @name public function dealing with mapping of positions in case of a deforming bounding
      //! box
      //! @{

      //! transform from undeformed to global
      void transform_from_undeformed_bounding_box_system_to_global(
          Core::LinAlg::Matrix<3, 1> const& xi, Core::LinAlg::Matrix<3, 1>& x) const;

      void transform_from_undeformed_bounding_box_system_to_global(
          double const* xi, double* x) const;

      //! transform from global to undeformed
      bool transform_from_global_to_undeformed_bounding_box_system(
          Core::LinAlg::Matrix<3, 1> const& x,  ///< input  -> global position
          Core::LinAlg::Matrix<3, 1>& xi  ///< output -> position in undeformed bounding box system
      ) const;
      bool transform_from_global_to_undeformed_bounding_box_system(
          double const* x,  ///< input  -> global position
          double* xi        ///< output -> position in undeformed bounding box system
      ) const;

      //! @}

     protected:
      //! returns init state
      inline bool is_init() const { return isinit_; };

      //! returns setup state
      inline bool is_setup() const { return issetup_; };

      //! Check the init state
      inline void throw_if_not_init() const
      {
        if (not is_init()) FOUR_C_THROW("Call init() first!");
      }

      //! Check the init and setup state
      inline void throw_if_not_init_or_setup() const
      {
        if (not is_init() or not is_setup()) FOUR_C_THROW("Call init() and setup() first!");
      }

     private:
      /*!
      \brief shift node (if outside) back in box if periodic boundary conditions
      */
      bool shift_1d(int dim, double& d, double const& X = 0.0) const;

      /*!
      \brief shift node out of box if it was shifted in previously
      */
      bool un_shift_1d(int dim, double& d, double const& ref, double const& X = 0.0) const;

      bool in_between(double smin, double smax, double omin, double omax) const;

      //! @name private function dealing with mapping of positions in case of a deforming bounding
      //! box
      //! @{

      //! evaluate lagrange polynomial that maps from undeformed to global at xi
      void lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global(
          Core::LinAlg::Matrix<8, 1>& funct,  ///< to be filled with shape function values
          double r, double s, double t) const;

      //! evaluate first derivative of lagrange polynomial that maps from undeformed to global at xi
      void lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global_deriv1(
          Core::LinAlg::Matrix<3, 8>&
              deriv1,  ///< to be filled with shape function derivative values
          double r, double s, double t) const;

      //! @}

     protected:
      //! @name member variables

      //! indicates if the init() function has been called
      bool isinit_;

      //! indicates if the setup() function has been called
      bool issetup_;

     private:
      /// discretization with one volume element representing the box ( used e.g. for output)
      std::shared_ptr<Core::FE::Discretization> boxdiscret_;
      /// box displacement vector
      std::shared_ptr<Core::LinAlg::Vector<double>> disn_row_;
      std::shared_ptr<Core::LinAlg::Vector<double>> disn_col_;

      bool empty_;
      /// set global pbc flag
      bool haveperiodicbc_;
      /// set global dbc flag
      bool havedirichletbc_;
      /// box corners
      Core::LinAlg::Matrix<3, 2> box_;
      /// flags for existence of periodic boundary conditions in x, y, z direction
      bool pbconoff_[3];
      ///< box edge lengths in x, y, z direction
      double edgelength_[3];

      //! bounding box discretization runtime visualization writer
      std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh>
          visualization_output_writer_ptr_;
    };

  }  // namespace MeshFree
}  // namespace Core::Geo

FOUR_C_NAMESPACE_CLOSE

#endif
