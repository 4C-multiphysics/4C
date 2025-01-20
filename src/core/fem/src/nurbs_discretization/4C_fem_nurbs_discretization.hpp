// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_NURBS_DISCRETIZATION_HPP
#define FOUR_C_FEM_NURBS_DISCRETIZATION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class Solver;
  class MapExtractor;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::FE
{
  namespace Nurbs
  {
    /*!
    \brief A class to manage a nurbs discretization in parallel

    Up to now, it's only a standard discretisation extendend by a knotvector.

    The nodes are replaced by control points on construction/input; the control points are
    derived from the node class and are hence managed by the original discretisation

    */
    class NurbsDiscretization : public Core::FE::Discretization
    {
     public:
      /*!
      \brief Standard Constructor

      \param name: name of this nurbs discretization
      \param comm: An epetra comm object associated with this discretization
      \param n_dim: number of space dimensions of this discretization
      */
      NurbsDiscretization(const std::string name, MPI_Comm comm, unsigned int n_dim);

      /*!
      \brief Set a knot vector

      Store a the knot vector in the discretization.
      It can then be accessed with the GetKnotVector method.

      \note Knot vectors attached to the discretization have to be
            completely redundant meaning that they are the same on
            each processor. I think its affordable.

      \param knots : The Knotvector class

      \author gammi

      */
      virtual void set_knot_vector(std::shared_ptr<Core::FE::Nurbs::Knotvector> knots);

      /*!
      \brief get a pointer to the knotvector from the discretization

      \return knots : The Knotvector class

      \author gammi

      */
      std::shared_ptr<Core::FE::Nurbs::Knotvector> get_knot_vector();
      std::shared_ptr<const Core::FE::Nurbs::Knotvector> get_knot_vector() const;

      /*!
      \brief return number of knots in each direction

      \param npatch (i)
             the number of the patch we are interested in

      \return  : The number of knots in each direction

      \author gammi

      */
      virtual std::vector<int> return_n_x_m_x_l(const int npatch)
      {
        return (knots_->return_n_x_m_x_l(npatch));
      }

      /*!
      \brief return degree in each direction

      \param npatch (i)
             the number of the patch we are interested in

      \return  : The degree in each direction

      \author gammi

      */
      virtual std::vector<int> return_degree(const int npatch)
      {
        return (knots_->return_degree(npatch));
      }

      /*!
      \brief return the offsets

      \return  : The element offsets of all patches

      \author gammi

      */
      virtual std::vector<int> return_offsets() { return (knots_->return_offsets()); }

      /*!
      \brief return number of elements in each direction

      \param npatch (i)
             the number of the patch we are interested in

      \return  : The number of elements in each direction

      \author gammi

      */
      virtual std::vector<int> return_nele_x_mele_x_lele(const int npatch)
      {
        return (knots_->return_nele_x_mele_x_lele(npatch));
      }

     private:
      //! don't want = operator
      NurbsDiscretization operator=(const NurbsDiscretization& old) = delete;

      //! don't want copy constructor
      NurbsDiscretization(const Core::FE::Nurbs::NurbsDiscretization& old) = delete;

      /*!
      \brief The knot vector

      dimension  u     : nurbs curve   (n)
      dimensions u,v   : nurbs surface (n x m)
      dimensions u,v,w : nurbs volume  (n x m x l)
      */
      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots_;

    };  // class NurbsDiscretization
  }  // namespace Nurbs

  namespace Utils
  {
    class DbcNurbs : public Core::FE::Utils::Dbc
    {
      using Dbc::do_dirichlet_condition;

     public:
      /// constructor
      DbcNurbs() = default;


     protected:
      void evaluate(const Teuchos::ParameterList& params, const Core::FE::Discretization& discret,
          double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
          DbcInfo& info, std::shared_ptr<std::set<int>>* dbcgids) const override;

      void do_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
          double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
          const Core::LinAlg::Vector<int>& toggle,
          const std::shared_ptr<std::set<int>>* dbcgids) const override;

     private:
      /*!
      \brief Fill mass matrix and rhs vector for evaluation of least squares dirichlet on a boundary

      \param ele          The element that is to be evaluated
      \param knots        element knot vector
      \param lm           reduced location vector of element (DBC DOFs only)
      \param funct        function information (read from the condition)
      \param val          value information (read from the condition)
      \param deg          degree of time derivative needed
      \param time         current time
      \param elemass      element matrix to be filled
      \param elerhs       element right hand side to be filled

      */
      template <Core::FE::CellType distype>
      void fill_matrix_and_rhs_for_ls_dirichlet_boundary(Core::Elements::Element& ele,
          const std::vector<Core::LinAlg::SerialDenseVector>* knots, const std::vector<int>& lm,
          const std::vector<Core::IO::Noneable<int>>& funct, const std::vector<double>& val,
          const unsigned deg, const double time, Core::LinAlg::SerialDenseMatrix& elemass,
          std::vector<Core::LinAlg::SerialDenseVector>& elerhs,
          const Core::Utils::FunctionManager& function_manager) const;

      /*!
      \brief Fill mass matrix and rhs vector for evaluation of least squares dirichlet on a domain

      \param ele          The element that is to be evaluated
        \param knots        element knot vector
      \param lm           reduced location vector of element (DBC DOFs only)
      \param funct        function information (read from the condition)
      \param val          value information (read from the condition)
      \param deg          degree of time derivative needed
      \param time         current time
      \param elemass      element matrix to be filled
      \param elerhs       element right hand side to be filled

      */
      template <Core::FE::CellType distype>
      void fill_matrix_and_rhs_for_ls_dirichlet_domain(Core::Elements::Element& ele,
          const std::vector<Core::LinAlg::SerialDenseVector>* knots, const std::vector<int>& lm,
          const std::vector<Core::IO::Noneable<int>>& funct, const std::vector<double>& val,
          const unsigned deg, const double time, Core::LinAlg::SerialDenseMatrix& elemass,
          std::vector<Core::LinAlg::SerialDenseVector>& elerhs,
          const Core::Utils::FunctionManager& function_manager) const;

    };  // class DbcNurbs
  }  // namespace Utils
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
