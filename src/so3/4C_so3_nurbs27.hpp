// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_NURBS27_HPP
#define FOUR_C_SO3_NURBS27_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_so3_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/// Several parameters which are fixed for Solid nurbs27
const int NUMNOD_SONURBS27 = 27;  ///< number of nodes
const int NODDOF_SONURBS27 = 3;   ///< number of dofs per node
const int NUMDOF_SONURBS27 = 81;  ///< total dofs per element
const int NUMGPT_SONURBS27 = 27;  ///< total gauss points per element
const int NUMDIM_SONURBS27 = 3;   ///< number of dimensions

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    namespace Nurbs
    {
      class SoNurbs27Type : public Core::Elements::ElementType
      {
       public:
        std::string name() const override { return "So_nurbs27Type"; }

        static SoNurbs27Type& instance();

        Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

        std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
            const std::string eledistype, const int id, const int owner) override;

        std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

        int initialize(Core::FE::Discretization& dis) override;

        void nodal_block_information(
            Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

        Core::LinAlg::SerialDenseMatrix compute_null_space(
            Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

        void setup_element_definition(
            std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
            override;

       private:
        static SoNurbs27Type instance_;

        std::string get_element_type_string() const { return "SONURBS27_DEPRECATED"; }
      };


      /*!
      \brief A C++ version of a 27-control point nurbs solid element

      A structural 27-control points nurbs solid displacement element for large deformations.
      As its discretization is fixed many data structures are evaluated just once and kept
      for performance.

      \author gammi
      */
      class SoNurbs27 : public SoBase
      {
       public:
        //! @name Friends
        friend class SoNurbs27Type;

        //@}
        //! @name Constructors and destructors and related methods

        /*!
        \brief Standard Constructor

        \param id : A unique global id
        \param owner : elements owner
        */
        SoNurbs27(int id, int owner);

        /*!
        \brief Copy Constructor

        Makes a deep copy of a Element

        */
        SoNurbs27(const SoNurbs27& old);

        /*!
        \brief Deep copy this instance of Solid3 and return pointer to the copy

        The clone() method is used from the virtual base class Element in cases
        where the type of the derived class is unknown and a copy-ctor is needed

        */
        Core::Elements::Element* clone() const override;

        /*!
        \brief Get shape type of element
        */
        Core::FE::CellType shape() const override;

        /*!
        \brief Return number of volumes of this element
        */
        int num_volume() const override { return 1; }

        /*!
        \brief Return number of surfaces of this element
        */
        int num_surface() const override { return 6; }

        /*!
        \brief Return number of lines of this element
        */
        int num_line() const override { return 12; }

        /*!
        \brief Get vector of std::shared_ptrs to the lines of this element

        */
        std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

        /*!
        \brief Get vector of std::shared_ptrs to the surfaces of this element

        */
        std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

        /*!
        \brief Return unique ParObject id

        every class implementing ParObject needs a unique id defined at the
        top of this file.
        */
        int unique_par_object_id() const override
        {
          return SoNurbs27Type::instance().unique_par_object_id();
        }

        /*!
        \brief Pack this class so it can be communicated

        \ref pack and \ref unpack are used to communicate this element

        */
        void pack(Core::Communication::PackBuffer& data) const override;

        /*!
        \brief Unpack data from a char vector into this class

        \ref pack and \ref unpack are used to communicate this element

        */
        void unpack(Core::Communication::UnpackBuffer& buffer) override;


        //@}

        //! @name Access methods


        /*!
        \brief Get number of degrees of freedom of a certain node
               (implements pure virtual Core::Elements::Element)

        The element decides how many degrees of freedom its nodes must have.
        As this may vary along a simulation, the element can redecide the
        number of degrees of freedom per node along the way for each of it's nodes
        separately.
        */
        int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

        /*!
        \brief Get number of degrees of freedom per element
               (implements pure virtual Core::Elements::Element)

        The element decides how many element degrees of freedom it has.
        It can redecide along the way of a simulation.

        \note Element degrees of freedom mentioned here are dofs that are visible
              at the level of the total system of equations. Purely internal
              element dofs that are condensed internally should NOT be considered.
        */
        int num_dof_per_element() const override { return 0; }

        /*!
        \brief Print this element
        */
        void print(std::ostream& os) const override;

        Core::Elements::ElementType& element_type() const override
        {
          return SoNurbs27Type::instance();
        }

        //! @name Input and Creation

        /*!
        \brief Read input for this element
        */
        bool read_element(const std::string& eletype, const std::string& distype,
            const Core::IO::InputParameterContainer& container) override;

        //@}

        //! @name Evaluation

        /*!
        \brief Evaluate an element

        Evaluate So_nurbs27 element stiffness, mass, internal forces, etc.

        If nullptr on input, the controlling method does not expect the element
        to fill these matrices or vectors.

        \return 0 if successful, negative otherwise
        */
        int evaluate(
            Teuchos::ParameterList&
                params,  ///< ParameterList for communication between control routine and elements
            Core::FE::Discretization&
                discretization,    ///< pointer to discretization for de-assembly
            std::vector<int>& lm,  ///< location matrix for de-assembly
            Core::LinAlg::SerialDenseMatrix&
                elemat1,  ///< (stiffness-)matrix to be filled by element.
            Core::LinAlg::SerialDenseMatrix& elemat2,  ///< (mass-)matrix to be filled by element.
            Core::LinAlg::SerialDenseVector&
                elevec1,  ///< (internal force-)vector to be filled by element
            Core::LinAlg::SerialDenseVector& elevec2,  ///< vector to be filled by element
            Core::LinAlg::SerialDenseVector& elevec3   ///< vector to be filled by element
            ) override;


        /*!
        \brief Evaluate a Neumann boundary condition

        this method evaluates a surface Neumann condition on the solid3 element

        \param params (in/out)    : ParameterList for communication between control routine
                                    and elements
        \param discretization (in): A reference to the underlying discretization
        \param condition (in)     : The condition to be evaluated
        \param lm (in)            : location vector of this element
        \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

        \return 0 if successful, negative otherwise
        */
        int evaluate_neumann(Teuchos::ParameterList& params,
            Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
            std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
            Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

        //@}

        /*!
         \brief calculate a. scaled director matrix for thin shell-like structures
           or its inverse

         \return void
         */
        void do_calc_stc_matrix(Core::LinAlg::Matrix<81, 81>& elemat1,
            const Inpar::Solid::StcScale stc_scaling, const int stc_layer, std::vector<int>& lm,
            Core::FE::Discretization& discretization, bool do_inverse);

       protected:
        //! action parameters recognized by So_nurbs27
        enum ActionType
        {
          none,
          calc_struct_linstiff,
          calc_struct_nlnstiff,
          calc_struct_internalforce,
          calc_struct_linstiffmass,
          calc_struct_nlnstiffmass,
          calc_struct_eleload,
          calc_struct_fsiload,
          calc_struct_update_istep,
          calc_stc_matrix,  //! calculate scaled director matrix for thin shell-like structures
          calc_stc_matrix_inverse,  //! calculate inverse scaled director matrix for thin shell-like
                                    //! structures
          calc_struct_reset_istep,
          calc_struct_energy,
          calc_struct_stifftemp,  //!< TSI specific: mechanical-thermal stiffness
          calc_struct_nlnstifflmass
        };

        //! vector of inverses of the jacobian in material frame
        std::vector<Core::LinAlg::Matrix<3, 3>> invJ_;
        //! determinant of Jacobian in material frame
        std::vector<double> detJ_;


        // internal calculation methods

        //! don't want = operator
        SoNurbs27& operator=(const SoNurbs27& old);

        //! init the inverse of the jacobian and its determinant in
        //! the material configuration
        virtual void init_jacobian_mapping(Core::FE::Discretization& dis);

        //! Calculate nonlinear stiffness and mass matrix
        virtual void sonurbs27_nlnstiffmass(std::vector<int>& lm,  ///< location matrix
            Core::FE::Discretization& discretization,   ///< discretisation to extract knot vector
            std::vector<double>& disp,                  ///< current displacements
            std::vector<double>& residual,              ///< current residual displ
            Core::LinAlg::Matrix<81, 81>* stiffmatrix,  ///< element stiffness matrix
            Core::LinAlg::Matrix<81, 81>* massmatrix,   ///< element mass matrix
            Core::LinAlg::Matrix<81, 1>* force,         ///< element internal force vector
            Teuchos::ParameterList& params);

        //! Evaluate Nurbs27 Shapefcts to keep them static
        std::vector<Core::LinAlg::Matrix<27, 1>> sonurbs27_shapefcts(
            const std::vector<Core::LinAlg::SerialDenseVector>& myknots,
            const Core::LinAlg::Matrix<27, 1>& weights);
        //! Evaluate Nurbs27 Derivs to keep them static
        std::vector<Core::LinAlg::Matrix<3, 27>> sonurbs27_derivs(
            const std::vector<Core::LinAlg::SerialDenseVector>& myknots,
            const Core::LinAlg::Matrix<27, 1>& weights);
        //! Evaluate Nurbs27 Weights to keep them static
        std::vector<double> sonurbs27_gpweights();

        //! calculate internal energy
        double calc_int_energy(Core::FE::Discretization& discretization, std::vector<double>& disp,
            Teuchos::ParameterList& params);

        //! Lump mass matrix (bborn 07/08)
        void lumpmass(Core::LinAlg::Matrix<NUMDOF_SONURBS27, NUMDOF_SONURBS27>* emass);

       private:
        std::string get_element_type_string() const { return "SONURBS27_DEPRECATED"; }
      };  // class So_nurbs27



    }  // namespace Nurbs
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
