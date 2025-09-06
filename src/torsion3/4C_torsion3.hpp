// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TORSION3_HPP
#define FOUR_C_TORSION3_HPP


#include "4C_config.hpp"

#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declaration ...
namespace Solid
{
  namespace Elements
  {
    class ParamsInterface;
  }  // namespace Elements
}  // namespace Solid

namespace Discret
{
  namespace Elements
  {
    class Torsion3Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Torsion3Type"; }

      static Torsion3Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static Torsion3Type instance_;
    };

    /*!
    \brief three dimensional torsion element

    */
    class Torsion3 : public Core::Elements::Element
    {
     public:
      //! @name Friends


      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id    (in): A globally unique element id
      \param etype (in): Type of element
      \param owner (in): owner processor of the element
      */
      Torsion3(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      Torsion3(const Torsion3& old);



      /*!
      \brief Deep copy this instance of Torsion3 and return pointer to the copy

      The clone() method is used by the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed
    .
      */
      Core::Elements::Element* clone() const override;

      /*!
     \brief Get shape type of element
     */
      Core::FE::CellType shape() const override;


      /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H
      */
      int unique_par_object_id() const override
      {
        return Torsion3Type::instance().unique_par_object_id();
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

      Core::Elements::ElementType& element_type() const override
      {
        return Torsion3Type::instance();
      }

      //@}

      /*!
      \brief Return number of lines to this element
      */
      int num_line() const override { return 1; }


      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;


      /*!
      \brief Get number of degrees of freedom of a single node
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        /*note: this is not necessarily the number of DOF assigned to this node by the
         *discretization finally, but only the number of DOF requested for this node by this
         *element; the discretization will finally assign the maximal number of DOF to this node
         *requested by any element connected to this node*/
        return 3;
      }


      /*!
      \brief Get number of degrees of freedom per element not including nodal degrees of freedom
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;


      //@}

      //! @name Construction


      /*!
      \brief Read input for this element

      This class implements a dummy of this method that prints a warning and
      returns false. A derived class would read one line from the input file and
      store all necessary information.

      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;


      //@}


      //! @name Evaluation methods


      /*!
      \brief Evaluate an element

      An element derived from this class uses the Evaluate method to receive commands
      and parameters from some control routine in params and evaluates element matrices and
      vectors according to the command in params.

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param lm (in)            : location vector of this element
      \param elemat1 (out)      : matrix to be filled by element depending on commands
                                  given in params
      \param elemat2 (out)      : matrix to be filled by element depending on commands
                                  given in params
      \param elevec1 (out)      : vector to be filled by element depending on commands
                                  given in params
      \param elevec2 (out)      : vector to be filled by element depending on commands
                                  given in params
      \param elevec3 (out)      : vector to be filled by element depending on commands
                                  given in params
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      /*!
      \brief Evaluate a Neumann boundary condition

      An element derived from this class uses the evaluate_neumann method to receive commands
      and parameters from some control routine in params and evaluates a Neumann boundary condition
      given in condition

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : Force vector to be filled by element

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

      /*! \brief set the parameter interface ptr for the solid elements
       *
       *  \param p (in): Parameter list coming from the time integrator.
       *
       */
      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      /*! \brief returns true if the parameter interface is defined and initialized, otherwise false
       *
       */
      inline bool is_params_interface() const override { return (interface_ptr_ != nullptr); }

      /*! \brief get access to the parameter interface pointer
       *
       */
      std::shared_ptr<Core::Elements::ParamsInterface> params_interface_ptr() override;

     protected:
      /** \brief get access to the interface
       *
       */
      inline FourC::Solid::Elements::ParamsInterface& params_interface()
      {
        if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
        return *interface_ptr_;
      }

     private:
      //! possible bending potentials
      enum BendingPotential
      {
        quadratic,
        cosine
      };

      /*! \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      std::shared_ptr<FourC::Solid::Elements::ParamsInterface> interface_ptr_;

      //! Bending potential
      BendingPotential bendingpotential_;

      //! @name Internal calculation methods

      //! calculation of nonlinear stiffness and mass matrix
      void t3_nlnstiffmass(std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseMatrix* massmatrix, Core::LinAlg::SerialDenseVector* force);

      //! calculation of elastic energy
      void t3_energy(Teuchos::ParameterList& params, std::vector<double>& disp,
          Core::LinAlg::SerialDenseVector* intenergy);


      //@}


      //! @name Methods for Brownian dynamics simulations

      //! shifts nodes so that proper evaluation is possible even in case of periodic boundary
      //! conditions
      template <int nnode, int ndim>                   // number of nodes, number of dimensions
      void node_shift(Teuchos::ParameterList& params,  //!< parameter list
          std::vector<double>& disp);                  //!< element disp vector

      //@}

      // don't want = operator
      Torsion3& operator=(const Torsion3& old);


    };  // class Torsion3



    // << operator
    std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);


  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
