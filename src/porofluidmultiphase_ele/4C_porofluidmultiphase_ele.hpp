// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUIDMULTIPHASE_ELE_HPP
#define FOUR_C_POROFLUIDMULTIPHASE_ELE_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace ELEMENTS
  {
    class PoroFluidMultiPhaseType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "PoroFluidMultiPhaseType"; }

      /// singleton instance method
      static PoroFluidMultiPhaseType& instance();

      /// create an element from data
      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      /// create an element from a dat file specifier
      Teuchos::RCP<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      /// create an empty element
      Teuchos::RCP<Core::Elements::Element> create(const int id, const int owner) override;

      /// nodal block information to create a null space description
      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      /// do the null space computation
      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      /// setup the dat file input line definitions for this type of element
      void setup_element_definition(
          std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
          override;

      /// initialize element
      int initialize(Core::FE::Discretization& dis) override;

      /// pre-evaluation
      void pre_evaluate(Core::FE::Discretization& dis, Teuchos::ParameterList& p,
          Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
          Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
          Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector1,
          Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector2,
          Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector3) override;

     private:
      /// the actual instance
      static PoroFluidMultiPhaseType instance_;
    };

    /*!
    \brief The PoroFluidMultiPhase element
    */
    class PoroFluidMultiPhase : public Core::Elements::Element
    {
      //! @name Friends
      friend class PoroFluidMultiPhaseType;

     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor
      */
      PoroFluidMultiPhase(int id,  ///< A unique global id of this element
          int owner                ///< processor id who owns a certain instance of this class
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      PoroFluidMultiPhase(const PoroFluidMultiPhase& old);

      /*!
      \brief Deep copy this instance of PoroFluidMultiPhase and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;


      //@}

      /// Set element material
      /*!
        Material numbers are read from the input file. The element stores
        a corresponding material object. These material objects can be
        anything from very simple (just a little calculation) to highly
        sophisticated with history data. The material is packed and
        unpacked along with its element.

        \param matnum : material number from input file

        \note reimplementation of this method, due to initialising
              numdofpernode_, since the material is known now.
       */
      void set_material(const int index, Teuchos::RCP<Core::Mat::Material> mat) override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief set discretization type of element
      */
      virtual void set_dis_type(Core::FE::CellType shape)
      {
        distype_ = shape;
        return;
      };

      /*!
      \brief Return number of lines of this element
      */
      int num_line() const override;

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override;

      /*!
      \brief Return number of volumes of this element
      */
      int num_volume() const override;

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element
      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element
      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return PoroFluidMultiPhaseType::instance().unique_par_object_id();
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

      //! @name Access methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can re-decide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        if (numdofpernode_ < 1) FOUR_C_THROW("NumDofPerNode is < 1");
        return numdofpernode_;
      }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual Core::Elements::Element)

      The element decides how many element degrees of freedom it has.
      It can re-decide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      /*!
      \brief Return ElementType
      */
      Core::Elements::ElementType& element_type() const override
      {
        return PoroFluidMultiPhaseType::instance();
      }

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element (multiple dofset version)

      An element derived from this class uses the Evaluate method to receive commands
      and parameters from some control routine in params and evaluates element matrices and
      vectors accoring to the command in params.

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param la (in)            : location data for all dofsets of the discretization
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
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a surfaces Neumann condition on the shell element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     protected:
      virtual void initialize();

     private:
      //! the element discretization type (shape)
      Core::FE::CellType distype_;

      /*!
       * \brief number of dofs per node (for systems of transport equations)
       * (storage neccessary because we dont know the material in the post filters anymore)
       */
      int numdofpernode_;

      //! don't want = operator
      PoroFluidMultiPhase& operator=(const PoroFluidMultiPhase& old);
    };  // class PoroFluidMultiPhase


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================

    class PoroFluidMultiPhaseBoundaryType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "PoroFluidMultiPhaseBoundaryType"; }

      static PoroFluidMultiPhaseBoundaryType& instance();

      Teuchos::RCP<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented");
        return nullspace;
      }

     private:
      static PoroFluidMultiPhaseBoundaryType instance_;
    };


    /*!
    \brief An element representing a boundary element of a PoroFluidMultiPhase element

    \note This is a pure boundary condition element. Its only
          purpose is to evaluate certain boundary conditions that might be
          adjacent to a parent PoroFluidMultiPhase element.
    */
    class PoroFluidMultiPhaseBoundary : public Core::Elements::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this surface
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent PoroFluidMultiPhase element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      PoroFluidMultiPhaseBoundary(int id, int owner, int nnode, const int* nodeids,
          Core::Nodes::Node** nodes, Discret::ELEMENTS::PoroFluidMultiPhase* parent,
          const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      PoroFluidMultiPhaseBoundary(const PoroFluidMultiPhaseBoundary& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief Return number of lines of boundary element
      */
      int num_line() const override;

      /*!
      \brief Return number of surfaces of boundary element
       */
      int num_surface() const override;

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element

      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element
      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int unique_par_object_id() const override
      {
        return PoroFluidMultiPhaseBoundaryType::instance().unique_par_object_id();
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
      As this may vary along a simulation, the element can re-decide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        return parent_element()->num_dof_per_node(node);
      }

      //! Return a pointer to the parent element of this boundary element
      Discret::ELEMENTS::PoroFluidMultiPhase* parent_element() const override
      {
        Core::Elements::Element* parent = Core::Elements::FaceElement::parent_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::ELEMENTS::PoroFluidMultiPhase*>(parent) != nullptr,
            "Master element is no PoroFluidMultiPhase element");
        return static_cast<Discret::ELEMENTS::PoroFluidMultiPhase*>(parent);
      }


      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual Core::Elements::Element)

      The element decides how many element degrees of freedom it has.
      It can re-decide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      /*!
      \brief Return ElementType
      */
      Core::Elements::ElementType& element_type() const override
      {
        return PoroFluidMultiPhaseBoundaryType::instance();
      }

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      An element derived from this class uses the Evaluate method to receive commands
      and parameters from some control routine in params and evaluates element matrices and
      vectors accoring to the command in params.

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param la (in)            : location data for all dofsets of the discretization
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
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      /*!
      \brief Evaluate an element

      An element derived from this class uses the Evaluate method to receive commands
      and parameters from some control routine in params and evaluates element matrices and
      vectors accoring to the command in params.

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

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a surface Neumann condition on the PoroFluidMultiPhase element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      // don't want = operator
      PoroFluidMultiPhaseBoundary& operator=(const PoroFluidMultiPhaseBoundary& old);

    };  // class PoroFluidMultiPhaseBoundary


  }  // namespace ELEMENTS
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
