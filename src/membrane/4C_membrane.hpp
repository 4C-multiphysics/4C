/*----------------------------------------------------------------------*/
/*! \file

\level 3


\brief Nonlinear Membrane Finite Element

The input line (in the header file) should read
MAT x KINEM nonlinear THICK x STRESS_STRAIN [plane_stress/plane_strain]


*----------------------------------------------------------------------*/
#ifndef FOUR_C_MEMBRANE_HPP
#define FOUR_C_MEMBRANE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_membrane_eletypes.hpp"
#include "4C_thermo_ele_impl_utils.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace STR
{
  namespace ELEMENTS
  {
    class ParamsInterface;
  }  // namespace ELEMENTS
}  // namespace STR

// forward declaration
namespace Mat
{
  class So3Material;
}  // namespace Mat

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace ELEMENTS
  {
    // forward declarations
    template <Core::FE::CellType distype2>
    class MembraneLine;

    /*!
    \brief A C++ wrapper for the membrane element
    */
    template <Core::FE::CellType distype>
    class Membrane : public Core::Elements::Element
    {
     public:
      //! @name Friends
      friend class MembraneTri3Type;
      friend class MembraneTri6Type;
      friend class MembraneQuad4Type;
      friend class MembraneQuad9Type;
      template <Core::FE::CellType distype2>
      friend class MembraneLine;

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Membrane(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Membrane(const Membrane<distype>& old);

      /*!
      \brief Deep copy this instance of Membrane and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-constructor is needed

      */
      Core::Elements::Element* Clone() const override;

      //! number of element nodes
      static constexpr int numnod_ = Core::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr int numdim_ = Core::FE::dim<distype>;

      //! number of dofs per node
      static constexpr int noddof_ = 3;

      //! total dofs per element
      static constexpr int numdof_ = noddof_ * numnod_;

      //! static const is required for fixedsizematrices
      static constexpr int numgpt_post_ = THR::DisTypeToNumGaussPoints<distype>::nquad;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType Shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int NumLine() const override;

      /*!
      \brief Return number of surfaces of this element
      */
      int NumSurface() const override { return 1; }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element

      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element

      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneTri3Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::tri6:
          {
            return MembraneTri6Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::quad4:
          {
            return MembraneQuad4Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::quad9:
          {
            return MembraneQuad9Type::Instance().UniqueParObjectId();
          }
          default:
            FOUR_C_THROW("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return -1;
      };

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Unpack(const std::vector<char>& data) override;


      //@}

      //! @name Acess methods

      /*!
      \brief Return the material of this element

      Note: The input parameter nummat is not the material number from input file
            as in SetMaterial(int matnum), but the number of the material within
            the vector of materials the element holds

      \param nummat (in): number of requested material
      */
      virtual Teuchos::RCP<Mat::So3Material> SolidMaterial(int nummat = 0) const;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const Core::Nodes::Node& node) const override { return noddof_; }

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
      void Print(std::ostream& os) const override;

      Core::Elements::ElementType& ElementType() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneTri3Type::Instance();
          }
          break;
          case Core::FE::CellType::tri6:
          {
            return MembraneTri6Type::Instance();
          }
          break;
          case Core::FE::CellType::quad4:
          {
            return MembraneQuad4Type::Instance();
          }
          break;
          case Core::FE::CellType::quad9:
          {
            return MembraneQuad9Type::Instance();
          }
          break;
          default:
            FOUR_C_THROW("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return MembraneQuad4Type::Instance();
      };

      /*!
      \brief Query names of element data to be visualized using BINIO

      The element fills the provided map with key names of
      visualization data the element wants to visualize AT THE CENTER
      of the element geometry. The values is supposed to be dimension of the
      data to be visualized. It can either be 1 (scalar), 3 (vector), 6 (sym. tensor)
      or 9 (nonsym. tensor)

      Example:
      \code
        // Name of data is 'Owner', dimension is 1 (scalar value)
        names.insert(std::pair<string,int>("Owner",1));
        // Name of data is 'StressesXYZ', dimension is 6 (sym. tensor value)
        names.insert(std::pair<string,int>("StressesXYZ",6));
      \endcode

      \param names (out): On return, the derived class has filled names with
                          key names of data it wants to visualize and with int dimensions
                          of that data.
      */
      void VisNames(std::map<std::string, int>& names) override;

      /*!
      \brief Query data to be visualized using BINIO of a given name

      The method is supposed to call this base method to visualize the owner of
      the element.
      If the derived method recognizes a supported data name, it shall fill it
      with corresponding data.
      If it does NOT recognizes the name, it shall do nothing.

      \warning The method must not change size of data

      \param name (in):   Name of data that is currently processed for visualization
      \param data (out):  data to be filled by element if element recognizes the name
      */
      bool VisData(const std::string& name, std::vector<double>& data) override;

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& eledistype,
          Input::LineDefinition* linedef) override;

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate Membrane element stiffness, mass, internal forces etc

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param discretization : pointer to discretization for de-assembly
      \param lm (in)        : location matrix for de-assembly
      \param elemat1 (out)  : (stiffness-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : (mass-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : (internal force-)vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) override;


      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a surfaces Neumann condition on the membrane element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,
                                  the controlling method does not expect the element
                                  to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;


      //@}

     protected:
      /// Update history variables at the end of time step (fiber direction, inelastic deformation)
      /// (braeu 07/16)
      void update_element(std::vector<double>& disp,  // current displacements
          Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
          Teuchos::RCP<Core::Mat::Material> mat);     // material

     public:
      /** \brief set the parameter interface ptr for the solid elements
       *
       *  \param p (in): Parameter list coming from the time integrator.
       *
       *  \author hiermeier
       *  \date 04/16 */
      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      /** \brief returns true if the parameter interface is defined and initialized, otherwise false
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline bool IsParamsInterface() const override { return (not interface_ptr_.is_null()); }

      /** \brief get access to the parameter interface pointer
       *
       *  \author hiermeier
       *  \date 04/16 */
      Teuchos::RCP<Core::Elements::ParamsInterface> ParamsInterfacePtr() override;

     protected:
      /** \brief get access to the interface
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline Core::Elements::ParamsInterface& params_interface()
      {
        if (not IsParamsInterface()) FOUR_C_THROW("The interface ptr is not set!");
        return *interface_ptr_;
      }

      /** \brief get access to the structure interface
       *
       *  \author vuong
       *  \date 11/16 */
      STR::ELEMENTS::ParamsInterface& str_params_interface();

     private:
      /** \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      Teuchos::RCP<Core::Elements::ParamsInterface> interface_ptr_;

      /// type of 2D dimension reduction
      enum DimensionalReduction
      {
        plane_stress,  ///< plane stress, i.e. lateral stress is zero \f$S_{33}=S_{13}=S_{23}=0\f$

        // Membrane not intended for plane strain evaluation (mentioned here for completeness)
        plane_strain  ///< plane strain, i.e. lateral strain is zero \f$E_{33}=E_{13}=E_{23}=0\f$
      };

      //! membrane thickness
      double thickness_;

      //! current membrane thickness at gauss point
      std::vector<double> cur_thickness_;

      //! membrane stress/strain state
      DimensionalReduction planetype_;

     protected:
      Core::FE::IntegrationPoints2D intpoints_;

     private:
      // internal calculation methods

      // don't want = operator
      Membrane<distype>& operator=(const Membrane<distype>& old);

      //! calculate nonlinear stiffness and mass matrix
      void mem_nlnstiffmass(std::vector<int>& lm,               // location matrix
          std::vector<double>& disp,                            // current displacements
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
          Core::LinAlg::Matrix<numdof_, numdof_>* massmatrix,   // element mass matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              // element internal force vector
          Core::LinAlg::Matrix<numgpt_post_, 6>* elestress,     // stresses at GP
          Core::LinAlg::Matrix<numgpt_post_, 6>* elestrain,     // strains at GP
          Teuchos::ParameterList& params,                       // algorithmic parameters e.g. time
          const Inpar::STR::StressType iostress,                // stress output option
          const Inpar::STR::StrainType iostrain);               // strain output option

      //! get reference and current configuration
      void mem_configuration(const std::vector<double>& disp,
          Core::LinAlg::Matrix<numnod_, noddof_>& xrefe,
          Core::LinAlg::Matrix<numnod_, noddof_>& xcurr);

      //! introduce orthonormal base in the undeformed configuration at current Gauss point
      void mem_orthonormalbase(const Core::LinAlg::Matrix<numnod_, noddof_>& xrefe,
          const Core::LinAlg::Matrix<numnod_, noddof_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& derivs,
          Core::LinAlg::Matrix<numdim_, numnod_>& derivs_ortho, double& G1G2_cn,
          Core::LinAlg::Matrix<noddof_, 1>& dXds1, Core::LinAlg::Matrix<noddof_, 1>& dXds2,
          Core::LinAlg::Matrix<noddof_, 1>& dxds1, Core::LinAlg::Matrix<noddof_, 1>& dxds2,
          Core::LinAlg::Matrix<noddof_, noddof_>& Q_localToGlobal) const;

      //! pushforward of 2nd Piola-Kirchhoff stresses to Cauchy stresses at Gauss point
      void mem_p_k2to_cauchy(const Core::LinAlg::Matrix<noddof_, noddof_>& pkstress_global,
          const Core::LinAlg::Matrix<noddof_, noddof_>& defgrd,
          Core::LinAlg::Matrix<noddof_, noddof_>& cauchy) const;

      // pushforward of Green-Lagrange to Euler-Almansi strains at Gauss point
      void mem_g_lto_ea(const Core::LinAlg::Matrix<noddof_, noddof_>& glstrain_global,
          const Core::LinAlg::Matrix<noddof_, noddof_>& defgrd,
          Core::LinAlg::Matrix<noddof_, noddof_>& euler_almansi) const;

      // determine deformation gradient in global frame on membrane surface
      void mem_defgrd_global(const Core::LinAlg::Matrix<noddof_, 1>& dXds1,
          const Core::LinAlg::Matrix<noddof_, 1>& dXds2,
          const Core::LinAlg::Matrix<noddof_, 1>& dxds1,
          const Core::LinAlg::Matrix<noddof_, 1>& dxds2, const double& lambda3,
          Core::LinAlg::Matrix<noddof_, noddof_>& defgrd_global) const;

      // determine extrapolation matrix for postprocessing purposes
      Core::LinAlg::Matrix<Core::FE::num_nodes<distype>,
          THR::DisTypeToNumGaussPoints<distype>::nquad>
      mem_extrapolmat() const;

    };  // class Membrane


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================

    /*----------------------------------------------------------------------*
     |  LINE 2 Element                                         fbraeu 06/16 |
     *----------------------------------------------------------------------*/
    class MembraneLine2Type : public Core::Elements::ElementType
    {
     public:
      std::string Name() const override { return "Membrane_line2Type"; }

      static MembraneLine2Type& Instance();

      Teuchos::RCP<Core::Elements::Element> Create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      Core::LinAlg::SerialDenseMatrix ComputeNullSpace(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static MembraneLine2Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  LINE 3 Element                                         fbraeu 06/16 |
     *----------------------------------------------------------------------*/
    class MembraneLine3Type : public Core::Elements::ElementType
    {
     public:
      std::string Name() const override { return "Membrane_line3Type"; }

      static MembraneLine3Type& Instance();

      Teuchos::RCP<Core::Elements::Element> Create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      Core::LinAlg::SerialDenseMatrix ComputeNullSpace(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static MembraneLine3Type instance_;
    };

    /*!
    \brief An element representing a line edge of a membrane element

    \note This is a pure Neumann boundary condition element. It's only
          purpose is to evaluate line Neumann boundary conditions that might be
          adjacent to a parent membrane element. It therefore does not implement
          the Core::Elements::Element::Evaluate method and does not have its own ElementRegister
    class.

    */
    template <Core::FE::CellType distype2>
    class MembraneLine : public Core::Elements::FaceElement
    {
     public:
      //! @name Friends
      friend class MembraneLine2Type;
      friend class MembraneLine3Type;

      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this line
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes
      \param parent: The parent shell element of this line
      \param lline: the local line number of this line w.r.t. the parent element
      */
      MembraneLine(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::ELEMENTS::Membrane<distype2>* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      MembraneLine(const MembraneLine<distype2>& old);

      //! number of nodes per line
      static constexpr int numnod_line_ =
          Core::FE::num_nodes<Core::FE::DisTypeToFaceShapeType<distype2>::shape>;

      static constexpr int noddof_ = 3;

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* Clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int UniqueParObjectId() const override
      {
        switch (Core::FE::DisTypeToFaceShapeType<distype2>::shape)
        {
          case Core::FE::CellType::line2:
          {
            return MembraneLine2Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::line3:
          {
            return MembraneLine3Type::Instance().UniqueParObjectId();
          }
          default:
            FOUR_C_THROW("unknown line type!");
            break;
        }
        // Intel compiler needs a return so
        return -1;
      };


      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Unpack(const std::vector<char>& data) override;


      //@}

      //! @name Acess methods

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType Shape() const override;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const Core::Nodes::Node& node) const override { return 3; }

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
       * \brief Return pointer to the parent element
       */
      virtual Discret::ELEMENTS::Membrane<distype2>* parent_element() const
      {
        Core::Elements::Element* parent = this->Core::Elements::FaceElement::parent_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::ELEMENTS::Membrane<distype2>*>(parent) != nullptr,
            "Parent element is no membrane element");
        return static_cast<Discret::ELEMENTS::Membrane<distype2>*>(parent);
      }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      Core::Elements::ElementType& ElementType() const override
      {
        switch (Core::FE::DisTypeToFaceShapeType<distype2>::shape)
        {
          case Core::FE::CellType::line2:
          {
            return MembraneLine2Type::Instance();
          }
          case Core::FE::CellType::line3:
          {
            return MembraneLine3Type::Instance();
          }
          default:
            FOUR_C_THROW("unknown line type!");
            break;
        }
        // Intel compiler needs a return so
        return MembraneLine2Type::Instance();
      };

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a line Neumann condition on the membrane element

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
      MembraneLine<distype2>& operator=(const MembraneLine<distype2>& old);

      Core::FE::IntegrationPoints1D intpointsline_;

    };  // class MembraneLine

  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
