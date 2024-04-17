/*----------------------------------------------------------------------*/
/*! \file

\level 3


\brief Nonlinear Membrane Finite Element

The input line (in the header file) should read
MAT x KINEM nonlinear THICK x STRESS_STRAIN [plane_stress/plane_strain]


*----------------------------------------------------------------------*/
#ifndef FOUR_C_MEMBRANE_HPP
#define FOUR_C_MEMBRANE_HPP

#include "baci_config.hpp"

#include "baci_discretization_fem_general_utils_local_connectivity_matrices.hpp"
#include "baci_inpar_structure.hpp"
#include "baci_lib_element.hpp"
#include "baci_membrane_eletypes.hpp"
#include "baci_thermo_ele_impl_utils.hpp"

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
namespace MAT
{
  class So3Material;
}  // namespace MAT

namespace DRT
{
  // forward declarations
  class Discretization;

  namespace ELEMENTS
  {
    // forward declarations
    template <CORE::FE::CellType distype2>
    class MembraneLine;

    /*!
    \brief A C++ wrapper for the membrane element
    */
    template <CORE::FE::CellType distype>
    class Membrane : public DRT::Element
    {
     public:
      //! @name Friends
      friend class Membrane_tri3Type;
      friend class Membrane_tri6Type;
      friend class Membrane_quad4Type;
      friend class Membrane_quad9Type;
      template <CORE::FE::CellType distype2>
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
      DRT::Element* Clone() const override;

      //! number of element nodes
      static constexpr int numnod_ = CORE::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr int numdim_ = CORE::FE::dim<distype>;

      //! number of dofs per node
      static constexpr int noddof_ = 3;

      //! total dofs per element
      static constexpr int numdof_ = noddof_ * numnod_;

      //! static const is required for fixedsizematrices
      static constexpr int numgpt_post_ = THR::DisTypeToNumGaussPoints<distype>::nquad;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

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
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        switch (distype)
        {
          case CORE::FE::CellType::tri3:
          {
            return Membrane_tri3Type::Instance().UniqueParObjectId();
          }
          case CORE::FE::CellType::tri6:
          {
            return Membrane_tri6Type::Instance().UniqueParObjectId();
          }
          case CORE::FE::CellType::quad4:
          {
            return Membrane_quad4Type::Instance().UniqueParObjectId();
          }
          case CORE::FE::CellType::quad9:
          {
            return Membrane_quad9Type::Instance().UniqueParObjectId();
          }
          default:
            dserror("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return -1;
      };

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

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
      virtual Teuchos::RCP<MAT::So3Material> SolidMaterial(int nummat = 0) const;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override { return noddof_; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override
      {
        switch (distype)
        {
          case CORE::FE::CellType::tri3:
          {
            return Membrane_tri3Type::Instance();
          }
          break;
          case CORE::FE::CellType::tri6:
          {
            return Membrane_tri6Type::Instance();
          }
          break;
          case CORE::FE::CellType::quad4:
          {
            return Membrane_quad4Type::Instance();
          }
          break;
          case CORE::FE::CellType::quad9:
          {
            return Membrane_quad9Type::Instance();
          }
          break;
          default:
            dserror("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return Membrane_quad4Type::Instance();
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
          INPUT::LineDefinition* linedef) override;

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
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          std::vector<int>& lm, CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
          CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
          CORE::LINALG::SerialDenseVector& elevec1_epetra,
          CORE::LINALG::SerialDenseVector& elevec2_epetra,
          CORE::LINALG::SerialDenseVector& elevec3_epetra) override;


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
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;


      //@}

     protected:
      /// Update history variables at the end of time step (fiber direction, inelastic deformation)
      /// (braeu 07/16)
      void Update_element(std::vector<double>& disp,  // current displacements
          Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
          Teuchos::RCP<MAT::Material> mat);           // material

     public:
      /** \brief set the parameter interface ptr for the solid elements
       *
       *  \param p (in): Parameter list coming from the time integrator.
       *
       *  \author hiermeier
       *  \date 04/16 */
      void SetParamsInterfacePtr(const Teuchos::ParameterList& p) override;

      /** \brief returns true if the parameter interface is defined and initialized, otherwise false
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline bool IsParamsInterface() const override { return (not interface_ptr_.is_null()); }

      /** \brief get access to the parameter interface pointer
       *
       *  \author hiermeier
       *  \date 04/16 */
      Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> ParamsInterfacePtr() override;

     protected:
      /** \brief get access to the interface
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline DRT::ELEMENTS::ParamsInterface& ParamsInterface()
      {
        if (not IsParamsInterface()) dserror("The interface ptr is not set!");
        return *interface_ptr_;
      }

      /** \brief get access to the structure interface
       *
       *  \author vuong
       *  \date 11/16 */
      STR::ELEMENTS::ParamsInterface& StrParamsInterface();

     private:
      /** \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> interface_ptr_;

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
      CORE::FE::IntegrationPoints2D intpoints_;

     private:
      // internal calculation methods

      // don't want = operator
      Membrane<distype>& operator=(const Membrane<distype>& old);

      //! calculate nonlinear stiffness and mass matrix
      void mem_nlnstiffmass(std::vector<int>& lm,               // location matrix
          std::vector<double>& disp,                            // current displacements
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
          CORE::LINALG::Matrix<numdof_, numdof_>* massmatrix,   // element mass matrix
          CORE::LINALG::Matrix<numdof_, 1>* force,              // element internal force vector
          CORE::LINALG::Matrix<numgpt_post_, 6>* elestress,     // stresses at GP
          CORE::LINALG::Matrix<numgpt_post_, 6>* elestrain,     // strains at GP
          Teuchos::ParameterList& params,                       // algorithmic parameters e.g. time
          const INPAR::STR::StressType iostress,                // stress output option
          const INPAR::STR::StrainType iostrain);               // strain output option

      //! get reference and current configuration
      void mem_configuration(const std::vector<double>& disp,
          CORE::LINALG::Matrix<numnod_, noddof_>& xrefe,
          CORE::LINALG::Matrix<numnod_, noddof_>& xcurr);

      //! introduce orthonormal base in the undeformed configuration at current Gauss point
      void mem_orthonormalbase(const CORE::LINALG::Matrix<numnod_, noddof_>& xrefe,
          const CORE::LINALG::Matrix<numnod_, noddof_>& xcurr,
          const CORE::LINALG::Matrix<numdim_, numnod_>& derivs,
          CORE::LINALG::Matrix<numdim_, numnod_>& derivs_ortho, double& G1G2_cn,
          CORE::LINALG::Matrix<noddof_, 1>& dXds1, CORE::LINALG::Matrix<noddof_, 1>& dXds2,
          CORE::LINALG::Matrix<noddof_, 1>& dxds1, CORE::LINALG::Matrix<noddof_, 1>& dxds2,
          CORE::LINALG::Matrix<noddof_, noddof_>& Q_localToGlobal) const;

      //! pushforward of 2nd Piola-Kirchhoff stresses to Cauchy stresses at Gauss point
      void mem_PK2toCauchy(const CORE::LINALG::Matrix<noddof_, noddof_>& pkstress_global,
          const CORE::LINALG::Matrix<noddof_, noddof_>& defgrd,
          CORE::LINALG::Matrix<noddof_, noddof_>& cauchy) const;

      // pushforward of Green-Lagrange to Euler-Almansi strains at Gauss point
      void mem_GLtoEA(const CORE::LINALG::Matrix<noddof_, noddof_>& glstrain_global,
          const CORE::LINALG::Matrix<noddof_, noddof_>& defgrd,
          CORE::LINALG::Matrix<noddof_, noddof_>& euler_almansi) const;

      // determine deformation gradient in global frame on membrane surface
      void mem_defgrd_global(const CORE::LINALG::Matrix<noddof_, 1>& dXds1,
          const CORE::LINALG::Matrix<noddof_, 1>& dXds2,
          const CORE::LINALG::Matrix<noddof_, 1>& dxds1,
          const CORE::LINALG::Matrix<noddof_, 1>& dxds2, const double& lambda3,
          CORE::LINALG::Matrix<noddof_, noddof_>& defgrd_global) const;

      // determine extrapolation matrix for postprocessing purposes
      CORE::LINALG::Matrix<CORE::FE::num_nodes<distype>,
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
    class Membrane_line2Type : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "Membrane_line2Type"; }

      static Membrane_line2Type& Instance();

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(
          DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        CORE::LINALG::SerialDenseMatrix nullspace;
        dserror("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static Membrane_line2Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  LINE 3 Element                                         fbraeu 06/16 |
     *----------------------------------------------------------------------*/
    class Membrane_line3Type : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "Membrane_line3Type"; }

      static Membrane_line3Type& Instance();

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(
          DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        CORE::LINALG::SerialDenseMatrix nullspace;
        dserror("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static Membrane_line3Type instance_;
    };

    /*!
    \brief An element representing a line edge of a membrane element

    \note This is a pure Neumann boundary condition element. It's only
          purpose is to evaluate line Neumann boundary conditions that might be
          adjacent to a parent membrane element. It therefore does not implement
          the DRT::Element::Evaluate method and does not have its own ElementRegister class.

    */
    template <CORE::FE::CellType distype2>
    class MembraneLine : public DRT::FaceElement
    {
     public:
      //! @name Friends
      friend class Membrane_line2Type;
      friend class Membrane_line3Type;

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
      MembraneLine(int id, int owner, int nnode, const int* nodeids, DRT::Node** nodes,
          DRT::ELEMENTS::Membrane<distype2>* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      MembraneLine(const MembraneLine<distype2>& old);

      //! number of nodes per line
      static constexpr int numnod_line_ =
          CORE::FE::num_nodes<CORE::FE::DisTypeToFaceShapeType<distype2>::shape>;

      static constexpr int noddof_ = 3;

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int UniqueParObjectId() const override
      {
        switch (CORE::FE::DisTypeToFaceShapeType<distype2>::shape)
        {
          case CORE::FE::CellType::line2:
          {
            return Membrane_line2Type::Instance().UniqueParObjectId();
          }
          case CORE::FE::CellType::line3:
          {
            return Membrane_line3Type::Instance().UniqueParObjectId();
          }
          default:
            dserror("unknown line type!");
            break;
        }
        // Intel compiler needs a return so
        return -1;
      };


      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

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
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override { return 3; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
       * \brief Return pointer to the parent element
       */
      virtual DRT::ELEMENTS::Membrane<distype2>* ParentElement() const
      {
        DRT::Element* parent = this->DRT::FaceElement::ParentElement();
        // make sure the static cast below is really valid
        dsassert(dynamic_cast<DRT::ELEMENTS::Membrane<distype2>*>(parent) != nullptr,
            "Parent element is no membrane element");
        return static_cast<DRT::ELEMENTS::Membrane<distype2>*>(parent);
      }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override
      {
        switch (CORE::FE::DisTypeToFaceShapeType<distype2>::shape)
        {
          case CORE::FE::CellType::line2:
          {
            return Membrane_line2Type::Instance();
          }
          case CORE::FE::CellType::line3:
          {
            return Membrane_line3Type::Instance();
          }
          default:
            dserror("unknown line type!");
            break;
        }
        // Intel compiler needs a return so
        return Membrane_line2Type::Instance();
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
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      // don't want = operator
      MembraneLine<distype2>& operator=(const MembraneLine<distype2>& old);

      CORE::FE::IntegrationPoints1D intpointsline_;

    };  // class MembraneLine

  }  // namespace ELEMENTS
}  // namespace DRT

FOUR_C_NAMESPACE_CLOSE

#endif
