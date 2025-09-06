// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_HPP
#define FOUR_C_W1_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    // forward declarations
    class Wall1Line;

    class Wall1Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Wall1Type"; }

      static Wall1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, int const numdof, int const dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static Wall1Type instance_;
    };

    /*======================================================================*/
    /*!
     * \brief A C++ wrapper for the wall1 element
     *
     * A planar solid element for geometrically non-linear analyses.
     *
     *
     * <b>Typical quantities and their storage pattern</b>
     * - Green-Lagrange strain vector
     *   \f$E = \left[\begin{array}{c} E_{11} \\ E_{22} \\ 2E_{12} \end{array}\right]\f$
     * - Deformation gradient vector
     *   \f$\tilde{F} = \left[\begin{array}{c} F_{11} \\ F_{22} \\ F_{12} \\ F_{21}
     * \end{array}\right]\f$
     * - Constitutive matrix
     *   \f$C = \left[\begin{array}{ccc} C_{11} & C_{12} & C_{13} \\ C_{21} & C_{22} & C_{23} \\
     * C_{31} & C_{32} & C_{33} \end{array}\right]\f$
     * - Second Piola--Kirchhoff stress vector
     *   \f$S = \left[\begin{array}{c} S_{11} \\ S_{22} \\ S_{12} \end{array}\right]\f$
     * - Second Piola--Kirchhoff stress matrix
     *   \f$\bar{S} = \left[\begin{array}{cccc} S_{11} & 0 & S_{12} & 0 \\ 0 & S_{22} & 0 & S_{12}
     * \\ S_{12} & 0 & S_{22} & 0 \\ 0 & S_{12} & 0 & S_{11} \end{array}\right]\f$
     * - First Piola-Kirchhoff stress vector
     *   \f$\tilde{P} = \left[\begin{array}{c} P_{11} \\ P_{22} \\ P_{12} \\ P_{21}
     * \end{array}\right]\f$
     *
     * <b>References</b>
     * - [1] JC Simo and F Armero, Geometrically non-linear enhanced strain
     *   mixed methods and the method of incompatible modes, International
     *   Journal for Numerical Methods in Engineering, 33:1413-1449, 1992.
     * - [2] S Glaser and F Armero, On the formulation of enhanced strain finite
     *   elements in finite deformations, Engineering Computations, 14:759-791,
     *   1997.
     * - [3] P Wriggers, Nichtlineare Finite-Element-Methods, Springer, 2001.
     * - [4] WA Wall and B Bornemann, Nichtlineare Finite-Element-Methods,
     *   Vorlesungsskript, Lehrstuhl fuer Numerische Mechanik, SS 2007.
     */
    class Wall1 : public Core::Elements::Element
    {
     public:
      /// @name Friends
      //@{
      friend class Wall1Line;
      //@}

      /// @name object-wide constants
      //@{
      static constexpr int maxnod_ = 9;  ///< maximally permitted number of nodes
      static constexpr int numdim_ = 2;  ///< 2-dimensional/planar element
      static constexpr int noddof_ = 2;  ///< number of nodal DOFs
      static constexpr int numstr_ =
          4;  ///< number of (symmetric) strain/stress components --- THIS SHOULD BE 3
      static constexpr int numnstr_ = 4;  ///< number of non-symmetric strain/stress components
      static constexpr int neas_ = 4;     ///< number of EAS parameters
      //@}

      /// @name Constructors and destructors and related methods

      /// Standard Constructor
      Wall1(int id,  ///< A unique global id
          int owner);

      /// Copy Constructor
      ///
      /// Makes a deep copy of a Element
      Wall1(const Wall1& old);

      /// Deep copy this instance of Wall1 and return pointer to the copy
      ///
      /// The clone() method is used from the virtual base class Element in cases
      /// where the type of the derived class is unknown and a copy-ctor is needed
      Core::Elements::Element* clone() const override;

      /// Get shape type of element
      Core::FE::CellType shape() const override;

      /// Set discretization type of element
      virtual void set_dis_type(Core::FE::CellType shape)
      {
        distype_ = shape;
        return;
      };


      /// Return number of lines of this element
      int num_line() const override
      {
        if (num_node() == 4 || num_node() == 8 || num_node() == 9)
          return 4;
        else
          return 3;
      }

      /// Return number of surfaces of this element
      int num_surface() const override { return 1; }

      /// Get vector of std::shared_ptrs to the lines of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /// Get vector of std::shared_ptrs to the surfaces of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      /// Return unique ParObject id
      ///
      /// every class implementing ParObject needs a unique id defined at the
      /// top of this file.
      int unique_par_object_id() const override
      {
        return Wall1Type::instance().unique_par_object_id();
      }

      /// Pack this class so it can be communicated
      ///
      /// \ref pack and \ref unpack are used to communicate this element
      void pack(Core::Communication::PackBuffer& data) const override;

      /// Unpack data from a char vector into this class
      ///
      /// \ref pack and \ref unpack are used to communicate this element
      void unpack(Core::Communication::UnpackBuffer& buffer) override;


      //@}

      /// @name Access methods
      //@{
      /*!
      \brief Does this element use EAS?
      */
      [[nodiscard]] bool have_eas() const { return (eastype_ != eas_vague); };

      /// Get number of degrees of freedom of a certain node
      /// (implements pure virtual Core::Elements::Element)
      ///
      /// The element decides how many degrees of freedom its nodes must have.
      /// As this may vary along a simulation, the element can redecide the
      /// number of degrees of freedom per node along the way for each of it's nodes
      /// separately.
      int num_dof_per_node(const Core::Nodes::Node& node) const override { return Wall1::noddof_; }

      /// Get number of degrees of freedom per element
      /// (implements pure virtual Core::Elements::Element)
      ///
      /// The element decides how many element degrees of freedom it has.
      /// It can redecide along the way of a simulation.
      ///
      /// \note Element degrees of freedom mentioned here are dofs that are visible
      ///      at the level of the total system of equations. Purely internal
      ///      element dofs that are condensed internally should NOT be considered.
      int num_dof_per_element() const override { return 0; }

      Core::Elements::ElementType& element_type() const override { return Wall1Type::instance(); }

      //@}

      /// @name Input and Creation
      //@{

      /// Read input for this element
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      //@}

      /// @name Evaluation
      //@{

      /// Evaluate an element
      ///
      /// Evaluate Wall1 element stiffness, mass, internal forces etc
      ///
      /// \return 0 if successful, negative otherwise
      int evaluate(Teuchos::ParameterList& params,  ///< (in/out) ParameterList for communication
                                                    ///< between control routine and elements
          Core::FE::Discretization&
              discretization,    ///< A reference to the underlying discretization
          std::vector<int>& lm,  ///< location vector of this element
          Core::LinAlg::SerialDenseMatrix& elemat1,  ///< matrix to be filled by element. If nullptr
                                                     ///< on input, the controlling method does not
                                                     ///< epxect the element to fill this matrix.
          Core::LinAlg::SerialDenseMatrix& elemat2,  ///< matrix to be filled by element. If nullptr
                                                     ///< on input, the controlling method does not
                                                     ///< epxect the element to fill this matrix.
          Core::LinAlg::SerialDenseVector& elevec1,  ///< vector to be filled by element. If nullptr
                                                     ///< on input, the controlling method does not
                                                     ///< epxect the element to fill this vector
          Core::LinAlg::SerialDenseVector& elevec2,  ///< vector to be filled by element. If nullptr
                                                     ///< on input, the controlling method does not
                                                     ///< epxect the element to fill this vector
          Core::LinAlg::SerialDenseVector& elevec3   ///< vector to be filled by element. If nullptr
                                                     ///< on input, the controlling method does not
                                                     ///< epxect the element to fill this vector
          ) override;

      /// Evaluate a Neumann boundary condition
      ///
      /// this method evaluates a surfaces Neumann condition on the wall element
      ///
      /// \return 0 if successful, negative otherwise
      int evaluate_neumann(
          Teuchos::ParameterList& params,  ///< (in/out) ParameterList for communication between
                                           ///< control routine and elements
          Core::FE::Discretization&
              discretization,  ///< A reference to the underlying discretization
          const Core::Conditions::Condition& condition,  ///<  The condition to be evaluated
          std::vector<int>& lm,                          ///< location vector of this element
          Core::LinAlg::SerialDenseVector&
              elevec1,  ///< vector to be filled by element. If nullptr on input
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

      /*!
      \brief Return the material of this element

      Note: The input parameter nummat is not the material number from input file
            as in set_material(int matnum), but the number of the material within
            the vector of materials the element holds

      \param nummat (in): number of requested material
      */
      std::shared_ptr<Mat::So3Material> solid_material(int nummat = 0) const;

      /** \brief get access to the interface
       *
       */
      inline Core::Elements::ParamsInterface& params_interface()
      {
        if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
        return *interface_ptr_;
      }

      [[nodiscard]] inline bool is_params_interface() const override
      {
        return (interface_ptr_ != nullptr);
      }

      std::shared_ptr<Core::Elements::ParamsInterface> params_interface_ptr() override;

      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      FourC::Solid::Elements::ParamsInterface& str_params_interface();

      // get the kinematic type from the element
      [[nodiscard]] Inpar::Solid::KinemType kinematic_type() const { return kintype_; }

     protected:
      /// type of 2D dimension reduction
      enum DimensionalReduction
      {
        plane_none,    ///< undetermined
        plane_stress,  ///< plane stress, i.e. lateral stress is zero \f$S_{33}=S_{13}=S_{23}=0\f$
        plane_strain   ///< plane strain, i.e. lateral strain is zero \f$E_{33}=E_{13}=E_{23}=0\f$
      };

      /// type of stresses to be calculatecd in postprocessing
      ///
      /// describes in co-ordinate system the stress components are expressed
      enum StressType
      {
        w1_none,  ///< undetermined
        w1_xy,    ///< Cartesian
        w1_rs     ///< local/parametric/natural
      };

      /// EAS type
      enum EasType
      {
        eas_vague,  ///< unknown type
        eas_q1e4,   ///< Q1E4 formulation due to Simo and Armero [1]
        eas_q1et4   ///< Q1ET4 formulation due to Glaser and Armero [2]
      };

      /// number of the material law
      int material_;
      /// element thickness
      double thickness_;
      // line search parameter (old step length)
      double old_step_length_;
      /// gaussian points
      Core::FE::GaussRule2D gaussrule_;
      /// problem type
      DimensionalReduction wtype_;
      /// type of stress calculation
      StressType stresstype_;
      /// eas or not
      bool iseas_;
      /// EAS type
      enum EasType eastype_;

      /** \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      std::shared_ptr<Core::Elements::ParamsInterface> interface_ptr_ = nullptr;

      //! Flag of the status of the material post setup routine
      bool material_post_setup_ = false;

      //! kinematic type
      Inpar::Solid::KinemType kintype_ = Inpar::Solid::KinemType::vague;

      struct EASData
      {
        Core::LinAlg::SerialDenseMatrix alpha{};
        Core::LinAlg::SerialDenseMatrix alphao{};
        Core::LinAlg::SerialDenseMatrix feas{};
        Core::LinAlg::SerialDenseMatrix invKaa{};
        Core::LinAlg::SerialDenseMatrix Kda{};
        Core::LinAlg::SerialDenseMatrix Kad{};
        Core::LinAlg::SerialDenseMatrix eas_inc{};
      };

      // EAS data
      EASData easdata_;

      //! the element discretization type (shape)
      Core::FE::CellType distype_;

      /// @name Internal calculation methods
      //@{
      /** recover elementwise stored stuff
       *
       */
      void w1_recover(const std::vector<int>& lm, const std::vector<double>& disp,
          const std::vector<double>& residual);

      /// evaluate the element forces and stiffness and mass
      void w1_nlnstiffmass(const std::vector<int>& lm,  ///< location vector
          const std::vector<double>& disp,              ///< element displacements
          const std::vector<double>& residual,          ///< residual displacements
          std::vector<Core::LinAlg::SerialDenseVector>&
              myknots,                                          ///< knot vector for nurbs elements
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,         ///< element stiffness matrix
          Core::LinAlg::SerialDenseMatrix* massmatrix,          ///< element mass matrix
          Core::LinAlg::SerialDenseVector* force,               ///< element internal force vector
          Core::LinAlg::SerialDenseMatrix* elestress,           ///< element stresses
          Core::LinAlg::SerialDenseMatrix* elestrain,           ///< element strains
          std::shared_ptr<const Core::Mat::Material> material,  ///< element material
          Teuchos::ParameterList& params,           ///< algorithmic parameters e.g. time
          const Inpar::Solid::StressType iostress,  ///< stress output option
          const Inpar::Solid::StrainType iostrain   ///< strain output option
      );

      /// evaluate the geometrically linear element forces and stiffness and mass
      void w1_linstiffmass(const std::vector<int>& lm,  ///< location vector
          const std::vector<double>& disp,              ///< element displacements
          const std::vector<double>& residual,          ///< residual displacements
          std::vector<Core::LinAlg::SerialDenseVector>&
              myknots,                                          ///< knot vector for nurbs elements
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,         ///< element stiffness matrix
          Core::LinAlg::SerialDenseMatrix* massmatrix,          ///< element mass matrix
          Core::LinAlg::SerialDenseVector* force,               ///< element internal force vector
          Core::LinAlg::SerialDenseMatrix* elestress,           ///< element stresses
          Core::LinAlg::SerialDenseMatrix* elestrain,           ///< element strains
          std::shared_ptr<const Core::Mat::Material> material,  ///< element material
          Teuchos::ParameterList& params,           ///< algorithmic parameters e.g. time
          const Inpar::Solid::StressType iostress,  ///< stress output option
          const Inpar::Solid::StrainType iostrain   ///< strain output option
      );

      /// Jacobian matrix for mapping from parameter space in physical material space
      /// at point parameter space
      void w1_jacobianmatrix(const Core::LinAlg::SerialDenseMatrix&
                                 xrefe,  ///< reference/material co-ordinates of element nodes
          const Core::LinAlg::SerialDenseMatrix&
              deriv,  ///< derivatives of shape functions at parameter point
          Core::LinAlg::SerialDenseMatrix& xjm,  ///< Jacobi matrix
          double* det,                           ///< determinant of Jacobi matrix
          const int iel                          ///< actual number of element nodes
      );

      /// Linear B-operator in reference configuration at point parameter space
      void w1_boplin(Core::LinAlg::SerialDenseMatrix& boplin,  ///< the B-operator
          Core::LinAlg::SerialDenseMatrix&
              deriv,  ///< derivatives of shape functions at parameter point
          Core::LinAlg::SerialDenseMatrix& xjm,  ///< Jacobian at parameter point
          double& det,                           ///< Jacobian determinant at parameter point
          const int iel                          ///< number of element nodes
      );

      /// (Material) Deformation gradient \f$F\f$ and Green-Lagrange strains \f$E\f$
      /// at parameter point
      void w1_defgrad(Core::LinAlg::SerialDenseVector& F,  ///< deformation gradient
          Core::LinAlg::SerialDenseVector&
              strain,  ///< GL strain \f$E^T=[E_{11} \; E_{22} \; E_{12}]\f$
          const Core::LinAlg::SerialDenseMatrix&
              xrefe,  ///< reference/material co-ordinates of element nodes
          const Core::LinAlg::SerialDenseMatrix&
              xcure,  ///< current/spatial co-ordinates of element nodes
          Core::LinAlg::SerialDenseMatrix& boplin,  ///< linear B-operator
          const int iel                             ///< number of element nodes
      );

      /// Non-linear B-operator in reference configuration
      void w1_boplin_cure(Core::LinAlg::SerialDenseMatrix& b_cure,  ///< non-linear B-operator
          const Core::LinAlg::SerialDenseMatrix& boplin,            ///< linear B-operator
          const Core::LinAlg::SerialDenseVector& F,  ///< deformation gradient as Voigt-vector
          const int numeps,                          ///< number of strains
          const int nd                               ///< number of element nodes
      );

      /// Geometric stiffness constribution (total Lagrange)
      void w1_kg(Core::LinAlg::SerialDenseMatrix& estif,  ///< (in/out) element stiffness matrix
          const Core::LinAlg::SerialDenseMatrix& boplin,  ///< linear B-operator
          const Core::LinAlg::SerialDenseMatrix& stress,  ///< PK2 stress vector
          const double fac,                               ///< integration factor
          const int nd,                                   ///< number of element DOFs
          const int numeps                                ///< number of strains
      );

      /// elastic and initial displacement stiffness contribution (total Lagrange)
      void w1_keu(Core::LinAlg::SerialDenseMatrix& estif,  ///< (in/out) element stiffness matrix
          const Core::LinAlg::SerialDenseMatrix& b_cure,   ///< non-linear B-operator
          const Core::LinAlg::SerialDenseMatrix& C,        ///< elasticity matrix
          const double fac,                                ///< integration factor
          const int nd,                                    ///< number of element DOFs
          const int numeps                                 ///< number of strains
      );

      /// Evaluate internal element forces for large def (total Lagr)
      void w1_fint(const Core::LinAlg::SerialDenseMatrix& stress,  ///< PK2 stress vector
          const Core::LinAlg::SerialDenseMatrix& b_cure,           ///< non-linear B-op
          Core::LinAlg::SerialDenseVector& intforce,               ///< internal force vector
          const double fac,                                        ///< integration factor
          const int nd                                             ///< number of element DOFs
      );

      /// lump mass matrix
      void w1_lumpmass(Core::LinAlg::SerialDenseMatrix* emass  ///< (in/out) element mass matrix
      );

      /// determine cauchy stress and store it
      void stress_cauchy(const int ip,                    ///< Gauss point index
          const double& F11,                              ///< F_{11} component of def.grad.
          const double& F22,                              ///< F_{22} component of def.grad.
          const double& F12,                              ///< F_{12} component of def.grad.
          const double& F21,                              ///< F_{21} component of def.grad.
          const Core::LinAlg::SerialDenseMatrix& stress,  ///< PK2 stress matrix
          Core::LinAlg::SerialDenseMatrix* elestress      ///< element stress array
      );

      /// determine internal energy
      void energy(Teuchos::ParameterList& params,  ///<  a list containing GEMM coefficients
          const std::vector<int>& lm,              ///< location vector
          const std::vector<double>&
              dis,  ///< element displacements \f$d_{n}^{(e)}\f$ at \f$t_{n}\f$
          Core::LinAlg::SerialDenseVector* energies,           ///< (in/out) energies
          std::shared_ptr<const Core::Mat::Material> material  ///< element material
      );

      //@}

      /// @name EAS tools
      //@{
      /// return the number of eas dofs
      int w1_neas() const { return neas_; };

      /// setup of constant EAS data
      void w1_eassetup(Core::LinAlg::SerialDenseMatrix& boplin0,  ///< linear B-op at origin
          Core::LinAlg::SerialDenseVector& F0,           ///< deformation gradient at origin
          Core::LinAlg::SerialDenseMatrix& xjm0,         ///< jacobian matrix at origin
          double& detJ0,                                 ///< det of Jacobian at origin
          const Core::LinAlg::SerialDenseMatrix& xrefe,  ///< material element coords
          const Core::LinAlg::SerialDenseMatrix& xcure,  ///< current element coords
          const Core::FE::CellType& distype              ///< discretisation type
      );

      /// Get the enhanced deformation gradient and
      /// also the operators G,W0 and Z
      /// at point in parameter space
      void w1_call_defgrad_enh(
          Core::LinAlg::SerialDenseMatrix& F_enh,       ///< enhanced deformation gradient vector
          const Core::LinAlg::SerialDenseMatrix xjm0,   ///< Jacobian at origin
          const Core::LinAlg::SerialDenseMatrix xjm,    ///< Jacobian at parameter point
          const double detJ0,                           ///< det of Jacobian at origin
          const double det,                             ///< det of Jacobian at parameter point
          const Core::LinAlg::SerialDenseVector F0,     ///< deformation gradient at origin
          const Core::LinAlg::SerialDenseMatrix alpha,  ///< alpha parameters (EAS params)
          const double e1,                      ///< \f$\xi\f$ co-ordinate of parameter point
          const double e2,                      ///< \f$\eta\f$ co-ordinate of parameter point
          Core::LinAlg::SerialDenseMatrix& G,   ///< G-operator
          Core::LinAlg::SerialDenseMatrix& W0,  ///< W_0-operator
          const Core::LinAlg::SerialDenseMatrix boplin0,  ///< linear B-op at origin
          Core::LinAlg::SerialDenseMatrix& Z              ///< Z-operator
      );

      /// Total deformation gradient and (enhanced) Green-Lagrange strain
      void w1_call_defgrad_tot(
          const Core::LinAlg::SerialDenseMatrix& F_enh,  ///< enhanced def.grad.
          Core::LinAlg::SerialDenseMatrix& F_tot,        ///< total def.grad.
          const Core::LinAlg::SerialDenseVector& F,      ///< displ.-based def.grad.
          Core::LinAlg::SerialDenseVector& strain        ///< GL strains
      );

      /// first Piola-Kirchhoff stress vector
      void w1_stress_eas(const Core::LinAlg::SerialDenseMatrix& stress,  ///< PK2 stress vector
          const Core::LinAlg::SerialDenseMatrix& F_tot,                  ///< total def.grad.
          Core::LinAlg::SerialDenseMatrix& p_stress                      ///< PK1 stress vector
      );

      /// calculate stiffness matrix kdd \f$\partial f_{int}/\partial d\f$
      void w1_kdd(const Core::LinAlg::SerialDenseMatrix& boplin,  ///< linear B-op
          const Core::LinAlg::SerialDenseMatrix& W0,              ///< W-operator at origin
          const Core::LinAlg::SerialDenseMatrix& F,               ///< total def.grad.
          const Core::LinAlg::SerialDenseMatrix& C,               ///< consititutive matrix
          const Core::LinAlg::SerialDenseMatrix& stress,          ///< PK2 stress vector
          Core::LinAlg::SerialDenseMatrix& FCF,                   ///< \f$F^T \dot C \dot F\f$
          Core::LinAlg::SerialDenseMatrix& estif,                 ///< element stiff matrix
          const double fac                                        ///< integration factor
      );

      /// calculate tangential matrix kda \f$\partial f_{int}/\partial \alpha\f$
      void w1_kda(const Core::LinAlg::SerialDenseMatrix& FCF,  ///< a product
          const Core::LinAlg::SerialDenseMatrix& W0,           ///< W-operator at origin
          const Core::LinAlg::SerialDenseMatrix& boplin,       ///< linear B-op
          const Core::LinAlg::SerialDenseMatrix& stress,       ///< PK2 stress
          const Core::LinAlg::SerialDenseMatrix& G,            ///< G-operator
          const Core::LinAlg::SerialDenseMatrix& Z,            ///< Z-operator
          Core::LinAlg::SerialDenseMatrix& Kda,                ///< target: kda
          const Core::LinAlg::SerialDenseMatrix& p_stress,     ///< PK1 stress
          const double fac                                     ///< integration factor
      );

      /// calculate tangential matrix kaa \f$\partial s/\partial \alpha\f$
      void w1_kaa(
          const Core::LinAlg::SerialDenseMatrix& FCF,     ///< a product \f$F^T \dot C \dot F\f$
          const Core::LinAlg::SerialDenseMatrix& stress,  ///< PK2 stress
          const Core::LinAlg::SerialDenseMatrix& G,       ///< G-op
          Core::LinAlg::SerialDenseMatrix& Kaa,           ///< target: kaa
          const double fac                                ///< integration factor
      );

      /// calculate internal forces fint(displacements u) and feas
      void w1_fint_eas(const Core::LinAlg::SerialDenseMatrix& W0,  ///< W-op at origin
          const Core::LinAlg::SerialDenseMatrix& boplin,           ///< linear B-op
          const Core::LinAlg::SerialDenseMatrix& G,                ///< G-op
          const Core::LinAlg::SerialDenseMatrix& p_stress,         ///< PK1 stress
          Core::LinAlg::SerialDenseVector& intforce,               ///< internal force vector
          Core::LinAlg::SerialDenseVector& feas,  ///< internal EAS constraint condition
          const double fac                        ///< integration factor
      );

      void pack_eas_data(Core::Communication::PackBuffer& data) const
      {
        add_to_pack(data, easdata_.alpha);
        add_to_pack(data, easdata_.alphao);
        add_to_pack(data, easdata_.feas);
        add_to_pack(data, easdata_.invKaa);
        add_to_pack(data, easdata_.Kda);
        add_to_pack(data, easdata_.Kad);
        add_to_pack(data, easdata_.eas_inc);
      }

      void unpack_eas_data(Core::Communication::UnpackBuffer& buffer)
      {
        extract_from_pack(buffer, easdata_.alpha);
        extract_from_pack(buffer, easdata_.alphao);
        extract_from_pack(buffer, easdata_.feas);
        extract_from_pack(buffer, easdata_.invKaa);
        extract_from_pack(buffer, easdata_.Kda);
        extract_from_pack(buffer, easdata_.Kad);
        extract_from_pack(buffer, easdata_.eas_inc);
      }
      //@}

      /// @name Material matters
      //@{

      /// Constitutive matrix \f$C\f$ and stresses
      void w1_call_matgeononl(
          const Core::LinAlg::SerialDenseVector& strain,        ///< Green-Lagrange strain vector
          Core::LinAlg::SerialDenseMatrix& stress,              ///< stress matrix
          Core::LinAlg::SerialDenseMatrix& C,                   ///< elasticity matrix
          const int numeps,                                     ///< number of strains
          std::shared_ptr<const Core::Mat::Material> material,  ///< the material data
          Teuchos::ParameterList& params,                       ///< element parameter list
          int gp                                                ///< Gauss point
      );

      /// Stress and constitutive matrix mapper from 3d to 2d
      /// due to dimensional reduction #wtype_
      ///
      void material_response3d_plane(Core::LinAlg::SerialDenseMatrix& stress,  ///< stress matrix
          Core::LinAlg::SerialDenseMatrix& C,             ///< elasticity matrix
          const Core::LinAlg::SerialDenseVector& strain,  ///< Green-Lagrange strain vector
          Teuchos::ParameterList& params,                 ///< element parameter list
          int gp                                          ///< Gauss point
      );

      /// Generic 3D stress response
      ///
      void material_response3d(Core::LinAlg::SymmetricTensor<double, 3, 3>&
                                   stress,  ///< 3D 2nd Piola-Kirchhoff stress vector
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< 3D elasticity matrix
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              glstrain,                    ///< 3D Green-Lagrange strain vector
          Teuchos::ParameterList& params,  ///< element parameter list
          int gp                           ///< Gauss point
      );

      /// Map plane Green-Lagrange strains to 3d
      void green_lagrange_plane3d(const Core::LinAlg::SerialDenseVector&
                                      glplane,  ///< 2d version of GL strain (Voigt notation)
          Core::LinAlg::SymmetricTensor<double, 3, 3>&
              gl3d  ///< 3d version of GL strain (Voigt notation)
      );

      /// Internal/strain energy
      double energy_internal(
          std::shared_ptr<const Core::Mat::Material> material,  ///< element material
          Teuchos::ParameterList& params,                       ///< element parameter list
          const Core::LinAlg::SerialDenseVector& Ev,            ///< Green-Lagrange strain vector
          int gp                                                ///< Gauss point
      );

      /// Kinetic Energy
      /// \return kinetic energy of element
      double energy_kinetic(const Core::LinAlg::SerialDenseMatrix& mass,  ///< element mass matrix
          const std::vector<double>& vel  ///< element velocity vector
      );

      //@}



      /// don't want = operator
      Wall1& operator=(const Wall1& old);


      /// set number of gauss points to element shape default
      Core::FE::GaussRule2D get_gaussrule(int* ngp  ///< number of Gauss points
      );

      /*!
       * \brief This method executes the material_post_setup if not already executed.
       *
       * This method should be placed in the Evaluate call. It will internally check, whether the
       * material post_setup() routine was already called in if not, it invokes this call directly.
       *
       * @param params Container for additional information
       */
      void ensure_material_post_setup(Teuchos::ParameterList& params);

      /*!
       * \brief This method calls the post_setup routine of all materials.
       *
       * It can be used to pass information from the element to the materials after everything
       * is set up. For a simple element, the ParameterList is passed unchanged to the materials.
       */
      virtual void material_post_setup(Teuchos::ParameterList& params);
    };  // class Wall1



    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================

    class Wall1LineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Wall1LineType"; }

      static Wall1LineType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& actnode, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static Wall1LineType instance_;
    };


    /*======================================================================*/
    /*!
     * \brief An element representing a line edge of a Wall1 element
     *
     * \note This is a pure Neumann boundary condition element. It's only
     *      purpose is to evaluate line Neumann boundary conditions that might be
     *      adjacent to a parent wall1 element. It therefore does not implement
     *      the Core::Elements::Element::Evaluate method and does not have its own ElementRegister
     * class.
     *
     */
    class Wall1Line : public Core::Elements::FaceElement
    {
     public:
      /// @name Constructors and destructors and related methods
      //@{

      /// Standard Constructor
      Wall1Line(int id,        ///< A unique global id
          int owner,           ///< Processor owning this line
          int nnode,           ///< Number of nodes attached to this element
          const int* nodeids,  ///< global ids of nodes attached to this element
          Core::Nodes::Node**
              nodes,  ///<  the discretizations map of nodes to build ptrs to nodes from
          Discret::Elements::Wall1* parent,  ///< The parent wall element of this line
          const int lline  ///< the local line number of this line w.r.t. the parent element
      );

      /// Copy Constructor
      ///
      /// Makes a deep copy of a Element
      Wall1Line(const Wall1Line& old);


      /// Deep copy this instance of an element and return pointer to the copy
      ///
      /// The clone() method is used from the virtual base class Element in cases
      /// where the type of the derived class is unknown and a copy-ctor is needed
      Core::Elements::Element* clone() const override;


      /// Get shape type of element
      Core::FE::CellType shape() const override;


      /// Return unique ParObject id
      ///
      /// every class implementing ParObject needs a unique id defined at the
      /// top of the parobject.H file.
      int unique_par_object_id() const override
      {
        return Wall1LineType::instance().unique_par_object_id();
      }

      /// Pack this class so it can be communicated
      ///
      /// \ref pack and \ref unpack are used to communicate this element
      void pack(Core::Communication::PackBuffer& data) const override;

      /// Unpack data from a char vector into this class
      ///
      /// \ref pack and \ref unpack are used to communicate this element
      void unpack(Core::Communication::UnpackBuffer& buffer) override;


      //@}

      /// @name Access methods
      //@{

      /// Get number of degrees of freedom of a certain node
      /// (implements pure virtual Core::Elements::Element)
      ///
      /// The element decides how many degrees of freedom its nodes must have.
      /// As this may vary along a simulation, the element can redecide the
      /// number of degrees of freedom per node along the way for each of it's nodes
      /// separately.
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        return parent_master_element()->num_dof_per_node(node);
      }

      /// Get number of degrees of freedom per element
      /// (implements pure virtual Core::Elements::Element)
      ///
      /// The element decides how many element degrees of freedom it has.
      /// It can redecide along the way of a simulation.
      ///
      /// \note Element degrees of freedom mentioned here are dofs that are visible
      ///       at the level of the total system of equations. Purely internal
      ///       element dofs that are condensed internally should NOT be considered.
      int num_dof_per_element() const override { return 0; }

      /// Print this element
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return Wall1LineType::instance();
      }

      //@}

      /// @name Evaluate methods
      //@{

      /// Evaluate a Neumann boundary condition
      ///
      /// this method evaluates a line Neumann condition on the wall element
      ///
      /// \return 0 if successful, negative otherwise
      int evaluate_neumann(
          Teuchos::ParameterList&
              params,  ///< (in/out) ParameterList for communication between control routine
                       ///<   and elements
          Core::FE::Discretization&
              discretization,  ///< (in) A reference to the underlying discretization
          const Core::Conditions::Condition& condition,  ///< (in) The condition to be evaluated
          std::vector<int>& lm,                          ///< (in) location vector of this element
          Core::LinAlg::SerialDenseVector&
              elevec1,  ///< (out) vector to be filled by element. If nullptr on input
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      /// Evaluate a boundary condition
      ///
      /// this method evaluates a line Neumann condition on the wall element
      ///
      /// \return 0 if successful, negative otherwise
      int evaluate(Teuchos::ParameterList& params,  ///< (in/out) ParameterList for communication
                                                    ///< between control routine and elements
          Core::FE::Discretization&
              discretization,    ///< (in) A reference to the underlying discretization
          std::vector<int>& lm,  ///< (in) location vector of this element
          Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
          Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2,
          Core::LinAlg::SerialDenseVector& elevector3) override;

      //! Evaluate method on multiple dofsets for wall element
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elematrix1,
          Core::LinAlg::SerialDenseMatrix& elematrix2, Core::LinAlg::SerialDenseVector& elevector1,
          Core::LinAlg::SerialDenseVector& elevector2,
          Core::LinAlg::SerialDenseVector& elevector3) override;

      //@}

     private:
      /// digestible actions
      enum ActionType
      {
        none,  ///< undetermined
        calc_struct_areaconstrstiff,
        calc_struct_centerdisp,
        calc_struct_constrarea,
        calc_struct_area_poro
      };

      /// don't want = operator
      Wall1Line& operator=(const Wall1Line& old);

      /// Submethod to compute necessary change to stiffness matrix due to the constraint
      void compute_area_constr_stiff(
          Core::LinAlg::SerialDenseMatrix xscurr, Core::LinAlg::SerialDenseMatrix& elematrix);

      /// Submethod to compute first derivatives of constraint area w.r.t. the displacements
      void compute_area_constr_deriv(
          Core::LinAlg::SerialDenseMatrix xscurr, Core::LinAlg::SerialDenseVector& elevector);

      /// compute infinitesimal line element dr for integration along the line
      double w1_substitution(const Core::LinAlg::SerialDenseMatrix& xyze,
          const Core::LinAlg::SerialDenseMatrix& deriv, std::vector<double>* unrm, const int iel);

      /// set number of gauss points to element shape default
      Core::FE::GaussRule1D get_optimal_gaussrule(const Core::FE::CellType& distype);

    };  // class Wall1Line



  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
