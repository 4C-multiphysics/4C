// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_ALE2_HPP
#define FOUR_C_ALE_ALE2_HPP

/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_config.hpp"

#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* forward declarations */
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class Ale2Line;

    /*----------------------------------------------------------------------------*/
    /* definition of classes */
    class Ale2Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Ale2Type"; }

      static Ale2Type& instance();

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
      static Ale2Type instance_;
    };

    /*----------------------------------------------------------------------------*/
    /*!
    \brief A C++ wrapper for the ale2 element
    */
    class Ale2 : public Core::Elements::Element
    {
     public:
      //! @name Friends
      friend class Ale2Surface;

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      */
      Ale2(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale2(const Ale2& old);

      /*!
      \brief Deep copy this instance of Ale2 and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int num_line() const override
      {
        if (num_node() == 9 || num_node() == 8 || num_node() == 4)
          return 4;
        else if (num_node() == 3 || num_node() == 6)
          return 3;
        else
        {
          FOUR_C_THROW("Could not determine number of lines");
          return -1;
        }
      }

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override { return 1; }

      /*!
      \brief Return number of volumes of this element (always 1)
      */
      int num_volume() const override { return -1; }

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
        return Ale2Type::instance().unique_par_object_id();
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
      int num_dof_per_node(const Core::Nodes::Node& node) const override { return 2; }

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

      Core::Elements::ElementType& element_type() const override { return Ale2Type::instance(); }

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
      \brief Evaluate an element

      Evaluate ale2 element stiffness, mass, internal forces etc

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
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
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;


      //@}

      //! @name Other

      /// set number of gauss points to element shape default
      Core::FE::GaussRule2D get_optimal_gaussrule(const Core::FE::CellType& distype);

      //@}


     private:
      /*! \brief ALE mesh motion via Laplacian smoohting
       *
       *  Solve Laplace equation
       *  \f[
       *    \nabla_{config}\cdot\nabla_{config} d = 0, \quad d = \hat d \text{ on } \Gamma
       *  \f]
       *  with \f$\nabla_{config}\f$ denoting the material or spatial gradient
       *  operator and the displacement field \f$d\f$ and satisfying prescribed
       *  displacement on the entire boundary \f$\Gamma\f$.
       *
       *  \note For spatial configuration, displacement vector equals current
       *  displacements. For material configuration, displacement vector is zero.
       */
      void static_ke_laplace(Core::FE::Discretization& dis,  ///< discretization
          std::vector<int>& lm,                              ///< node owning procs
          Core::LinAlg::SerialDenseMatrix* sys_mat,   ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          std::vector<double>& displacements,         ///< nodal discplacements
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
      );

      void static_ke_spring(
          Core::LinAlg::SerialDenseMatrix* sys_mat,   ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          std::vector<double>& displacements,         ///, nodal displacements
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
      );

      /*! \brief evaluate element quantities for nonlinear case
       *
       *  We can handle St.-Venant-Kirchhoff material and many materials
       *  from the elast hyper toolbox.
       *
       *  \note this routine was copied from Discret::Elements::Wall1
       */
      void static_ke_nonlinear(const std::vector<int>& lm,  ///< node owning procs
          const std::vector<double>& disp,                  ///< nodal displacements
          Core::LinAlg::SerialDenseMatrix*
              stiffmatrix,                         ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector* force,  ///< element residual vector (to be filled)
          Teuchos::ParameterList& params,          ///< parameter list
          const bool spatialconfiguration,         ///< use spatial configuration (true), material
                                                   ///< configuration (false)
          const bool pseudolinear  ///< compute residual as stiffness * displacements (pseudo-linear
                                   ///< approach)
      );

      /*! \brief Calculate determinant of Jacobian mapping and check for validity
       *
       *  Use current displacement state, i.e. current/spatial configuration.
       *  Check for invalid mappings (detJ <= 0)
       *
       */
      void compute_det_jac(
          Core::LinAlg::SerialDenseVector& elevec1,  ///< vector with element result data
          const std::vector<int>& lm,                ///< node owning procs
          const std::vector<double>& disp            ///< nodal displacements
      );

      /*! \brief Element quality measure according to [Oddy et al. 1988a]
       *
       *  Distortion metric for quadrilaterals and hexahedrals. Value is zero for
       *  squares/cubes and increases to large values for distorted elements.
       *
       *  Reference: Oddy A, Goldak J, McDill M, Bibby M (1988): A distortion metric
       *  for isoparametric finite elements, Trans. Can. Soc. Mech. Engrg.,
       *  Vol. 12 (4), pp. 213-217
       */
      void evaluate_oddy(const Core::LinAlg::SerialDenseMatrix& xjm, double det, double& qm);

      void call_mat_geo_nonl(
          const Core::LinAlg::SerialDenseVector& strain,        ///< Green-Lagrange strain vector
          Core::LinAlg::SerialDenseMatrix& stress,              ///< stress vector
          Core::LinAlg::SerialDenseMatrix& C,                   ///< elasticity matrix
          const int numeps,                                     ///< number of strains
          std::shared_ptr<const Core::Mat::Material> material,  ///< the material data
          Teuchos::ParameterList& params,                       ///< element parameter list
          int gp                                                ///< Integration point
      );

      void material_response3d_plane(
          Core::LinAlg::SerialDenseMatrix& stress,        ///< stress state (output)
          Core::LinAlg::SerialDenseMatrix& C,             ///< material tensor (output)
          const Core::LinAlg::SerialDenseVector& strain,  ///< strain state (input)
          Teuchos::ParameterList& params,                 ///< parameter list
          int gp                                          ///< Integration point
      );

      void material_response3d(Core::LinAlg::Matrix<6, 1>* stress,  ///< stress state (output)
          Core::LinAlg::Matrix<6, 6>* cmat,                         ///< material tensor (output)
          const Core::LinAlg::Matrix<6, 1>* glstrain,               ///< strain state (input)
          Teuchos::ParameterList& params,                           ///< parameter list
          int gp                                                    ///< Integration point
      );

      //! Transform Green-Lagrange notation from 2D to 3D
      void green_lagrange_plane3d(const Core::LinAlg::SerialDenseVector&
                                      glplane,  ///< Green-Lagrange strains in 2D notation
          Core::LinAlg::Matrix<6, 1>& gl3d);    ///< Green-Lagrange strains in 2D notation

      void edge_geometry(int i, int j, const Core::LinAlg::SerialDenseMatrix& xyze, double* length,
          double* sin_alpha, double* cos_alpha);
      double ale2_area_tria(const Core::LinAlg::SerialDenseMatrix& xyze, int i, int j, int k);
      void ale2_tors_spring_tri3(
          Core::LinAlg::SerialDenseMatrix* sys_mat, const Core::LinAlg::SerialDenseMatrix& xyze);
      void ale2_tors_spring_quad4(
          Core::LinAlg::SerialDenseMatrix* sys_mat, const Core::LinAlg::SerialDenseMatrix& xyze);
      void ale2_torsional(int i, int j, int k, const Core::LinAlg::SerialDenseMatrix& xyze,
          Core::LinAlg::SerialDenseMatrix* k_torsion);

      void calc_b_op_lin(Core::LinAlg::SerialDenseMatrix& boplin,
          Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm, double& det,
          const int iel);

      void b_op_lin_cure(Core::LinAlg::SerialDenseMatrix& b_cure,
          const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseVector& F,
          const int numeps, const int nd);

      void jacobian_matrix(const Core::LinAlg::SerialDenseMatrix& xrefe,
          const Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm,
          double* det, const int iel);

      ///! Compute deformation gradient
      void def_grad(Core::LinAlg::SerialDenseVector& F,  ///< deformation gradient (to be filled)
          Core::LinAlg::SerialDenseVector& strain,       ///< strain tensor (to be filled)
          const Core::LinAlg::SerialDenseMatrix&
              xrefe,  ///< nodal positions of reference configuration
          const Core::LinAlg::SerialDenseMatrix&
              xcure,                                ///< nodal positions of current configuration
          Core::LinAlg::SerialDenseMatrix& boplin,  ///< B-operator
          const int iel);

      //! Compute geometric part of stiffness matrix
      void kg(Core::LinAlg::SerialDenseMatrix& estif,  ///< element stiffness matrix (to be filled)
          const Core::LinAlg::SerialDenseMatrix& boplin,  ///< B-operator
          const Core::LinAlg::SerialDenseMatrix& stress,  ///< 2. Piola-Kirchhoff stress tensor
          const double fac,                               ///< factor for Gaussian quadrature
          const int nd,                                   ///< number of DOFs in this element
          const int numeps);

      //! Compute elastic part of stiffness matrix
      void keu(Core::LinAlg::SerialDenseMatrix& estif,  ///< element stiffness matrix (to be filled)
          const Core::LinAlg::SerialDenseMatrix&
              b_cure,  ///< B-operator for current configuration (input)
          const Core::LinAlg::SerialDenseMatrix& C,  ///< material tensor (input)
          const double fac,                          ///< factor for Gaussian quadrature
          const int nd,                              ///< number of DOFs in this element
          const int numeps);

      //! Compute internal forces for solid approach
      void fint(
          const Core::LinAlg::SerialDenseMatrix& stress,  ///< 2. Piola-Kirchhoff stress (input)
          const Core::LinAlg::SerialDenseMatrix&
              b_cure,  ///< B-operator for current configuration (input)
          Core::LinAlg::SerialDenseVector& intforce,  ///< force vector (to be filled)
          const double fac,                           ///< factor for Gaussian quadrature
          const int nd                                ///< number of DOFs in this element
      );

      //! action parameters recognized by ale2
      enum ActionType
      {
        none,
        calc_ale_solid,         ///< compute stiffness based on fully nonlinear elastic solid with
                                ///< hyperelastic material law
        calc_ale_solid_linear,  ///< compute stiffness based on linear elastic solid with
                                ///< hyperelastic material law
        calc_ale_springs_material,  ///< compute stiffness based on springs algorithm in material
                                    ///< configuration
        calc_ale_springs_spatial,   ///< compute stiffness based on springs algorithm in spatial
                                    ///< configuration
        calc_ale_laplace_material,  ///< compute stiffness based on laplacian smoothing based on
                                    ///< material configuration
        calc_ale_laplace_spatial,   ///< compute stiffness based on laplacian smoothing based on
                                    ///< spatial configuration
        setup_material,             ///< Setup material in case of ElastHyper Tool Box
        calc_det_jac  ///< calculate Jacobian determinant and mesh quality measure according to
                      ///< [Oddy et al. 1988]
      };

      //! don't want = operator
      Ale2& operator=(const Ale2& old);
    };


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================


    class Ale2LineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Ale2LineType"; }

      static Ale2LineType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static Ale2LineType instance_;
    };


    /*!
    \brief An element representing a line of a ale2 element
    */
    class Ale2Line : public Core::Elements::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this line
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent ale element of this line
      \param lline: the local line number of this line w.r.t. the parent element
      */
      Ale2Line(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::Ale2* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale2Line(const Ale2Line& old);

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
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int unique_par_object_id() const override
      {
        return Ale2LineType::instance().unique_par_object_id();
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
      int num_dof_per_node(const Core::Nodes::Node& node) const override { return 2; }

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
        return Ale2LineType::instance();
      }

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a line Neumann condition on the ale2 element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      // don't want = operator
      Ale2Line& operator=(const Ale2Line& old);

    };  // class Ale2Line



  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
