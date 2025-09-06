// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_ALE3_HPP
#define FOUR_C_ALE_ALE3_HPP

/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

const int NUMDIM_ALE3 = 3;  ///< number of dimensions
const int NODDOF_ALE3 = 3;  ///< number of dofs per node

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
    // where should that be put?
    template <Core::FE::CellType dtype>
    struct DisTypeToNumCornerNodes
    {
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::tet4>
    {
      static constexpr int numCornerNodes = 4;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::tet10>
    {
      static constexpr int numCornerNodes = 4;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::hex8>
    {
      static constexpr int numCornerNodes = 8;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::hex20>
    {
      static constexpr int numCornerNodes = 8;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::hex27>
    {
      static constexpr int numCornerNodes = 8;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::pyramid5>
    {
      static constexpr int numCornerNodes = 5;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::wedge6>
    {
      static constexpr int numCornerNodes = 6;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::wedge15>
    {
      static constexpr int numCornerNodes = 6;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::nurbs8>
    {
      static constexpr int numCornerNodes = 8;
    };
    template <>
    struct DisTypeToNumCornerNodes<Core::FE::CellType::nurbs27>
    {
      static constexpr int numCornerNodes = 8;
    };


    /*----------------------------------------------------------------------------*/
    /*----------------------------------------------------------------------------*/
    class Ale3Surface;
    class Ale3ImplInterface;
    template <Core::FE::CellType distype>
    class Ale3Impl;
    class Ale3SurfaceImplInterface;
    template <Core::FE::CellType distype>
    class Ale3SurfaceImpl;


    class Ale3Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Ale3Type"; }

      static Ale3Type& instance();

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
      static Ale3Type instance_;
    };



    /*!
    \brief A C++ wrapper for the ale3 element
    */
    class Ale3 : public Core::Elements::Element
    {
     public:
      //! @name Friends
      friend class Ale3Surface;
      // friend class Ale3ImplInterface;
      // friend class Ale3Impl<

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      */
      Ale3(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale3(const Ale3& old);

      /*!
      \brief Deep copy this instance of Ale3 and return pointer to the copy

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
      int num_line() const override { return 0; }

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override
      {
        switch (num_node())
        {
          case 8:
          case 20:
          case 27:
            return 6;  // hex
          case 4:
          case 10:
            return 4;  // tet
          case 6:
          case 15:
          case 5:
            return 5;  // wedge or pyramid
          default:
            FOUR_C_THROW("Could not determine number of surfaces");
            return -1;
        }
      }

      /*!
      \brief Return number of volumes of this element (always 1)
      */
      inline int num_volume() const override { return 1; }

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
        return Ale3Type::instance().unique_par_object_id();
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

      Core::Elements::ElementType& element_type() const override { return Ale3Type::instance(); }

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

      Evaluate ale3 element stiffness, mass, internal forces etc

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

      //@}

      //! action parameters recognized by ale3
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
        calc_ale_node_normal,       ///< Calculate boundary node normal
        boundary_calc_ale_node_normal,  ///< Calculate boundary node normal
        setup_material,                 ///< Setup material in case of ElastHyper Tool Box
        calc_det_jac  ///< calculate Jacobian determinant and mesh quality measure according to
                      ///< [Oddy et al. 1988]
      };

     private:
      //! don't want = operator
      Ale3& operator=(const Ale3& old);
    };

    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================

    class Ale3ImplInterface
    {
     public:
      virtual ~Ale3ImplInterface() = default;

      /// Internal implementation class for fluid element
      static Ale3ImplInterface* impl(Discret::Elements::Ale3* ele);

      virtual void static_ke_spring(Ale3* ele,        ///< pointer to element
          Core::LinAlg::SerialDenseMatrix& sys_mat,   ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          const std::vector<double>& displacements,   ///< nodal displacements
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
          ) = 0;

      virtual void static_ke_nonlinear(Ale3* ele,    ///< pointer to element
          Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& lm,                      ///< node owner procs
          Core::LinAlg::SerialDenseMatrix&
              element_matrix,  ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector&
              element_residual,            ///< element residual vector (to be filled)
          std::vector<double>& my_dispnp,  ///< nodal displacements
          Teuchos::ParameterList& params,  ///< parameter list
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
          ) = 0;

      virtual void static_ke_laplace(Ale3* ele,  ///< pointer to element
          Core::FE::Discretization& dis,         ///< discretization
          Core::LinAlg::SerialDenseMatrix&
              element_matrix,                         ///< element stiffnes matrix (to be filled)
          Core::LinAlg::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          std::vector<double>& my_dispnp,             ///< nodal displacements
          std::shared_ptr<Core::Mat::Material> material,  ///< material law
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
          ) = 0;

      virtual void element_node_normal(
          Ale3* ele, Core::LinAlg::SerialDenseVector& elevec1, std::vector<double>& my_dispnp) = 0;
    };

    template <Core::FE::CellType distype>
    class Ale3Impl : public Ale3ImplInterface
    {
     public:
      /// Singleton access method
      static Ale3Impl<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      void static_ke_laplace(Ale3* ele,   ///< pointer to element
          Core::FE::Discretization& dis,  ///< discretization
          Core::LinAlg::SerialDenseMatrix&
              element_matrix,                         ///< element stiffnes matrix (to be filled)
          Core::LinAlg::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          std::vector<double>& my_dispnp,             ///< nodal displacements
          std::shared_ptr<Core::Mat::Material> material,  ///< material law
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
          ) override;

      void static_ke_spring(Ale3* ele,  ///< pointer to element
          Core::LinAlg::SerialDenseMatrix&
              element_matrix,  ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector&
              element_residual,                      ///< element residual vector (to be filled)
          const std::vector<double>& displacements,  ///< nodal displacements
          const bool spatialconfiguration            ///< use spatial configuration (true), material
                                                     ///< configuration (false)
          ) override;

      void static_ke_nonlinear(Ale3* ele,            ///< pointer to element
          Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& lm,                      ///< node owner procs
          Core::LinAlg::SerialDenseMatrix&
              element_matrix,  ///< element stiffness matrix (to be filled)
          Core::LinAlg::SerialDenseVector&
              element_residual,            ///< element residual vector (to be filled)
          std::vector<double>& my_dispnp,  ///< nodal displacements
          Teuchos::ParameterList& params,  ///< parameter list
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
          ) override;

      void element_node_normal(Ale3* ele,            ///< pointer to element
          Core::LinAlg::SerialDenseVector& elevec1,  ///< normal vector (to be filled)
          std::vector<double>& my_dispnp             ///< nodal displacements
          ) override;

      //! Calculate Jacobian matrix and its determinant
      void calc_jacobian(Ale3* ele,  ///< pointer to element
          double& detJ               ///< determinant of Jacobian matrix
      );

     private:
      Ale3Impl() = default;
      static constexpr int iel = Core::FE::num_nodes(distype);
      static constexpr int numcnd = DisTypeToNumCornerNodes<distype>::numCornerNodes;

      inline void ale3_edge_geometry(int i, int j, const Core::LinAlg::Matrix<3, iel>& xyze,
          double& length, double& dx, double& dy, double& dz);

      /*!
      \brief Prevents node s from penetrating face pqr.

      The calculated dynamic triangle sjq is perpendicular to edge pr and face
      pqr. According to Farhat et al.

      The interface of this function is rather weird, I changed it to reduce
      re-calculation of vectors that are used multiple times.

      \param node_i (in)    : Determine position of triangle sjq in
      tetrahedra.
      \param sq             : edge vector
      \param len_sq         : length
      \param rp             : edge vector
      \param len_rp         : length
      \param qp             : edge vector
      \param local_x        : vector needed in calculation
      \param sysmat (in/out): The element's sys_mat
      */
      void ale3_add_tria_stiffness(int node_p, int node_q, int node_r, int node_s,
          const Core::LinAlg::Matrix<3, 1>& sq, const double len_sq,
          const Core::LinAlg::Matrix<3, 1>& rp, const double len_rp,
          const Core::LinAlg::Matrix<3, 1>& qp, const Core::LinAlg::Matrix<3, 1>& local_x,
          Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat);

      /*!
      \brief Prevents node-face-penetration for given nodes.

      Twelve-triangle configuration. According to Farhat et al.

      \param tet_i (in)    : Nodes. Arbitrary succession.
      \param sysmat (in/out): The element's sys_mat
      \param xyze (in)     : The actual element coordinates
      */
      void ale3_add_tetra_stiffness(int tet_0, int tet_1, int tet_2, int tet_3,
          Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);
      inline void ale3_tors_spring_tet4(Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);
      inline void ale3_tors_spring_pyramid5(Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);
      inline void ale3_tors_spring_wedge6(Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);
      inline void ale3_tors_spring_hex8(Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);
      inline void ale3_tors_spring_nurbs27(Core::LinAlg::Matrix<3 * iel, 3 * iel>& sys_mat,
          const Core::LinAlg::Matrix<3, iel>& xyze);


      inline Core::FE::GaussRule3D get_optimal_gaussrule();

      Ale3Impl<distype> operator=(const Ale3Impl<distype> other);
    };


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================


    class Ale3SurfaceType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Ale3SurfaceType"; }

      static Ale3SurfaceType& instance();

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
      static Ale3SurfaceType instance_;
    };


    /*!
    \brief An element representing a surface of a ale3 element

    \note This is a pure Neumann boundary condition element. It's only
          purpose is to evaluate surface Neumann boundary conditions that might be
          adjacent to a parent ale3 element. It therefore does not implement
          the Core::Elements::Element::Evaluate method and does not have its own ElementRegister
    class.

    */
    class Ale3Surface : public Core::Elements::FaceElement
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
      \param parent: The parent ale element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      Ale3Surface(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::Ale3* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale3Surface(const Ale3Surface& old);

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
        return Ale3SurfaceType::instance().unique_par_object_id();
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
        return Ale3SurfaceType::instance();
      }

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate the ale3 surface element

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

      this method evaluates a surface Neumann condition on the ale3 element

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
      Ale3Surface& operator=(const Ale3Surface& old);

      //  compute kovariant metric tensor G for ale surface element
      //                                                  gammi 04/07
      void f3_metric_tensor_for_surface(const Core::LinAlg::SerialDenseMatrix xyze,
          const Core::LinAlg::SerialDenseMatrix deriv,
          Core::LinAlg::SerialDenseMatrix& metrictensor, double* drs);

    };  // class Ale3Surface

    class Ale3SurfaceImplInterface
    {
     public:
      /// Empty constructor
      Ale3SurfaceImplInterface() {}

      virtual ~Ale3SurfaceImplInterface() = default;
      /// Internal implementation class for ale surface element
      static Ale3SurfaceImplInterface* impl(Discret::Elements::Ale3Surface* ele);

      virtual void element_node_normal(Ale3Surface* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1, std::vector<double>& mydispnp) = 0;
    };

    template <Core::FE::CellType distype>
    class Ale3SurfaceImpl : public Ale3SurfaceImplInterface
    {
      Ale3SurfaceImpl() {}

     public:
      /// Singleton access method
      static Ale3SurfaceImpl<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      //! number of element nodes
      static constexpr int bdrynen_ = Core::FE::num_nodes(distype);

      //! number of spatial dimensions of boundary element
      static constexpr int bdrynsd_ = Core::FE::dim<distype>;

      //! number of spatial dimensions of parent element
      static constexpr int nsd_ = bdrynsd_ + 1;

      //! number of degrees of freedom per node
      static constexpr int numdofpernode_ = nsd_;

      void element_node_normal(Ale3Surface* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1, std::vector<double>& mydispnp) override;

     private:
      //  static constexpr int iel =
      //  Core::FE::num_nodes(distype); static const int
      //  numcnd = DisTypeToNumCornerNodes<distype>::numCornerNodes;

      //  inline void ale3_edge_geometry(int i, int j, const Core::LinAlg::Matrix<3, iel>& xyze,
      //                                 double& length,
      //                                 double& dx,
      //                                 double& dy,
      //                                 double& dz);

      //  /*!
      //  \brief Prevents node s from penetrating face pqr.
      //
      //  The calculated dynamic triangle sjq is perpendicular to edge pr and face
      //  pqr. According to Farhat et al.
      //
      //  The interface of this function is rather weird, I changed it to reduce
      //  re-calculation of vectors that are used multiple times.
      //
      //  \param node_i (in)    : Determine position of triangle sjq in
      //  tetrahedra.
      //  \param sq             : edge vector
      //  \param len_sq         : length
      //  \param rp             : edge vector
      //  \param len_rp         : length
      //  \param qp             : edge vector
      //  \param local_x        : vector needed in calculation
      //  \param sysmat (in/out): The element's sys_mat
      //  */
      //  void ale3_add_tria_stiffness(
      //      int node_p, int node_q, int node_r, int node_s,
      //      const Core::LinAlg::Matrix<3, 1>& sq,
      //      const double len_sq,
      //      const Core::LinAlg::Matrix<3, 1>& rp,
      //      const double len_rp,
      //      const Core::LinAlg::Matrix<3, 1>& qp,
      //      const Core::LinAlg::Matrix<3, 1>& local_x,
      //      Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat);
      //
      //  /*!
      //  \brief Prevents node-face-penetration for given nodes.
      //
      //  Twelve-triangle configuration. According to Farhat et al.
      //
      //  \param tet_i (in)    : Nodes. Arbitrary succession.
      //  \param sysmat (in/out): The element's sys_mat
      //  \param xyze (in)     : The actual element coordinates
      //  */
      //  void ale3_add_tetra_stiffness(int tet_0, int tet_1, int tet_2, int tet_3,
      //                                Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //                                const Core::LinAlg::Matrix<3,iel>& xyze);
      //  inline void ale3_tors_spring_tet4(Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //                                    const Core::LinAlg::Matrix<3,iel>& xyze);
      //  inline void ale3_tors_spring_pyramid5(Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //                                        const Core::LinAlg::Matrix<3,iel>& xyze);
      //  inline void ale3_tors_spring_wedge6(Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //                                      const Core::LinAlg::Matrix<3,iel>& xyze);
      //  inline void ale3_tors_spring_hex8(Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //                                    const Core::LinAlg::Matrix<3,iel>& xyze);
      //  inline void ale3_tors_spring_nurbs27(Core::LinAlg::Matrix<3*iel,3*iel>& sys_mat,
      //               const Core::LinAlg::Matrix<3,iel>& xyze);
      //
      //
      //  inline Core::FE::GaussRule3D get_optimal_gaussrule();

      // Ale3SurfaceImpl<distype> operator=(const Ale3SurfaceImpl<distype> other);


     protected:
      //! array for shape functions for boundary element
      Core::LinAlg::Matrix<bdrynen_, 1> funct_;
      //! array for shape function derivatives for boundary element
      Core::LinAlg::Matrix<bdrynsd_, bdrynen_> deriv_;
      //! integration factor
      double fac_;
      //! normal vector pointing out of the domain
      Core::LinAlg::Matrix<nsd_, 1> unitnormal_;
      //! infinitesimal area element drs
      double drs_;
      //! coordinates of current integration point in reference coordinates
      Core::LinAlg::Matrix<bdrynsd_, 1> xsi_;
      //! node coordinates for boundary element
      Core::LinAlg::Matrix<nsd_, bdrynen_> xyze_;
    };



  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
