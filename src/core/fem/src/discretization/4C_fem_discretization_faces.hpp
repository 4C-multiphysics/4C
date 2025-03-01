// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DISCRETIZATION_FACES_HPP
#define FOUR_C_FEM_DISCRETIZATION_FACES_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class MapExtractor;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::Elements
{
  class FaceElement;
}

namespace Core::FE
{
  class DiscretizationFaces : public Core::FE::Discretization
  {
   public:
    /*!
     * \brief internal class that holds the information used to create face elements
     *
     */
    class InternalFacesData
    {
     public:
      /*!
      \brief Standard Constructor

      \param master_peid (in): element id of master parent element
      \param slave_peid (in): element id of slave parent element
      \param lsurface_master (in): local index of surface w.r.t master parent element
      \param nodes (in): vector of nodes building the surface element
      */
      InternalFacesData(int master_peid, std::vector<Core::Nodes::Node*> nodes, int lsurface_master)
      {
        master_peid_ = master_peid;
        slave_peid_ = -1;
        lsurface_master_ = lsurface_master;
        lsurface_slave_ = -1;
        nodes_ = nodes;
      }

      /*--- set ------------------------------------------*/

      //! set the parent element id for slave parent element
      void set_slave_peid(int eid) { slave_peid_ = eid; }

      //! set the local surface number w.r.t slave parent element
      void set_l_surface_slave(int lsurface_slave) { lsurface_slave_ = lsurface_slave; }

      /*!
      \brief set the map for the face's nodes between the local coordinate systems of the face w.r.t
      the master parent element's face's coordinate system and the slave element's face's coordinate
      system
      */
      void set_local_numbering_map(std::vector<int> localtrafomap)
      {
        localtrafomap_ = localtrafomap;
      }


      /*--- get ------------------------------------------*/

      //! get the master parent element id
      int get_master_peid() const { return master_peid_; }

      //! get the slave parent element id
      int get_slave_peid() const { return slave_peid_; }

      //! get the local surface number w.r.t master parent element
      int get_l_surface_master() const { return lsurface_master_; }

      //! get the local surface number w.r.t slave parent element
      int get_l_surface_slave() const { return lsurface_slave_; }

      //! get the transformation map between the local coordinate systems of the face w.r.t the
      //! master parent element's face's coordinate system and the slave element's face's coordinate
      //! system
      const std::vector<int>& get_local_numbering_map() const { return localtrafomap_; }

      //! get surface's nodes (unsorted, original)
      const std::vector<Core::Nodes::Node*>& get_nodes() const { return nodes_; }

     private:
      int master_peid_;  //!< master parent element id
      int slave_peid_;   //!< slave parent element id

      int lsurface_master_;  //!< local surface number w.r.t master parent element
      int lsurface_slave_;   //!< local surface number w.r.t slave parent element

      std::vector<Core::Nodes::Node*>
          nodes_;  //!< vector of surface nodes, order w.r.t master parent element

      /*!
       \brief map for the face's nodes between the local coordinate systems of the face w.r.t the
       master parent element's face's coordinate system and the slave element's face's coordinate
       system
       */
      std::vector<int> localtrafomap_;
    };



    /*!
    \brief Standard Constructor

    \param name: name of this discretization
    \param comm: Epetra comm object associated with this discretization
    \param n_dim: number of space dimensions of this discretization
    */
    DiscretizationFaces(const std::string name, MPI_Comm comm, unsigned int n_dim);



    /*!
    \brief Compute the nullspace of the discretization

    This method looks in the solver parameters whether algebraic multigrid (AMG)
    is used as preconditioner. AMG desires the nullspace of the
    system of equations which is then computed here if it does not already exist
    in the parameter list.

    \note This method is supposed to go away and live somewhere else soon....

    \param solveparams (in): List of parameters
    \param recompute (in)  : force method to recompute the nullspace
    */
    void compute_null_space_if_necessary(
        Teuchos::ParameterList& solveparams, bool recompute = false) override
    {
      // remark: the null space is not computed correctly for XFEM discretizations, since the number
      // of
      //         degrees of freedom per node is not fixed.
      //         - it is not clear what happens with respect to the Krylov projection
      //           (having XFEM dofs seems to render the system non-singular, but it should be
      //           singular so the null space has a non-zero dimension)
      //         - the ML preconditioner also relies on a fixed number of dofs per node
      Core::FE::Discretization::compute_null_space_if_necessary(solveparams, recompute);
    }

    /*!
    \brief Complete construction of a discretization  (Filled()==true NOT prerequisite)

    After adding or deleting nodes or elements or redistributing them in parallel,
    or adding/deleting boundary conditions, this method has to be called to (re)construct
    pointer topologies.<br>
    It builds in this order:<br>
    Standard fill_complete of base class
    - row map of nodes
    - column map of nodes
    - row map of elements
    - column map of elements
    - pointers from elements to nodes
    - pointers from nodes to elements
    - assigns degrees of freedoms
    - map of element register classes
    - calls all element register initialize methods
    - build geometries of all Dirichlet and Neumann boundary conditions

    Additional features
    - build internal faces elements
    - build maps and pointers for internal faces

    \param assigndegreesoffreedom (in) : if true, resets existing dofsets and performs
                                         assigning of degrees of freedoms to nodes and
                                         elements.
    \param initelements (in) : if true, build element register classes and call initialize()
                               on each type of finite element present
    \param doboundaryconditions (in) : if true, build geometry of boundary conditions
                                       present.
    \param createinternalfaces (in) : if true, build geometry of internal faces.
    \param createboundaryfaces (in) : if true,

    \note In order to receive a fully functional discretization, this method must be called
          with all parameters set to true (the default). The parameters though can be
          used to turn off specific tasks to allow for more flexibility in the
          construction of a discretization, where it is known that this method will
          be called more than once.

    \note Sets Filled()=true
    */
    int fill_complete_faces(bool assigndegreesoffreedom = true, bool initelements = true,
        bool doboundaryconditions = true, bool createinternalfaces = false);

    /*!
    \brief Get flag indicating whether create_internal_faces_extension() has been called
    */
    virtual inline bool filled_extension() const { return extension_filled_; }

    /*!
    \brief Get map associated with the distribution of the ownership of faces
           (Filled()==true prerequisite)

    This map includes all faces stored on this proc and also owned by this proc.
    This map is non-ambiguous, meaning that it is a non-overlapping map.

    \return nullptr if Filled() is false. A call to fill_complete() is a prerequisite.
    */
    virtual const Epetra_Map* face_row_map() const;

    /*!
    \brief Get map associated with the distribution of elements including ghosted faces
           (Filled()==true prerequisite)

    This map includes all internal faces stored on this proc including any ghosted faces
    This map is ambiguous, meaning that it is an overlapping map

    \return nullptr if Filled() is false. A call to fill_complete() is a prerequisite.
    */
    virtual const Epetra_Map* face_col_map() const;

    /*!
    \brief Get global number of internal faces (true number of total elements)
           (Filled()==true prerequisite)

    This is a collective call
    */
    virtual int num_global_faces() const;

    /*!
    \brief Get processor local number of internal faces owned by this processor
           (Filled()==true prerequisite)
    */
    virtual int num_my_row_faces() const;

    /*!
    \brief Get processor local number of internal faces including ghost elements
           (Filled()==true NOT prerequisite)
    */
    virtual int num_my_col_faces() const;

    /*!
    \brief Get the internal face element with local row id lid (Filled()==true prerequisite)

    Returns the internal face element with local row index lid.
    Will not return any ghosted element.
    This is an individual call and Filled()=true is a prerequisite

    \return Address of internal face element if element is owned by calling proc
    */
    virtual inline Core::Elements::Element* l_row_face(int lid) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (!filled()) FOUR_C_THROW("Core::FE::DiscretizationFaces::lRowIntFace: Filled() != true");
#endif
      return facerowptr_[lid];
    }

    /*!
    \brief Get the element with local column id lid (Filled()==true prerequisite)

    Returns the internal face element with local column index lid.
    Will also return any ghosted element.
    This is an individual call and Filled()=true is a prerequisite

    \return Address of internal face element if element is stored by calling proc
    */
    virtual inline Core::Elements::Element* l_col_face(int lid) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (!filled()) FOUR_C_THROW("Core::FE::DiscretizationFaces::lColIntFace: Filled() != true");
#endif
      return facecolptr_[lid];
    }

    /*!
    \brief Build internal faces extension
    */
    void create_internal_faces_extension(const bool verbose = false);

    /*!
    \brief Complete construction of a face elements
    */
    void build_faces(const bool verbose = false);

    /*!
    \brief Build intfacerowmap_ (Filled()==true NOT prerequisite)

    Build the parallel layout of internal faces in this
    discretization and store it as an Epetra_Map in intfacerowmap_
    intfacerowmap_ is unique.
    It considers internal faces owned by a proc only

    \note This is a collective call

    */
    virtual void build_face_row_map();

    /*!
    \brief Build intfacecolmap_ (Filled()==true NOT prerequisite)

    Build the potentially overlapping parallel layout of internal faces in this
    discretization and store it as an Epetra_Map in intfacecolmap_
    intfacecolmap_ includes ghosted internal faces and is potentially overlapping.

    \note This is a collective call

    */
    virtual void build_face_col_map();

    /*!
    \brief Print Print internal faces discretization to os (Filled()==true NOT prerequisite)
           (ostream << also supported)

    \note This is a collective call
    */
    void print_faces(std::ostream& os) const;


   protected:
    bool extension_filled_;  ///< flag indicating whether faces extension has been filled
    bool doboundaryfaces_;   ///< flag set to true by derived HDG class for boundary face elements

    std::shared_ptr<Epetra_Map> facerowmap_;  ///< unique distribution of element ownerships
    std::shared_ptr<Epetra_Map> facecolmap_;  ///< distribution of elements including ghost elements
    std::vector<Core::Elements::Element*>
        facerowptr_;  ///< vector of pointers to row elements for faster access
    std::vector<Core::Elements::Element*>
        facecolptr_;  ///< vector of pointers to column elements for faster access
    std::map<int, std::shared_ptr<Core::Elements::FaceElement>>
        faces_;  ///< map of internal faces elements


  };  // class DiscretizationXFEM
}  // namespace Core::FE

/// << operator
std::ostream& operator<<(std::ostream& os, const Core::FE::DiscretizationFaces& dis);


FOUR_C_NAMESPACE_CLOSE

#endif
