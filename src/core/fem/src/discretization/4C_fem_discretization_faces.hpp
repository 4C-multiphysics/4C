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

      \param target_peid (in): element id of target parent element
      \param source_peid (in): element id of source parent element
      \param lsurface_target (in): local index of surface w.r.t target parent element
      \param nodes (in): vector of nodes building the surface element
      */
      InternalFacesData(int target_peid, std::vector<Core::Nodes::Node*> nodes, int lsurface_target)
      {
        target_peid_ = target_peid;
        source_peid_ = -1;
        lsurface_target_ = lsurface_target;
        lsurface_source_ = -1;
        nodes_ = nodes;
      }

      /*--- set ------------------------------------------*/

      //! set the parent element id for source parent element
      void set_source_peid(int eid) { source_peid_ = eid; }

      //! set the local surface number w.r.t source parent element
      void set_l_surface_source(int lsurface_source) { lsurface_source_ = lsurface_source; }

      /*!
      \brief set the map for the face's nodes between the local coordinate systems of the face w.r.t
      the target parent element's face's coordinate system and the source element's face's
      coordinate system
      */
      void set_local_numbering_map(std::vector<int> localtrafomap)
      {
        localtrafomap_ = localtrafomap;
      }


      /*--- get ------------------------------------------*/

      //! get the target parent element id
      int get_target_peid() const { return target_peid_; }

      //! get the source parent element id
      int get_source_peid() const { return source_peid_; }

      //! get the local surface number w.r.t target parent element
      int get_l_surface_target() const { return lsurface_target_; }

      //! get the local surface number w.r.t source parent element
      int get_l_surface_source() const { return lsurface_source_; }

      //! get the transformation map between the local coordinate systems of the face w.r.t the
      //! target parent element's face's coordinate system and the source element's face's
      //! coordinate system
      const std::vector<int>& get_local_numbering_map() const { return localtrafomap_; }

      //! get surface's nodes (unsorted, original)
      const std::vector<Core::Nodes::Node*>& get_nodes() const { return nodes_; }

     private:
      int target_peid_;  //!< target parent element id
      int source_peid_;  //!< source parent element id

      int lsurface_target_;  //!< local surface number w.r.t target parent element
      int lsurface_source_;  //!< local surface number w.r.t source parent element

      std::vector<Core::Nodes::Node*>
          nodes_;  //!< vector of surface nodes, order w.r.t target parent element

      /*!
       \brief map for the face's nodes between the local coordinate systems of the face w.r.t the
       target parent element's face's coordinate system and the source element's face's coordinate
       system
       */
      std::vector<int> localtrafomap_;
    };



    /*!
    \brief Standard Constructor

    \param name: name of this discretization
    \param comm: MPI comm object associated with this discretization
    \param n_dim: number of space dimensions of this discretization
    */
    DiscretizationFaces(const std::string name, MPI_Comm comm, unsigned int n_dim);


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

    \param options (in) : options struct to set flags for specific tasks
    \param createinternalfaces (in) : if true, build geometry of internal faces.

    \note In order to receive a fully functional discretization, this method must be called
          with all parameters set to true (the default). The parameters though can be
          used to turn off specific tasks to allow for more flexibility in the
          construction of a discretization, where it is known that this method will
          be called more than once.

    \note Sets Filled()=true
    */
    int fill_complete_faces(OptionsFillComplete options, bool createinternalfaces = false);

    /*!
    \brief Get flag indicating whether create_internal_faces_extension() has been called
    */
    inline bool filled_extension() const { return extension_filled_; }

    /*!
    \brief Get map associated with the distribution of the ownership of faces
           (Filled()==true prerequisite)

    This map includes all faces stored on this proc and also owned by this proc.
    This map is non-ambiguous, meaning that it is a non-overlapping map.

    \return nullptr if Filled() is false. A call to fill_complete() is a prerequisite.
    */
    const Core::LinAlg::Map* face_row_map() const;

    /*!
    \brief Get map associated with the distribution of elements including ghosted faces
           (Filled()==true prerequisite)

    This map includes all internal faces stored on this proc including any ghosted faces
    This map is ambiguous, meaning that it is an overlapping map

    \return nullptr if Filled() is false. A call to fill_complete() is a prerequisite.
    */
    const Core::LinAlg::Map* face_col_map() const;

    /*!
    \brief Get global number of internal faces (true number of total elements)
           (Filled()==true prerequisite)

    This is a collective call
    */
    int num_global_faces() const;

    /*!
    \brief Get processor local number of internal faces owned by this processor
           (Filled()==true prerequisite)
    */
    int num_my_row_faces() const;

    /*!
    \brief Get processor local number of internal faces including ghost elements
           (Filled()==true NOT prerequisite)
    */
    int num_my_col_faces() const;

    /*!
    \brief Get the internal face element with local row id lid (Filled()==true prerequisite)

    Returns the internal face element with local row index lid.
    Will not return any ghosted element.
    This is an individual call and Filled()=true is a prerequisite

    \return Address of internal face element if element is owned by calling proc
    */
    inline Core::Elements::Element* l_row_face(int lid) const
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
    inline Core::Elements::Element* l_col_face(int lid) const
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
    discretization and store it as an Core::LinAlg::Map in intfacerowmap_
    intfacerowmap_ is unique.
    It considers internal faces owned by a proc only

    \note This is a collective call

    */
    void build_face_row_map();

    /*!
    \brief Build intfacecolmap_ (Filled()==true NOT prerequisite)

    Build the potentially overlapping parallel layout of internal faces in this
    discretization and store it as an Core::LinAlg::Map in intfacecolmap_
    intfacecolmap_ includes ghosted internal faces and is potentially overlapping.

    \note This is a collective call

    */
    void build_face_col_map();

    /*!
    \brief Print Print internal faces discretization to os (Filled()==true NOT prerequisite)
           (ostream << also supported)

    \note This is a collective call
    */
    void print_faces(std::ostream& os) const;


   protected:
    bool extension_filled_;  ///< flag indicating whether faces extension has been filled
    bool doboundaryfaces_;   ///< flag set to true by derived HDG class for boundary face elements

    std::shared_ptr<Core::LinAlg::Map> facerowmap_;  ///< unique distribution of element ownerships
    std::shared_ptr<Core::LinAlg::Map>
        facecolmap_;  ///< distribution of elements including ghost elements
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
