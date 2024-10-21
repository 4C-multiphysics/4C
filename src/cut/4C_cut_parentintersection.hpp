#ifndef FOUR_C_CUT_PARENTINTERSECTION_HPP
#define FOUR_C_CUT_PARENTINTERSECTION_HPP

#include "4C_config.hpp"

#include "4C_cut_meshhandle.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_pointpool.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Element;
  class ElementHandle;

  /*!
  \brief Interface class for the general mesh cut. This class contains shared functionality
  between level set intersection and mesh intersection routines.
  */
  class ParentIntersection
  {
   public:
    /*!
    \brief This class holds data for all volumecells for that dofsets (dofset numbers) of non-row
    nodes have to be communicated between processors.
     */
    class DofSetData
    {
     public:
      //! constructor for creating volumecells dofset data for communication between processors
      //! (constructor during the Robin round)
      DofSetData(int set_index,  ///< set index for Volumecell
          bool inside_cell,      ///< cell inside or outside
          std::vector<Core::LinAlg::Matrix<3, 1>>&
              cut_points_coords,  ///< coordinates of cut_points
          int peid,               ///< parent element id
          std::map<int, int>&
              node_dofsetnumber_map  ///< for the current volumecell in a parent element, for each
                                     ///< node (nid) the current dofset number
          )
          : set_index_(set_index), inside_cell_(inside_cell), peid_(peid)
      {
        std::copy(node_dofsetnumber_map.begin(), node_dofsetnumber_map.end(),
            std::inserter(node_dofsetnumber_map_, node_dofsetnumber_map_.begin()));
        std::copy(cut_points_coords.begin(), cut_points_coords.end(),
            std::inserter(cut_points_coords_, cut_points_coords_.begin()));
      }

      /*!
      \brief print the dofset data to screen
       */
      void print()
      {
        // print volumecell information
        std::cout << "Volumecell-DofSetData: " << std::endl;

        // print parent element Id
        std::cout << "\tparent element id: " << peid_ << std::endl;

        // print node_dofsetnumber_map
        std::cout << "\tnode_dofsetnumber_map" << std::endl;
        for (std::map<int, int>::iterator i = node_dofsetnumber_map_.begin();
             i != node_dofsetnumber_map_.end(); ++i)
        {
          std::cout << "\t\tnodeId \t" << i->first << "\t dofsetnumber \t" << i->second
                    << std::endl;
        }
      }

      int set_index_;     ///< set index for Volumecell
      bool inside_cell_;  ///< bool inside or outside cell
      std::vector<Core::LinAlg::Matrix<3, 1>>
          cut_points_coords_;                     ///< coordinates for points of Volumecell
      int peid_;                                  ///< parent element Id for volumecell
      std::map<int, int> node_dofsetnumber_map_;  ///< node Ids and dofset numbers w.r.t volumecell
                                                  ///< for that data have to be communicated


     private:
    };  // end class DofSetData


    /// constructur for ParentIntersecton class
    ParentIntersection(int myrank = -1)
        : pp_(Teuchos::make_rcp<PointPool>()),
          mesh_(options_, 1, pp_, false, myrank),
          myrank_(myrank)
    {
    }

    /// destructor
    virtual ~ParentIntersection() = default;

    /*========================================================================*/
    //! @name set and get routines for options
    /*========================================================================*/

    /// set the option if positions have to be determined or not
    void set_find_positions(bool positions) { options_.set_find_positions(positions); }

    /// set the option if positions have to be determined or not
    void set_nodal_dof_set_strategy(Cut::NodalDofSetStrategy nodal_dofset_strategy)
    {
      options_.set_nodal_dof_set_strategy(nodal_dofset_strategy);
    }

    /// Set the position for the boundary cell creation
    void set_gen_boundary_cell_position(BoundaryCellPosition gen_bcell_position)
    {
      options_.set_gen_boundary_cell_position(gen_bcell_position);
    }

    /// get the options
    void get_options(Options& options) { options = options_; };

    /// get the options
    Options& get_options() { return options_; };

    /*========================================================================*/
    //! @name Cut functionality routines
    /*========================================================================*/


    virtual void cut_self_cut(bool include_inner, bool screenoutput)
    {
      if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t * 2/6 Cut_SelfCut ... not performed";
      return;
    };

    virtual void cut_collision_detection(bool include_inner, bool screenoutput)
    {
      if (myrank_ == 0 and screenoutput)
        Core::IO::cout << "\t * 3/6 cut_collision_detection ... not performed";
      return;
    };

    virtual void cut_mesh(bool screenoutput)
    {
      if (myrank_ == 0 and screenoutput)
        Core::IO::cout << "\t * 4/6 Cut_Mesh (LevelSet-Cut) ... not performed";
      return;
    };

    virtual void cut_mesh_intersection(bool screenoutput)
    {
      if (myrank_ == 0 and screenoutput)
        Core::IO::cout << "\t * 4/6 cut_mesh_intersection (Mesh-Cut) ... not performed";
      return;
    };

    /*!
    \brief The routine which splits the volumecell into integrationcells by tessellation, or
    create Gaussian integration rules by moment fitting equations
     */
    void cut_finalize(bool include_inner, VCellGaussPts VCellgausstype,
        Cut::BCellGaussPts BCellgausstype, bool tetcellsonly, bool screenoutput);

    /*========================================================================*/
    //! @name nodal dofset routines
    /*========================================================================*/

    /// Create nodal dofset sets within the parallel cut framework
    void create_nodal_dof_set(bool include_inner, const Core::FE::Discretization& dis);

    /// fill parallel DofSetData with information that has to be communicated
    void fill_parallel_dof_set_data(std::vector<Teuchos::RCP<DofSetData>>& parallel_dofSetData,
        const Core::FE::Discretization& dis, bool include_inner);

    /// create parallel DofSetData for a volumecell that has to be communicated
    void create_parallel_dof_set_data_vc(std::vector<Teuchos::RCP<DofSetData>>& parallel_dofSetData,
        int eid, int set_index, bool inside, VolumeCell* cell, std::map<int, int>& node_dofset_map);

    /// find cell sets around each node (especially for quadratic elements)
    void find_nodal_cell_sets(bool include_inner, std::set<int>& eids,
        std::vector<int>& sourrounding_elements,
        std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets_inside,
        std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets_outside,
        std::vector<plain_volumecell_set>& cell_sets_inside,
        std::vector<plain_volumecell_set>& cell_sets_outside,
        std::vector<plain_volumecell_set>& cell_sets);

    /// connect sets of volumecells for neighboring elements around a node
    void connect_nodal_dof_sets(std::vector<Node*>& nodes, bool include_inner,
        const Core::FE::Discretization& dis,
        const std::vector<plain_volumecell_set>& connected_vc_sets,
        std::vector<std::vector<int>>& nodaldofset_vc_sets,
        std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm);


    /*========================================================================*/
    //! @name get routines for nodes, elements, sides, mesh, meshhandles
    /*========================================================================*/

    /// get the node based on node id
    Node* get_node(int nid) const;

    /// get the mesh's side based on node ids and return the side
    SideHandle* get_side(std::vector<int>& nodeids) const;

    /// get the mesh's side based on side id
    SideHandle* get_side(int sid) const;

    /// get the mesh's element based on element id
    ElementHandle* get_element(int eid) const;

    /// get the linear mesh
    Mesh& normal_mesh() { return mesh_.linear_mesh(); }

    /// get the mesh handle
    MeshHandle& get_mesh_handle() { return mesh_; }

    /*========================================================================*/
    //! @name GMSH output
    /*========================================================================*/

    /// write gmsh debug output for nodal cell sets
    void dump_gmsh_nodal_cell_set(
        std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets,
        const Core::FE::Discretization& dis);

    /// write gmsh debug output for CellSets
    void dump_gmsh_cell_sets(
        std::vector<plain_volumecell_set>& cell_sets, const Core::FE::Discretization& dis);

    /// write gmsh cut output for number of dofsets and the connected vc sets
    void dump_gmsh_num_dof_sets(
        std::string filename, bool include_inner, const Core::FE::Discretization& dis);

    /// write gmsh output for volumecells
    void dump_gmsh_volume_cells(std::string name, bool include_inner);

    /// write gmsh output for volumecells
    void dump_gmsh_integration_cells(std::string name);

    /// write gmsh output for volumecells
    void dump_gmsh_volume_cells(std::string name);


    /*========================================================================*/
    //! @name statistics, output
    /*========================================================================*/

    /// print cell statistics
    void print_cell_stats();

   protected:
    /*========================================================================*/
    //! @name protected class variables
    /*========================================================================*/

    Teuchos::RCP<PointPool> pp_;  ///< pointpool (octTree) whose nodes consist of bounding boxes,
                                  ///< each bb contains a set of Teuchos::RCPs to points
    Options options_;             ///< options
    MeshHandle mesh_;             ///< the background mesh
    int myrank_;                  ///< my processor Id
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
