// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PRE_EXODUS_READER_HPP
#define FOUR_C_PRE_EXODUS_READER_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_utils_exceptions.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace EXODUS
{
  // forward declaration
  class ElementBlock;
  class NodeSet;
  class SideSet;

  /*!
  \brief Mesh will in future store all information necessary to build a mesh

  */
  class Mesh
  {
   public:
    //! constructor
    Mesh(std::string exofilename);

    //! extension constructor, adds Elementblock and nodes to a basemesh
    Mesh(const Mesh& basemesh, const std::shared_ptr<std::map<int, std::vector<double>>> extNodes,
        const std::map<int, std::shared_ptr<ElementBlock>>& extBlocks,
        const std::map<int, NodeSet>& extNodesets, const std::map<int, SideSet>& extSidesets,
        const std::string newtitle);

    //! empty constructor
    Mesh();

    //! destructor
    virtual ~Mesh() = default;

    //! Print mesh info
    void print(std::ostream& os, bool verbose = false) const;

    //! Print Nodes and Coords
    void print_nodes(std::ostream& os, bool storeid = false) const;

    //! Get number of nodes in mesh
    int get_num_nodes() const { return nodes_->size(); }

    //! Get number of elements in mesh
    int get_num_ele() const { return num_elem_; }

    //! Get number of dimensions
    int get_num_dim() const { return num_dim_; }

    //! Get number of dimensions
    int get_four_c_dim() const { return four_c_dim_; }

    //! Get exodus file id
    int get_exo_id() const { return exoid_; }

    //! Get mesh title
    std::string get_title() const;

    //! Get ElementBlock map
    std::map<int, std::shared_ptr<ElementBlock>> get_element_blocks() const
    {
      return element_blocks_;
    }

    //! Get Number of ElementBlocks
    int get_num_element_blocks() const { return element_blocks_.size(); }

    //! Get one ElementBlock
    std::shared_ptr<ElementBlock> get_element_block(const int id) const;

    //! Get NodeSet map
    std::map<int, NodeSet> get_node_sets() const { return node_sets_; }

    //! Get Number of NodeSets
    int get_num_node_sets() const { return node_sets_.size(); }

    //! Get one NodeSet
    NodeSet get_node_set(const int id) const;

    //! Get SideSet map
    std::map<int, SideSet> get_side_sets() const { return side_sets_; }

    //! Get Number of SideSets
    int get_num_side_sets() const { return side_sets_.size(); }

    //! Get one SideSet
    SideSet get_side_set(const int id) const;

    //! Get Side Set Connectivity with Global Nodes
    std::map<int, std::vector<int>> get_side_set_conn(const SideSet sideset) const;

    //! Get Side Set Connectivity with Global Nodes
    std::map<int, std::vector<int>> get_side_set_conn(
        const SideSet sideset, bool checkoutside) const;

    //! Make sure child ele (SideSet) is outward oriented w.r.t. parent ele
    std::vector<int> outside_oriented_side(
        const std::vector<int> parentele, const std::vector<int> sidemap) const;

    //! Get edge Normal at node
    std::vector<double> normal(const int head1, const int origin, const int head2) const;

    //! Get normalized Vector between 2 nodes
    std::vector<double> node_vec(const int tail, const int head) const;

    //! Transform SideSet into ElementBlock
    std::vector<ElementBlock> side_set_to_e_blocks(
        const SideSet& sideset, const std::map<int, std::vector<int>>& sidesetconn) const;

    //! Transform SideSet into NodeSet
    NodeSet side_set_to_node_set(
        const SideSet& sideset, const std::map<int, std::vector<int>>& sidesetconn) const;

    //! Get Set of Nodes in SideSet
    std::set<int> get_side_set_nodes(
        const EXODUS::SideSet& sideset, const std::map<int, std::vector<int>>& sidesetconn) const;

    //! Get Node map
    std::shared_ptr<std::map<int, std::vector<double>>> get_nodes() const { return nodes_; }

    //! Get one nodal coords
    std::vector<double> get_node(const int NodeID) const;

    //! Set one nodal coords
    void set_node(const int NodeID, const std::vector<double> coord);

    //! Set number of space dimensions
    void set_nsd(const int nsd);

    //! Close Exodus File
    void close_exo() const;

    //! Write Mesh into exodus file
    void write_mesh(const std::string newexofilename) const;

    //! Add Element Block to mesh
    void add_element_block(const std::shared_ptr<EXODUS::ElementBlock> eblock) const;

    //! Erase Element Block from mesh
    void erase_element_block(const int id);

    //! Erase SideSet from mesh
    void erase_side_set(const int id);

    //! Calculate the midpoint of all elements and return map<midpoint,map<eb,ele> >
    std::map<int, std::pair<int, int>> create_midpoints(
        std::map<int, std::vector<double>>& midpoints, const std::vector<int>& eb_ids) const;

    //! Adjust local element ids referenced in SideSet to global ids
    std::map<int, std::vector<int>> globalify_s_seleids(const int ssid) const;

    //! Plot Nodes in Gmsh-file
    void plot_nodes_gmsh() const;

    //! Plot all ElementBlocks into Gmsh-file
    void plot_element_blocks_gmsh(const std::string fname, const EXODUS::Mesh& mymesh) const;
    void plot_element_blocks_gmsh(
        const std::string fname, const EXODUS::Mesh& mymesh, const std::vector<int>& ebids) const;

    //! Plot Connectivity into Gmsh-file
    void plot_conn_gmsh(const std::string fname, const EXODUS::Mesh& mymesh,
        const std::map<int, std::vector<int>>& conn) const;

   private:
    std::shared_ptr<std::map<int, std::vector<double>>> nodes_;

    std::map<int, std::shared_ptr<ElementBlock>> element_blocks_;

    std::map<int, NodeSet> node_sets_;

    std::map<int, SideSet> side_sets_;

    //! number of dimensions
    int num_dim_;
    //! number of dimensions for 4C problem (wall and fluid2 elements require 2d, although we have
    //! spatial dimensions)
    int four_c_dim_;
    //! number of elements
    int num_elem_;
    //! exoid
    int exoid_;
    //! title
    std::string title_;
  };


  /*!
  \brief ElementBlock is a set of Elements of same discretization Type

  A Element Block is a tiny class storing element-type, name, etc. of a ElementBlock
  It implements its printout.

  */
  class ElementBlock
  {
   public:
    enum Shape
    {
      dis_none,  ///< unknown dis type
      quad4,     ///< 4 noded quadrilateral
      quad8,     ///< 8 noded quadrilateral
      quad9,     ///< 9 noded quadrilateral
      shell4,
      shell8,
      shell9,
      tri3,        ///< 3 noded triangle
      tri6,        ///< 6 noded triangle
      hex8,        ///< 8 noded hexahedra
      hex20,       ///< 20 noded hexahedra
      hex27,       ///< 27 noded hexahedra
      tet4,        ///< 4 noded tetrahedra
      tet10,       ///< 10 noded tetrahedra
      wedge6,      ///< 6 noded wedge
      wedge15,     ///< 15 noded wedge
      pyramid5,    ///< 5 noded pyramid
      bar2,        ///< 2 noded line
      bar3,        ///< 3 noded line
      point1,      ///< 1 noded point
      max_distype  ///<  end marker. must be the last entry
    };

    ElementBlock(ElementBlock::Shape DisType,
        std::shared_ptr<std::map<int, std::vector<int>>>& eleconn,  // Element connectivity
        std::string name);

    virtual ~ElementBlock() = default;

    ElementBlock::Shape get_shape() const { return distype_; }

    int get_num_ele() const { return eleconn_->size(); }

    std::shared_ptr<std::map<int, std::vector<int>>> get_ele_conn() const { return eleconn_; }

    std::vector<int> get_ele_nodes(int i) const;

    std::string get_name() const { return name_; }

    int get_ele_node(int ele, int node) const;

    void fill_econn_array(int* connarray) const;

    void print(std::ostream& os, bool verbose = false) const;

   private:
    Shape distype_;

    //! Element Connectivity
    std::shared_ptr<std::map<int, std::vector<int>>> eleconn_;

    std::string name_;
  };

  class NodeSet
  {
   public:
    NodeSet(const std::set<int>& nodeids, const std::string& name, const std::string& propname);

    virtual ~NodeSet() = default;

    std::set<int> get_node_set() const { return nodeids_; };

    std::string get_name() const { return name_; };

    std::string get_prop_name() const { return propname_; };

    void fill_nodelist_array(int* nodelist) const;

    inline int get_num_nodes() const { return nodeids_.size(); }

    void print(std::ostream& os, bool verbose = false) const;

   private:
    std::set<int> nodeids_;  // nodids in NodeSet
    std::string name_;       // NodeSet name
    std::string propname_;   // Icem assigns part names as property names
  };

  class SideSet
  {
   public:
    SideSet(const std::map<int, std::vector<int>>& sides, const std::string& name);

    virtual ~SideSet() = default;

    inline int get_num_sides() const { return sides_.size(); }

    std::string get_name() const { return name_; }

    std::map<int, std::vector<int>> get_side_set() const { return sides_; }

    void replace_sides(std::map<int, std::vector<int>> newsides)
    {
      sides_ = newsides;
      return;
    };

    std::vector<int> get_first_side_set() const { return sides_.begin()->second; }

    void fill_side_lists(int* elemlist, int* sidelist) const;
    void fill_side_lists(
        int* elemlist, int* sidelist, const std::map<int, std::vector<int>>& sides) const;

    void print(std::ostream& os, bool verbose = false) const;

   private:
    std::map<int, std::vector<int>> sides_;
    std::string name_;
  };
  // *********** end of classes

  Mesh quadto_tri(EXODUS::Mesh& basemesh);

  inline ElementBlock::Shape string_to_shape(const std::string shape)
  {
    if (shape.compare("SPHERE") == 0)
      return ElementBlock::point1;
    else if (shape.compare("QUAD4") == 0)
      return ElementBlock::quad4;
    else if (shape.compare("QUAD8") == 0)
      return ElementBlock::quad8;
    else if (shape.compare("QUAD9") == 0)
      return ElementBlock::quad9;
    else if (shape.compare("SHELL4") == 0)
      return ElementBlock::shell4;
    else if (shape.compare("SHELL8") == 0)
      return ElementBlock::shell8;
    else if (shape.compare("SHELL9") == 0)
      return ElementBlock::shell9;
    else if (shape.compare("TRI3") == 0)
      return ElementBlock::tri3;
    else if (shape.compare("TRI6") == 0)
      return ElementBlock::tri6;
    else if (shape.compare("HEX8") == 0)
      return ElementBlock::hex8;
    else if (shape.compare("HEX20") == 0)
      return ElementBlock::hex20;
    else if (shape.compare("HEX27") == 0)
      return ElementBlock::hex27;
    else if (shape.compare("HEX") == 0)
      return ElementBlock::hex8;  // really needed????? a.g. 08/08
    else if (shape.compare("TET4") == 0)
      return ElementBlock::tet4;  // TODO:: gibts das eigentlich?
    else if (shape.compare("TETRA4") == 0)
      return ElementBlock::tet4;
    else if (shape.compare("TETRA10") == 0)
      return ElementBlock::tet10;
    else if (shape.compare("TETRA") == 0)
      return ElementBlock::tet4;  // really needed????? a.g. 08/08
    else if (shape.compare("WEDGE6") == 0)
      return ElementBlock::wedge6;
    else if (shape.compare("WEDGE15") == 0)
      return ElementBlock::wedge15;
    else if (shape.compare("WEDGE") == 0)
      return ElementBlock::wedge6;  // really needed????? a.g. 08/08
    else if (shape.compare("PYRAMID5") == 0)
      return ElementBlock::pyramid5;
    else if (shape.compare("PYRAMID") == 0)
      return ElementBlock::pyramid5;  // really needed????? a.g. 08/08
    else if (shape.compare("BAR2") == 0)
      return ElementBlock::bar2;
    else if (shape.compare("BAR3") == 0)
      return ElementBlock::bar3;
    else
    {
      std::cout << "Unknown Exodus Element Shape Name: " << shape;
      FOUR_C_THROW("Unknown Exodus Element Shape Name!");
      return ElementBlock::dis_none;
    }
  }

  inline std::string shape_to_string(const ElementBlock::Shape shape)
  {
    switch (shape)
    {
      case ElementBlock::point1:
        return "SPHERE";
        break;
      case ElementBlock::quad4:
        return "QUAD4";
        break;
      case ElementBlock::quad8:
        return "QUAD8";
        break;
      case ElementBlock::quad9:
        return "QUAD9";
        break;
      case ElementBlock::shell4:
        return "SHELL4";
        break;
      case ElementBlock::shell8:
        return "SHELL8";
        break;
      case ElementBlock::shell9:
        return "SHELL9";
        break;
      case ElementBlock::tri3:
        return "TRI3";
        break;
      case ElementBlock::tri6:
        return "TRI6";
        break;
      case ElementBlock::hex8:
        return "HEX8";
        break;
      case ElementBlock::hex20:
        return "HEX20";
        break;
      case ElementBlock::hex27:
        return "HEX27";
        break;
      case ElementBlock::tet4:
        return "TET4";
        break;
      case ElementBlock::tet10:
        return "TET10";
        break;
      case ElementBlock::wedge6:
        return "WEDGE6";
        break;
      case ElementBlock::wedge15:
        return "WEDGE15";
        break;
      case ElementBlock::pyramid5:
        return "PYRAMID5";
        break;
      case ElementBlock::bar2:
        return "BAR2";
        break;
      case ElementBlock::bar3:
        return "BAR3";
        break;
      default:
        FOUR_C_THROW("Unknown ElementBlock::Shape");
    }
    return "xxx";
  }

  inline Core::FE::CellType pre_shape_to_drt(const ElementBlock::Shape shape)
  {
    switch (shape)
    {
      case ElementBlock::point1:
        return Core::FE::CellType::point1;
        break;
      case ElementBlock::quad4:
        return Core::FE::CellType::quad4;
        break;
      case ElementBlock::quad8:
        return Core::FE::CellType::quad8;
        break;
      case ElementBlock::quad9:
        return Core::FE::CellType::quad9;
        break;
      case ElementBlock::shell4:
        return Core::FE::CellType::quad4;
        break;
      case ElementBlock::shell8:
        return Core::FE::CellType::quad8;
        break;
      case ElementBlock::shell9:
        return Core::FE::CellType::quad9;
        break;
      case ElementBlock::tri3:
        return Core::FE::CellType::tri3;
        break;
      case ElementBlock::tri6:
        return Core::FE::CellType::tri6;
        break;
      case ElementBlock::hex8:
        return Core::FE::CellType::hex8;
        break;
      case ElementBlock::hex20:
        return Core::FE::CellType::hex20;
        break;
      case ElementBlock::hex27:
        return Core::FE::CellType::hex27;
        break;
      case ElementBlock::tet4:
        return Core::FE::CellType::tet4;
        break;
      case ElementBlock::tet10:
        return Core::FE::CellType::tet10;
        break;
      case ElementBlock::wedge6:
        return Core::FE::CellType::wedge6;
        break;
      case ElementBlock::wedge15:
        return Core::FE::CellType::wedge15;
        break;
      case ElementBlock::pyramid5:
        return Core::FE::CellType::pyramid5;
        break;
      case ElementBlock::bar2:
        return Core::FE::CellType::line2;
        break;
      case ElementBlock::bar3:
        return Core::FE::CellType::line3;
        break;
      default:
        FOUR_C_THROW("Unknown ElementBlock::Shape");
    }
    return Core::FE::CellType::max_distype;
  }

  void print_map(std::ostream& os, const std::map<int, std::vector<int>> mymap);
  void print_map(std::ostream& os, const std::map<int, std::vector<double>> mymap);
  void print_map(std::ostream& os, const std::map<int, std::set<int>> mymap);
  void print_map(std::ostream& os, const std::map<int, std::map<int, int>> mymap);
  void print_map(std::ostream& os, const std::map<int, std::pair<int, int>> mymap);
  void print_map(std::ostream& os, const std::map<int, int> mymap);
  void print_map(std::ostream& os, const std::map<int, double> mymap);
  void print_map(std::ostream& os, const std::map<double, int> mymap);
  void print_vec(std::ostream& os, const std::vector<int> actvec);
  void print_vec(std::ostream& os, const std::vector<double> actvec);
  void print_set(std::ostream& os, const std::set<int> actset);


  int hex_side_number_exo_to_four_c(const int exoface);
  int pyr_side_number_exo_to_four_c(const int exoface);

}  // namespace EXODUS

FOUR_C_NAMESPACE_CLOSE

#endif
