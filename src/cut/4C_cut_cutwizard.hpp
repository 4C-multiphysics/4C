// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_CUTWIZARD_HPP
#define FOUR_C_CUT_CUTWIZARD_HPP

#include "4C_config.hpp"

#include "4C_cut_enum.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <filesystem>
#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}

namespace XFEM
{
  class ConditionManager;
}


namespace Cut
{
  class CombIntersection;
  class ElementHandle;
  class Node;
  class SideHandle;

  /// contains the cut, and shared functionality between the level set and mesh cut.
  class CutWizard
  {
   public:
    /*------------------------------------------------------------------------*/
    /*! \brief Container class for the background mesh object
     *
     *  */
    class BackMesh
    {
     public:
      /// constructor
      explicit BackMesh(
          const std::shared_ptr<Core::FE::Discretization>& backdis, Cut::CutWizard* wizard)
          : wizard_(wizard),
            back_discret_(backdis),
            back_disp_col_(nullptr),
            back_levelset_col_(nullptr)
      {
        if (!backdis) FOUR_C_THROW("null pointer to background dis, invalid!");
      }

      virtual ~BackMesh() = default;

      void init(const std::shared_ptr<const Core::LinAlg::Vector<double>>& back_disp_col,
          const std::shared_ptr<const Core::LinAlg::Vector<double>>& back_levelset_col);

      const std::shared_ptr<Core::FE::Discretization>& get_ptr() { return back_discret_; }

      Core::FE::Discretization& get() { return *back_discret_; }

      const Core::FE::Discretization& get() const { return *back_discret_; }

      virtual int num_my_col_elements() const;

      virtual const Core::Elements::Element* l_col_element(int lid) const;

      inline bool is_back_disp() const { return (back_disp_col_ != nullptr); }

      const Core::LinAlg::Vector<double>& back_disp_col() const
      {
        if (not is_back_disp())
          FOUR_C_THROW("The background displacement was not initialized correctly!");

        return *back_disp_col_;
      }

      inline bool is_level_set() const { return (back_levelset_col_ != nullptr); }

      const Core::LinAlg::Vector<double>& back_level_set_col() const
      {
        if (not is_level_set())
          FOUR_C_THROW("No level-set values set for the background discretization!");

        return *back_levelset_col_;
      }


     protected:
      Cut::CutWizard* wizard_;

     private:
      /// background discretization
      std::shared_ptr<Core::FE::Discretization> back_discret_;

      /// col vector holding background ALE displacements for backdis
      std::shared_ptr<const Core::LinAlg::Vector<double>> back_disp_col_;

      /// col vector holding nodal level-set values based on backdis
      std::shared_ptr<const Core::LinAlg::Vector<double>> back_levelset_col_;
    };

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Container class for a certain cutting mesh objects
     */
    class CutterMesh
    {
     public:
      //! ctor
      CutterMesh(std::shared_ptr<Core::FE::Discretization> cutterdis,
          std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
          const int start_ele_gid)
          : cutterdis_(cutterdis), cutter_disp_col_(cutter_disp_col), start_ele_gid_(start_ele_gid)
      {
      }

      std::shared_ptr<Core::FE::Discretization> get_cutter_discretization() { return cutterdis_; }

      //---------------------------------discretization-----------------------------

      //! @name cutter discretization
      std::shared_ptr<Core::FE::Discretization> cutterdis_;  ///< cutter discretization
      //@}

      //---------------------------------state vectors ----------------------------

      //! @name state vectors holding displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>>
          cutter_disp_col_;  ///< col vector holding interface displacements for cutterdis
      //@}

      //!
      int start_ele_gid_;
    };

    /*========================================================================*/
    //! @name Constructor and Destructor
    /*========================================================================*/

    /**
     * \brief Constructor.
     *
     * Create CutWizard object with the given background discretization. The optional
     * function @p global_dof_indices can be used to retrieve the global dof indices from
     * the background discretization.
     */
    CutWizard(const std::shared_ptr<Core::FE::Discretization>& backdis,
        std::function<void(const Core::Nodes::Node& node, std::vector<int>& lm)>
            global_dof_indices = nullptr);


    /*!
    \brief Destructor
    */
    virtual ~CutWizard() = default;

    //@}

    /*========================================================================*/
    //! @name Setters
    /*========================================================================*/

    //! set options and flags used during the cut
    void set_options(const Teuchos::ParameterList& cutparams,  //!< parameter list for cut options
        Cut::NodalDofSetStrategy nodal_dofset_strategy,  //!< strategy for nodal dofset management
        Cut::VCellGaussPts VCellgausstype,  //!< Gauss point generation method for Volumecell
        Cut::BCellGaussPts BCellgausstype,  //!< Gauss point generation method for Boundarycell
        std::string output_prefix,          //!< prefix for output files
        bool gmsh_output,                   //!< print write gmsh output for cut
        bool positions,     //!< set inside and outside point, facet and volumecell positions
        bool tetcellsonly,  //!< generate only tet cells
        bool screenoutput   //!< print screen output
    );

    virtual void set_background_state(
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            back_disp_col,  //!< col vector holding background ALE displacements for backdis
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            back_levelset_col,  //!< col vector holding nodal level-set values based on backdis
        int level_set_sid       //!< global id for level-set side
    );

    void add_cutter_state(const int mc_idx, std::shared_ptr<Core::FE::Discretization> cutter_dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col);

    void add_cutter_state(const int mc_idx, std::shared_ptr<Core::FE::Discretization> cutter_dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
        const int start_ele_gid);

    // Find marked background-boundary sides.
    //  Extract these sides and create boundary cell for these!
    void set_marked_condition_sides(
        // const int mc_idx,
        Core::FE::Discretization& cutter_dis,
        // std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
        const int start_ele_gid);

    //@}

    /*========================================================================*/
    //! @name main Cut call
    /*========================================================================*/

    //! prepare the cut, add background elements and cutting sides
    void prepare();

    void cut(bool include_inner  //!< perform cut in the interior of the cutting mesh
    );

    /*========================================================================*/
    //! @name Accessors
    /*========================================================================*/

    //! Get this side (not from cut meshes) (faces of background elements) from the cut libraries
    Cut::SideHandle* get_side(std::vector<int>& nodeids);

    //! Get this side (not from cut meshes) from the cut libraries
    Cut::SideHandle* get_side(int sid);

    //! Get this side from cut meshes from the cut libraries
    Cut::SideHandle* get_cut_side(int sid);

    //! Get this element from the cut libraries by element id
    Cut::ElementHandle* get_element(const int eleid) const;

    //! Get this element from the cut libraries by element pointer
    Cut::ElementHandle* get_element(const Core::Elements::Element* ele) const;

    //! Get this node from the cut libraries
    Cut::Node* get_node(int nid);

    //! Get the sidehandle for cutting sides
    Cut::SideHandle* get_mesh_cutting_side(int sid, int mi);

    //! is there a level-set side with the given sid?
    bool has_ls_cutting_side(int sid);

    //! update the coordinates of the cut boundary cells
    void update_boundary_cell_coords(std::shared_ptr<Core::FE::Discretization> cutterdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
        const int start_ele_gid);

    //! Cubaturedegree for creating of integrationpoints on boundarycells
    int get_bc_cubaturedegree() const;

    //! From the cut options, get if the cells marked as inside cells have a physical meaning
    //! and if they should be integrated
    bool do_inside_cells_have_physical_meaning();

    //! Get the main intersection
    std::shared_ptr<Cut::CombIntersection> get_intersection();

    //! Check if the construction of the coupling pairs can be performed
    void check_if_mesh_intersection_and_cut();

   protected:
    /** \brief hidden constructor for derived classes only
     *
     *  */
    CutWizard(MPI_Comm comm);

    std::shared_ptr<BackMesh>& back_mesh_ptr() { return back_mesh_; }

    [[nodiscard]] std::shared_ptr<const BackMesh> back_mesh_ptr() const { return back_mesh_; }

    //! Get the current position of a given element
    Core::LinAlg::SerialDenseMatrix get_current_element_position(
        const Core::Elements::Element* element);

    Cut::CombIntersection& intersection()
    {
      if (!intersection_) FOUR_C_THROW("nullptr pointer!");

      return *intersection_;
    }

   private:
    /*========================================================================*/
    //! @name Add functionality for elements and cutting sides
    /*========================================================================*/

    //! add all cutting sides (mesh and level-set sides)
    void add_cutting_sides();

    //! add level-set cutting side
    void add_ls_cutting_side();

    //! add all cutting sides from the cut-discretization
    void add_mesh_cutting_side();

    //! add elements from the background discretization
    void add_background_elements();

    //! add elements from the background discretization for the cases of mesh and level set
    //! intersection
    void add_background_elements_general();

    //! add elements from the background discretization for embedded mesh applications,
    void add_background_elements_embeddedmesh(
        std::vector<Core::Conditions::Condition*>& embeddedmesh_cond);

    //! Add all cutting side elements of given cutter discretization with given displacement field
    //! to the intersection class
    void add_mesh_cutting_side(std::shared_ptr<Core::FE::Discretization> cutterdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> cutter_disp_col,
        const int start_ele_gid = 0  ///< global start index for element id numbering
    );

    //! Add this cutting side element with given global coordinates to the intersection class
    void add_mesh_cutting_side(int mi, Core::Elements::Element* element,
        const Core::LinAlg::SerialDenseMatrix& element_current_position, int sid);

    //! Add this background mesh element to the intersection class
    void add_element(const Core::Elements::Element* ele,
        const Core::LinAlg::SerialDenseMatrix& xyze, double* myphinp = nullptr,
        bool lsv_only_plus_domain = false);

    //@}


    /*========================================================================*/
    //! @name Major steps to prepare the cut, to perform it and to do postprocessing
    /*========================================================================*/

    //! perform the actual cut, the intersection
    void run_cut(bool include_inner  //!< perform cut in the interior of the cutting mesh
    );

    //! routine for finding node positions and computing volume-cell dofsets in a parallel way
    void find_position_dof_sets(bool include_inner);

    //! write statistics and output to screen and files
    void output(bool include_inner);

    //! Check that cut is initialized correctly
    bool safety_checks(bool is_prepare_cut_call);

    //@}

    /*========================================================================*/
    //! @name Output routines
    /*========================================================================*/

    /*! Print the number of volumecells and boundarycells generated over the
     *  whole mesh during the cut */
    void print_cell_stats();

    //! Write the DOF details of the nodes
    void dump_gmsh_num_dof_sets(bool include_inner);

    //! Write volumecell output in GMSH format throughout the domain
    void dump_gmsh_volume_cells(bool include_inner);

    //! Write the integrationcells and boundarycells in GMSH format throughout the domain
    void dump_gmsh_integration_cells();

    //@}

    //---------------------------------discretizations----------------------------

    //! @name meshes
    std::shared_ptr<BackMesh> back_mesh_;
    std::function<void(const Core::Nodes::Node& node, std::vector<int>& lm)> global_dof_indices_;
    std::map<int, std::shared_ptr<CutterMesh>> cutter_meshes_;
    MPI_Comm comm_;
    int myrank_;  ///< my processor Id
    //@}

    //---------------------------------main intersection class----------------------------
    //! @name main intersection class and flags
    std::shared_ptr<Cut::CombIntersection>
        intersection_;  ///< combined intersection object which handles cutting mesh sides and a
                        ///< level-set side

    bool do_mesh_intersection_;      ///< flag to perform intersection with mesh sides
    bool do_levelset_intersection_;  ///< flag to perform intersection with a level-set side
    //@}

    //---------------------------------state vectors ----------------------------

    //! @name state vectors holding displacements and level-set values
    int level_set_sid_;
    //@}

    //---------------------------------Options ----------------------------

    //! @name Options
    Cut::VCellGaussPts v_cellgausstype_;  ///< integration type for volume-cells
    Cut::BCellGaussPts b_cellgausstype_;  ///< integration type for boundary-cells
    std::string output_prefix_;           ///< prefix for output files
    bool gmsh_output_;                    ///< write gmsh output?
    bool tetcellsonly_;          ///< enforce to create tetrahedral integration cells exclusively
    bool screenoutput_;          ///< write output to screen
    bool lsv_only_plus_domain_;  ///< consider only plus domain of level-set field as physical field
    //@}

    //--------------------------------- Initialization flags ----------------------------

    //! @name Flags whether wizard is initialized correctly
    bool is_set_options_;
    bool is_cut_prepare_performed_;

    //! @name Flag to check that the cut operation was done
    bool is_cut_performed_;
    //@}

  };  // class CutWizard
}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
