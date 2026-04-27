// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_XFLUID_OUTPUTSERVICE_HPP
#define FOUR_C_FLUID_XFLUID_OUTPUTSERVICE_HPP


/*! header inclusions */
#include "4C_config.hpp"

#include "4C_cut_enum.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_xfem_input.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::DOFSets
{
  class IndependentDofSet;
}

namespace Core::LinAlg
{
  class MapExtractor;
}



namespace Cut
{
  class CutWizard;
  class ElementHandle;
  class VolumeCell;
}  // namespace Cut


namespace XFEM
{
  class ConditionManager;
  class DiscretizationXFEM;
  class XfemEdgeStab;
}  // namespace XFEM

namespace FLD
{
  class XFluidState;

  /*!
   * \brief Class handles output of XFluid and derived classes
   */
  class XFluidOutputService
  {
   public:
    XFluidOutputService(const std::shared_ptr<XFEM::DiscretizationXFEM>& discret,
        const std::shared_ptr<XFEM::ConditionManager>& cond_manager);

    virtual ~XFluidOutputService() = default;

    /// prepare standard output
    void prepare_output();

    /// standard output routine
    void output(int step, double time, bool write_restart_data, const FLD::XFluidState& state,
        std::shared_ptr<Core::LinAlg::Vector<double>> dispnp = nullptr,
        std::shared_ptr<Core::LinAlg::Vector<double>> gridvnp = nullptr);

    /// Gmsh solution output
    virtual void gmsh_solution_output(const std::string& filename_base,  ///< name for output file
        int step,                                                        ///< step number
        const std::shared_ptr<FLD::XFluidState>& state,                  ///< state
        int count = -1) {};

    /// Gmsh solution output for previous time step
    virtual void gmsh_solution_output_previous(
        const std::string& filename_base,                ///< name for output file
        int step,                                        ///< step number
        const std::shared_ptr<FLD::XFluidState>& state,  ///< state
        int count = -1) {};

    /// Gmsh output of solution (debug)
    virtual void gmsh_solution_output_debug(
        const std::string& filename_base,  ///< name for output file
        int step,                          ///< step number
        int count,                         ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
    ) {};

    /// Gmsh output of residual (debug)
    virtual void gmsh_residual_output_debug(
        const std::string& filename_base,  ///< name for output file
        int step,                          ///< step number
        int count,                         ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
    ) {};

    /// Gmsh output of increment (debug)
    virtual void gmsh_increment_output_debug(
        const std::string& filename_base,  ///< name for output file
        int step,                          ///< step number
        int count,                         ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
    ) {};

    /// Gmsh output of discretization
    virtual void gmsh_output_discretization(bool print_faces, int step,
        std::map<int, Core::LinAlg::Matrix<3, 1>>* curr_pos = nullptr) {};

    /// Main output routine for gmsh output
    virtual void gmsh_output(const std::string& filename_base,  ///< name for output file
        const std::string& prefix,                              ///< data prefix
        int step,                                               ///< step number
        int count,  ///< counter for iterations within a global time step
        const std::shared_ptr<Cut::CutWizard>& wizard,  ///< cut wizard
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            vel,  ///< vector holding velocity and pressure dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
            nullptr  ///< vector holding acceleration
    ) {};

    /// Gmsh output for EOS
    virtual void gmsh_output_eos(int step,             ///< step number
        std::shared_ptr<XFEM::XfemEdgeStab> edge_stab  ///< stabilization handler
    ) {};

   protected:
    //! @name XFEM discretization
    //@{
    const std::shared_ptr<XFEM::DiscretizationXFEM> discret_;
    //@}

    //! XFEM condition manager
    const std::shared_ptr<XFEM::ConditionManager> cond_manager_;

    //! dofset for fluid output
    std::shared_ptr<Core::DOFSets::IndependentDofSet> dofset_out_;

    //! output vector (mapped to initial fluid dofrowmap)
    std::shared_ptr<Core::LinAlg::Vector<double>> outvec_fluid_;

    //! vel-pres splitter for output purpose
    std::shared_ptr<Core::LinAlg::MapExtractor> velpressplitter_out_;

    bool firstoutputofrun_;

    //! how many restart steps have already been written
    int restart_count_;
  };

  /*!
   * \brief Class handles output of XFluid and derived classes, capable of handling gmsh output
   */
  class XFluidOutputServiceGmsh : public XFluidOutputService
  {
   public:
    XFluidOutputServiceGmsh(Teuchos::ParameterList& params_xfem,
        const std::shared_ptr<XFEM::DiscretizationXFEM>& discret,
        const std::shared_ptr<XFEM::ConditionManager>& cond_manager, const bool include_inner);

    /// Gmsh solution output
    void gmsh_solution_output(const std::string& filename_base,  ///< name for output file
        int step,                                                ///< step number
        const std::shared_ptr<FLD::XFluidState>& state,          ///< state
        int count = -1) override;

    /// Gmsh solution output for previous time step
    void gmsh_solution_output_previous(const std::string& filename_base,  ///< name for output file
        int step,                                                         ///< step number
        const std::shared_ptr<FLD::XFluidState>& state,                   ///< state
        int count = -1) override;

    /// Gmsh output of solution (debug)
    void gmsh_solution_output_debug(const std::string& filename_base,  ///< name for output file
        int step,                                                      ///< step number
        int count,  ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
        ) override;

    /// Gmsh output of residual (debug)
    void gmsh_residual_output_debug(const std::string& filename_base,  ///< name for output file
        int step,                                                      ///< step number
        int count,  ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
        ) override;

    /// Gmsh output of increment (debug)
    void gmsh_increment_output_debug(const std::string& filename_base,  ///< name for output file
        int step,                                                       ///< step number
        int count,  ///< counter for iterations within a global time step
        const std::shared_ptr<FLD::XFluidState>& state  ///< state
        ) override;

    /// Gmsh output of discretization
    void gmsh_output_discretization(bool print_faces, int step,
        std::map<int, Core::LinAlg::Matrix<3, 1>>* curr_pos = nullptr) override;

    /// Main output routine for gmsh output
    void gmsh_output(const std::string& filename_base,  ///< name for output file
        const std::string& prefix,                      ///< data prefix (e.g. "SOL")
        int step,                                       ///< step number
        int count,               ///< counter for iterations within a global time step
        Cut::CutWizard& wizard,  ///< cut wizard
        const Core::LinAlg::Vector<double>& vel,  ///< vector holding velocity and pressure dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
            nullptr,  ///< vector holding acceleration
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
            nullptr  ///< vector holding ale displacements
    );

    /// Gmsh output for EOS
    void gmsh_output_eos(int step,                     ///< step number
        std::shared_ptr<XFEM::XfemEdgeStab> edge_stab  ///< stabilization handler
        ) override;

   private:
    /// Gmsh output function for elements without an Cut::ElementHandle
    void gmsh_output_element(
        Core::FE::Discretization& discret,        ///< background fluid discretization
        std::ofstream& vel_f,                     ///< output file stream for velocity
        std::ofstream& press_f,                   ///< output file stream for pressure
        std::ofstream& acc_f,                     ///< output file stream for acceleration
        Core::Elements::Element* actele,          ///< element
        std::vector<int>& nds,                    ///< vector holding the nodal dofsets
        const Core::LinAlg::Vector<double>& vel,  ///< vector holding velocity and pressure dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
            nullptr,  ///< vector holding acceleration
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
            nullptr  ///< vector holding ale displacements
    );

    /// Gmsh output function for volumecells
    void gmsh_output_volume_cell(
        Core::FE::Discretization& discret,           ///< background fluid discretization
        std::ofstream& vel_f,                        ///< output file stream for velocity
        std::ofstream& press_f,                      ///< output file stream for pressure
        std::ofstream& acc_f,                        ///< output file stream for acceleration
        Core::Elements::Element* actele,             ///< element
        Cut::ElementHandle* e,                       ///< elementhandle
        Cut::VolumeCell* vc,                         ///< volumecell
        const std::vector<int>& nds,                 ///< vector holding the nodal dofsets
        const Core::LinAlg::Vector<double>& velvec,  ///< vector holding velocity and pressure dofs
        std::shared_ptr<const Core::LinAlg::Vector<double>> accvec =
            nullptr  ///< vector holding acceleration
    );

    /// Gmsh output function for boundarycells
    void gmsh_output_boundary_cell(
        Core::FE::Discretization& discret,  ///< background fluid discretization
        std::ofstream& bound_f,             ///< output file stream for boundary mesh
        Cut::VolumeCell* vc,                ///< volumecell
        Cut::CutWizard& wizard              ///< cut wizard
    );

    //! @name flags for detailed gmsh output
    const bool gmsh_sol_out_;           ///< Gmsh solution output for each timestep
    const bool gmsh_ref_sol_out_;       ///< Gmsh reference solution output
    const bool gmsh_debug_out_;         ///< Gmsh debug output (increment, residual, etc.)
    const bool gmsh_debug_out_screen_;  ///< print information about output to screen
    const bool gmsh_eos_out_;      ///< output for edge-oriented stabilization and ghost-penalty
                                   ///< stabilization
    const bool gmsh_discret_out_;  ///< output of XFEM discretization
    const int gmsh_step_diff_;     ///< no. of kept steps
    //@}

    //! integration approach
    const Cut::VCellGaussPts volume_cell_gauss_point_by_;

    //! include elements with inside position?
    const bool include_inner_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
