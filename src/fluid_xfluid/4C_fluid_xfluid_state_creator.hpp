// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_XFLUID_STATE_CREATOR_HPP
#define FOUR_C_FLUID_XFLUID_STATE_CREATOR_HPP

#include "4C_config.hpp"

#include "4C_cut_enum.hpp"
#include "4C_cut_input.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_xfem_input.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Cut
{
  class CutWizard;
}

namespace Core::LinAlg
{
  class SparseMatrix;
  class MultiMapExtractor;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace XFEM
{
  class ConditionManager;
  class DiscretizationXFEM;
  class XFEMDofSet;
}  // namespace XFEM

namespace FLD
{
  class XFluidState;
  class XFluidFluidState;

  namespace Utils
  {
    class KSPMapExtractor;
  }

  /**
   * Builder class for XFluid(Fluid)State.
   * Creates the appropriate wizard & handles the cut state (level-set field or boundary
   * discretization).
   */
  class XFluidStateCreator
  {
   public:
    /// ctor
    XFluidStateCreator(std::shared_ptr<XFEM::ConditionManager> condition_manager,
        Teuchos::ParameterList& params_xfem, int maxnumdofsets, int minnumdofsets,
        bool include_inner)
        : condition_manager_(condition_manager),
          nodal_dofset_strategy_(Teuchos::getIntegralValue<Cut::NodalDofSetStrategy>(
              params_xfem, "NODAL_DOFSET_STRATEGY")),
          volume_cell_gauss_point_by_(
              Teuchos::getIntegralValue<Cut::VCellGaussPts>(params_xfem, "VOLUME_GAUSS_POINTS_BY")),
          bound_cell_gauss_point_by_(Teuchos::getIntegralValue<Cut::BCellGaussPts>(
              params_xfem, "BOUNDARY_GAUSS_POINTS_BY")),
          gmsh_cut_out_(params_xfem.get<bool>("GMSH_CUT_OUT")),
          maxnumdofsets_(maxnumdofsets),
          minnumdofsets_(minnumdofsets),
          include_inner_(include_inner)
    {
    }

    /// create a state-object after a cut (pure XFEM fluid)
    std::shared_ptr<XFluidState> create(const std::shared_ptr<XFEM::DiscretizationXFEM>&
                                            xdiscret,  //!< xfluid background discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            back_disp_col,  //!< col vector holding background ALE displacements for backdis
        Teuchos::ParameterList& solver_params,  //!< solver parameters
        const int step,                         //!< current time step
        const double& time                      //!< current time
    );

    /// create a state-object after a cut (XFEM fluid with embedded fluid mesh)
    std::shared_ptr<XFluidFluidState> create(const std::shared_ptr<XFEM::DiscretizationXFEM>&
                                                 xdiscret,  //!< xfluid background discretization
        const std::shared_ptr<Core::FE::Discretization>&
            embfluiddiscret,  //!< embedded fluid discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            back_disp_col,  //!< col vector holding background ALE displacements for backdis
        Teuchos::ParameterList& solver_params,  //!< solver parameters
        const int step,                         //!< current time step
        const double& time                      //!< current time
    );


   private:
    /// create wizard, perform cut, create new dofset and update xfem discretization
    void create_new_cut_state(
        std::shared_ptr<XFEM::XFEMDofSet>& dofset,  //!< xfem dofset obtained from the new wizard
        std::shared_ptr<Cut::CutWizard>&
            wizard,  //!< cut wizard associated with current intersection state
        const std::shared_ptr<XFEM::DiscretizationXFEM>&
            xdiscret,  //!< xfluid background discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            back_disp_col,  //!< col vector holding background ALE displacements for backdis
        Teuchos::ParameterList& solver_params,  //!< solver parameters
        const int step                          //!< current time step
    );


    //! condition manager which handles all coupling objects and the coupling/boundary conditions
    std::shared_ptr<XFEM::ConditionManager> condition_manager_;

    //! strategy for nodal dofset management
    const Cut::NodalDofSetStrategy nodal_dofset_strategy_;

    const Cut::VCellGaussPts volume_cell_gauss_point_by_;
    const Cut::BCellGaussPts bound_cell_gauss_point_by_;

    /// is gmsh-output active?
    const bool gmsh_cut_out_;

    //! @name size limits for dofsets with variable size
    //@{
    const int maxnumdofsets_;
    int minnumdofsets_;
    //@}

    bool include_inner_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
