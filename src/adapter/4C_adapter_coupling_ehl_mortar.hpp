// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_COUPLING_EHL_MORTAR_HPP
#define FOUR_C_ADAPTER_COUPLING_EHL_MORTAR_HPP

/*---------------------------------------------------------------------*
 | headers                                                             |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                                |
 *---------------------------------------------------------------------*/
namespace Core::LinAlg
{
  class SparseMatrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Adapter
{
  class CouplingEhlMortar : public CouplingNonLinMortar
  {
   public:
    //! @name Construction / Destruction
    //@{
    /*!
    \brief Constructor.
    */
    CouplingEhlMortar(Global::Problem& problem, int spatial_dimension,
        Teuchos::ParameterList mortar_coupling_params,
        Teuchos::ParameterList contact_dynamic_params,
        Core::FE::ShapeFunctionType shape_function_type);

    //@}


    //! @name Evaluation
    //@{
    /*!
    \brief Read Mortar Condition

    */
    void read_mortar_condition(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<Core::FE::Discretization> source_dis, std::vector<int> coupleddof,
        const std::string& couplingcond, Teuchos::ParameterList& input,
        std::map<int, Core::Nodes::Node*>& target_global_nodes,
        std::map<int, Core::Nodes::Node*>& source_global_nodes,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& target_elements,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& source_elements) override;

    /*!
    \brief initialize routine

    */
    void setup(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<Core::FE::Discretization> source_dis, std::vector<int> coupleddof,
        const std::string& couplingcond) override;

    /// perform interface integration
    virtual void integrate(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp, const double dt);

    /// perform condensation of contact Lagrange multipliers
    virtual void condense_contact(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& combined_RHS,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp, const double dt);
    virtual void recover_coupled(std::shared_ptr<Core::LinAlg::Vector<double>> sinc,
        std::shared_ptr<Core::LinAlg::Vector<double>> tinc);

    void evaluate_rel_mov();
    void store_dirichlet_status(const Core::LinAlg::MapExtractor& dbcmaps);

    /// check whether this displacement state has already been evaluated
    virtual bool already_evaluated(std::shared_ptr<const Core::LinAlg::Vector<double>> disp);

    /// Assemble linearization D_{ij,k}*x_i
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> assemble_ehl_lin_d(
        const std::shared_ptr<Core::LinAlg::Vector<double>> x);

    /// Assemble linearization M_{il,k}*x_i
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> assemble_ehl_lin_m(
        const std::shared_ptr<Core::LinAlg::Vector<double>> x);

    /// Assemble linearization G_{ij,k}*x_i
    std::shared_ptr<Core::LinAlg::SparseMatrix> assemble_surf_grad_deriv(
        const Core::LinAlg::Vector<double>& x);

    virtual void write_restart(Core::IO::DiscretizationWriter& output);
    virtual void read_restart(Core::IO::DiscretizationReader& reader);
    void create_active_slip_toggle(std::shared_ptr<Core::LinAlg::Vector<double>>* active,
        std::shared_ptr<Core::LinAlg::Vector<double>>* slip,
        std::shared_ptr<Core::LinAlg::Vector<double>>* active_old = nullptr);
    void create_force_vec(std::shared_ptr<Core::LinAlg::Vector<double>>& n,
        std::shared_ptr<Core::LinAlg::Vector<double>>& t);
    bool has_contact() { return contact_regularization_; }
    double contact_res() { return contact_rhs_norm_; }
    double contact_incr() { return contact_LM_incr_norm_; }
    int active_contact();
    int slip_contact();
    //@}

    //! @name Access
    //@{

    /// relative tangential velocity
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> rel_tang_vel() { return relTangVel_; }
    /// relative tangential velocity derivative
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> rel_tang_vel_deriv()
    {
      return relTangVel_deriv_;
    }
    /// average tangential velocity
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> av_tang_vel() { return avTangVel_; }
    /// average tangential velocity derivative
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> av_tang_vel_deriv()
    {
      return avTangVel_deriv_;
    }
    /// nodal gap (not weighted)
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> nodal_gap() { return nodal_gap_; }
    /// nodal gap (not weighted) derivative
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> nodal_gap_deriv()
    {
      return deriv_nodal_gap_;
    }
    /// nodal normals
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> normals() { return normals_; }
    /// nodal normals derivative
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> nderiv_matrix() { return Nderiv_; }
    /// surfrace gradient operator
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> surf_grad_matrix() { return SurfGrad_; }
    /// source+target dof map
    virtual std::shared_ptr<const Core::LinAlg::Map> source_target_dof_map()
    {
      return source_target_dof_row_map_;
    }
    //@}

   private:
    //! @name Unwanted parent functions
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> gap() override
    {
      FOUR_C_THROW("stop");
      return nullptr;
    }
    std::shared_ptr<Core::LinAlg::SparseMatrix> n_matrix() override
    {
      FOUR_C_THROW("stop");
      return nullptr;
    }
    void evaluate() override { FOUR_C_THROW("stop"); }
    void evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> idisp) override
    {
      FOUR_C_THROW("stop");
    }
    void evaluate_with_mesh_relocation(
        std::shared_ptr<Core::FE::Discretization> source_dis,  ///< source discretization
        std::shared_ptr<Core::FE::Discretization> aledis,      ///< ALE discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>& idisp,  ///< ALE displacements
        MPI_Comm comm,                                         ///< communicator
        bool source_is_ale                                     ///< flag defining if source is ALE
        ) override
    {
      FOUR_C_THROW("stop");
    }

    void evaluate_geometry(std::vector<std::shared_ptr<Mortar::IntCell>>&
            intcells  //!< vector of mortar integration cells
        ) override
    {
      FOUR_C_THROW("stop");
    }
    //@}

   protected:
    //! @name Assemlby of global vectors/matrices
    //@{

    /// real (not weighted) gap
    void assemble_real_gap();
    /// real (not weighted) gap derivative wrt displacements
    void assemble_real_gap_deriv();
    /// source sided nodal normals
    void assemble_normals();
    /// source sided nodal normals, derivative wrt displacements
    void assemble_normals_deriv();
    /// source sided tangential gradient operator
    void assemble_surf_grad();
    /// relative and average tangential velocities and their derivatives
    void assemble_interface_velocities(const double dt);
    //@}

    std::shared_ptr<Core::LinAlg::SparseMatrix>
        SurfGrad_;  ///< matrix to compute surface gradient at nodes: Grad(x) \approx TangGrad * x
    std::shared_ptr<Core::LinAlg::Vector<double>> relTangVel_;  ///< relative tangential velocity
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        relTangVel_deriv_;  ///< derivative of relative tangential velocity
    std::shared_ptr<Core::LinAlg::Vector<double>> avTangVel_;  ///< average tangential velocity
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        avTangVel_deriv_;  ///< derivative of average tangential velocity
    std::shared_ptr<Core::LinAlg::Vector<double>>
        nodal_gap_;  ///< NOT the weighted gap but the real distance
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        deriv_nodal_gap_;  ///< derivative of nodal_gap_ w.r.t. displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> normals_;  ///< nodal normals (source side)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        Nderiv_;  ///< derivtative of nodal source-sided normals

    std::shared_ptr<Core::LinAlg::Vector<double>>
        evaluated_state_;  ///< displacement state that has last been evaluated

    bool as_converged_;
    double contact_rhs_norm_;
    double contact_LM_incr_norm_;
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fscn_;  // structural contact forces of last time step (needed for time integration)
    std::shared_ptr<Core::LinAlg::Vector<double>> z_;  //  contact lagrange multiplier

    // recovery of contact LM
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        dinvA_;  // dinv on active displacement dofs (for recovery)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        kss_a_;  // Part of structure-stiffness (kss) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        kst_a_;  // Part of coupling-stiffness  (kst) that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::Vector<double>>
        rs_a_;  // Part of structural residual that corresponds to active source rows
    std::shared_ptr<Core::LinAlg::Vector<double>>
        sdirichtoggle_;  // global dirichlet toggle of all source dofs

    bool contact_regularization_;
    double regularization_thickness_;
    double regularization_compliance_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
