// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NONCONFORMING_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NONCONFORMING_HPP

#include "4C_config.hpp"

#include "4C_art_net_input.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseVector;
}
namespace PoroPressureBased
{
  // forward declaration
  class PorofluidElastScatraArteryCouplingPairBase;

  //! Non-conforming coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingNonConformingAlgorithm
      : public PorofluidElastScatraArteryCouplingBaseAlgorithm
  {
   public:
    PorofluidElastScatraArteryCouplingNonConformingAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name);

   protected:
    //! Evaluate the 1D-3D coupling
    void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override;

    //! set up the global system of equations of the coupled problem
    void setup_system(Core::LinAlg::BlockSparseMatrixBase& sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        Core::LinAlg::SparseMatrix& sysmat_homogenized, Core::LinAlg::SparseMatrix& sysmat_artery,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
        const Core::LinAlg::MapExtractor& dbcmap_homogenized,
        const Core::LinAlg::Map& dbcmap_artery,
        const Core::LinAlg::Map& dbcmap_artery_with_collapsed) const;

    //! set up the strategy
    void setup() override;

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    virtual void evaluate_additional_linearization_of_integrated_diameter() = 0;

    //! FE-assemble into global force and stiffness matrix
    virtual void assemble(const int& ele1_gid, const int& ele2_gid,
        const double& integrated_diameter,
        std::vector<Core::LinAlg::SerialDenseVector> const& ele_rhs,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& ele_matrix,
        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! set flag if variable diameter has to be calculated
    virtual void set_flag_variable_diameter();

    //! print the coupling method
    void print_coupling_method() const override;

    //! coupling parameters
    const Teuchos::ParameterList& coupling_params_;

    //! name of the condition
    const std::string condition_name_;

    //! have the porofluid-managers been initialized?
    bool porofluid_managers_initialized_;

    //! has setup() been called
    bool is_setup_;

    //! is it a pure fluid problem
    bool pure_porofluid_problem_;

    //! does it have a variable (by-function)-diameter
    bool has_variable_diameter_;

    //! flag to determine if 'free-hanging' elements should be deleted
    bool delete_free_hanging_elements_;

    //! small connected components whose size is smaller than this fraction of the overall network
    //! size are additionally deleted (even if they have a Dirichlet boundary condition)
    double threshold_delete_free_hanging_elements_;

    //! interacting pairs of artery and continuous-discretization elements
    std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>> coupled_ele_pairs_;

    //! vector with 1D coupling nodes for node-to-point coupling
    std::vector<int> coupling_nodes_for_node_to_point_;

    //! type of coupling method
    ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod artery_coupling_method_;

    //! phinp for artery discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_art_;

    //! coupling matrix (FE)
    std::shared_ptr<Core::LinAlg::SparseMatrix> coupling_matrix_;

   private:
    //! check if initial fields on coupled DOFs are equal
    //!  \note not performed here since penalty approach will force solution to be
    //!        equal anyway
    void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) override {};

    //! access artery (1D) dof row map
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const override;

    //! access full dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const override;

    //! Setup global vector
    void setup_global_vector(std::shared_ptr<Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) override;

    void extract_single_field_vectors(
        std::shared_ptr<const Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& artery_vector) override;

    //! set solution vector of single fields
    void set_solution_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phin_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_artery) override;

    //! init the strategy
    void init() override;

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    //! access to the blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

    //! create interaction pairs for line- or surface-based coupling (GPTS and MP)
    void create_coupling_pairs_line_surface_based();

    //! create interaction pairs for node-to-point coupling
    void create_coupling_pairs_node_to_point();

    //! get the node Id from 1D nodes from the input file for node-to-point coupling
    void get_coupling_nodes_from_input_node_to_point();

    //! evaluate the pairs
    void evaluate_coupling_pairs(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! FE-assemble into global D, M and kappa (MP case)
    void assemble_mortar_matrices_and_vector(const int& ele1_gid, const int& ele2_gid,
        const Core::LinAlg::SerialDenseMatrix& D_ele, const Core::LinAlg::SerialDenseMatrix& M_ele,
        const Core::LinAlg::SerialDenseVector& Kappa_ele) const;

    //! sum D and M into global force and stiffness matrix
    void sum_mortar_matrices_into_global_matrix(Core::LinAlg::BlockSparseMatrixBase& sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) const;

    //! return appropriate internal implementation class
    //! (acts as a simple factory to create single pairs)
    static std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>
    create_new_artery_coupling_pair(std::vector<Core::Elements::Element const*> const& elements);

    //! set the artery diameter in material to be able to use it on 1D discretization
    virtual void set_artery_diameter_in_material() = 0;

    //! get the segment length
    virtual std::vector<double> get_ele_segment_length(int artery_ele_gid) = 0;

    //! reset the integrated diameter vector to zero
    virtual void reset_integrated_diameter_to_zero() = 0;

    //! fill Function and Scale vectors as read from input file
    void fill_function_and_scale_vectors();

    //! set the right-hand side factor for time integration
    void set_time_fac_rhs();

    //! right-hand side factor for artery time integration
    double timefacrhs_artery_;

    //! right-hand side factor for time integration of 2D/3D discretization
    double timefacrhs_homogenized_;

    //! result of brute force search
    std::map<int, std::set<int>> nearby_ele_pairs_;

    //! phinp for homogenized discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_homogenized_;

    //! phin for homogenized discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin_homogenized_;

    //! zeros for homogenized discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> zeros_homogenized_;

    //! zeros for artery discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> zeros_artery_;

    //! scale and function-vector
    std::vector<std::vector<int>> scale_vector_;
    std::vector<std::vector<int>> function_vector_;

    //! mortar coupling matrices
    std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_matrix_d_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_matrix_m_;
    std::shared_ptr<Core::LinAlg::FEVector<double>> mortar_kappa_inv_;

    //! penalty parameter
    double penalty_parameter_;

    //! coupling rhs-vector (FE)
    std::shared_ptr<Core::LinAlg::FEVector<double>> coupling_rhs_vector_;
  };
}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
