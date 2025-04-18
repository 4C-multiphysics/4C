// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NONCONFORMING_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NONCONFORMING_HPP

#include "4C_config.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_base.hpp"

#include <Epetra_FEVector.h>

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
  class PoroMultiPhaseScatraArteryCouplingPairBase;

  //! Line based coupling between artery network and poromultiphasescatra algorithm
  class PoroMultiPhaseScaTraArtCouplNonConforming : public PoroMultiPhaseScaTraArtCouplBase
  {
   public:
    PoroMultiPhaseScaTraArtCouplNonConforming(std::shared_ptr<Core::FE::Discretization> arterydis,
        std::shared_ptr<Core::FE::Discretization> contdis,
        const Teuchos::ParameterList& couplingparams, const std::string& condname,
        const std::string& artcoupleddofname, const std::string& contcoupleddofname);

   protected:
    //! Evaluate the 1D-3D coupling
    void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override;

    //! set-up of linear system of equations of coupled problem
    void setup_system(Core::LinAlg::BlockSparseMatrixBase& sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs, Core::LinAlg::SparseMatrix& sysmat_cont,
        Core::LinAlg::SparseMatrix& sysmat_art,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
        const Core::LinAlg::MapExtractor& dbcmap_cont, const Core::LinAlg::Map& dbcmap_art,
        const Core::LinAlg::Map& dbcmap_art_with_collapsed);

    //! setup the strategy
    void setup() override;

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    virtual void evaluate_additional_linearizationof_integrated_diam() = 0;

    //! FE-assemble into global force and stiffness matrix
    virtual void fe_assemble_ele_force_stiff_into_system_vector_matrix(const int& ele1gid,
        const int& ele2gid, const double& integrated_diam,
        std::vector<Core::LinAlg::SerialDenseVector> const& elevec,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elemat,
        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! set flag if varying diameter has to be calculated
    virtual void set_varying_diam_flag();

    //! print out the coupling method
    void print_out_coupling_method() const override;

    //! the parameters
    const Teuchos::ParameterList& couplingparams_;

    //! name of the condition
    const std::string condname_;

    //! have the managers been set?
    bool porofluidmanagersset_;

    //! has setup() been called
    bool issetup_;

    //! is it a pure fluid problem
    bool porofluidprob_;

    //! does it have a varying (by-function)-diameter
    bool has_varying_diam_;

    //! flag to determine if 'free-hanging' elements should be deleted
    bool delete_free_hanging_eles_;

    //! small connected components whose size is smaller than this fraction of the overall network
    //! size are additionally deleted (even if they have a Dirichlet boundary condition)
    double delete_free_hanging_eles_threshold_;

    //! interacting pairs of artery and continuous-discretization elements
    std::vector<std::shared_ptr<PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>
        coupl_elepairs_;

    //! vector with 1D coupling nodes for ntp coupling
    std::vector<int> couplingnodes_ntp_;

    //! type of coupling method
    Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod coupling_method_;

    //! phinp for artery dis
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_art_;

    //! coupling matrix (FE)
    std::shared_ptr<Core::LinAlg::SparseMatrix> FEmat_;

   private:
    //! check if initial fields on coupled DOFs are equal
    //!  \note not performed here since penalty approach will force solution to be
    //!        equal anyway
    void check_initial_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec_art) override {};

    //! access artery (1D) dof row map
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const override;

    //! access full dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const override;

    //! Setup global vector
    void setup_vector(std::shared_ptr<Core::LinAlg::Vector<double>> vec,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec_art) override;

    void extract_single_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> globalvec,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_art) override;

    //! set solution vector of single fields
    void set_solution_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phin_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_art) override;

    //! init the strategy
    void init() override;

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearbyelepairs) override;

    //! access to blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

    //! create interaction pairs for line- or surfacebased coupling (GPTS and MP)
    void create_coupling_pairs_line_surf_based();

    //! create interaction pairs for ntp coupling
    void create_coupling_pairs_ntp();

    //! calculate the volume fraction occupied by blood vessels
    void calculate_blood_vessel_volume_fraction();

    //! get the Id from 1D nodes for coupling
    void get_coupling_idsfrom_input();

    //! evaluate the pairs
    void evaluate_coupling_pairs(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! FE-assemble into global D, M and kappa (MP case)
    void fe_assemble_dm_kappa(const int& ele1gid, const int& ele2gid,
        const Core::LinAlg::SerialDenseMatrix& D_ele, const Core::LinAlg::SerialDenseMatrix& M_ele,
        const Core::LinAlg::SerialDenseVector& Kappa_ele);

    //! sum D and M into global force and stiffness matrix
    void sum_dm_into_global_force_stiff(Core::LinAlg::BlockSparseMatrixBase& sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs);

    //! invert kappa vector
    void invert_kappa();

    //! return appropriate internal implementation class (acts as a simple factory to create single
    //! pairs)
    static std::shared_ptr<PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>
    create_new_artery_coupling_pair(std::vector<Core::Elements::Element const*> const& ele_ptrs);

    //! set the artery diameter in material to be able to use it on 1D discretization
    virtual void set_artery_diam_in_material() = 0;

    //! get the segment lengths of element 'artelegid'
    virtual std::vector<double> get_ele_segment_lengths(const int artelegid) = 0;

    //! reset the integrated diameter vector to zero
    virtual void reset_integrated_diam_to_zero() = 0;

    //! fill Function and Scale vectors as read from input file
    void fill_function_and_scale_vectors();

    //! set the right-hand side factor for time integration
    void set_time_fac_rhs();

    //! right hand side factor for artery time integration
    double timefacrhs_art_;
    //! right hand side factor for time integration of 2D/3D discretization
    double timefacrhs_cont_;

    //! result of brute force search
    std::map<int, std::set<int>> nearbyelepairs_;

    //! phinp for continuous dis
    std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_cont_;

    //! phin for continuous dis
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin_cont_;

    //! zeros for continuous dis
    std::shared_ptr<const Core::LinAlg::Vector<double>> zeros_cont_;

    //! zeros for artery dis
    std::shared_ptr<const Core::LinAlg::Vector<double>> zeros_art_;

    //! scale and function-vector
    std::vector<std::vector<int>> scale_vec_;
    std::vector<std::vector<int>> funct_vec_;

    //! mortar coupling matrices
    std::shared_ptr<Core::LinAlg::SparseMatrix> d_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> m_;
    std::shared_ptr<Epetra_FEVector> kappa_inv_;

    //! penalty parameter
    double pp_;

    //! coupling rhs-vector (FE)
    std::shared_ptr<Epetra_FEVector> fe_rhs_;
  };
}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
