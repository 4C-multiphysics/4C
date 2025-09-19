// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CARDIOVASCULAR0D_MANAGER_HPP
#define FOUR_C_CARDIOVASCULAR0D_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_sparse_pod.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_ParameterList.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class DiscretizationReader;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
  class MapExtractor;
  class MultiMapExtractor;
  class Solver;
}  // namespace Core::LinAlg

namespace Cardiovascular0D
{
  class ProperOrthogonalDecomposition;
}

namespace Utils
{
  // forward declarations
  class Cardiovascular0D;
  class Cardiovascular0DDofSet;

  class Cardiovascular0DManager
  {
   public:
    /*!
      \brief Constructor of cardiovascular0d manager
    */
    Cardiovascular0DManager(
        std::shared_ptr<Core::FE::Discretization> disc,            ///< standard discretization
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp,  ///< current displacement
        Teuchos::ParameterList
            strparams,  ///<  parameterlist from structural time integration algorithm
        Teuchos::ParameterList cv0dparams,  ///<  parameterlist from cardiovascular0d
        Core::LinAlg::Solver& solver,       ///< Solver to solve linear subproblem in iteration
        std::shared_ptr<Core::LinAlg::ProperOrthogonalDecomposition> mor  ///< model order reduction
    );

    /*!
      \brief Assemble cardiovascular0d stiffness and rhs contributions to full coupled problem
    */
    void evaluate_force_stiff(const double time,  ///< time at end of time step
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            disp,  ///< displacement at end of time step
        std::shared_ptr<Core::LinAlg::Vector<double>>
            fint,  ///< vector of internal structural forces
        std::shared_ptr<Core::LinAlg::SparseOperator> stiff,  ///< structural stiffness matrix
        Teuchos::ParameterList scalelist);

    /*!
     \brief Return cardiovascular0d rhs norm at generalized midpoint \f$t_{n+\theta}\f$
    */
    double get_cardiovascular0_drhs_norm() const
    {
      double foo;
      cardvasc0d_res_m_->norm_2(&foo);
      return foo;
    };

    /*!
     \brief Return cardiovascular0d rhs norm at generalized midpoint \f$t_{n+\theta}\f$
    */
    double get_cardiovascular0_drhs_inf_norm() const
    {
      double foo;
      cardvasc0d_res_m_->norm_inf(&foo);
      return foo;
    };

    /*!
     \brief Return cardiovascular0d dof inbcr norm
    */
    double get_cardiovascular0_d_dof_incr_norm() const
    {
      double foo;
      cv0ddofincrement_->norm_2(&foo);
      return foo;
    };

    /*!
     \brief Return cardiovascular0d rhs norm at generalized midpoint \f$t_{n+\theta}\f$
    */
    int get_cardiovascular0_d_lin_solve_error() const { return linsolveerror_; };

    /*!
         \brief Update cardiovascular0d dofs
    */
    void update_time_step();

    /*!
         \brief check periodic state of cardiovascular model
    */
    void check_periodic();


    bool is_relative_equal_to(const double A, const double B, const double Ref);

    bool modulo_is_relative_zero(const double value, const double modulo, const double Ref);

    /*!
         \brief Update cardiovascular0d dofs
    */
    void reset_step();

    /// Add a vector as residual increment to the cardiovascular0d dof vector
    void update_cv0_d_dof(Core::LinAlg::Vector<double>& cv0ddofincrement  ///< vector to add
    );

    ///
    void evaluate_neumann_cardiovascular0_d_coupling(
        Teuchos::ParameterList params, Core::LinAlg::Vector<double>& actpres,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,  ///< structural rhs
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix  ///< structural stiffness matrix
    );

    /*!
         \brief Return cardiovascular0d rhs at generalized midpoint \f$t_{n+\theta}\f$
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_cardiovascular0_drhs() const
    {
      return cardvasc0d_res_m_;
    }

    /*!
     \brief Return map that determined distribution of Cardiovascular0D functions and
     pressures over processors
    */
    std::shared_ptr<Core::LinAlg::Map> get_cardiovascular0_d_map() const
    {
      return cardiovascular0dmap_full_;
    };

    //! Return the additional rectangular matrix, constructed for pressure evaluation
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_mat_dcardvasc0d_dd()  // const
    {
      return mat_dcardvasc0d_dd_;
    };

    //! Return the additional rectangular matrix, constructed for pressure evaluation
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_mat_dstruct_dcv0ddof()  // const
    {
      return mat_dstruct_dcv0ddof_;
    };

    //! Return the additional rectangular matrix, constructed for pressure evaluation
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_cardiovascular0_d_stiffness()  // const
    {
      return cardiovascular0dstiffness_;
    };

    /*!
      \brief Return dof vector
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_dof_np() const { return cv0ddof_np_; };

    /*!
      \brief Return dof vector of last converged step
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_dof_n() const { return cv0ddof_n_; };

    /*!
      \brief Return dof vector of last converged step
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_dof_m() const { return cv0ddof_m_; };

    /*!
      \brief Return vol vector
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_vol_np() const { return v_np_; };


    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_df_np() const
    {
      return cardvasc0d_df_np_;
    };

    /*!
      \brief Return dof vector of last converged step
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_df_n() const { return cardvasc0d_df_n_; };

    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_f_np() const { return cardvasc0d_f_np_; };

    /*!
      \brief Return dof vector of last converged step
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get0_d_f_n() const { return cardvasc0d_f_n_; };

    /*!
     \brief Return if there are Cardiovascular0Ds
    */
    bool have_cardiovascular0_d() const { return havecardiovascular0d_; };

    /*!
     \brief Read restart information
    */
    void read_restart(Core::IO::DiscretizationReader& reader, const double& time);

    /*!
     \brief Return structural input parameter list
    */
    Teuchos::ParameterList& str_params() { return strparams_; }

    /*!
     \brief Return cardiovascular0d input parameter list
    */
    Teuchos::ParameterList& cardvasc0_d_params() { return cv0dparams_; }

    std::shared_ptr<Core::LinAlg::Solver>& get_solver() { return solver_; }

    /// Reset reference base values for restart computations
    void set0_d_v_n(Core::LinAlg::Vector<double>& newval  ///< new reference base values
    )
    {
      v_n_->update(1.0, newval, 0.0);
    }

    /// set df_n, f_n
    void set0_d_df_n(Core::LinAlg::Vector<double>& newval  ///< new Cardiovascular0D dofs
    )
    {
      cardvasc0d_df_n_->update(1.0, newval, 0.0);
      return;
    }
    void set0_d_f_n(Core::LinAlg::Vector<double>& newval  ///< new Cardiovascular0D dofs
    )
    {
      cardvasc0d_f_n_->update(1.0, newval, 0.0);
      return;
    }

    /// Reset dofs
    void set0_d_dof_n(Core::LinAlg::Vector<double>& newdof  ///< new Cardiovascular0D dofs
    )
    {
      cv0ddof_np_->update(1.0, newdof, 0.0);
      cv0ddof_n_->update(1.0, newdof, 0.0);
      return;
    }

    void print_pres_flux(bool init) const;



    //! switch Cardiovascular0D matrix to block matrix
    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps);


    void solver_setup(Core::LinAlg::Solver& solver, Teuchos::ParameterList params);


    int solve(Core::LinAlg::SparseMatrix& stiff,  ///< stiffness matrix
        Core::LinAlg::Vector<double>& dispinc,    ///< displacement increment to compute
        Core::LinAlg::Vector<double>& rhsstruct,  ///< standard right hand side
        const double k_ptc                        ///< for 3D-0D PTC
    );

    std::shared_ptr<Cardiovascular0D> get_cardvasc0_d4_element_windkessel()
    {
      return cardvasc0d_4elementwindkessel_;
    }

    std::shared_ptr<Cardiovascular0D> get_cardvasc0_d_arterial_prox_dist()
    {
      return cardvasc0d_arterialproxdist_;
    }

    std::shared_ptr<Cardiovascular0D> get_cardvasc0_d_sys_pul_circulation()
    {
      return cardvasc0d_syspulcirculation_;
    }

    std::shared_ptr<Cardiovascular0D> get_cardvasc_respir0_d_sys_pul_periph_circulation()
    {
      return cardvascrespir0d_syspulperiphcirculation_;
    }

    bool get_is_periodic() const { return is_periodic_; };

    double get_k_ptc() const { return k_ptc_; };

    void modify_k_ptc(const double sum, const double fac)
    {
      // increase PTC factor
      if (k_ptc_ == 0.0)
        k_ptc_ += sum;
      else
        k_ptc_ *= fac;
    };

    void reset_k_ptc()
    {
      // reset PTC factor - for adaptivity, if divcont flag is set to "adapt_3D0Dptc_ele_err"
      k_ptc_ = 0.0;
    };


   private:
    // don't want = operator, cctor and destructor
    Cardiovascular0DManager operator=(const Cardiovascular0DManager& old);
    Cardiovascular0DManager(const Cardiovascular0DManager& old);


    std::shared_ptr<Core::FE::Discretization>
        actdisc_;  ///< discretization where elements of cardiovascular0d boundary live in
    int myrank_;   ///< processor
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;  ///< map for Dirichlet DOFs
    std::shared_ptr<Cardiovascular0DDofSet>
        cardiovascular0ddofset_full_;  ///< degrees of freedom of pressures
    std::shared_ptr<Cardiovascular0DDofSet>
        cardiovascular0ddofset_;  ///< (reduced) degrees of freedom of pressures
    std::shared_ptr<Core::LinAlg::Map>
        cardiovascular0dmap_full_;  ///< unique map of Cardiovascular0D values
    std::shared_ptr<Core::LinAlg::Map>
        cardiovascular0dmap_;  ///< unique map of (reduced) Cardiovascular0D values
    std::shared_ptr<Core::LinAlg::Map>
        redcardiovascular0dmap_;  ///< fully redundant map of Cardiovascular0D values
    std::shared_ptr<Core::LinAlg::Export>
        cardvasc0dimpo_;  ///< importer for fully redundant Cardiovascular0D
                          ///< vector into distributed one
    std::shared_ptr<Core::LinAlg::Vector<double>> cv0ddofincrement_;  ///< increment of cvdof
    std::shared_ptr<Core::LinAlg::Vector<double>> cv0ddof_n_;         ///< cvdof vector at t_{n}
    std::shared_ptr<Core::LinAlg::Vector<double>> cv0ddof_np_;        ///< cvdof vector at t_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> cv0ddof_m_;         ///< cvdof vector at mid-point
    std::shared_ptr<Core::LinAlg::Vector<double>> dcv0ddof_m_;  ///< cvdof rate vector at mid-point
    std::shared_ptr<Core::LinAlg::Vector<double>> v_n_;         ///< vol vector at t_{n}
    std::shared_ptr<Core::LinAlg::Vector<double>> v_np_;        ///< vol vector at t_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> v_m_;         ///< vol vector at mid-point
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cv0ddof_t_n_;  ///< cvdof vector at periodic time T_{N}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cv0ddof_t_np_;  ///< cvdof vector at periodic time T_{N+1}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_res_m_;  ///< Cardiovascular0D full rhs vector, at t_{n+theta}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_df_n_;  ///< Cardiovascular0D rhs part associated with time derivatives, at t_{n}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_df_np_;  ///< Cardiovascular0D rhs part associated
                            ///< with time derivatives, at t_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_df_m_;  ///< Cardiovascular0D rhs part associated
                           ///< with time derivatives, at t_{n+theta}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_f_n_;  ///< Cardiovascular0D rhs part associated with non-derivatives, at t_{n}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_f_np_;  ///< Cardiovascular0D rhs part associated
                           ///< with non-derivatives, at t_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>>
        cardvasc0d_f_m_;           ///< Cardiovascular0D rhs part associated
                                   ///< with non-derivatives, at t_{n+theta}
    const double t_period_;        ///< periodic time
    const double eps_periodic_;    ///< tolerance for periodic state
    bool is_periodic_;             ///< true, if periodic state is reached, false otherwise
    double cycle_error_;           ///< perdiodicity error
    int num_cardiovascular0_did_;  ///< number of Cardiovascular0D bcs
    int cardiovascular0_did_;      ///< smallest Cardiovascular0D bc id
    int offset_id_;                ///< smallest Cardiovascular0D bc id
    std::vector<int> current_id_;  ///< bc id
    bool havecardiovascular0d_;    ///< are there Cardiovascular0D bcs at all?
    std::shared_ptr<Cardiovascular0D> cardvasc0d_model_;
    std::shared_ptr<Cardiovascular0D> cardvasc0d_4elementwindkessel_;
    std::shared_ptr<Cardiovascular0D> cardvasc0d_arterialproxdist_;
    std::shared_ptr<Cardiovascular0D> cardvasc0d_syspulcirculation_;
    std::shared_ptr<Cardiovascular0D> cardvascrespir0d_syspulperiphcirculation_;
    std::shared_ptr<Core::LinAlg::Solver> solver_;  ///< solver for linear standard linear system
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        cardiovascular0dstiffness_;  ///< additional rectangular matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        mat_dcardvasc0d_dd_;  ///< additional rectangular matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        mat_dstruct_dcv0ddof_;  ///< additional rectangular matrix
    int counter_;               ///< counts how often #Solve is called
    bool isadapttol_;           ///< adaptive tolerance for solver?
    double adaptolbetter_;      ///< adaptive tolerance for solver useful?
    double tolres_struct_;      ///< tolerace for structural residual
    double tolres_cardvasc0d_;  ///< tolerace for cardiovascular0d residual
    Inpar::Cardiovascular0D::Cardvasc0DSolveAlgo algochoice_;
    std::shared_ptr<Core::LinAlg::Vector<double>>
        dirichtoggle_;                                     ///< \b only for compatibility: dirichlet
                                                           ///< toggle -- monitor its target change!
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;  ///< a zero vector of full length
    const double theta_;                 ///< time-integration factor for One Step Theta scheme
    const bool enhanced_output_;         ///< for enhanced output
    const bool ptc_3d0d_;                ///< if we want to use PTC (pseudo-transient continuation)
    double k_ptc_;                       ///< PTC factor, stiff*incr + k_ptc_ * Id = -red
    double totaltime_;                   ///< total simulation time
    int linsolveerror_;                  ///< indicates error / problem in linear solver
    Teuchos::ParameterList strparams_;   ///< structure input parameters
    Teuchos::ParameterList cv0dparams_;  ///< 0D cardiovascular input parameters
    Inpar::Solid::IntegrationStrategy
        intstrat_;  ///< structural time-integration strategy (old vs. standard)
    std::shared_ptr<Core::LinAlg::ProperOrthogonalDecomposition> mor_;  ///< model order reduction
    bool have_mor_;  ///< model order reduction is used

  };  // class
}  // namespace Utils
FOUR_C_NAMESPACE_CLOSE

#endif
